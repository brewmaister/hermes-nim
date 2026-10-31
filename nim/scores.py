"""Quality prior, fetched instead of typed.

`benchmarks.py` used to hold three hand-typed Artificial Analysis numbers. Accurate the day they were
written, silently wrong afterwards, and the only part of this plugin that was not mechanical.

Two sources now, each on its own scale, each with its own glyph, never averaged:

* **Hermes Index** (`🧠`) — portal.nousresearch.com/bench, an agentic index (task completion and cost
  per task, measured inside Hermes Agent). Server-rendered HTML, 18 models. Covers the NVIDIA line;
  blind to MiniMax and Mistral.
* **LMArena** (`🏛`) — the public leaderboard dataset, queried by name through the datasets-server
  `/search` endpoint. ~420 rated models. The reverse: it rates MiniMax, Mistral, Muse, GPT-OSS, and
  rates no Nemotron at all.

Neither is a superset, which is why both exist. They are *not* comparable: 24.1 on a 0–100 agentic
index and 1431 Elo are different measurements, so nothing here adds, averages or ranks one against the
other. The star's rule is index first, LMArena only when the index is silent — a second opinion, never
a blended one.

Why `/search` and not pagination: the dataset has 1.2M rows and its `/filter` endpoint 500s at almost
every offset past zero (offset 100 with length 50 answered, offset 150 did not). Asking for the eleven
model names we actually score is both robust and cheap, and it cannot half-download a table.

What remains non-mechanical is the id -> leaderboard-name match, and it is labelled on screen: exact
normalised match wins, otherwise the most specific shared token set, marked `inherited` — `Nemotron
Super 3.5` is the family, `nemotron-3.5-super-vl-120b-a12b` the variant. The token comparison exists
because the two names order their words differently.

Failures are not fatal: no cache and no network means no score, and every view says "N of M published"
rather than inventing one. A fetch that comes back shrunken keeps the previous rows.
"""

from __future__ import annotations

import html as html_module
import json
import logging
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

CACHE_NAME = "scores.json"
CACHE_VERSION = 4

INDEX_URL = "https://portal.nousresearch.com/bench"
INDEX_SOURCE = "hermes"
INDEX_GLYPH = "🧠"

ARENA_SOURCE = "lmarena"
ARENA_GLYPH = "🏛"
ARENA_SEARCH = ("https://datasets-server.huggingface.co/search"
                "?dataset=lmarena-ai%2Fleaderboard-dataset&config=text&split=latest")

# The glyph an id carries on screen. One per scale, so a reader never has to guess which number it is.
GLYPHS = {INDEX_SOURCE: INDEX_GLYPH, ARENA_SOURCE: ARENA_GLYPH}

# Backwards-compatible aliases: earlier code and the benchmarks shim spoke of one "SOURCE".
SOURCE = INDEX_URL
GLYPH = INDEX_GLYPH


@dataclass(frozen=True)
class Score:
    value: float          # in its own source's units: 0-100 index, or Elo
    rank: int
    name: str             # the leaderboard's own name for the model, so a match can be judged
    org: str
    published: str
    inherited: bool       # True when this is a family match, not an exact one
    cost: float | None = None
    source: str = INDEX_SOURCE
    lower: float | None = None      # published 95% interval, when the source gives one
    upper: float | None = None
    votes: int | None = None

    @property
    def glyph(self) -> str:
        return GLYPHS.get(self.source, "")

    @property
    def label(self) -> str:
        """One number, in its own units and with its own glyph."""
        return f"{self.glyph}{self.value:.1f}" if self.source == INDEX_SOURCE else f"{self.glyph}{self.value:.0f}"


def _age_days(stamp: str | None) -> float | None:
    """Days since an ISO stamp, or None when it cannot be read."""
    if not stamp:
        return None
    try:
        when = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
    except ValueError:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - when).total_seconds() / 86400


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _norm(text: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text or "").lower())


def _tokens(text: object) -> set[str]:
    """Words of a model name, punctuation dropped: `Nemotron Super 3.5` -> {nemotron, super, 3, 5}."""
    return {token for token in re.split(r"[^a-z0-9]+", str(text or "").lower()) if token}


_PATH_CACHE: dict[str, Path] = {}
# Renders ask for a score per row; resolving the path means reading config.yaml and stat-ing a file.
# A memo keyed on (path, mtime) keeps a board render off the disk without ever serving a stale fetch.
_MEMO: dict[str, object] = {}


def cache_path() -> Path:
    """Resolved once per process: it costs a config.yaml read, and renders ask for it per row."""
    cached = _PATH_CACHE.get("path")
    if cached is not None:
        return cached
    from . import engine, settings

    path = engine.state_dir(settings.settings().state_dir or "") / CACHE_NAME
    _PATH_CACHE["path"] = path
    return path


def reset() -> None:
    """Drop the in-process path and content memos (tests, or a run that changes state_dir)."""
    _PATH_CACHE.clear()
    _MEMO.clear()


# --------------------------------------------------------------------------- fetch: Hermes Index

def _get_text(url: str, timeout: float = 30.0) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "hermes-nim"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def parse_index_page(page: str) -> list[dict]:
    """Rows out of the server-rendered leaderboard table.

    Deliberately narrow: `<td data-column="…">` plus the class the site puts on its numbers. A markup
    redesign makes this return nothing, and `refresh` then keeps the previous rows — a stale table is
    visible in `/nim scores`, whereas an empty one would look like the models simply had no scores.
    """
    rows = [row for row in re.findall(r'<tr[^>]*data-slot="table-row".*?</tr>', page, re.S)
            if "<td" in row]
    parsed: list[dict] = []

    def cell(row: str, name: str) -> str:
        found = re.search(rf'<td[^>]*data-column="{name}"[^>]*>(.*?)</td>', row, re.S)
        return found.group(1) if found else ""

    def plain(fragment: str) -> str:
        return re.sub(r"\s+", " ", html_module.unescape(re.sub(r"<[^>]+>", " ", fragment))).strip()

    for row in rows:
        model_cell = cell(row, "model")
        names = re.findall(r'<span class="truncate">(.*?)</span>', model_cell, re.S)
        vendor = re.findall(r"opacity-60[^>]*>(.*?)</span>", model_cell, re.S)
        index = re.findall(r'class="tabular-nums">([\d.]+)<', cell(row, "index"))
        rank = re.findall(r'class="tabular-nums">(\d+)<', cell(row, "rank"))
        cost = re.findall(r'class="tabular-nums[^"]*">([\d.]+)<', cell(row, "cost"))
        if not names or not index:
            continue
        parsed.append({
            "model_name": plain(names[0]),
            "organization": plain(vendor[0]) if vendor else "",
            "rating": float(index[0]),
            "rank": int(rank[0]) if rank else len(parsed) + 1,
            "cost_per_task": float(cost[0]) if cost else None,
        })
    return parsed


def fetch_rows() -> list[dict]:
    return parse_index_page(_get_text(INDEX_URL))


# --------------------------------------------------------------------------- fetch: LMArena

def arena_search(name: str, length: int = 25) -> list[dict]:
    """Rows for one model name out of the arena dataset.

    `/search` is used instead of `/filter`+offset because the latter 500s at most offsets past zero on
    this 1.2M-row dataset. Retries once: the datasets-server is flaky in a way that is not our bug.
    """
    query = " ".join(str(name).split("/")[-1].replace("-", " ").replace("_", " ").split())
    if not query:
        return []
    url = f"{ARENA_SEARCH}&query={urllib.parse.quote(query)}&offset=0&length={length}"
    last: Exception | None = None
    # Two attempts of 20 s, not three of 45: this runs inside a survey the user is waiting on, and a
    # datasets-server that 500s twice in a row is not going to answer on the third.
    for attempt in range(2):
        try:
            payload = json.loads(_get_text(url, timeout=20.0))
            rows = [row.get("row") or {} for row in (payload.get("rows") or [])]
            return [row for row in rows if str(row.get("category") or "overall") == "overall"]
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(1.0 + attempt)
    logger.warning("hermes-nim: arena search for %r failed: %s", name, last)
    return []


def _contains_either_way(candidate: set[str], tokens: set[str]) -> bool:
    """Is one name's word set inside the other's?

    One direction was not enough: the board lists `NVIDIA-Nemotron-3-Ultra-550B-A55B (NVFP4)` — more
    words than our id — and a subset test that only allowed shorter board names silently dropped the
    model that matters most here. Both directions, and the intersection size decides between candidates.
    """
    return len(candidate) >= 2 and (candidate <= tokens or tokens <= candidate)


def _arena_fields(row: dict) -> dict:
    """The stored shape for one arena row, including the interval the leaderboard publishes.

    `rating_lower`/`rating_upper` are a 95% interval, and they are the difference between a ranking
    and a decoration: 1431 (1428-1435) and 1430 (1424-1436) are the same number with two labels.
    """
    def num(key: str) -> float | None:
        value = row.get(key)
        return float(value) if isinstance(value, (int, float)) else None

    votes = row.get("vote_count")
    return {"value": float(row.get("rating") or 0.0), "rank": int(row.get("rank") or 0),
            "name": str(row.get("model_name") or ""), "org": str(row.get("organization") or ""),
            "lower": num("rating_lower"), "upper": num("rating_upper"),
            "votes": int(votes) if isinstance(votes, (int, float)) else None}


def _arena_entry(rows: list[dict], bare: str) -> dict | None:
    """Best row for this id: exact normalised name first, then the closest word-set match."""
    tokens = _tokens(bare)
    name = _norm(bare)
    best: tuple[int, dict] | None = None
    for row in rows:
        key = _norm(row.get("model_name"))
        if not key:
            continue
        if key == name:
            return {**_arena_fields(row), "inherited": False}
        candidate = _tokens(row.get("model_name"))
        if not _contains_either_way(candidate, tokens):
            continue
        if best is None or len(candidate & tokens) > best[0]:
            best = (len(candidate & tokens), {**_arena_fields(row), "inherited": True})
    return best[1] if best else None


MISS_TTL_DAYS = 7.0     # how long "this model is not on the leaderboard" stays a fact


def refresh_arena(names: list[str], *, max_lookups: int = 24, gap: float = 0.25,
                  budget: float = 12.0) -> dict | None:
    """Look up *names* on LMArena and fold the hits into the cache. Never raises.

    Three rules, all learned by making the survey crawl:

    1. **A miss is cached.** Storing only hits meant every survey re-searched every unrated model —
       six requests of 2-10 s each, on every run, forever, to relearn the same "not on the board".
       Misses are kept per id for ``MISS_TTL_DAYS`` and skipped until they expire.
    2. **The whole pass has a budget.** This runs inside a survey the user is waiting on, so it stops
       at ``budget`` seconds and keeps whatever it has; the rest is looked up next time.
    3. **A render never calls this.** Only a survey does, because a board that blocks on someone
       else's server is a board that looks broken.

    Only called from a survey.
    """
    cache = load() or {}
    arena = dict(cache.get("arena") or {})
    models: dict[str, dict] = dict(arena.get("models") or {})
    misses: dict[str, str] = dict(arena.get("misses") or {})

    looked = 0
    started = time.monotonic()
    for model in names:
        bare = str(model).split("/", 1)[-1]
        key = _norm(bare)
        if not bare or key in models:
            continue
        age = _age_days(misses.get(key))
        if age is not None and age < MISS_TTL_DAYS:
            continue                      # asked recently, still absent: do not ask again
        if looked >= max_lookups or time.monotonic() - started > budget:
            break
        looked += 1
        entry = _arena_entry(arena_search(bare), bare)
        if entry is not None:
            models[key] = entry
            misses.pop(key, None)
        else:
            misses[key] = _now()
        if gap:
            time.sleep(gap)
    if looked:
        arena = {"models": models, "misses": misses, "fetched_at": _now(),
                 "published": _now()[:10], "count": len(models)}
        cache = {**cache, "arena": arena}
        _write(cache)
    return cache or None


# --------------------------------------------------------------------------- cache

def build_cache(rows: list[dict]) -> dict:
    """Rows -> cache. One entry per normalised name; the highest-rated spelling wins a collision."""
    models: dict[str, dict] = {}
    for row in rows:
        key = _norm(row.get("model_name"))
        if not key:
            continue
        entry = {
            "value": float(row.get("rating") or 0.0),
            "rank": int(row.get("rank") or 0),
            "name": str(row.get("model_name") or ""),
            "org": str(row.get("organization") or ""),
            "cost": row.get("cost_per_task"),
        }
        if key not in models or entry["value"] > models[key]["value"]:
            models[key] = entry
    return {"version": CACHE_VERSION, "fetched_at": _now(), "source": INDEX_URL,
            "published": _now()[:10], "count": len(models), "models": models}


def _write(cache: dict) -> None:
    try:
        path = cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache, indent=1, sort_keys=True))
        _MEMO.clear()          # a fresh write must not be served the pre-write copy
    except OSError as exc:  # noqa: BLE001
        logger.warning("hermes-nim: could not cache scores: %s", exc)


def refresh(force: bool = False) -> dict | None:
    """Fetch and cache the Hermes Index. A failed or shrunken fetch leaves the cache alone."""
    cached = load()
    if cached and not force and age_hours(cached) < 0:
        return cached
    try:
        cache = build_cache(fetch_rows())
    except Exception as exc:  # noqa: BLE001 - offline is a state, not a crash
        logger.warning("hermes-nim: index fetch failed: %s", exc)
        return cached
    if not cache.get("models"):
        return cached
    previous = (cached or {}).get("count") or 0
    if previous and cache["count"] < previous * 0.8:
        # A truncated scrape must not replace a full table: losing half the entries to a layout change
        # would quietly degrade every ranking instead of showing up as a problem.
        logger.warning("hermes-nim: index returned %s rows, keeping the cached %s",
                       cache["count"], previous)
        return cached
    merged = {**cache, "arena": (cached or {}).get("arena") or {}}   # keep the other source
    _write(merged)
    return merged


def load() -> dict | None:
    try:
        path = cache_path()
        mtime = path.stat().st_mtime
    except OSError:
        return None  # no cache yet: no score, and no guessing
    key = f"{path}:{mtime}"
    if _MEMO.get("key") == key:
        return _MEMO["data"]      # type: ignore[return-value]
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if not data.get("models") and not (data.get("arena") or {}).get("models"):
        return None
    _MEMO["key"], _MEMO["data"] = key, data
    return data


def age_hours(cache: dict | None) -> float:
    if not cache:
        return float("inf")
    try:
        fetched = datetime.fromisoformat(str(cache.get("fetched_at")).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return float("inf")
    return (datetime.now(timezone.utc) - fetched).total_seconds() / 3600.0


def maybe_refresh(max_age_hours: float = 24.0) -> dict | None:
    """Called by a survey: fetch only when the cache is missing or old. Never raises."""
    cached = load()
    if cached and age_hours(cached) < max_age_hours:
        return cached
    return refresh(force=True) or cached


def sources_meta(cache: dict | None = None) -> list[dict]:
    """What `/nim scores` prints as provenance: one line per source, with its own date and count."""
    cache = cache or load() or {}
    arena = cache.get("arena") or {}
    return [
        {"key": INDEX_SOURCE, "source": "Hermes Index", "glyph": INDEX_GLYPH,
         "scale": "0-100 agentic index",
         "published": str(cache.get("published") or ""), "count": int(cache.get("count") or 0),
         "url": INDEX_URL},
        {"key": ARENA_SOURCE, "source": "LMArena", "glyph": ARENA_GLYPH,
         "scale": "Elo, public leaderboard",
         "published": str(arena.get("published") or ""), "count": int(arena.get("count") or 0),
         "url": "https://lmarena.ai/leaderboard"},
    ]


# --------------------------------------------------------------------------- match

def _match(models: dict[str, dict], name: str, tokens: set[str]) -> tuple[dict, bool] | None:
    """Exact normalised name, then the closest word-set match (either direction)."""
    found = models.get(name)
    if found:
        return found, bool(found.get("inherited"))
    best: tuple[int, dict] | None = None
    for key, entry in models.items():
        candidate = _tokens(entry.get("name") or key)
        if not _contains_either_way(candidate, tokens):
            continue
        if best is None or len(candidate & tokens) > best[0]:
            best = (len(candidate & tokens), entry)
    return (best[1], True) if best else None


def lookup(model: str, cache: dict | None = None, *, source: str | None = None) -> Score | None:
    """The published score for *model*.

    Index first, LMArena only when the index is silent — unless `source` names one explicitly, which
    is what `/nim scores` does when it lists each source separately.
    """
    cache = cache if cache is not None else load()
    if not cache or not model:
        return None
    bare = str(model).split("/", 1)[-1]
    name = _norm(bare)
    if not name:
        return None
    tokens = _tokens(bare)
    order = [source] if source else [INDEX_SOURCE, ARENA_SOURCE]
    for candidate in order:
        if candidate == INDEX_SOURCE:
            table, published = cache.get("models") or {}, str(cache.get("published") or "")
            cost_of = lambda e: e.get("cost")  # noqa: E731
        elif candidate == ARENA_SOURCE:
            arena = cache.get("arena") or {}
            table, published = arena.get("models") or {}, str(arena.get("published") or "")
            cost_of = lambda e: None  # noqa: E731
        else:
            continue
        matched = _match(table, name, tokens)
        if matched is None:
            continue
        found, inherited = matched
        return Score(found["value"], found["rank"], found["name"], found["org"], published,
                     inherited, cost_of(found), candidate,
                     found.get("lower"), found.get("upper"), found.get("votes"))
    return None


def coverage(models: list[str], cache: dict | None = None) -> tuple[int, int]:
    """(how many of *models* carry any published score, how many were asked about)."""
    cache = cache if cache is not None else load()
    scored = sum(1 for model in models if lookup(model, cache) is not None)
    return scored, len(models)


__all__ = ["Score", "SOURCE", "GLYPH", "GLYPHS", "INDEX_SOURCE", "INDEX_GLYPH", "ARENA_SOURCE",
           "ARENA_GLYPH", "CACHE_NAME", "parse_index_page", "fetch_rows", "arena_search",
           "refresh_arena", "build_cache", "refresh", "load", "age_hours", "maybe_refresh",
           "sources_meta", "lookup", "coverage", "cache_path", "reset"]
