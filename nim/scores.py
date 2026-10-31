"""Quality prior, fetched instead of typed.

`benchmarks.py` used to hold three hand-typed Artificial Analysis numbers. Accurate the day they were
written, silently wrong afterwards, and the only part of this plugin that was not mechanical.

The prior now comes from the **Hermes Index** — portal.nousresearch.com/bench, an agentic index (task
completion and cost per task, measured inside Hermes Agent). It is server-rendered HTML, so this module
parses the leaderboard table, caches it to disk with a fetch timestamp, and a survey refetches it when
the cache is old.

One source on purpose. LMArena was wired in alongside it for a while and removed: two leaderboards mean
two scales, and a board that shows `24.1` next to `1431` is asking the reader to do arithmetic that
does not exist. Percentiles papered over it; one source is honest.

What is not mechanical, and is therefore labelled on screen: matching a NIM id to a leaderboard name.
Exact normalised match wins; otherwise the most specific shared token set, marked as a family match —
`Nemotron Super 3.5` is the family, `nemotron-3.5-super-vl-120b-a12b` is the variant.

Failures are not fatal: no cache and no network means no score, and every view says "0 of N published"
rather than inventing one. A fetch that returns fewer rows than the cache keeps the cache.
"""

from __future__ import annotations

import html as html_module
import json
import logging
import re
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

CACHE_NAME = "scores.json"
CACHE_VERSION = 3

INDEX_URL = "https://portal.nousresearch.com/bench"
SOURCE = INDEX_URL
GLYPH = "🧠"


@dataclass(frozen=True)
class Score:
    value: float
    rank: int
    name: str        # the leaderboard's own name for the model, so a match can be judged
    org: str
    published: str
    inherited: bool  # True when this is a family match, not an exact one
    cost: float | None = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _norm(text: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text or "").lower())


def _tokens(text: object) -> set[str]:
    """Words of a model name, punctuation dropped: `Nemotron Super 3.5` -> {nemotron, super, 3, 5}.

    A prefix matcher was fine until the leaderboard called it `Nemotron Super 3.5` while the NIM id is
    `nemotron-3.5-super-vl-120b-a12b` — same words, different order. Comparing token sets handles the
    order without loosening the rule into substring soup.
    """
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


# --------------------------------------------------------------------------- fetch

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
    return {"version": CACHE_VERSION, "fetched_at": _now(), "source": SOURCE,
            "published": _now()[:10], "count": len(models), "models": models}


def refresh(force: bool = False) -> dict | None:
    """Fetch and cache the leaderboard. A failed or shrunken fetch leaves the cache alone."""
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
    try:
        path = cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache, indent=1, sort_keys=True))
    except OSError as exc:  # noqa: BLE001
        logger.warning("hermes-nim: could not cache scores: %s", exc)
    return cache


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
    if not data.get("models"):
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


# --------------------------------------------------------------------------- match

def _match(models: dict[str, dict], name: str, tokens: set[str]) -> tuple[dict, bool] | None:
    """Exact normalised name, then the most specific token-subset match."""
    found = models.get(name)
    if found:
        return found, False
    best: tuple[int, dict] | None = None
    for key, entry in models.items():
        candidate = _tokens(entry.get("name") or key)
        if len(candidate) < 2 or not candidate <= tokens:
            continue
        if best is None or len(candidate) > best[0]:
            best = (len(candidate), entry)
    return (best[1], True) if best else None


def lookup(model: str, cache: dict | None = None) -> Score | None:
    """The published score for *model*, or None when the leaderboard does not cover it."""
    cache = cache if cache is not None else load()
    if not cache or not model:
        return None
    bare = str(model).split("/", 1)[-1]
    name = _norm(bare)
    if not name:
        return None
    matched = _match(cache.get("models") or {}, name, _tokens(bare))
    if matched is None:
        return None
    found, inherited = matched
    return Score(found["value"], found["rank"], found["name"], found["org"],
                 str(cache.get("published") or ""), inherited, found.get("cost"))


def coverage(models: list[str], cache: dict | None = None) -> tuple[int, int]:
    cache = cache if cache is not None else load()
    scored = sum(1 for model in models if lookup(model, cache) is not None)
    return scored, len(models)


__all__ = ["Score", "SOURCE", "GLYPH", "CACHE_NAME", "parse_index_page", "fetch_rows", "build_cache",
           "refresh", "load", "age_hours", "maybe_refresh", "lookup", "coverage", "cache_path",
           "reset"]
