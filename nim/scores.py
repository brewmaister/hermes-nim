"""Quality prior, fetched instead of typed.

Why this module exists: `benchmarks.py` used to be a hand-kept table of three Artificial Analysis
scores. It was accurate the day it was written and silently wrong afterwards — the one part of this
plugin that could rot without anything noticing, and the only part that was not mechanical.

LMArena publishes its leaderboard as a public HuggingFace dataset, and the datasets-server exposes it
as JSON with a server-side filter, so the prior can be *fetched*: 400-odd ranked models, each with an
Elo rating, a vote count and a publish date, refreshed on our own schedule and cached on disk.

What is still not mechanical, and is therefore labelled on screen:
  * matching a NIM id to an arena name is a normalisation heuristic (exact, else longest shared
    prefix). A prefix match is marked `inherited` — it is the family's rating, not this variant's.
  * LMArena measures human preference in pairwise battles. It is not a capability benchmark, and the
    number is never compared with a number from another scale.

Failures are not fatal: no cache and no network means no score, and every view says "0 of N
published" rather than inventing one.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

DATASET = "lmarena-ai/leaderboard-dataset"
CONFIG = "text"
SPLIT = "latest"
CATEGORY = "overall"
PAGE = 100
MAX_PAGES = 8                      # 800 rows is well past the current 419; a bound beats a hang
SOURCE = ("https://datasets-server.huggingface.co/filter?dataset=lmarena-ai%2Fleaderboard-dataset"
          f"&config={CONFIG}&split={SPLIT}&where=%22category%22%3D%27{CATEGORY}%27")
CACHE_NAME = "scores.json"
CACHE_VERSION = 1


@dataclass(frozen=True)
class Score:
    elo: float
    rank: int
    name: str        # the arena's name for the model, so the match can be judged
    org: str
    published: str   # leaderboard publish date, not our fetch time
    inherited: bool  # True when this is a family/prefix match, not an exact one


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _norm(text: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text or "").lower())


_PATH_CACHE: dict[str, Path] = {}


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

def _get(url: str, timeout: float = 30.0) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "hermes-nim"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def fetch_rows() -> list[dict]:
    """Every overall text-leaderboard row, paginated.

    A page that fails stops the walk but does not discard what came back: the datasets-server 500s
    under load now and then, and 400 rows plus a warning beats throwing away a usable leaderboard.
    Raises only when nothing at all was retrieved.
    """
    rows: list[dict] = []
    total = None
    for page in range(MAX_PAGES):
        url = f"{SOURCE}&offset={page * PAGE}&length={PAGE}"
        try:
            payload = _get(url)
        except Exception as exc:  # noqa: BLE001
            if rows:
                logger.warning("hermes-nim: leaderboard page %s failed (%s); keeping %s rows",
                               page + 1, exc, len(rows))
                break
            raise
        batch = [item.get("row") or {} for item in payload.get("rows") or []]
        total = payload.get("num_rows_total") or total
        rows.extend(row for row in batch if row.get("model_name"))
        if not batch or (total is not None and len(rows) >= total):
            break
    return rows


def build_cache(rows: list[dict]) -> dict:
    """Rows -> cache. One entry per normalised name; the highest-rated spelling wins a collision."""
    models: dict[str, dict] = {}
    published = ""
    for row in rows:
        key = _norm(row.get("model_name"))
        if not key:
            continue
        published = str(row.get("leaderboard_publish_date") or published)
        entry = {
            "elo": float(row.get("rating") or 0.0),
            "rank": int(row.get("rank") or 0),
            "name": str(row.get("model_name") or ""),
            "org": str(row.get("organization") or ""),
            "published": str(row.get("leaderboard_publish_date") or ""),
        }
        if key not in models or entry["elo"] > models[key]["elo"]:
            models[key] = entry
    return {"version": CACHE_VERSION, "fetched_at": _now(), "source": SOURCE,
            "published": published, "count": len(models), "models": models}


def refresh(force: bool = False) -> dict | None:
    """Fetch and cache the leaderboard. Returns the cache, or None when the fetch failed."""
    cached = load()
    if cached and not force and age_hours(cached) < 0:
        return cached
    try:
        cache = build_cache(fetch_rows())
    except Exception as exc:  # noqa: BLE001 - offline is a state, not a crash
        logger.warning("hermes-nim: leaderboard fetch failed: %s", exc)
        return cached
    if not cache.get("models"):
        return cached
    try:
        path = cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache, indent=1, sort_keys=True))
    except OSError as exc:  # noqa: BLE001
        logger.warning("hermes-nim: could not cache scores: %s", exc)
    return cache


# Renders ask for a score per row; resolving the path means reading config.yaml and stat-ing a file.
# A memo keyed on (path, mtime) keeps a board render off the disk without ever serving a stale fetch.
_MEMO: dict[str, object] = {}


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

def lookup(model: str, cache: dict | None = None) -> Score | None:
    """The arena score for *model*, exact match first, then the longest shared prefix.

    A prefix match is labelled `inherited`: `MiniMax M3.1` says nothing about the Flash variant's
    own rating, so the number must never be read as a measurement of it.
    """
    cache = cache if cache is not None else load()
    if not cache or not model:
        return None
    models = cache.get("models") or {}
    name = _norm(str(model).split("/", 1)[-1])
    if not name:
        return None
    found = models.get(name)
    if found:
        return Score(found["elo"], found["rank"], found["name"], found["org"],
                     found["published"], False)
    best_key = None
    for key in models:
        if len(key) >= 8 and name.startswith(key) and (best_key is None or len(key) > len(best_key)):
            best_key = key
    if best_key is None:
        return None
    found = models[best_key]
    return Score(found["elo"], found["rank"], found["name"], found["org"], found["published"], True)


def coverage(models: list[str], cache: dict | None = None) -> tuple[int, int]:
    cache = cache if cache is not None else load()
    scored = sum(1 for model in models if lookup(model, cache) is not None)
    return scored, len(models)


__all__ = ["Score", "DATASET", "SOURCE", "CACHE_NAME", "fetch_rows", "build_cache", "refresh",
           "load", "age_hours", "maybe_refresh", "lookup", "coverage", "cache_path"]
