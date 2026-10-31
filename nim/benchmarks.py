"""The quality prior — a compatibility surface over `nim.scores`, which fetches it.

History worth keeping: this module used to hold a hand-typed table of three Artificial Analysis
scores, under the rule "only add a score you can point at". The rule was right; the maintenance model
was not. A typed table is accurate on the day it is written and silently stale afterwards — no update
path, no date check, nothing to notice when a model was re-rated or withdrawn.

It is now mechanical: `nim.scores` pulls LMArena's public leaderboard dataset (a server-filtered JSON
API), caches it with a fetch timestamp, and refreshes on a schedule. What remains non-mechanical is the
id → arena-name match, which is why every prefix match is flagged `inherited`.

Two scales are never mixed. This module reports LMArena Elo only; the old AA index is gone rather than
kept alongside, because a column of numbers drawn from two sources is a lie about comparability.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import scores

SOURCE = scores.SOURCE


@dataclass(frozen=True)
class Benchmark:
    index: float          # LMArena Elo, overall text leaderboard
    label: str            # the arena's own name for the model, so a match can be judged
    context: int | None   # LMArena publishes no context window; kept for interface compatibility
    source: str
    inherited: bool = False  # True when the score is the family's, not this exact variant's


def as_of() -> str:
    """The leaderboard's publish date — not our fetch date. Empty when nothing is cached."""
    cache = scores.load() or {}
    return str(cache.get("published") or "")


def lookup(model: str) -> Benchmark | None:
    """The arena rating for *model*, or None when nothing published covers it."""
    found = scores.lookup(model)
    if found is None:
        return None
    return Benchmark(index=found.elo, label=found.name, context=None, source=SOURCE,
                     inherited=found.inherited)


def quality(model: str) -> float | None:
    """Comparable quality number: the Elo, or None to fall back on id size."""
    found = lookup(model)
    return found.index if found else None


def describe(model: str) -> str:
    """A short, honest tag for the model's line: `arena 1487` or nothing at all."""
    found = lookup(model)
    if found is None:
        return ""
    return f"arena {found.index:.0f}{' family' if found.inherited else ''}"


def known_models() -> list[str]:
    cache = scores.load() or {}
    return sorted((cache.get("models") or {}).keys())


__all__ = ["Benchmark", "SOURCE", "as_of", "lookup", "quality", "describe", "known_models"]
