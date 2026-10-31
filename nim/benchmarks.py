"""The quality prior — a compatibility surface over `nim.scores`, which fetches the Hermes Index.

History worth keeping: this module used to hold a hand-typed table of three Artificial Analysis scores
under the rule "only add a score you can point at". The rule was right; the maintenance model was not.
A typed table is accurate the day it is written and silently stale afterwards.

It is now mechanical: `nim.scores` parses portal.nousresearch.com/bench (the Hermes Index — agentic task
completion and cost per task, measured inside Hermes Agent), caches it with a fetch timestamp, and a
survey refetches it on a schedule. One source on purpose: a second leaderboard meant a second scale, and
two scales in one column is a lie about comparability.

What remains non-mechanical is the id → leaderboard-name match, which is why every family match is
flagged `inherited`: `Nemotron Super 3.5` scoring 24.1 is the family's number, not the variant's.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import scores

SOURCE = scores.SOURCE
GLYPH = scores.GLYPH


@dataclass(frozen=True)
class Benchmark:
    index: float             # Hermes Index, 0-100
    label: str               # the leaderboard's own name for the model, so a match can be judged
    context: int | None      # the board publishes no context window; kept for the interface
    source: str
    published: str = ""
    inherited: bool = False  # True when the score is the family's, not this exact variant's
    cost: float | None = None

    @property
    def glyph(self) -> str:
        return scores.GLYPHS.get(self.source, GLYPH)

    @property
    def scaled(self) -> str:
        """The number in its own units: one decimal for the 0-100 index, integer for Elo."""
        return f"{self.index:.1f}" if self.source == scores.INDEX_SOURCE else f"{self.index:.0f}"


def as_of() -> str:
    """The leaderboard date we last fetched. Empty when nothing is cached."""
    cache = scores.load() or {}
    return str(cache.get("published") or "")


def lookup(model: str, source: str | None = None) -> Benchmark | None:
    """The published score for *model* — index first, LMArena when the index is silent."""
    found = scores.lookup(model, source=source)
    if found is None:
        return None
    return Benchmark(index=found.value, label=found.name, context=None, source=found.source,
                     published=found.published, inherited=found.inherited, cost=found.cost)


def quality(model: str) -> float | None:
    """Comparable quality number: the Hermes Index, or None to fall back on id size."""
    found = lookup(model)
    return found.index if found else None


def describe(model: str) -> str:
    """A short, honest tag for the model's line: `🧠24.1` or nothing at all."""
    found = lookup(model)
    if found is None:
        return ""
    return f"{found.glyph}{found.scaled}{' family' if found.inherited else ''}"


def known_models() -> list[str]:
    cache = scores.load() or {}
    return sorted((cache.get("models") or {}).keys())


__all__ = ["Benchmark", "SOURCE", "GLYPH", "as_of", "lookup", "quality", "describe", "known_models"]
