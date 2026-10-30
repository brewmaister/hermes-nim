"""Published quality scores, kept deliberately small and dated.

Why a static table and not a live lookup: NIM exposes no benchmarks and no metadata beyond
`id`/`owned_by`, so any quality prior has to come from somewhere outside the API. This is that
somewhere — and it is *external published evidence*, not something this plugin measured. Every entry
names its source and the date it was read, because a stale benchmark silently becomes a lie.

Rules for this file:

1. **Only add a score you can point at.** No interpolation, no "this is probably about the same as
   its predecessor".
2. **Family matches must say so.** `minimaxai/minimax-m3` covers the M3 line; a variant with no
   published score of its own inherits the family number *and is labelled as inherited*, so nobody
   reads "29" as a measurement of the Flash variant.
3. **Id-size fallback stays.** A model with no entry still ranks by the parameter count in its id —
   a weak prior, but an honest one, and better than pretending every unlisted model is equal.

Artificial Analysis Intelligence Index, read 2026-10-30:
https://artificialanalysis.ai/models/comparisons/nvidia-nemotron-3-ultra-550b-a55b-vs-minimax-m3
"""

from __future__ import annotations

from dataclasses import dataclass

AS_OF = "2026-10-30"


@dataclass(frozen=True)
class Benchmark:
    index: float          # Artificial Analysis Intelligence Index
    label: str            # what the number actually measures, in the user's words
    context: int | None   # published context window, tokens
    source: str
    inherited: bool = False  # True when the score is the family's, not this exact variant's


# Prefix-matched, longest prefix wins. Keep entries in descending specificity.
_TABLE: tuple[tuple[str, Benchmark], ...] = (
    (
        "minimaxai/minimax-m3.1",
        Benchmark(
            index=29,
            label="MiniMax-M3 family score (M3.1-Flash variant not published separately)",
            context=1_000_000,
            source="artificialanalysis.ai/models/comparisons/"
                   "nvidia-nemotron-3-ultra-550b-a55b-vs-minimax-m3",
            inherited=True,
        ),
    ),
    (
        "minimaxai/minimax-m3",
        Benchmark(
            index=29,
            label="MiniMax-M3",
            context=1_000_000,
            source="artificialanalysis.ai/models/comparisons/"
                   "nvidia-nemotron-3-ultra-550b-a55b-vs-minimax-m3",
        ),
    ),
    (
        "nvidia/nemotron-3-ultra",
        Benchmark(
            index=23,
            label="Nemotron 3 Ultra 550B A55B (Reasoning)",
            context=262_144,
            source="artificialanalysis.ai/models/nvidia-nemotron-3-ultra-550b-a55b",
        ),
    ),
)


def lookup(model: str) -> Benchmark | None:
    """The published score for *model*, or None when nothing here covers it."""
    if not model:
        return None
    best: tuple[str, Benchmark] | None = None
    for prefix, benchmark in _TABLE:
        if model.startswith(prefix) and (best is None or len(prefix) > len(best[0])):
            best = (prefix, benchmark)
    return best[1] if best else None


def quality(model: str) -> float | None:
    """Comparable quality number: the published index, or None to fall back on id size."""
    found = lookup(model)
    return found.index if found else None


def describe(model: str) -> str:
    """A short, honest tag for the model's line: `AA 29` or nothing at all."""
    found = lookup(model)
    if found is None:
        return ""
    return f"AA {found.index:g}{' family' if found.inherited else ''}"


def known_models() -> list[str]:
    return [prefix for prefix, _ in _TABLE]


__all__ = ["AS_OF", "Benchmark", "lookup", "quality", "describe", "known_models"]
