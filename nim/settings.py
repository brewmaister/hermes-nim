"""Plugin settings: the manifest's config_schema values, plus env overrides for cron and one-offs.

Every knob has a working default, so the plugin behaves the same on a fresh box as it does here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace

from .engine import DEFAULT_BASE

_CTX = None


@dataclass(frozen=True)
class Settings:
    api_base: str = DEFAULT_BASE
    ttft_seconds: float = 25.0
    attempts: int = 2
    stagger_seconds: float = 1.5
    state_dir: str = ""
    # The candidate pool for swaps — not in config, so without a default `/nim probe` would only
    # ever look at models already in use and never surface a better one.
    shortlist: tuple[str, ...] = ("minimaxai/minimax-m3.1-flash", "meta/muse-glimmer-30b",
                                  "stepfun-ai/step-5-preview")

    def with_env(self) -> "Settings":
        """Env wins over config: cron scripts have no ctx, and ad-hoc runs need to override."""
        changes: dict = {}
        if os.environ.get("NIM_API_BASE"):
            changes["api_base"] = os.environ["NIM_API_BASE"]
        if os.environ.get("NIM_TTFT"):
            try:
                changes["ttft_seconds"] = float(os.environ["NIM_TTFT"])
            except ValueError:
                pass
        if os.environ.get("NIM_ATTEMPTS"):
            try:
                changes["attempts"] = max(1, int(os.environ["NIM_ATTEMPTS"]))
            except ValueError:
                pass
        if os.environ.get("NIM_STATE_DIR"):
            changes["state_dir"] = os.environ["NIM_STATE_DIR"]
        if os.environ.get("NIM_SHORTLIST"):
            changes["shortlist"] = tuple(
                part.strip() for part in os.environ["NIM_SHORTLIST"].split(",") if part.strip())
        return replace(self, **changes) if changes else self

    def shortlist_models(self) -> list[str]:
        return [model for model in self.shortlist if model]


def _as_float(value, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _as_int(value, fallback: int) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return fallback


def configure(ctx) -> None:
    """Called once from register(): keep the ctx and resolve the manifest settings."""
    global _CTX
    _CTX = ctx


def settings() -> Settings:
    values = Settings()
    if _CTX is not None:
        try:
            raw = _CTX.get_config() or {}
        except Exception:  # noqa: BLE001
            raw = {}
        entry = ((raw.get("plugins") or {}).get("entries") or {}).get("hermes-nim") or {}
        configured = entry.get("settings") or {}
        shortlist_raw = str(configured.get("shortlist") or "") or ",".join(values.shortlist)
        values = Settings(
            api_base=str(configured.get("api_base") or values.api_base),
            ttft_seconds=_as_float(configured.get("ttft_seconds"), values.ttft_seconds),
            attempts=_as_int(configured.get("attempts"), values.attempts),
            stagger_seconds=_as_float(configured.get("stagger_seconds"), values.stagger_seconds),
            state_dir=str(configured.get("state_dir") or ""),
            shortlist=tuple(
                part.strip() for part in shortlist_raw.split(",") if part.strip()),
        )
    return values.with_env()


def describe() -> str:
    current = settings()
    models = ", ".join(current.shortlist_models()) or "none"
    return (
        "settings (config.yaml → plugins.entries.hermes-nim.settings, or NIM_* env):\n"
        f"· ttft_seconds: {current.ttft_seconds:g} (below 20 misreads cold starts as dead)\n"
        f"· attempts: {current.attempts}\n"
        f"· state_dir: {current.state_dir or '~/hermes-outputs/nim-probe'}\n"
        f"· shortlist: {models}"
    )


__all__ = ["Settings", "settings", "configure", "describe"]
