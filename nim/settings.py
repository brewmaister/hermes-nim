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
    # `/nim` measures by default: a cached verdict with an age on it still needs the reader to do
    # arithmetic about whether it means anything. `/nim cached` is the instant view.
    live_default: bool = True
    # Reuse a scan this young instead of probing again — a double-tap on /nim should not launch two
    # full probes at the same model.
    live_grace_seconds: float = 20.0
    # Re-confirming a death is a cheaper question than the first verdict: a model already marked dead
    # is probed with this shorter first-byte cap instead of ttft_seconds. 0 disables the shortcut.
    recheck_ttft_seconds: float = 8.0
    # One attempt for the interactive view: two samples doubled the worst case to ~50 s, and a model
    # that black-holes for 25 s is unusable in chat whether or not a second attempt agrees. The
    # hysteresis still needs two *runs* before anything is called dead, so this cannot fake a death.
    live_attempts: int = 1
    # "auto" follows NIM_LANGUAGE → HERMES_LANGUAGE → display.language; or pin "en"/"es".
    language: str = "auto"

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
        if os.environ.get("NIM_LIVE"):
            changes["live_default"] = os.environ["NIM_LIVE"].strip().lower() not in {
                "0", "false", "no", "off"}
        if os.environ.get("NIM_LANGUAGE"):
            changes["language"] = os.environ["NIM_LANGUAGE"]
        if os.environ.get("NIM_LIVE_ATTEMPTS"):
            try:
                changes["live_attempts"] = max(1, int(os.environ["NIM_LIVE_ATTEMPTS"]))
            except ValueError:
                pass
        if os.environ.get("NIM_LIVE_GRACE"):
            try:
                changes["live_grace_seconds"] = max(0.0, float(os.environ["NIM_LIVE_GRACE"]))
            except ValueError:
                pass
        if os.environ.get("NIM_RECHECK_TTFT"):
            try:
                changes["recheck_ttft_seconds"] = max(0.0, float(os.environ["NIM_RECHECK_TTFT"]))
            except ValueError:
                pass
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


def _as_bool(value, fallback: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return fallback
    return str(value).strip().lower() not in {"0", "false", "no", "off"}


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
            live_default=_as_bool(configured.get("live_default"), values.live_default),
            live_grace_seconds=_as_float(configured.get("live_grace_seconds"),
                                         values.live_grace_seconds),
            recheck_ttft_seconds=_as_float(configured.get("recheck_ttft_seconds"),
                                           values.recheck_ttft_seconds),
            live_attempts=_as_int(configured.get("live_attempts"), values.live_attempts),
            language=str(configured.get("language") or values.language),
        )
    return values.with_env()


def describe() -> str:
    """The settings block under /nim help — translated, because half a translated page is a bug."""
    from .i18n import t

    current = settings()
    models = ", ".join(current.shortlist_models()) or t("settings.none")
    lines = [
        t("settings.intro"),
        t("settings.ttft", value=f"{current.ttft_seconds:g}"),
        t("settings.attempts", value=current.attempts),
        t("settings.state_dir", value=current.state_dir or "~/hermes-outputs/nim-probe"),
        t("settings.shortlist", value=models),
        t("settings.live_default", value=current.live_default),
        t("settings.live_grace", value=f"{current.live_grace_seconds:g}"),
        t("settings.recheck", value=f"{current.recheck_ttft_seconds:g}"),
        t("settings.live_attempts", value=current.live_attempts, attempts=current.attempts),
        t("settings.language", value=current.language),
    ]
    return "\n".join(lines)


__all__ = ["Settings", "settings", "configure", "describe"]
