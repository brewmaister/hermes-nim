"""Plugin settings: the manifest's config_schema values, plus env overrides for cron and one-offs.

Every knob has a working default, so the plugin behaves the same on a fresh box as it does here.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, replace

from .engine import DEFAULT_BASE, hermes_home

logger = logging.getLogger("hermes-nim.settings")

_CTX = None


@dataclass(frozen=True)
class Settings:
    api_base: str = DEFAULT_BASE
    ttft_seconds: float = 25.0
    attempts: int = 2
    stagger_seconds: float = 1.5
    state_dir: str = ""
    # Empty by default, on purpose: a plugin that probes models you never configured is spending your
    # latency on its author's taste. `/nim` then measures exactly the models config.yaml references,
    # and the candidate pool grows only when you name something here. (`/nim catalog` still reports
    # what NVIDIA lists, without probing it.)
    shortlist: tuple[str, ...] = ()
    # `/nim` measures by default: a cached verdict with an age on it still needs the reader to do
    # arithmetic about whether it means anything. `/nim cached` is the instant view.
    live_default: bool = True
    # Reuse a scan this young instead of probing again — a double-tap on /nim should not launch two
    # full probes at the same model.
    live_grace_seconds: float = 20.0
    # Re-confirming a death is a cheaper question than the first verdict: a model already marked dead
    # is probed with this shorter first-byte cap instead of ttft_seconds. 0 disables the shortcut.
    recheck_ttft_seconds: float = 8.0
    # A survey measures every model NVIDIA lists, so it must not inherit the fallback-chain cap: a
    # catalog sweep that waited 25 s on each black hole would take an hour. 10 s is enough to tell
    # "answers" from "does not", which is the only question the catalog view asks.
    survey_ttft_seconds: float = 10.0
    survey_workers: int = 20
    # Every `/nim` sweeps the whole catalog: the question "what can I actually use?" is a survey, not
    # a peek at the four models already in config. Costs ~17 s instead of ~1 s, and the answer is the
    # whole board rather than a footnote.
    survey_catalog: bool = True
    survey_attempts: int = 1
    # One attempt for the interactive view: two samples doubled the worst case to ~50 s, and a model
    # that black-holes for 25 s is unusable in chat whether or not a second attempt agrees. The
    # hysteresis still needs two *runs* before anything is called dead, so this cannot fake a death.
    live_attempts: int = 1
    # The quality prior is fetched, not typed: a survey re-reads the Hermes Index when the
    # cache is older than scores_max_age_hours. False pins whatever is cached and stops the fetch.
    # What the star means: the model to point Hermes at for chat. First byte must clear
    # recommend_max_ttft_s, and a *scored* model must also write at least recommend_min_tps to keep its
    # top tier — a smart model that crawls is still recommended, just below one that does not.
    recommend_max_ttft_s: float = 5.0
    recommend_min_tps: float = 25.0
    scores_refresh: bool = True
    # Second opinion, never a blended number: LMArena (🏛, Elo) fills in where the Hermes Index (🧠)
    # has no row. Each keeps its own glyph, its own scale and its own publish date.
    arena_scores: bool = True
    # What the chat list is sorted by: "arena" (🏛 Elo — widest coverage), "index" (🧠 agentic),
    # or "speed" (measured ✍️). Models the chosen source does not rate sort below the ones it does,
    # so two scales never interleave in one column.
    order_by: str = "arena"
    scores_max_age_hours: float = 24.0
    # "auto" follows NIM_LANGUAGE → HERMES_LANGUAGE → display.language; or pin "en"/"es".
    language: str = "auto"
    # Background scanner (no cron needed). 0 disables that loop.
    background_config_scan_min: float = 10.0
    background_survey_min: float = 60.0

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


def _config_entry_from_file() -> dict:
    """The plugin's settings out of ``$HERMES_HOME/config.yaml``, for the ctx-less doors.

    `hermes nim` and the cron watchdog run outside the gateway, so there is no ctx to ask, and
    reading nothing meant they silently used the defaults while `/nim` in chat honoured the config.
    Two doors disagreeing about the shortlist is a bug, not a nuance.
    """
    path = hermes_home() / "config.yaml"
    try:
        import yaml

        raw = yaml.safe_load(path.read_text()) or {}
    except FileNotFoundError:
        return {}
    except Exception as exc:  # noqa: BLE001 - a broken config must not break the command
        logger.warning("hermes-nim: could not read %s: %s", path, exc)
        return {}
    entry = ((raw.get("plugins") or {}).get("entries") or {}).get("hermes-nim") or {}
    return entry.get("settings") or {}


def reset_cache() -> None:
    """Kept for callers that expect a cache hook; settings are resolved per call on purpose."""
    return None


def _configured() -> dict:
    """Settings dict from the live ctx, or from config.yaml when there is no ctx."""
    if _CTX is None:
        return _config_entry_from_file()
    try:
        raw = _CTX.get_config() or {}
    except Exception:  # noqa: BLE001
        return _config_entry_from_file()
    entry = ((raw.get("plugins") or {}).get("entries") or {}).get("hermes-nim") or {}
    return entry.get("settings") or {}


def settings() -> Settings:
    values = Settings()
    configured = _configured()
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
        survey_ttft_seconds=_as_float(configured.get("survey_ttft_seconds"),
                                      values.survey_ttft_seconds),
        survey_workers=_as_int(configured.get("survey_workers"), values.survey_workers),
        survey_catalog=_as_bool(configured.get("survey_catalog"), values.survey_catalog),
        live_attempts=_as_int(configured.get("live_attempts"), values.live_attempts),
        language=str(configured.get("language") or values.language),
        background_config_scan_min=_as_float(configured.get("background_config_scan_min"), values.background_config_scan_min),
        background_survey_min=_as_float(configured.get("background_survey_min"), values.background_survey_min),
    )
    return values.with_env()


def describe() -> str:
    """The settings block under /nim help — translated, because half a translated page is a bug.

    Rich markdown on purpose: this block and the command list above it are the two things a reader
    copies out of Telegram, and only the rich path sends raw markdown the client renders itself, so a
    copy keeps headings and monospace. It also means no column padding — rich rendering collapses runs
    of spaces, so a padded column would arrive ragged. One bullet per setting instead.
    """
    from .i18n import t

    current = settings()
    # Grouped the same way as the verbs: measuring, the live door, the survey, ranking, files. Every
    # setting keeps the reason it holds its value — a knob whose why is missing is a knob you cannot
    # safely change.
    def row(key: str, value: object, why: str | None = None, **fmt: object) -> str:
        head = f"- `{key}` {value}"
        if not why:
            return head
        reason = t(f"settings.why_{why}", **{name: str(value) for name, value in fmt.items()})
        return f"{head} — {reason}"

    lines = [
        t("settings.intro"),
        t("settings.intro_sub"),
        "",
        t("settings.g_measure"),
        row("ttft_seconds", f"{current.ttft_seconds:g}", "ttft"),
        row("attempts", current.attempts, "attempts"),
        row("recheck_ttft_seconds", f"{current.recheck_ttft_seconds:g}", "recheck"),
        "",
        t("settings.g_live"),
        row("live_default", current.live_default, "live_default"),
        row("live_attempts", current.live_attempts, "live_attempts", attempts=current.attempts),
        row("live_grace_seconds", f"{current.live_grace_seconds:g}", "live_grace"),
        "",
        t("settings.g_survey"),
        row("survey_catalog", current.survey_catalog, "survey", workers=current.survey_workers,
            cap=f"{current.survey_ttft_seconds:g}"),
        "",
        t("settings.g_rank"),
        row("order_by", current.order_by, "order"),
        row("recommend_max_ttft_s", f"{current.recommend_max_ttft_s:g}", "recommend"),
        row("recommend_min_tps", f"{current.recommend_min_tps:g}", "min_tps"),
        row("scores_refresh", current.scores_refresh, "scores",
            hours=f"{current.scores_max_age_hours:g}"),
        row("arena_scores", current.arena_scores, "arena"),
        "",
        t("settings.g_files"),
        row("state_dir", current.state_dir or "~/hermes-outputs/nim-probe"),
        # One id per line: a comma run of seven 30-character ids is a paragraph, not a list.
        t("settings.shortlist", count=len(current.shortlist_models())),
        *[t("settings.shortlist_item", model=model) for model in current.shortlist_models()],
        row("language", current.language, "language"),
        "",
        t("settings.g_background"),
        row("background_config_scan_min", f"{current.background_config_scan_min:g}", "bg_config"),
        row("background_survey_min", f"{current.background_survey_min:g}", "bg_survey"),
    ]
    return "\n".join(lines)


__all__ = ["Settings", "settings", "configure", "describe"]
