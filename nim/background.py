"""Background scanner: keeps the NIM state fresh without a cron job.

Two independent loops, both writing the same ``state.json`` the engine already uses:

1. **Config scan** — the models your config points at (fallback chain + shortlist).
2. **Survey scan** — every model NVIDIA lists.

Why a plain daemon thread and not ``ctx.spawn_task`` or a startup hook: plugin ``register()`` runs on
a loader worker thread that has no running event loop (``get_running_loop()`` raises there), and a
``gateway:startup`` hook registered during a reload never fires again because the gateway already
started. The result was a scanner that logged "starting" and then probed nothing at all. A thread
with its own ``asyncio.run`` loop starts from every load path — first boot, force reload, CLI — and
asks the host for nothing.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass
from typing import Optional

from . import engine, settings

# __name__, not a hand-written name: the gateway installs its handlers on loggers under
# hermes_plugins.*, so a logger called "hermes-nim.background" logs into the void.
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BackgroundConfig:
    """Timing for the two independent scan loops."""

    # How often to scan your config models (fallback chain + shortlist). 10 min catches a stalled
    # fallback before you notice, and costs well under 2% of one core.
    config_scan_interval_min: float = 10.0
    # How often to survey the whole catalog. NVIDIA adds or removes models maybe weekly.
    survey_interval_min: float = 60.0
    # First run delay after load, so a gateway restart does not fire ~96 probes at once.
    initial_delay_sec: float = 5.0


async def _scan_config_models(current: settings.Settings) -> dict:
    """Probe only the models your config points at (plus the shortlist)."""
    return await asyncio.to_thread(
        engine.scan,
        base=current.api_base,
        ttft=current.ttft_seconds,
        attempts=current.attempts,
        stagger=current.stagger_seconds,
        extra=current.shortlist_models(),
        configured_dir=current.state_dir,
        recheck_ttft=current.recheck_ttft_seconds,
        survey=False,
    )


async def _scan_full_survey(current: settings.Settings) -> dict:
    """Probe every model NVIDIA lists."""
    return await asyncio.to_thread(
        engine.scan,
        base=current.api_base,
        ttft=current.survey_ttft_seconds,
        attempts=current.survey_attempts,
        stagger=current.stagger_seconds,
        extra=current.shortlist_models(),
        configured_dir=current.state_dir,
        recheck_ttft=0.0,
        survey=True,
        survey_ttft=current.survey_ttft_seconds,
    )


async def _sleep(seconds: float, stop: threading.Event) -> bool:
    """Sleep in slices so a stop request lands promptly. True means it is time to exit."""
    remaining = max(0.0, seconds)
    while remaining > 0:
        if stop.is_set():
            return True
        await asyncio.sleep(min(1.0, remaining))
        remaining -= 1.0
    return stop.is_set()


async def _background_loop(cfg: BackgroundConfig, stop: threading.Event) -> None:
    """Run both loops until *stop* is set."""
    if await _sleep(cfg.initial_delay_sec, stop):
        return
    next_config_scan = 0.0
    next_survey = 0.0
    started = asyncio.get_event_loop().time()

    while not stop.is_set():
        elapsed = asyncio.get_event_loop().time() - started
        current = settings.settings()

        if cfg.config_scan_interval_min > 0 and elapsed >= next_config_scan:
            logger.info("hermes-nim: config scan starting")
            try:
                report = await _scan_config_models(current)
                logger.info("hermes-nim: config scan done (%d probed)",
                            len(report.get("probed") or []))
            except Exception as exc:  # noqa: BLE001 - one bad sweep must not kill the loop
                logger.warning("hermes-nim: config scan failed: %s", exc)
            next_config_scan = elapsed + cfg.config_scan_interval_min * 60

        if cfg.survey_interval_min > 0 and elapsed >= next_survey:
            logger.info("hermes-nim: catalog survey starting")
            try:
                report = await _scan_full_survey(current)
                survey = report.get("survey") or {}
                logger.info("hermes-nim: catalog survey done (%s of %s online)",
                            survey.get("online"), survey.get("listed"))
            except Exception as exc:  # noqa: BLE001
                logger.warning("hermes-nim: catalog survey failed: %s", exc)
            next_survey = elapsed + cfg.survey_interval_min * 60

        due = [moment for moment, interval in (
            (next_config_scan, cfg.config_scan_interval_min),
            (next_survey, cfg.survey_interval_min),
        ) if interval > 0]
        if not due:  # both loops disabled: nothing left to wait for
            return
        if await _sleep(min(due) - elapsed, stop):
            return


_stop = threading.Event()
_thread: Optional[threading.Thread] = None


def _thread_main(cfg: BackgroundConfig) -> None:
    try:
        asyncio.run(_background_loop(cfg, _stop))
    except Exception:  # noqa: BLE001 - a dead scanner must not take the gateway with it
        logger.exception("hermes-nim: background scanner crashed")


def start_background_scanner(ctx=None, cfg: Optional[BackgroundConfig] = None) -> None:
    """Start the scanner thread, once. A second call while it runs does nothing."""
    global _thread
    if _thread is not None and _thread.is_alive():
        logger.debug("hermes-nim: background scanner already running")
        return
    cfg = cfg or BackgroundConfig()
    _stop.clear()
    _thread = threading.Thread(target=_thread_main, args=(cfg,),
                               name="hermes-nim-scanner", daemon=True)
    _thread.start()
    logger.info("hermes-nim: background scanner started (%s min config / %s min survey)",
                cfg.config_scan_interval_min, cfg.survey_interval_min)


def stop_background_scanner() -> None:
    """Ask the scanner thread to finish: an in-flight sweep completes, then the loop exits."""
    global _thread
    if _thread is not None and _thread.is_alive():
        _stop.set()
        logger.info("hermes-nim: background scanner stopping")
    _thread = None


__all__ = ["BackgroundConfig", "start_background_scanner", "stop_background_scanner"]
