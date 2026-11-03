"""Background scanner: keeps the NIM state fresh without a cron job.

The scanner runs on the gateway's event loop as a supervised task. It does two things
on independent timers:

1. Config scan — probes the models your config actually uses (fallback chain + shortlist)
   every N minutes. This is the data the fallback chain and /nim probe read.

2. Survey scan — probes every model NVIDIA lists every M minutes. This is what keeps
   /nim catalog, /nim added, /nim dead, and the podium fresh.

Both write the same state.json the engine uses, so there's no second source of truth.
The cron job becomes a formatter only — it reads the already-fresh state and sends
a message if anything changed.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

from . import engine, settings

logger = logging.getLogger("hermes-nim.background")


@dataclass(frozen=True)
class BackgroundConfig:
    """Timing for the two independent scan loops."""

    # How often to scan your config models (fallback chain + shortlist).
    # 10 min default: fast enough to catch a stalled fallback before you notice,
    # slow enough that a 10s TTFT cap on 11 models costs < 2% of a CPU.
    config_scan_interval_min: float = 10.0

    # How often to survey the whole catalog (all 85 models).
    # 60 min default: NVIDIA adds/removes models maybe weekly; 1h is plenty.
    survey_interval_min: float = 60.0

    # First run delay after plugin load — stagger so we don't hammer NIM on startup.
    initial_delay_sec: float = 5.0


async def _scan_config_models(s: settings.Settings) -> dict:
    """Scan only the models your config points at (plus shortlist)."""
    # engine.configured_models takes an optional config_path; state_dir is not a config path.
    used = engine.configured_models()
    # Merge configured with shortlist for the scan targets
    extra = s.shortlist_models()
    return await asyncio.to_thread(
        engine.scan,
        base=s.api_base,
        ttft=s.ttft_seconds,
        attempts=s.attempts,
        stagger=s.stagger_seconds,
        extra=extra,
        configured_dir=s.state_dir,
        recheck_ttft=s.recheck_ttft_seconds,
        survey=False,
    )


async def _scan_full_survey(s: settings.Settings) -> dict:
    """Scan every model NVIDIA lists."""
    return await asyncio.to_thread(
        engine.scan,
        base=s.api_base,
        ttft=s.survey_ttft_seconds,
        attempts=s.survey_attempts,
        stagger=s.stagger_seconds,
        extra=s.shortlist_models(),
        configured_dir=s.state_dir,
        recheck_ttft=0.0,
        survey=True,
        survey_ttft=s.survey_ttft_seconds,
    )


async def _background_loop(cfg: BackgroundConfig) -> None:
    """Run both scan loops until cancelled."""
    # Initial delay to let the gateway settle
    await asyncio.sleep(cfg.initial_delay_sec)

    # Load settings once per loop iteration (they can change via config reload)
    next_config_scan = 0.0
    next_survey = 0.0
    loop_start = asyncio.get_event_loop().time()

    while True:
        now = asyncio.get_event_loop().time()
        elapsed = now - loop_start

        # Load current settings (respects live config changes)
        s = settings.settings()

        # --- Config scan ---
        if elapsed >= next_config_scan:
            logger.info("hermes-nim: starting config scan")
            try:
                await _scan_config_models(s)
                logger.info("hermes-nim: config scan done")
            except Exception as exc:  # noqa: BLE001 - a failed scan must not kill the loop
                logger.warning("hermes-nim: config scan failed: %s", exc)
            next_config_scan = elapsed + cfg.config_scan_interval_min * 60

        # --- Survey scan ---
        if elapsed >= next_survey:
            logger.info("hermes-nim: starting catalog survey")
            try:
                await _scan_full_survey(s)
                logger.info("hermes-nim: catalog survey done")
            except Exception as exc:  # noqa: BLE001
                logger.warning("hermes-nim: catalog survey failed: %s", exc)
            next_survey = elapsed + cfg.survey_interval_min * 60

        # Sleep until the next scheduled scan (whichever is sooner)
        next_due = min(next_config_scan, next_survey)
        sleep_for = max(1.0, next_due - elapsed)
        await asyncio.sleep(sleep_for)


_task: Optional[asyncio.Task] = None


def start_background_scanner(ctx, cfg: Optional[BackgroundConfig] = None) -> None:
    """Spawn the supervised background scanner task.

    Called from plugin.register(). The task is attached to the plugin context so
    unload/force-reload cancels it cleanly.
    """
    global _task
    if _task is not None and not _task.done():
        logger.debug("hermes-nim: background scanner already running")
        return

    cfg = cfg or BackgroundConfig()

    async def _runner():
        try:
            await _background_loop(cfg)
        except asyncio.CancelledError:
            logger.info("hermes-nim: background scanner cancelled")
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("hermes-nim: background scanner crashed: %s", exc)

    _task = ctx.spawn_task(_runner(), name=f"plugin:{ctx.plugin_id}:background-scanner")
    logger.info(
        "hermes-nim: background scanner started (config_every=%.1fm survey_every=%.1fm)",
        cfg.config_scan_interval_min,
        cfg.survey_interval_min,
    )


def stop_background_scanner() -> None:
    """Cancel the scanner task (called on unload)."""
    global _task
    if _task is not None and not _task.done():
        _task.cancel()
        _task = None
        logger.info("hermes-nim: background scanner stopped")


__all__ = ["BackgroundConfig", "start_background_scanner", "stop_background_scanner"]
