"""hermes-nim — NVIDIA NIM model health, in chat.

Registers:
  * ``/nim`` — cached scan, live scan, catalog, dead, added, and the config view;
  * ``hermes nim`` — the same engine as a CLI door, for scripts and cron.

Why this exists: NVIDIA publishes no usable lifecycle signal for its NIM free endpoints. Model
cards keep advertising "Free Endpoint — Available" for ids that return zero bytes,
``/v1/models`` lists models this account cannot call, and status.ngc.nvidia.com only tracks
infrastructure and stays green throughout. The only trustworthy signal is the wire, so this
plugin probes it.

The imports are deliberately relative: the loader imports this directory as
``hermes_plugins.hermes_nim`` and evicts that prefix on a force reload. An absolute import would
cache the engine under a name the eviction never touches, so an edit would appear to do nothing
until the gateway restarted.
"""

from __future__ import annotations

import logging
import os

from .nim.commands import (handle_cli, setup_argparse, slash_command,
                           slash_command_with_progress)
from .nim.settings import configure as configure_settings
from .nim.version import PLUGIN_VERSION
from .nim.background import start_background_scanner, stop_background_scanner, BackgroundConfig

logger = logging.getLogger(__name__)


def register(ctx) -> None:
    logger.info("hermes-nim %s loaded (pid %s)", PLUGIN_VERSION, os.getpid())
    configure_settings(ctx)

    # Start the background scanner — keeps state.json fresh without a cron job.
    # Intervals come from settings (config.yaml → plugins.entries.hermes-nim.settings).
    # Try to start immediately if the gateway loop is already running; otherwise
    # register a gateway:startup hook to start after the gateway is fully up.
    from .nim.settings import settings as _current_settings
    s = _current_settings()
    logger.info("hermes-nim: starting background scanner (config_every=%.1fm survey_every=%.1fm)",
                s.background_config_scan_min, s.background_survey_min)

    def _do_start_scanner():
        from .nim.settings import settings as _s
        _settings = _s()
        start_background_scanner(ctx, BackgroundConfig(
            config_scan_interval_min=_settings.background_config_scan_min,
            survey_interval_min=_settings.background_survey_min,
        ))

    # Where the loop is decides how the scanner starts. Plugin register() often runs on a loader
    # worker thread with no loop (so `get_running_loop()` raises), and `gateway:startup` alone would
    # then only fire on the NEXT gateway start — the scanner silently never ran after a reload.
    # session:start / agent:start fire on the loop for every turn, and start_background_scanner is
    # idempotent, so the first message after a reload is what brings the scanner up.
    import asyncio

    try:
        asyncio.get_running_loop()
        _do_start_scanner()
    except RuntimeError:
        for hook in ("gateway:startup", "session:start", "agent:start"):
            ctx.register_hook(hook, lambda _payload: _do_start_scanner())

    def _chat_command(raw_args: str = "", **kwargs):
        """Bind the plugin context so the handler can say "measuring…" before a live probe.

        The wrapper edits that one message with the result instead of leaving a dangling progress
        line, and returns None in that case — the gateway then sends nothing further.
        """
        return slash_command_with_progress(ctx, raw_args, **kwargs)

    ctx.register_command(
        "nim",
        handler=_chat_command,
        description="NVIDIA NIM model health: what is alive, stalled, retired or new (hermes-nim)",
        args_hint="[cached|probe|speed|catalog|dead|added|cfg|json|version|help]",
    )
    ctx.register_cli_command(
        name="nim",
        help="NVIDIA NIM model health — which endpoints actually answer (hermes-nim plugin)",
        setup_fn=setup_argparse,
        handler_fn=handle_cli,
        description="NIM model health (hermes-nim)",
    )

    # Clean shutdown: cancel the background scanner on unload/force-reload
    ctx.on_unload(stop_background_scanner)


__all__ = ["register"]
