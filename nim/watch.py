"""The unattended door: a `no_agent` cron job whose STDOUT is the message.

Contract, and it is the whole contract: **print nothing when there is nothing to report.** An empty
stdout makes the cron engine send nothing at all. A watcher that always prints something gets muted
within a week, and then a real alert is invisible.

Everything expensive is guarded by the scan's own hysteresis (see :func:`nim.engine.scan`): a model
must look dead on two consecutive runs before it is called dead, and each condition is reported once
until it recovers. Without that, NIM's replica-level flapping produces alarm spam every tick.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from . import engine, render, settings


def _header(report: dict) -> str:
    from datetime import datetime, timezone

    stamp = datetime.now(timezone.utc).strftime("%b %d %H:%M UTC")
    return f"NIM watchdog — {stamp}"


def render_watch(report: dict, *, verbose: bool = False) -> str:
    """The cron message: what changed, and what is not usable right now."""
    if report.get("error"):
        return f"NIM watchdog — {report['error']}"

    changes = report.get("changes") or []
    # A survey measures the whole catalog, most of which is 404 for this account. The cron message is
    # about the models you actually point at — otherwise every tick is 70 lines of other people's
    # models and you mute the watcher.
    watched = set(report.get("configured_in") or {})
    dead = [result for result in (report.get("models") or {}).values()
            if engine.severity(result) == "dead" and result["model"] in watched]
    if not changes and not verbose:
        return ""

    lines = [_header(report)]
    if changes:
        lines += [""] + changes
    if dead:
        used = report.get("configured_in") or {}
        lines += ["", "Not usable (models you point at):"]
        for result in sorted(dead, key=lambda r: r["model"]):
            lines.append(f"• {result['model']} — {engine.label(result)}")
            where = render.short_where(used.get(result["model"]) or [])
            if where and where != "shortlist":
                lines.append(f"   in config: {where}")
        lines += ["", "Model cards won't warn you — swap the id (/nim added shows candidates)."]
    if verbose:
        lines += ["", render.summary_line(report)]
    return "\n".join(lines)


def deliver(text: str) -> None:
    """Fallback delivery for a non-cron caller. The cron engine delivers stdout itself, so a job
    must NOT also pass --send, or every alert arrives twice."""
    binary = Path.home() / ".local" / "bin" / "hermes"
    if not binary.exists():
        return
    try:
        subprocess.run([str(binary), "send", "-t", "telegram", "--", text],
                       check=False, timeout=90, capture_output=True, text=True)
    except Exception:  # noqa: BLE001 - delivery is best-effort here; cron owns the real path
        pass


def run_watch(*, verbose: bool = False, send: bool = False) -> int:
    current = settings.settings()
    report = engine.scan(
        base=current.api_base,
        ttft=current.ttft_seconds,
        attempts=current.attempts,
        stagger=current.stagger_seconds,
        extra=current.shortlist_models(),
        configured_dir=current.state_dir,
        # The cron sweep uses `attempts` per model, so this is the difference between 2x25 s and
        # 2x8 s per dead model — on a 6h tick that is the whole point of rechecking cheaply.
        recheck_ttft=current.recheck_ttft_seconds,
        # The 6h sweep is the survey: it is the only door that measures the whole catalog, which is
        # what keeps "who is online right now" true without you maintaining a list of names.
        survey=current.survey_catalog,
        survey_ttft=current.survey_ttft_seconds,
    )
    text = render_watch(report, verbose=verbose)
    if text:
        print(text)
        if send:
            deliver(text)
    return 0
