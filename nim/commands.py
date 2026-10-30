"""The two doors onto the engine: ``/nim`` in chat and ``hermes nim`` from a shell.

Both render through :mod:`nim.render`, so the chat reply and the terminal output cannot drift.
"""

from __future__ import annotations

import argparse
import json as jsonlib
import sys

from . import engine, render, settings, version

USAGE = (
    "/nim — cached scan (instant)\n"
    "/nim probe — live scan of the models your config uses + the shortlist\n"
    "            (~30-60 s: a model that never answers costs two 25 s stalls)\n"
    "/nim catalog — what /v1/models lists, and what dropped out of it\n"
    "/nim dead — everything currently unusable\n"
    "/nim added [days] — models that appeared recently (default 7)\n"
    "/nim speed [n] — timed scan: first byte and full answer, median of n attempts (default 3)\n"
    "/nim cfg — which NIM models config.yaml references, no network\n"
    "/nim json — the last scan as JSON\n"
    "/nim version · /nim help"
)


def _verb_and_rest(raw: str) -> tuple[str, list[str]]:
    pieces = [piece for piece in (raw or "").split() if piece]
    if not pieces:
        return "cached", []
    return pieces[0].lower(), pieces[1:]


def _days(args: list[str], fallback: float = 7.0) -> float:
    for arg in args:
        try:
            return max(0.0, float(arg))
        except ValueError:
            continue
    return fallback


def do_scan() -> dict:
    current = settings.settings()
    return engine.scan(
        base=current.api_base,
        ttft=current.ttft_seconds,
        attempts=current.attempts,
        stagger=current.stagger_seconds,
        extra=current.shortlist_models(),
        configured_dir=current.state_dir,
    )


def do_speed_scan(attempts: int) -> dict:
    """A timed scan: more attempts than a health check, because one sample is not a speed."""
    current = settings.settings()
    return engine.scan(
        base=current.api_base,
        ttft=current.ttft_seconds,
        attempts=max(1, min(6, attempts)),
        stagger=current.stagger_seconds,
        extra=current.shortlist_models(),
        configured_dir=current.state_dir,
    )


def dispatch(verb: str, extra_args: list[str], *, force_scan: bool = False) -> tuple[str, int]:
    """Return (text, exit_code). Exit: 0 healthy · 1 something unusable · 2 could not scan."""
    current = settings.settings()
    if verb in {"help", "-h", "--help", "?"}:
        return USAGE + "\n\n" + settings.describe(), 0
    if verb in {"version", "-v", "--version"}:
        return f"🧩 {version.stamp()} — {version.in_gateway_hint()}", 0
    if verb == "cfg":
        used = engine.configured_models()
        return render.render_cfg(used, str(engine.hermes_home() / "config.yaml")), 0
    if verb == "json":
        report = engine.load_state(current.state_dir)
        if not report:
            return ("🧠 NIM — no scan yet; run /nim probe first."), 2
        return jsonlib.dumps(report, indent=1), 0

    if verb == "speed":
        attempts = int(_days(extra_args, 3))
        report = do_speed_scan(attempts)
        if report.get("error"):
            return f"⚡ NIM speed — {report['error']}", 2
        return render.render_speed(report), 0

    report = None
    if force_scan or verb == "probe":
        report = do_scan()
        if report.get("error"):
            return f"🧠 NIM — {report['error']}", 2
    else:
        report = engine.load_state(current.state_dir)

    if verb in {"cached", "status", ""}:
        return render.render_cached(report), 0
    if verb == "probe":
        text = render.render_probe(report)
        return text, (1 if _dead(report) else 0)
    if verb == "catalog":
        return render.render_catalog(report), 0
    if verb == "dead":
        return render.render_dead(report), (1 if _dead(report) else 0)
    if verb == "added":
        return render.render_added(report, _days(extra_args)), 0
    return f"Unknown argument '{verb}'.\n\n{USAGE}", 2


def _dead(report: dict) -> list[dict]:
    return [result for result in (report.get("models") or {}).values()
            if engine.severity(result) == "dead"]


# ---------------------------------------------------------------------------- slash command

def slash_command(raw_args: str = "", *, session_id: str = "", **_ignored) -> str:
    verb, extra = _verb_and_rest(raw_args)
    text, _code = dispatch(verb, extra)
    return text


# ---------------------------------------------------------------------------- CLI door

def setup_argparse(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "verb",
        nargs="?",
        default="cached",
        choices=["cached", "status", "probe", "speed", "catalog", "dead", "added", "cfg", "json",
                 "watch", "version", "help"],
        help="What to print (default: cached — the last scan, no network)",
    )
    parser.add_argument("days", nargs="?", default=None,
                        help="For `added`: how many days back; for `speed`: how many attempts")
    parser.add_argument("--json", action="store_true", help="Print the report as JSON")
    parser.add_argument("--ttft", type=float, default=None,
                        help="Override seconds-to-first-byte before a model counts as stalled")
    parser.add_argument("--attempts", type=int, default=None,
                        help="Override calls per model (default 2)")
    parser.add_argument("--verbose", action="store_true",
                        help="For `watch`: print the full table, not only changes")
    parser.add_argument("--send", action="store_true",
                        help="For `watch`: deliver the report through `hermes send`")
    parser.add_argument("--version", action="store_true", help="Print the plugin version and exit")


def handle_cli(args: argparse.Namespace) -> int:
    if getattr(args, "version", False):
        print(version.stamp())
        return 0
    # CLI overrides ride the env layer: settings() already gives env precedence, and this keeps
    # the resolution order in exactly one place.
    import os

    if getattr(args, "ttft", None) is not None:
        os.environ["NIM_TTFT"] = str(args.ttft)
    if getattr(args, "attempts", None) is not None:
        os.environ["NIM_ATTEMPTS"] = str(args.attempts)

    verb = str(getattr(args, "verb", "cached") or "cached")
    if verb == "watch":
        from .watch import run_watch

        return run_watch(verbose=bool(getattr(args, "verbose", False)),
                         send=bool(getattr(args, "send", False)))

    extra = []
    days = getattr(args, "days", None)
    if days not in (None, ""):
        extra.append(str(days))
    text, code = dispatch(verb, extra)
    if getattr(args, "json", False) and verb != "json":
        report = engine.load_state(settings.settings().state_dir)
        print(jsonlib.dumps(report, indent=1))
        return code
    print(text)
    return code


if __name__ == "__main__":  # pragma: no cover - convenience for `python -m nim.commands`
    sys.exit(0)
