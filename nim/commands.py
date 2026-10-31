"""The two doors onto the engine: ``/nim`` in chat and ``hermes nim`` from a shell.

Both render through :mod:`nim.render`, so the chat reply and the terminal output cannot drift.
"""

from __future__ import annotations

import argparse
import json as jsonlib
import os
import sys

from . import engine, render, settings, version
from .i18n import t

# Help text is localized too: a user who reads /nim in Spanish should not get English help.
USAGE_KEYS = ("nim", "cached", "probe", "speed", "catalog", "vendor", "scores", "survey", "dead",
              "added", "cfg", "json", "legend", "tail")


def usage() -> str:
    return "\n".join(t(f"usage.{key}") for key in USAGE_KEYS)


def _print(text: str) -> None:
    """print(), but a closed pipe is not a crash.

    `hermes nim probe | head -3` is a normal thing to type, and without this the shell gets a
    BrokenPipeError traceback and a non-zero exit for a command that did exactly what was asked.
    The devnull dup2 is Python's documented workaround: it also silences the interpreter's final
    flush of the closed stream.
    """
    try:
        print(text)
    except BrokenPipeError:
        try:
            os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        except Exception:  # noqa: BLE001 - nothing useful left to do
            pass


def _verb_and_rest(raw: str) -> tuple[str, list[str]]:
    pieces = [piece for piece in (raw or "").split() if piece]
    if not pieces:
        # Empty means "the default verb", and the default is now a live scan. Mapping it to
        # "cached" here silently disabled the whole feature: /nim rendered the cache every time.
        return "", []
    return pieces[0].lower(), pieces[1:]


def _days(args: list[str], fallback: float = 7.0) -> float:
    for arg in args:
        try:
            return max(0.0, float(arg))
        except ValueError:
            continue
    return fallback


def do_scan(*, attempts: int | None = None, workers: int | None = None) -> dict:
    current = settings.settings()
    return engine.scan(
        base=current.api_base,
        ttft=current.ttft_seconds,
        attempts=current.attempts if attempts is None else attempts,
        stagger=current.stagger_seconds,
        extra=current.shortlist_models(),
        configured_dir=current.state_dir,
        recheck_ttft=current.recheck_ttft_seconds,
        workers=workers,
    )


def do_survey() -> dict:
    """Measure every model NVIDIA lists, once each — the online/offline board, on demand."""
    current = settings.settings()
    return engine.scan(
        base=current.api_base,
        ttft=current.ttft_seconds,
        attempts=current.survey_attempts,
        stagger=current.stagger_seconds,
        extra=current.shortlist_models(),
        configured_dir=current.state_dir,
        recheck_ttft=current.recheck_ttft_seconds,
        workers=current.survey_workers,
        survey=True,
        survey_ttft=current.survey_ttft_seconds,
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
        recheck_ttft=current.recheck_ttft_seconds,
    )


def dispatch(verb: str, extra_args: list[str], *, force_scan: bool = False) -> tuple[str, int]:
    """Return (text, exit_code). Exit: 0 healthy · 1 something unusable · 2 could not scan."""
    current = settings.settings()
    if verb in {"help", "-h", "--help", "?"}:
        return usage() + "\n\n" + settings.describe(), 0
    if verb in {"version", "-v", "--version"}:
        return f"🧩 {version.stamp()} — {version.in_gateway_hint()}", 0
    if verb == "cfg":
        used = engine.configured_models()
        return render.render_cfg(used, str(engine.hermes_home() / "config.yaml")), 0
    if verb == "json":
        report = engine.load_state(current.state_dir)
        if not report:
            return t("empty.no_scan_probe"), 2
        return jsonlib.dumps(report, indent=1), 0

    if verb == "speed":
        attempts = int(_days(extra_args, 3))
        report = do_speed_scan(attempts)
        if report.get("error"):
            return t("speed.error", error=report["error"]), 2
        return render.render_speed(report), 0

    # `/nim` measures. A cached verdict needs the reader to judge whether "22 min ago" still means
    # anything; the whole point of this plugin is the wire, so the default path uses it.
    if verb in {"", "status", "live", "cached"}:
        if verb == "cached" or not current.live_default:
            report = engine.load_state(current.state_dir)
            return (render.render_board(report) if current.survey_catalog
                    else render.render_cached(report)), 0
        cached = engine.load_state(current.state_dir)
        age = engine.scan_age_seconds(cached)
        reusable = (verb != "live" and cached.get("models") and age is not None
                    and age < current.live_grace_seconds)
        if reusable and current.survey_catalog and not cached.get("survey"):
            # A cached config-only scan cannot stand in for a survey: the board would show five
            # verdicts under a header claiming the catalog was measured.
            reusable = False
        if reusable:
            # A double-tap must not fire a second survey: that one is still about now.
            return (render.render_board(cached) if current.survey_catalog
                    else render.render_cached(cached)), 0
        if current.survey_catalog:
            # The default door is the survey: "what can I use?" is answered by measuring everything
            # NVIDIA lists, not by checking the four models already in config.
            report = do_survey()
            if report.get("error"):
                return t("empty.error", error=report["error"]), 2
            return render.render_board(report), 0
        # One attempt and a wide fan-out: the floor is one ttft cap, not two waves of two attempts.
        report = do_scan(attempts=current.live_attempts, workers=12)
        if report.get("error"):
            return t("empty.error", error=report["error"]), 2
        return render.render_probe(report), 0

    report = None
    if force_scan or verb == "probe":
        report = do_scan()
        if report.get("error"):
            return t("empty.error", error=report["error"]), 2
    else:
        report = engine.load_state(current.state_dir)

    if verb in {"cached", "status", ""}:
        return render.render_cached(report), 0
    if verb == "probe":
        text = render.render_probe(report)
        return text, (1 if _dead(report) else 0)
    if verb == "catalog":
        return render.render_catalog(report), 0
    if verb == "vendor":
        return render.render_vendors(report), 0
    if verb == "scores":
        return render.render_scores(report), 0
    if verb == "survey":
        surveyed = do_survey()
        if surveyed.get("error"):
            return t("empty.error", error=surveyed["error"]), 2
        return render.render_catalog(surveyed), (1 if engine.removed_ids(surveyed) else 0)
    if verb == "dead":
        return render.render_dead(report), (1 if _dead(report) else 0)
    if verb == "added":
        return render.render_added(report, _days(extra_args)), 0
    return t("cmd.unknown", verb=verb) + "\n\n" + usage(), 2


def _dead(report: dict) -> list[dict]:
    """Dead models you point at. A stranger's 404 is not a broken config, so it is not exit code 1."""
    watched = render.watched(report)
    return [result for result in (watched.get("models") or {}).values()
            if engine.severity(result) == "dead"]


# ---------------------------------------------------------------------------- slash command

def slash_command(raw_args: str = "", *, session_id: str = "", **_ignored) -> str:
    verb, extra = _verb_and_rest(raw_args)
    text, _code = dispatch(verb, extra)
    return text


def will_measure(verb: str) -> bool:
    """Would this verb hit the network? Used to decide whether to say "measuring" first."""
    current = settings.settings()
    if verb in {"probe", "speed", "live"}:
        return True
    if verb in {"", "status"}:
        return current.live_default
    return False


def measuring_note() -> str:
    """The one line you see for ~1 s while the scan runs — same vocabulary as the result.

    It used to say "measuring 5 models live", which is the same unexplained count the result line just
    stopped printing. Badge comes from the last survey, so the placeholder already tells you how much
    of the catalog is even usable.
    """
    current = settings.settings()
    if current.survey_catalog:
        listed = len(engine.load_state(current.state_dir).get("catalog") or []) or 85
        return t("progress.surveying", listed=listed,
                 seconds=f"{(listed / max(1, current.survey_workers)) * current.survey_ttft_seconds:.0f}")
    models = len(engine.configured_models()) + len(current.shortlist_models())
    board = render.board_badge(engine.load_state(current.state_dir))
    return t("progress.measuring", models=models, ttft=f"{current.ttft_seconds:g}", board=board)


def slash_command_with_progress(ctx, raw_args: str = "", **kwargs) -> str | None:
    """The chat door: say "measuring" first, then edit that same message with the result.

    A live probe is 15-40 s of silence, and Telegram gives no spinner — silence that long reads as a
    broken bot. Posting one line and editing it in place keeps the chat clean (no leftover
    "measuring…" message) and works the same on a surface that cannot edit: then the note stays and
    the report arrives as its own message.
    """
    verb, extra = _verb_and_rest(raw_args)
    if not will_measure(verb):
        return slash_command(raw_args, **kwargs)
    handle = None
    try:
        handle = ctx.post_message(measuring_note())
    except Exception:  # noqa: BLE001 - a progress note must never break the command
        handle = None
    text = slash_command(raw_args, **kwargs)
    if handle is not None and getattr(handle, "ok", False):
        try:
            handle.edit(text)
            return None   # already delivered, in place
        except Exception:  # noqa: BLE001
            pass
    return text


# ---------------------------------------------------------------------------- CLI door

def setup_argparse(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "verb",
        nargs="?",
        default="",
        choices=["", "live", "cached", "probe", "speed", "catalog", "vendor", "scores", "survey",
                 "dead", "added", "cfg", "json", "watch", "status", "version", "help"],
        help="What to print (default: live scan; `cached` prints the last scan with no network)",
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

    verb = str(getattr(args, "verb", "") or "")
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
        _print(jsonlib.dumps(report, indent=1))
        return code
    _print(render.plain(text))
    return code


if __name__ == "__main__":  # pragma: no cover - convenience for `python -m nim.commands`
    sys.exit(0)
