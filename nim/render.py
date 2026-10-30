"""Rendering: one place that turns a report into words, so chat, CLI and cron cannot drift.

Bullets, no tables: these land in Telegram and a terminal alike, and a table dies at 40 columns.
"""

from __future__ import annotations

from datetime import datetime, timezone

from . import engine

MAX_LINES = 26  # keep a chat reply readable; the artifact files hold the full picture


def human_age(seconds: float | None) -> str:
    if seconds is None:
        return "unknown age"
    seconds = max(0.0, seconds)
    if seconds < 90:
        return f"{int(seconds)}s ago"
    if seconds < 5400:
        return f"{int(seconds / 60)} min ago"
    if seconds < 172800:
        return f"{seconds / 3600:.1f} h ago"
    return f"{seconds / 86400:.1f} d ago"


def short_where(where: list[str]) -> str:
    """Config paths are precise and unreadable; report the role instead."""
    labels: list[str] = []
    for path in where:
        if path.startswith("fallback_providers["):
            index = path.split("[")[1].split("]")[0]
            labels.append(f"fallback {int(index) + 1}")
        elif path.startswith("delegation"):
            labels.append("free route")
        elif path.startswith("moa"):
            labels.append("moa reference")
        elif path == "shortlist":
            labels.append("shortlist")
        elif path:
            labels.append(path)
    seen: list[str] = []
    for label in labels:
        if label not in seen:
            seen.append(label)
    return ", ".join(seen)


def _grouped(report: dict) -> tuple[list[dict], list[dict], list[dict]]:
    models = list((report.get("models") or {}).values())
    alive = [m for m in models if engine.severity(m) == "ok"]
    warn = [m for m in models if engine.severity(m) == "warn"]
    dead = [m for m in models if engine.severity(m) == "dead"]
    return alive, warn, dead


def _size_tag(result: dict) -> str:
    """The vendor's own parameter count, when the id states one."""
    size = engine.params_from_id(str(result.get("model") or ""))
    return f"{size:g}B · " if size is not None else ""


def _line(result: dict, used: dict[str, list[str]], mark: str, best: bool = False) -> str:
    where = short_where((used or {}).get(result["model"]) or [])
    suffix = f" · {where}" if where else ""
    tail = " ← best" if best else ""
    return f"{mark} {result['model']} — {_size_tag(result)}{engine.label(result)}{suffix}{tail}"


def model_lines(report: dict, limit: int = MAX_LINES) -> list[str]:
    used = report.get("configured_in") or {}
    alive, warn, dead = _grouped(report)
    lines: list[str] = []
    # Best-first, and the ordering is all facts: parameter count from the model id (NVIDIA's own
    # number), then how many attempts answered, then measured latency. Nothing invented.
    for position, result in enumerate(sorted(alive, key=engine.rank_key)):
        lines.append(_line(result, used, "✅", best=position == 0))
    for result in warn:
        lines.append(_line(result, used, "⚠️"))
    for result in dead:
        lines.append(_line(result, used, "❌"))
    if len(lines) > limit:
        hidden = len(lines) - limit
        lines = lines[:limit] + [f"… and {hidden} more (see /nim catalog)"]
    return lines


def best_line(report: dict) -> str:
    """One line naming the top pick and exactly why it is on top."""
    alive = [result for result in (report.get("models") or {}).values()
             if engine.severity(result) == "ok"]
    if not alive:
        return "best right now: nothing — no model answered every attempt"
    top = sorted(alive, key=engine.rank_key)[0]
    size = engine.params_from_id(str(top.get("model") or ""))
    why = [f"{top.get('ok_attempts')}/{top.get('attempts')} answered", f"{top.get('ttft_s')}s"]
    if size is not None:
        why.insert(0, f"{size:g}B")
    return f"best right now: {top['model']} — " + ", ".join(why)


def changes_block(report: dict) -> list[str]:
    changes = report.get("changes") or []
    if not changes:
        return []
    return ["", "since the last scan:"] + [f"• {change}" for change in changes]


def render_probe(report: dict) -> str:
    if report.get("error"):
        return f"🧠 NIM — {report['error']}"
    models = report.get("models") or {}
    header = (f"🧠 NIM scan — {len(models)} models · "
              f"{report.get('attempts')} attempt(s) each · "
              f"just now")
    lines = [header, ""] + model_lines(report) + changes_block(report)
    lines += ["", best_line(report), _catalog_footnote(report)]
    return "\n".join(lines)


def _catalog_footnote(report: dict) -> str:
    size = len(report.get("catalog") or [])
    if not report.get("catalog_fetch_ok", True) and not size:
        return "catalog: /v1/models did not answer — arrival/removal tracking skipped"
    tracked = len(report.get("catalog_seen") or {})
    note = f"catalog: {size} listed · {tracked} tracked"
    if size and tracked and tracked != size:
        note += " (removals kept in history)"
    return note


def render_cached(report: dict) -> str:
    if not report:
        return ("🧠 NIM — no scan yet.\n\n"
                "/nim probe scans the models your config uses (~30 s).\n"
                "/nim cfg shows what those are without touching the network.")
    if report.get("error"):
        return f"🧠 NIM — {report['error']}"
    age = engine.scan_age_seconds(report)
    models = report.get("models") or {}
    lines = [f"🧠 NIM — last scan {human_age(age)} · {len(models)} models", ""]
    lines += model_lines(report)
    lines += ["", best_line(report), _catalog_footnote(report)]
    lines += ["", "/nim probe to re-scan · /nim dead · /nim added · /nim catalog"]
    return "\n".join(lines)


def render_catalog(report: dict) -> str:
    if not report:
        return "🧠 NIM — no scan yet; run /nim probe first (the catalog is read during a scan)."
    live = report.get("catalog") or []
    if not report.get("catalog_fetch_ok", True):
        live = report.get("catalog") or []
        if not live:
            return "🧠 NIM — /v1/models did not answer on the last scan."
    dead = engine.removed_ids(report)
    usable = [model for model in live
              if (report.get("models") or {}).get(model, {}).get("status") == "ok"]
    lines = [f"🧠 NIM catalog — {len(live)} models listed"]
    if dead:
        lines.append(f"· {len(dead)} dropped since the last scan: {', '.join(dead[:5])}"
                     + ("…" if len(dead) > 5 else ""))
    if usable:
        lines.append(f"· verified answering: {', '.join(usable)}")
    unprobed = len(live) - len([m for m in live if m in (report.get("models") or {})])
    if unprobed:
        lines.append(f"· {unprobed} never probed (not in your config or shortlist)")
    lines += ["", "Listing a model proves nothing: /v1/models includes ids this account "
                  "cannot call (404) and ids that answer zero bytes."]
    return "\n".join(lines)


def render_dead(report: dict) -> str:
    if not report:
        return "🧠 NIM — no scan yet; run /nim probe."
    _, _, dead = _grouped(report)
    removed = engine.removed_ids(report)
    lines = []
    if dead:
        lines.append(f"🧠 NIM — {len(dead)} not usable")
        used = report.get("configured_in") or {}
        lines += [f"• {result['model']} — {engine.label(result)}"
                  f"{' · ' + short_where((used or {}).get(result['model']) or []) if short_where((used or {}).get(result['model']) or []) else ''}"
                  for result in sorted(dead, key=lambda r: r["model"])]
    if removed:
        lines += ["", f"dropped from /v1/models ({len(removed)}): "
                      + ", ".join(removed[:8]) + ("…" if len(removed) > 8 else "")]
    if not lines:
        return "🧠 NIM — nothing is currently dead in the last scan."
    return "\n".join(lines)


def render_added(report: dict, days: float = 7.0) -> str:
    if not report:
        return "🧠 NIM — no scan yet; run /nim probe."
    added = engine.added_ids(report, days)
    if not added:
        seeded = report.get("catalog_seen") or {}
        since = min((str(r.get("first_seen")) for r in seeded.values() if r.get("first_seen")),
                    default="")
        hint = f" (tracking since {since[:10]})" if since else ""
        return f"🧠 NIM — no new models in the last {days:g} days{hint}."
    probed = report.get("new_models_probed") or {}
    lines = [f"🧠 NIM — {len(added)} model(s) first seen in the last {days:g} days"]
    for model in added:
        verdict = probed.get(model)
        detail = f" — {verdict['status']}" if verdict else ""
        lines.append(f"• {model}{detail}")
    lines += ["", "New arrivals are probed once when they appear; most turn out to be "
                  "404-unprovisioned for this account."]
    return "\n".join(lines)


def render_cfg(used: dict[str, list[str]], config_path: str) -> str:
    if not used:
        return ("🧠 NIM — no `provider: nvidia` models in config.yaml.\n"
                "Nothing to watch; add one, or use /nim probe to check the shortlist only.")
    lines = [f"🧠 NIM — {len(used)} NIM model(s) in config.yaml"]
    for model, where in used.items():
        lines.append(f"• {model} — {short_where(where) or ', '.join(where)}")
    lines += ["", config_path]
    return "\n".join(lines)


def summary_line(report: dict) -> str:
    """One line for logs and CLI exit paths."""
    alive, warn, dead = _grouped(report)
    return f"{len(alive)} ok · {len(warn)} degraded · {len(dead)} not usable"


def report_age_note(report: dict) -> str:
    age = engine.scan_age_seconds(report)
    if age is None:
        return "unknown age"
    return human_age(age)


def iso_today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")
