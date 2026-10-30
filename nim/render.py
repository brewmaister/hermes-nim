"""Rendering: one place that turns a report into words, so chat, CLI and cron cannot drift.

Bullets, no tables: these land in Telegram and a terminal alike, and a table dies at 40 columns.
"""

from __future__ import annotations

from datetime import datetime, timezone

from . import benchmarks, engine

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


MARKS = {"ok": "✓", "warn": "⚠", "dead": "✗"}

# Full role names cost 10-15 characters each on a line that already carries a 30-character model id.
ROLE_SHORT = (("moa reference", "moa"), ("free route", "free"), ("shortlist", "shortlist"))


def human_ctx(tokens: int | None) -> str:
    """1_000_000 -> 1M, 262_144 -> 262k: a context window is a headline, not a measurement."""
    if not tokens:
        return ""
    if tokens >= 1_000_000:
        return f"{tokens / 1_000_000:g}M"
    return f"{round(tokens / 1000)}k"


def short_model(model: str) -> str:
    """Drop the org prefix. `nvidia/nemotron-3-ultra-550b-a55b` is unambiguous without `nvidia/`."""
    return str(model or "").split("/", 1)[-1]


def _quality_tag(result: dict) -> str:
    """Published score when one exists, else the id's parameter count — and never both."""
    model = str(result.get("model") or "")
    published = benchmarks.lookup(model)
    if published is not None:
        return f"AA {published.index:g}"
    size = engine.params_from_id(model)
    if size is None:
        return ""
    # The size is already in the name for most ids; repeating it is just width.
    return "" if f"{size:g}b" in model.lower() else f"{size:g}B"


def _roles(result: dict, used: dict[str, list[str]]) -> str:
    where = short_where((used or {}).get(result["model"]) or [])
    for full, short in ROLE_SHORT:
        where = where.replace(full, short)
    return where


def _line(result: dict, used: dict[str, list[str]], recommended: bool = False) -> str:
    """One model, one line: mark, short name, then the facts, separated by middots."""
    bits = [bit for bit in (_quality_tag(result), engine.metrics(result),
                            _roles(result, used)) if bit]
    mark = "★" if recommended else MARKS.get(engine.severity(result), "·")
    return f"{mark} {short_model(result['model'])} — " + " · ".join(bits)


def model_lines(report: dict, limit: int = MAX_LINES) -> list[str]:
    used = report.get("configured_in") or {}
    alive, warn, dead = _grouped(report)
    lines: list[str] = []
    # Ranked best-first on evidence: published benchmark index, else the parameter count in the id,
    # then attempts-answered, then measured latency. Exactly one model carries "(recommended)" —
    # and it is not always the top of the list, because a 40 s first byte is not a recommendation.
    pick = engine.recommended(report)
    pick_model = pick.get("model") if pick else None
    for result in sorted(alive, key=engine.rank_key):
        lines.append(_line(result, used, recommended=result["model"] == pick_model))
    for group in (warn, dead):
        if group:
            lines.append("")   # a blank line is cheaper than a header row
            lines += [_line(result, used) for result in group]
    if len(lines) > limit:
        hidden = len(lines) - limit
        lines = lines[:limit] + [f"… and {hidden} more (see /nim catalog)"]
    return lines


def best_line(report: dict) -> str:
    """One line naming the recommendation and the evidence behind it."""
    pick = engine.recommended(report)
    if pick is None:
        alive = [result for result in (report.get("models") or {}).values()
                 if engine.severity(result) == "ok"]
        if not alive:
            return "★ recommended: nothing — no model answered every attempt"
        return (f"★ recommended: nothing under {engine.RECOMMEND_MAX_TTFT_S:g}s — "
                f"the healthy models are all slower than that right now")
    model = str(pick.get("model") or "")
    published = benchmarks.lookup(model)
    why: list[str] = []
    if published is not None:
        why.append(f"AA {published.index:g}")
        if published.context:
            why.append(f"{human_ctx(published.context)} ctx")
    else:
        size = engine.params_from_id(model)
        if size is not None:
            why.append(f"{size:g}B")
    why += [f"{pick.get('ttft_s')}s", f"{pick.get('ok_attempts')}/{pick.get('attempts')}"]
    # The footer keeps the full id on purpose: this is the line you paste into config.yaml.
    return f"★ recommended: {model} — " + " · ".join(why)


def credential_block(report: dict) -> list[str]:
    """The key-rejected banner: one line about the credential, never 85 lines about models."""
    failure = report.get("credential_failure")
    if not failure:
        return []
    code = failure.get("code") or "?"
    detail = str(failure.get("detail") or "").strip() or "no detail in the response"
    lines = [f"🔑 NVIDIA key rejected — HTTP {code}: {detail}",
             "   Models were NOT probed: this is the credential, not the models.",
             "   Fix NVIDIA_API_KEY in ~/.hermes/.env, then /nim probe."]
    age = engine.scan_age_seconds({"run_at": report.get("last_good_run_at")})
    if report.get("models"):
        lines.append(f"   Below is the last good scan ({human_age(age)}), unchanged.")
    return lines


def render_speed(report: dict) -> str:
    """Latency, measured now: to first byte and to the end of the answer, median of the attempts."""
    if report.get("error"):
        return f"⚡ NIM speed — {report['error']}"
    banner = credential_block(report)
    if banner:
        return "\n".join(["⚡ NIM speed — cannot measure"] + [""] + banner)
    models = list((report.get("models") or {}).values())
    answered = [m for m in models if m.get("ttft_median_s") is not None]
    silent = [m for m in models if m.get("ttft_median_s") is None]
    answered.sort(key=lambda m: (m.get("ttft_median_s") or 9e9, m.get("total_s") or 9e9))
    lines = [f"⚡ NIM speed · just now · {report.get('attempts')}× each · "
             f"median time to first byte"]
    lines.append("")
    if not answered:
        lines.append("No model answered at all — nothing to time.")
    for position, result in enumerate(answered, start=1):
        model = str(result.get("model") or "")
        trials = [trial.split(":")[1] for trial in (result.get("trials") or [])
                  if trial.startswith("ok:")]
        spread = " / ".join(trials[:4]) + ("…" if len(trials) > 4 else "")
        total = result.get("total_s")
        tail = f" · {total}s to finish" if total is not None else ""
        flaky = ("" if result.get("ok_attempts") == result.get("attempts")
                 else f" · only {result.get('ok_attempts')}/{result.get('attempts')} answered")
        lines.append(f"{position}. {short_model(model)} — {result.get('ttft_median_s')}s"
                     f" ({spread}){tail}{flaky}")
    if silent:
        lines += ["", "no measurement (never emitted a byte): "
                  + ", ".join(str(m.get("model")) for m in silent)]
    fastest = answered[0] if answered else None
    lines += ["", "first byte is what feels fast; 'to finish' is the whole 32-token answer, so it "
                  "tracks tokens/s. Median, not mean: one stall is an event, not a speed."]
    if fastest:
        lines.append(f"fastest right now: {short_model(fastest.get('model'))} — "
                     f"{fastest.get('ttft_median_s')}s")
    return "\n".join(lines)


def changes_block(report: dict) -> list[str]:
    changes = report.get("changes") or []
    if not changes:
        return []
    return ["", "since the last scan:"] + [f"• {change}" for change in changes]


def render_probe(report: dict) -> str:
    if report.get("error"):
        return f"🧠 NIM — {report['error']}"
    banner = credential_block(report)
    if banner:
        return "\n".join([f"🧠 NIM scan — aborted", ""] + banner)
    models = report.get("models") or {}
    header = (f"🧠 NIM scan · just now · {len(models)} models · "
              f"{report.get('attempts')}× each")
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
    banner = credential_block(report)
    if banner:
        lines = [f"🧠 NIM — {len(report.get('models') or {})} models known", ""] + banner + [""]
        lines += model_lines(report)
        lines += ["", _catalog_footnote(report)]
        return "\n".join(lines)
    age = engine.scan_age_seconds(report)
    models = report.get("models") or {}
    lines = [f"🧠 NIM · {human_age(age)} · {len(models)} models", ""]
    lines += model_lines(report)
    lines += ["", best_line(report), _catalog_footnote(report)]
    lines += ["", "/nim probe · /nim speed · /nim dead · /nim catalog"]
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
