"""Rendering: one place that turns a report into words, so chat, CLI and cron cannot drift.

Bullets, no tables: these land in Telegram and a terminal alike, and a table dies at 40 columns.

Every user-facing string comes from :mod:`nim.i18n` (see ``locales/*.yaml``). Nothing here hardcodes
English: the engine describes state, this module says it in the reader's language. That split is why
`label()` and `metrics()` live here rather than in the engine — they are words, not measurements.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

from . import benchmarks, engine
from .i18n import t

MAX_LINES = 26  # keep a chat reply readable; the artifact files hold the full picture

# The watchdog runs every 6h. Past a full cycle plus slack, "the last scan" is no longer evidence
# about now — say so in the header rather than leaving the reader to do arithmetic on "7.2 h ago".
STALE_AFTER_S = 7 * 3600

# Say the status, not a tick: ✅/🆗 reads "works", 💀 reads "gone". A tick next to a model you cannot
# call was the confusing part — it looked like approval rather than a health verdict.
MARKS = {"ok": "🆗", "warn": "⚠️", "dead": "💀"}

# One glyph per fact, so a row can be read at a glance instead of parsed:
#   🧠 published benchmark index   ⚡ time to first byte
# The glyphs replace words, they do not invent data: a model with no published score shows no 🧠.
#
# There is deliberately no glyph for parameter count. An id only yields a size when it spells the
# size out (`…-30b-a3b`), so a size tag would always repeat what the name already says. The number
# still appears where it is not redundant: the recommendation footer, for models with no benchmark.
QUALITY_MARK = "🧠"
SPEED_MARK = "⚡"
NEW_MARK = "🆕"   # appeared in /v1/models within NEW_WINDOW_DAYS, and is not the install baseline
NEW_WINDOW_DAYS = 7.0


def plain(text: str) -> str:
    """Strip the emphasis for surfaces that print raw text.

    The canonical render carries markdown because chat is the primary surface; a terminal must not
    show literal ``**`` around a title, so the CLI door calls this rather than a second renderer
    growing a variant of every line.
    """
    return re.sub(r"\*\*(.+?)\*\*", r"\1", text or "")


def board_badge(report: dict) -> str:
    """A compact "/13 of 85 answer" next to the model count, so one line explains the other two.

    Without it the header says "5 models" and the footer says "85 listed", and the reader has to
    remember which measurement is which. The badge makes the header self-contained.
    """
    survey = report.get("survey") or {}
    if not survey or not survey.get("listed"):
        return ""
    return t("header.board", online=survey.get("online") or 0, listed=survey.get("listed"))


def human_age(seconds: float | None) -> str:
    if seconds is None:
        return t("age.unknown")
    seconds = max(0.0, seconds)
    if seconds < 90:
        return t("age.seconds", n=int(seconds))
    if seconds < 5400:
        return t("age.minutes", n=int(seconds / 60))
    if seconds < 172800:
        return t("age.hours", n=f"{seconds / 3600:.1f}")
    return t("age.days", n=f"{seconds / 86400:.1f}")


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


def metrics(result: dict) -> str:
    """Compact evidence for a one-line list: ``0.41s``, ``flaky 1/2``, ``stalled``.

    Deliberately separate from :func:`label`: that one is the sentence a dead-model alert needs, this
    is what fits after a model name without wrapping a phone screen three times.
    """
    status = str(result.get("status") or "?")
    attempts = result.get("attempts") or 1
    okays = result.get("ok_attempts") or 0
    if status == "ok":
        return f"{result.get('ttft_s')}s"
    if status in {"flaky", "throttled"}:
        return t(f"status.{status}", ok=okays, attempts=attempts)
    word = {"stall": "stalled", "eol": "retired", "not_provisioned": "not_provisioned",
            "rate_limited": "rate_limited", "auth_error": "auth_rejected"}.get(status)
    return t(f"status.{word}") if word else status


def label(result: dict) -> str:
    """One sentence per model — what a dead-model alert and /nim dead need."""
    status = str(result.get("status") or "?")
    attempts = result.get("attempts") or 1
    okays = result.get("ok_attempts") or 0
    if status == "ok":
        return t("label.ok", seconds=result.get("ttft_s"), ok=okays, attempts=attempts)
    if status == "flaky":
        fails = [trial for trial in result.get("trials") or [] if not trial.startswith("ok")]
        stalls = [float(trial.split(":")[1]) for trial in fails if trial.startswith("stall")]
        if stalls:
            why = t("label.flaky_worst", seconds=f"{max(stalls):.0f}")
        else:
            why = ", ".join(sorted({trial.split(":")[0] for trial in fails}))
        return t("label.flaky", ok=okays, attempts=attempts, why=why)
    if status == "throttled":
        return t("label.throttled", ok=okays, attempts=attempts)
    if status == "stall":
        return t("label.stall", attempts=attempts)
    if status == "eol":
        when = result.get("eol_at")
        return t("label.eol", when=when) if when else t("label.eol_unknown")
    if status == "not_provisioned":
        return t("label.not_provisioned")
    if status == "rate_limited":
        return t("label.rate_limited")
    if status == "auth_error":
        return t("label.auth_error")
    return status


def short_where(where: list[str]) -> str:
    """Config paths are precise and unreadable; report the role instead."""
    labels: list[str] = []
    for path in where:
        if path.startswith("fallback_providers["):
            index = path.split("[")[1].split("]")[0]
            labels.append(t("role.fallback", n=int(index) + 1))
        elif path.startswith("delegation"):
            labels.append(t("role.free"))
        elif path.startswith("moa"):
            labels.append(t("role.moa"))
        elif path == "shortlist":
            labels.append(t("role.shortlist"))
        elif path:
            labels.append(path)
    seen: list[str] = []
    for text in labels:
        if text not in seen:
            seen.append(text)
    return ", ".join(seen)


def watched(report: dict) -> dict:
    """The report narrowed to models you actually point at (config or shortlist).

    A survey measures the whole catalog and stores every verdict, so the raw report can hold 85
    models. Chat views must not: a stranger's verdict is not your uptime, and a recommendation drawn
    from models you do not use is advice to change providers, not to fix a fallback chain.
    """
    used = report.get("configured_in")
    if not used:
        # No map at all (an older state file, or a hand-built report): we cannot tell watched from
        # stranger, so show what we have rather than an empty list that reads like "all dead".
        return report
    watched_models = {model: result for model, result in (report.get("models") or {}).items()
                      if model in used}
    return {**report, "models": watched_models}


def _grouped(report: dict) -> tuple[list[dict], list[dict], list[dict]]:
    models = list((report.get("models") or {}).values())
    alive = [m for m in models if engine.severity(m) == "ok"]
    warn = [m for m in models if engine.severity(m) == "warn"]
    dead = [m for m in models if engine.severity(m) == "dead"]
    return alive, warn, dead


def _quality_tag(result: dict) -> str:
    """🧠 + published index, or nothing at all."""
    published = benchmarks.lookup(str(result.get("model") or ""))
    if published is None:
        return ""
    # "family" is a claim about the number: it is the parent model's score, not this variant's.
    family = f" {t('quality.family')}" if published.inherited else ""
    return f"{QUALITY_MARK}{published.index:g}{family}"


def _speed_tag(result: dict) -> str:
    """⚡ only in front of a measurement. `stalled` is a verdict, not a duration."""
    measured = metrics(result)
    if not measured:
        return ""
    return f"{SPEED_MARK}{measured}" if measured[0].isdigit() else measured


def _roles(result: dict, used: dict[str, list[str]]) -> str:
    return short_where((used or {}).get(result["model"]) or [])


def new_models(report: dict, days: float = NEW_WINDOW_DAYS) -> set[str]:
    """Ids NVIDIA added recently — the arrivals nobody announces."""
    return set(engine.added_ids(report, days))


def _line(result: dict, used: dict[str, list[str]], recommended: bool = False,
          new: set[str] | None = None) -> str:
    """One model, one line: mark, short name, then the facts, separated by middots.

    The recommended row says so in words and drops its config role: showing `★ … · shortlist` told
    the reader "candidate" next to the star, which contradicts the star. One row, one claim.
    """
    fresh = NEW_MARK if new and result["model"] in new else ""
    bits = [bit for bit in (fresh, _quality_tag(result), _speed_tag(result),
                            t("role.recommended") if recommended else _roles(result, used)) if bit]
    mark = "★" if recommended else MARKS.get(engine.severity(result), "·")
    return f"{mark} {short_model(result['model'])} — " + " · ".join(bits)


def model_lines(report: dict, limit: int = MAX_LINES) -> list[str]:
    used = report.get("configured_in") or {}
    alive, warn, dead = _grouped(report)
    lines: list[str] = []
    # Ranked best-first on evidence: published benchmark index, else the parameter count in the id,
    # then attempts-answered, then measured latency. Exactly one model carries the star — and it is
    # not always the top of the list, because a 40 s first byte is not a recommendation.
    pick = engine.recommended(report)
    pick_model = pick.get("model") if pick else None
    new = new_models(report)
    for result in sorted(alive, key=engine.rank_key):
        lines.append(_line(result, used, recommended=result["model"] == pick_model, new=new))
    for group in (warn, dead):
        if group:
            lines.append("")   # a blank line is cheaper than a header row
            lines += [_line(result, used) for result in group]
    if len(lines) > limit:
        hidden = len(lines) - limit
        lines = lines[:limit] + [t("foot.more", hidden=hidden)]
    return lines


def best_line(report: dict) -> str:
    """One line naming the recommendation and the evidence behind it."""
    pick = engine.recommended(report)
    if pick is None:
        alive = [result for result in (report.get("models") or {}).values()
                 if engine.severity(result) == "ok"]
        if not alive:
            return t("rec.none")
        return t("rec.none_slow", cap=f"{engine.RECOMMEND_MAX_TTFT_S:g}")
    model = str(pick.get("model") or "")
    published = benchmarks.lookup(model)
    why: list[str] = []
    if published is not None:
        why.append(f"{QUALITY_MARK}{published.index:g}")
        if published.context:
            why.append(t("rec.ctx", ctx=human_ctx(published.context)))
    else:
        size = engine.params_from_id(model)
        if size is not None:
            why.append(f"{size:g}B")
    why += [f"{SPEED_MARK}{pick.get('ttft_s')}s",
            f"{pick.get('ok_attempts')}/{pick.get('attempts')}"]
    # The footer keeps the full id on purpose: this is the line you paste into config.yaml.
    return t("rec.line", model=model, evidence=" · ".join(why))


def credential_block(report: dict) -> list[str]:
    """The key-rejected banner: one line about the credential, never 85 lines about models."""
    failure = report.get("credential_failure")
    if not failure:
        return []
    code = failure.get("code") or "?"
    detail = str(failure.get("detail") or "").strip() or t("cred.no_detail")
    lines = [t("cred.key", code=code, detail=detail),
             t("cred.not_probed"),
             t("cred.fix")]
    age = engine.scan_age_seconds({"run_at": report.get("last_good_run_at")})
    if report.get("models"):
        lines.append(t("cred.below", age=human_age(age)))
    return lines


def render_speed(report: dict) -> str:
    """Latency, measured now: to first byte and to the end of the answer, median of the attempts."""
    if report.get("error"):
        return t("speed.error", error=report["error"])
    banner = credential_block(report)
    if banner:
        return "\n".join([t("speed.cannot")] + [""] + banner)
    models = list((report.get("models") or {}).values())
    answered = [m for m in models if m.get("ttft_median_s") is not None]
    silent = [m for m in models if m.get("ttft_median_s") is None]
    answered.sort(key=lambda m: (m.get("ttft_median_s") or 9e9, m.get("total_s") or 9e9))
    lines = [t("speed.header", attempts=report.get("attempts")), ""]
    if not answered:
        lines.append(t("speed.none"))
    for position, result in enumerate(answered, start=1):
        model = str(result.get("model") or "")
        trials = [trial.split(":")[1] for trial in (result.get("trials") or [])
                  if trial.startswith("ok:")]
        spread = " / ".join(trials[:4]) + ("…" if len(trials) > 4 else "")
        total = result.get("total_s")
        tail = f" · {t('speed.finish', seconds=total)}" if total is not None else ""
        flaky = ("" if result.get("ok_attempts") == result.get("attempts") else
                 f" · {t('speed.flaky', ok=result.get('ok_attempts'), attempts=result.get('attempts'))}")
        lines.append(f"{position}. {short_model(model)} — {SPEED_MARK}{result.get('ttft_median_s')}s"
                     f" ({spread}){tail}{flaky}")
    if silent:
        lines += ["", t("speed.unmeasured",
                        models=", ".join(str(m.get("model")) for m in silent))]
    fastest = answered[0] if answered else None
    lines += ["", t("speed.note")]
    if fastest:
        lines.append(t("speed.fastest", model=short_model(fastest.get("model")),
                       seconds=fastest.get("ttft_median_s")))
    return "\n".join(lines)


def changes_block(report: dict) -> list[str]:
    changes = report.get("changes") or []
    if not changes:
        return []
    return ["", t("foot.since")] + [f"• {change}" for change in changes]


def render_probe(report: dict) -> str:
    if report.get("error"):
        return t("empty.error", error=report["error"])
    report = watched(report)
    banner = credential_block(report)
    if banner:
        return "\n".join([t("header.aborted"), ""] + banner)
    models = report.get("models") or {}
    header = t("header.probe", models=len(models), attempts=report.get("attempts"),
               board=board_badge(report))
    lines = [header, ""] + model_lines(report) + changes_block(report)
    lines += ["", best_line(report), _catalog_footnote(report)]
    return "\n".join(lines)


def _catalog_footnote(report: dict) -> str:
    """Two counts sit next to each other here, so say which is which.

    `{models} measured` in the header is what this scan touched — the models you point at. The board
    is a different measurement (every listed id, once). Printing "5 models" above "85 tracked" invited
    the reading "it scanned 85 and only 5 answered", which is the opposite of what happened.
    """
    size = len(report.get("catalog") or [])
    if not report.get("catalog_fetch_ok", True) and not size:
        return t("foot.catalog_failed")
    survey = report.get("survey") or {}
    if not survey:
        return t("foot.catalog_unsurveyed", listed=size)
    age = engine.scan_age_seconds({"run_at": survey.get("at")})
    return t("foot.catalog_survey", listed=size, online=survey.get("online") or 0,
             surveyed=survey.get("measured") or 0, age=human_age(age))


def render_cached(report: dict) -> str:
    if not report:
        return t("empty.no_scan")
    if report.get("error"):
        return t("empty.error", error=report["error"])
    report = watched(report)
    banner = credential_block(report)
    if banner:
        lines = [t("header.known", models=len(report.get("models") or {})), ""] + banner + [""]
        lines += model_lines(report)
        lines += ["", _catalog_footnote(report)]
        return "\n".join(lines)
    age = engine.scan_age_seconds(report)
    models = report.get("models") or {}
    stale = age is None or age >= STALE_AFTER_S
    header = t("header.cached", age=human_age(age), models=len(models),
               board=board_badge(report))
    if stale:
        header += t("header.stale_badge")
    lines = [header, ""]
    lines += model_lines(report)
    lines += ["", best_line(report), _catalog_footnote(report)]
    if stale:
        lines += ["", t("foot.stale")]
    lines += ["", t("foot.hints")]
    return "\n".join(lines)


def render_board(report: dict, limit: int = 14) -> str:
    """The whole answer to "what can I use?": every listed id, measured, sorted into chat / utility /
    offline. This is `/nim` itself — a survey, not a peek at the models already in config."""
    if not report:
        return t("empty.no_scan_catalog")
    if report.get("error"):
        return t("empty.error", error=report["error"])
    models = report.get("models") or {}
    catalog = report.get("catalog") or []
    survey = report.get("survey") or {}
    rows = {model: verdict for model, verdict in models.items() if model in catalog}
    if not rows:
        return t("empty.no_scan_catalog")

    alive = {model: verdict for model, verdict in rows.items()
             if verdict.get("status") == "ok"}
    chat = sorted((model for model in alive if engine.capability(model) == "chat"),
                  key=lambda model: engine.rank_key(alive[model]))
    utility = sorted((model for model in alive if engine.capability(model) == "utility"),
                     key=lambda model: engine.rank_key(alive[model]))
    offline = [(model, verdict) for model, verdict in rows.items()
               if verdict.get("status") != "ok"]

    mine = report.get("configured_in") or {}
    # "yours" means config.yaml routes to it. A shortlist entry is watched, not used — saying both
    # were yours made an 11-model scan look like an 11-model config.
    routed = {model for model, where in mine.items()
              if any(path != "shortlist" for path in where)}
    watched_only = set(mine) - routed
    pick = engine.recommended(watched(report))
    pick_model = pick.get("model") if pick else None
    new = new_models(report)

    header = t("board.header", age=human_age(engine.scan_age_seconds({"run_at": survey.get("at")}))
               if survey else t("catalog.never_surveyed"), listed=len(catalog))
    counts = t("board.counts", chat=len(chat), utility=len(utility), offline=len(offline))
    if new:
        counts += t("board.new_badge", count=len(new), days=f"{NEW_WINDOW_DAYS:g}")
    lines = [header, counts, ""]

    if chat:
        lines.append(t("board.section_chat"))
        for model in chat[:limit]:
            verdict = alive[model]
            bits = [f"{SPEED_MARK}{verdict.get('ttft_s')}s"]
            if model in new:
                bits.insert(0, NEW_MARK)
            published = benchmarks.lookup(model)
            if published is not None:
                bits.insert(0, f"{QUALITY_MARK}{published.index:g}"
                               + (f" {t('quality.family')}" if published.inherited else ""))
            if model in routed:
                bits.append(t("board.yours"))
            elif model in watched_only:
                bits.append(t("board.watching"))
            mark = "★" if model == pick_model else MARKS["ok"]
            lines.append(f"{mark} {short_model(model)} — " + " · ".join(bits))
        if len(chat) > limit:
            lines.append(t("foot.more", hidden=len(chat) - limit))
    if utility:
        lines += ["", t("board.section_utility"),
                  "· " + ", ".join(short_model(model) for model in utility[:limit])]

    if offline:
        lines += ["", t("board.offline_header", count=len(offline))]
        counts: dict[str, int] = {}
        for _model, verdict in offline:
            counts[verdict.get("status") or "?"] = counts.get(verdict.get("status") or "?", 0) + 1
        for status, count in sorted(counts.items(), key=lambda item: -item[1]):
            lines.append(t("board.offline_group", count=count, why=metrics({"status": status})))
    lines += ["", t("board.hints")]
    return "\n".join(lines)


def render_catalog(report: dict, limit: int = 12) -> str:
    """What NVIDIA lists, and what actually answers — the online/offline board.

    Fed by a survey (every catalog id probed once). Without a survey the verdicts are simply absent,
    and the view says so instead of implying a model is dead because it was never asked.
    """
    if not report:
        return t("empty.no_scan_catalog")
    live = report.get("catalog") or []
    if not report.get("catalog_fetch_ok", True) and not live:
        return t("empty.catalog_failed")
    models = report.get("models") or {}
    surveyed = [model for model in live if model in models]
    usable = sorted((model for model in surveyed if models[model].get("status") == "ok"),
                    key=lambda model: engine.rank_key(models[model]))
    broken = sorted((model for model in surveyed if models[model].get("status") != "ok"),
                    key=lambda model: str(models[model].get("status")))
    survey = report.get("survey") or {}
    header = t("catalog.header", models=len(live))
    if survey:
        header += " · " + t(
            "catalog.surveyed", age=human_age(engine.scan_age_seconds({"run_at": survey.get("at")})))
    else:
        header += " · " + t("catalog.never_surveyed")
    lines = [header]
    removed = engine.removed_ids(report)
    if removed:
        lines.append(t("catalog.dropped", count=len(removed),
                       models=", ".join(removed[:5]) + ("…" if len(removed) > 5 else "")))
    if usable:
        lines.append(t("catalog.online", count=len(usable),
                       models=", ".join(short_model(model) for model in usable[:limit])
                       + ("…" if len(usable) > limit else "")))
    if broken:
        lines.append(t("catalog.offline", count=len(broken)))
        for model in broken[:limit]:
            result = models[model]
            lines.append(t("catalog.line", model=short_model(model),
                           why=metrics(result) if result.get("status") != "ok" else "ok"))
        if len(broken) > limit:
            lines.append(t("catalog.more", count=len(broken) - limit))
    unprobed = len(live) - len(surveyed)
    if unprobed:
        lines.append(t("catalog.never_probed", count=unprobed))
    lines += ["", t("catalog.footnote")]
    return "\n".join(lines)


def render_dead(report: dict) -> str:
    if not report:
        return t("empty.no_scan_probe")
    report = watched(report)
    _, _, dead = _grouped(report)
    removed = engine.removed_ids(report)
    lines = []
    if dead:
        lines.append(t("dead.header", count=len(dead)))
        used = report.get("configured_in") or {}
        for result in sorted(dead, key=lambda r: r["model"]):
            where = short_where((used or {}).get(result["model"]) or [])
            suffix = f" · {where}" if where else ""
            lines.append(f"• {result['model']} — {label(result)}{suffix}")
    if removed:
        lines += ["", t("dead.dropped", count=len(removed),
                        models=", ".join(removed[:8]) + ("…" if len(removed) > 8 else ""))]
    if not lines:
        return t("dead.none")
    return "\n".join(lines)


def render_added(report: dict, days: float = 7.0) -> str:
    if not report:
        return t("empty.no_scan_probe")
    added = engine.added_ids(report, days)
    if not added:
        seeded = report.get("catalog_seen") or {}
        since = min((str(r.get("first_seen")) for r in seeded.values() if r.get("first_seen")),
                    default="")
        hint = t("added.tracking", date=since[:10]) if since else ""
        return t("added.none", days=f"{days:g}", hint=hint)
    probed = report.get("new_models_probed") or {}
    lines = [t("added.header", count=len(added), days=f"{days:g}")]
    for model in added:
        verdict = probed.get(model)
        detail = f" — {verdict['status']}" if verdict else ""
        lines.append(f"• {model}{detail}")
    lines += ["", t("added.footnote")]
    return "\n".join(lines)


def render_cfg(used: dict[str, list[str]], config_path: str) -> str:
    if not used:
        return t("empty.cfg_none")
    lines = [t("cfg.header", count=len(used))]
    for model, where in used.items():
        lines.append(f"• {model} — {short_where(where) or ', '.join(where)}")
    lines += ["", config_path]
    return "\n".join(lines)


def summary_line(report: dict) -> str:
    """One line for logs and CLI exit paths — deliberately not translated (logs, not prose)."""
    alive, warn, dead = _grouped(report)
    return f"{len(alive)} ok · {len(warn)} degraded · {len(dead)} not usable"


def report_age_note(report: dict) -> str:
    age = engine.scan_age_seconds(report)
    if age is None:
        return t("age.unknown")
    return human_age(age)


def iso_today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")
