"""The engine: probe NIM endpoints for liveness, and remember what was seen.

Standard library only, on purpose — the plugin then declares no dependencies and never drags the
host's dependency union along with it.

Nothing here trusts NVIDIA's published state. Every verdict comes from a streaming request:
a model that connects and never sends a byte is dead in practice, and a non-streaming call would
just hang instead of telling you so.
"""

from __future__ import annotations

import getpass
import json
import os
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

DEFAULT_BASE = "https://integrate.api.nvidia.com/v1"
STATE_VERSION = 2
PROBE_PROMPT = "ping"

# Status vocabulary. `severity` collapses it into the three buckets the renderers and the
# hysteresis care about.
DEAD_KINDS = {"eol", "not_provisioned", "auth_error", "stall"}
WARN_KINDS = {"flaky", "rate_limited", "throttled"}


# --------------------------------------------------------------------------- settings / state paths

def hermes_home() -> Path:
    """The Hermes home for the profile this process belongs to."""
    try:  # in-process, the authoritative answer
        from hermes_constants import get_hermes_home  # type: ignore

        return Path(get_hermes_home()).expanduser()
    except Exception:  # noqa: BLE001 - CLI shim or a different install layout
        return Path(os.environ.get("HERMES_HOME") or (Path.home() / ".hermes")).expanduser()


def state_dir(configured: str = "") -> Path:
    """Where the scan state lives. Env beats the manifest setting, for cron and one-off runs."""
    for candidate in (os.environ.get("NIM_STATE_DIR"), configured):
        if candidate and str(candidate).strip():
            return Path(str(candidate)).expanduser()
    return Path.home() / "hermes-outputs" / "nim-probe"


def state_path(configured: str = "") -> Path:
    return state_dir(configured) / "state.json"


# --------------------------------------------------------------------------- credentials

def _parse_dotenv(path: Path) -> str:
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return ""
    for line in text.splitlines():
        match = re.match(r"\s*(?:export\s+)?NVIDIA_API_KEY\s*=\s*['\"]?([^'\"\s]+)", line)
        if match and match.group(1).startswith("nvapi-"):
            return match.group(1)
    return ""


def api_key(home: Path | None = None) -> str:
    """NVIDIA key: environment first, then the profile's .env, then the default home's."""
    key = (os.environ.get("NVIDIA_API_KEY") or "").strip()
    if key.startswith("nvapi-"):
        return key
    homes = [home or hermes_home()]
    default = Path.home() / ".hermes"
    if default not in homes:
        homes.append(default)
    for candidate in homes:
        found = _parse_dotenv(candidate / ".env")
        if found:
            return found
    return ""


def key_hint() -> str:
    """What to say when there is no key — name the file, not the secret."""
    return (f"NVIDIA_API_KEY is not set. Put it in {hermes_home() / '.env'} "
            f"(NVIDIA_API_KEY=nvapi-...) or export it.")


# --------------------------------------------------------------------------- one attempt

def _urlopen(req: urllib.request.Request, timeout: float):
    return urllib.request.urlopen(req, timeout=timeout)


def attempt(base: str, model: str, key: str, ttft: float) -> dict:
    """One streaming call. Returns {kind, seconds, detail[, eol_at]}.

    `kind` is one of ok / stall / eol / not_provisioned / rate_limited / auth_error / http_<code>.
    Time-to-first-byte is the measurement: a json body only arrives after the model has spoken.
    """
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": PROBE_PROMPT}],
        "max_tokens": 32,
        "temperature": 0,
        "stream": True,
    }).encode()
    req = urllib.request.Request(
        f"{base.rstrip('/')}/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
    )
    started = time.time()
    try:
        with _urlopen(req, ttft) as response:
            for raw in response:
                if raw.strip():
                    return {"kind": "ok", "seconds": round(time.time() - started, 2), "detail": ""}
        return {"kind": "stall", "seconds": round(time.time() - started, 2),
                "detail": "empty stream"}
    except urllib.error.HTTPError as exc:
        try:
            payload = json.loads(exc.read().decode(errors="replace"))
        except Exception:  # noqa: BLE001
            payload = {}
        detail = str(payload.get("detail") or payload.get("title") or "")[:300]
        kind = {
            410: "eol",
            404: "not_provisioned",
            429: "rate_limited",
            401: "auth_error",
            403: "auth_error",
        }.get(exc.code, f"http_{exc.code}")
        result = {"kind": kind, "seconds": round(time.time() - started, 2), "detail": detail}
        if kind == "eol":
            found = re.search(r"end of life on ([0-9T:\-]+Z)", detail)
            if found:
                result["eol_at"] = found.group(1)
        return result
    except Exception as exc:  # noqa: BLE001 - read timeout, reset, refused, no data
        return {"kind": "stall", "seconds": round(time.time() - started, 2),
                "detail": type(exc).__name__}


def probe(base: str, model: str, key: str, ttft: float = 25.0, attempts: int = 2,
          stagger: float = 1.5) -> dict:
    """Several attempts, because one is a coin flip on a replica-pooled gateway."""
    trials: list[dict] = []
    for index in range(max(1, attempts)):
        if index:
            time.sleep(max(0.0, stagger))
        trials.append(attempt(base, model, key, ttft))
        # A hard verdict will not change on retry.
        if trials[-1]["kind"] in {"eol", "not_provisioned", "auth_error"}:
            break
    kinds = [trial["kind"] for trial in trials]
    okays = [trial for trial in trials if trial["kind"] == "ok"]
    others = [kind for kind in kinds if kind != "ok"]
    if not others:
        status = "ok"
    elif okays and all(kind == "rate_limited" for kind in others):
        status = "throttled"  # answered, just not every time: quota, not rot
    elif okays:
        status = "flaky"
    elif all(kind == "rate_limited" for kind in kinds):
        status = "rate_limited"
    else:
        status = next((kind for kind in kinds if kind not in {"ok", "rate_limited"}), kinds[-1])
    result = {
        "model": model,
        "status": status,
        "ok_attempts": len(okays),
        "attempts": len(trials),
        "ttft_s": min((trial["seconds"] for trial in okays), default=None),
        "detail": next((trial["detail"] for trial in trials if trial["detail"]), ""),
        "trials": [f"{trial['kind']}:{trial['seconds']}" for trial in trials],
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    for trial in trials:
        if trial.get("eol_at"):
            result["eol_at"] = trial["eol_at"]
    return result


# --------------------------------------------------------------------------- catalog

def catalog(base: str, key: str) -> list[str]:
    """Live /v1/models ids. Empty list means the fetch failed — callers must treat it as unknown."""
    req = urllib.request.Request(
        f"{base.rstrip('/')}/models", headers={"Authorization": f"Bearer {key}"})
    try:
        with _urlopen(req, 30) as response:
            return sorted(entry["id"] for entry in json.load(response).get("data", []))
    except Exception:  # noqa: BLE001
        return []


# --------------------------------------------------------------------------- config scan

def configured_models(config_path: Path | None = None) -> dict[str, list[str]]:
    """Every model referenced with `provider: nvidia`, mapped to where it is used."""
    path = config_path or (hermes_home() / "config.yaml")
    try:
        text = path.read_text()
    except OSError:
        return {}
    try:
        import yaml
    except ImportError:  # degrade to a scanner rather than lie about having nothing configured
        return _configured_models_without_yaml(text)

    try:
        tree = yaml.safe_load(text)
    except Exception:  # noqa: BLE001
        return _configured_models_without_yaml(text)

    found: dict[str, set[str]] = {}

    def walk(node, path_parts: str) -> None:
        if isinstance(node, dict):
            if node.get("provider") == "nvidia" and isinstance(node.get("model"), str):
                found.setdefault(node["model"], set()).add(path_parts or "root")
            for key_name, value in node.items():
                walk(value, f"{path_parts}.{key_name}" if path_parts else str(key_name))
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{path_parts}[{index}]")

    walk(tree, "")
    return {model: sorted(where) for model, where in sorted(found.items())}


def _configured_models_without_yaml(text: str) -> dict[str, list[str]]:
    """Fallback for a host without PyYAML: treat each list item as one mapping and read its pair.

    Every nvidia reference in a Hermes config sits in a list of small dicts (fallback_providers,
    delegation model_map options, moa reference_models), so splitting on `- ` item starts is enough
    structure to pair `model:` with `provider:` without a YAML parser — and it never pairs across
    items, which a line-neighbour heuristic does.
    """
    items: list[list[tuple[str, str]]] = []
    current: list[tuple[str, str]] | None = None
    for line in text.splitlines():
        start = re.match(r"\s*-\s+(\S.*)$", line)
        body = start.group(1) if start else line
        if start is not None:
            current = []
            items.append(current)
        pair = re.match(r"\s*([A-Za-z_][\w-]*):\s*(.*?)\s*$", body)
        if pair and current is not None:
            current.append((pair.group(1), pair.group(2).strip("'\"")))
    found: dict[str, list[str]] = {}
    for item in items:
        keys = dict(item)
        model = keys.get("model", "")
        if keys.get("provider") == "nvidia" and model:
            found.setdefault(model, set()).add("config list item")
    return {model: sorted(where) for model, where in sorted(found.items())}


# --------------------------------------------------------------------------- ranking inputs

_PARAMS_RE = re.compile(r"(\d+(?:\.\d+)?)b(?![a-z])", re.IGNORECASE)


def params_from_id(model: str) -> float | None:
    """Parameter count in billions, parsed from the vendor's own model id.

    `nvidia/nemotron-3-ultra-550b-a55b` -> 550 (total; the a-suffix is active params in an MoE),
    `openai/gpt-oss-20b` -> 20. None when the id does not say — never a guess.

    This is the only *quality* prior available without inventing one: NIM exposes no benchmarks and
    no metadata beyond id/owned_by, but the vendor puts the size in the name. It is displayed next
    to the model so nobody has to take the ordering on faith.
    """
    if not model:
        return None
    tail = str(model).rsplit("/", 1)[-1]
    found = _PARAMS_RE.search(tail)
    return float(found.group(1)) if found else None


def reliability(result: dict) -> float:
    """Fraction of attempts that answered — 1.0 beats 0.5 even when the fast one is flakier."""
    attempts = int(result.get("attempts") or 1) or 1
    return float(result.get("ok_attempts") or 0) / attempts


def rank_key(result: dict) -> tuple:
    """Best-first, on evidence in descending order of trust.

    1. A published benchmark index (`nim.benchmarks`) — external evidence, labelled as such.
    2. Otherwise the parameter count in the model id — a weak prior, better than nothing.
    3. Then measured behaviour: attempts-answered ratio, then time to first byte.

    Every position is explainable from something printed on the line. A model with neither an index
    nor a stated size sorts last and is labelled, never given an invented number.
    """
    from . import benchmarks

    model = str(result.get("model") or "")
    index = benchmarks.quality(model)
    size = params_from_id(model)
    latency = result.get("ttft_s")
    return (
        0 if index is not None else 1,     # published scores first
        -(index or 0.0),
        0 if size is not None else 1,      # then known sizes
        -(size or 0.0),
        -reliability(result),              # then: answered every time
        latency if latency is not None else 9e9,
    )


# A model that takes longer than this to emit a first byte is not what anyone wants recommended,
# however strong it is on paper. Kept as a constant so the rule is visible and testable.
RECOMMEND_MAX_TTFT_S = 5.0


def recommended(report: dict) -> dict | None:
    """The single model to use right now, or None when nothing qualifies.

    Highest-ranked model that answered *every* attempt and is quick enough to be pleasant. Ranking
    alone is not a recommendation: a 40 s first byte is unusable in a chat loop no matter how good
    the benchmark is, so speed is a gate here rather than a tie-break.
    """
    for result in sorted((report.get("models") or {}).values(), key=rank_key):
        if severity(result) != "ok":
            continue
        latency = result.get("ttft_s")
        if latency is None or latency <= RECOMMEND_MAX_TTFT_S:
            return result
    return None


# --------------------------------------------------------------------------- verdicts

def severity(result: dict) -> str:
    status = str(result.get("status") or "stall")
    if status in DEAD_KINDS or status.startswith("http_"):
        return "dead"
    if status in WARN_KINDS:
        return "warn"
    return "ok"


def label(result: dict) -> str:
    """One line per model, in the words a human would use."""
    status = str(result.get("status") or "?")
    attempts = result.get("attempts") or 1
    if status == "ok":
        return f"ok {result.get('ttft_s')}s ({result.get('ok_attempts')}/{attempts})"
    if status == "flaky":
        fails = [trial for trial in result.get("trials") or [] if not trial.startswith("ok")]
        stalls = [float(trial.split(":")[1]) for trial in fails if trial.startswith("stall")]
        why = f"worst stall {max(stalls):.0f}s" if stalls else ", ".join(
            sorted({trial.split(":")[0] for trial in fails}))
        return f"FLAKY — only {result.get('ok_attempts')}/{attempts} answered ({why})"
    if status == "throttled":
        return f"ok but throttled — {result.get('ok_attempts')}/{attempts} answered, rest 429"
    if status == "stall":
        return f"STALL — zero bytes on all {attempts} attempts"
    if status == "eol":
        return f"EOL {result.get('eol_at', 'unknown date')}"
    if status == "not_provisioned":
        return "404 not provisioned for this account"
    if status == "rate_limited":
        return "429 throttled on every attempt"
    if status == "auth_error":
        return "auth rejected — check NVIDIA_API_KEY"
    return status


# --------------------------------------------------------------------------- state

def load_state(configured_dir: str = "") -> dict:
    path = state_path(configured_dir)
    try:
        return json.loads(path.read_text())
    except Exception:  # noqa: BLE001 - missing or corrupt state is just "no history"
        return {}


def save_state(report: dict, configured_dir: str = "") -> Path:
    directory = state_dir(configured_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "state.json"
    path.write_text(json.dumps(report, indent=1))
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M")
    (directory / f"{stamp}.json").write_text(json.dumps(report, indent=1))
    return path


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def merge_seen(previous: dict, live: list[str]) -> dict:
    """Track first/last sighting per model id, so arrivals and removals survive between runs."""
    now = _now_iso()
    seen: dict[str, dict] = {}
    for model in live:
        record = dict(previous.get(model) or {})
        record.setdefault("first_seen", now)
        record["last_seen"] = now
        seen[model] = record
    for model, record in previous.items():
        if model not in seen:  # keep the history of what left, without touching last_seen
            seen[model] = dict(record)
    return seen


def removed_ids(report: dict) -> list[str]:
    """Models in the history that the latest successful catalog fetch no longer lists."""
    live = set(report.get("catalog") or [])
    if not live or not report.get("catalog_fetch_ok", True):
        return []
    return sorted(model for model in (report.get("catalog_seen") or {}) if model not in live)


def added_ids(report: dict, days: float = 7.0) -> list[str]:
    """Models first seen within `days` — the swap candidates nobody announces."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    live = set(report.get("catalog") or [])
    added: list[str] = []
    for model, record in (report.get("catalog_seen") or {}).items():
        if live and model not in live:
            continue
        try:
            first = datetime.fromisoformat(str(record.get("first_seen")).replace("Z", "+00:00"))
        except Exception:  # noqa: BLE001
            continue
        if first >= cutoff:
            added.append(model)
    return sorted(added)


# --------------------------------------------------------------------------- the scan

def scan(*, base: str = DEFAULT_BASE, ttft: float = 25.0, attempts: int = 2,
         stagger: float = 1.5, extra: list[str] | None = None, key: str | None = None,
         config_path: Path | None = None, configured_dir: str = "") -> dict:
    """Probe the models your config uses (plus `extra`), diff the catalog, persist, return the report.

    `report["changes"]` holds only transitions worth telling a human about: hysteresis means a model
    must look dead on two runs before it is called dead, and each condition is reported once until
    it recovers.
    """
    resolved_key = key if key is not None else api_key()
    if not resolved_key:
        return {"error": key_hint(), "models": {}, "changes": [], "catalog": []}

    used = configured_models(config_path)
    targets = list(used)
    for model in extra or []:
        if model and model not in targets:
            targets.append(model)
            used.setdefault(model, ["shortlist"])
    if not targets:
        return {"error": "No `provider: nvidia` models in config.yaml and nothing on the shortlist.",
                "models": {}, "changes": [], "catalog": []}

    workers = max(1, min(4, len(targets)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(
            lambda model: probe(base, model, resolved_key, ttft, attempts, stagger), targets))
    live_catalog = catalog(base, resolved_key)
    catalog_ok = bool(live_catalog)

    previous = load_state(configured_dir)
    previous_models = previous.get("models") or {}
    previous_seen = previous.get("catalog_seen") or {}
    if not previous_seen:  # first run under this schema: seed from the catalog we just read
        previous_seen = {model: {"first_seen": _now_iso()} for model in live_catalog}

    changes: list[str] = []
    history: dict[str, dict] = {}
    for result in results:
        model = result["model"]
        current = severity(result)
        before = previous_models.get(model) or {}
        before_severity = before.get("severity")
        was_alerted = bool(before.get("alerted"))
        confirmed = before_severity == current
        alerted = was_alerted
        if current == "ok":
            if was_alerted:
                changes.append(f"RECOV    {model}: back to {label(result)}")
            alerted = False
        elif confirmed and not was_alerted:
            changes.append(f"{'DEAD' if current == 'dead' else 'DEGRADED'}     "
                           f"{model}: {label(result)}")
            alerted = True
        history[model] = {
            "severity": current, "status": result["status"], "alerted": alerted,
            "where": used.get(model, []),
        }

    previous_catalog = previous.get("catalog") or []
    removed = sorted(set(previous_catalog) - set(live_catalog)) if (previous_catalog and catalog_ok) else []
    added = sorted(set(live_catalog) - set(previous_catalog)) if (previous_catalog and catalog_ok) else []
    for model in removed:
        changes.append(f"GONE     {model}: dropped from /v1/models"
                       + (" (CONFIG STILL USES IT)" if model in used else ""))
    arrivals = {}
    if added:  # a new arrival only matters if it answers, so probe it once before reporting
        with ThreadPoolExecutor(max_workers=min(4, len(added))) as pool:
            arrivals = {result["model"]: result for result in pool.map(
                lambda model: probe(base, model, resolved_key, ttft, 1, 0.0), added)}
        for model in added:
            result = arrivals.get(model)
            changes.append(f"NEW MODEL {model}: listed in /v1/models — "
                           f"{label(result) if result else 'not probed'}")

    models_by_id = {result["model"]: result for result in results}
    # Severity and the alerted flag must live beside the probe result: the next run reads them back
    # to decide whether a condition is confirmed, and therefore whether it is worth saying out loud.
    for model, info in history.items():
        if model in models_by_id:
            models_by_id[model].update(info)

    report = {
        "version": STATE_VERSION,
        "run_at": _now_iso(),
        "base": base,
        "ttft_cap_s": ttft,
        "attempts": attempts,
        "models": models_by_id,
        "configured_in": used,
        "catalog": live_catalog or previous_catalog,
        "catalog_fetch_ok": catalog_ok,
        "catalog_seen": merge_seen(previous_seen, live_catalog),
        "new_models_probed": {model: {"status": r["status"], "ttft_s": r["ttft_s"]}
                              for model, r in arrivals.items()},
        "history": history,
        "changes": changes,
        "removed": removed,
        "added": added,
        "host": socket.gethostname(),
        "user": getpass.getuser(),
    }
    save_state(report, configured_dir)
    return report


def scan_age_seconds(report: dict) -> float | None:
    try:
        when = datetime.fromisoformat(str(report.get("run_at")).replace("Z", "+00:00"))
    except Exception:  # noqa: BLE001
        return None
    return (datetime.now(timezone.utc) - when).total_seconds()


__all__ = [
    "params_from_id", "rank_key", "reliability",
    "DEFAULT_BASE", "STATE_VERSION", "api_key", "key_hint", "attempt", "probe", "catalog",
    "configured_models", "severity", "label", "load_state", "save_state", "scan",
    "scan_age_seconds", "merge_seen", "removed_ids", "added_ids", "state_dir", "state_path",
    "hermes_home",
]
