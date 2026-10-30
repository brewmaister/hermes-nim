#!/usr/bin/env bash
# Reload plugin code in the RUNNING gateway, in place — no restart, no dropped sessions.
#
# Why this exists: installing or updating a plugin moves the clone on disk, and nothing reloads the
# gateway, so the process keeps serving the handlers it imported at startup. `/nim` (the in-gateway
# slash command) stays on the old code until something reloads it. `hermes nim` from a shell is
# always current, because it is a fresh process — that difference is exactly what makes the
# staleness invisible.
#
#   ./scripts/reload-gateway.sh [name]     name narrows the output to activations whose key contains it
#
# Socket resolution: ask the gateway serving $HERMES_HOME first. When that path has no live socket,
# retry through the ROOT home ($HERMES_ROOT_HOME, default $HOME/.hermes) and pass the profile as
# profile_home. A caller whose socket answers nothing gets exit 2 and a diagnostic: a reload that
# never happened must never read as success.
#
# Exit: 0 an activation list came back · 2 no core venv, no socket client, or no gateway answered ·
#       3 the gateway answered but reloaded nothing (no activations, or the name filter matched none).
#
# Do not spell out a restart command in this file, in a comment or anywhere else: the terminal
# lifecycle guard scans script text for one and kills the script on a match, even in prose.

set -uo pipefail

CORE="${HERMES_CORE:-$HOME/.hermes/hermes-agent}"
HERMES_DIR="${HERMES_HOME:-$HOME/.hermes}"
export HERMES_ROOT_HOME="${HERMES_ROOT_HOME:-$HOME/.hermes}"

if [ ! -x "$CORE/venv/bin/python" ]; then
  echo "no core venv at $CORE — cycle the service by hand, outside this process" >&2
  exit 2
fi

exec "$CORE/venv/bin/python" - "$CORE" "$HERMES_DIR" "$@" <<'PY'
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, sys.argv[1])
try:
    from gateway.control_socket import reload_gateway_plugins, resolve_client_socket_path
except Exception as exc:
    print(f"no control-socket client ({exc}) — cycle the service by hand, outside this process",
          file=sys.stderr)
    raise SystemExit(2)

home = Path(sys.argv[2]).expanduser()
root = Path(os.environ.get("HERMES_ROOT_HOME") or (Path.home() / ".hermes")).expanduser()
wanted = sys.argv[3] if len(sys.argv) > 3 else ""


def describe(path: Path) -> str:
    sock = resolve_client_socket_path(path)
    if sock is None:
        return f"{path}: no control socket"
    try:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(sock.stat().st_mtime))
    except OSError:
        stamp = "unknown"
    return f"{path}: socket {sock} (last written {stamp}) exists but nobody answered — stale socket?"


tried = [home]
out = reload_gateway_plugins(home) or {}
via = home
if not out and root != home:
    tried.append(root)
    out = reload_gateway_plugins(root, profile_home=home) or {}
    via = root

if not out:
    print("NOT reloaded — no gateway answered; the running process still serves its startup code.",
          file=sys.stderr)
    for path in tried:
        print("  " + describe(path), file=sys.stderr)
    print("  cycle the service by hand from a separate shell, then re-run with its home set",
          file=sys.stderr)
    raise SystemExit(2)

if via != home:
    print(f"note: {describe(home)} reloaded through the root gateway {root} "
          f"with profile_home={home}", file=sys.stderr)

shown = 0
activations = out.get("activations") or []
if not activations:
    print(f"NOT reloaded — the gateway at {via} answered but reported no plugin activations for "
          f"profile_home={home}; nothing was re-imported.", file=sys.stderr)
    raise SystemExit(3)

for activation in activations:
    key = str(activation.get("key") or "")
    if wanted and wanted not in key:
        continue
    now = activation.get("activated_now")
    print(f"{key}: {now if now else 'unchanged'}")
    shown += 1
if wanted and not shown:
    keys = [str(a.get("key") or "") for a in activations]
    print(f"NOT reloaded — {wanted!r} matched no activation; the gateway reported {keys}. "
          f"The name you filtered on does not exist there.", file=sys.stderr)
    raise SystemExit(3)
PY
