#!/usr/bin/env python3
"""Cron door: print the NIM report only when something changed, so a healthy tick is silent.

This is the file to point a `no_agent` cron job at. From the installed plugin:

    ~/.hermes/scripts/nim-watch.py        # a two-line shim, see README
    hermes nim watch                      # same thing via the CLI

The cron engine delivers stdout itself — do NOT also pass `--send`, or every alert arrives twice.

Exit codes: 0 always (a watcher that fails loudly on a transient network blip is a watcher that
gets muted).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Resolve the plugin whether this file is run from the plugin tree or from ~/.hermes/scripts.
_HERE = Path(__file__).resolve()
for _candidate in (_HERE.parents[1], Path.home() / ".hermes" / "plugins" / "hermes-nim"):
    if (_candidate / "nim" / "watch.py").exists():
        sys.path.insert(0, str(_candidate))
        break

from nim.watch import run_watch  # noqa: E402

if __name__ == "__main__":
    argv = set(sys.argv[1:])
    sys.exit(run_watch(verbose="--verbose" in argv or "-v" in argv, send="--send" in argv))
