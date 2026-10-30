"""Which build of this plugin is actually executing — the answer to "did the reload land?".

A gateway reload receipt proves that *something* was re-imported; it carries no version string. So
``/nim version`` (the code the running process imported) and ``hermes nim version`` (a fresh
process, always the disk copy) are the pair: if the fresh-process number is newer, the gateway is
still serving an older import.
"""

from __future__ import annotations

import os
from pathlib import Path

_PLUGIN_YAML = Path(__file__).resolve().parents[1] / "plugin.yaml"


def _read_version() -> str:
    try:
        for line in _PLUGIN_YAML.read_text(encoding="utf-8").splitlines():
            if line.startswith("version:"):
                return line.split(":", 1)[1].strip().strip('"').strip("'")
    except OSError:
        pass
    return "unknown"


# Bound at import: a force reload evicts this module too, so the next import re-reads plugin.yaml.
PLUGIN_VERSION = _read_version()


def stamp() -> str:
    return f"hermes-nim {PLUGIN_VERSION} (pid {os.getpid()})"


def in_gateway_hint() -> str:
    return ("the build this process imported — `hermes nim version` in a fresh shell always "
            "reads the disk copy, so if that number is higher the running process is stale")
