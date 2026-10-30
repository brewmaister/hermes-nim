"""hermes-nim engine package: probing, state, rendering, and the two doors onto them.

Note the deliberate absence of a `settings` re-export: binding the *function* `settings.settings`
onto this package would shadow the `nim.settings` submodule, and every `from . import settings`
elsewhere would then get a function where it expects the module. Re-export only what is unambiguous.
"""

from __future__ import annotations

from .settings import configure

__all__ = ["configure"]
