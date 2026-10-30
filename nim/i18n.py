"""Localization for hermes-nim: one lookup, one fallback chain, no hardcoded user-facing English.

Resolution order for a key — highest layer first:

1. ``$HERMES_HOME/locales/<lang>.yaml`` — the *user's* overlay, so any single line can be reworded
   without forking the repo;
2. this plugin's ``locales/<lang>.yaml``;
3. this plugin's ``locales/en.yaml``;
4. the bare key (a visible miss, never a crash).

The host's own ``agent.i18n.t`` is deliberately *not* in this chain. Its bundled catalogs fail to
load outside the Hermes install directory (they import a repo-root ``hermes_yaml`` module that is not
on ``sys.path`` for a plugin run), it prints a warning for each such file, and it resolves the
language from a config cache we cannot pin. Reading our own files is deterministic, silent, and works
identically in the gateway, the CLI and the cron script.

Resolution of *which* language:

1. an explicit ``lang=`` argument (tests, one-off renders);
2. ``NIM_LANGUAGE`` — the plugin's own setting / env override;
3. ``HERMES_LANGUAGE`` — the process-wide override Hermes itself honours;
4. ``display.language`` from config.yaml;
5. ``en``.

Fallback order for a *key* is the opposite direction and matters more: a partly translated language
must degrade to English per key, never to a dotted key name in the user's face, and never raise. A
probe watchdog that dies because a translation file is missing is far worse than an English sentence.
"""

from __future__ import annotations

import logging
import os
import time
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

LOCALE_DIR = Path(__file__).resolve().parent.parent / "locales"
FALLBACK_LANGUAGE = "en"
LANGUAGE_ENV = "NIM_LANGUAGE"


@lru_cache(maxsize=8)
def _catalog(language: str) -> dict[str, str]:
    """Flat ``{key: text}`` for *language*, read from this plugin's own locales directory."""
    path = LOCALE_DIR / f"{language}.yaml"
    try:
        import yaml

        raw = yaml.safe_load(path.read_text()) or {}
    except FileNotFoundError:
        return {}
    except Exception as exc:  # noqa: BLE001 - a broken catalog must not break the command
        logger.warning("hermes-nim locale %s unreadable: %s", path.name, exc)
        return {}
    flat: dict[str, str] = {}

    def walk(node, prefix=""):
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, f"{prefix}{key}.")
        elif isinstance(node, str):
            flat[prefix[:-1]] = node

    walk(raw)
    return flat


@lru_cache(maxsize=8)
def _overlay(language: str) -> dict[str, str]:
    """The user's own ``$HERMES_HOME/locales/<lang>.yaml``, flattened — highest priority."""
    try:
        from .engine import hermes_home

        path = hermes_home() / "locales" / f"{language}.yaml"
    except Exception:  # noqa: BLE001
        return {}
    try:
        import yaml

        raw = yaml.safe_load(path.read_text()) or {}
    except FileNotFoundError:
        return {}
    except Exception as exc:  # noqa: BLE001 - a typo in an override must not break the command
        logger.warning("hermes-nim: user overlay %s unreadable: %s", path, exc)
        return {}
    flat: dict[str, str] = {}

    def walk(node, prefix=""):
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, f"{prefix}{key}.")
        elif isinstance(node, str):
            flat[prefix[:-1]] = node

    walk(raw)
    return flat


def available() -> tuple[str, ...]:
    """Language ids this plugin ships, English first."""
    try:
        found = sorted(path.stem for path in LOCALE_DIR.glob("*.yaml"))
    except OSError:
        found = []
    return tuple([FALLBACK_LANGUAGE] + [lang for lang in found if lang != FALLBACK_LANGUAGE])


def _config_language() -> str:
    try:
        from hermes_cli.config import load_config_readonly

        display = (load_config_readonly() or {}).get("display") or {}
        return str(display.get("language") or "").strip().lower()
    except Exception:  # noqa: BLE001 - CLI without the host package, or an unreadable config
        return ""


_CACHE: tuple[float, str] | None = None
_CACHE_TTL_S = 5.0


def _plugin_language() -> str:
    """``plugins.entries.hermes-nim.settings.language``; empty when unset or ``auto``."""
    try:
        from .settings import settings

        value = str(getattr(settings(), "language", "") or "").strip().lower()
    except Exception:  # noqa: BLE001 - no ctx (cron, CLI before register) is not an error
        return ""
    return "" if value in {"", "auto"} else value


def language(*, override: str | bool | None = None) -> str:
    """The language to render in. Never returns an id this plugin cannot serve.

    Cached for a few seconds: ``t()`` runs dozens of times per report and re-reading config.yaml for
    each line would be silly, while a long-lived cache would make ``/nim`` ignore a fresh setting.
    """
    explicit = str(override or "").strip().lower()
    if explicit and explicit not in {"auto", "true"}:
        return explicit
    global _CACHE
    now = time.monotonic()
    if _CACHE is not None and now - _CACHE[0] < _CACHE_TTL_S:
        return _CACHE[1]
    resolved = FALLBACK_LANGUAGE
    for candidate in (os.environ.get(LANGUAGE_ENV), os.environ.get("HERMES_LANGUAGE"),
                      _plugin_language(), _config_language()):
        value = str(candidate or "").strip().lower()
        if value and value not in {"auto", "true"}:
            resolved = value
            break
    _CACHE = (now, resolved)
    return resolved


def reset_cache() -> None:
    """Drop cached language and catalogs — for tests and after a settings change."""
    global _CACHE
    _CACHE = None
    _catalog.cache_clear()
    _overlay.cache_clear()


def t(key: str, *, lang: str | None = None, **fields) -> str:
    """Translate ``key`` for the active language, formatting ``{placeholders}`` afterwards.

    Formatting is attempted per catalog entry, so a translation that forgot a placeholder can never
    turn into a ``KeyError`` in the middle of a report — the entry with the missing placeholder is
    skipped and the next language is tried.
    """
    resolved = language() if lang is None else lang
    overlay = _overlay(resolved).get(key)
    if overlay is not None:
        try:
            return overlay.format(**fields) if fields else overlay
        except (KeyError, IndexError, ValueError):
            logger.warning("hermes-nim overlay %s: %r does not fit %s", resolved, key, fields)
    for candidate in (resolved, FALLBACK_LANGUAGE):
        text = _catalog(candidate).get(key)
        if text is None:
            continue
        try:
            return text.format(**fields) if fields else text
        except (KeyError, IndexError, ValueError):
            logger.warning("hermes-nim locale %s: %r does not fit %s", candidate, key, fields)
            continue
    if fields:
        try:
            return _catalog(FALLBACK_LANGUAGE).get(key, key).format(**fields)
        except Exception:  # noqa: BLE001
            pass
    return key


def has(key: str, *, lang: str | None = None) -> bool:
    """Whether *lang* translates *key* — used by the parity test, not by rendering."""
    return key in _catalog(lang or language())


__all__ = ["LOCALE_DIR", "available", "language", "t", "has", "reset_cache"]
