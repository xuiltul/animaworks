# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Dynamic discovery of the "mode + model" catalog from the installed CLIs.

Each execution path (Claude SDK, Codex CLI, Grok CLI, OpenAI-compatible
endpoints, Ollama) is probed on demand so the UI can offer the models that
are *actually* usable right now, rather than a hard-coded snapshot.

Every probe is split into a pure *parse* function (unit-testable with no
subprocess) and an *execute* wrapper that runs the CLI / HTTP round-trip.
All probes run concurrently with a short timeout; a single failure never
blocks the others.  Results are cached for ``CACHE_TTL_SECONDS``.  Expired
results remain usable while the server refreshes them in the background, so
opening a model picker never pays the recurring probe latency.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import re
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import httpx

from core.config.local_llm import normalize_ollama_base_url, normalize_ollama_model_name
from core.config.model_mode import resolve_execution_mode
from core.config.models import load_config
from core.config.schemas import LocalLLMConfig
from core.platform.codex import get_codex_executable, is_codex_login_available
from core.platform.grok import get_grok_executable, is_grok_authenticated

logger = logging.getLogger(__name__)

# Cache TTL for the resolved catalog (seconds).
CACHE_TTL_SECONDS = 300.0
# Per-probe timeouts (seconds): CLI probes are given more head-room than HTTP.
CLI_TIMEOUT = 10.0
HTTP_TIMEOUT = 5.0

# Preferred group order for the UI dropdown; anything else sorts alphabetically.
_GROUP_ORDER = {"Claude": 0, "Codex": 1, "Grok": 2, "Ollama": 3}


@dataclass(frozen=True)
class DiscoveredModel:
    id: str  # value that goes straight into ChatRequest.model, e.g. "c:codex/gpt-5.6-sol"
    mode: str  # lowercase single letter, e.g. "c"
    model: str  # mode stripped, e.g. "codex/gpt-5.6-sol"
    label: str  # UI display name, e.g. "GPT-5.6-Sol"
    group: str  # UI group heading, e.g. "Codex" / "Claude" / "<credential>"
    note: str = ""  # optional description (tooltip)
    source: str = ""  # "codex-cli" / "grok-cli" / "claude-cli" / "openai-compatible" / "ollama" / "static"


# ── Module-level cache ────────────────────────────────────────────────────
_cache: dict[str, Any] = {}
_cache_lock = threading.Lock()
_cache_refreshing = False


def invalidate_cache() -> None:
    """Clear the cached discovery result (force a fresh probe on next call)."""
    global _cache, _cache_refreshing
    with _cache_lock:
        _cache = {}
        _cache_refreshing = False


# ── Parse functions (pure, unit-testable, no network) ────────────────────


def _parse_codex_models(raw: str) -> list[dict[str, Any]]:
    """Parse ``codex debug models`` JSON into UI-visible models.

    Only ``visibility == "list"`` entries are returned, sorted by
    ``priority`` ascending (``"hide"`` entries are internal-use only).
    """
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return []
    entries = data.get("models", []) if isinstance(data, dict) else []
    models: list[dict[str, Any]] = []
    for item in entries:
        if not isinstance(item, dict):
            continue
        slug = str(item.get("slug", "")).strip()
        if not slug:
            continue
        if str(item.get("visibility", "")).strip() != "list":
            continue
        try:
            priority = int(item.get("priority", 0) or 0)
        except (TypeError, ValueError):
            priority = 0
        models.append(
            {
                "slug": slug,
                "label": str(item.get("display_name", "")).strip() or slug,
                "priority": priority,
            }
        )
    models.sort(key=lambda m: m["priority"])
    return models


def _parse_grok_models(text: str) -> list[str]:
    """Parse ``grok models`` output into model names.

    Only bullet entries under ``Available models:`` are collected and the
    ``(default)`` marker is never part of the returned name.
    """
    models: list[str] = []
    in_available = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("Available models:"):
            in_available = True
            continue
        if not in_available:
            continue
        if stripped.startswith(("*", "-")) and " " in stripped:
            parts = stripped.split()
            models.append(parts[1])
        elif stripped:
            # A new (non-bullet) section after the list — stop collecting.
            break
    seen: set[str] = set()
    result: list[str] = []
    for name in models:
        if name not in seen:
            seen.add(name)
            result.append(name)
    return result


def _parse_claude_help(help_text: str) -> list[str]:
    """Extract the latest-model aliases from ``claude --help``.

    Parses the ``--model`` option block and returns the single-quoted
    aliases listed beside "an alias for the latest model (e.g. 'fable',
    'opus', or 'sonnet')".  The full-name example (e.g. ``claude-fable-5``)
    from the "a model's full name" sentence is ignored.
    """
    lines = help_text.splitlines()
    idx = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("--model") and "<model>" in stripped:
            idx = i
            break
    if idx is None:
        return []

    # Join the description block up to the next option line.
    block = lines[idx]
    for line in lines[idx + 1 :]:
        stripped = line.strip()
        if stripped.startswith("--"):
            break
        block += " " + stripped

    marker = "an alias for the latest model"
    pos = block.find(marker)
    if pos == -1:
        return []
    tail = block[pos:]
    start = tail.find("(")
    end = tail.find(")", start)
    if start == -1 or end == -1:
        return []
    return re.findall(r"'([^']+)'", tail[start + 1 : end])


# ── HTTP helpers ─────────────────────────────────────────────────────────


def _http_get_models(base_url: str, api_key: str, timeout: float = HTTP_TIMEOUT) -> list[str]:
    """Fetch ``data[].id`` list from an OpenAI-compatible ``{base_url}/models``."""
    try:
        response = httpx.get(
            f"{base_url}/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(timeout, connect=timeout),
        )
        response.raise_for_status()
        data = response.json()
        return sorted({str(item.get("id", "")).strip() for item in data.get("data", []) if item.get("id")})
    except Exception:  # noqa: BLE001 - probe results are best-effort
        return []


def _list_ollama_models(base_url: str, timeout: float = HTTP_TIMEOUT) -> list[str]:
    """Fetch local Ollama model names from ``{base_url}/api/tags``."""
    try:
        response = httpx.get(
            f"{base_url}/api/tags",
            timeout=httpx.Timeout(timeout, connect=timeout),
        )
        response.raise_for_status()
        data = response.json()
        models = sorted(
            {
                normalize_ollama_model_name(str(item.get("name", "")).strip())
                for item in data.get("models", [])
                if item.get("name")
            }
        )
        return [model for model in models if model]
    except Exception:  # noqa: BLE001 - probe results are best-effort
        return []


# ── Probe implementations (execute the CLI / HTTP, build models) ──────────


def _probe_codex() -> list[DiscoveredModel]:
    if not is_codex_login_available():
        return []
    executable = get_codex_executable()
    if not executable:
        return []
    try:
        result = subprocess.run(
            [executable, "debug", "models"],
            capture_output=True,
            text=True,
            timeout=CLI_TIMEOUT,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        logger.debug("codex debug models failed")
        return []
    parsed = _parse_codex_models(result.stdout or "")
    return [
        DiscoveredModel(
            id=f"c:codex/{m['slug']}",
            mode="c",
            model=f"codex/{m['slug']}",
            label=m["label"],
            group="Codex",
            note="",
            source="codex-cli",
        )
        for m in parsed
    ]


def _probe_grok() -> list[DiscoveredModel]:
    if not is_grok_authenticated() or not get_grok_executable():
        return []
    executable = get_grok_executable()
    try:
        result = subprocess.run(
            [executable, "models"],
            capture_output=True,
            text=True,
            timeout=CLI_TIMEOUT,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        logger.debug("grok models failed")
        return []
    names = _parse_grok_models((result.stdout or "") + (result.stderr or ""))
    return [
        DiscoveredModel(
            id=f"x:grok/{name}",
            mode="x",
            model=f"grok/{name}",
            label=name,
            group="Grok",
            source="grok-cli",
        )
        for name in names
    ]


def _probe_claude() -> list[DiscoveredModel]:
    if shutil.which("claude") is None:
        return []
    try:
        result = subprocess.run(
            ["claude", "--help"],
            capture_output=True,
            text=True,
            timeout=CLI_TIMEOUT,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        logger.debug("claude --help failed")
        return []
    aliases = _parse_claude_help((result.stdout or "") + (result.stderr or ""))
    return [
        DiscoveredModel(
            id=f"s:{alias}",
            mode="s",
            model=alias,
            label=alias,
            group="Claude",
            source="claude-cli",
        )
        for alias in aliases
    ]


def _probe_openai_compatible(config: Any) -> list[DiscoveredModel]:
    """Probe each OpenAI-compatible credential's ``GET /models`` endpoint."""
    newline_models: list[DiscoveredModel] = []
    credentials = getattr(config, "credentials", {}) or {}
    for name, cred in credentials.items():
        base = (getattr(cred, "base_url", None) or "").strip().rstrip("/")
        api_key = (getattr(cred, "api_key", None) or "").strip()
        if not base or not api_key:
            continue
        if "openai.azure.com" in base or "api.aws" in base:
            continue  # Azure / Bedrock have no classic /models endpoint.
        for model in _http_get_models(base, api_key):
            newline_models.append(
                DiscoveredModel(
                    id=f"a:openai/{model}",
                    mode="a",
                    model=f"openai/{model}",
                    label=model,
                    group=str(name),
                    source="openai-compatible",
                )
            )
    return newline_models


def _probe_ollama(config: Any) -> list[DiscoveredModel]:
    try:
        local_llm = LocalLLMConfig.model_validate(config.local_llm.model_dump())
        base_url = normalize_ollama_base_url(local_llm.base_url)
    except Exception:  # noqa: BLE001
        return []
    results: list[DiscoveredModel] = []
    for name in _list_ollama_models(base_url):
        norm = name if name.startswith("ollama/") else f"ollama/{name}"
        bare = norm.removeprefix("ollama/")
        mode = resolve_execution_mode(config, norm).lower()
        results.append(
            DiscoveredModel(
                id=f"{mode}:{norm}",
                mode=mode,
                model=norm,
                label=bare,
                group="Ollama",
                source="ollama",
            )
        )
    return results


def _static_fallback(config: Any) -> list[DiscoveredModel]:
    """Fall back to the static catalog when every dynamic probe came up empty."""
    from core.config.model_catalog import _build_static_model_catalog  # local import to avoid cycle

    results: list[DiscoveredModel] = []
    for entry in _build_static_model_catalog(config) or []:
        mid = str(entry.get("id", ""))
        if not mid:
            continue
        mode = resolve_execution_mode(config, mid, entry.get("mode") or None).lower()
        full_id = f"{mode}:{mid}"
        results.append(
            DiscoveredModel(
                id=full_id,
                mode=mode,
                model=mid,
                label=str(entry.get("label", mid)),
                group=str(entry.get("credential", "")),
                source="static",
            )
        )
    return results


# ── Orchestration ────────────────────────────────────────────────────────


def _dedupe_and_sort(models: list[DiscoveredModel]) -> list[DiscoveredModel]:
    """Drop duplicate ids (first wins) and order by group preference."""
    seen: set[str] = set()
    result: list[DiscoveredModel] = []
    for model in models:
        if model.id in seen:
            continue
        seen.add(model.id)
        result.append(model)
    result.sort(key=lambda m: (_GROUP_ORDER.get(m.group, 100), m.group))
    return result


def _discover_uncached(config: Any = None) -> list[DiscoveredModel]:
    """Run all discovery probes and return a newly built catalog."""
    if config is None:
        config = load_config()

    probes: list[Callable[[], list[DiscoveredModel]]] = [
        _probe_codex,
        _probe_grok,
        _probe_claude,
        lambda: _probe_openai_compatible(config),
        lambda: _probe_ollama(config),
    ]

    merged: list[DiscoveredModel] = []
    with ThreadPoolExecutor(max_workers=len(probes)) as executor:
        futures = [executor.submit(probe) for probe in probes]
        for future in concurrent.futures.as_completed(futures):
            try:
                merged.extend(future.result())
            except Exception:  # noqa: BLE001 - one probe must never block the rest
                logger.warning("model discovery probe raised", exc_info=True)

    result = _dedupe_and_sort(merged)
    if not result:
        result = _dedupe_and_sort(_static_fallback(config))
    else:
        from core.config.model_catalog import _configured_model_entries

        configured = []
        for entry in _configured_model_entries(config):
            model = entry["id"]
            mode = resolve_execution_mode(config, model, entry.get("mode") or None).lower()
            configured.append(
                DiscoveredModel(
                    id=f"{mode}:{model}",
                    mode=mode,
                    model=model,
                    label=entry["label"],
                    group=entry["credential"] or "Configured",
                    source="configured",
                )
            )
        result = _dedupe_and_sort([*result, *configured])
    return result


def _refresh_cache_in_background(config: Any = None) -> None:
    """Refresh an expired catalog without delaying the current caller."""
    global _cache, _cache_refreshing
    try:
        result = _discover_uncached(config)
        with _cache_lock:
            _cache = {"timestamp": time.monotonic(), "models": list(result)}
    except Exception:  # noqa: BLE001 - preserve the last usable catalog
        logger.warning("background model catalog refresh failed", exc_info=True)
    finally:
        with _cache_lock:
            _cache_refreshing = False


def discover_models(*, refresh: bool = False, config: Any = None) -> list[DiscoveredModel]:
    """Return the dynamically discovered catalog of DiscoveredModel objects.

    The first call (or ``refresh=True``) probes all execution backends.
    Normal calls return the server-side cache immediately.  Once its TTL has
    elapsed, that stale-but-usable value is returned and one daemon thread
    refreshes it for subsequent calls.
    """
    now = time.monotonic()
    global _cache, _cache_refreshing
    with _cache_lock:
        if not refresh and _cache:
            if now - _cache.get("timestamp", 0.0) >= CACHE_TTL_SECONDS and not _cache_refreshing:
                _cache_refreshing = True
                threading.Thread(
                    target=_refresh_cache_in_background,
                    args=(config,),
                    name="model-catalog-refresh",
                    daemon=True,
                ).start()
            return list(_cache["models"])

    result = _discover_uncached(config)

    with _cache_lock:
        _cache = {"timestamp": time.monotonic(), "models": list(result)}
    return list(result)


def discovered_model_ids(*, refresh: bool = False, config: Any = None) -> set[str]:
    """Return the set of all acceptable override ids (3 forms per model).

    For each DiscoveredModel this adds its full ``mode:model`` id, its bare
    ``model``, and the ``/``-separated tail so ``validate_model_override``
    can match a ``mode:model`` request against just the model part.
    """
    models = discover_models(refresh=refresh, config=config)
    ids: set[str] = set()
    for model in models:
        ids.add(model.id)
        ids.add(model.model)
        tail = model.model.rpartition("/")[2]
        if tail:
            ids.add(tail)
    return ids


__all__ = [
    "DiscoveredModel",
    "discover_models",
    "discovered_model_ids",
    "invalidate_cache",
]
