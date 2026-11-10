"""Congestion probe for self-hosted fallback models (vLLM ``/metrics`` or llm-gate).

A ``models.json`` entry may declare either probe::

    "busy_probe": {"metrics_url": "http://host:port/metrics",
                   "max_running": 2, "max_waiting": 0}

    "busy_probe": {"gate_status_url": "http://host:port/gate/status",
                   "max_waiting": 0}

``metrics_url`` reads a single vLLM server.  ``gate_status_url`` reads an
llm-gate router, which fronts several servers: it is busy when no backend that
is up has a free seat, or when more than ``max_waiting`` requests are queued
(the gate seats queued dsh requests first, so any queue delays an anima).

The fallback resolver skips such a candidate while it is congested so that a
slower queue does not stall the cycle when a later candidate is free.  The
probe is fail-open: any fetch/parse error reports "not busy".
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.request
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

_CACHE_TTL_S = 15.0
_TIMEOUT_S = 2.0
_cache: dict[str, tuple[float, tuple[float, float] | None]] = {}
_lock = threading.Lock()


def _parse_vllm_load(text: str) -> tuple[float, float]:
    """Sum ``vllm:num_requests_running`` / ``waiting`` across engines."""
    running = waiting = 0.0
    for line in text.splitlines():
        if line.startswith("vllm:num_requests_running"):
            running += float(line.rsplit(" ", 1)[1])
        elif line.startswith("vllm:num_requests_waiting") and not line.startswith(
            "vllm:num_requests_waiting_by_reason"
        ):
            waiting += float(line.rsplit(" ", 1)[1])
    return running, waiting


def _parse_gate_load(text: str) -> tuple[float, float]:
    """Return ``(free_seats, waiting)`` from an llm-gate ``/gate/status`` body."""
    data = json.loads(text)
    free = 0.0
    for backend in data.get("backends") or []:
        if backend.get("up"):
            free += max(0.0, float(backend.get("slots", 0)) - float(backend.get("busy", 0)))
    return free, float(data.get("waiting", 0))


def _fetch_load(
    url: str,
    parse: Callable[[str], tuple[float, float]] = _parse_vllm_load,
) -> tuple[float, float] | None:
    now = time.monotonic()
    with _lock:
        hit = _cache.get(url)
        if hit is not None and now - hit[0] < _CACHE_TTL_S:
            return hit[1]
    try:
        with urllib.request.urlopen(url, timeout=_TIMEOUT_S) as resp:  # noqa: S310 - operator-configured URL
            load: tuple[float, float] | None = parse(resp.read().decode("utf-8", "replace"))
    except Exception:
        logger.debug("busy probe failed for %s", url, exc_info=True)
        load = None
    with _lock:
        _cache[url] = (now, load)
    return load


def is_model_busy(entry: dict[str, Any] | None) -> bool:
    """Return True when the models.json *entry*'s ``busy_probe`` reports congestion."""
    probe = (entry or {}).get("busy_probe")
    if not isinstance(probe, dict):
        return False
    if probe.get("gate_status_url"):
        load = _fetch_load(str(probe["gate_status_url"]), _parse_gate_load)
        if load is None:
            return False
        free, waiting = load
        return free <= 0 or waiting > float(probe.get("max_waiting", 0))
    if not probe.get("metrics_url"):
        return False
    load = _fetch_load(str(probe["metrics_url"]))
    if load is None:
        return False
    running, waiting = load
    return running >= float(probe.get("max_running", 2)) or waiting > float(probe.get("max_waiting", 0))
