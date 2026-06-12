from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared GPU device resolution and degradation state."""

import logging
import threading
from datetime import UTC, datetime
from typing import Any, Literal

logger = logging.getLogger(__name__)

DevicePreference = Literal["auto", "cuda", "cpu"]
ResolvedDevice = Literal["cuda", "cpu"]

_VALID_COMPONENTS = {"embedding", "nli", "reranker"}
_CUDA_FAILURE_MARKERS = (
    "cuda",
    "cublas",
    "cudnn",
    "gpu",
    "nvidia",
    "xid",
    "device",
)

_state_lock = threading.Lock()
_component_devices: dict[str, ResolvedDevice] = {}
_failure_state: dict[str, dict[str, str]] = {}
_cuda_unavailable_warned = False


def _cuda_available_safely() -> bool:
    """Return whether CUDA can be initialized without raising."""
    global _cuda_unavailable_warned

    try:
        import torch

        return bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
    except Exception as exc:
        if not _cuda_unavailable_warned:
            logger.warning("CUDA unavailable; using CPU: %s", exc)
            _cuda_unavailable_warned = True
        return False


def _normalize_component(component: str) -> str:
    normalized = component.strip().lower()
    if normalized not in _VALID_COMPONENTS:
        raise ValueError(f"Unknown GPU component: {component!r}")
    return normalized


def _preference_from_config(config: Any, component: str) -> DevicePreference:
    defaults: dict[str, DevicePreference] = {
        "embedding": "auto",
        "nli": "cpu",
        "reranker": "cpu",
    }
    attr_by_component = {
        "embedding": "embedding_device",
        "nli": "nli_device",
        "reranker": "reranker_device",
    }
    gpu_cfg = getattr(config, "gpu", None)
    value = getattr(gpu_cfg, attr_by_component[component], defaults[component])
    if value in {"auto", "cuda", "cpu"}:
        return value
    return defaults[component]


def _embedding_auto_uses_gpu(config: Any) -> bool:
    rag_cfg = getattr(config, "rag", None)
    return bool(getattr(rag_cfg, "use_gpu", False))


def resolve_device(component: str) -> ResolvedDevice:
    """Resolve a component's execution device from config and CUDA health.

    ``embedding`` preserves the historical ``rag.use_gpu`` behavior when the
    new ``gpu.embedding_device`` preference is ``auto``.  Other components use
    CPU by default, and ``auto`` means "use CUDA only when it is safely
    available."
    """
    component = _normalize_component(component)
    try:
        from core.config import load_config

        config = load_config()
    except Exception:
        config = None

    preference = _preference_from_config(config, component)
    if preference == "cpu":
        return "cpu"
    if preference == "cuda":
        if _cuda_available_safely():
            return "cuda"
        logger.warning("CUDA requested for %s but unavailable; using CPU", component)
        return "cpu"

    if component == "embedding" and not _embedding_auto_uses_gpu(config):
        return "cpu"
    return "cuda" if _cuda_available_safely() else "cpu"


def is_cuda_failure(exc: BaseException) -> bool:
    """Return whether an exception looks like a CUDA/GPU runtime failure."""
    if isinstance(exc, RuntimeError):
        message = str(exc).lower()
        return any(marker in message for marker in _CUDA_FAILURE_MARKERS)
    return False


def record_component_device(component: str, device: str) -> None:
    """Record the actual device currently used by a component."""
    component = _normalize_component(component)
    if device not in {"cuda", "cpu"}:
        return
    with _state_lock:
        _component_devices[component] = device


def record_gpu_failure(component: str, exc: BaseException) -> None:
    """Record a CUDA/GPU failure for status endpoints and future fallbacks."""
    component = _normalize_component(component)
    detected_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    message = f"{type(exc).__name__}: {exc}"
    with _state_lock:
        _failure_state[component] = {
            "component": component,
            "last_error": message[:1000],
            "detected_at": detected_at,
        }


def is_component_degraded(component: str) -> bool:
    """Return whether a component has recorded a GPU failure."""
    component = _normalize_component(component)
    with _state_lock:
        return component in _failure_state


def get_gpu_status() -> dict[str, object]:
    """Return a compact GPU status payload for health/status endpoints."""
    with _state_lock:
        embedding_device = _component_devices.get("embedding")
        failures = list(_failure_state.values())

    if embedding_device is None:
        try:
            embedding_device = resolve_device("embedding")
        except Exception:
            embedding_device = "cpu"

    latest_failure = max(failures, key=lambda item: item["detected_at"]) if failures else None
    return {
        "embedding_device": embedding_device,
        "degraded": latest_failure is not None,
        "last_error": latest_failure["last_error"] if latest_failure else None,
        "detected_at": latest_failure["detected_at"] if latest_failure else None,
    }


def reset_gpu_status_for_testing() -> None:
    """Reset GPU status state for unit tests."""
    global _cuda_unavailable_warned
    with _state_lock:
        _component_devices.clear()
        _failure_state.clear()
        _cuda_unavailable_warned = False
