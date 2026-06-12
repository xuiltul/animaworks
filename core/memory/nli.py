from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared Natural Language Inference helper for memory consistency checks."""

import logging
import threading
from typing import Any

logger = logging.getLogger("animaworks.memory.nli")

_cache_lock = threading.Lock()
_inference_lock = threading.Lock()
_pipeline_cache: dict[tuple[str, str], Any] = {}


def _pipeline_device_arg(device: str) -> int:
    return 0 if device == "cuda" else -1


def _load_cached_pipeline(model_name: str, device: str):
    from transformers import pipeline as hf_pipeline

    key = (model_name, device)
    with _cache_lock:
        cached = _pipeline_cache.get(key)
        if cached is not None:
            return cached
        pipeline = hf_pipeline(
            "text-classification",
            model=model_name,
            device=_pipeline_device_arg(device),
        )
        _pipeline_cache[key] = pipeline
        return pipeline


def _reset_for_testing() -> None:
    with _cache_lock:
        _pipeline_cache.clear()


class SharedNLIModel:
    """Lazy wrapper around the multilingual NLI classifier.

    This helper is intentionally small: contradiction detection needs local
    entailment/contradiction labels, while LLM review stays in the detector.
    """

    NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"

    def __init__(self) -> None:
        self._nli_pipeline = None
        self._nli_available = True
        from core.gpu import is_component_degraded, resolve_device

        self._device = "cpu" if is_component_degraded("nli") else resolve_device("nli")

    def _load_nli_model(self) -> None:
        """Load the NLI model. GPU -> CPU fallback. On failure, NLI is skipped."""
        from core.gpu import record_component_device, record_gpu_failure

        try:
            try:
                self._nli_pipeline = _load_cached_pipeline(self.NLI_MODEL, self._device)
                record_component_device("nli", self._device)
                logger.info("NLI model loaded on %s", self._device)
            except Exception as exc:
                if self._device != "cuda":
                    raise
                record_gpu_failure("nli", exc)
                self._device = "cpu"
                self._nli_pipeline = _load_cached_pipeline(self.NLI_MODEL, "cpu")
                record_component_device("nli", "cpu")
                logger.warning("NLI model GPU load failed; falling back to CPU: %s", exc)
        except Exception as exc:
            logger.warning("NLI model load failed; NLI checks disabled: %s", exc)
            self._nli_available = False

    def check(self, hypothesis: str, premise: str) -> tuple[str, float]:
        """Run NLI inference on a premise-hypothesis pair."""
        from core.gpu import is_cuda_failure, record_component_device, record_gpu_failure

        if self._nli_pipeline is None and self._nli_available:
            self._load_nli_model()
        if not self._nli_available or self._nli_pipeline is None:
            return ("neutral", 0.0)
        try:
            with _inference_lock:
                result = self._nli_pipeline(
                    f"{premise} [SEP] {hypothesis}",
                    truncation=True,
                )
            label = str(result[0]["label"]).lower()
            score = float(result[0]["score"])
            return (label, score)
        except Exception as exc:
            if self._device == "cuda" and is_cuda_failure(exc):
                logger.error("GPU failure detected - falling back to CPU NLI", exc_info=True)
                record_gpu_failure("nli", exc)
                try:
                    self._device = "cpu"
                    self._nli_pipeline = _load_cached_pipeline(self.NLI_MODEL, "cpu")
                    record_component_device("nli", "cpu")
                    with _inference_lock:
                        result = self._nli_pipeline(
                            f"{premise} [SEP] {hypothesis}",
                            truncation=True,
                        )
                    label = str(result[0]["label"]).lower()
                    score = float(result[0]["score"])
                    return (label, score)
                except Exception as retry_exc:
                    logger.warning("NLI CPU fallback check failed: %s", retry_exc)
            logger.warning("NLI check failed: %s", exc)
            return ("neutral", 0.0)
