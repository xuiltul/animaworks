"""E2E-specific fixtures for constructing AgentCore and DigitalAnima instances."""

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from core.agent import AgentCore
from core.anima import DigitalAnima
from core.memory import MemoryManager
from core.messenger import Messenger


class _DeterministicEmbeddingModel:
    """Small local embedding stand-in for E2E tests.

    The E2E suite verifies AnimaWorks behavior, not Hugging Face model
    availability.  Keeping embeddings deterministic avoids CI failures when
    hosted runners hit external model rate limits.
    """

    dimension = 384

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimension

    def encode(
        self,
        texts: str | list[str],
        *,
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False,
    ):
        if isinstance(texts, str):
            vector = self._embed(texts)
            if convert_to_numpy:
                return np.array(vector, dtype=float)
            return vector

        vectors = [self._embed(text) for text in texts]
        if convert_to_numpy:
            return np.array(vectors, dtype=float)
        return vectors

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in self._tokens(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = digest[0] % self.dimension
            sign = 1.0 if digest[1] % 2 == 0 else -1.0
            weight = 1.0 + digest[2] / 255.0
            vector[index] += sign * weight
        norm = sum(value * value for value in vector) ** 0.5
        if not norm:
            return vector
        return [value / norm for value in vector]

    def _tokens(self, text: str) -> list[str]:
        normalized = text.lower()
        word_tokens = re.findall(r"[\w]+", normalized)
        compact = re.sub(r"\s+", "", normalized)
        char_tokens = [compact[i : i + 3] for i in range(max(0, len(compact) - 2))]
        return word_tokens + char_tokens or [normalized]


class _DeterministicReranker:
    """Local cross-encoder stand-in for E2E retrieval tests."""

    _model_name = "e2e-deterministic-reranker"

    def rerank_sync(
        self,
        query: str,
        items: list[dict],
        *,
        text_field: str | Any = "content",
        top_k: int = 10,
        min_candidates: int = 2,
    ) -> list[dict]:
        return self._rank(items, top_k)

    async def rerank(
        self,
        query: str,
        items: list[dict],
        *,
        text_field: str | Any = "fact",
        top_k: int = 10,
    ) -> list[dict]:
        return self._rank(items, top_k)

    def _rank(self, items: list[dict], top_k: int) -> list[dict]:
        ranked: list[dict] = []
        for offset, item in enumerate(items[:top_k]):
            row = dict(item)
            score = 1.0 - (offset * 0.01)
            row["ce_score"] = score
            row["score"] = score
            row["search_method"] = "cross_encoder"
            ranked.append(row)
        return ranked


@pytest.fixture(autouse=True)
def _bypass_completion_gate():
    """Disable completion_gate enforcement for all e2e tests.

    E2E tests mock LLM responses with a fixed number of side effects.
    completion_gate injects an extra retry loop iteration when the gate
    is not called, causing StopIteration on the mock.  Since e2e tests
    test specific agent behaviors (not completion_gate itself), bypass
    enforcement globally here.  Unit-level gate tests live in tests/unit/.
    """
    with patch(
        "core.execution.litellm_loop.completion_gate_applies_to_trigger",
        return_value=False,
    ):
        yield


@pytest.fixture(autouse=True)
def _deterministic_embedding_model(monkeypatch: pytest.MonkeyPatch):
    """Prevent E2E tests from reaching external embedding model hosts."""

    model = _DeterministicEmbeddingModel()
    monkeypatch.setattr("core.memory.rag.singleton.get_embedding_model", lambda model_name=None: model)


@pytest.fixture(autouse=True)
def _deterministic_reranker(monkeypatch: pytest.MonkeyPatch):
    """Prevent E2E tests from downloading external cross-encoder models."""

    reranker = _DeterministicReranker()
    monkeypatch.setattr("core.memory.retrieval.reranker.get_reranker", lambda model_name=None: reranker)
    monkeypatch.setattr("core.memory.retrieval.pipeline.get_reranker", lambda model_name=None: reranker)
    monkeypatch.setattr("core.memory.graph.reranker.get_reranker", lambda model_name=None: reranker)


@pytest.fixture
def make_agent_core(data_dir: Path, make_anima):
    """Factory to create an AgentCore instance with isolated filesystem.

    Bypasses DigitalAnima to test AgentCore directly.
    """

    def _make(name: str = "test-agent", **kwargs: Any) -> AgentCore:
        anima_dir = make_anima(name, **kwargs)
        memory = MemoryManager(anima_dir)
        model_config = memory.read_model_config()
        messenger = Messenger(data_dir / "shared", name)
        return AgentCore(anima_dir, memory, model_config, messenger)

    return _make


@pytest.fixture
def make_digital_anima(data_dir: Path, make_anima):
    """Factory to create a DigitalAnima instance with isolated filesystem."""

    def _make(name: str = "test-anima", **kwargs: Any) -> DigitalAnima:
        anima_dir = make_anima(name, **kwargs)
        shared_dir = data_dir / "shared"
        return DigitalAnima(anima_dir, shared_dir)

    return _make
