"""Inference requests must not hold up task operations or each other's priorities."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import threading
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from server.routes.internal import create_internal_router

_TIMEOUT = 20.0


@pytest.fixture
def app() -> FastAPI:
    application = FastAPI()
    application.state.shared_dir = Path("/tmp/test-shared")
    application.include_router(create_internal_router(), prefix="/api")
    return application


class _Blocked:
    """Stands in for model work: records its thread and waits to be released."""

    def __init__(self) -> None:
        self.release = threading.Event()
        self.lock = threading.Lock()
        self.calls: list[tuple[str, str]] = []

    def __call__(self, kind: str) -> None:
        with self.lock:
            self.calls.append((kind, threading.current_thread().name))
        assert self.release.wait(timeout=_TIMEOUT)

    def started(self, kind: str) -> int:
        with self.lock:
            return sum(1 for seen, _ in self.calls if seen == kind)


@pytest.fixture
def blocked() -> Iterator[_Blocked]:
    state = _Blocked()

    def encode(texts: list[str], *, purpose: str = "document", priority: str = "interactive") -> list[list[float]]:
        state(priority)
        return [[0.0] for _ in texts]

    def score(query: str, documents: list[str]) -> list[float]:
        state("rerank")
        return [0.0 for _ in documents]

    reranker = MagicMock()
    reranker.score_sync.side_effect = score
    with (
        patch("core.memory.rag.embedding.thread_safe_encode", side_effect=encode),
        patch("core.memory.retrieval.reranker.get_reranker", return_value=reranker),
    ):
        try:
            yield state
        finally:
            state.release.set()


async def _wait_for(condition, *, timeout: float = _TIMEOUT) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not condition():
        assert asyncio.get_running_loop().time() < deadline, "condition was not reached"
        await asyncio.sleep(0.01)


async def _post_embed(client: AsyncClient, priority: str):
    return await client.post("/api/internal/embed", json={"texts": ["text"], "priority": priority})


async def _post_rerank(client: AsyncClient):
    return await client.post("/api/internal/rerank", json={"query": "q", "documents": ["doc"]})


async def test_task_operations_run_while_inference_is_saturated(app: FastAPI, blocked: _Blocked) -> None:
    """Every native-ops worker was once taken by embedding, delaying task updates by minutes."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test", timeout=_TIMEOUT) as client:
        requests = [
            *(asyncio.create_task(_post_embed(client, "interactive")) for _ in range(12)),
            *(asyncio.create_task(_post_embed(client, "bulk")) for _ in range(6)),
            *(asyncio.create_task(_post_rerank(client)) for _ in range(6)),
        ]
        await _wait_for(lambda: blocked.started("interactive") + blocked.started("rerank") >= 8)

        with patch("core.tasks.board.board_actions.run_board_action", return_value={"done": True}) as action:
            response = await client.post(
                "/api/internal/task-board-action",
                json={"actor": "human", "action": "note", "task_id": "t", "text": "progress"},
            )
        assert response.status_code == 200, response.text
        action.assert_called_once()
        assert not blocked.release.is_set()

        blocked.release.set()
        results = await asyncio.gather(*requests)
    assert all(result.status_code == 200 for result in results)


async def test_inference_never_runs_on_the_task_executor(app: FastAPI, blocked: _Blocked) -> None:
    blocked.release.set()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test", timeout=_TIMEOUT) as client:
        await _post_embed(client, "interactive")
        await _post_embed(client, "bulk")
        await _post_rerank(client)
    threads = dict(blocked.calls)
    assert threads["interactive"].startswith("inference-interactive")
    assert threads["rerank"].startswith("inference-interactive")
    assert threads["bulk"].startswith("inference-bulk")


async def test_interactive_requests_are_not_queued_behind_bulk(app: FastAPI, blocked: _Blocked) -> None:
    """A free thread is what lets an interactive request reach the priority logic."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test", timeout=_TIMEOUT) as client:
        bulk = [asyncio.create_task(_post_embed(client, "bulk")) for _ in range(8)]
        await _wait_for(lambda: blocked.started("bulk") >= 1)

        interactive = [asyncio.create_task(_post_embed(client, "interactive")) for _ in range(4)]
        rerank = asyncio.create_task(_post_rerank(client))
        await _wait_for(lambda: blocked.started("interactive") == 4 and blocked.started("rerank") == 1)
        assert blocked.started("bulk") <= 2

        blocked.release.set()
        results = await asyncio.gather(*bulk, *interactive, rerank)
    assert all(result.status_code == 200 for result in results)
