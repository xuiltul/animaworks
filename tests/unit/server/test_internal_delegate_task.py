from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for POST /api/internal/delegate-task."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient


def _make_test_app():
    from fastapi import FastAPI

    from server.routes.internal import create_internal_router

    app = FastAPI()
    app.state.ws_manager = MagicMock()
    app.state.ws_manager.broadcast = AsyncMock()
    app.include_router(create_internal_router(), prefix="/api")
    return app


def _setup_animas(
    tmp_path: Path,
    *,
    delegator: str = "rin",
    target: str = "natsume",
    delegator_company: str | None = None,
    target_company: str | None = None,
) -> Path:
    animas = tmp_path / "animas"
    for name, company in ((delegator, delegator_company), (target, target_company)):
        d = animas / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "state").mkdir(exist_ok=True)
        status: dict = {"enabled": True}
        if company is not None:
            status["company"] = company
        (d / "status.json").write_text(
            json.dumps(status, indent=2) + "\n",
            encoding="utf-8",
        )
    return animas


def _base_payload(**overrides) -> dict:
    payload = {
        "delegator": "rin",
        "target": "natsume",
        "instruction": "fix the merge conflict on PR #3553",
        "summary": "PR #3553 conflict",
        "deadline": "2h",
        "sub_task_id": "aabbccddeeff",
        "tracking_task_id": "112233445566",
        "workspace": "",
        "persist_sub": True,
        "persist_tracking": True,
        "persist_pending": True,
    }
    payload.update(overrides)
    return payload


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class TestInternalDelegateTask:
    @pytest.mark.anyio
    async def test_success_writes_queues_and_pending(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        animas = _setup_animas(tmp_path)
        monkeypatch.setattr("core.paths.get_animas_dir", lambda: animas)

        app = _make_test_app()
        transport = ASGITransport(app=app)
        with patch(
            "core.tooling.handler_delegation._record_taskboard_delegation"
        ) as mock_tb:
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/api/internal/delegate-task",
                    json=_base_payload(),
                )

        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["sub_task_id"] == "aabbccddeeff"
        assert body["tracking_task_id"] == "112233445566"

        sub_queue = animas / "natsume" / "state" / "task_queue.jsonl"
        assert sub_queue.exists()
        sub_entry = json.loads(sub_queue.read_text(encoding="utf-8").strip().split("\n")[-1])
        assert sub_entry["task_id"] == "aabbccddeeff"
        assert sub_entry["status"] == "pending"
        assert sub_entry["assignee"] == "natsume"
        assert sub_entry["relay_chain"] == ["rin"]

        own_queue = animas / "rin" / "state" / "task_queue.jsonl"
        assert own_queue.exists()
        own_entry = json.loads(own_queue.read_text(encoding="utf-8").strip().split("\n")[-1])
        assert own_entry["task_id"] == "112233445566"
        assert own_entry["status"] == "delegated"
        assert own_entry["meta"]["delegated_to"] == "natsume"
        assert own_entry["meta"]["delegated_task_id"] == "aabbccddeeff"

        pending = animas / "natsume" / "state" / "pending" / "aabbccddeeff.json"
        assert pending.exists()
        pending_data = json.loads(pending.read_text(encoding="utf-8"))
        assert pending_data["task_type"] == "llm"
        assert pending_data["task_id"] == "aabbccddeeff"
        assert pending_data["submitted_by"] == "rin"
        assert pending_data["reply_to"] == "rin"
        assert pending_data["source"] == "delegation"
        assert pending_data["description"] == "fix the merge conflict on PR #3553"

        mock_tb.assert_called_once()
        kwargs = mock_tb.call_args.kwargs
        assert kwargs["delegated_to"] == "natsume"
        assert kwargs["delegated_task_id"] == "aabbccddeeff"
        assert kwargs["delegator"] == "rin"
        assert kwargs["tracking_task_id"] == "112233445566"

    @pytest.mark.anyio
    async def test_persist_sub_false_skips_subordinate_queue(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        animas = _setup_animas(tmp_path)
        monkeypatch.setattr("core.paths.get_animas_dir", lambda: animas)

        app = _make_test_app()
        transport = ASGITransport(app=app)
        with patch("core.tooling.handler_delegation._record_taskboard_delegation"):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/api/internal/delegate-task",
                    json=_base_payload(persist_sub=False),
                )

        assert resp.status_code == 200
        sub_queue = animas / "natsume" / "state" / "task_queue.jsonl"
        assert not sub_queue.exists()
        own_queue = animas / "rin" / "state" / "task_queue.jsonl"
        assert own_queue.exists()
        pending = animas / "natsume" / "state" / "pending" / "aabbccddeeff.json"
        assert pending.exists()

    @pytest.mark.anyio
    async def test_invalid_anima_name_returns_400(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        animas = _setup_animas(tmp_path)
        monkeypatch.setattr("core.paths.get_animas_dir", lambda: animas)

        app = _make_test_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/internal/delegate-task",
                json=_base_payload(target="../evil"),
            )

        assert resp.status_code == 400
        assert "Invalid anima name" in resp.json()["detail"]

    @pytest.mark.anyio
    async def test_missing_target_dir_returns_404(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        animas = _setup_animas(tmp_path)
        # remove target
        import shutil

        shutil.rmtree(animas / "natsume")
        monkeypatch.setattr("core.paths.get_animas_dir", lambda: animas)

        app = _make_test_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/internal/delegate-task",
                json=_base_payload(),
            )

        assert resp.status_code == 404
        assert "natsume" in resp.json()["detail"]

    @pytest.mark.anyio
    async def test_invalid_deadline_returns_422(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        animas = _setup_animas(tmp_path)
        monkeypatch.setattr("core.paths.get_animas_dir", lambda: animas)

        app = _make_test_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/internal/delegate-task",
                json=_base_payload(deadline="not-a-deadline"),
            )

        assert resp.status_code == 422
        assert "deadline" in resp.json()["detail"].lower() or "Invalid" in resp.json()["detail"]

    @pytest.mark.anyio
    async def test_cross_company_returns_403(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        animas = _setup_animas(
            tmp_path,
            delegator_company="alpha",
            target_company="beta",
        )
        monkeypatch.setattr("core.paths.get_animas_dir", lambda: animas)

        app = _make_test_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/internal/delegate-task",
                json=_base_payload(),
            )

        assert resp.status_code == 403
        assert "Cross-company" in resp.json()["detail"]
