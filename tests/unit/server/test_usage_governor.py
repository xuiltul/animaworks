from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from server.usage_governor import DEFAULT_POLICY, UsageGovernor


def _write_status(animas_dir, name: str, credential: str) -> None:
    anima_dir = animas_dir / name
    anima_dir.mkdir(parents=True, exist_ok=True)
    (anima_dir / "status.json").write_text(
        json.dumps({"credential": credential}),
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_tick_keeps_suspended_anima_when_usage_fetch_fails(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    animas_dir = tmp_path / "animas"
    _write_status(animas_dir, "alice", "anthropic")

    supervisor = SimpleNamespace(
        processes={},
        start_anima=AsyncMock(),
        stop_anima=AsyncMock(),
    )
    app = SimpleNamespace(state=SimpleNamespace(supervisor=supervisor))
    governor = UsageGovernor(app, data_dir, animas_dir)
    governor.state.suspended_animas = ["alice"]
    governor.state.since = "2026-03-25T18:00:00+0900"

    monkeypatch.setattr(
        "server.routes.usage_routes._fetch_claude_usage",
        lambda **kwargs: {"error": "unauthorized", "message": "expired"},
    )
    monkeypatch.setattr(
        "server.routes.usage_routes._fetch_openai_usage",
        lambda **kwargs: {"provider": "openai"},
    )

    await governor._tick(DEFAULT_POLICY)

    assert governor.state.suspended_animas == ["alice"]
    assert "claude usage unavailable" in governor.state.reason
    supervisor.start_anima.assert_not_called()
    supervisor.stop_anima.assert_not_called()


@pytest.mark.asyncio
async def test_tick_only_keeps_suspended_animas_for_provider_with_fetch_failure(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    animas_dir = tmp_path / "animas"
    _write_status(animas_dir, "alice", "anthropic")
    _write_status(animas_dir, "bob", "openai")

    supervisor = SimpleNamespace(
        processes={},
        start_anima=AsyncMock(),
        stop_anima=AsyncMock(),
    )
    app = SimpleNamespace(state=SimpleNamespace(supervisor=supervisor))
    governor = UsageGovernor(app, data_dir, animas_dir)
    governor.state.suspended_animas = ["alice", "bob"]

    monkeypatch.setattr(
        "server.routes.usage_routes._fetch_claude_usage",
        lambda **kwargs: {"error": "rate_limited", "message": "retry shortly"},
    )
    monkeypatch.setattr(
        "server.routes.usage_routes._fetch_openai_usage",
        lambda **kwargs: {
            "provider": "openai",
            "5h": {
                "remaining": 80,
                "resets_at": 4102444800,
                "window_seconds": 18000,
            },
            "Week": {
                "remaining": 85,
                "resets_at": 4102444800,
                "window_seconds": 604800,
            },
        },
    )

    await governor._tick(DEFAULT_POLICY)

    assert governor.state.suspended_animas == ["alice"]
    supervisor.start_anima.assert_awaited_once_with("bob")
    supervisor.stop_anima.assert_not_called()
