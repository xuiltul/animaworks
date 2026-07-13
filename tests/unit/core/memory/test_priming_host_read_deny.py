from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.memory.backend.base import RetrievedMemory
from core.memory.priming.channel_a import channel_a_sender_profile
from core.memory.priming.channel_b import channel_b_recent_activity, read_shared_channels
from core.memory.priming.channel_e import channel_e_pending_tasks
from core.memory.priming.channel_g import collect_graph_context
from core.prompt.org_context import _build_org_context, _discover_other_animas


def _write_permissions(anima_dir: Path, denied: list[Path]) -> None:
    anima_dir.mkdir(parents=True, exist_ok=True)
    (anima_dir / "permissions.json").write_text(
        json.dumps({"version": 1, "file_roots_denied": [str(path) for path in denied]}),
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_channel_a_omits_denied_sender_profile_symlink(tmp_path: Path) -> None:
    anima_dir = tmp_path / "animas" / "agent"
    shared = tmp_path / "shared"
    denied = tmp_path / "private"
    denied.mkdir()
    secret = denied / "profile.md"
    secret.write_text("DENIED PROFILE CANARY", encoding="utf-8")
    profile_dir = shared / "users" / "alice"
    profile_dir.mkdir(parents=True)
    (profile_dir / "index.md").symlink_to(secret)
    _write_permissions(anima_dir, [denied])

    with patch("core.paths.get_shared_dir", return_value=shared):
        result = await channel_a_sender_profile(anima_dir, "alice")

    assert result == ""


def test_channel_b_omits_denied_shared_channel_symlink(tmp_path: Path) -> None:
    anima_dir = tmp_path / "animas" / "agent"
    shared = tmp_path / "shared"
    channels = shared / "channels"
    channels.mkdir(parents=True)
    denied = tmp_path / "private"
    denied.mkdir()
    secret = denied / "general.jsonl"
    secret.write_text('{"ts":"2026-07-13T12:00:00+09:00","text":"DENIED CHANNEL CANARY"}\n')
    (channels / "general.jsonl").symlink_to(secret)
    _write_permissions(anima_dir, [denied])

    assert read_shared_channels(anima_dir, shared) == []


@pytest.mark.asyncio
async def test_channel_b_does_not_open_denied_activity_tree(tmp_path: Path) -> None:
    anima_dir = tmp_path / "animas" / "agent"
    activity_dir = anima_dir / "activity_log"
    activity_dir.mkdir(parents=True)
    _write_permissions(anima_dir, [activity_dir])

    with patch("core.memory.activity.ActivityLogger.recent", side_effect=AssertionError("must not read")):
        result = await channel_b_recent_activity(anima_dir, None, "human", [])

    assert result == ""


@pytest.mark.asyncio
async def test_channel_e_omits_denied_task_result(tmp_path: Path) -> None:
    anima_dir = tmp_path / "animas" / "agent"
    result_dir = anima_dir / "state" / "task_results"
    result_dir.mkdir(parents=True)
    denied_result = result_dir / "secret.md"
    denied_result.write_text("DENIED TASK RESULT CANARY", encoding="utf-8")
    _write_permissions(anima_dir, [denied_result])

    result = await channel_e_pending_tasks(anima_dir, None)

    assert "DENIED TASK RESULT CANARY" not in result
    assert "secret.md" not in result


@pytest.mark.asyncio
async def test_channel_g_filters_denied_and_unknown_graph_sources(tmp_path: Path) -> None:
    anima_dir = tmp_path / "animas" / "agent"
    denied = anima_dir / "facts" / "private"
    denied.mkdir(parents=True)
    _write_permissions(anima_dir, [denied])
    backend = MagicMock()
    backend.get_community_context = AsyncMock(
        return_value=[RetrievedMemory("UNKNOWN GRAPH CANARY", 1.0, "community:opaque")]
    )
    backend.get_recent_facts = AsyncMock(
        return_value=[
            RetrievedMemory("DENIED GRAPH CANARY", 1.0, "facts/private/secret.json"),
            RetrievedMemory("VISIBLE GRAPH FACT", 0.8, "facts/public.json"),
        ]
    )

    result = await collect_graph_context(backend, "test", anima_dir=anima_dir)

    assert "VISIBLE GRAPH FACT" in result
    assert "DENIED GRAPH CANARY" not in result
    assert "UNKNOWN GRAPH CANARY" not in result


def test_org_context_omits_denied_anima_tree_and_status(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    current = data_dir / "animas" / "agent"
    peer = data_dir / "animas" / "secret-peer"
    for path in (current, peer):
        path.mkdir(parents=True)
        (path / "identity.md").write_text("identity", encoding="utf-8")
    (current / "status.json").write_text('{"supervisor": null, "role": "main"}', encoding="utf-8")
    (peer / "status.json").write_text(
        '{"supervisor": "agent", "speciality": "DENIED STATUS CANARY"}',
        encoding="utf-8",
    )
    _write_permissions(current, [peer])

    config = SimpleNamespace(animas={})
    with (
        patch("core.prompt.org_context.get_data_dir", return_value=data_dir),
        patch("core.config.load_config", return_value=config),
        patch(
            "core.prompt.org_context.load_prompt",
            side_effect=lambda _name, **kwargs: str(kwargs.get("tree_text", kwargs)),
        ),
        patch("core.prompt.org_context._load_fallback_strings", return_value={}),
    ):
        assert _discover_other_animas(current) == []
        result = _build_org_context("agent", ["secret-peer"])

    assert "secret-peer" not in result
    assert "DENIED STATUS CANARY" not in result
