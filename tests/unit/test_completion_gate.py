"""Unit tests for completion_gate Stop hook IPC and ToolHandler."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from core.execution._sdk_hooks import (
    _build_stop_hook,
    _cleanup_gate_marker,
    _completion_gate_marker_path,
    _gate_marker_exists,
)
from core.tooling.handler import ToolHandler


def _sdk_available() -> bool:
    try:
        import claude_agent_sdk  # noqa: F401

        return True
    except ImportError:
        return False


def _stop_input(*, stop_hook_active: bool = False) -> dict:
    return {
        "session_id": "sess",
        "transcript_path": "/tmp/t.jsonl",
        "cwd": "/",
        "permission_mode": "default",
        "hook_event_name": "Stop",
        "stop_hook_active": stop_hook_active,
    }


def _decision_from(result: object) -> str | None:
    if isinstance(result, dict):
        return result.get("decision")
    return getattr(result, "decision", None)


@pytest.fixture
def anima_dir(tmp_path: Path) -> Path:
    d = tmp_path / "animas" / "gate-test"
    d.mkdir(parents=True)
    (d / "run").mkdir(parents=True, exist_ok=True)
    (d / "permissions.json").write_text(
        '{"version": 1, "file_roots": ["/"], "commands": {"allow_all": true, "allow": [], "deny": []}, '
        '"external_tools": {"allow_all": true}, "tool_creation": {"personal": true, "shared": false}}',
        encoding="utf-8",
    )
    return d


@pytest.fixture
def memory(anima_dir: Path) -> MagicMock:
    m = MagicMock()
    m.read_permissions.return_value = ""
    m.search_memory_text.return_value = []
    return m


# ── Marker helpers ────────────────────────────────────────────


class TestGateMarkerHelpers:
    def test_gate_marker_exists_false(self, anima_dir: Path):
        assert _gate_marker_exists(anima_dir) is False

    def test_gate_marker_exists_true(self, anima_dir: Path):
        p = _completion_gate_marker_path(anima_dir)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("", encoding="utf-8")
        assert _gate_marker_exists(anima_dir) is True

    def test_cleanup_gate_marker(self, anima_dir: Path):
        p = _completion_gate_marker_path(anima_dir)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("", encoding="utf-8")
        _cleanup_gate_marker(anima_dir)
        assert not p.exists()

    def test_cleanup_gate_marker_no_file(self, anima_dir: Path):
        _cleanup_gate_marker(anima_dir)


# ── Stop hook ─────────────────────────────────────────────────


@pytest.mark.skipif(not _sdk_available(), reason="claude_agent_sdk not installed")
class TestStopHookCompletionGate:
    @pytest.mark.asyncio
    async def test_stop_hook_allows_when_stop_hook_active(self, anima_dir: Path):
        p = _completion_gate_marker_path(anima_dir)
        p.write_text("", encoding="utf-8")
        hook = _build_stop_hook(anima_dir, session_stats={"trigger": "chat"})
        result = await hook(_stop_input(stop_hook_active=True), None, {})
        assert result == {} or _decision_from(result) is None
        assert not p.exists()

    @pytest.mark.asyncio
    async def test_stop_hook_blocks_when_no_gate_called(self, anima_dir: Path):
        hook = _build_stop_hook(anima_dir, session_stats={"trigger": "chat"})
        result = await hook(_stop_input(stop_hook_active=False), None, {})
        assert _decision_from(result) == "block"

    @pytest.mark.asyncio
    async def test_stop_hook_allows_when_gate_called(self, anima_dir: Path):
        p = _completion_gate_marker_path(anima_dir)
        p.write_text("", encoding="utf-8")
        hook = _build_stop_hook(anima_dir, session_stats={"trigger": "chat"})
        result = await hook(_stop_input(stop_hook_active=False), None, {})
        assert _decision_from(result) is None
        assert not p.exists()

    @pytest.mark.asyncio
    async def test_stop_hook_skips_heartbeat(self, anima_dir: Path):
        hook = _build_stop_hook(anima_dir, session_stats={"trigger": "heartbeat"})
        result = await hook(_stop_input(stop_hook_active=False), None, {})
        assert _decision_from(result) is None

    @pytest.mark.asyncio
    async def test_stop_hook_skips_inbox(self, anima_dir: Path):
        hook = _build_stop_hook(anima_dir, session_stats={"trigger": "inbox:dm"})
        result = await hook(_stop_input(stop_hook_active=False), None, {})
        assert _decision_from(result) is None

    @pytest.mark.asyncio
    async def test_stop_hook_blocks_for_chat(self, anima_dir: Path):
        hook = _build_stop_hook(anima_dir, session_stats={"trigger": "chat"})
        result = await hook(_stop_input(stop_hook_active=False), None, {})
        assert _decision_from(result) == "block"

    @pytest.mark.asyncio
    async def test_stop_hook_blocks_for_task(self, anima_dir: Path):
        hook = _build_stop_hook(anima_dir, session_stats={"trigger": "task:exec"})
        result = await hook(_stop_input(stop_hook_active=False), None, {})
        assert _decision_from(result) == "block"

    @pytest.mark.asyncio
    async def test_stop_hook_blocks_for_cron(self, anima_dir: Path):
        hook = _build_stop_hook(anima_dir, session_stats={"trigger": "cron:0"})
        result = await hook(_stop_input(stop_hook_active=False), None, {})
        assert _decision_from(result) == "block"


# ── completion_gate tool ────────────────────────────────────────


class TestCompletionGateTool:
    def test_completion_gate_writes_marker(self, anima_dir: Path, memory: MagicMock):
        h = ToolHandler(anima_dir=anima_dir, memory=memory, tool_registry=[])
        h.handle("completion_gate", {})
        assert _gate_marker_exists(anima_dir)

    def test_completion_gate_returns_checklist(self, anima_dir: Path, memory: MagicMock):
        h = ToolHandler(anima_dir=anima_dir, memory=memory, tool_registry=[])
        raw = h.handle("completion_gate", {})
        assert isinstance(raw, str)
        assert len(raw) > 10
        assert "Pre-Completion" in raw or "完了前検証" in raw
