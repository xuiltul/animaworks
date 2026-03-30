"""Unit tests for completion_gate: shared helpers, Stop hook, ToolHandler, and Mode A gate."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from core.execution._completion_gate import (
    cleanup_gate_marker,
    completion_gate_applies_to_trigger,
    completion_gate_marker_path,
    gate_marker_exists,
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


def _reason_from(result: object) -> str | None:
    if isinstance(result, dict):
        return result.get("reason")
    return getattr(result, "reason", None)


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


# ── Shared helpers ────────────────────────────────────────────


class TestGateMarkerHelpers:
    def test_gate_marker_exists_false(self, anima_dir: Path):
        assert gate_marker_exists(anima_dir) is False

    def test_gate_marker_exists_true(self, anima_dir: Path):
        p = completion_gate_marker_path(anima_dir)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("", encoding="utf-8")
        assert gate_marker_exists(anima_dir) is True

    def test_cleanup_gate_marker(self, anima_dir: Path):
        p = completion_gate_marker_path(anima_dir)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("", encoding="utf-8")
        cleanup_gate_marker(anima_dir)
        assert not p.exists()

    def test_cleanup_gate_marker_no_file(self, anima_dir: Path):
        cleanup_gate_marker(anima_dir)


class TestTriggerApplicability:
    def test_none_trigger_applies(self):
        assert completion_gate_applies_to_trigger(None) is True

    def test_chat_applies(self):
        assert completion_gate_applies_to_trigger("chat") is True

    def test_task_applies(self):
        assert completion_gate_applies_to_trigger("task:exec") is True

    def test_cron_applies(self):
        assert completion_gate_applies_to_trigger("cron:0") is True

    def test_heartbeat_skipped(self):
        assert completion_gate_applies_to_trigger("heartbeat") is False

    def test_inbox_skipped(self):
        assert completion_gate_applies_to_trigger("inbox:dm") is False

    def test_inbox_mention_skipped(self):
        assert completion_gate_applies_to_trigger("inbox:mention") is False


# ── Stop hook (Mode S) — direct checklist injection ──────────


@pytest.mark.skipif(not _sdk_available(), reason="claude_agent_sdk not installed")
class TestStopHookCompletionGate:
    @pytest.mark.asyncio
    async def test_stop_hook_allows_when_stop_hook_active(self, anima_dir: Path):
        from core.execution._sdk_hooks import _build_stop_hook

        hook = _build_stop_hook(anima_dir, session_stats={"trigger": "chat"})
        result = await hook(_stop_input(stop_hook_active=True), None, {})
        assert result == {} or _decision_from(result) is None

    @pytest.mark.asyncio
    async def test_stop_hook_active_cleans_up_stale_marker(self, anima_dir: Path):
        """Stale marker from Mode A is cleaned up on stop_hook_active pass."""
        from core.execution._sdk_hooks import _build_stop_hook

        p = completion_gate_marker_path(anima_dir)
        p.write_text("", encoding="utf-8")
        hook = _build_stop_hook(anima_dir, session_stats={"trigger": "chat"})
        result = await hook(_stop_input(stop_hook_active=True), None, {})
        assert result == {} or _decision_from(result) is None
        assert not p.exists()

    @pytest.mark.asyncio
    async def test_stop_hook_blocks_with_checklist_for_chat(self, anima_dir: Path):
        from core.execution._sdk_hooks import _build_stop_hook

        hook = _build_stop_hook(anima_dir, session_stats={"trigger": "chat"})
        result = await hook(_stop_input(stop_hook_active=False), None, {})
        assert _decision_from(result) == "block"
        reason = _reason_from(result)
        assert reason is not None
        assert "完了前検証" in reason or "Pre-Completion" in reason

    @pytest.mark.asyncio
    async def test_stop_hook_blocks_with_checklist_for_task(self, anima_dir: Path):
        from core.execution._sdk_hooks import _build_stop_hook

        hook = _build_stop_hook(anima_dir, session_stats={"trigger": "task:exec"})
        result = await hook(_stop_input(stop_hook_active=False), None, {})
        assert _decision_from(result) == "block"
        reason = _reason_from(result)
        assert reason is not None
        assert "完了前検証" in reason or "Pre-Completion" in reason

    @pytest.mark.asyncio
    async def test_stop_hook_blocks_with_checklist_for_cron(self, anima_dir: Path):
        from core.execution._sdk_hooks import _build_stop_hook

        hook = _build_stop_hook(anima_dir, session_stats={"trigger": "cron:0"})
        result = await hook(_stop_input(stop_hook_active=False), None, {})
        assert _decision_from(result) == "block"
        reason = _reason_from(result)
        assert reason is not None
        assert "完了前検証" in reason or "Pre-Completion" in reason

    @pytest.mark.asyncio
    async def test_stop_hook_reason_contains_checklist_items(self, anima_dir: Path):
        """Verify the injected reason contains actionable checklist items."""
        from core.execution._sdk_hooks import _build_stop_hook

        hook = _build_stop_hook(anima_dir, session_stats={"trigger": "chat"})
        result = await hook(_stop_input(stop_hook_active=False), None, {})
        reason = _reason_from(result)
        assert "- [ ]" in reason

    @pytest.mark.asyncio
    async def test_stop_hook_no_tool_call_instruction(self, anima_dir: Path):
        """Verify the reason does NOT tell the model to call completion_gate tool."""
        from core.execution._sdk_hooks import _build_stop_hook

        hook = _build_stop_hook(anima_dir, session_stats={"trigger": "chat"})
        result = await hook(_stop_input(stop_hook_active=False), None, {})
        reason = _reason_from(result)
        assert "completion_gate" not in reason.lower()

    @pytest.mark.asyncio
    async def test_stop_hook_skips_heartbeat(self, anima_dir: Path):
        from core.execution._sdk_hooks import _build_stop_hook

        hook = _build_stop_hook(anima_dir, session_stats={"trigger": "heartbeat"})
        result = await hook(_stop_input(stop_hook_active=False), None, {})
        assert _decision_from(result) is None

    @pytest.mark.asyncio
    async def test_stop_hook_skips_inbox(self, anima_dir: Path):
        from core.execution._sdk_hooks import _build_stop_hook

        hook = _build_stop_hook(anima_dir, session_stats={"trigger": "inbox:dm"})
        result = await hook(_stop_input(stop_hook_active=False), None, {})
        assert _decision_from(result) is None

    @pytest.mark.asyncio
    async def test_stop_hook_does_not_depend_on_marker(self, anima_dir: Path):
        """Even if marker exists, stop hook still blocks (marker is for Mode A only)."""
        from core.execution._sdk_hooks import _build_stop_hook

        p = completion_gate_marker_path(anima_dir)
        p.write_text("", encoding="utf-8")
        hook = _build_stop_hook(anima_dir, session_stats={"trigger": "chat"})
        result = await hook(_stop_input(stop_hook_active=False), None, {})
        assert _decision_from(result) == "block"


# ── MCP tool list (Mode S) ──────────────────────────────────


class TestMCPToolListInclusion:
    def test_completion_gate_in_exposed_names(self):
        from core.mcp.server import _EXPOSED_TOOL_NAMES

        assert "completion_gate" in _EXPOSED_TOOL_NAMES


# ── TOOL_TRUST_LEVELS ─────────────────────────────────────────


class TestToolTrustLevels:
    def test_completion_gate_is_trusted(self):
        from core.execution._sanitize import TOOL_TRUST_LEVELS

        assert TOOL_TRUST_LEVELS.get("completion_gate") == "trusted"


# ── completion_gate tool (Mode A/B) ────────────────────────────


class TestCompletionGateTool:
    def test_completion_gate_writes_marker(self, anima_dir: Path, memory: MagicMock):
        h = ToolHandler(anima_dir=anima_dir, memory=memory, tool_registry=[])
        h.handle("completion_gate", {})
        assert gate_marker_exists(anima_dir)

    def test_completion_gate_returns_checklist(self, anima_dir: Path, memory: MagicMock):
        h = ToolHandler(anima_dir=anima_dir, memory=memory, tool_registry=[])
        raw = h.handle("completion_gate", {})
        assert isinstance(raw, str)
        assert len(raw) > 10
        assert "Pre-Completion" in raw or "完了前検証" in raw


# ── Mode A loop gate ────────────────────────────────────────


class TestModeALoopGate:
    """Verify that completion_gate helpers are importable from the shared module
    and that the unified tool list includes completion_gate."""

    def test_completion_gate_in_unified_tool_list(self):
        from core.tooling.schemas import build_unified_tool_list

        tools = build_unified_tool_list()
        names = {t["name"] for t in tools}
        assert "completion_gate" in names

    def test_completion_gate_not_in_consolidation_tools(self):
        from core.tooling.schemas import build_unified_tool_list

        tools = build_unified_tool_list(trigger="consolidation:daily")
        names = {t["name"] for t in tools}
        assert "completion_gate" not in names

    def test_shared_helpers_importable_from_litellm_loop(self):
        """Ensure litellm_loop.py can use the shared helpers."""
        from core.execution._completion_gate import (  # noqa: F401
            cleanup_gate_marker,
            completion_gate_applies_to_trigger,
            gate_marker_exists,
        )

    def test_gate_marker_roundtrip(self, anima_dir: Path):
        assert not gate_marker_exists(anima_dir)
        marker = completion_gate_marker_path(anima_dir)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("", encoding="utf-8")
        assert gate_marker_exists(anima_dir)
        cleanup_gate_marker(anima_dir)
        assert not gate_marker_exists(anima_dir)


# ── non_s guide ─────────────────────────────────────────────


class TestNonSGuide:
    def test_non_s_guide_mentions_completion_gate(self):
        from core.tooling.prompt_db import get_default_guide

        ja = get_default_guide("non_s", locale="ja")
        assert "completion_gate" in ja

        en = get_default_guide("non_s", locale="en")
        assert "completion_gate" in en
