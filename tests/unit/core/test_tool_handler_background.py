"""Tests for background execution path in core/tooling/handler.py — ToolHandler."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from core.background import BackgroundTaskManager
from core.tooling.handler import ToolHandler


# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def anima_dir(tmp_path: Path) -> Path:
    d = tmp_path / "animas" / "test-anima"
    d.mkdir(parents=True)
    (d / "permissions.md").write_text("", encoding="utf-8")
    return d


@pytest.fixture
def memory() -> MagicMock:
    m = MagicMock()
    m.read_permissions.return_value = ""
    m.search_memory_text.return_value = []
    return m


@pytest.fixture
def external_dispatcher() -> MagicMock:
    return MagicMock()


@pytest.fixture
def background_manager(anima_dir: Path) -> BackgroundTaskManager:
    return BackgroundTaskManager(anima_dir, anima_name="test-anima")


# ── Background execution in handle() ─────────────────────────
# Note: Tests that trigger submit() on BackgroundTaskManager must be async
# because submit() calls asyncio.create_task() which requires a running loop.


class TestHandleBackgroundExecution:
    async def test_handle_eligible_tool_returns_background_response(
        self,
        anima_dir: Path,
        memory: MagicMock,
        background_manager: BackgroundTaskManager,
    ):
        """When BackgroundTaskManager is set and tool is eligible,
        handle() returns JSON with status='background'."""
        handler = ToolHandler(
            anima_dir=anima_dir,
            memory=memory,
            messenger=None,
            tool_registry=[],
            background_manager=background_manager,
        )
        # Replace external dispatcher so we can verify dispatch is not called directly
        handler._external = MagicMock()
        handler._external.dispatch.return_value = "direct result"

        result = handler.handle("generate_character_assets", {"prompt": "cat"})
        parsed = json.loads(result)

        assert parsed["status"] == "background"
        assert "task_id" in parsed
        assert isinstance(parsed["task_id"], str)
        assert len(parsed["task_id"]) == 12
        assert "message" in parsed

        # The external dispatcher's dispatch should NOT have been called directly
        # (it will be called by the background task manager in the background)
        handler._external.dispatch.assert_not_called()

    def test_handle_non_eligible_tool_proceeds_normally(
        self,
        anima_dir: Path,
        memory: MagicMock,
        background_manager: BackgroundTaskManager,
    ):
        """Non-eligible tools still go through normal dispatch."""
        handler = ToolHandler(
            anima_dir=anima_dir,
            memory=memory,
            messenger=None,
            tool_registry=[],
            background_manager=background_manager,
        )
        handler._external = MagicMock()
        handler._external.dispatch.return_value = "normal result"

        # "web_search" is NOT in the default eligible tools
        result = handler.handle("web_search", {"query": "hello"})
        assert result == "normal result"

        # External dispatch should have been called
        handler._external.dispatch.assert_called_once()
        call_args = handler._external.dispatch.call_args
        assert call_args[0][0] == "web_search"

    def test_handle_no_background_manager(
        self,
        anima_dir: Path,
        memory: MagicMock,
    ):
        """When background_manager is None, all tools go through normal path."""
        handler = ToolHandler(
            anima_dir=anima_dir,
            memory=memory,
            messenger=None,
            tool_registry=[],
            background_manager=None,
        )
        handler._external = MagicMock()
        handler._external.dispatch.return_value = "sync result"

        # Even an eligible tool should go through normal dispatch
        result = handler.handle("generate_character_assets", {"prompt": "cat"})
        assert result == "sync result"
        handler._external.dispatch.assert_called_once()

    async def test_handle_eligible_tool_passes_anima_dir_in_args(
        self,
        anima_dir: Path,
        memory: MagicMock,
        background_manager: BackgroundTaskManager,
    ):
        """Background submission includes anima_dir in the tool args."""
        handler = ToolHandler(
            anima_dir=anima_dir,
            memory=memory,
            messenger=None,
            tool_registry=[],
            background_manager=background_manager,
        )
        handler._external = MagicMock()

        handler.handle("generate_character_assets", {"prompt": "cat"})

        # Verify the task was submitted with anima_dir in args
        task = background_manager.list_tasks()[0]
        assert "anima_dir" in task.tool_args
        assert task.tool_args["anima_dir"] == str(anima_dir)
        assert task.tool_args["prompt"] == "cat"

    async def test_handle_multiple_eligible_tools(
        self,
        anima_dir: Path,
        memory: MagicMock,
        background_manager: BackgroundTaskManager,
    ):
        """Multiple eligible tool calls each get their own background task."""
        handler = ToolHandler(
            anima_dir=anima_dir,
            memory=memory,
            messenger=None,
            tool_registry=[],
            background_manager=background_manager,
        )
        handler._external = MagicMock()

        result1 = handler.handle("generate_character_assets", {"prompt": "cat"})
        result2 = handler.handle("local_llm", {"query": "hello"})
        result3 = handler.handle("run_command", {"cmd": "ls"})

        parsed1 = json.loads(result1)
        parsed2 = json.loads(result2)
        parsed3 = json.loads(result3)

        # All three should be background tasks
        assert parsed1["status"] == "background"
        assert parsed2["status"] == "background"
        assert parsed3["status"] == "background"

        # All task_ids should be unique
        ids = {parsed1["task_id"], parsed2["task_id"], parsed3["task_id"]}
        assert len(ids) == 3

    def test_handle_memory_tools_not_affected_by_background_manager(
        self,
        anima_dir: Path,
        memory: MagicMock,
        background_manager: BackgroundTaskManager,
    ):
        """Memory tools (search_memory, etc.) are handled before the background
        check and should work normally regardless of background_manager."""
        memory.search_memory_text.return_value = [
            ("knowledge/test.md", "found it"),
        ]
        handler = ToolHandler(
            anima_dir=anima_dir,
            memory=memory,
            messenger=None,
            tool_registry=[],
            background_manager=background_manager,
        )

        result = handler.handle("search_memory", {"query": "test"})
        assert "found it" in result
        # No background tasks should have been created
        assert background_manager.active_count() == 0
