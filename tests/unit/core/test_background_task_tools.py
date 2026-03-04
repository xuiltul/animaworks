"""Tests for check_background_task / list_background_tasks handler methods."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from core.background import BackgroundTaskManager, TaskStatus
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
def background_manager(anima_dir: Path) -> BackgroundTaskManager:
    return BackgroundTaskManager(anima_dir, anima_name="test-anima")


@pytest.fixture
def handler_with_bg(
    anima_dir: Path, memory: MagicMock, background_manager: BackgroundTaskManager,
) -> ToolHandler:
    return ToolHandler(
        anima_dir=anima_dir,
        memory=memory,
        messenger=None,
        tool_registry=[],
        background_manager=background_manager,
    )


@pytest.fixture
def handler_without_bg(anima_dir: Path, memory: MagicMock) -> ToolHandler:
    return ToolHandler(
        anima_dir=anima_dir,
        memory=memory,
        messenger=None,
        tool_registry=[],
        background_manager=None,
    )


# ── check_background_task ─────────────────────────────────────


class TestCheckBackgroundTask:
    def test_returns_error_when_bg_disabled(self, handler_without_bg: ToolHandler):
        result = handler_without_bg.handle("check_background_task", {"task_id": "abc"})
        assert "Error" in result

    def test_returns_error_when_task_id_missing(self, handler_with_bg: ToolHandler):
        result = handler_with_bg.handle("check_background_task", {})
        assert "Error" in result

    def test_returns_error_when_task_not_found(self, handler_with_bg: ToolHandler):
        result = handler_with_bg.handle("check_background_task", {"task_id": "nonexistent"})
        assert "Error" in result
        assert "nonexistent" in result

    async def test_returns_task_details(
        self,
        handler_with_bg: ToolHandler,
        background_manager: BackgroundTaskManager,
    ):
        handler_with_bg._external = MagicMock()
        handler_with_bg._external.dispatch.return_value = "result"

        submit_result = handler_with_bg.handle(
            "generate_character_assets", {"prompt": "cat"},
        )
        task_id = json.loads(submit_result)["task_id"]

        result = handler_with_bg.handle("check_background_task", {"task_id": task_id})
        parsed = json.loads(result)

        assert parsed["task_id"] == task_id
        assert parsed["tool_name"] == "generate_character_assets"
        assert parsed["status"] in ("running", "completed", "failed", "pending")


# ── list_background_tasks ─────────────────────────────────────


class TestListBackgroundTasks:
    def test_returns_error_when_bg_disabled(self, handler_without_bg: ToolHandler):
        result = handler_without_bg.handle("list_background_tasks", {})
        assert "Error" in result

    def test_returns_empty_list_when_no_tasks(self, handler_with_bg: ToolHandler):
        result = handler_with_bg.handle("list_background_tasks", {})
        parsed = json.loads(result)
        assert parsed == []

    async def test_returns_all_tasks(
        self,
        handler_with_bg: ToolHandler,
        background_manager: BackgroundTaskManager,
    ):
        handler_with_bg._external = MagicMock()
        handler_with_bg._external.dispatch.return_value = "result"

        handler_with_bg.handle("generate_character_assets", {"prompt": "cat"})
        handler_with_bg.handle("local_llm", {"query": "hello"})

        result = handler_with_bg.handle("list_background_tasks", {})
        parsed = json.loads(result)
        assert len(parsed) == 2

    async def test_filters_by_status(
        self,
        handler_with_bg: ToolHandler,
        background_manager: BackgroundTaskManager,
    ):
        handler_with_bg._external = MagicMock()
        handler_with_bg._external.dispatch.return_value = "result"

        handler_with_bg.handle("generate_character_assets", {"prompt": "cat"})

        result = handler_with_bg.handle(
            "list_background_tasks", {"status": "running"},
        )
        parsed = json.loads(result)
        assert all(t["status"] == "running" for t in parsed)

    def test_invalid_status_returns_error(self, handler_with_bg: ToolHandler):
        result = handler_with_bg.handle(
            "list_background_tasks", {"status": "invalid_status"},
        )
        assert "Error" in result
        assert "invalid_status" in result
