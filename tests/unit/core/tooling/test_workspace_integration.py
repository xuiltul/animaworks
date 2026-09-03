# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for workspace resolution with tool handlers.

Validates that workspace resolution integrates correctly with:
- submit_tasks (handler_skills)
- delegate_task (handler_org)
- _intercept_task_to_pending (_sdk_hooks)
- TaskExec prompt injection (pending_executor + task_exec.md templates)
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.config.models import AnimaModelConfig, AnimaWorksConfig

# ── TestSubmitTasksWorkspace ─────────────────────────────────────


class TestSubmitTasksWorkspace:
    """Workspace resolution in submit_tasks handler."""

    @pytest.fixture
    def handler(self, tmp_path: Path):
        """Create ToolHandler with minimal deps for submit_tasks."""
        from core.memory import MemoryManager
        from core.tooling.handler import ToolHandler

        anima_dir = tmp_path / "animas" / "test"
        for d in ["state", "episodes", "knowledge", "procedures", "skills"]:
            (anima_dir / d).mkdir(parents=True)
        memory = MemoryManager(anima_dir)
        return ToolHandler(anima_dir, memory)

    def test_valid_workspace_resolves_to_working_directory_in_pending_json(self, handler, tmp_path: Path) -> None:
        """When task has workspace 'myproject' and it resolves, pending JSON has working_directory."""
        resolved_path = tmp_path / "myproject"
        resolved_path.mkdir()

        with patch("core.workspace.resolve_workspace", return_value=Path(resolved_path)):
            result = handler.handle(
                "submit_tasks",
                {
                    "batch_id": "b1",
                    "tasks": [
                        {
                            "task_id": "t1",
                            "title": "Task 1",
                            "description": "Do something",
                            "workspace": "myproject",
                        }
                    ],
                },
            )
        data = json.loads(result)
        assert data.get("status") == "submitted"
        assert "t1" in data.get("task_ids", [])

        pending_path = handler._anima_dir / "state" / "pending" / "t1.json"
        assert pending_path.exists()
        task_json = json.loads(pending_path.read_text(encoding="utf-8"))
        assert task_json.get("working_directory") == str(resolved_path.resolve())

    def test_invalid_workspace_returns_error(self, handler) -> None:
        """Invalid workspace returns error result."""
        with patch(
            "core.workspace.resolve_workspace",
            side_effect=ValueError("Workspace 'bad' not found"),
        ):
            result = handler.handle(
                "submit_tasks",
                {
                    "batch_id": "b1",
                    "tasks": [
                        {
                            "task_id": "t1",
                            "title": "Task 1",
                            "description": "Do something",
                            "workspace": "bad",
                        }
                    ],
                },
            )
        data = json.loads(result)
        assert "error" in data or data.get("status") == "error"
        assert "Workspace" in result or "workspace" in result.lower()

    def test_task_without_workspace_has_empty_working_directory(self, handler, tmp_path: Path) -> None:
        """Task without workspace field has working_directory empty string."""
        with patch("core.workspace.resolve_workspace"):
            result = handler.handle(
                "submit_tasks",
                {
                    "batch_id": "b1",
                    "tasks": [
                        {
                            "task_id": "t1",
                            "title": "Task 1",
                            "description": "Do something",
                        }
                    ],
                },
            )
        data = json.loads(result)
        assert data.get("status") == "submitted"

        pending_path = handler._anima_dir / "state" / "pending" / "t1.json"
        task_json = json.loads(pending_path.read_text(encoding="utf-8"))
        assert task_json.get("working_directory") == ""

    def test_multiple_tasks_with_workspace_each_resolved(self, handler, tmp_path: Path) -> None:
        """Multiple tasks with same workspace all get resolved working_directory."""
        resolved_path = tmp_path / "proj"
        resolved_path.mkdir()

        def resolve(alias: str):
            if alias == "proj":
                return Path(resolved_path)
            raise ValueError(f"Unknown: {alias}")

        with patch("core.workspace.resolve_workspace", side_effect=resolve):
            result = handler.handle(
                "submit_tasks",
                {
                    "batch_id": "b2",
                    "tasks": [
                        {
                            "task_id": "t2a",
                            "title": "A",
                            "description": "A",
                            "workspace": "proj",
                        },
                        {
                            "task_id": "t2b",
                            "title": "B",
                            "description": "B",
                            "workspace": "proj",
                        },
                    ],
                },
            )
        data = json.loads(result)
        assert data.get("status") == "submitted"
        assert len(data.get("task_ids", [])) == 2

        for tid in ("t2a", "t2b"):
            task_json = json.loads(
                (handler._anima_dir / "state" / "pending" / f"{tid}.json").read_text(encoding="utf-8")
            )
            assert task_json.get("working_directory") == str(resolved_path.resolve())


# ── TestDelegateTaskWorkspace ────────────────────────────────────


class TestDelegateTaskWorkspace:
    """Workspace resolution in delegate_task handler."""

    @pytest.fixture
    def handler(self, tmp_path: Path):
        """Create ToolHandler with subordinate config for delegate_task."""
        from core.memory import MemoryManager
        from core.tooling.handler import ToolHandler

        anima_dir = tmp_path / "animas" / "boss"
        sub_dir = tmp_path / "animas" / "sub"
        for d in ["state", "episodes", "knowledge", "procedures", "skills"]:
            (anima_dir / d).mkdir(parents=True)
            (sub_dir / d).mkdir(parents=True)
        (anima_dir / "status.json").write_text("{}", encoding="utf-8")
        (sub_dir / "status.json").write_text("{}", encoding="utf-8")
        memory = MemoryManager(anima_dir)
        cfg = AnimaWorksConfig()
        cfg.animas = {
            "boss": AnimaModelConfig(supervisor=None),
            "sub": AnimaModelConfig(supervisor="boss"),
        }
        with (
            patch("core.config.models.load_config", return_value=cfg),
            patch("core.paths.get_animas_dir", return_value=tmp_path / "animas"),
            patch("core.paths.get_data_dir", return_value=tmp_path),
        ):
            return ToolHandler(anima_dir, memory)

    def test_delegate_task_with_workspace_writes_working_directory(self, handler, tmp_path: Path) -> None:
        """delegate_task with workspace resolves and writes working_directory to pending JSON."""
        resolved_path = tmp_path / "project"
        resolved_path.mkdir()
        cfg = AnimaWorksConfig()
        cfg.animas = {
            "boss": AnimaModelConfig(supervisor=None),
            "sub": AnimaModelConfig(supervisor="boss"),
        }

        with (
            patch("core.config.models.load_config", return_value=cfg),
            patch("core.paths.get_animas_dir", return_value=tmp_path / "animas"),
            patch("core.paths.get_data_dir", return_value=tmp_path),
            patch("core.workspace.resolve_workspace", return_value=Path(resolved_path)),
        ):
            result = handler.handle(
                "delegate_task",
                {
                    "name": "sub",
                    "instruction": "Implement feature X",
                    "summary": "Feature X",
                    "deadline": "1h",
                    "workspace": "project",
                },
            )
        # Success: pending JSON written with working_directory
        pending_dir = tmp_path / "animas" / "sub" / "state" / "pending"
        assert pending_dir.exists(), f"Expected pending dir; result: {result}"
        json_files = list(pending_dir.glob("*.json"))
        assert json_files, f"No pending JSON; result: {result}"
        task_json = json.loads(json_files[0].read_text(encoding="utf-8"))
        assert task_json.get("working_directory") == str(resolved_path.resolve())

    def test_delegate_task_invalid_workspace_returns_error(self, handler) -> None:
        """delegate_task with invalid workspace returns error."""
        with patch(
            "core.workspace.resolve_workspace",
            side_effect=ValueError("Workspace 'nonexistent' not found"),
        ):
            result = handler.handle(
                "delegate_task",
                {
                    "name": "sub",
                    "instruction": "Do X",
                    "summary": "X",
                    "deadline": "1h",
                    "workspace": "nonexistent",
                },
            )
        data = json.loads(result)
        assert "error" in data or "InvalidArguments" in result
        assert "Workspace" in result or "workspace" in result.lower()


# ── TestInterceptWorkingDirectory ─────────────────────────────────


class TestInterceptWorkingDirectory:
    """_intercept_task_to_pending includes working_directory in task JSON."""

    def test_intercept_includes_working_directory_key(self, tmp_path: Path) -> None:
        """Intercepted task JSON has working_directory key."""
        from core.execution._sdk_hooks import _intercept_task_to_pending

        anima_dir = tmp_path / "animas" / "test"
        (anima_dir / "state" / "pending").mkdir(parents=True)

        task_id = _intercept_task_to_pending(
            anima_dir,
            {"description": "Background task", "prompt": "Do X"},
            tool_use_id=None,
        )
        pending_path = anima_dir / "state" / "pending" / f"{task_id}.json"
        assert pending_path.exists()
        task_json = json.loads(pending_path.read_text(encoding="utf-8"))
        assert "working_directory" in task_json
        assert task_json["working_directory"] == ""

    def test_working_directory_defaults_to_empty_string(self, tmp_path: Path) -> None:
        """working_directory in intercepted task defaults to empty string."""
        from core.execution._sdk_hooks import _intercept_task_to_pending

        anima_dir = tmp_path / "animas" / "a"
        (anima_dir / "state" / "pending").mkdir(parents=True)

        task_id = _intercept_task_to_pending(
            anima_dir,
            {"description": "Task", "prompt": "Task"},
            tool_use_id=None,
        )
        task_json = json.loads((anima_dir / "state" / "pending" / f"{task_id}.json").read_text(encoding="utf-8"))
        assert task_json["working_directory"] == ""


# ── TestTaskExecPromptInjection ─────────────────────────────────


class TestTaskExecPromptInjection:
    """TaskExec prompt templates contain workspace placeholder."""

    @pytest.fixture
    def repo_root(self) -> Path:
        """Project root for template paths (tests/unit/core/tooling -> project root)."""
        return Path(__file__).resolve().parents[4]

    def test_ja_task_exec_template_has_workspace_placeholder(self, repo_root: Path) -> None:
        """Japanese task_exec.md contains {workspace} placeholder."""
        template_path = repo_root / "templates" / "ja" / "prompts" / "task_exec.md"
        assert template_path.exists(), f"Template not found: {template_path}"
        content = template_path.read_text(encoding="utf-8")
        assert "{workspace}" in content, "ja task_exec.md must contain {workspace}"

    def test_en_task_exec_template_has_workspace_placeholder(self, repo_root: Path) -> None:
        """English task_exec.md contains {workspace} placeholder."""
        template_path = repo_root / "templates" / "en" / "prompts" / "task_exec.md"
        assert template_path.exists(), f"Template not found: {template_path}"
        content = template_path.read_text(encoding="utf-8")
        assert "{workspace}" in content, "en task_exec.md must contain {workspace}"

    def test_load_prompt_task_exec_accepts_workspace_param(self) -> None:
        """load_prompt('task_exec', workspace=...) injects workspace into output."""
        from core.paths import load_prompt

        result = load_prompt(
            "task_exec",
            task_id="t1",
            title="Test",
            submitted_by="alice",
            workspace="/path/to/project",
            description="Do X",
            context="",
            acceptance_criteria="",
            constraints="",
            file_paths="",
        )
        assert "/path/to/project" in result
        assert "{workspace}" not in result
