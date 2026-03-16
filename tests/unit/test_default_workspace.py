# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for default_workspace in status.json.

Tests that default_workspace is injected into the prompt when set,
and that pending_executor uses it as working_directory fallback.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.helpers.filesystem import create_test_data_dir

# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """Create isolated data dir and redirect ANIMAWORKS_DATA_DIR."""
    from core.config import invalidate_cache
    from core.paths import _prompt_cache
    from core.tooling.prompt_db import reset_prompt_store

    d = create_test_data_dir(tmp_path)
    monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(d))
    invalidate_cache()
    _prompt_cache.clear()
    reset_prompt_store()

    yield d

    invalidate_cache()
    _prompt_cache.clear()
    reset_prompt_store()


def _write_status(data_dir: Path, anima_name: str, **kwargs) -> Path:
    """Write status.json for anima with given fields."""
    anima_dir = data_dir / "animas" / anima_name
    anima_dir.mkdir(parents=True, exist_ok=True)
    (anima_dir / "identity.md").write_text("# Test\n", encoding="utf-8")
    for sub in ["episodes", "knowledge", "skills", "state", "shortterm"]:
        (anima_dir / sub).mkdir(parents=True, exist_ok=True)
    status_path = anima_dir / "status.json"
    status_path.write_text(json.dumps(kwargs, indent=2, ensure_ascii=False), encoding="utf-8")
    return anima_dir


# ── build_system_prompt: default_workspace in prompt ────────────────────────────


class TestDefaultWorkspaceInPrompt:
    """default_workspace from status.json appears in prompt environment section."""

    def test_default_workspace_in_prompt_when_set(self, data_dir):
        """When status.json has default_workspace, path is shown in prompt."""
        config_path = data_dir / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["workspaces"] = {"myproj": "/home/user/myproj"}
        config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

        from core.config import invalidate_cache

        invalidate_cache()

        anima_dir = _write_status(data_dir, "test-anima", model="claude-sonnet-4-6", default_workspace="myproj")

        from core.memory import MemoryManager
        from core.prompt.builder import build_system_prompt

        with patch("core.workspace.resolve_workspace", return_value=Path("/home/user/myproj")):
            memory = MemoryManager(anima_dir)
            result = build_system_prompt(memory, trigger="chat")

        assert "default_workspace" in result.system_prompt
        assert "/home/user/myproj" in result.system_prompt
        assert "myproj" in result.system_prompt

    def test_no_injection_when_default_workspace_unset(self, data_dir):
        """When default_workspace is not set, nothing is injected."""
        anima_dir = _write_status(data_dir, "test-anima", model="claude-sonnet-4-6")

        from core.memory import MemoryManager
        from core.prompt.builder import build_system_prompt

        memory = MemoryManager(anima_dir)
        result = build_system_prompt(memory, trigger="chat")

        assert "default_workspace" not in result.system_prompt

    def test_unresolved_display_when_alias_fails(self, data_dir):
        """When workspace alias resolution fails, 'unresolved' is shown."""
        anima_dir = _write_status(data_dir, "test-anima", model="claude-sonnet-4-6", default_workspace="nonexistent")

        from core.memory import MemoryManager
        from core.prompt.builder import build_system_prompt

        with patch("core.workspace.resolve_workspace", side_effect=ValueError("Not found")):
            memory = MemoryManager(anima_dir)
            result = build_system_prompt(memory, trigger="chat")

        assert "default_workspace" in result.system_prompt
        assert "未解決" in result.system_prompt or "unresolved" in result.system_prompt
        assert "nonexistent" in result.system_prompt


# ── pending_executor: working_directory fallback ───────────────────────────────


class TestPendingExecutorDefaultWorkspaceFallback:
    """PendingTaskExecutor uses default_workspace when working_directory is unspecified."""

    def test_working_directory_fallback_to_default_workspace(self, data_dir):
        """When task has no working_directory, default_workspace is used."""
        config_path = data_dir / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["workspaces"] = {"proj": "/abs/path/proj"}
        config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

        from core.config import invalidate_cache

        invalidate_cache()

        anima_dir = _write_status(data_dir, "test-anima", model="claude-sonnet-4-6", default_workspace="proj")
        (anima_dir / "state" / "pending").mkdir(parents=True, exist_ok=True)

        from core.supervisor.pending_executor import _resolve_default_workspace

        with patch("core.workspace.resolve_workspace", return_value=Path("/abs/path/proj")):
            resolved = _resolve_default_workspace(anima_dir)

        assert resolved == "/abs/path/proj"

    def test_empty_when_default_workspace_unset(self, data_dir):
        """_resolve_default_workspace returns empty when not set."""
        anima_dir = _write_status(data_dir, "test-anima", model="claude-sonnet-4-6")

        from core.supervisor.pending_executor import _resolve_default_workspace

        resolved = _resolve_default_workspace(anima_dir)

        assert resolved == ""
