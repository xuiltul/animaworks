"""E2E tests for tool auto-discovery, creation, hot-reload, and sharing."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from core.tooling.handler import ToolHandler


# Test flow:
# 1. Create a anima_dir with tools/ containing a custom Python tool
# 2. Verify discover_personal_tools finds it
# 3. Create ToolHandler, call refresh_tools, verify the tool is discovered
# 4. Call the custom tool via handler.handle() and verify it works
# 5. Test share_tool copies to common_tools
# 6. Test discover_common_tools finds the shared tool


@pytest.mark.e2e
class TestToolHotReload:
    """Verify the tool creation -> refresh -> use flow end-to-end."""

    # Test 1: Anima writes a tool file, refresh discovers it, handler can call it
    def test_create_tool_refresh_and_use(self, tmp_path: Path) -> None:
        # Set up anima directory with a custom tool
        anima_dir = tmp_path / "animas" / "alice"
        anima_dir.mkdir(parents=True)
        (anima_dir / "permissions.md").write_text(
            "## ツール作成\n- 個人ツール: yes\n- 共有ツール: yes",
            encoding="utf-8",
        )

        # Create a simple tool module
        tools_dir = anima_dir / "tools"
        tools_dir.mkdir()
        (tools_dir / "greet.py").write_text(
            '''def get_tool_schemas():
    return [{
        "name": "greet",
        "description": "Greet someone",
        "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
    }]

def dispatch(name, args):
    if name == "greet":
        return f"Hello, {args['name']}!"
    raise ValueError(f"Unknown: {name}")
''',
            encoding="utf-8",
        )

        # Discover and verify
        from core.tools import discover_personal_tools

        personal = discover_personal_tools(anima_dir)
        assert "greet" in personal

        # Create handler and test refresh + use
        memory = MagicMock()
        memory.read_permissions.return_value = (
            "## ツール作成\n- 個人ツール: yes\n- 共有ツール: yes"
        )
        memory.search_memory_text.return_value = []

        handler = ToolHandler(
            anima_dir=anima_dir,
            memory=memory,
            messenger=None,
            tool_registry=[],
            personal_tools=personal,
        )

        # Call the custom tool
        result = handler.handle("greet", {"name": "World"})
        assert "Hello, World!" in result

    # Test 2: refresh_tools discovers newly added tool
    def test_refresh_discovers_new_tool(self, tmp_path: Path) -> None:
        anima_dir = tmp_path / "animas" / "alice"
        anima_dir.mkdir(parents=True)
        (anima_dir / "permissions.md").write_text("", encoding="utf-8")

        memory = MagicMock()
        memory.read_permissions.return_value = ""
        memory.search_memory_text.return_value = []

        handler = ToolHandler(
            anima_dir=anima_dir,
            memory=memory,
            messenger=None,
            tool_registry=[],
        )

        # Initially no personal tools
        result = handler.handle("refresh_tools", {})
        assert "No personal or common tools found" in result

        # Now create a tool file
        tools_dir = anima_dir / "tools"
        tools_dir.mkdir()
        (tools_dir / "calc.py").write_text(
            '''def get_tool_schemas():
    return [{"name": "calc_add", "description": "Add", "parameters": {"type": "object"}}]

def dispatch(name, args):
    return "42"
''',
            encoding="utf-8",
        )

        # Refresh should find it
        result = handler.handle("refresh_tools", {})
        assert "calc" in result
        assert "Refreshed tools" in result

        # Now the tool should be callable
        result = handler.handle("calc_add", {})
        assert "42" in result

    # Test 3: share_tool copies to common_tools
    def test_share_tool_copies_to_common(self, tmp_path: Path) -> None:
        anima_dir = tmp_path / "animas" / "alice"
        anima_dir.mkdir(parents=True)
        (anima_dir / "permissions.md").write_text("", encoding="utf-8")
        tools_dir = anima_dir / "tools"
        tools_dir.mkdir()
        (tools_dir / "helper.py").write_text("# helper tool", encoding="utf-8")

        common_dir = tmp_path / "common_tools"

        memory = MagicMock()
        memory.read_permissions.return_value = (
            "## ツール作成\n- 共有ツール: yes"
        )
        memory.search_memory_text.return_value = []

        handler = ToolHandler(
            anima_dir=anima_dir,
            memory=memory,
            messenger=None,
            tool_registry=[],
        )

        from unittest.mock import patch

        with patch("core.paths.get_data_dir", return_value=tmp_path):
            result = handler.handle("share_tool", {"tool_name": "helper"})

        assert "Shared tool" in result
        assert (common_dir / "helper.py").exists()
        assert (
            (common_dir / "helper.py").read_text(encoding="utf-8")
            == "# helper tool"
        )

    # Test 4: tool creation blocked without permission
    def test_tool_creation_blocked_without_permission(
        self, tmp_path: Path
    ) -> None:
        anima_dir = tmp_path / "animas" / "alice"
        anima_dir.mkdir(parents=True)
        (anima_dir / "permissions.md").write_text("", encoding="utf-8")

        memory = MagicMock()
        memory.read_permissions.return_value = ""  # No tool creation section
        memory.search_memory_text.return_value = []

        handler = ToolHandler(
            anima_dir=anima_dir,
            memory=memory,
            messenger=None,
            tool_registry=[],
        )

        result = handler.handle(
            "write_memory_file",
            {"path": "tools/evil.py", "content": "# evil"},
        )
        parsed = json.loads(result)
        assert parsed["error_type"] == "PermissionDenied"
        assert "ツール作成" in parsed["message"]
        # File should NOT be created
        assert not (anima_dir / "tools" / "evil.py").exists()

    # Test 5: discover_common_tools finds shared tools
    def test_discover_common_tools_finds_shared(self, tmp_path: Path) -> None:
        common_dir = tmp_path / "common_tools"
        common_dir.mkdir()
        (common_dir / "shared_calc.py").write_text(
            "# shared", encoding="utf-8"
        )
        (common_dir / "_internal.py").write_text("# skip", encoding="utf-8")

        from core.tools import discover_common_tools

        result = discover_common_tools(data_dir=tmp_path)
        assert "shared_calc" in result
        assert "_internal" not in result
