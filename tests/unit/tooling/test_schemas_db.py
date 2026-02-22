from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for DB overlay integration in core.tooling.schemas.apply_db_descriptions."""

from pathlib import Path
from unittest.mock import patch

import pytest

from core.tooling.prompt_db import ToolPromptStore
from core.tooling.schemas import apply_db_descriptions


# ── apply_db_descriptions ───────────────────────────────────


class TestApplyDbDescriptions:
    def test_tools_pass_through_when_store_is_none(self) -> None:
        """When the prompt store is unavailable (None), tools pass through unchanged."""
        tools = [
            {"name": "search_memory", "description": "Original desc", "parameters": {}},
            {"name": "read_file", "description": "Read a file", "parameters": {}},
        ]

        with patch("core.tooling.prompt_db.get_prompt_store", return_value=None):
            result = apply_db_descriptions(tools)

        assert result is tools  # exact same list object
        assert result[0]["description"] == "Original desc"
        assert result[1]["description"] == "Read a file"

    def test_matching_description_overrides_tool_description(
        self, tmp_path: Path
    ) -> None:
        """When the store has a matching description, it overrides the tool's."""
        store = ToolPromptStore(tmp_path / "prompts.sqlite3")
        store.set_description("search_memory", "DB-overridden description")

        tools = [
            {"name": "search_memory", "description": "Original desc", "parameters": {}},
        ]

        with patch("core.tooling.prompt_db.get_prompt_store", return_value=store):
            result = apply_db_descriptions(tools)

        assert len(result) == 1
        assert result[0]["description"] == "DB-overridden description"
        assert result[0]["name"] == "search_memory"

    def test_no_match_preserves_original_description(
        self, tmp_path: Path
    ) -> None:
        """When the store has no match, the original description is preserved."""
        store = ToolPromptStore(tmp_path / "prompts.sqlite3")
        # Store has no entries for 'read_file'

        tools = [
            {"name": "read_file", "description": "Original desc", "parameters": {}},
        ]

        with patch("core.tooling.prompt_db.get_prompt_store", return_value=store):
            result = apply_db_descriptions(tools)

        assert len(result) == 1
        assert result[0]["description"] == "Original desc"

    def test_original_tool_dict_not_mutated(self, tmp_path: Path) -> None:
        """The original tool dict should not be mutated; a new dict is created."""
        store = ToolPromptStore(tmp_path / "prompts.sqlite3")
        store.set_description("my_tool", "New description from DB")

        original_tool = {
            "name": "my_tool",
            "description": "Original description",
            "parameters": {"type": "object"},
        }
        tools = [original_tool]

        with patch("core.tooling.prompt_db.get_prompt_store", return_value=store):
            result = apply_db_descriptions(tools)

        # Result should have the new description
        assert result[0]["description"] == "New description from DB"
        # Original dict should NOT be mutated
        assert original_tool["description"] == "Original description"

    def test_mixed_case_some_overridden_some_not(self, tmp_path: Path) -> None:
        """Some tools have DB overrides, others don't."""
        store = ToolPromptStore(tmp_path / "prompts.sqlite3")
        store.set_description("tool_a", "Overridden A")
        store.set_description("tool_c", "Overridden C")

        tools = [
            {"name": "tool_a", "description": "Original A", "parameters": {}},
            {"name": "tool_b", "description": "Original B", "parameters": {}},
            {"name": "tool_c", "description": "Original C", "parameters": {}},
        ]

        with patch("core.tooling.prompt_db.get_prompt_store", return_value=store):
            result = apply_db_descriptions(tools)

        assert len(result) == 3
        assert result[0]["name"] == "tool_a"
        assert result[0]["description"] == "Overridden A"
        assert result[1]["name"] == "tool_b"
        assert result[1]["description"] == "Original B"  # preserved
        assert result[2]["name"] == "tool_c"
        assert result[2]["description"] == "Overridden C"
