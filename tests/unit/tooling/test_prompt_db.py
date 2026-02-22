from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for core.tooling.prompt_db — ToolPromptStore and defaults."""

import sqlite3
from pathlib import Path

import pytest

from core.tooling.prompt_db import (
    DEFAULT_DESCRIPTIONS,
    DEFAULT_GUIDES,
    ToolPromptStore,
    get_prompt_store,
    reset_prompt_store,
)


# ── Schema / Init ────────────────────────────────────────────


class TestToolPromptStoreInit:
    def test_schema_created_on_init(self, tmp_path: Path) -> None:
        db_path = tmp_path / "prompts.sqlite3"
        ToolPromptStore(db_path)

        conn = sqlite3.connect(str(db_path))
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        conn.close()
        assert "tool_descriptions" in tables
        assert "tool_guides" in tables

    def test_wal_mode_enabled(self, tmp_path: Path) -> None:
        db_path = tmp_path / "prompts.sqlite3"
        store = ToolPromptStore(db_path)

        conn = sqlite3.connect(str(db_path))
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        conn.close()
        assert mode == "wal"

    def test_parent_directories_created(self, tmp_path: Path) -> None:
        db_path = tmp_path / "nested" / "dir" / "prompts.sqlite3"
        ToolPromptStore(db_path)
        assert db_path.exists()


# ── Descriptions CRUD ────────────────────────────────────────


class TestDescriptionsCRUD:
    @pytest.fixture
    def store(self, tmp_path: Path) -> ToolPromptStore:
        return ToolPromptStore(tmp_path / "prompts.sqlite3")

    def test_set_description_creates_new_entry(self, store: ToolPromptStore) -> None:
        result = store.set_description("search_memory", "Search long-term memory")
        assert result["name"] == "search_memory"
        assert result["description"] == "Search long-term memory"
        assert "updated_at" in result

    def test_get_description_returns_correct_value(
        self, store: ToolPromptStore
    ) -> None:
        store.set_description("read_file", "Read a file by path")
        assert store.get_description("read_file") == "Read a file by path"

    def test_get_description_returns_none_for_missing_key(
        self, store: ToolPromptStore
    ) -> None:
        assert store.get_description("nonexistent_tool") is None

    def test_set_description_overwrites_existing(
        self, store: ToolPromptStore
    ) -> None:
        store.set_description("tool_a", "Original description")
        store.set_description("tool_a", "Updated description")
        assert store.get_description("tool_a") == "Updated description"

    def test_list_descriptions_returns_all_sorted_by_name(
        self, store: ToolPromptStore
    ) -> None:
        store.set_description("zebra_tool", "Z tool")
        store.set_description("alpha_tool", "A tool")
        store.set_description("middle_tool", "M tool")

        result = store.list_descriptions()
        assert len(result) == 3
        names = [r["name"] for r in result]
        assert names == ["alpha_tool", "middle_tool", "zebra_tool"]
        # Verify dict structure
        for entry in result:
            assert "name" in entry
            assert "description" in entry
            assert "updated_at" in entry

    def test_empty_description_rejected_by_check_constraint(
        self, store: ToolPromptStore
    ) -> None:
        with pytest.raises(sqlite3.IntegrityError):
            store.set_description("bad_tool", "")


# ── Guides CRUD ──────────────────────────────────────────────


class TestGuidesCRUD:
    @pytest.fixture
    def store(self, tmp_path: Path) -> ToolPromptStore:
        return ToolPromptStore(tmp_path / "prompts.sqlite3")

    def test_set_guide_creates_new_entry(self, store: ToolPromptStore) -> None:
        result = store.set_guide("a1_builtin", "## Builtin tools guide")
        assert result["key"] == "a1_builtin"
        assert result["content"] == "## Builtin tools guide"
        assert "updated_at" in result

    def test_get_guide_returns_correct_value(self, store: ToolPromptStore) -> None:
        store.set_guide("non_a1", "## Non-A1 mode guide")
        assert store.get_guide("non_a1") == "## Non-A1 mode guide"

    def test_get_guide_returns_none_for_missing_key(
        self, store: ToolPromptStore
    ) -> None:
        assert store.get_guide("nonexistent_guide") is None

    def test_set_guide_overwrites_existing(self, store: ToolPromptStore) -> None:
        store.set_guide("a1_mcp", "Original guide")
        store.set_guide("a1_mcp", "Updated guide")
        assert store.get_guide("a1_mcp") == "Updated guide"

    def test_list_guides_returns_all_sorted_by_key(
        self, store: ToolPromptStore
    ) -> None:
        store.set_guide("z_guide", "Z content")
        store.set_guide("a_guide", "A content")
        store.set_guide("m_guide", "M content")

        result = store.list_guides()
        assert len(result) == 3
        keys = [r["key"] for r in result]
        assert keys == ["a_guide", "m_guide", "z_guide"]
        # Verify dict structure
        for entry in result:
            assert "key" in entry
            assert "content" in entry
            assert "updated_at" in entry

    def test_empty_content_rejected_by_check_constraint(
        self, store: ToolPromptStore
    ) -> None:
        with pytest.raises(sqlite3.IntegrityError):
            store.set_guide("bad_guide", "")


# ── Seed Defaults ────────────────────────────────────────────


class TestSeedDefaults:
    @pytest.fixture
    def store(self, tmp_path: Path) -> ToolPromptStore:
        return ToolPromptStore(tmp_path / "prompts.sqlite3")

    def test_seed_defaults_populates_empty_db(
        self, store: ToolPromptStore
    ) -> None:
        descs = {"tool_x": "X desc", "tool_y": "Y desc"}
        store.seed_defaults(descriptions=descs)

        assert store.get_description("tool_x") == "X desc"
        assert store.get_description("tool_y") == "Y desc"

    def test_seed_defaults_does_not_overwrite_existing(
        self, store: ToolPromptStore
    ) -> None:
        # Pre-populate with a custom value
        store.set_description("tool_x", "Custom description")

        # Seed with a different value for the same key
        descs = {"tool_x": "Default description", "tool_z": "Z desc"}
        store.seed_defaults(descriptions=descs)

        # Custom value should be preserved (INSERT OR IGNORE)
        assert store.get_description("tool_x") == "Custom description"
        # New key should be inserted
        assert store.get_description("tool_z") == "Z desc"

    def test_seed_defaults_with_both_descriptions_and_guides(
        self, store: ToolPromptStore
    ) -> None:
        descs = {"tool_a": "A desc"}
        guides = {"guide_a": "A guide content"}
        store.seed_defaults(descriptions=descs, guides=guides)

        assert store.get_description("tool_a") == "A desc"
        assert store.get_guide("guide_a") == "A guide content"

    def test_seed_defaults_with_none_arguments(
        self, store: ToolPromptStore
    ) -> None:
        # Should be a no-op, not raise
        store.seed_defaults(descriptions=None, guides=None)

        assert store.list_descriptions() == []
        assert store.list_guides() == []


# ── Singleton ────────────────────────────────────────────────


class TestSingleton:
    def setup_method(self) -> None:
        """Reset singleton state before each test."""
        reset_prompt_store()

    def teardown_method(self) -> None:
        """Reset singleton state after each test."""
        reset_prompt_store()

    def test_get_prompt_store_returns_none_when_data_dir_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        nonexistent = tmp_path / "does_not_exist"
        # get_prompt_store does `from core.paths import get_data_dir` at call
        # time, so we patch the function on the module object.
        import core.paths

        monkeypatch.setattr(core.paths, "get_data_dir", lambda: nonexistent)

        result = get_prompt_store()
        assert result is None

    def test_reset_prompt_store_clears_singleton(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import core.paths

        monkeypatch.setattr(core.paths, "get_data_dir", lambda: tmp_path)

        store1 = get_prompt_store()
        assert store1 is not None

        reset_prompt_store()

        store2 = get_prompt_store()
        assert store2 is not None
        # After reset, a new instance should be created
        assert store1 is not store2

    def test_get_prompt_store_returns_store_when_path_exists(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import core.paths

        monkeypatch.setattr(core.paths, "get_data_dir", lambda: tmp_path)

        result = get_prompt_store()
        assert isinstance(result, ToolPromptStore)


# ── Default Data ─────────────────────────────────────────────


class TestDefaultData:
    def test_default_descriptions_has_expected_tools(self) -> None:
        expected_names = [
            "search_memory",
            "read_memory_file",
            "write_memory_file",
            "send_message",
            "post_channel",
            "read_channel",
            "call_human",
            "discover_tools",
            "execute_command",
        ]
        for name in expected_names:
            assert name in DEFAULT_DESCRIPTIONS, (
                f"Expected '{name}' in DEFAULT_DESCRIPTIONS"
            )

    def test_default_guides_has_exactly_three_keys(self) -> None:
        assert set(DEFAULT_GUIDES.keys()) == {"a1_builtin", "a1_mcp", "non_a1"}

    def test_guide_content_is_non_empty(self) -> None:
        for key, content in DEFAULT_GUIDES.items():
            assert len(content) > 0, f"Guide '{key}' has empty content"
            # Also verify it contains some markdown structure
            assert "#" in content, f"Guide '{key}' should contain markdown headings"

    def test_guide_content_has_no_unresolved_template_variables(self) -> None:
        """Ensure no {data_dir} or {name} template variables remain in guides."""
        for key, content in DEFAULT_GUIDES.items():
            assert "{data_dir}" not in content, (
                f"Guide '{key}' contains unresolved {{data_dir}} template variable"
            )
            assert "{name}" not in content, (
                f"Guide '{key}' contains unresolved {{name}} template variable"
            )
