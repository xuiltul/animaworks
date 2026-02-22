from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for system_sections table CRUD in core.tooling.prompt_db."""

import sqlite3
from pathlib import Path

import pytest

from core.tooling.prompt_db import SECTION_CONDITIONS, ToolPromptStore


@pytest.fixture
def store(tmp_path: Path) -> ToolPromptStore:
    """Create a fresh ToolPromptStore in a temp directory."""
    return ToolPromptStore(tmp_path / "test.sqlite3")


# ── Schema ────────────────────────────────────────────────────


class TestSystemSectionsSchema:
    """Verify the system_sections table is created alongside existing tables."""

    def test_table_exists(self, store: ToolPromptStore) -> None:
        conn = sqlite3.connect(str(store._db_path))
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        table_names = [t[0] for t in tables]
        conn.close()
        assert "system_sections" in table_names
        assert "tool_descriptions" in table_names
        assert "tool_guides" in table_names


# ── Sections CRUD ─────────────────────────────────────────────


class TestSectionsCRUD:
    """Test get/set/list operations on system_sections."""

    def test_get_section_returns_none_for_missing(self, store: ToolPromptStore) -> None:
        assert store.get_section("nonexistent") is None

    def test_set_and_get_section(self, store: ToolPromptStore) -> None:
        store.set_section("test_key", "test content", "mode:a1")
        assert store.get_section("test_key") == "test content"

    def test_get_section_with_condition(self, store: ToolPromptStore) -> None:
        store.set_section("test_key", "content", "mode:a2")
        result = store.get_section_with_condition("test_key")
        assert result is not None
        content, condition = result
        assert content == "content"
        assert condition == "mode:a2"

    def test_get_section_with_condition_returns_none_for_missing(
        self, store: ToolPromptStore
    ) -> None:
        assert store.get_section_with_condition("nonexistent") is None

    def test_get_section_with_null_condition(self, store: ToolPromptStore) -> None:
        store.set_section("test_key", "content", None)
        result = store.get_section_with_condition("test_key")
        assert result is not None
        content, condition = result
        assert content == "content"
        assert condition is None

    def test_set_section_upsert(self, store: ToolPromptStore) -> None:
        store.set_section("key", "v1", "mode:a1")
        store.set_section("key", "v2", "mode:a2")
        assert store.get_section("key") == "v2"
        _, cond = store.get_section_with_condition("key")
        assert cond == "mode:a2"

    def test_list_sections_empty(self, store: ToolPromptStore) -> None:
        assert store.list_sections() == []

    def test_list_sections_ordered(self, store: ToolPromptStore) -> None:
        store.set_section("beta", "b", None)
        store.set_section("alpha", "a", "mode:a1")
        sections = store.list_sections()
        assert len(sections) == 2
        assert sections[0]["key"] == "alpha"
        assert sections[1]["key"] == "beta"

    def test_list_sections_has_all_fields(self, store: ToolPromptStore) -> None:
        store.set_section("key", "content", "mode:a1")
        sections = store.list_sections()
        assert len(sections) == 1
        s = sections[0]
        assert "key" in s
        assert "content" in s
        assert "condition" in s
        assert "updated_at" in s

    def test_set_section_returns_record(self, store: ToolPromptStore) -> None:
        result = store.set_section("key", "content", "mode:a1")
        assert result["key"] == "key"
        assert result["content"] == "content"
        assert result["condition"] == "mode:a1"
        assert "updated_at" in result

    def test_set_section_returns_record_with_null_condition(
        self, store: ToolPromptStore
    ) -> None:
        result = store.set_section("key", "content", None)
        assert result["key"] == "key"
        assert result["condition"] is None


# ── Section constraints ───────────────────────────────────────


class TestSectionConstraints:
    """Test CHECK constraints on system_sections."""

    def test_empty_content_rejected(self, store: ToolPromptStore) -> None:
        with pytest.raises(sqlite3.IntegrityError):
            store.set_section("key", "", None)


# ── Seed defaults with sections ───────────────────────────────


class TestSeedWithSections:
    """Test seed_defaults with sections parameter."""

    def test_seed_sections(self, store: ToolPromptStore) -> None:
        sections = {
            "behavior_rules": ("content1", None),
            "environment": ("content2", None),
        }
        store.seed_defaults(sections=sections)
        assert store.get_section("behavior_rules") == "content1"
        assert store.get_section("environment") == "content2"

    def test_seed_preserves_existing(self, store: ToolPromptStore) -> None:
        store.set_section("behavior_rules", "user edited", None)
        sections = {"behavior_rules": ("original", None)}
        store.seed_defaults(sections=sections)
        # INSERT OR IGNORE should preserve existing user edits
        assert store.get_section("behavior_rules") == "user edited"

    def test_seed_backward_compatible(self, store: ToolPromptStore) -> None:
        """seed_defaults still works without sections parameter."""
        store.seed_defaults(descriptions={"test": "desc"})
        assert store.get_description("test") == "desc"

    def test_seed_sections_with_conditions(self, store: ToolPromptStore) -> None:
        sections = {
            "messaging_a1": ("a1 msg", "mode:a1"),
            "a2_reflection": ("a2 ref", "mode:a2"),
        }
        store.seed_defaults(sections=sections)

        result = store.get_section_with_condition("messaging_a1")
        assert result is not None
        assert result[0] == "a1 msg"
        assert result[1] == "mode:a1"

        result = store.get_section_with_condition("a2_reflection")
        assert result is not None
        assert result[0] == "a2 ref"
        assert result[1] == "mode:a2"

    def test_seed_all_three_types(self, store: ToolPromptStore) -> None:
        """Seed descriptions, guides, and sections together."""
        store.seed_defaults(
            descriptions={"tool_a": "A desc"},
            guides={"guide_a": "A guide"},
            sections={"sec_a": ("A section", None)},
        )
        assert store.get_description("tool_a") == "A desc"
        assert store.get_guide("guide_a") == "A guide"
        assert store.get_section("sec_a") == "A section"

    def test_seed_with_none_sections(self, store: ToolPromptStore) -> None:
        """seed_defaults with sections=None is a no-op for sections."""
        store.seed_defaults(sections=None)
        assert store.list_sections() == []


# ── SECTION_CONDITIONS constant ───────────────────────────────


class TestSectionConditions:
    """Test SECTION_CONDITIONS constant."""

    def test_has_expected_keys(self) -> None:
        expected = {
            "behavior_rules",
            "environment",
            "messaging_a1",
            "messaging",
            "communication_rules_a1",
            "communication_rules",
            "emotion_instruction",
            "a2_reflection",
            "hiring_context",
        }
        assert set(SECTION_CONDITIONS.keys()) == expected

    def test_conditions_are_correct_types(self) -> None:
        for key, cond in SECTION_CONDITIONS.items():
            assert cond is None or isinstance(cond, str), (
                f"{key}: expected None or str, got {type(cond)}"
            )

    def test_specific_conditions(self) -> None:
        """Verify a few well-known condition values."""
        assert SECTION_CONDITIONS["behavior_rules"] is None
        assert SECTION_CONDITIONS["messaging_a1"] == "mode:a1"
        assert SECTION_CONDITIONS["messaging"] == "mode:non_a1"
        assert SECTION_CONDITIONS["a2_reflection"] == "mode:a2"
        assert SECTION_CONDITIONS["hiring_context"] == "solo_top_level"
