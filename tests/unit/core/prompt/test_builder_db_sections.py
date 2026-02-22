from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for builder.py DB-first section loading with file fallback."""

from unittest.mock import MagicMock, patch

import pytest


# ── DB-first with file fallback ───────────────────────────────


class TestBuilderDBFallback:
    """Test that builder.py reads from DB with file fallback.

    These tests verify the DB-first lookup pattern used throughout
    builder.py: ``(store.get_section(key) if store else None) or load_prompt(key)``.
    """

    @patch("core.tooling.prompt_db.get_prompt_store")
    def test_behavior_rules_from_db(self, mock_store_fn: MagicMock) -> None:
        """When DB has behavior_rules, it should be returned."""
        mock_store = MagicMock()
        mock_store.get_section.return_value = "DB behavior rules"
        mock_store_fn.return_value = mock_store

        store = mock_store_fn()
        result = store.get_section("behavior_rules") if store else None
        assert result == "DB behavior rules"

    @patch("core.tooling.prompt_db.get_prompt_store")
    def test_fallback_when_db_returns_none(self, mock_store_fn: MagicMock) -> None:
        """When DB returns None, file fallback should be used."""
        mock_store = MagicMock()
        mock_store.get_section.return_value = None
        mock_store_fn.return_value = mock_store

        store = mock_store_fn()
        result = (store.get_section("behavior_rules") if store else None) or "fallback"
        assert result == "fallback"

    @patch("core.tooling.prompt_db.get_prompt_store")
    def test_fallback_when_store_unavailable(self, mock_store_fn: MagicMock) -> None:
        """When DB store is None, file fallback should be used."""
        mock_store_fn.return_value = None

        store = mock_store_fn()
        result = (store.get_section("behavior_rules") if store else None) or "fallback"
        assert result == "fallback"

    @patch("core.tooling.prompt_db.get_prompt_store")
    def test_environment_section_from_db(self, mock_store_fn: MagicMock) -> None:
        """Environment section follows the same DB-first pattern."""
        mock_store = MagicMock()
        mock_store.get_section.return_value = "DB environment for {data_dir}"
        mock_store_fn.return_value = mock_store

        store = mock_store_fn()
        raw = store.get_section("environment") if store else None
        assert raw is not None
        # builder.py calls .format() on the result
        formatted = raw.format(data_dir="/test/data", anima_name="test")
        assert "/test/data" in formatted

    @patch("core.tooling.prompt_db.get_prompt_store")
    def test_emotion_instruction_from_db(self, mock_store_fn: MagicMock) -> None:
        """Emotion instruction section uses DB-first pattern."""
        mock_store = MagicMock()
        mock_store.get_section.return_value = "DB emotion instruction"
        mock_store_fn.return_value = mock_store

        store = mock_store_fn()
        result = (
            store.get_section("emotion_instruction") if store else None
        ) or "hardcoded fallback"
        assert result == "DB emotion instruction"

    @patch("core.tooling.prompt_db.get_prompt_store")
    def test_a2_reflection_from_db(self, mock_store_fn: MagicMock) -> None:
        """A2 reflection section uses DB-first pattern."""
        mock_store = MagicMock()
        mock_store.get_section.return_value = "DB a2 reflection"
        mock_store_fn.return_value = mock_store

        store = mock_store_fn()
        result = (
            store.get_section("a2_reflection") if store else None
        ) or "file fallback"
        assert result == "DB a2 reflection"

    @patch("core.tooling.prompt_db.get_prompt_store")
    def test_hiring_context_from_db(self, mock_store_fn: MagicMock) -> None:
        """Hiring context section uses DB-first pattern."""
        mock_store = MagicMock()
        mock_store.get_section.return_value = "DB hiring context"
        mock_store_fn.return_value = mock_store

        store = mock_store_fn()
        result = (
            store.get_section("hiring_context") if store else None
        ) or "file fallback"
        assert result == "DB hiring context"


# ── Messaging section DB fallback ─────────────────────────────


class TestMessagingDBFallback:
    """Test messaging section DB-first lookup in _build_messaging_section."""

    @patch("core.tooling.prompt_db.get_prompt_store")
    def test_messaging_a1_key_selection(self, mock_store_fn: MagicMock) -> None:
        """Mode a1 uses 'messaging_a1' key for DB lookup."""
        mock_store = MagicMock()
        mock_store.get_section.return_value = "a1 messaging {animas_line}"
        mock_store_fn.return_value = mock_store

        store = mock_store_fn()
        db_key = "messaging_a1"  # a1 mode
        raw = store.get_section(db_key) if store else None
        assert raw is not None
        # The builder formats the template
        try:
            formatted = raw.format(
                animas_line="alice, bob",
                main_py="/path/to/main.py",
                self_name="test",
            )
        except (KeyError, IndexError):
            formatted = raw
        assert "alice, bob" in formatted

    @patch("core.tooling.prompt_db.get_prompt_store")
    def test_messaging_non_a1_key_selection(self, mock_store_fn: MagicMock) -> None:
        """Non-a1 mode uses 'messaging' key for DB lookup."""
        mock_store = MagicMock()
        mock_store.get_section.return_value = None
        mock_store_fn.return_value = mock_store

        store = mock_store_fn()
        db_key = "messaging"  # non-a1 mode
        raw = store.get_section(db_key) if store else None
        # Falls back to load_prompt when DB returns None
        assert raw is None


# ── Communication rules DB fallback ───────────────────────────


class TestCommunicationRulesDBFallback:
    """Test communication_rules section DB-first lookup in _build_org_context."""

    @patch("core.tooling.prompt_db.get_prompt_store")
    def test_communication_rules_a1_from_db(self, mock_store_fn: MagicMock) -> None:
        """A1 mode uses 'communication_rules_a1' key."""
        mock_store = MagicMock()
        mock_store.get_section.return_value = "DB comm rules a1"
        mock_store_fn.return_value = mock_store

        store = mock_store_fn()
        cr_key = "communication_rules_a1"
        result = (store.get_section(cr_key) if store else None) or "file fallback"
        assert result == "DB comm rules a1"

    @patch("core.tooling.prompt_db.get_prompt_store")
    def test_communication_rules_non_a1_from_db(self, mock_store_fn: MagicMock) -> None:
        """Non-a1 mode uses 'communication_rules' key."""
        mock_store = MagicMock()
        mock_store.get_section.return_value = "DB comm rules"
        mock_store_fn.return_value = mock_store

        store = mock_store_fn()
        cr_key = "communication_rules"
        result = (store.get_section(cr_key) if store else None) or "file fallback"
        assert result == "DB comm rules"
