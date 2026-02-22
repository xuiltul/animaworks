from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for praise-loop prevention changes.

Tests cover three code changes:
1. Messenger.send() — board_mention no longer exempt from depth limiter
2. ToolHandler._handle_post_channel() — _suppress_board_fanout flag
3. _migrate_praise_loop_prevention_v1() — SQLite migration for prompt DB
"""

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from core.schemas import Message


# ── Test 1: board_mention is no longer exempt from depth check ────


@pytest.mark.unit
class TestBoardMentionDepthCheck:
    """Verify that board_mention goes through the depth limiter,
    while ack/error/system_alert remain exempt."""

    @pytest.fixture
    def shared_dir(self, tmp_path: Path) -> Path:
        """Create a shared directory with inbox structure."""
        d = tmp_path / "shared"
        d.mkdir()
        return d

    @pytest.fixture
    def animas_dir(self, tmp_path: Path) -> Path:
        """Create animas directory with a target Anima."""
        d = tmp_path / "animas"
        d.mkdir()
        (d / "target-anima").mkdir()
        return d

    @pytest.fixture
    def messenger(self, shared_dir: Path) -> "Messenger":
        from core.messenger import Messenger
        return Messenger(shared_dir=shared_dir, anima_name="sender-anima")

    def test_board_mention_calls_depth_limiter(
        self, messenger: "Messenger", animas_dir: Path
    ) -> None:
        """board_mention should NOT be exempt — depth_limiter.check_and_record must be called."""
        mock_limiter = MagicMock()
        mock_limiter.check_and_record.return_value = True

        with (
            patch("core.paths.get_animas_dir", return_value=animas_dir),
            patch("core.cascade_limiter.depth_limiter", mock_limiter),
        ):
            messenger.send(to="target-anima", content="Great job!", msg_type="board_mention")

        mock_limiter.check_and_record.assert_called_once_with("sender-anima", "target-anima")

    def test_board_mention_blocked_when_depth_exceeded(
        self, messenger: "Messenger", animas_dir: Path
    ) -> None:
        """board_mention should be blocked when depth limiter returns False."""
        mock_limiter = MagicMock()
        mock_limiter.check_and_record.return_value = False

        with (
            patch("core.paths.get_animas_dir", return_value=animas_dir),
            patch("core.cascade_limiter.depth_limiter", mock_limiter),
        ):
            result = messenger.send(
                to="target-anima", content="Great job!", msg_type="board_mention",
            )

        assert result.type == "error"
        assert result.from_person == "system"
        assert "ConversationDepthExceeded" in result.content

    def test_ack_exempt_from_depth_limiter(
        self, messenger: "Messenger", animas_dir: Path
    ) -> None:
        """msg_type='ack' should bypass the depth limiter entirely."""
        mock_limiter = MagicMock()

        # Ensure inbox exists for target
        inbox = messenger.shared_dir / "inbox" / "target-anima"
        inbox.mkdir(parents=True, exist_ok=True)

        with (
            patch("core.paths.get_animas_dir", return_value=animas_dir),
            patch("core.cascade_limiter.depth_limiter", mock_limiter),
        ):
            result = messenger.send(to="target-anima", content="ok", msg_type="ack")

        mock_limiter.check_and_record.assert_not_called()
        assert result.type == "ack"

    def test_error_exempt_from_depth_limiter(
        self, messenger: "Messenger", animas_dir: Path
    ) -> None:
        """msg_type='error' should bypass the depth limiter entirely."""
        mock_limiter = MagicMock()

        inbox = messenger.shared_dir / "inbox" / "target-anima"
        inbox.mkdir(parents=True, exist_ok=True)

        with (
            patch("core.paths.get_animas_dir", return_value=animas_dir),
            patch("core.cascade_limiter.depth_limiter", mock_limiter),
        ):
            result = messenger.send(to="target-anima", content="err", msg_type="error")

        mock_limiter.check_and_record.assert_not_called()
        assert result.type == "error"

    def test_system_alert_exempt_from_depth_limiter(
        self, messenger: "Messenger", animas_dir: Path
    ) -> None:
        """msg_type='system_alert' should bypass the depth limiter entirely."""
        mock_limiter = MagicMock()

        inbox = messenger.shared_dir / "inbox" / "target-anima"
        inbox.mkdir(parents=True, exist_ok=True)

        with (
            patch("core.paths.get_animas_dir", return_value=animas_dir),
            patch("core.cascade_limiter.depth_limiter", mock_limiter),
        ):
            result = messenger.send(
                to="target-anima", content="alert", msg_type="system_alert",
            )

        mock_limiter.check_and_record.assert_not_called()
        assert result.type == "system_alert"

    def test_regular_message_calls_depth_limiter(
        self, messenger: "Messenger", animas_dir: Path
    ) -> None:
        """Regular 'message' type should also go through the depth limiter."""
        mock_limiter = MagicMock()
        mock_limiter.check_and_record.return_value = True

        with (
            patch("core.paths.get_animas_dir", return_value=animas_dir),
            patch("core.cascade_limiter.depth_limiter", mock_limiter),
        ):
            messenger.send(to="target-anima", content="hello", msg_type="message")

        mock_limiter.check_and_record.assert_called_once_with("sender-anima", "target-anima")


# ── Test 2: _suppress_board_fanout flag in handler ────────────────


@pytest.mark.unit
class TestSuppressBoardFanout:
    """Verify that _suppress_board_fanout flag controls fanout in _handle_post_channel."""

    @pytest.fixture
    def anima_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / "animas" / "test-anima"
        d.mkdir(parents=True)
        (d / "permissions.md").write_text("", encoding="utf-8")
        return d

    @pytest.fixture
    def memory(self) -> MagicMock:
        m = MagicMock()
        m.read_permissions.return_value = ""
        m.search_memory_text.return_value = []
        return m

    @pytest.fixture
    def messenger(self) -> MagicMock:
        m = MagicMock()
        m.anima_name = "test-anima"
        msg = MagicMock()
        msg.id = "msg_001"
        msg.thread_id = "thread_001"
        m.send.return_value = msg
        return m

    @pytest.fixture
    def handler(
        self, anima_dir: Path, memory: MagicMock, messenger: MagicMock,
    ) -> "ToolHandler":
        from core.tooling.handler import ToolHandler
        return ToolHandler(
            anima_dir=anima_dir,
            memory=memory,
            messenger=messenger,
            tool_registry=[],
        )

    def test_post_channel_calls_fanout_when_flag_not_set(
        self,
        handler: "ToolHandler",
        messenger: MagicMock,
    ) -> None:
        """Without _suppress_board_fanout, _fanout_board_mentions should be called."""
        with patch.object(handler, "_fanout_board_mentions") as mock_fanout:
            handler._handle_post_channel({"channel": "general", "text": "@all hello"})

        mock_fanout.assert_called_once_with("general", "@all hello")

    def test_post_channel_calls_fanout_when_flag_explicitly_false(
        self,
        handler: "ToolHandler",
        messenger: MagicMock,
    ) -> None:
        """_suppress_board_fanout=False should still call fanout normally."""
        handler._suppress_board_fanout = False

        with patch.object(handler, "_fanout_board_mentions") as mock_fanout:
            handler._handle_post_channel({"channel": "dev", "text": "@bob check"})

        mock_fanout.assert_called_once_with("dev", "@bob check")

    def test_post_channel_suppresses_fanout_when_flag_true(
        self,
        handler: "ToolHandler",
        messenger: MagicMock,
    ) -> None:
        """_suppress_board_fanout=True should skip _fanout_board_mentions."""
        handler._suppress_board_fanout = True

        with patch.object(handler, "_fanout_board_mentions") as mock_fanout:
            handler._handle_post_channel({"channel": "general", "text": "@all thanks"})

        mock_fanout.assert_not_called()

    def test_post_channel_still_posts_when_fanout_suppressed(
        self,
        handler: "ToolHandler",
        messenger: MagicMock,
    ) -> None:
        """Even with fanout suppressed, the channel post itself should still succeed."""
        handler._suppress_board_fanout = True

        result = handler._handle_post_channel({"channel": "general", "text": "hello"})

        messenger.post_channel.assert_called_once_with("general", "hello")
        assert "Posted to #general" in result

    def test_post_channel_logs_suppression(
        self,
        handler: "ToolHandler",
        messenger: MagicMock,
    ) -> None:
        """Suppressed fanout should be logged."""
        handler._suppress_board_fanout = True

        with patch("core.tooling.handler.logger") as mock_logger:
            handler._handle_post_channel({"channel": "ops", "text": "@all acknowledged"})

        # Look for the suppression log message
        log_messages = [
            call.args[0] if call.args else ""
            for call in mock_logger.info.call_args_list
        ]
        assert any("Suppressed board fanout" in msg for msg in log_messages), (
            f"Expected suppression log message, got: {log_messages}"
        )

    def test_suppress_flag_defaults_to_false_via_getattr(
        self,
        handler: "ToolHandler",
    ) -> None:
        """Handler should not have _suppress_board_fanout attribute by default;
        getattr(..., False) should return False."""
        # Verify the attribute doesn't exist on the handler by default
        assert not getattr(handler, "_suppress_board_fanout", False)


# ── Test 3: SQLite migration ─────────────────────────────────────


@pytest.mark.unit
class TestPraiseLoopPreventionMigration:
    """Verify _migrate_praise_loop_prevention_v1 creates migration records
    and updates the 4 section keys correctly."""

    @pytest.fixture
    def prompts_dir(self, tmp_path: Path) -> Path:
        """Create a prompts directory with test prompt files."""
        d = tmp_path / "prompts"
        d.mkdir()

        # Create the 4 section files that the migration reads
        (d / "communication_rules_a1.md").write_text(
            "# Communication Rules (A1)\nNew A1 communication rules with 1往復ルール",
            encoding="utf-8",
        )
        (d / "communication_rules.md").write_text(
            "# Communication Rules\nNew communication rules with 1往復ルール",
            encoding="utf-8",
        )
        (d / "messaging_a1.md").write_text(
            "# Messaging (A1)\nNew A1 messaging with Board投稿ルール",
            encoding="utf-8",
        )
        (d / "messaging.md").write_text(
            "# Messaging\nNew messaging with Board投稿ルール",
            encoding="utf-8",
        )
        return d

    @pytest.fixture
    def tool_store(self, tmp_path: Path) -> "ToolPromptStore":
        from core.tooling.prompt_db import ToolPromptStore
        return ToolPromptStore(tmp_path / "prompts.sqlite3")

    def test_migration_creates_migrations_table(
        self, tool_store: "ToolPromptStore", prompts_dir: Path
    ) -> None:
        """Migration should create the 'migrations' table in the DB."""
        from core.init import _migrate_praise_loop_prevention_v1

        _migrate_praise_loop_prevention_v1(tool_store, prompts_dir)

        conn = sqlite3.connect(str(tool_store._db_path))
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        conn.close()
        assert "migrations" in tables

    def test_migration_records_key_in_migrations_table(
        self, tool_store: "ToolPromptStore", prompts_dir: Path
    ) -> None:
        """Migration should insert a 'praise_loop_prevention_v1' record."""
        from core.init import _migrate_praise_loop_prevention_v1

        _migrate_praise_loop_prevention_v1(tool_store, prompts_dir)

        conn = sqlite3.connect(str(tool_store._db_path))
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT key, applied_at FROM migrations WHERE key = ?",
            ("praise_loop_prevention_v1",),
        ).fetchone()
        conn.close()

        assert row is not None
        assert row["key"] == "praise_loop_prevention_v1"
        assert row["applied_at"]  # non-empty ISO timestamp

    def test_migration_updates_all_four_sections(
        self, tool_store: "ToolPromptStore", prompts_dir: Path
    ) -> None:
        """Migration should update communication_rules_a1, communication_rules,
        messaging_a1, and messaging sections."""
        from core.init import _migrate_praise_loop_prevention_v1

        _migrate_praise_loop_prevention_v1(tool_store, prompts_dir)

        for key in (
            "communication_rules_a1",
            "communication_rules",
            "messaging_a1",
            "messaging",
        ):
            content = tool_store.get_section(key)
            assert content is not None, f"Section '{key}' should exist after migration"
            assert len(content) > 0, f"Section '{key}' should have non-empty content"

    def test_migration_sets_correct_conditions(
        self, tool_store: "ToolPromptStore", prompts_dir: Path
    ) -> None:
        """Migration should set correct conditions from SECTION_CONDITIONS."""
        from core.init import _migrate_praise_loop_prevention_v1
        from core.tooling.prompt_db import SECTION_CONDITIONS

        _migrate_praise_loop_prevention_v1(tool_store, prompts_dir)

        for key in (
            "communication_rules_a1",
            "communication_rules",
            "messaging_a1",
            "messaging",
        ):
            result = tool_store.get_section_with_condition(key)
            assert result is not None, f"Section '{key}' should exist"
            _content, condition = result
            expected_condition = SECTION_CONDITIONS.get(key)
            assert condition == expected_condition, (
                f"Section '{key}' condition: expected {expected_condition!r}, got {condition!r}"
            )

    def test_migration_content_matches_prompt_files(
        self, tool_store: "ToolPromptStore", prompts_dir: Path
    ) -> None:
        """Migration should load content from the prompt files verbatim (stripped)."""
        from core.init import _migrate_praise_loop_prevention_v1

        _migrate_praise_loop_prevention_v1(tool_store, prompts_dir)

        for key in (
            "communication_rules_a1",
            "communication_rules",
            "messaging_a1",
            "messaging",
        ):
            path = prompts_dir / f"{key}.md"
            expected = path.read_text(encoding="utf-8").strip()
            actual = tool_store.get_section(key)
            assert actual == expected, (
                f"Section '{key}' content mismatch"
            )

    def test_migration_is_idempotent(
        self, tool_store: "ToolPromptStore", prompts_dir: Path
    ) -> None:
        """Running migration twice should not fail or re-apply changes."""
        from core.init import _migrate_praise_loop_prevention_v1

        # Apply first time
        _migrate_praise_loop_prevention_v1(tool_store, prompts_dir)

        # Record the state after first application
        first_content = tool_store.get_section("communication_rules_a1")

        # Modify the prompt file to something different
        (prompts_dir / "communication_rules_a1.md").write_text(
            "# Modified content that should NOT be applied",
            encoding="utf-8",
        )

        # Apply again — should be a no-op
        _migrate_praise_loop_prevention_v1(tool_store, prompts_dir)

        # Content should remain unchanged from the first application
        second_content = tool_store.get_section("communication_rules_a1")
        assert second_content == first_content

    def test_migration_skips_missing_prompt_files(
        self, tool_store: "ToolPromptStore", tmp_path: Path
    ) -> None:
        """Migration should gracefully handle missing prompt files."""
        from core.init import _migrate_praise_loop_prevention_v1

        # Create prompts dir with only 2 of the 4 files
        partial_dir = tmp_path / "partial_prompts"
        partial_dir.mkdir()
        (partial_dir / "communication_rules_a1.md").write_text(
            "A1 rules content", encoding="utf-8",
        )
        (partial_dir / "messaging.md").write_text(
            "Messaging content", encoding="utf-8",
        )

        _migrate_praise_loop_prevention_v1(tool_store, partial_dir)

        # Present files should have their content
        assert tool_store.get_section("communication_rules_a1") == "A1 rules content"
        assert tool_store.get_section("messaging") == "Messaging content"

        # Missing files should not create sections
        assert tool_store.get_section("communication_rules") is None
        assert tool_store.get_section("messaging_a1") is None

    def test_migration_skips_empty_prompt_files(
        self, tool_store: "ToolPromptStore", tmp_path: Path
    ) -> None:
        """Migration should skip prompt files that are empty or whitespace-only."""
        from core.init import _migrate_praise_loop_prevention_v1

        empty_dir = tmp_path / "empty_prompts"
        empty_dir.mkdir()
        (empty_dir / "communication_rules_a1.md").write_text(
            "Valid content", encoding="utf-8",
        )
        (empty_dir / "communication_rules.md").write_text(
            "   \n  \n  ", encoding="utf-8",
        )
        (empty_dir / "messaging_a1.md").write_text(
            "", encoding="utf-8",
        )
        (empty_dir / "messaging.md").write_text(
            "Valid messaging", encoding="utf-8",
        )

        _migrate_praise_loop_prevention_v1(tool_store, empty_dir)

        assert tool_store.get_section("communication_rules_a1") == "Valid content"
        assert tool_store.get_section("communication_rules") is None  # whitespace-only
        assert tool_store.get_section("messaging_a1") is None  # empty
        assert tool_store.get_section("messaging") == "Valid messaging"
