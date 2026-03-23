"""Unit tests for Slack mention resolution, annotations, intent detection, and inbox trigger logic."""

from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import MagicMock, patch

from core.config.schemas import UserAliasConfig
from core.schemas import EXTERNAL_PLATFORM_SOURCES, Message

# ── _resolve_slack_mentions ────────────────────────────────


class TestResolveSlackMentions:
    """Tests for :func:`server.slack_socket._resolve_slack_mentions`."""

    @patch("core.tools.slack.SlackClient")
    def test_returns_empty_for_empty_text(self, _mock_slack: MagicMock) -> None:
        from server.slack_socket import _resolve_slack_mentions, _user_name_cache

        _user_name_cache.clear()
        assert _resolve_slack_mentions("", "xoxb-token") == ""

    @patch("core.tools.slack.SlackClient")
    def test_returns_text_unchanged_without_mentions(self, _mock_slack: MagicMock) -> None:
        from server.slack_socket import _resolve_slack_mentions, _user_name_cache

        _user_name_cache.clear()
        assert _resolve_slack_mentions("hello world", "xoxb-token") == "hello world"

    @patch("core.tools.slack.SlackClient")
    def test_resolves_single_mention_via_api(self, mock_cls: MagicMock) -> None:
        from server.slack_socket import _resolve_slack_mentions, _user_name_cache

        _user_name_cache.clear()
        mock_cls.return_value.resolve_user_name.return_value = "DisplayName"
        out = _resolve_slack_mentions("hey <@U123ABC>", "xoxb-token")
        assert out == "hey @DisplayName"
        mock_cls.assert_called_once_with(token="xoxb-token")
        mock_cls.return_value.resolve_user_name.assert_called_once_with("U123ABC")

    @patch("core.tools.slack.SlackClient")
    def test_uses_cache_on_second_call(self, mock_cls: MagicMock) -> None:
        from server.slack_socket import _resolve_slack_mentions, _user_name_cache

        _user_name_cache.clear()
        mock_cls.return_value.resolve_user_name.return_value = "Alice"
        _resolve_slack_mentions("hi <@U123ABC>", "tok")
        _resolve_slack_mentions("<@U123ABC> again", "tok")
        assert mock_cls.return_value.resolve_user_name.call_count == 1

    @patch("core.tools.slack.SlackClient")
    def test_resolves_multiple_mentions(self, mock_cls: MagicMock) -> None:
        from server.slack_socket import _resolve_slack_mentions, _user_name_cache

        _user_name_cache.clear()

        def _resolve(uid: str) -> str:
            return {"U111": "One", "U222": "Two"}[uid]

        mock_cls.return_value.resolve_user_name.side_effect = _resolve
        out = _resolve_slack_mentions("<@U111> said hi to <@U222>", "tok")
        assert out == "@One said hi to @Two"
        assert mock_cls.return_value.resolve_user_name.call_count == 2
        called_uids = {c.args[0] for c in mock_cls.return_value.resolve_user_name.call_args_list}
        assert called_uids == {"U111", "U222"}

    @patch("core.tools.slack.SlackClient")
    def test_graceful_on_api_failure(self, mock_cls: MagicMock) -> None:
        from server.slack_socket import _resolve_slack_mentions, _user_name_cache

        _user_name_cache.clear()
        mock_cls.side_effect = RuntimeError("slack down")
        out = _resolve_slack_mentions("ping <@U999ZZZ>", "tok")
        assert out == "ping @U999ZZZ"

    @patch("core.tools.slack.SlackClient")
    def test_empty_token_still_converts_cached(self, mock_cls: MagicMock) -> None:
        from server.slack_socket import _resolve_slack_mentions, _user_name_cache

        _user_name_cache.clear()
        _user_name_cache["U123ABC"] = "CachedUser"
        out = _resolve_slack_mentions("yo <@U123ABC>", "")
        assert out == "yo @CachedUser"
        mock_cls.assert_not_called()


# ── _build_slack_annotation ────────────────────────────────


class TestBuildSlackAnnotation:
    """Tests for :func:`server.slack_socket._build_slack_annotation`."""

    def test_dm_channel(self) -> None:
        from server.slack_socket import _build_slack_annotation

        assert _build_slack_annotation("D01234567", False) == "[slack:DM]\n"
        assert _build_slack_annotation("D01234567", True) == "[slack:DM]\n"

    def test_channel_with_mention(self) -> None:
        from server.slack_socket import _build_slack_annotation

        assert _build_slack_annotation("C12345", True) == "[slack:channel — あなたがメンションされています]\n"

    def test_channel_without_mention(self) -> None:
        from server.slack_socket import _build_slack_annotation

        assert _build_slack_annotation("C12345", False) == "[slack:channel — あなたへの直接メンションはありません]\n"


# ── _detect_mention_intent ───────────────────────────────────


class TestDetectMentionIntent:
    """Tests for :func:`server.slack_socket._detect_mention_intent`."""

    def test_bot_mention_returns_question(self) -> None:
        from server.slack_socket import _detect_mention_intent

        assert _detect_mention_intent("hi <@U_BOT123>", "U_BOT123", set()) == "question"

    def test_alias_mention_returns_question(self) -> None:
        from server.slack_socket import _detect_mention_intent

        alias_ids = {"U_ALICE", "U_BOB"}
        assert _detect_mention_intent("hey <@U_ALICE>", "U_OTHER", alias_ids) == "question"

    def test_no_mention_returns_empty(self) -> None:
        from server.slack_socket import _detect_mention_intent

        assert _detect_mention_intent("plain text", "U_BOT", {"U_ALICE"}) == ""

    def test_empty_text_returns_empty(self) -> None:
        from server.slack_socket import _detect_mention_intent

        assert _detect_mention_intent("", "U_BOT", {"U_ALICE"}) == ""

    def test_empty_bot_id_with_alias_match(self) -> None:
        from server.slack_socket import _detect_mention_intent

        assert _detect_mention_intent("call <@U_ALIAS>", "", {"U_ALIAS"}) == "question"


# ── _load_alias_user_ids ─────────────────────────────────────


class TestLoadAliasUserIds:
    """Tests for :func:`server.slack_socket._load_alias_user_ids`."""

    @patch("server.slack_socket.load_config")
    def test_returns_slack_user_ids_from_config(self, mock_load: MagicMock) -> None:
        from server.slack_socket import _load_alias_user_ids

        cfg = MagicMock()
        cfg.external_messaging.user_aliases = {
            "alice": UserAliasConfig(slack_user_id="U111"),
            "bob": UserAliasConfig(slack_user_id="U222"),
        }
        mock_load.return_value = cfg
        assert _load_alias_user_ids() == {"U111", "U222"}

    @patch("server.slack_socket.load_config")
    def test_returns_empty_on_config_error(self, mock_load: MagicMock) -> None:
        from server.slack_socket import _load_alias_user_ids

        mock_load.side_effect = RuntimeError("no config")
        assert _load_alias_user_ids() == set()

    @patch("server.slack_socket.load_config")
    def test_ignores_empty_slack_user_ids(self, mock_load: MagicMock) -> None:
        from server.slack_socket import _load_alias_user_ids

        cfg = MagicMock()
        cfg.external_messaging.user_aliases = {
            "alice": UserAliasConfig(slack_user_id="U111"),
            "ghost": UserAliasConfig(slack_user_id=""),
        }
        mock_load.return_value = cfg
        assert _load_alias_user_ids() == {"U111"}


# ── External platform inbox trigger (shared condition) ───────


class TestExternalDirectedTrigger:
    """Boolean logic for immediate heartbeat vs defer (inbox_watcher / inbox_rate_limiter)."""

    def test_external_with_intent_triggers_immediately(self) -> None:
        messages = [
            Message(
                from_person="slack:U1",
                to_person="anima",
                content="hi",
                source="slack",
                intent="question",
            )
        ]
        has_external_directed = any(m.source in EXTERNAL_PLATFORM_SOURCES and m.intent for m in messages)
        assert has_external_directed is True

    def test_external_without_intent_defers(self) -> None:
        messages = [
            Message(
                from_person="slack:U1",
                to_person="anima",
                content="hi",
                source="slack",
                intent="",
            )
        ]
        has_external_directed = any(m.source in EXTERNAL_PLATFORM_SOURCES and m.intent for m in messages)
        assert has_external_directed is False

    def test_human_message_still_triggers(self) -> None:
        """Human source satisfies ``has_human`` so the defer branch is not taken."""
        inbox_messages = [
            Message(
                from_person="human:alice",
                to_person="anima",
                content="hi",
                source="human",
                intent="",
            )
        ]
        has_human = any(m.source == "human" for m in inbox_messages)
        has_external_directed = any(m.source in EXTERNAL_PLATFORM_SOURCES and m.intent for m in inbox_messages)
        actionable_intents = ("report", "question")
        has_actionable = any(m.intent in actionable_intents for m in inbox_messages)
        defer = not has_human and not has_external_directed and not has_actionable
        assert defer is False
