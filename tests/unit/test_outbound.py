"""Tests for core.outbound — outbound message routing."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from core.config.models import ExternalMessagingConfig, UserAliasConfig
from core.outbound import (
    ResolvedRecipient,
    _build_channel_order,
    _resolve_from_alias,
    resolve_recipient,
    send_external,
)


# ── Fixtures ──────────────────────────────────────────────


@pytest.fixture
def known_animas() -> set[str]:
    return {"sakura", "kotoha"}


@pytest.fixture
def config_with_aliases() -> ExternalMessagingConfig:
    return ExternalMessagingConfig(
        preferred_channel="slack",
        user_aliases={
            "user": UserAliasConfig(slack_user_id="U06MJKLV0TG"),
            "taka": UserAliasConfig(
                slack_user_id="U06MJKLV0TG",
                chatwork_room_id="12345",
            ),
        },
    )


@pytest.fixture
def empty_config() -> ExternalMessagingConfig:
    return ExternalMessagingConfig()


# ── TestResolveRecipient ─────────────────────────────────


class TestResolveRecipient:
    def test_exact_anima_name_match(self, known_animas, empty_config):
        result = resolve_recipient("sakura", known_animas, empty_config)
        assert result.is_internal is True
        assert result.name == "sakura"

    def test_exact_anima_case_sensitive(self, known_animas, empty_config):
        """'Sakura' (capital S) should not match step 1 (exact), falls to step 6."""
        result = resolve_recipient("Sakura", known_animas, empty_config)
        assert result.is_internal is True
        assert result.name == "sakura"  # normalized via step 6

    def test_user_alias_match(self, known_animas, config_with_aliases):
        result = resolve_recipient("user", known_animas, config_with_aliases)
        assert result.is_internal is False
        assert result.channel == "slack"
        assert result.slack_user_id == "U06MJKLV0TG"
        assert result.alias_used == "user"

    def test_user_alias_case_insensitive(self, known_animas, config_with_aliases):
        for variant in ("User", "USER", "uSeR"):
            result = resolve_recipient(variant, known_animas, config_with_aliases)
            assert result.is_internal is False
            assert result.channel == "slack"

    def test_slack_prefix(self, known_animas, empty_config):
        result = resolve_recipient("slack:U06MJKLV0TG", known_animas, empty_config)
        assert result.is_internal is False
        assert result.channel == "slack"
        assert result.slack_user_id == "U06MJKLV0TG"

    def test_slack_prefix_case_insensitive(self, known_animas, empty_config):
        result = resolve_recipient("Slack:u06mjklv0tg", known_animas, empty_config)
        assert result.is_internal is False
        assert result.channel == "slack"
        assert result.slack_user_id == "U06MJKLV0TG"

    def test_chatwork_prefix(self, known_animas, empty_config):
        result = resolve_recipient("chatwork:12345", known_animas, empty_config)
        assert result.is_internal is False
        assert result.channel == "chatwork"
        assert result.chatwork_room_id == "12345"

    def test_bare_slack_user_id(self, known_animas, empty_config):
        result = resolve_recipient("U06MJKLV0TG", known_animas, empty_config)
        assert result.is_internal is False
        assert result.channel == "slack"
        assert result.slack_user_id == "U06MJKLV0TG"

    def test_bare_slack_user_id_short_rejected(self, known_animas, empty_config):
        """Too short to be a Slack ID — falls through to unknown."""
        with pytest.raises(ValueError, match="Unknown recipient"):
            resolve_recipient("U12345", known_animas, empty_config)

    def test_case_insensitive_anima_fallback(self, known_animas, empty_config):
        result = resolve_recipient("SAKURA", known_animas, empty_config)
        assert result.is_internal is True
        assert result.name == "sakura"

    def test_unknown_recipient_raises(self, known_animas, empty_config):
        with pytest.raises(ValueError, match="Unknown recipient 'nobody'"):
            resolve_recipient("nobody", known_animas, empty_config)

    def test_unknown_recipient_error_lists_known(self, known_animas, empty_config):
        with pytest.raises(ValueError, match="Known animas"):
            resolve_recipient("xyz", known_animas, empty_config)

    def test_empty_recipient_raises(self, known_animas, empty_config):
        with pytest.raises(ValueError, match="empty"):
            resolve_recipient("", known_animas, empty_config)

    def test_whitespace_stripping(self, known_animas, config_with_aliases):
        result = resolve_recipient("  user  ", known_animas, config_with_aliases)
        assert result.is_internal is False
        assert result.channel == "slack"

    def test_anima_takes_priority_over_alias(self, config_with_aliases):
        """If 'sakura' is both a known anima and a user alias, anima wins."""
        animas = {"sakura"}
        config = ExternalMessagingConfig(
            user_aliases={"sakura": UserAliasConfig(slack_user_id="U999")},
        )
        result = resolve_recipient("sakura", animas, config)
        assert result.is_internal is True
        assert result.name == "sakura"


# ── TestResolveFromAlias ─────────────────────────────────


class TestResolveFromAlias:
    def test_preferred_slack(self):
        alias_cfg = UserAliasConfig(slack_user_id="U123", chatwork_room_id="R456")
        result = _resolve_from_alias("user", alias_cfg, "slack")
        assert result.channel == "slack"
        assert result.slack_user_id == "U123"

    def test_preferred_chatwork(self):
        alias_cfg = UserAliasConfig(slack_user_id="U123", chatwork_room_id="R456")
        result = _resolve_from_alias("user", alias_cfg, "chatwork")
        assert result.channel == "chatwork"
        assert result.chatwork_room_id == "R456"

    def test_fallback_when_preferred_unavailable(self):
        alias_cfg = UserAliasConfig(slack_user_id="U123")
        result = _resolve_from_alias("user", alias_cfg, "chatwork")
        assert result.channel == "slack"  # fallback

    def test_fallback_to_chatwork(self):
        alias_cfg = UserAliasConfig(chatwork_room_id="R456")
        result = _resolve_from_alias("user", alias_cfg, "slack")
        assert result.channel == "chatwork"

    def test_no_contact_info_raises(self):
        alias_cfg = UserAliasConfig()
        with pytest.raises(ValueError, match="no contact info"):
            _resolve_from_alias("user", alias_cfg, "slack")


# ── TestBuildChannelOrder ────────────────────────────────


class TestBuildChannelOrder:
    def test_single_channel(self):
        r = ResolvedRecipient(is_internal=False, name="x", channel="slack", slack_user_id="U1")
        assert _build_channel_order(r) == ["slack"]

    def test_fallback_added(self):
        r = ResolvedRecipient(
            is_internal=False, name="x", channel="slack",
            slack_user_id="U1", chatwork_room_id="R1",
        )
        assert _build_channel_order(r) == ["slack", "chatwork"]

    def test_chatwork_primary_with_slack_fallback(self):
        r = ResolvedRecipient(
            is_internal=False, name="x", channel="chatwork",
            slack_user_id="U1", chatwork_room_id="R1",
        )
        assert _build_channel_order(r) == ["chatwork", "slack"]

    def test_no_duplicate(self):
        r = ResolvedRecipient(
            is_internal=False, name="x", channel="slack",
            slack_user_id="U1",
        )
        order = _build_channel_order(r)
        assert order == ["slack"]
        assert len(order) == 1


# ── TestSendExternal ─────────────────────────────────────


class TestSendExternal:
    @patch("core.outbound._send_via_slack")
    def test_slack_success(self, mock_slack):
        mock_slack.return_value = json.dumps({"status": "sent", "channel": "slack"})
        r = ResolvedRecipient(
            is_internal=False, name="user", channel="slack", slack_user_id="U1",
        )
        result = send_external(r, "hello", sender_name="sakura")
        data = json.loads(result)
        assert data["status"] == "sent"
        mock_slack.assert_called_once_with("U1", "hello", "sakura")

    @patch("core.outbound._send_via_chatwork")
    def test_chatwork_success(self, mock_cw):
        mock_cw.return_value = json.dumps({"status": "sent", "channel": "chatwork"})
        r = ResolvedRecipient(
            is_internal=False, name="user", channel="chatwork", chatwork_room_id="R1",
        )
        result = send_external(r, "hello")
        data = json.loads(result)
        assert data["status"] == "sent"

    @patch("core.outbound._send_via_chatwork")
    @patch("core.outbound._send_via_slack")
    def test_slack_failure_fallback_chatwork(self, mock_slack, mock_cw):
        mock_slack.side_effect = RuntimeError("API error")
        mock_cw.return_value = json.dumps({"status": "sent", "channel": "chatwork"})
        r = ResolvedRecipient(
            is_internal=False, name="user", channel="slack",
            slack_user_id="U1", chatwork_room_id="R1",
        )
        result = send_external(r, "hello")
        data = json.loads(result)
        assert data["status"] == "sent"
        assert data["channel"] == "chatwork"

    @patch("core.outbound._send_via_slack")
    def test_all_channels_fail(self, mock_slack):
        mock_slack.side_effect = RuntimeError("fail")
        r = ResolvedRecipient(
            is_internal=False, name="user", channel="slack", slack_user_id="U1",
        )
        result = send_external(r, "hello")
        data = json.loads(result)
        assert data["status"] == "error"
        assert "DeliveryFailed" in data["error_type"]

    @patch("core.outbound._send_via_slack")
    def test_sender_name_prefix(self, mock_slack):
        mock_slack.return_value = json.dumps({"status": "sent"})
        r = ResolvedRecipient(
            is_internal=False, name="user", channel="slack", slack_user_id="U1",
        )
        send_external(r, "hello", sender_name="sakura")
        mock_slack.assert_called_once_with("U1", "hello", "sakura")

    @patch("core.outbound._send_via_slack")
    def test_sender_name_empty_no_prefix(self, mock_slack):
        mock_slack.return_value = json.dumps({"status": "sent"})
        r = ResolvedRecipient(
            is_internal=False, name="user", channel="slack", slack_user_id="U1",
        )
        send_external(r, "hello", sender_name="")
        mock_slack.assert_called_once_with("U1", "hello", "")


# ── TestUserAliasConfig ──────────────────────────────────


class TestUserAliasConfig:
    def test_default_values(self):
        cfg = UserAliasConfig()
        assert cfg.slack_user_id == ""
        assert cfg.chatwork_room_id == ""

    def test_round_trip(self):
        cfg = UserAliasConfig(slack_user_id="U123", chatwork_room_id="R456")
        data = cfg.model_dump()
        restored = UserAliasConfig(**data)
        assert restored.slack_user_id == "U123"
        assert restored.chatwork_room_id == "R456"

    def test_json_round_trip(self):
        cfg = UserAliasConfig(slack_user_id="U123")
        json_str = cfg.model_dump_json()
        restored = UserAliasConfig.model_validate_json(json_str)
        assert restored.slack_user_id == "U123"


# ── TestExternalMessagingConfig ──────────────────────────


class TestExternalMessagingConfig:
    def test_default_preferred_channel(self):
        cfg = ExternalMessagingConfig()
        assert cfg.preferred_channel == "slack"

    def test_default_user_aliases_empty(self):
        cfg = ExternalMessagingConfig()
        assert cfg.user_aliases == {}

    def test_with_user_aliases(self):
        cfg = ExternalMessagingConfig(
            user_aliases={"user": UserAliasConfig(slack_user_id="U1")},
        )
        assert "user" in cfg.user_aliases
        assert cfg.user_aliases["user"].slack_user_id == "U1"

    def test_backward_compatible(self):
        """Old config without preferred_channel/user_aliases should load with defaults."""
        data = {"slack": {"enabled": True}, "chatwork": {"enabled": False}}
        cfg = ExternalMessagingConfig(**data)
        assert cfg.preferred_channel == "slack"
        assert cfg.user_aliases == {}
        assert cfg.slack.enabled is True

    def test_full_round_trip(self):
        cfg = ExternalMessagingConfig(
            preferred_channel="chatwork",
            user_aliases={
                "user": UserAliasConfig(slack_user_id="U1", chatwork_room_id="R1"),
            },
        )
        data = cfg.model_dump()
        restored = ExternalMessagingConfig(**data)
        assert restored.preferred_channel == "chatwork"
        assert restored.user_aliases["user"].chatwork_room_id == "R1"
