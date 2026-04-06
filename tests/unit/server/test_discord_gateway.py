# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Discord gateway routing logic."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from server.discord_gateway import (
    DiscordGatewayManager,
    _build_discord_annotation,
    _is_duplicate_id,
)


# ── _build_discord_annotation ────────────────────────────


class TestBuildDiscordAnnotation:
    def test_dm(self):
        result = _build_discord_annotation(is_dm=True, has_mention=False)
        assert "[discord:DM]" in result

    def test_channel_with_mention(self):
        result = _build_discord_annotation(is_dm=False, has_mention=True)
        assert "メンションされています" in result

    def test_channel_without_mention(self):
        result = _build_discord_annotation(is_dm=False, has_mention=False)
        assert "直接メンションはありません" in result


# ── _is_duplicate_id ─────────────────────────────────────


class TestIsDuplicateId:
    def test_first_call_not_duplicate(self):
        assert _is_duplicate_id("unique_test_id_001") is False

    def test_second_call_is_duplicate(self):
        _is_duplicate_id("unique_test_id_002")
        assert _is_duplicate_id("unique_test_id_002") is True


# ── DiscordGatewayManager routing ────────────────────────


class TestDiscordGatewayManagerRouting:
    @pytest.fixture
    def _mock_config(self, monkeypatch):
        """Provide a mock config for routing tests — must be applied before manager."""
        mock_cfg = MagicMock()
        mock_cfg.animas = {
            "sakura": MagicMock(aliases=["さくら"]),
            "kotoha": MagicMock(aliases=[]),
        }
        mock_cfg.external_messaging.discord.channel_members = {
            "ch1": ["sakura", "kotoha"],
            "ch2": ["kotoha"],
            "dm-sakura": ["sakura"],
        }
        mock_cfg.external_messaging.discord.default_anima = "sakura"
        mock_cfg.external_messaging.discord.board_mapping = {}

        monkeypatch.setattr(
            "server.discord_gateway.load_config",
            lambda: mock_cfg,
        )
        return mock_cfg

    @pytest.fixture
    def manager(self, _mock_config) -> DiscordGatewayManager:
        mgr = DiscordGatewayManager()
        mgr._build_anima_patterns()
        return mgr

    def test_detect_anima_by_name_in_text(self, manager: DiscordGatewayManager, _mock_config):
        discord_cfg = _mock_config.external_messaging.discord
        result = manager._detect_target_anima("sakuraに聞いて", "ch1", discord_cfg)
        assert result == "sakura"

    def test_detect_anima_by_japanese_alias(self, manager: DiscordGatewayManager, _mock_config):
        discord_cfg = _mock_config.external_messaging.discord
        result = manager._detect_target_anima("今日はさくらに連絡", "ch1", discord_cfg)
        assert result == "sakura"

    def test_detect_anima_single_member_channel(self, manager: DiscordGatewayManager, _mock_config):
        discord_cfg = _mock_config.external_messaging.discord
        result = manager._detect_target_anima("何か教えて", "dm-sakura", discord_cfg)
        assert result == "sakura"

    def test_detect_anima_no_match(self, manager: DiscordGatewayManager, _mock_config):
        discord_cfg = _mock_config.external_messaging.discord
        result = manager._detect_target_anima("こんにちは", "ch1", discord_cfg)
        assert result is None

    def test_is_anima_in_channel_member(self, _mock_config):
        discord_cfg = _mock_config.external_messaging.discord
        assert DiscordGatewayManager._is_anima_in_channel("sakura", "ch1", discord_cfg) is True

    def test_is_anima_not_in_channel(self, _mock_config):
        discord_cfg = _mock_config.external_messaging.discord
        assert DiscordGatewayManager._is_anima_in_channel("sakura", "ch2", discord_cfg) is False

    def test_is_anima_in_channel_no_config(self, _mock_config):
        """No membership config → allow all."""
        discord_cfg = _mock_config.external_messaging.discord
        assert DiscordGatewayManager._is_anima_in_channel("sakura", "unknown_ch", discord_cfg) is True

    def test_anima_name_regex_case_insensitive(self, manager: DiscordGatewayManager, _mock_config):
        discord_cfg = _mock_config.external_messaging.discord
        result = manager._detect_target_anima("Sakuraに聞いて", "ch1", discord_cfg)
        assert result == "sakura"
