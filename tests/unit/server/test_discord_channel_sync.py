# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Discord channel sync name conversion logic."""

from __future__ import annotations

from server.discord_channel_sync import _board_to_channel_name, _channel_to_board_name


class TestChannelToBoardName:
    def test_simple(self):
        assert _channel_to_board_name("general") == "general"

    def test_with_dashes(self):
        assert _channel_to_board_name("my-channel") == "my-channel"

    def test_spaces_to_dashes(self):
        assert _channel_to_board_name("my channel") == "my-channel"

    def test_uppercase_lowered(self):
        assert _channel_to_board_name("General") == "general"

    def test_special_chars(self):
        assert _channel_to_board_name("dev#ops!") == "dev-ops"

    def test_empty_string(self):
        assert _channel_to_board_name("") == "unnamed"

    def test_only_special_chars(self):
        assert _channel_to_board_name("!!!") == "unnamed"


class TestBoardToChannelName:
    def test_simple(self):
        assert _board_to_channel_name("general") == "general"

    def test_round_trip(self):
        name = "my-channel"
        assert _board_to_channel_name(_channel_to_board_name(name)) == name

    def test_empty(self):
        assert _board_to_channel_name("") == "unnamed"
