"""Tests for human notification configuration models."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import pytest

from core.config.models import (
    AnimaWorksConfig,
    HumanNotificationConfig,
    NotificationChannelConfig,
)


class TestNotificationChannelConfig:
    def test_defaults(self):
        ch = NotificationChannelConfig(type="slack")
        assert ch.type == "slack"
        assert ch.enabled is True
        assert ch.config == {}

    def test_with_config(self):
        ch = NotificationChannelConfig(
            type="ntfy",
            config={"server_url": "https://ntfy.sh", "topic": "test"},
        )
        assert ch.config["server_url"] == "https://ntfy.sh"
        assert ch.config["topic"] == "test"

    def test_disabled(self):
        ch = NotificationChannelConfig(type="slack", enabled=False)
        assert ch.enabled is False


class TestHumanNotificationConfig:
    def test_defaults(self):
        config = HumanNotificationConfig()
        assert config.enabled is False
        assert config.channels == []

    def test_with_channels(self):
        config = HumanNotificationConfig(
            enabled=True,
            channels=[
                NotificationChannelConfig(
                    type="slack",
                    config={"webhook_url": "https://hooks.slack.com/test"},
                ),
                NotificationChannelConfig(
                    type="telegram",
                    config={
                        "bot_token_env": "TELEGRAM_BOT_TOKEN",
                        "chat_id": "123",
                    },
                ),
            ],
        )
        assert config.enabled is True
        assert len(config.channels) == 2
        assert config.channels[0].type == "slack"
        assert config.channels[1].type == "telegram"


class TestAnimaWorksConfigIntegration:
    def test_default_has_human_notification(self):
        config = AnimaWorksConfig()
        assert hasattr(config, "human_notification")
        assert isinstance(config.human_notification, HumanNotificationConfig)
        assert config.human_notification.enabled is False

    def test_roundtrip_json(self):
        config = AnimaWorksConfig(
            human_notification=HumanNotificationConfig(
                enabled=True,
                channels=[
                    NotificationChannelConfig(
                        type="slack",
                        config={"webhook_url": "https://hooks.slack.com/x"},
                    ),
                    NotificationChannelConfig(
                        type="line",
                        config={
                            "channel_access_token_env": "LINE_TOKEN",
                            "user_id": "U999",
                        },
                    ),
                    NotificationChannelConfig(
                        type="telegram",
                        config={
                            "bot_token_env": "TG_TOKEN",
                            "chat_id": "42",
                        },
                    ),
                    NotificationChannelConfig(
                        type="chatwork",
                        config={
                            "api_token_env": "CW_TOKEN",
                            "room_id": "100",
                        },
                    ),
                    NotificationChannelConfig(
                        type="ntfy",
                        config={
                            "server_url": "https://ntfy.sh",
                            "topic": "aw-reports",
                        },
                    ),
                ],
            ),
        )

        # Serialize to JSON and back
        data = json.loads(config.model_dump_json())
        restored = AnimaWorksConfig.model_validate(data)

        assert restored.human_notification.enabled is True
        assert len(restored.human_notification.channels) == 5
        types = [ch.type for ch in restored.human_notification.channels]
        assert types == ["slack", "line", "telegram", "chatwork", "ntfy"]

    def test_backward_compatible_without_field(self):
        """Config JSON without human_notification should load fine."""
        data = {
            "version": 1,
            "system": {"mode": "server"},
            "credentials": {"anthropic": {"api_key": ""}},
            "animas": {},
        }
        config = AnimaWorksConfig.model_validate(data)
        assert config.human_notification.enabled is False
        assert config.human_notification.channels == []
