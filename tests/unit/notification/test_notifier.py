"""Tests for core.notification.notifier — HumanNotifier and channel framework."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.config.models import (
    HumanNotificationConfig,
    NotificationChannelConfig,
)
from core.notification.notifier import (
    HumanNotifier,
    NotificationChannel,
    PRIORITY_LEVELS,
    _CHANNEL_REGISTRY,
    create_channel,
    register_channel,
)


# ── Test Channel Implementation ──────────────────────────────


class MockChannel(NotificationChannel):
    """A mock channel for testing."""

    def __init__(self, config: dict[str, Any], *, fail: bool = False) -> None:
        super().__init__(config)
        self._fail = fail
        self.sent: list[dict[str, str]] = []

    @property
    def channel_type(self) -> str:
        return "mock"

    async def send(
        self,
        subject: str,
        body: str,
        priority: str = "normal",
        *,
        anima_name: str = "",
    ) -> str:
        if self._fail:
            raise ConnectionError("Mock connection error")
        self.sent.append({
            "subject": subject,
            "body": body,
            "priority": priority,
            "anima_name": anima_name,
        })
        return "mock: OK"


# ── Priority Levels ──────────────────────────────────────────


class TestPriorityLevels:
    def test_all_levels_present(self):
        assert PRIORITY_LEVELS == ("low", "normal", "high", "urgent")


# ── HumanNotifier ────────────────────────────────────────────


class TestHumanNotifier:
    @pytest.fixture
    def notifier_with_channels(self) -> tuple[HumanNotifier, list[MockChannel]]:
        ch1 = MockChannel({})
        ch2 = MockChannel({})
        return HumanNotifier([ch1, ch2]), [ch1, ch2]

    def test_channel_count(self, notifier_with_channels):
        notifier, channels = notifier_with_channels
        assert notifier.channel_count == 2

    def test_empty_notifier(self):
        notifier = HumanNotifier([])
        assert notifier.channel_count == 0

    @pytest.mark.asyncio
    async def test_notify_sends_to_all_channels(self, notifier_with_channels):
        notifier, channels = notifier_with_channels
        results = await notifier.notify("Test Subject", "Test Body")
        assert len(results) == 2
        assert all(r == "mock: OK" for r in results)
        for ch in channels:
            assert len(ch.sent) == 1
            assert ch.sent[0]["subject"] == "Test Subject"
            assert ch.sent[0]["body"] == "Test Body"
            assert ch.sent[0]["priority"] == "normal"

    @pytest.mark.asyncio
    async def test_notify_with_priority(self, notifier_with_channels):
        notifier, channels = notifier_with_channels
        await notifier.notify("Alert", "Something broke", "urgent")
        for ch in channels:
            assert ch.sent[0]["priority"] == "urgent"

    @pytest.mark.asyncio
    async def test_notify_with_anima_name(self, notifier_with_channels):
        notifier, channels = notifier_with_channels
        await notifier.notify("Report", "All good", anima_name="alice")
        for ch in channels:
            assert ch.sent[0]["anima_name"] == "alice"

    @pytest.mark.asyncio
    async def test_notify_invalid_priority_defaults_to_normal(
        self, notifier_with_channels,
    ):
        notifier, channels = notifier_with_channels
        await notifier.notify("Test", "Body", "invalid_priority")
        for ch in channels:
            # Priority is normalized to "normal" by the notifier
            assert ch.sent[0]["priority"] == "normal"

    @pytest.mark.asyncio
    async def test_notify_no_channels(self):
        notifier = HumanNotifier([])
        results = await notifier.notify("Test", "Body")
        assert results == ["No notification channels configured"]

    @pytest.mark.asyncio
    async def test_notify_handles_channel_failure(self):
        ok_ch = MockChannel({})
        fail_ch = MockChannel({}, fail=True)
        notifier = HumanNotifier([ok_ch, fail_ch])
        results = await notifier.notify("Test", "Body")
        assert len(results) == 2
        assert results[0] == "mock: OK"
        assert "ERROR" in results[1]

    @pytest.mark.asyncio
    async def test_notify_all_channels_fail(self):
        fail1 = MockChannel({}, fail=True)
        fail2 = MockChannel({}, fail=True)
        notifier = HumanNotifier([fail1, fail2])
        results = await notifier.notify("Test", "Body")
        assert len(results) == 2
        assert all("ERROR" in r for r in results)


# ── Channel Registry ─────────────────────────────────────────


class TestChannelRegistry:
    def test_register_and_create(self):
        @register_channel("test_registered")
        class TestRegistered(NotificationChannel):
            @property
            def channel_type(self) -> str:
                return "test_registered"

            async def send(self, subject, body, priority="normal", *, anima_name=""):
                return "test_registered: OK"

        assert "test_registered" in _CHANNEL_REGISTRY
        channel = create_channel(
            NotificationChannelConfig(type="test_registered", config={})
        )
        assert channel.channel_type == "test_registered"

        # Cleanup
        del _CHANNEL_REGISTRY["test_registered"]

    def test_create_unknown_channel_raises(self):
        with pytest.raises(ValueError, match="Unknown notification channel"):
            create_channel(
                NotificationChannelConfig(type="nonexistent", config={})
            )


# ── from_config ──────────────────────────────────────────────


class TestFromConfig:
    def test_from_config_creates_channels(self):
        config = HumanNotificationConfig(
            enabled=True,
            channels=[
                NotificationChannelConfig(
                    type="ntfy",
                    config={"server_url": "https://ntfy.sh", "topic": "test"},
                ),
            ],
        )
        notifier = HumanNotifier.from_config(config)
        assert notifier.channel_count == 1

    def test_from_config_skips_disabled(self):
        config = HumanNotificationConfig(
            enabled=True,
            channels=[
                NotificationChannelConfig(
                    type="ntfy",
                    enabled=False,
                    config={"server_url": "https://ntfy.sh", "topic": "test"},
                ),
            ],
        )
        notifier = HumanNotifier.from_config(config)
        assert notifier.channel_count == 0

    def test_from_config_skips_unknown_type(self):
        config = HumanNotificationConfig(
            enabled=True,
            channels=[
                NotificationChannelConfig(
                    type="does_not_exist_xyz",
                    config={},
                ),
            ],
        )
        notifier = HumanNotifier.from_config(config)
        assert notifier.channel_count == 0

    def test_from_config_empty_channels(self):
        config = HumanNotificationConfig(enabled=True, channels=[])
        notifier = HumanNotifier.from_config(config)
        assert notifier.channel_count == 0
