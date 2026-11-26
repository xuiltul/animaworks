"""Tests for core.notification.notifier — HumanNotifier and channel framework."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from core.config.models import (
    HumanNotificationConfig,
    NotificationChannelConfig,
)
from core.hub.timeline import list_items
from core.notification import notification_key_for
from core.notification.interactive import InteractionRequest
from core.notification.notifier import (
    _CHANNEL_REGISTRY,
    PRIORITY_LEVELS,
    HumanNotifier,
    NotificationChannel,
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
        interaction: Any | None = None,
    ) -> str:
        if self._fail:
            raise ConnectionError("Mock connection error")
        self.sent.append(
            {
                "subject": subject,
                "body": body,
                "priority": priority,
                "anima_name": anima_name,
            }
        )
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
        self,
        notifier_with_channels,
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
    @pytest.mark.parametrize("with_channel", [False, True])
    async def test_notify_records_timeline_key_and_callback(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch, with_channel: bool
    ):
        shared = tmp_path / "shared"
        shared.mkdir()
        monkeypatch.setattr(
            "core.config.models.load_config",
            lambda: SimpleNamespace(hub=SimpleNamespace(timeline_enabled=True, company="a")),
        )
        monkeypatch.setattr("core.notification.notifier._shared_dir", lambda: shared)
        interaction = InteractionRequest(
            callback_id="cb-1",
            anima_name="alice",
            category="approval",
            options=["yes", "no"],
            created_at=datetime.now(timezone.utc),
            approval_token="token",
        )
        notifier = HumanNotifier([MockChannel({})] if with_channel else [])

        await notifier.notify("Subject", "Body", interaction=interaction)

        items = list_items(shared, company="a")
        assert len(items) == 1
        item = items[0]
        assert item["kind"] == "notified"
        assert item["source"] == "system"
        assert item["from"] == "system"
        assert item["key"] == notification_key_for("Subject", "Body")
        assert item["callback_id"] == "cb-1"
        assert item["delivered"] is with_channel

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

            async def send(self, subject, body, priority="normal", *, anima_name="", interaction=None):
                return "test_registered: OK"

        assert "test_registered" in _CHANNEL_REGISTRY
        channel = create_channel(NotificationChannelConfig(type="test_registered", config={}))
        assert channel.channel_type == "test_registered"

        # Cleanup
        del _CHANNEL_REGISTRY["test_registered"]

    def test_create_unknown_channel_raises(self):
        with pytest.raises(ValueError, match="Unknown notification channel"):
            create_channel(NotificationChannelConfig(type="nonexistent", config={}))


# ── from_config ──────────────────────────────────────────────


class TestFromConfig:
    def test_from_config_creates_channels(self):
        config = HumanNotificationConfig(
            enabled=True,
            web_ui=False,
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
            web_ui=False,
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
            web_ui=False,
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
        config = HumanNotificationConfig(enabled=True, channels=[], web_ui=False)
        notifier = HumanNotifier.from_config(config)
        assert notifier.channel_count == 0

    def test_web_ui_channel_is_built_in(self):
        notifier = HumanNotifier.from_config(HumanNotificationConfig())
        assert notifier.channel_count == 1
        assert notifier.delivers_to_web is True

    def test_include_external_false_keeps_only_web(self):
        config = HumanNotificationConfig(
            enabled=False,
            channels=[
                NotificationChannelConfig(type="ntfy", config={"server_url": "https://ntfy.sh", "topic": "t"}),
            ],
        )
        notifier = HumanNotifier.from_config(config, include_external=False)
        assert notifier.channel_count == 1
        assert notifier.delivers_to_web is True

    def test_web_ui_off_and_no_channels_is_empty(self):
        notifier = HumanNotifier.from_config(HumanNotificationConfig(web_ui=False))
        assert notifier.channel_count == 0
        assert notifier.delivers_to_web is False


def test_has_external_channels_reflects_configured_channels():
    web_only = HumanNotifier.from_config(HumanNotificationConfig())
    assert web_only.has_external_channels is False
    with_ntfy = HumanNotifier.from_config(
        HumanNotificationConfig(
            enabled=True,
            channels=[NotificationChannelConfig(type="ntfy", config={"server_url": "https://ntfy.sh", "topic": "t"})],
        )
    )
    assert with_ntfy.has_external_channels is True


# ── Enclave egress for external notifications ───────────────


class _WebMock(MockChannel):
    @property
    def channel_type(self) -> str:
        return "web"


def _enclave_config(stages: list[dict[str, Any]]) -> SimpleNamespace:
    return SimpleNamespace(enclave=SimpleNamespace(enabled=True, egress={"stages": stages}))


class TestEnclaveEgress:
    @pytest.fixture(autouse=True)
    def _data_dir(self, tmp_path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr("core.paths.get_data_dir", lambda: tmp_path)
        monkeypatch.setattr("core.notification.notifier._record_timeline_notification", lambda *a, **k: None)
        return tmp_path

    @pytest.mark.asyncio
    async def test_external_text_is_masked_but_web_keeps_the_original(self, monkeypatch: pytest.MonkeyPatch):
        stages = [{"type": "regex_denylist", "patterns": [{"regex": "PATIENT-[0-9]+", "action": "redact"}]}]
        monkeypatch.setattr("core.config.models.load_config", lambda: _enclave_config(stages))
        web, ext = _WebMock({}), MockChannel({})
        status = await HumanNotifier([web, ext]).notify("about PATIENT-12", "see PATIENT-34 now")
        assert web.sent[0]["body"] == "see PATIENT-34 now"
        assert "PATIENT-34" not in ext.sent[0]["body"] and "PATIENT-12" not in ext.sent[0]["subject"]
        assert all("ERROR" not in s for s in status)

    @pytest.mark.asyncio
    async def test_blocked_text_never_reaches_external_channels(self, monkeypatch: pytest.MonkeyPatch):
        stages = [{"type": "regex_denylist", "patterns": [{"regex": "sk-ant-[A-Za-z0-9_-]{20,}", "action": "block"}]}]
        monkeypatch.setattr("core.config.models.load_config", lambda: _enclave_config(stages))
        web, ext = _WebMock({}), MockChannel({})
        status = await HumanNotifier([web, ext]).notify("key", "sk-ant-" + "a" * 30)
        assert ext.sent == []
        assert len(web.sent) == 1
        assert any("blocked by the enclave egress stages" in s for s in status)

    @pytest.mark.asyncio
    async def test_outside_an_enclave_text_is_unchanged(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            "core.config.models.load_config", lambda: SimpleNamespace(enclave=SimpleNamespace(enabled=False))
        )
        ext = MockChannel({})
        await HumanNotifier([ext]).notify("s", "PATIENT-1")
        assert ext.sent[0]["body"] == "PATIENT-1"
