# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from core.config.schemas import HubConfig
from core.hub.timeline import list_items
from core.notification.notifier import HumanNotifier, NotificationChannel
from core.paths import get_animas_dir, get_shared_dir


class MockChannel(NotificationChannel):
    def __init__(self, *, fail: bool = False) -> None:
        super().__init__({})
        self.fail = fail

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
        if self.fail:
            raise ConnectionError("boom")
        return "mock: OK"


def _set_company(monkeypatch: pytest.MonkeyPatch, company: str) -> None:
    anima_dir = get_animas_dir() / "alice"
    anima_dir.mkdir(parents=True, exist_ok=True)
    (anima_dir / "status.json").write_text(json.dumps({"company": company}), encoding="utf-8")


def _set_hub(monkeypatch: pytest.MonkeyPatch, **kwargs) -> None:
    hub = HubConfig(**kwargs)
    monkeypatch.setattr("core.config.models.load_config", lambda: SimpleNamespace(hub=hub))


@pytest.mark.asyncio
async def test_notify_records_timeline_item(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_company(monkeypatch, "a")
    _set_hub(monkeypatch, timeline_enabled=True, company="a")

    notifier = HumanNotifier([MockChannel()])
    status = await notifier.notify("Subject", "Body", "high", anima_name="alice")

    assert status == ["mock: OK"]
    items = list_items(get_shared_dir(), company="a")
    assert len(items) == 1
    item = items[0]
    assert item["from"] == "alice"
    assert item["kind"] == "notified"
    assert item["text"] == "Subject: Body"
    assert item["priority"] == "high"
    assert item["urgent"] is True
    assert item["delivered"] is True
    assert item["source"] == "anima"


@pytest.mark.asyncio
async def test_notify_records_failed_delivery(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_company(monkeypatch, "a")
    _set_hub(monkeypatch, timeline_enabled=True, company="a")

    notifier = HumanNotifier([MockChannel(fail=True)])
    status = await notifier.notify("Subject", "Body", anima_name="alice")

    assert "ERROR" in status[0]
    items = list_items(get_shared_dir(), company="a")
    assert items[0]["delivered"] is False


@pytest.mark.asyncio
async def test_notify_without_channels_records_failed_delivery(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_company(monkeypatch, "a")
    _set_hub(monkeypatch, timeline_enabled=True, company="a")

    status = await HumanNotifier([]).notify("Subject", "Body", anima_name="alice")

    assert status == ["No notification channels configured"]
    items = list_items(get_shared_dir(), company="a")
    assert items[0]["delivered"] is False


@pytest.mark.asyncio
async def test_notify_records_system_when_anima_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_hub(monkeypatch, timeline_enabled=True, company="a")

    await HumanNotifier([MockChannel()]).notify("Subject", "Body")

    items = list_items(get_shared_dir(), company="a")
    assert items[0]["from"] == "system"
    assert items[0]["source"] == "system"


@pytest.mark.asyncio
async def test_notify_skips_other_company(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_company(monkeypatch, "b")
    _set_hub(monkeypatch, timeline_enabled=True, company="a")

    await HumanNotifier([MockChannel()]).notify("Subject", "Body", anima_name="alice")

    assert list_items(get_shared_dir(), company="a") == []
    assert list_items(get_shared_dir(), company="b") == []


@pytest.mark.asyncio
async def test_notify_skips_when_timeline_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_company(monkeypatch, "a")
    _set_hub(monkeypatch, company="a")

    await HumanNotifier([MockChannel()]).notify("Subject", "Body", anima_name="alice")

    assert list_items(get_shared_dir(), company="a") == []


@pytest.mark.asyncio
async def test_recording_failure_does_not_break_notification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_company(monkeypatch, "a")
    _set_hub(monkeypatch, timeline_enabled=True, company="a")

    def _raise(*args, **kwargs):
        raise OSError("disk full")

    from core.hub import timeline

    monkeypatch.setattr(timeline, "append_item", _raise)

    status = await HumanNotifier([MockChannel()]).notify("Subject", "Body", anima_name="alice")
    assert status == ["mock: OK"]
