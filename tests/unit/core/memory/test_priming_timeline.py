# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.config.schemas import HubConfig
from core.hub.timeline import advance_timeline_read, append_item
from core.i18n import t
from core.memory.priming.outbound import collect_pending_human_notifications
from core.paths import get_shared_dir
from core.time_utils import now_iso


def _set_hub(monkeypatch: pytest.MonkeyPatch, **kwargs) -> None:
    hub = HubConfig(**kwargs)
    monkeypatch.setattr("core.config.models.load_config", lambda: SimpleNamespace(hub=hub))


def _set_company(anima_dir: Path, company: str) -> None:
    (anima_dir / "status.json").write_text(json.dumps({"company": company}), encoding="utf-8")


@pytest.fixture
def anima_dir(tmp_path: Path) -> Path:
    path = tmp_path / "animas" / "rin"
    path.mkdir(parents=True)
    return path


@pytest.mark.asyncio
async def test_timeline_replaces_activity_notifications(
    anima_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_company(anima_dir, "a")
    _set_hub(monkeypatch, timeline_enabled=True, company="a")
    item = append_item(
        get_shared_dir(),
        from_name="alice",
        company="a",
        kind="done",
        text="Fixed the deployment",
        source="anima",
    )

    text = await collect_pending_human_notifications(anima_dir, channel="chat")

    assert t("priming.timeline_header") in text
    assert "done" in text
    assert "alice" in text
    assert "Fixed the deployment" in text
    assert t("priming.timeline_unread_mark") in text
    assert t("priming.timeline_update_mark") not in text

    advance_timeline_read(get_shared_dir(), "human:alice", item["id"])
    read_text = await collect_pending_human_notifications(anima_dir, channel="chat")
    assert t("priming.timeline_unread_mark") not in read_text

    append_item(
        get_shared_dir(),
        from_name="alice",
        company="a",
        kind="decided",
        text="Will roll back",
        source="anima",
        about=item["id"],
    )
    update_text = await collect_pending_human_notifications(anima_dir, channel="chat")
    assert t("priming.timeline_update_mark") in update_text


@pytest.mark.asyncio
async def test_timeline_requires_matching_company(
    anima_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_company(anima_dir, "b")
    _set_hub(monkeypatch, timeline_enabled=True, company="a")
    append_item(
        get_shared_dir(),
        from_name="alice",
        company="a",
        kind="done",
        text="Other company work",
        source="anima",
    )

    assert await collect_pending_human_notifications(anima_dir, channel="chat") == ""


@pytest.mark.asyncio
async def test_timeline_only_for_chat_and_heartbeat(
    anima_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_company(anima_dir, "a")
    _set_hub(monkeypatch, timeline_enabled=True, company="a")
    append_item(
        get_shared_dir(),
        from_name="alice",
        company="a",
        kind="done",
        text="Fixed the deployment",
        source="anima",
    )

    assert await collect_pending_human_notifications(anima_dir, channel="task") == ""


@pytest.mark.asyncio
async def test_disabled_timeline_falls_back_to_activity(
    anima_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_company(anima_dir, "a")
    _set_hub(monkeypatch, company="a")
    activity_dir = anima_dir / "activity_log"
    activity_dir.mkdir()
    ts = now_iso()
    entry = {
        "ts": ts,
        "type": "human_notify",
        "content": "Activity notification",
        "via": "slack",
    }
    (activity_dir / f"{ts[:10]}.jsonl").write_text(json.dumps(entry), encoding="utf-8")

    text = await collect_pending_human_notifications(anima_dir, channel="chat")

    assert t("priming.pending_human_notifications_header") in text
    assert "Activity notification" in text
