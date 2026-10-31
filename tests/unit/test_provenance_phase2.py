from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for provenance Phase 2: input boundary origin tracking."""

import json
from pathlib import Path
from typing import Any

import pytest

from core.activity.logger import ActivityLogger
from core.activity.models import ActivityEntry
from core.schemas import Message
from core.trust import (
    ORIGIN_ANIMA,
    ORIGIN_EXTERNAL_PLATFORM,
    ORIGIN_HUMAN,
    ORIGIN_UNKNOWN,
)

# ── Message.origin_chain ──────────────────────────────────────


class TestMessageOriginChain:
    """Message schema extension with origin_chain field."""

    def test_default_origin_chain_is_empty_list(self) -> None:
        msg = Message(from_person="alice", to_person="bob", content="hi")
        assert msg.origin_chain == []

    def test_origin_chain_set_explicitly(self) -> None:
        msg = Message(
            from_person="slack:U123",
            to_person="anima1",
            content="hello",
            origin_chain=["external_platform"],
        )
        assert msg.origin_chain == ["external_platform"]

    def test_origin_chain_roundtrip_json(self) -> None:
        msg = Message(
            from_person="chatwork:456",
            to_person="anima2",
            content="test",
            source="chatwork",
            origin_chain=["external_platform"],
        )
        data = json.loads(msg.model_dump_json())
        restored = Message(**data)
        assert restored.origin_chain == ["external_platform"]

    def test_backward_compat_no_origin_chain_in_json(self) -> None:
        """Existing inbox JSON without origin_chain parses correctly."""
        raw = {
            "from_person": "anima1",
            "to_person": "anima2",
            "content": "hello",
            "source": "anima",
        }
        msg = Message(**raw)
        assert msg.origin_chain == []

    def test_origin_chain_multi_hop(self) -> None:
        msg = Message(
            from_person="relay",
            to_person="target",
            content="data",
            origin_chain=["external_platform", "anima"],
        )
        assert len(msg.origin_chain) == 2
        assert msg.origin_chain[0] == "external_platform"
        assert msg.origin_chain[1] == "anima"


# ── ActivityEntry origin/origin_chain ─────────────────────────


class TestActivityEntryOrigin:
    """ActivityEntry schema extension with origin and origin_chain."""

    def test_default_origin_empty(self) -> None:
        entry = ActivityEntry(ts="2026-02-28T10:00:00", type="message_received")
        assert entry.origin == ""
        assert entry.origin_chain == []

    def test_origin_set(self) -> None:
        entry = ActivityEntry(
            ts="2026-02-28T10:00:00",
            type="message_received",
            origin="external_platform",
            origin_chain=["external_platform"],
        )
        assert entry.origin == "external_platform"
        assert entry.origin_chain == ["external_platform"]

    def test_to_dict_omits_empty_origin(self) -> None:
        """Empty origin and origin_chain are excluded from to_dict()."""
        entry = ActivityEntry(
            ts="2026-02-28T10:00:00",
            type="message_received",
            content="test",
        )
        d = entry.to_dict()
        assert "origin" not in d
        assert "origin_chain" not in d

    def test_to_dict_includes_non_empty_origin(self) -> None:
        entry = ActivityEntry(
            ts="2026-02-28T10:00:00",
            type="message_received",
            content="test",
            origin="human",
            origin_chain=["human"],
        )
        d = entry.to_dict()
        assert d["origin"] == "human"
        assert d["origin_chain"] == ["human"]

    def test_to_dict_origin_chain_only(self) -> None:
        """origin_chain present but origin empty → only chain in dict."""
        entry = ActivityEntry(
            ts="2026-02-28T10:00:00",
            type="message_received",
            content="test",
            origin_chain=["anima", "external_platform"],
        )
        d = entry.to_dict()
        assert "origin" not in d
        assert d["origin_chain"] == ["anima", "external_platform"]

    def test_backward_compat_load_without_origin(self) -> None:
        """JSONL entry without origin fields loads correctly."""
        raw = {
            "ts": "2026-02-28T10:00:00",
            "type": "message_received",
            "content": "old entry",
            "from": "alice",
        }
        filtered = {k: v for k, v in raw.items() if k in ActivityEntry.__dataclass_fields__}
        if "from" in raw:
            filtered["from_person"] = raw["from"]
        entry = ActivityEntry(**filtered)
        assert entry.origin == ""
        assert entry.origin_chain == []


# ── ActivityLogger.log() with origin ──────────────────────────


class TestActivityLoggerOrigin:
    """ActivityLogger.log() origin argument propagation."""

    @pytest.fixture
    def anima_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / "animas" / "test-anima"
        (d / "activity_log").mkdir(parents=True)
        return d

    @pytest.fixture
    def logger(self, anima_dir: Path) -> ActivityLogger:
        return ActivityLogger(anima_dir)

    def test_log_without_origin_backward_compat(self, logger: ActivityLogger) -> None:
        entry = logger.log("message_received", content="hello")
        assert entry.origin == ""
        assert entry.origin_chain == []

    def test_log_with_origin_human(self, logger: ActivityLogger) -> None:
        entry = logger.log(
            "message_received",
            content="user input",
            origin="human",
        )
        assert entry.origin == "human"
        assert entry.origin_chain == []

    def test_log_with_origin_and_chain(self, logger: ActivityLogger) -> None:
        entry = logger.log(
            "message_received",
            content="relayed",
            origin="external_platform",
            origin_chain=["external_platform"],
        )
        assert entry.origin == "external_platform"
        assert entry.origin_chain == ["external_platform"]

    def test_origin_persisted_to_jsonl(
        self,
        logger: ActivityLogger,
        anima_dir: Path,
    ) -> None:
        logger.log(
            "message_received",
            content="from slack",
            from_person="slack:U123",
            origin="external_platform",
            origin_chain=["external_platform"],
        )

        log_dir = anima_dir / "activity_log"
        jsonl_files = list(log_dir.glob("*.jsonl"))
        assert len(jsonl_files) == 1

        line = jsonl_files[0].read_text(encoding="utf-8").strip()
        data = json.loads(line)
        assert data["origin"] == "external_platform"
        assert data["origin_chain"] == ["external_platform"]

    def test_origin_empty_not_persisted(
        self,
        logger: ActivityLogger,
        anima_dir: Path,
    ) -> None:
        """Empty origin fields are not written to JSONL."""
        logger.log("heartbeat_start", summary="check")

        log_dir = anima_dir / "activity_log"
        jsonl_files = list(log_dir.glob("*.jsonl"))
        assert len(jsonl_files) == 1

        line = jsonl_files[0].read_text(encoding="utf-8").strip()
        data = json.loads(line)
        assert "origin" not in data
        assert "origin_chain" not in data

    def test_origin_survives_load_entries(
        self,
        logger: ActivityLogger,
    ) -> None:
        """Origin fields are restored when loading entries from JSONL."""
        logger.log(
            "message_received",
            content="external msg",
            origin="external_platform",
            origin_chain=["external_platform"],
        )
        entries = logger._load_entries(days=1)
        assert len(entries) == 1
        assert entries[0].origin == "external_platform"
        assert entries[0].origin_chain == ["external_platform"]

    def test_old_entries_without_origin_load_ok(
        self,
        anima_dir: Path,
    ) -> None:
        """Pre-Phase 2 entries without origin fields load with defaults."""
        from core.time_utils import now_jst

        log_dir = anima_dir / "activity_log"
        today = now_jst().strftime("%Y-%m-%d")
        entry = json.dumps(
            {
                "ts": now_jst().isoformat(),
                "type": "message_received",
                "content": "legacy entry",
                "from": "alice",
            }
        )
        (log_dir / f"{today}.jsonl").write_text(entry + "\n", encoding="utf-8")

        al = ActivityLogger(anima_dir)
        entries = al._load_entries(days=1)
        assert len(entries) == 1
        assert entries[0].origin == ""
        assert entries[0].origin_chain == []


# ── Messenger.receive_external() origin_chain ────────────────


class TestReceiveExternalOrigin:
    """Messenger.receive_external() sets origin_chain."""

    @pytest.fixture
    def messenger(self, tmp_path: Path) -> Any:
        from core.messaging.messenger import Messenger

        shared = tmp_path / "shared"
        shared.mkdir(parents=True)
        return Messenger(shared, "test-anima")

    def test_receive_external_sets_origin_chain(self, messenger: Any) -> None:
        msg = messenger.receive_external(
            content="hello from slack",
            source="slack",
            external_user_id="U123",
        )
        assert msg.origin_chain == [ORIGIN_EXTERNAL_PLATFORM]

    def test_receive_external_chatwork(self, messenger: Any) -> None:
        msg = messenger.receive_external(
            content="hello from chatwork",
            source="chatwork",
            external_channel_id="R456",
        )
        assert msg.origin_chain == [ORIGIN_EXTERNAL_PLATFORM]

    def test_receive_external_persists_origin_chain(
        self,
        messenger: Any,
    ) -> None:
        """origin_chain survives JSON serialization to inbox file."""
        msg = messenger.receive_external(
            content="test",
            source="slack",
            external_user_id="U789",
        )
        inbox_file = messenger.inbox_dir / f"{msg.id}.json"
        assert inbox_file.exists()
        data = json.loads(inbox_file.read_text(encoding="utf-8"))
        assert data["origin_chain"] == ["external_platform"]

    def test_receive_external_message_source_preserved(
        self,
        messenger: Any,
    ) -> None:
        """source field retains platform name (not origin category)."""
        msg = messenger.receive_external(
            content="test",
            source="slack",
        )
        assert msg.source == "slack"


# ── _SOURCE_TO_ORIGIN mapping ────────────────────────────────


class TestSourceToOriginMapping:
    """_SOURCE_TO_ORIGIN in _anima_inbox.py maps correctly."""

    def test_slack_maps_to_external_platform(self) -> None:
        from core.anima.inbox import _SOURCE_TO_ORIGIN

        assert _SOURCE_TO_ORIGIN["slack"] == ORIGIN_EXTERNAL_PLATFORM

    def test_chatwork_maps_to_external_platform(self) -> None:
        from core.anima.inbox import _SOURCE_TO_ORIGIN

        assert _SOURCE_TO_ORIGIN["chatwork"] == ORIGIN_EXTERNAL_PLATFORM

    def test_human_maps_to_human(self) -> None:
        from core.anima.inbox import _SOURCE_TO_ORIGIN

        assert _SOURCE_TO_ORIGIN["human"] == ORIGIN_HUMAN

    def test_anima_maps_to_anima(self) -> None:
        from core.anima.inbox import _SOURCE_TO_ORIGIN

        assert _SOURCE_TO_ORIGIN["anima"] == ORIGIN_ANIMA

    def test_unknown_source_fallback(self) -> None:
        from core.anima.inbox import _SOURCE_TO_ORIGIN

        assert _SOURCE_TO_ORIGIN.get("future_platform", ORIGIN_UNKNOWN) == ORIGIN_UNKNOWN


# ── Template files ────────────────────────────────────────────


class TestToolDataInterpretationTemplate:
    """Verify trust-boundary / origin_chain rules are in the prompt templates."""

    def test_ja_template_has_origin_chain_rule(self) -> None:
        path = Path("templates/ja/prompts/behavior_rules.md")
        content = path.read_text(encoding="utf-8")
        assert "origin_chain" in content
        assert 'trust="untrusted"' in content

    def test_en_template_has_origin_chain_rule(self) -> None:
        path = Path("templates/en/prompts/behavior_rules.md")
        content = path.read_text(encoding="utf-8")
        assert "origin_chain" in content
        assert 'trust="untrusted"' in content
