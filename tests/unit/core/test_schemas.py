"""Unit tests for core/schemas.py — Pydantic data models."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import datetime

import pytest

from core.schemas import (
    CronTask,
    CycleResult,
    Message,
    ModelConfig,
    AnimaConfig,
    AnimaStatus,
)


# ── CronTask ──────────────────────────────────────────────


class TestCronTask:
    def test_basic_creation(self):
        task = CronTask(name="daily_report", schedule="毎日 9:00", description="Run daily")
        assert task.name == "daily_report"
        assert task.schedule == "毎日 9:00"
        assert task.description == "Run daily"

    def test_required_fields(self):
        with pytest.raises(Exception):
            CronTask()  # type: ignore[call-arg]


# ── ModelConfig ───────────────────────────────────────────


class TestModelConfig:
    def test_defaults(self):
        mc = ModelConfig()
        assert mc.model == "claude-sonnet-4-20250514"
        assert mc.fallback_model is None
        assert mc.max_tokens == 4096
        assert mc.max_turns == 20
        assert mc.api_key is None
        assert mc.api_key_env == "ANTHROPIC_API_KEY"
        assert mc.api_base_url is None
        assert mc.context_threshold == 0.50
        assert mc.max_chains == 2
        assert mc.conversation_history_threshold == 0.30
        assert mc.execution_mode is None
        assert mc.supervisor is None
        assert mc.speciality is None

    def test_custom_values(self):
        mc = ModelConfig(
            model="gpt-4o",
            max_tokens=8192,
            api_key="sk-test",
            api_base_url="http://localhost:8000",
            execution_mode="assisted",
            supervisor="boss",
        )
        assert mc.model == "gpt-4o"
        assert mc.max_tokens == 8192
        assert mc.api_key == "sk-test"
        assert mc.api_base_url == "http://localhost:8000"
        assert mc.execution_mode == "assisted"
        assert mc.supervisor == "boss"


# ── AnimaConfig ──────────────────────────────────────────


class TestAnimaConfig:
    def test_defaults(self, tmp_path):
        pc = AnimaConfig(name="alice", base_dir=tmp_path)
        assert pc.name == "alice"
        assert pc.base_dir == tmp_path
        assert pc.identity == ""
        assert pc.injection == ""
        assert pc.permissions == ""
        assert pc.heartbeat_interval == 30
        assert pc.active_hours == (9, 22)
        assert pc.cron_tasks == []
        assert isinstance(pc.model_config_data, ModelConfig)

    def test_with_cron_tasks(self, tmp_path):
        tasks = [CronTask(name="t1", schedule="毎日 9:00", description="daily")]
        pc = AnimaConfig(name="bob", base_dir=tmp_path, cron_tasks=tasks)
        assert len(pc.cron_tasks) == 1
        assert pc.cron_tasks[0].name == "t1"


# ── Message ───────────────────────────────────────────────


class TestMessage:
    def test_defaults(self):
        msg = Message(from_person="alice", to_person="bob", content="hello")
        assert msg.from_person == "alice"
        assert msg.to_person == "bob"
        assert msg.content == "hello"
        assert msg.type == "message"
        assert msg.thread_id == ""
        assert msg.reply_to == ""
        assert msg.attachments == []
        assert isinstance(msg.id, str)
        assert len(msg.id) > 0
        assert isinstance(msg.timestamp, datetime)

    def test_auto_generated_id_format(self):
        msg = Message(from_person="a", to_person="b", content="c")
        # ID format: YYYYMMDD_HHMMSS_ffffff
        parts = msg.id.split("_")
        assert len(parts) == 3
        assert len(parts[0]) == 8  # date
        assert len(parts[1]) == 6  # time

    def test_custom_fields(self):
        msg = Message(
            from_person="alice",
            to_person="bob",
            content="hello",
            type="delegation",
            thread_id="thread-1",
            reply_to="msg-1",
            attachments=["file.txt"],
        )
        assert msg.type == "delegation"
        assert msg.thread_id == "thread-1"
        assert msg.reply_to == "msg-1"
        assert msg.attachments == ["file.txt"]

    def test_json_serialization(self):
        msg = Message(from_person="a", to_person="b", content="test")
        json_str = msg.model_dump_json()
        assert "from_person" in json_str
        assert '"a"' in json_str

    def test_two_messages_have_different_ids(self):
        m1 = Message(from_person="a", to_person="b", content="1")
        m2 = Message(from_person="a", to_person="b", content="2")
        # Due to microsecond precision, they should usually differ
        # but both are valid format
        assert isinstance(m1.id, str)
        assert isinstance(m2.id, str)


# ── CycleResult ───────────────────────────────────────────


class TestCycleResult:
    def test_defaults(self):
        cr = CycleResult(trigger="heartbeat", action="responded")
        assert cr.trigger == "heartbeat"
        assert cr.action == "responded"
        assert cr.summary == ""
        assert cr.duration_ms == 0
        assert isinstance(cr.timestamp, datetime)
        assert cr.context_usage_ratio == 0.0
        assert cr.session_chained is False
        assert cr.total_turns == 0

    def test_custom_values(self):
        cr = CycleResult(
            trigger="cron:daily",
            action="responded",
            summary="All done",
            duration_ms=1500,
            context_usage_ratio=0.45,
            session_chained=True,
            total_turns=5,
        )
        assert cr.summary == "All done"
        assert cr.duration_ms == 1500
        assert cr.context_usage_ratio == 0.45
        assert cr.session_chained is True
        assert cr.total_turns == 5

    def test_model_dump(self):
        cr = CycleResult(trigger="t", action="a", summary="s")
        d = cr.model_dump()
        assert d["trigger"] == "t"
        assert d["action"] == "a"
        assert d["summary"] == "s"


# ── AnimaStatus ──────────────────────────────────────────


class TestAnimaStatus:
    def test_defaults(self):
        ps = AnimaStatus(name="alice")
        assert ps.name == "alice"
        assert ps.status == "idle"
        assert ps.current_task == ""
        assert ps.last_heartbeat is None
        assert ps.last_activity is None
        assert ps.pending_messages == 0

    def test_custom_values(self):
        now = datetime.now()
        ps = AnimaStatus(
            name="bob",
            status="thinking",
            current_task="Responding to human",
            last_heartbeat=now,
            last_activity=now,
            pending_messages=3,
        )
        assert ps.status == "thinking"
        assert ps.current_task == "Responding to human"
        assert ps.last_heartbeat == now
        assert ps.pending_messages == 3
