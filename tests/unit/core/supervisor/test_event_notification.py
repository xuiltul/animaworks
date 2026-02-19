# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for event file notification system (Fix 2a).

Tests AnimaRunner._emit_event() (core/supervisor/runner.py) and
ProcessSupervisor._poll_anima_events() (core/supervisor/manager.py).
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.supervisor.runner import AnimaRunner
from core.supervisor.manager import ProcessSupervisor


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def runner(tmp_path: Path) -> AnimaRunner:
    """Create a AnimaRunner with tmp_path-based directories."""
    return AnimaRunner(
        anima_name="test",
        socket_path=Path("/tmp/test.sock"),
        animas_dir=tmp_path / "animas",
        shared_dir=tmp_path / "animaworks" / "shared",
    )


@pytest.fixture
def events_dir(runner: AnimaRunner) -> Path:
    """Return the expected events directory for the runner."""
    return runner.shared_dir.parent / "run" / "events" / runner.anima_name


@pytest.fixture
def supervisor(tmp_path: Path) -> MagicMock:
    """Create a mock ProcessSupervisor with run_dir set."""
    sup = MagicMock(spec=ProcessSupervisor)
    sup.run_dir = tmp_path / "run"
    sup._broadcast_event = AsyncMock()
    # Bind the real unbound method so we can call it with our mock
    sup._poll_anima_events = (
        ProcessSupervisor._poll_anima_events.__get__(sup, ProcessSupervisor)
    )
    return sup


# ── TestEmitEvent ─────────────────────────────────────────────────


class TestEmitEvent:
    """Tests for AnimaRunner._emit_event()."""

    def test_creates_event_file(self, runner: AnimaRunner, events_dir: Path) -> None:
        """_emit_event creates a .json file in the correct directory."""
        runner._emit_event("anima.heartbeat", {"name": "test"})

        json_files = list(events_dir.glob("*.json"))
        assert len(json_files) == 1
        assert json_files[0].suffix == ".json"

    def test_event_file_content(self, runner: AnimaRunner, events_dir: Path) -> None:
        """File contains correct JSON with event type and data."""
        data = {"name": "test", "result": {"summary": "ok"}}
        runner._emit_event("anima.heartbeat", data)

        json_files = list(events_dir.glob("*.json"))
        assert len(json_files) == 1

        content = json.loads(json_files[0].read_text(encoding="utf-8"))
        assert content["event"] == "anima.heartbeat"
        assert content["data"] == data

    def test_creates_directory_if_missing(
        self, runner: AnimaRunner, events_dir: Path
    ) -> None:
        """Events dir is created automatically if it doesn't exist."""
        assert not events_dir.exists()

        runner._emit_event("anima.cron", {"task": "cleanup"})

        assert events_dir.is_dir()
        assert len(list(events_dir.glob("*.json"))) == 1

    def test_atomic_write(self, runner: AnimaRunner, events_dir: Path) -> None:
        """No partial/temp files (dotfiles) remain after _emit_event."""
        runner._emit_event("anima.heartbeat", {"name": "test"})

        # Check there are no hidden temp files (.{filename})
        all_files = list(events_dir.iterdir())
        dotfiles = [f for f in all_files if f.name.startswith(".")]
        assert dotfiles == [], f"Temp files remain: {dotfiles}"

        # Only the final .json file should exist
        json_files = [f for f in all_files if f.suffix == ".json"]
        assert len(json_files) == 1


# ── TestPollAnimaEvents ──────────────────────────────────────────


class TestPollAnimaEvents:
    """Tests for ProcessSupervisor._poll_anima_events()."""

    @pytest.mark.asyncio
    async def test_reads_and_broadcasts_events(
        self, supervisor: MagicMock
    ) -> None:
        """Events are read and _broadcast_event is called with correct data."""
        events_dir = supervisor.run_dir / "events" / "alice"
        events_dir.mkdir(parents=True)

        event = {"event": "anima.heartbeat", "data": {"name": "alice"}}
        (events_dir / "0001.json").write_text(
            json.dumps(event), encoding="utf-8"
        )

        await supervisor._poll_anima_events()

        supervisor._broadcast_event.assert_awaited_once_with(
            "anima.heartbeat", {"name": "alice"}
        )

    @pytest.mark.asyncio
    async def test_deletes_processed_files(
        self, supervisor: MagicMock
    ) -> None:
        """Event files are deleted after successful processing."""
        events_dir = supervisor.run_dir / "events" / "bob"
        events_dir.mkdir(parents=True)

        event_file = events_dir / "0001.json"
        event_file.write_text(
            json.dumps({"event": "anima.cron", "data": {"task": "report"}}),
            encoding="utf-8",
        )
        assert event_file.exists()

        await supervisor._poll_anima_events()

        assert not event_file.exists()

    @pytest.mark.asyncio
    async def test_handles_corrupt_json(
        self, supervisor: MagicMock
    ) -> None:
        """Corrupt JSON files are removed without crash."""
        events_dir = supervisor.run_dir / "events" / "charlie"
        events_dir.mkdir(parents=True)

        corrupt_file = events_dir / "0001.json"
        corrupt_file.write_text("{broken json!!!", encoding="utf-8")

        # Should not raise
        await supervisor._poll_anima_events()

        # Corrupt file should be removed
        assert not corrupt_file.exists()
        # _broadcast_event should NOT have been called for corrupt data
        supervisor._broadcast_event.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_crash_on_missing_events_dir(
        self, supervisor: MagicMock
    ) -> None:
        """No error when the events/ directory doesn't exist."""
        assert not (supervisor.run_dir / "events").exists()

        # Should complete without raising
        await supervisor._poll_anima_events()

        supervisor._broadcast_event.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_processes_multiple_animas(
        self, supervisor: MagicMock
    ) -> None:
        """Events from different anima directories are all processed."""
        events_base = supervisor.run_dir / "events"

        # Create events for two different animas
        for name, event_type in [("alice", "anima.heartbeat"), ("bob", "anima.cron")]:
            anima_dir = events_base / name
            anima_dir.mkdir(parents=True)
            event = {"event": event_type, "data": {"name": name}}
            (anima_dir / "0001.json").write_text(
                json.dumps(event), encoding="utf-8"
            )

        await supervisor._poll_anima_events()

        # Both events should have been broadcast
        assert supervisor._broadcast_event.await_count == 2

        # Collect the call args
        calls = supervisor._broadcast_event.await_args_list
        event_types = {call.args[0] for call in calls}
        assert "anima.heartbeat" in event_types
        assert "anima.cron" in event_types

        # All event files should be deleted
        remaining = list(events_base.rglob("*.json"))
        assert remaining == []
