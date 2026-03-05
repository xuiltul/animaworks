from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for consolidation smart filtering and reflection extraction."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def temp_anima_dir(tmp_path: Path) -> Path:
    anima_dir = tmp_path / "test_anima"
    (anima_dir / "episodes").mkdir(parents=True)
    (anima_dir / "knowledge").mkdir(parents=True)
    return anima_dir


@pytest.fixture
def engine(temp_anima_dir: Path):
    from core.memory.consolidation import ConsolidationEngine

    return ConsolidationEngine(anima_dir=temp_anima_dir, anima_name="test_anima")


# ── _extract_reflections_from_episodes ───────────────────────────


class TestExtractReflections:
    """Tests for ConsolidationEngine._extract_reflections_from_episodes()."""

    def test_extracts_reflections(self, engine) -> None:
        text = (
            "## 2026-03-05 10:00\nSome episode content\n"
            "[REFLECTION]\n"
            "This is an important insight that I learned today about the system architecture and how it works.\n"
            "[/REFLECTION]\n"
            "More content\n"
        )
        result = engine._extract_reflections_from_episodes(text)
        assert "important insight" in result
        assert result.startswith("- ")

    def test_no_reflections(self, engine) -> None:
        text = "## 2026-03-05 10:00\nJust regular episode content without any reflections.\n"
        result = engine._extract_reflections_from_episodes(text)
        assert result == ""

    def test_short_entries_filtered(self, engine) -> None:
        text = (
            "[REFLECTION]\nToo short.\n[/REFLECTION]\n"
            "[REFLECTION]\n"
            "This is a sufficiently long reflection that should be included in the output for consolidation processing.\n"
            "[/REFLECTION]\n"
        )
        result = engine._extract_reflections_from_episodes(text)
        assert "Too short" not in result
        assert "sufficiently long reflection" in result

    def test_multiple_reflections(self, engine) -> None:
        text = (
            "[REFLECTION]\n"
            "First insight about how we handle error recovery in the production environment with automatic retries.\n"
            "[/REFLECTION]\n"
            "Some content\n"
            "[REFLECTION]\n"
            "Second insight about the deployment pipeline and how we can improve the CI/CD process for faster feedback.\n"
            "[/REFLECTION]\n"
        )
        result = engine._extract_reflections_from_episodes(text)
        assert "First insight" in result
        assert "Second insight" in result
        assert result.count("- ") == 2

    def test_empty_input(self, engine) -> None:
        assert engine._extract_reflections_from_episodes("") == ""

    def test_multiline_reflection(self, engine) -> None:
        text = (
            "[REFLECTION]\n"
            "Line 1 of the reflection about system design patterns.\n"
            "Line 2 continuing the thought about how we should structure modules.\n"
            "[/REFLECTION]\n"
        )
        result = engine._extract_reflections_from_episodes(text)
        assert "Line 1" in result
        assert "Line 2" in result


# ── _collect_activity_entries (smart filtering) ──────────────────


def _make_entry(type_: str, ts: str = "2026-03-05T10:00:00+09:00", **kwargs):
    from core.memory._activity_models import ActivityEntry
    return ActivityEntry(ts=ts, type=type_, **kwargs)


class TestCollectActivityEntries:
    """Tests for the smart-filtered _collect_activity_entries()."""

    def test_comm_events_prioritized(self, engine) -> None:
        entries = [
            _make_entry("message_received", ts="2026-03-05T10:00:00+09:00",
                        content="Hello from user", from_person="human"),
            _make_entry("response_sent", ts="2026-03-05T10:01:00+09:00",
                        content="Hello back"),
            _make_entry("tool_result", ts="2026-03-05T10:02:00+09:00",
                        tool="web_search", meta={"result_status": "ok", "result_bytes": 2048}),
            _make_entry("error", ts="2026-03-05T10:03:00+09:00",
                        content="Something failed"),
        ]

        with patch("core.memory.activity.ActivityLogger") as MockAL:
            mock_instance = MagicMock()
            mock_instance.recent.return_value = entries
            MockAL.return_value = mock_instance

            result = engine._collect_activity_entries(hours=24)

        assert "MSG< message_received" in result
        assert "RESP> response_sent" in result
        assert "ERR error" in result
        assert "TRES web_search" in result

    def test_tool_use_excluded(self, engine) -> None:
        entries = [
            _make_entry("message_received", ts="2026-03-05T10:00:00+09:00",
                        content="User message"),
            _make_entry("tool_result", ts="2026-03-05T10:01:00+09:00",
                        tool="search", meta={"result_status": "ok", "result_bytes": 100}),
        ]

        with patch("core.memory.activity.ActivityLogger") as MockAL:
            mock_instance = MagicMock()
            mock_instance.recent.return_value = entries
            MockAL.return_value = mock_instance

            result = engine._collect_activity_entries(hours=24)

        assert "TOOL" not in result or "TRES" in result

    def test_tool_result_fail_has_content(self, engine) -> None:
        entries = [
            _make_entry("tool_result", ts="2026-03-05T10:00:00+09:00",
                        tool="web_search", content="Connection timeout error",
                        meta={"result_status": "fail"}),
        ]

        with patch("core.memory.activity.ActivityLogger") as MockAL:
            mock_instance = MagicMock()
            mock_instance.recent.return_value = entries
            MockAL.return_value = mock_instance

            result = engine._collect_activity_entries(hours=24)

        assert "fail" in result
        assert "Connection timeout" in result

    def test_tool_result_ok_is_meta_only(self, engine) -> None:
        entries = [
            _make_entry("tool_result", ts="2026-03-05T10:00:00+09:00",
                        tool="web_search",
                        content="Very long search result content that should not appear",
                        meta={"result_status": "ok", "result_bytes": 5120, "result_count": 10}),
        ]

        with patch("core.memory.activity.ActivityLogger") as MockAL:
            mock_instance = MagicMock()
            mock_instance.recent.return_value = entries
            MockAL.return_value = mock_instance

            result = engine._collect_activity_entries(hours=24)

        assert "TRES web_search → ok" in result
        assert "5.0KB" in result
        assert "10件" in result
        assert "Very long search result" not in result

    def test_empty_entries(self, engine) -> None:
        with patch("core.memory.activity.ActivityLogger") as MockAL:
            mock_instance = MagicMock()
            mock_instance.recent.return_value = []
            MockAL.return_value = mock_instance

            result = engine._collect_activity_entries(hours=24)

        assert result == ""


# ── _format_tool_entries ─────────────────────────────────────────


class TestFormatToolEntries:
    """Tests for ConsolidationEngine._format_tool_entries()."""

    def test_ok_entries_meta_only(self) -> None:
        from core.memory.consolidation import ConsolidationEngine

        entries = [
            _make_entry("tool_result", ts="2026-03-05T10:00:00+09:00",
                        tool="search", meta={"result_status": "ok", "result_bytes": 3072, "result_count": 5}),
        ]
        lines = ConsolidationEngine._format_tool_entries(entries, 1000)
        assert len(lines) == 1
        assert "search → ok" in lines[0]
        assert "3.0KB" in lines[0]
        assert "5件" in lines[0]

    def test_fail_entries_have_content(self) -> None:
        from core.memory.consolidation import ConsolidationEngine

        entries = [
            _make_entry("tool_result", ts="2026-03-05T10:00:00+09:00",
                        tool="api_call", content="Auth failed 401",
                        meta={"result_status": "fail"}),
        ]
        lines = ConsolidationEngine._format_tool_entries(entries, 1000)
        assert "fail" in lines[0]
        assert "Auth failed" in lines[0]

    def test_budget_respected(self) -> None:
        from core.memory.consolidation import ConsolidationEngine

        entries = [
            _make_entry("tool_result", ts=f"2026-03-05T10:{i:02d}:00+09:00",
                        tool=f"tool_{i}", meta={"result_status": "ok", "result_bytes": 100})
            for i in range(50)
        ]
        lines = ConsolidationEngine._format_tool_entries(entries, 200)
        total = sum(len(l) + 1 for l in lines)
        assert total <= 201

    def test_small_bytes_shown_as_B(self) -> None:
        from core.memory.consolidation import ConsolidationEngine

        entries = [
            _make_entry("tool_result", ts="2026-03-05T10:00:00+09:00",
                        tool="ping", meta={"result_status": "ok", "result_bytes": 512}),
        ]
        lines = ConsolidationEngine._format_tool_entries(entries, 1000)
        assert "512B" in lines[0]
