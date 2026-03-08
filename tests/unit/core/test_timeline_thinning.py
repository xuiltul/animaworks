"""Unit tests for budget-aware timeline thinning in core/audit.py."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from core.audit import (
    _estimate_rendered_size,
    _is_command_cron,
    _render_event_lines,
    _thin_to_budget,
    generate_org_timeline,
)


def _make_entry(
    etype: str,
    ts: str = "2026-03-07T10:00:00+09:00",
    content: str = "",
    summary: str = "",
) -> dict:
    return {"type": etype, "ts": ts, "content": content, "summary": summary}


# ── _is_command_cron ─────────────────────────────────────


class TestIsCommandCron:
    def test_command_prefix_in_content(self):
        e = _make_entry("cron_executed", content="コマンド: PR新規コミット検出")
        assert _is_command_cron(e) is True

    def test_command_prefix_in_summary(self):
        e = _make_entry("cron_executed", summary="コマンド: Token Rotator")
        assert _is_command_cron(e) is True

    def test_llm_cron_not_command(self):
        e = _make_entry("cron_executed", content="データパイプライン監視を実行します。")
        assert _is_command_cron(e) is False

    def test_non_cron_event(self):
        e = _make_entry("heartbeat_end", content="コマンド: fake")
        assert _is_command_cron(e) is False


# ── _render_event_lines ──────────────────────────────────


class TestRenderEventLines:
    def test_basic_render(self):
        e = _make_entry("heartbeat_end", ts="2026-03-07T09:30:00+09:00", summary="Task check done")
        lines = _render_event_lines("alice", e)
        assert "[09:30] alice" in lines[0]
        assert "🔄" in lines[0]
        assert any("Task check done" in l for l in lines)

    def test_multiline_content_limited_to_3(self):
        content = "line1\nline2\nline3\nline4\nline5"
        e = _make_entry("response_sent", content=content)
        lines = _render_event_lines("bob", e)
        content_lines = [l for l in lines if l.startswith("  ")]
        assert len(content_lines) == 3


# ── _estimate_rendered_size ──────────────────────────────


class TestEstimateRenderedSize:
    def test_empty_list(self):
        assert _estimate_rendered_size([]) == 0

    def test_single_event(self):
        e = _make_entry("error", summary="Something failed")
        size = _estimate_rendered_size([("anima1", e)])
        assert size > 0

    def test_multiple_events_larger(self):
        e1 = _make_entry("error", summary="err1")
        e2 = _make_entry("message_sent", content="hello", summary="")
        size1 = _estimate_rendered_size([("a", e1)])
        size2 = _estimate_rendered_size([("a", e1), ("b", e2)])
        assert size2 > size1


# ── _thin_to_budget ──────────────────────────────────────


class TestThinToBudget:
    def _make_hb_events(self, n: int) -> list[tuple[str, dict]]:
        return [
            (
                "anima",
                _make_entry(
                    "heartbeat_end",
                    ts=f"2026-03-07T{h:02d}:{m:02d}:00+09:00",
                    summary=f"HB at {h:02d}:{m:02d}",
                ),
            )
            for i in range(n)
            for h, m in [(i // 2, (i % 2) * 30)]
        ]

    def test_all_fit(self):
        items = self._make_hb_events(5)
        kept, original = _thin_to_budget([], items, 100_000)
        assert len(kept) == 5
        assert original == 5

    def test_thinned_to_budget(self):
        items = self._make_hb_events(100)
        full_size = _estimate_rendered_size(items)
        budget = full_size // 3
        kept, original = _thin_to_budget([], items, budget)
        assert original == 100
        assert len(kept) < 100
        assert len(kept) >= 1

    def test_preserves_first_and_last(self):
        items = self._make_hb_events(50)
        full_size = _estimate_rendered_size(items)
        budget = full_size // 5
        kept, original = _thin_to_budget([], items, budget)
        assert kept[0] == items[0]
        assert kept[-1] == items[-1]

    def test_zero_budget(self):
        items = self._make_hb_events(10)
        kept, original = _thin_to_budget([], items, 0)
        assert kept == []
        assert original == 10

    def test_empty_thinnable(self):
        kept, original = _thin_to_budget([], [], 10_000)
        assert kept == []
        assert original == 0


# ── generate_org_timeline with max_chars ─────────────────


def _setup_anima_logs(tmp_path: Path, name: str, entries: list[dict]) -> Path:
    anima_dir = tmp_path / name
    anima_dir.mkdir(parents=True)
    (anima_dir / "status.json").write_text(
        json.dumps({"enabled": True, "model": "test", "role": "general"}),
        encoding="utf-8",
    )
    log_dir = anima_dir / "activity_log"
    log_dir.mkdir()
    path = log_dir / "2026-03-07.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    return anima_dir


class TestGenerateOrgTimelineMaxChars:
    def _make_entries(self) -> list[dict]:
        entries: list[dict] = []
        for h in range(24):
            entries.append(
                {
                    "type": "heartbeat_end",
                    "ts": f"2026-03-07T{h:02d}:00:00+09:00",
                    "summary": f"HB at {h:02d}:00",
                }
            )
            entries.append(
                {
                    "type": "cron_executed",
                    "ts": f"2026-03-07T{h:02d}:05:00+09:00",
                    "content": "コマンド: PR新規コミット検出",
                }
            )
            entries.append(
                {
                    "type": "message_sent",
                    "ts": f"2026-03-07T{h:02d}:10:00+09:00",
                    "content": f"Report at {h:02d}:10",
                    "to": "boss",
                }
            )
        entries.append(
            {
                "type": "error",
                "ts": "2026-03-07T15:30:00+09:00",
                "summary": "Something broke",
            }
        )
        return entries

    def test_no_max_chars_full_output(self, tmp_path):
        """max_chars=None returns all events including command crons."""
        entries = self._make_entries()
        _setup_anima_logs(tmp_path, "alice", entries)

        with patch("core.audit.get_animas_dir", return_value=tmp_path):
            result = generate_org_timeline("2026-03-07", max_chars=None)

        assert "コマンド:" in result
        assert result.count("alice 🔄") == 24

    def test_max_chars_removes_command_crons(self, tmp_path):
        """max_chars set should remove command-type crons."""
        entries = self._make_entries()
        _setup_anima_logs(tmp_path, "alice", entries)

        with patch("core.audit.get_animas_dir", return_value=tmp_path):
            result = generate_org_timeline("2026-03-07", max_chars=500_000)

        assert "コマンド:" not in result
        assert "cmd_cron" in result or "コマンドCron" in result

    def test_max_chars_preserves_protected_events(self, tmp_path):
        """Protected events (DM, error) are never thinned."""
        entries = self._make_entries()
        _setup_anima_logs(tmp_path, "alice", entries)

        with patch("core.audit.get_animas_dir", return_value=tmp_path):
            result = generate_org_timeline("2026-03-07", max_chars=5_000)

        assert "Something broke" in result
        assert result.count("📨") == 24

    def test_max_chars_thins_hb_events(self, tmp_path):
        """When budget is tight, HB events are thinned."""
        entries = self._make_entries()
        _setup_anima_logs(tmp_path, "alice", entries)

        with patch("core.audit.get_animas_dir", return_value=tmp_path):
            full = generate_org_timeline("2026-03-07", max_chars=None)
            tight = generate_org_timeline("2026-03-07", max_chars=1_500)

        full_hb_count = full.count("🔄")
        tight_hb_count = tight.count("🔄")
        assert tight_hb_count < full_hb_count

    def test_thinning_notice_shown(self, tmp_path):
        """Thinning notice appears when events are thinned."""
        entries = self._make_entries()
        _setup_anima_logs(tmp_path, "alice", entries)

        with patch("core.audit.get_animas_dir", return_value=tmp_path):
            result = generate_org_timeline("2026-03-07", max_chars=3_000)

        assert "24件中" in result or "of 24" in result or "HB/Cron" in result

    def test_multi_anima_thinning(self, tmp_path):
        """Multiple animas: all covered, HBs thinned proportionally."""
        for name in ("alice", "bob", "carol"):
            entries = [
                {
                    "type": "heartbeat_end",
                    "ts": f"2026-03-07T{h:02d}:00:00+09:00",
                    "summary": f"{name} HB",
                }
                for h in range(24)
            ]
            entries.append(
                {
                    "type": "message_sent",
                    "ts": "2026-03-07T12:00:00+09:00",
                    "content": f"{name} report",
                    "to": "boss",
                }
            )
            _setup_anima_logs(tmp_path, name, entries)

        with patch("core.audit.get_animas_dir", return_value=tmp_path):
            result = generate_org_timeline("2026-03-07", max_chars=5_000)

        for name in ("alice", "bob", "carol"):
            assert f"{name} report" in result
