"""Unit tests for audit_subordinate supervisor tool."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

from core.tooling.handler import ToolHandler


def _make_handler(tmp_path: Path, anima_name: str = "sakura") -> ToolHandler:
    """Create a ToolHandler with minimal mocked dependencies."""
    anima_dir = tmp_path / "animas" / anima_name
    anima_dir.mkdir(parents=True)
    (anima_dir / "permissions.md").write_text("", encoding="utf-8")
    (anima_dir / "activity_log").mkdir(parents=True, exist_ok=True)

    memory = MagicMock()
    memory.read_permissions.return_value = ""

    handler = ToolHandler(
        anima_dir=anima_dir,
        memory=memory,
        messenger=None,
        tool_registry=[],
    )
    return handler


def _setup_subordinate(
    tmp_path: Path,
    name: str,
    supervisor: str,
    *,
    enabled: bool = True,
    model: str = "claude-sonnet-4-6",
) -> Path:
    """Create a subordinate anima directory with status.json."""
    anima_dir = tmp_path / "animas" / name
    anima_dir.mkdir(parents=True, exist_ok=True)
    (anima_dir / "activity_log").mkdir(parents=True, exist_ok=True)
    (anima_dir / "state").mkdir(parents=True, exist_ok=True)
    status = {
        "enabled": enabled,
        "supervisor": supervisor,
        "model": model,
        "role": "general",
    }
    (anima_dir / "status.json").write_text(
        json.dumps(status, indent=2),
        encoding="utf-8",
    )
    return anima_dir


def _mock_config(animas: dict[str, dict]) -> MagicMock:
    """Build a mock config with AnimaModelConfig entries."""
    from core.config.models import AnimaModelConfig

    config = MagicMock()
    config.animas = {name: AnimaModelConfig(**fields) for name, fields in animas.items()}
    return config


def _write_activity(anima_dir: Path, entries: list[dict]) -> None:
    """Write activity entries to today's JSONL log."""
    from core.time_utils import now_jst

    log_dir = anima_dir / "activity_log"
    log_dir.mkdir(parents=True, exist_ok=True)
    date_str = now_jst().date().isoformat()
    path = log_dir / f"{date_str}.jsonl"
    with path.open("a", encoding="utf-8") as f:
        for entry in entries:
            if "ts" not in entry:
                from core.time_utils import now_iso

                entry["ts"] = now_iso()
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _patches(tmp_path, animas):
    """Return context managers for common patches."""
    mock_cfg = _mock_config(animas)
    return (
        patch("core.config.models.load_config", return_value=mock_cfg),
        patch("core.paths.get_animas_dir", return_value=tmp_path / "animas"),
        patch("core.paths.get_data_dir", return_value=tmp_path),
    )


class TestAuditSubordinateSummary:
    """Tests for audit_subordinate summary mode."""

    def test_summary_direct_subordinate(self, tmp_path):
        handler = _make_handler(tmp_path, "sakura")
        sub_dir = _setup_subordinate(tmp_path, "hinata", supervisor="sakura")

        _write_activity(sub_dir, [
            {"type": "tool_use", "tool": "read_file", "summary": "read foo.py"},
            {"type": "tool_use", "tool": "read_file", "summary": "read bar.py"},
            {"type": "tool_use", "tool": "execute_command", "summary": "ls"},
            {"type": "message_sent", "to": "sakura", "content": "done"},
        ])

        p1, p2, p3 = _patches(tmp_path, {
            "sakura": {},
            "hinata": {"supervisor": "sakura"},
        })
        with p1, p2, p3:
            result = handler.handle("audit_subordinate", {"name": "hinata"})

        assert "hinata" in result
        assert "sakura" in result

    def test_report_shows_model_info(self, tmp_path):
        handler = _make_handler(tmp_path, "sakura")
        _setup_subordinate(tmp_path, "hinata", supervisor="sakura", model="openai/gpt-4.1")

        p1, p2, p3 = _patches(tmp_path, {
            "sakura": {},
            "hinata": {"supervisor": "sakura"},
        })
        with p1, p2, p3:
            result = handler.handle("audit_subordinate", {"name": "hinata"})

        assert "openai/gpt-4.1" in result

    def test_report_no_activity(self, tmp_path):
        handler = _make_handler(tmp_path, "sakura")
        _setup_subordinate(tmp_path, "hinata", supervisor="sakura")

        p1, p2, p3 = _patches(tmp_path, {
            "sakura": {},
            "hinata": {"supervisor": "sakura"},
        })
        with p1, p2, p3:
            result = handler.handle("audit_subordinate", {"name": "hinata"})

        assert "hinata" in result

    def test_report_with_errors(self, tmp_path):
        handler = _make_handler(tmp_path, "sakura")
        sub_dir = _setup_subordinate(tmp_path, "hinata", supervisor="sakura")

        _write_activity(sub_dir, [
            {"type": "error", "summary": "API timeout on tool call", "meta": {"phase": "tool_use"}},
            {"type": "error", "summary": "Connection refused", "meta": {"phase": "heartbeat"}},
        ])

        p1, p2, p3 = _patches(tmp_path, {
            "sakura": {},
            "hinata": {"supervisor": "sakura"},
        })
        with p1, p2, p3:
            result = handler.handle("audit_subordinate", {"name": "hinata"})

        assert "API timeout" in result
        assert "Connection refused" in result

    def test_report_communication_patterns(self, tmp_path):
        handler = _make_handler(tmp_path, "sakura")
        sub_dir = _setup_subordinate(tmp_path, "hinata", supervisor="sakura")

        _write_activity(sub_dir, [
            {"type": "message_sent", "to": "sakura", "content": "report"},
            {"type": "message_sent", "to": "rin", "content": "hi"},
        ])

        p1, p2, p3 = _patches(tmp_path, {
            "sakura": {},
            "hinata": {"supervisor": "sakura"},
        })
        with p1, p2, p3:
            result = handler.handle("audit_subordinate", {"name": "hinata"})

        assert "sakura" in result
        assert "rin" in result


class TestAuditSubordinateReport:
    """Tests for audit_subordinate report mode."""

    def test_report_mode_priority_categories(self, tmp_path):
        """Report mode shows high-value events in category sections; tool_use only as summary."""
        handler = _make_handler(tmp_path, "sakura")
        sub_dir = _setup_subordinate(tmp_path, "hinata", supervisor="sakura")

        _write_activity(sub_dir, [
            {"type": "heartbeat_end", "summary": "Checked pending tasks, all clear"},
            {"type": "tool_use", "tool": "github_pr_review", "content": "Reviewed PR #42"},
            {"type": "message_sent", "to": "rin", "content": "Task complete"},
        ])

        p1, p2, p3 = _patches(tmp_path, {
            "sakura": {},
            "hinata": {"supervisor": "sakura"},
        })
        with p1, p2, p3:
            result = handler.handle("audit_subordinate", {"name": "hinata", "mode": "report"})

        assert "行動レポート" in result or "Activity Report" in result
        # High-value events appear in unified timeline
        assert "🔄" in result  # heartbeat
        assert "📨" in result  # message_sent
        # tool_use appears only in summary, not as individual entries
        assert "github_pr_review" in result
        assert "ツール使用サマリー" in result or "Tool Usage Summary" in result
        # Chronological: heartbeat appears before message_sent in output
        hb_pos = result.index("🔄")
        dm_pos = result.index("📨")
        assert hb_pos < dm_pos

    def test_report_tool_use_summary_only_no_individual_entries(self, tmp_path):
        """tool_use must appear only as summary line, never as individual entries."""
        handler = _make_handler(tmp_path, "sakura")
        sub_dir = _setup_subordinate(tmp_path, "hinata", supervisor="sakura")

        _write_activity(sub_dir, [
            {"type": "tool_use", "tool": "Read", "content": "read foo"},
            {"type": "tool_use", "tool": "Read", "content": "read bar"},
            {"type": "tool_use", "tool": "Write", "content": "write baz"},
        ])

        p1, p2, p3 = _patches(tmp_path, {
            "sakura": {},
            "hinata": {"supervisor": "sakura"},
        })
        with p1, p2, p3:
            result = handler.handle("audit_subordinate", {"name": "hinata", "mode": "report"})

        # Tool summary section must exist
        assert "ツール使用サマリー" in result or "Tool Usage Summary" in result
        assert "Read" in result and "Write" in result
        # Must NOT have individual tool entry lines (e.g. "[HH:MM] 🔧 ツール: Read")
        assert "ツール: Read" not in result
        assert "ツール: Write" not in result
        assert "Tool: Read" not in result
        assert "Tool: Write" not in result

    def test_report_mode_footer_stats(self, tmp_path):
        handler = _make_handler(tmp_path, "sakura")
        sub_dir = _setup_subordinate(tmp_path, "hinata", supervisor="sakura")

        _write_activity(sub_dir, [
            {"type": "heartbeat_end", "summary": "check"},
            {"type": "tool_use", "tool": "web_search", "content": "searching"},
            {"type": "error", "summary": "timeout", "meta": {"phase": "tool_use"}},
        ])

        p1, p2, p3 = _patches(tmp_path, {
            "sakura": {},
            "hinata": {"supervisor": "sakura"},
        })
        with p1, p2, p3:
            result = handler.handle("audit_subordinate", {"name": "hinata", "mode": "report"})

        assert "統計" in result or "Stats" in result

    def test_report_no_activity(self, tmp_path):
        handler = _make_handler(tmp_path, "sakura")
        _setup_subordinate(tmp_path, "hinata", supervisor="sakura")

        p1, p2, p3 = _patches(tmp_path, {
            "sakura": {},
            "hinata": {"supervisor": "sakura"},
        })
        with p1, p2, p3:
            result = handler.handle("audit_subordinate", {"name": "hinata", "mode": "report"})

        assert "行動レポート" in result or "Activity Report" in result
        assert "活動ログはありません" in result or "No activity" in result

    def test_report_thinking_text_excluded(self, tmp_path):
        handler = _make_handler(tmp_path, "sakura")
        sub_dir = _setup_subordinate(tmp_path, "hinata", supervisor="sakura")

        _write_activity(sub_dir, [
            {
                "type": "response_sent",
                "content": "Here is my answer",
                "meta": {"thinking_text": "SECRET_INTERNAL_THOUGHT"},
            },
        ])

        p1, p2, p3 = _patches(tmp_path, {
            "sakura": {},
            "hinata": {"supervisor": "sakura"},
        })
        with p1, p2, p3:
            result = handler.handle("audit_subordinate", {"name": "hinata", "mode": "report"})

        assert "Here is my answer" in result
        assert "SECRET_INTERNAL_THOUGHT" not in result


class TestAuditSubordinatePermissions:
    """Tests for permissions and edge cases."""

    def test_grandchild_allowed(self, tmp_path):
        handler = _make_handler(tmp_path, "sakura")
        _setup_subordinate(tmp_path, "hinata", supervisor="sakura")
        _setup_subordinate(tmp_path, "rin", supervisor="hinata")

        p1, p2, p3 = _patches(tmp_path, {
            "sakura": {},
            "hinata": {"supervisor": "sakura"},
            "rin": {"supervisor": "hinata"},
        })
        with p1, p2, p3:
            result = handler.handle("audit_subordinate", {"name": "rin"})

        assert "rin" in result
        assert "PermissionDenied" not in result

    def test_non_descendant_rejected(self, tmp_path):
        handler = _make_handler(tmp_path, "sakura")
        _setup_subordinate(tmp_path, "mio", supervisor="taka")

        p1, p2, _ = _patches(tmp_path, {
            "sakura": {},
            "mio": {"supervisor": "taka"},
        })
        with p1:
            result = handler.handle("audit_subordinate", {"name": "mio"})

        assert "PermissionDenied" in result

    def test_self_rejected(self, tmp_path):
        handler = _make_handler(tmp_path, "sakura")

        result = handler.handle("audit_subordinate", {"name": "sakura"})

        assert "自分自身" in result or "yourself" in result.lower()


class TestAuditSubordinateBatch:
    """Tests for name-omit batch audit."""

    def test_batch_all_direct_subordinates(self, tmp_path):
        handler = _make_handler(tmp_path, "sakura")
        dir_h = _setup_subordinate(tmp_path, "hinata", supervisor="sakura")
        dir_r = _setup_subordinate(tmp_path, "rin", supervisor="sakura")

        _write_activity(dir_h, [{"type": "heartbeat_end", "summary": "hinata check"}])
        _write_activity(dir_r, [{"type": "heartbeat_end", "summary": "rin check"}])

        p1, p2, p3 = _patches(tmp_path, {
            "sakura": {},
            "hinata": {"supervisor": "sakura"},
            "rin": {"supervisor": "sakura"},
        })
        with p1, p2, p3:
            result = handler.handle("audit_subordinate", {})

        assert "組織タイムライン" in result or "Org Timeline" in result
        assert "hinata" in result
        assert "rin" in result

    def test_batch_no_subordinates(self, tmp_path):
        handler = _make_handler(tmp_path, "sakura")

        p1, _, _ = _patches(tmp_path, {"sakura": {}})
        with p1:
            result = handler.handle("audit_subordinate", {})

        assert "いません" in result or "No subordinate" in result

    def test_batch_direct_only(self, tmp_path):
        handler = _make_handler(tmp_path, "sakura")
        dir_h = _setup_subordinate(tmp_path, "hinata", supervisor="sakura")
        _setup_subordinate(tmp_path, "rin", supervisor="hinata")

        _write_activity(dir_h, [{"type": "heartbeat_end", "summary": "hinata only"}])

        p1, p2, p3 = _patches(tmp_path, {
            "sakura": {},
            "hinata": {"supervisor": "sakura"},
            "rin": {"supervisor": "hinata"},
        })
        with p1, p2, p3:
            result = handler.handle("audit_subordinate", {"direct_only": True})

        assert "hinata" in result
        assert "rin" not in result


class TestAuditSubordinateParams:
    """Tests for hours parameter."""

    def test_hours_default_24(self, tmp_path):
        handler = _make_handler(tmp_path, "sakura")
        _setup_subordinate(tmp_path, "hinata", supervisor="sakura")

        p1, p2, p3 = _patches(tmp_path, {
            "sakura": {},
            "hinata": {"supervisor": "sakura"},
        })
        with p1, p2, p3:
            result = handler.handle("audit_subordinate", {"name": "hinata"})

        assert "24h" in result

    def test_hours_custom(self, tmp_path):
        handler = _make_handler(tmp_path, "sakura")
        _setup_subordinate(tmp_path, "hinata", supervisor="sakura")

        p1, p2, p3 = _patches(tmp_path, {
            "sakura": {},
            "hinata": {"supervisor": "sakura"},
        })
        with p1, p2, p3:
            result = handler.handle("audit_subordinate", {"name": "hinata", "hours": 48})

        assert "48h" in result

    def test_hours_clamped_max(self, tmp_path):
        handler = _make_handler(tmp_path, "sakura")
        _setup_subordinate(tmp_path, "hinata", supervisor="sakura")

        p1, p2, p3 = _patches(tmp_path, {
            "sakura": {},
            "hinata": {"supervisor": "sakura"},
        })
        with p1, p2, p3:
            result = handler.handle("audit_subordinate", {"name": "hinata", "hours": 999})

        assert "168h" in result

    def test_hours_clamped_min(self, tmp_path):
        handler = _make_handler(tmp_path, "sakura")
        _setup_subordinate(tmp_path, "hinata", supervisor="sakura")

        p1, p2, p3 = _patches(tmp_path, {
            "sakura": {},
            "hinata": {"supervisor": "sakura"},
        })
        with p1, p2, p3:
            result = handler.handle("audit_subordinate", {"name": "hinata", "hours": 0})

        assert "1h" in result

    def test_legacy_days_param_compat(self, tmp_path):
        """Legacy 'days' param is converted to hours (days * 24)."""
        handler = _make_handler(tmp_path, "sakura")
        _setup_subordinate(tmp_path, "hinata", supervisor="sakura")

        p1, p2, p3 = _patches(tmp_path, {
            "sakura": {},
            "hinata": {"supervisor": "sakura"},
        })
        with p1, p2, p3:
            result = handler.handle("audit_subordinate", {"name": "hinata", "days": 3})

        assert "72h" in result

    def test_hours_overrides_days(self, tmp_path):
        """When both hours and days are given, hours takes precedence."""
        handler = _make_handler(tmp_path, "sakura")
        _setup_subordinate(tmp_path, "hinata", supervisor="sakura")

        p1, p2, p3 = _patches(tmp_path, {
            "sakura": {},
            "hinata": {"supervisor": "sakura"},
        })
        with p1, p2, p3:
            result = handler.handle("audit_subordinate", {"name": "hinata", "hours": 12, "days": 3})

        assert "12h" in result


class TestAuditSubordinateSince:
    """Tests for since parameter (HH:MM time filter)."""

    def test_since_filters_entries(self, tmp_path):
        """Events before 'since' time are excluded."""
        from core.time_utils import now_jst

        handler = _make_handler(tmp_path, "sakura")
        sub_dir = _setup_subordinate(tmp_path, "hinata", supervisor="sakura")

        base = now_jst()
        ts_early = base.replace(hour=8, minute=0, second=0, microsecond=0).isoformat()
        ts_late = base.replace(hour=14, minute=0, second=0, microsecond=0).isoformat()

        _write_activity(sub_dir, [
            {"type": "heartbeat_end", "summary": "Early HB", "ts": ts_early},
            {"type": "heartbeat_end", "summary": "Late HB", "ts": ts_late},
        ])

        p1, p2, p3 = _patches(tmp_path, {
            "sakura": {},
            "hinata": {"supervisor": "sakura"},
        })
        with p1, p2, p3:
            result = handler.handle("audit_subordinate", {
                "name": "hinata", "mode": "report", "since": "13:00",
            })

        assert "Late HB" in result
        assert "Early HB" not in result

    def test_since_title_format(self, tmp_path):
        """Title should show 'since HH:MM' instead of 'last Xh'."""
        handler = _make_handler(tmp_path, "sakura")
        _setup_subordinate(tmp_path, "hinata", supervisor="sakura")

        p1, p2, p3 = _patches(tmp_path, {
            "sakura": {},
            "hinata": {"supervisor": "sakura"},
        })
        with p1, p2, p3:
            result = handler.handle("audit_subordinate", {
                "name": "hinata", "mode": "report", "since": "09:00",
            })

        assert "09:00" in result
        assert "24h" not in result

    def test_since_summary_mode(self, tmp_path):
        """since works with summary mode too."""
        handler = _make_handler(tmp_path, "sakura")
        _setup_subordinate(tmp_path, "hinata", supervisor="sakura")

        p1, p2, p3 = _patches(tmp_path, {
            "sakura": {},
            "hinata": {"supervisor": "sakura"},
        })
        with p1, p2, p3:
            result = handler.handle("audit_subordinate", {
                "name": "hinata", "since": "10:30",
            })

        assert "10:30" in result

    def test_since_batch_merged_timeline(self, tmp_path):
        """since works with merged timeline (batch mode)."""
        handler = _make_handler(tmp_path, "sakura")
        _setup_subordinate(tmp_path, "hinata", supervisor="sakura")
        _setup_subordinate(tmp_path, "rin", supervisor="sakura")

        p1, p2, p3 = _patches(tmp_path, {
            "sakura": {},
            "hinata": {"supervisor": "sakura"},
            "rin": {"supervisor": "sakura"},
        })
        with p1, p2, p3:
            result = handler.handle("audit_subordinate", {"since": "09:00"})

        assert "09:00" in result
        assert "組織タイムライン" in result or "Org Timeline" in result

    def test_since_invalid_format_ignored(self, tmp_path):
        """Invalid since format is silently ignored (falls back to hours)."""
        handler = _make_handler(tmp_path, "sakura")
        _setup_subordinate(tmp_path, "hinata", supervisor="sakura")

        p1, p2, p3 = _patches(tmp_path, {
            "sakura": {},
            "hinata": {"supervisor": "sakura"},
        })
        with p1, p2, p3:
            result = handler.handle("audit_subordinate", {
                "name": "hinata", "since": "invalid",
            })

        assert "24h" in result
        assert "PermissionDenied" not in result


class TestMergedTimeline:
    """Tests for generate_merged_timeline (cross-anima unified view)."""

    def test_merged_timeline_chronological_order(self, tmp_path):
        """Events from multiple animas are sorted by timestamp."""
        from core.time_utils import now_jst

        dir_a = _setup_subordinate(tmp_path, "alice", supervisor="boss")
        dir_b = _setup_subordinate(tmp_path, "bob", supervisor="boss")

        base = now_jst()
        ts_early = (base - timedelta(hours=3)).isoformat()
        ts_mid = (base - timedelta(hours=2)).isoformat()
        ts_late = (base - timedelta(hours=1)).isoformat()

        _write_activity(dir_a, [
            {"type": "heartbeat_end", "summary": "Alice HB", "ts": ts_early},
            {"type": "response_sent", "content": "Alice response", "ts": ts_late},
        ])
        _write_activity(dir_b, [
            {"type": "error", "summary": "Bob error", "ts": ts_mid, "meta": {"phase": "tool_use"}},
        ])

        from core.memory.audit import AuditAggregator

        result = AuditAggregator.generate_merged_timeline([dir_a, dir_b], hours=24)

        assert "組織タイムライン" in result or "Org Timeline" in result
        assert "2名" in result or "2 animas" in result
        alice_hb = result.index("alice")
        bob_err = result.index("bob")
        alice_resp = result.rindex("alice")
        assert alice_hb < bob_err < alice_resp

    def test_merged_timeline_includes_anima_names(self, tmp_path):
        """Each timeline entry is prefixed with the anima name."""
        dir_a = _setup_subordinate(tmp_path, "alice", supervisor="boss")
        _write_activity(dir_a, [
            {"type": "heartbeat_end", "summary": "checking tasks"},
        ])

        from core.memory.audit import AuditAggregator

        result = AuditAggregator.generate_merged_timeline([dir_a], hours=24)

        assert "alice 🔄" in result

    def test_merged_timeline_tool_summary_per_anima(self, tmp_path):
        """Tool usage summary is shown per-anima."""
        dir_a = _setup_subordinate(tmp_path, "alice", supervisor="boss")
        dir_b = _setup_subordinate(tmp_path, "bob", supervisor="boss")

        _write_activity(dir_a, [
            {"type": "tool_use", "tool": "Read", "content": "foo"},
            {"type": "tool_use", "tool": "Read", "content": "bar"},
        ])
        _write_activity(dir_b, [
            {"type": "tool_use", "tool": "Write", "content": "baz"},
        ])

        from core.memory.audit import AuditAggregator

        result = AuditAggregator.generate_merged_timeline([dir_a, dir_b], hours=24)

        assert "alice" in result and "Read: 2" in result
        assert "bob" in result and "Write: 1" in result

    def test_merged_timeline_no_individual_tool_entries(self, tmp_path):
        """tool_use events must not appear as individual timeline entries."""
        dir_a = _setup_subordinate(tmp_path, "alice", supervisor="boss")

        _write_activity(dir_a, [
            {"type": "tool_use", "tool": "Read", "content": "reading"},
            {"type": "heartbeat_end", "summary": "HB done"},
        ])

        from core.memory.audit import AuditAggregator

        result = AuditAggregator.generate_merged_timeline([dir_a], hours=24)

        assert "🔧" not in result
        assert "🔄" in result

    def test_merged_timeline_empty(self, tmp_path):
        """Empty anima dirs produce no-activity message."""
        dir_a = _setup_subordinate(tmp_path, "alice", supervisor="boss")

        from core.memory.audit import AuditAggregator

        result = AuditAggregator.generate_merged_timeline([dir_a], hours=24)

        assert "活動ログはありません" in result or "No activity" in result

    def test_merged_timeline_footer_stats(self, tmp_path):
        """Footer shows aggregate stats across all animas."""
        dir_a = _setup_subordinate(tmp_path, "alice", supervisor="boss")
        dir_b = _setup_subordinate(tmp_path, "bob", supervisor="boss")

        _write_activity(dir_a, [
            {"type": "heartbeat_end", "summary": "HB1"},
            {"type": "tool_use", "tool": "Read", "content": "x"},
        ])
        _write_activity(dir_b, [
            {"type": "error", "summary": "timeout", "meta": {}},
        ])

        from core.memory.audit import AuditAggregator

        result = AuditAggregator.generate_merged_timeline([dir_a, dir_b], hours=24)

        assert "全2名" in result or "2 animas" in result
        assert "HB1" in result

    def test_batch_report_mode_uses_merged_timeline(self, tmp_path):
        """audit_subordinate with multiple targets + report mode triggers merged timeline."""
        handler = _make_handler(tmp_path, "sakura")
        dir_a = _setup_subordinate(tmp_path, "alice", supervisor="sakura")
        dir_b = _setup_subordinate(tmp_path, "bob", supervisor="sakura")

        _write_activity(dir_a, [
            {"type": "heartbeat_end", "summary": "Alice check"},
        ])
        _write_activity(dir_b, [
            {"type": "response_sent", "content": "Bob reply"},
        ])

        p1, p2, p3 = _patches(tmp_path, {
            "sakura": {},
            "alice": {"supervisor": "sakura"},
            "bob": {"supervisor": "sakura"},
        })
        with p1, p2, p3:
            result = handler.handle("audit_subordinate", {"mode": "report"})

        assert "組織タイムライン" in result or "Org Timeline" in result
        assert "alice" in result
        assert "bob" in result
