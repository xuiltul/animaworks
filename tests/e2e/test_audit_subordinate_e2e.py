"""E2E tests for audit_subordinate supervisor tool.

Tests the full flow from handler.handle() through AuditAggregator
with real ActivityLogger file I/O and TaskQueueManager.
"""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.tooling.handler import ToolHandler


def _make_handler(tmp_path: Path, anima_name: str = "sakura") -> ToolHandler:
    anima_dir = tmp_path / "animas" / anima_name
    anima_dir.mkdir(parents=True)
    (anima_dir / "permissions.md").write_text("", encoding="utf-8")
    (anima_dir / "activity_log").mkdir(parents=True, exist_ok=True)
    memory = MagicMock()
    memory.read_permissions.return_value = ""
    return ToolHandler(anima_dir=anima_dir, memory=memory, messenger=None, tool_registry=[])


def _setup_sub(tmp_path: Path, name: str, supervisor: str, **kw) -> Path:
    anima_dir = tmp_path / "animas" / name
    anima_dir.mkdir(parents=True, exist_ok=True)
    (anima_dir / "activity_log").mkdir(parents=True, exist_ok=True)
    (anima_dir / "state").mkdir(parents=True, exist_ok=True)
    status = {"enabled": kw.get("enabled", True), "supervisor": supervisor, "model": kw.get("model", "claude-sonnet-4-6"), "role": "general"}
    (anima_dir / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    return anima_dir


def _mock_cfg(animas):
    from core.config.models import AnimaModelConfig

    cfg = MagicMock()
    cfg.animas = {n: AnimaModelConfig(**f) for n, f in animas.items()}
    return cfg


def _write_log(anima_dir: Path, entries: list[dict]) -> None:
    from core.time_utils import now_iso, now_jst

    log_dir = anima_dir / "activity_log"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{now_jst().date().isoformat()}.jsonl"
    with path.open("a", encoding="utf-8") as f:
        for e in entries:
            if "ts" not in e:
                e["ts"] = now_iso()
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


def _add_task(anima_dir: Path, task_id: str, status: str = "pending") -> None:
    tq = anima_dir / "state" / "task_queue.jsonl"
    entry = {
        "task_id": task_id,
        "summary": f"Task {task_id}",
        "status": status,
        "source": "anima",
        "assignee": anima_dir.name,
    }
    from core.time_utils import now_iso

    entry["ts"] = now_iso()
    with tq.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


class TestAuditSubordinateE2E:
    """E2E: 3-level hierarchy sakura → hinata → rin."""

    @pytest.fixture()
    def three_level(self, tmp_path):
        handler = _make_handler(tmp_path, "sakura")
        h_dir = _setup_sub(tmp_path, "hinata", "sakura")
        r_dir = _setup_sub(tmp_path, "rin", "hinata")

        _write_log(h_dir, [
            {"type": "heartbeat_end", "summary": "Observed: 3 pending PRs. Plan: review oldest first."},
            {"type": "tool_use", "tool": "github_pr_review", "content": "Reviewed PR #42"},
            {"type": "tool_use", "tool": "web_search", "content": "Searched for API docs"},
            {"type": "message_sent", "to": "rin", "content": "Please handle PR #43"},
            {"type": "message_sent", "to": "sakura", "content": "PR #42 approved"},
            {"type": "cron_executed", "content": "Daily report generated", "meta": {"task_name": "daily_report"}},
            {"type": "error", "summary": "Timeout calling web_search", "meta": {"phase": "tool_use"}},
        ])
        _add_task(h_dir, "task-a", "pending")
        _add_task(h_dir, "task-b", "done")

        _write_log(r_dir, [
            {"type": "heartbeat_end", "summary": "All tasks complete. Idle."},
            {"type": "response_sent", "content": "Got it, working on PR #43", "meta": {"thinking_text": "TOP_SECRET"}},
            {"type": "tool_use", "tool": "execute_command", "content": "git checkout -b fix-43"},
            {"type": "task_exec_end", "summary": "PR #43 fix implemented and tested"},
            {"type": "issue_resolved", "content": "Fixed null pointer in auth module"},
        ])

        cfg = _mock_cfg({
            "sakura": {},
            "hinata": {"supervisor": "sakura"},
            "rin": {"supervisor": "hinata"},
        })
        return handler, cfg, tmp_path

    def test_report_single_target(self, three_level):
        handler, cfg, tmp_path = three_level
        with (
            patch("core.config.models.load_config", return_value=cfg),
            patch("core.paths.get_animas_dir", return_value=tmp_path / "animas"),
            patch("core.paths.get_data_dir", return_value=tmp_path),
        ):
            result = handler.handle("audit_subordinate", {"name": "hinata"})

        assert "行動レポート" in result or "Activity Report" in result
        assert "🔄" in result  # heartbeat
        assert "📨" in result  # message_sent
        assert "❌" in result  # error
        assert "Timeout" in result
        # tool_use appears only in summary, not as individual entries
        assert "ツール使用サマリー" in result or "Tool Usage Summary" in result
        assert "github_pr_review" in result or "web_search" in result

    def test_report_grandchild_thinking_excluded(self, three_level):
        handler, cfg, tmp_path = three_level
        with (
            patch("core.config.models.load_config", return_value=cfg),
            patch("core.paths.get_animas_dir", return_value=tmp_path / "animas"),
            patch("core.paths.get_data_dir", return_value=tmp_path),
        ):
            result = handler.handle("audit_subordinate", {"name": "rin"})

        assert "rin" in result
        assert "TOP_SECRET" not in result
        assert "Got it" in result

    def test_batch_all_descendants(self, three_level):
        handler, cfg, tmp_path = three_level
        with (
            patch("core.config.models.load_config", return_value=cfg),
            patch("core.paths.get_animas_dir", return_value=tmp_path / "animas"),
            patch("core.paths.get_data_dir", return_value=tmp_path),
        ):
            result = handler.handle("audit_subordinate", {})

        assert "hinata" in result
        assert "rin" in result

    def test_batch_direct_only(self, three_level):
        handler, cfg, tmp_path = three_level
        with (
            patch("core.config.models.load_config", return_value=cfg),
            patch("core.paths.get_animas_dir", return_value=tmp_path / "animas"),
            patch("core.paths.get_data_dir", return_value=tmp_path),
        ):
            result = handler.handle("audit_subordinate", {"direct_only": True})

        assert "hinata" in result
        assert "═══ rin" not in result

    def test_hours_parameter(self, three_level):
        handler, cfg, tmp_path = three_level
        with (
            patch("core.config.models.load_config", return_value=cfg),
            patch("core.paths.get_animas_dir", return_value=tmp_path / "animas"),
            patch("core.paths.get_data_dir", return_value=tmp_path),
        ):
            result = handler.handle("audit_subordinate", {"name": "hinata", "hours": 72})

        assert "72h" in result

    def test_non_descendant_rejected(self, three_level):
        handler, cfg, tmp_path = three_level
        with patch("core.config.models.load_config", return_value=cfg):
            result = handler.handle("audit_subordinate", {"name": "nobody"})

        assert "PermissionDenied" in result

    def test_task_summary_integration(self, three_level):
        handler, cfg, tmp_path = three_level
        with (
            patch("core.config.models.load_config", return_value=cfg),
            patch("core.paths.get_animas_dir", return_value=tmp_path / "animas"),
            patch("core.paths.get_data_dir", return_value=tmp_path),
        ):
            result = handler.handle("audit_subordinate", {"name": "hinata"})

        assert "hinata" in result
