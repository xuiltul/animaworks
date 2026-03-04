"""Unit tests for audit_subordinate supervisor tool."""
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
        json.dumps(status, indent=2), encoding="utf-8",
    )
    return anima_dir


def _mock_config(animas: dict[str, dict]) -> MagicMock:
    """Build a mock config with AnimaModelConfig entries."""
    from core.config.models import AnimaModelConfig

    config = MagicMock()
    config.animas = {
        name: AnimaModelConfig(**fields)
        for name, fields in animas.items()
    }
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


class TestAuditSubordinate:
    """Tests for _handle_audit_subordinate."""

    def test_audit_direct_subordinate(self, tmp_path):
        handler = _make_handler(tmp_path, "sakura")
        sub_dir = _setup_subordinate(tmp_path, "hinata", supervisor="sakura")

        _write_activity(sub_dir, [
            {"type": "message_received", "from": "user", "content": "hello"},
            {"type": "tool_use", "tool": "read_file", "summary": "read foo.py"},
            {"type": "tool_use", "tool": "read_file", "summary": "read bar.py"},
            {"type": "tool_use", "tool": "execute_command", "summary": "ls"},
            {"type": "message_sent", "to": "sakura", "content": "done"},
        ])

        mock_cfg = _mock_config({
            "sakura": {},
            "hinata": {"supervisor": "sakura"},
        })

        with (
            patch("core.config.models.load_config", return_value=mock_cfg),
            patch("core.paths.get_animas_dir", return_value=tmp_path / "animas"),
            patch("core.paths.get_data_dir", return_value=tmp_path),
        ):
            result = handler.handle("audit_subordinate", {"name": "hinata"})

        assert "hinata" in result
        assert "監査レポート" in result or "Audit Report" in result
        assert "read_file" in result
        assert "execute_command" in result
        assert "sakura" in result

    def test_audit_grandchild_allowed(self, tmp_path):
        handler = _make_handler(tmp_path, "sakura")
        _setup_subordinate(tmp_path, "hinata", supervisor="sakura")
        _setup_subordinate(tmp_path, "rin", supervisor="hinata")

        mock_cfg = _mock_config({
            "sakura": {},
            "hinata": {"supervisor": "sakura"},
            "rin": {"supervisor": "hinata"},
        })

        with (
            patch("core.config.models.load_config", return_value=mock_cfg),
            patch("core.paths.get_animas_dir", return_value=tmp_path / "animas"),
            patch("core.paths.get_data_dir", return_value=tmp_path),
        ):
            result = handler.handle("audit_subordinate", {"name": "rin"})

        assert "rin" in result
        assert "Error" not in result or "エラーなし" in result or "No errors" in result

    def test_audit_non_descendant_rejected(self, tmp_path):
        handler = _make_handler(tmp_path, "sakura")
        _setup_subordinate(tmp_path, "mio", supervisor="taka")

        mock_cfg = _mock_config({
            "sakura": {},
            "mio": {"supervisor": "taka"},
        })

        with patch("core.config.models.load_config", return_value=mock_cfg):
            result = handler.handle("audit_subordinate", {"name": "mio"})

        assert "PermissionDenied" in result

    def test_audit_self_rejected(self, tmp_path):
        handler = _make_handler(tmp_path, "sakura")

        result = handler.handle("audit_subordinate", {"name": "sakura"})

        assert "自分自身" in result or "yourself" in result.lower()

    def test_audit_missing_name(self, tmp_path):
        handler = _make_handler(tmp_path, "sakura")

        result = handler.handle("audit_subordinate", {})

        assert "InvalidArguments" in result

    def test_audit_no_activity(self, tmp_path):
        handler = _make_handler(tmp_path, "sakura")
        _setup_subordinate(tmp_path, "hinata", supervisor="sakura")

        mock_cfg = _mock_config({
            "sakura": {},
            "hinata": {"supervisor": "sakura"},
        })

        with (
            patch("core.config.models.load_config", return_value=mock_cfg),
            patch("core.paths.get_animas_dir", return_value=tmp_path / "animas"),
            patch("core.paths.get_data_dir", return_value=tmp_path),
        ):
            result = handler.handle("audit_subordinate", {"name": "hinata"})

        assert "hinata" in result
        assert "活動ログはありません" in result or "No activity" in result

    def test_audit_with_errors(self, tmp_path):
        handler = _make_handler(tmp_path, "sakura")
        sub_dir = _setup_subordinate(tmp_path, "hinata", supervisor="sakura")

        _write_activity(sub_dir, [
            {"type": "error", "summary": "API timeout on tool call"},
            {"type": "error", "summary": "Connection refused"},
            {"type": "tool_use", "tool": "web_fetch", "summary": "fetch docs"},
        ])

        mock_cfg = _mock_config({
            "sakura": {},
            "hinata": {"supervisor": "sakura"},
        })

        with (
            patch("core.config.models.load_config", return_value=mock_cfg),
            patch("core.paths.get_animas_dir", return_value=tmp_path / "animas"),
            patch("core.paths.get_data_dir", return_value=tmp_path),
        ):
            result = handler.handle("audit_subordinate", {"name": "hinata"})

        assert "2" in result
        assert "API timeout" in result
        assert "Connection refused" in result

    def test_audit_shows_model_info(self, tmp_path):
        handler = _make_handler(tmp_path, "sakura")
        _setup_subordinate(
            tmp_path, "hinata", supervisor="sakura",
            model="openai/gpt-4.1",
        )

        mock_cfg = _mock_config({
            "sakura": {},
            "hinata": {"supervisor": "sakura"},
        })

        with (
            patch("core.config.models.load_config", return_value=mock_cfg),
            patch("core.paths.get_animas_dir", return_value=tmp_path / "animas"),
            patch("core.paths.get_data_dir", return_value=tmp_path),
        ):
            result = handler.handle("audit_subordinate", {"name": "hinata"})

        assert "openai/gpt-4.1" in result

    def test_audit_days_param(self, tmp_path):
        handler = _make_handler(tmp_path, "sakura")
        _setup_subordinate(tmp_path, "hinata", supervisor="sakura")

        mock_cfg = _mock_config({
            "sakura": {},
            "hinata": {"supervisor": "sakura"},
        })

        with (
            patch("core.config.models.load_config", return_value=mock_cfg),
            patch("core.paths.get_animas_dir", return_value=tmp_path / "animas"),
            patch("core.paths.get_data_dir", return_value=tmp_path),
        ):
            result = handler.handle("audit_subordinate", {"name": "hinata", "days": 7})

        assert "7" in result

    def test_audit_communication_patterns(self, tmp_path):
        handler = _make_handler(tmp_path, "sakura")
        sub_dir = _setup_subordinate(tmp_path, "hinata", supervisor="sakura")

        _write_activity(sub_dir, [
            {"type": "message_sent", "to": "sakura", "content": "report"},
            {"type": "message_sent", "to": "sakura", "content": "update"},
            {"type": "message_sent", "to": "rin", "content": "hi"},
            {"type": "message_received", "from": "sakura", "content": "ok"},
        ])

        mock_cfg = _mock_config({
            "sakura": {},
            "hinata": {"supervisor": "sakura"},
        })

        with (
            patch("core.config.models.load_config", return_value=mock_cfg),
            patch("core.paths.get_animas_dir", return_value=tmp_path / "animas"),
            patch("core.paths.get_data_dir", return_value=tmp_path),
        ):
            result = handler.handle("audit_subordinate", {"name": "hinata"})

        assert "sakura" in result
        assert "rin" in result

    def test_audit_days_clamped(self, tmp_path):
        """days should be clamped to [1, 30]."""
        handler = _make_handler(tmp_path, "sakura")
        _setup_subordinate(tmp_path, "hinata", supervisor="sakura")

        mock_cfg = _mock_config({
            "sakura": {},
            "hinata": {"supervisor": "sakura"},
        })

        with (
            patch("core.config.models.load_config", return_value=mock_cfg),
            patch("core.paths.get_animas_dir", return_value=tmp_path / "animas"),
            patch("core.paths.get_data_dir", return_value=tmp_path),
        ):
            result = handler.handle("audit_subordinate", {"name": "hinata", "days": 0})
            assert "1" in result

            result = handler.handle("audit_subordinate", {"name": "hinata", "days": 100})
            assert "30" in result
