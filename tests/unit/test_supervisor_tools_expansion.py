"""Unit tests for new supervisor tools: org_dashboard, ping_subordinate,
read_subordinate_state, check_permissions, delegate_task, task_tracker."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch


from core.tooling.handler import ToolHandler


# ── Fixtures ──────────────────────────────────────────────────


def _make_handler(
    tmp_path: Path,
    anima_name: str = "sakura",
    *,
    messenger: MagicMock | None = None,
    process_supervisor: Any | None = None,
) -> ToolHandler:
    """Create a ToolHandler with minimal mocked dependencies."""
    anima_dir = tmp_path / "animas" / anima_name
    anima_dir.mkdir(parents=True, exist_ok=True)
    (anima_dir / "permissions.md").write_text("", encoding="utf-8")
    (anima_dir / "state").mkdir(exist_ok=True)

    memory = MagicMock()
    memory.read_permissions.return_value = ""

    handler = ToolHandler(
        anima_dir=anima_dir,
        memory=memory,
        messenger=messenger,
        tool_registry=[],
        process_supervisor=process_supervisor,
    )
    return handler


def _setup_subordinate(
    tmp_path: Path,
    name: str,
    supervisor: str,
    *,
    enabled: bool = True,
) -> Path:
    anima_dir = tmp_path / "animas" / name
    anima_dir.mkdir(parents=True, exist_ok=True)
    (anima_dir / "state").mkdir(exist_ok=True)
    status = {
        "enabled": enabled,
        "supervisor": supervisor,
        "model": "claude-sonnet-4-6",
        "role": "general",
    }
    (anima_dir / "status.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8",
    )
    return anima_dir


def _mock_config(tmp_path: Path, animas: dict[str, dict]) -> MagicMock:
    from core.config.models import AnimaModelConfig

    config = MagicMock()
    config.animas = {
        name: AnimaModelConfig(**fields)
        for name, fields in animas.items()
    }
    return config


# ── _get_all_descendants tests ─────────────────────────────


class TestGetAllDescendants:
    """Tests for _get_all_descendants utility."""

    def test_direct_subordinates(self, tmp_path):
        handler = _make_handler(tmp_path, "sakura")
        mock_cfg = _mock_config(tmp_path, {
            "sakura": {},
            "hinata": {"supervisor": "sakura"},
            "kotoha": {"supervisor": "sakura"},
        })
        with patch("core.config.models.load_config", return_value=mock_cfg):
            descendants = handler._get_all_descendants()
        assert set(descendants) == {"hinata", "kotoha"}

    def test_grandchildren(self, tmp_path):
        handler = _make_handler(tmp_path, "sakura")
        mock_cfg = _mock_config(tmp_path, {
            "sakura": {},
            "hinata": {"supervisor": "sakura"},
            "natsume": {"supervisor": "hinata"},
        })
        with patch("core.config.models.load_config", return_value=mock_cfg):
            descendants = handler._get_all_descendants()
        assert set(descendants) == {"hinata", "natsume"}

    def test_circular_reference_no_infinite_loop(self, tmp_path):
        handler = _make_handler(tmp_path, "sakura")
        mock_cfg = _mock_config(tmp_path, {
            "sakura": {"supervisor": "hinata"},
            "hinata": {"supervisor": "sakura"},
        })
        with patch("core.config.models.load_config", return_value=mock_cfg):
            descendants = handler._get_all_descendants()
        assert "hinata" in descendants

    def test_no_descendants(self, tmp_path):
        handler = _make_handler(tmp_path, "sakura")
        mock_cfg = _mock_config(tmp_path, {
            "sakura": {},
            "mio": {"supervisor": "taka"},
        })
        with patch("core.config.models.load_config", return_value=mock_cfg):
            descendants = handler._get_all_descendants()
        assert descendants == []


# ── _check_descendant tests ────────────────────────────────


class TestCheckDescendant:

    def test_descendant_allowed(self, tmp_path):
        handler = _make_handler(tmp_path, "sakura")
        mock_cfg = _mock_config(tmp_path, {
            "sakura": {},
            "hinata": {"supervisor": "sakura"},
            "natsume": {"supervisor": "hinata"},
        })
        with patch("core.config.models.load_config", return_value=mock_cfg):
            assert handler._check_descendant("natsume") is None

    def test_non_descendant_denied(self, tmp_path):
        handler = _make_handler(tmp_path, "sakura")
        mock_cfg = _mock_config(tmp_path, {
            "sakura": {},
            "mio": {"supervisor": "taka"},
        })
        with patch("core.config.models.load_config", return_value=mock_cfg):
            result = handler._check_descendant("mio")
        assert result is not None
        assert "PermissionDenied" in result

    def test_self_denied(self, tmp_path):
        handler = _make_handler(tmp_path, "sakura")
        result = handler._check_descendant("sakura")
        assert "自分自身" in result


# ── org_dashboard tests ────────────────────────────────────


class TestOrgDashboard:

    def test_no_descendants(self, tmp_path):
        handler = _make_handler(tmp_path, "sakura")
        mock_cfg = _mock_config(tmp_path, {"sakura": {}})

        with patch("core.config.models.load_config", return_value=mock_cfg):
            result = handler.handle("org_dashboard", {})
        assert "配下の Anima はいません" in result

    def test_shows_descendants(self, tmp_path):
        handler = _make_handler(tmp_path, "sakura")
        _setup_subordinate(tmp_path, "hinata", supervisor="sakura")
        # Write a current task
        (tmp_path / "animas" / "hinata" / "state" / "current_task.md").write_text(
            "レポート作成中", encoding="utf-8",
        )

        mock_cfg = _mock_config(tmp_path, {
            "sakura": {},
            "hinata": {"supervisor": "sakura"},
        })

        with (
            patch("core.config.models.load_config", return_value=mock_cfg),
            patch("core.paths.get_animas_dir", return_value=tmp_path / "animas"),
        ):
            result = handler.handle("org_dashboard", {})

        assert "hinata" in result
        assert "レポート作成中" in result


# ── ping_subordinate tests ─────────────────────────────────


class TestPingSubordinate:

    def test_ping_all_no_descendants(self, tmp_path):
        handler = _make_handler(tmp_path, "sakura")
        mock_cfg = _mock_config(tmp_path, {"sakura": {}})

        with patch("core.config.models.load_config", return_value=mock_cfg):
            result = handler.handle("ping_subordinate", {})
        assert "配下の Anima はいません" in result

    def test_ping_single(self, tmp_path):
        handler = _make_handler(tmp_path, "sakura")
        _setup_subordinate(tmp_path, "hinata", supervisor="sakura")

        mock_cfg = _mock_config(tmp_path, {
            "sakura": {},
            "hinata": {"supervisor": "sakura"},
        })

        with (
            patch("core.config.models.load_config", return_value=mock_cfg),
            patch("core.paths.get_animas_dir", return_value=tmp_path / "animas"),
        ):
            result = handler.handle("ping_subordinate", {"name": "hinata"})

        parsed = json.loads(result)
        assert len(parsed) == 1
        assert parsed[0]["name"] == "hinata"

    def test_ping_non_descendant_denied(self, tmp_path):
        handler = _make_handler(tmp_path, "sakura")
        mock_cfg = _mock_config(tmp_path, {
            "sakura": {},
            "mio": {"supervisor": "taka"},
        })

        with patch("core.config.models.load_config", return_value=mock_cfg):
            result = handler.handle("ping_subordinate", {"name": "mio"})
        assert "PermissionDenied" in result


# ── read_subordinate_state tests ───────────────────────────


class TestReadSubordinateState:

    def test_read_with_task(self, tmp_path):
        handler = _make_handler(tmp_path, "sakura")
        sub_dir = _setup_subordinate(tmp_path, "hinata", supervisor="sakura")
        (sub_dir / "state" / "current_task.md").write_text(
            "API実装中", encoding="utf-8",
        )
        (sub_dir / "state" / "pending.md").write_text(
            "テスト作成待ち", encoding="utf-8",
        )

        mock_cfg = _mock_config(tmp_path, {
            "sakura": {},
            "hinata": {"supervisor": "sakura"},
        })

        with (
            patch("core.config.models.load_config", return_value=mock_cfg),
            patch("core.paths.get_animas_dir", return_value=tmp_path / "animas"),
        ):
            result = handler.handle("read_subordinate_state", {"name": "hinata"})

        assert "API実装中" in result
        assert "テスト作成待ち" in result

    def test_read_empty_state(self, tmp_path):
        handler = _make_handler(tmp_path, "sakura")
        _setup_subordinate(tmp_path, "hinata", supervisor="sakura")

        mock_cfg = _mock_config(tmp_path, {
            "sakura": {},
            "hinata": {"supervisor": "sakura"},
        })

        with (
            patch("core.config.models.load_config", return_value=mock_cfg),
            patch("core.paths.get_animas_dir", return_value=tmp_path / "animas"),
        ):
            result = handler.handle("read_subordinate_state", {"name": "hinata"})

        assert "(なし)" in result

    def test_read_missing_name(self, tmp_path):
        handler = _make_handler(tmp_path, "sakura")
        result = handler.handle("read_subordinate_state", {})
        assert "InvalidArguments" in result

    def test_read_grandchild(self, tmp_path):
        """Can read grandchild (not just direct subordinate)."""
        handler = _make_handler(tmp_path, "sakura")
        _setup_subordinate(tmp_path, "hinata", supervisor="sakura")
        sub_dir = _setup_subordinate(tmp_path, "natsume", supervisor="hinata")
        (sub_dir / "state" / "current_task.md").write_text(
            "設計中", encoding="utf-8",
        )

        mock_cfg = _mock_config(tmp_path, {
            "sakura": {},
            "hinata": {"supervisor": "sakura"},
            "natsume": {"supervisor": "hinata"},
        })

        with (
            patch("core.config.models.load_config", return_value=mock_cfg),
            patch("core.paths.get_animas_dir", return_value=tmp_path / "animas"),
        ):
            result = handler.handle("read_subordinate_state", {"name": "natsume"})

        assert "設計中" in result


# ── check_permissions tests ────────────────────────────────


class TestCheckPermissions:

    def test_returns_internal_tools(self, tmp_path):
        handler = _make_handler(tmp_path, "sakura")
        result = handler.handle("check_permissions", {})
        parsed = json.loads(result)
        assert "internal_tools" in parsed
        assert "search_memory" in parsed["internal_tools"]
        assert "check_permissions" in parsed["internal_tools"]

    def test_returns_file_access(self, tmp_path):
        handler = _make_handler(tmp_path, "sakura")
        result = handler.handle("check_permissions", {})
        parsed = json.loads(result)
        assert "file_access" in parsed
        assert "自分のディレクトリ" in parsed["file_access"]["read"]

    def test_always_available(self, tmp_path):
        """check_permissions should be available to all animas (not just supervisors)."""
        handler = _make_handler(tmp_path, "nobody")
        result = handler.handle("check_permissions", {})
        parsed = json.loads(result)
        assert "internal_tools" in parsed


# ── delegate_task tests ────────────────────────────────────


class TestDelegateTask:

    def test_delegate_to_subordinate(self, tmp_path):
        messenger = MagicMock()
        msg_mock = MagicMock()
        msg_mock.id = "msg1"
        msg_mock.thread_id = "t1"
        msg_mock.type = "message"
        messenger.send.return_value = msg_mock

        handler = _make_handler(tmp_path, "sakura", messenger=messenger)
        _setup_subordinate(tmp_path, "hinata", supervisor="sakura")

        mock_cfg = _mock_config(tmp_path, {
            "sakura": {},
            "hinata": {"supervisor": "sakura"},
        })

        with (
            patch("core.config.models.load_config", return_value=mock_cfg),
            patch("core.paths.get_animas_dir", return_value=tmp_path / "animas"),
        ):
            result = handler.handle("delegate_task", {
                "name": "hinata",
                "instruction": "レポートを作成してください",
                "summary": "レポート作成",
                "deadline": "2h",
            })

        assert "委譲しました" in result
        assert "hinata" in result

        # Check subordinate's task queue was created
        sub_queue = tmp_path / "animas" / "hinata" / "state" / "task_queue.jsonl"
        assert sub_queue.exists()

        # Check own tracking entry
        own_queue = tmp_path / "animas" / "sakura" / "state" / "task_queue.jsonl"
        assert own_queue.exists()
        lines = own_queue.read_text(encoding="utf-8").strip().split("\n")
        own_task = json.loads(lines[-1])
        assert own_task["status"] == "delegated"
        assert own_task["meta"]["delegated_to"] == "hinata"

    def test_delegate_to_non_descendant(self, tmp_path):
        handler = _make_handler(tmp_path, "sakura")
        mock_cfg = _mock_config(tmp_path, {
            "sakura": {},
            "mio": {"supervisor": "taka"},
        })

        with patch("core.config.models.load_config", return_value=mock_cfg):
            result = handler.handle("delegate_task", {
                "name": "mio",
                "instruction": "test",
                "deadline": "1h",
            })
        assert "PermissionDenied" in result
        assert "直属部下ではありません" in result

    def test_delegate_missing_fields(self, tmp_path):
        handler = _make_handler(tmp_path, "sakura")
        result = handler.handle("delegate_task", {})
        assert "InvalidArguments" in result

    def test_delegate_without_messenger(self, tmp_path):
        handler = _make_handler(tmp_path, "sakura", messenger=None)
        _setup_subordinate(tmp_path, "hinata", supervisor="sakura")

        mock_cfg = _mock_config(tmp_path, {
            "sakura": {},
            "hinata": {"supervisor": "sakura"},
        })

        with (
            patch("core.config.models.load_config", return_value=mock_cfg),
            patch("core.paths.get_animas_dir", return_value=tmp_path / "animas"),
        ):
            result = handler.handle("delegate_task", {
                "name": "hinata",
                "instruction": "test task",
                "deadline": "1h",
            })

        assert "メッセンジャー未設定" in result
        # Task should still be added to queues
        sub_queue = tmp_path / "animas" / "hinata" / "state" / "task_queue.jsonl"
        assert sub_queue.exists()


# ── task_tracker tests ─────────────────────────────────────


class TestTaskTracker:

    def test_no_delegated_tasks(self, tmp_path):
        handler = _make_handler(tmp_path, "sakura")
        result = handler.handle("task_tracker", {})
        assert "委譲済みタスクはありません" in result

    def test_tracks_delegated_task(self, tmp_path):
        messenger = MagicMock()
        msg_mock = MagicMock()
        msg_mock.id = "msg1"
        msg_mock.thread_id = "t1"
        msg_mock.type = "message"
        messenger.send.return_value = msg_mock

        handler = _make_handler(tmp_path, "sakura", messenger=messenger)
        _setup_subordinate(tmp_path, "hinata", supervisor="sakura")

        mock_cfg = _mock_config(tmp_path, {
            "sakura": {},
            "hinata": {"supervisor": "sakura"},
        })

        with (
            patch("core.config.models.load_config", return_value=mock_cfg),
            patch("core.paths.get_animas_dir", return_value=tmp_path / "animas"),
        ):
            # First delegate a task
            handler.handle("delegate_task", {
                "name": "hinata",
                "instruction": "テスト作成",
                "deadline": "2h",
            })

            # Then track it
            result = handler.handle("task_tracker", {"status": "all"})

        parsed = json.loads(result)
        assert len(parsed) >= 1
        assert parsed[0]["delegated_to"] == "hinata"
        assert parsed[0]["subordinate_status"] == "pending"


# ── File permission extension tests ────────────────────────


class TestDescendantFilePermission:

    def test_descendant_activity_log_readable(self, tmp_path):
        """Supervisor can read descendant's activity_log via read_file."""
        animas_dir = tmp_path / "animas"

        # Setup supervisor
        sakura_dir = animas_dir / "sakura"
        sakura_dir.mkdir(parents=True)
        (sakura_dir / "permissions.md").write_text("", encoding="utf-8")

        # Setup grandchild
        natsume_dir = animas_dir / "natsume"
        natsume_dir.mkdir(parents=True)
        (natsume_dir / "activity_log").mkdir()
        (natsume_dir / "activity_log" / "2026-02-25.jsonl").write_text(
            '{"ts":"2026-02-25T10:00:00","type":"test"}', encoding="utf-8",
        )

        mock_cfg = _mock_config(tmp_path, {
            "sakura": {},
            "hinata": {"supervisor": "sakura"},
            "natsume": {"supervisor": "hinata"},
        })

        memory = MagicMock()
        memory.read_permissions.return_value = ""

        with (
            patch("core.config.models.load_config", return_value=mock_cfg),
            patch("core.paths.get_animas_dir", return_value=animas_dir),
        ):
            handler = ToolHandler(
                anima_dir=sakura_dir,
                memory=memory,
                tool_registry=[],
            )

        # _check_file_permission should allow reading descendant's activity_log
        result = handler._check_file_permission(
            str(natsume_dir / "activity_log" / "2026-02-25.jsonl"),
        )
        assert result is None  # None means allowed

    def test_descendant_state_readable(self, tmp_path):
        """Supervisor can read descendant's state files."""
        animas_dir = tmp_path / "animas"

        sakura_dir = animas_dir / "sakura"
        sakura_dir.mkdir(parents=True)
        (sakura_dir / "permissions.md").write_text("", encoding="utf-8")

        natsume_dir = animas_dir / "natsume"
        natsume_dir.mkdir(parents=True)
        (natsume_dir / "state").mkdir()
        (natsume_dir / "state" / "current_task.md").write_text("busy", encoding="utf-8")

        mock_cfg = _mock_config(tmp_path, {
            "sakura": {},
            "hinata": {"supervisor": "sakura"},
            "natsume": {"supervisor": "hinata"},
        })

        memory = MagicMock()
        memory.read_permissions.return_value = ""

        with (
            patch("core.config.models.load_config", return_value=mock_cfg),
            patch("core.paths.get_animas_dir", return_value=animas_dir),
        ):
            handler = ToolHandler(
                anima_dir=sakura_dir,
                memory=memory,
                tool_registry=[],
            )

        result = handler._check_file_permission(
            str(natsume_dir / "state" / "current_task.md"),
        )
        assert result is None


# ── Descendant management file permission tests ───────────


def _make_handler_with_hierarchy(tmp_path: Path, *, anima_name: str = "sakura") -> tuple[ToolHandler, Path]:
    """Create a ToolHandler with a 3-level hierarchy: sakura -> hinata -> natsume."""
    animas_dir = tmp_path / "animas"
    sakura_dir = animas_dir / anima_name
    sakura_dir.mkdir(parents=True)
    (sakura_dir / "permissions.md").write_text("", encoding="utf-8")
    (sakura_dir / "state").mkdir(exist_ok=True)

    for sub in ("hinata", "natsume"):
        d = animas_dir / sub
        d.mkdir(parents=True, exist_ok=True)
        (d / "state").mkdir(exist_ok=True)
        (d / "state" / "pending").mkdir(exist_ok=True)
        (d / "activity_log").mkdir(exist_ok=True)
        for fname in ("cron.md", "heartbeat.md", "status.json", "injection.md", "identity.md"):
            (d / fname).write_text("", encoding="utf-8")
        (d / "state" / "current_task.md").write_text("", encoding="utf-8")
        (d / "state" / "pending.md").write_text("", encoding="utf-8")
        (d / "state" / "task_queue.jsonl").write_text("", encoding="utf-8")

    mock_cfg = _mock_config(tmp_path, {
        anima_name: {},
        "hinata": {"supervisor": anima_name},
        "natsume": {"supervisor": "hinata"},
    })

    memory = MagicMock()
    memory.read_permissions.return_value = ""

    with (
        patch("core.config.models.load_config", return_value=mock_cfg),
        patch("core.paths.get_animas_dir", return_value=animas_dir),
    ):
        handler = ToolHandler(
            anima_dir=sakura_dir,
            memory=memory,
            tool_registry=[],
        )
    return handler, animas_dir


class TestDescendantManagementFilePermission:
    """Grandchild management files (cron.md, heartbeat.md, status.json, injection.md) should be read/write."""

    def test_grandchild_cron_readable(self, tmp_path):
        handler, animas_dir = _make_handler_with_hierarchy(tmp_path)
        result = handler._check_file_permission(str(animas_dir / "natsume" / "cron.md"))
        assert result is None

    def test_grandchild_cron_writable(self, tmp_path):
        handler, animas_dir = _make_handler_with_hierarchy(tmp_path)
        result = handler._check_file_permission(str(animas_dir / "natsume" / "cron.md"), write=True)
        assert result is None

    def test_grandchild_heartbeat_readable(self, tmp_path):
        handler, animas_dir = _make_handler_with_hierarchy(tmp_path)
        result = handler._check_file_permission(str(animas_dir / "natsume" / "heartbeat.md"))
        assert result is None

    def test_grandchild_heartbeat_writable(self, tmp_path):
        handler, animas_dir = _make_handler_with_hierarchy(tmp_path)
        result = handler._check_file_permission(str(animas_dir / "natsume" / "heartbeat.md"), write=True)
        assert result is None

    def test_grandchild_injection_writable(self, tmp_path):
        handler, animas_dir = _make_handler_with_hierarchy(tmp_path)
        result = handler._check_file_permission(str(animas_dir / "natsume" / "injection.md"), write=True)
        assert result is None

    def test_grandchild_status_json_writable(self, tmp_path):
        handler, animas_dir = _make_handler_with_hierarchy(tmp_path)
        result = handler._check_file_permission(str(animas_dir / "natsume" / "status.json"), write=True)
        assert result is None

    def test_grandchild_identity_read_only(self, tmp_path):
        """identity.md must remain read-only even for descendants."""
        handler, animas_dir = _make_handler_with_hierarchy(tmp_path)
        result = handler._check_file_permission(str(animas_dir / "natsume" / "identity.md"))
        assert result is None
        result = handler._check_file_permission(str(animas_dir / "natsume" / "identity.md"), write=True)
        assert result is not None

    def test_grandchild_root_dir_listable(self, tmp_path):
        handler, animas_dir = _make_handler_with_hierarchy(tmp_path)
        result = handler._check_file_permission(str(animas_dir / "natsume"))
        assert result is None

    def test_direct_child_still_has_management_rw(self, tmp_path):
        """Direct child management files must remain read/write after refactor."""
        handler, animas_dir = _make_handler_with_hierarchy(tmp_path)
        for fname in ("cron.md", "heartbeat.md", "status.json", "injection.md"):
            result = handler._check_file_permission(str(animas_dir / "hinata" / fname), write=True)
            assert result is None, f"Write to direct child {fname} should be allowed"


class TestDescendantOrgToolPermission:
    """Org tools (disable/enable/restart/set_model/set_bg_model/delegate_task) should work for grandchildren."""

    def test_disable_grandchild(self, tmp_path):
        handler, animas_dir = _make_handler_with_hierarchy(tmp_path)
        (animas_dir / "natsume" / "status.json").write_text(
            json.dumps({"enabled": True, "supervisor": "hinata"}), encoding="utf-8",
        )
        mock_cfg = _mock_config(tmp_path, {
            "sakura": {},
            "hinata": {"supervisor": "sakura"},
            "natsume": {"supervisor": "hinata"},
        })
        with (
            patch("core.config.models.load_config", return_value=mock_cfg),
            patch("core.paths.get_animas_dir", return_value=animas_dir),
        ):
            result = handler.handle("disable_subordinate", {"name": "natsume", "reason": "test"})
        assert "PermissionDenied" not in result
        status = json.loads((animas_dir / "natsume" / "status.json").read_text(encoding="utf-8"))
        assert status["enabled"] is False

    def test_enable_grandchild(self, tmp_path):
        handler, animas_dir = _make_handler_with_hierarchy(tmp_path)
        (animas_dir / "natsume" / "status.json").write_text(
            json.dumps({"enabled": False, "supervisor": "hinata"}), encoding="utf-8",
        )
        mock_cfg = _mock_config(tmp_path, {
            "sakura": {},
            "hinata": {"supervisor": "sakura"},
            "natsume": {"supervisor": "hinata"},
        })
        with (
            patch("core.config.models.load_config", return_value=mock_cfg),
            patch("core.paths.get_animas_dir", return_value=animas_dir),
        ):
            result = handler.handle("enable_subordinate", {"name": "natsume"})
        assert "PermissionDenied" not in result

    def test_restart_grandchild(self, tmp_path):
        handler, animas_dir = _make_handler_with_hierarchy(tmp_path)
        (animas_dir / "natsume" / "status.json").write_text(
            json.dumps({"enabled": True, "supervisor": "hinata"}), encoding="utf-8",
        )
        mock_cfg = _mock_config(tmp_path, {
            "sakura": {},
            "hinata": {"supervisor": "sakura"},
            "natsume": {"supervisor": "hinata"},
        })
        with (
            patch("core.config.models.load_config", return_value=mock_cfg),
            patch("core.paths.get_animas_dir", return_value=animas_dir),
        ):
            result = handler.handle("restart_subordinate", {"name": "natsume", "reason": "test"})
        assert "PermissionDenied" not in result

    def test_delegate_task_to_grandchild(self, tmp_path):
        messenger = MagicMock()
        msg_mock = MagicMock()
        msg_mock.id = "msg1"
        msg_mock.thread_id = "t1"
        msg_mock.type = "message"
        messenger.send.return_value = msg_mock

        handler, animas_dir = _make_handler_with_hierarchy(tmp_path)
        handler._messenger = messenger

        mock_cfg = _mock_config(tmp_path, {
            "sakura": {},
            "hinata": {"supervisor": "sakura"},
            "natsume": {"supervisor": "hinata"},
        })
        with (
            patch("core.config.models.load_config", return_value=mock_cfg),
            patch("core.paths.get_animas_dir", return_value=animas_dir),
        ):
            result = handler.handle("delegate_task", {
                "name": "natsume",
                "instruction": "テストタスク",
                "summary": "テスト",
                "deadline": "2h",
            })
        assert "PermissionDenied" in result
        assert "直属部下ではありません" in result

    def test_non_descendant_still_blocked(self, tmp_path):
        """Org tools should still block non-descendant targets."""
        handler, animas_dir = _make_handler_with_hierarchy(tmp_path)
        _setup_subordinate(tmp_path, "mio", supervisor="taka")
        mock_cfg = _mock_config(tmp_path, {
            "sakura": {},
            "hinata": {"supervisor": "sakura"},
            "natsume": {"supervisor": "hinata"},
            "mio": {"supervisor": "taka"},
        })
        with patch("core.config.models.load_config", return_value=mock_cfg):
            result = handler.handle("disable_subordinate", {"name": "mio"})
        assert "PermissionDenied" in result


# ── Mode S descendant management file tests ────────────────


class TestSdkHooksDescendantManagementFiles:
    """_cache_subordinate_paths should include grandchild management files."""

    def test_grandchild_mgmt_files_in_cache(self, tmp_path):
        from core.execution._sdk_hooks import _cache_subordinate_paths

        animas_dir = tmp_path / "animas"
        sakura_dir = animas_dir / "sakura"
        sakura_dir.mkdir(parents=True)
        for name in ("hinata", "natsume"):
            d = animas_dir / name
            d.mkdir(parents=True, exist_ok=True)
            (d / "state").mkdir(exist_ok=True)

        mock_cfg = _mock_config(tmp_path, {
            "sakura": {},
            "hinata": {"supervisor": "sakura"},
            "natsume": {"supervisor": "hinata"},
        })
        with (
            patch("core.config.models.load_config", return_value=mock_cfg),
            patch("core.paths.get_animas_dir", return_value=animas_dir),
        ):
            _, sub_mgmt_files, _, _, _ = _cache_subordinate_paths(sakura_dir)

        natsume_dir = (animas_dir / "natsume").resolve()
        mgmt_names = {p.name for p in sub_mgmt_files if p.parent == natsume_dir}
        assert mgmt_names == {"cron.md", "heartbeat.md", "status.json", "injection.md"}

    def test_direct_child_mgmt_files_still_present(self, tmp_path):
        from core.execution._sdk_hooks import _cache_subordinate_paths

        animas_dir = tmp_path / "animas"
        sakura_dir = animas_dir / "sakura"
        sakura_dir.mkdir(parents=True)
        for name in ("hinata", "natsume"):
            d = animas_dir / name
            d.mkdir(parents=True, exist_ok=True)
            (d / "state").mkdir(exist_ok=True)

        mock_cfg = _mock_config(tmp_path, {
            "sakura": {},
            "hinata": {"supervisor": "sakura"},
            "natsume": {"supervisor": "hinata"},
        })
        with (
            patch("core.config.models.load_config", return_value=mock_cfg),
            patch("core.paths.get_animas_dir", return_value=animas_dir),
        ):
            _, sub_mgmt_files, _, _, _ = _cache_subordinate_paths(sakura_dir)

        hinata_dir = (animas_dir / "hinata").resolve()
        mgmt_names = {p.name for p in sub_mgmt_files if p.parent == hinata_dir}
        assert mgmt_names == {"cron.md", "heartbeat.md", "status.json", "injection.md"}


# ── build_tool_list integration tests ──────────────────────


class TestBuildToolListIntegration:

    def test_check_permissions_always_included(self):
        """check_permissions should always be in the tool list."""
        from core.tooling.schemas import build_tool_list

        tools = build_tool_list()
        names = {t["name"] for t in tools}
        assert "check_permissions" in names

    def test_supervisor_tools_include_new_tools(self):
        """When supervisor tools are enabled, new tools are included."""
        from core.tooling.schemas import build_tool_list

        tools = build_tool_list(include_supervisor_tools=True)
        names = {t["name"] for t in tools}
        for expected in [
            "org_dashboard", "ping_subordinate", "read_subordinate_state",
            "delegate_task", "task_tracker",
        ]:
            assert expected in names, f"Missing tool: {expected}"
