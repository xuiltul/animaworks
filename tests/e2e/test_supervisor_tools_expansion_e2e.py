"""E2E tests for supervisor tools expansion — full delegation workflow."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.config.models import AnimaModelConfig
from core.memory.manager import MemoryManager
from core.tooling.handler import ToolHandler


def _build_config(animas: dict[str, dict]) -> MagicMock:
    config = MagicMock()
    config.locale = "ja"
    config.animas = {
        name: AnimaModelConfig(**fields)
        for name, fields in animas.items()
    }
    config.heartbeat = MagicMock()
    config.heartbeat.channel_post_cooldown_s = 0
    return config


def _setup_org(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    """Create a 3-level org: sakura -> hinata -> natsume."""
    animas_dir = tmp_path / "animas"

    dirs = {}
    for name in ["sakura", "hinata", "natsume"]:
        d = animas_dir / name
        d.mkdir(parents=True)
        for sub in ["state", "episodes", "knowledge", "procedures", "skills", "activity_log"]:
            (d / sub).mkdir(exist_ok=True)
        (d / "permissions.md").write_text("", encoding="utf-8")
        dirs[name] = d

    # Status files
    for name, sup in [("sakura", None), ("hinata", "sakura"), ("natsume", "hinata")]:
        status = {"enabled": True, "role": "general"}
        if sup:
            status["supervisor"] = sup
        (animas_dir / name / "status.json").write_text(
            json.dumps(status), encoding="utf-8",
        )

    return animas_dir, dirs


class TestDelegationWorkflowE2E:
    """End-to-end test of the full delegation + tracking cycle."""

    def test_full_delegation_cycle(self, tmp_path):
        animas_dir, dirs = _setup_org(tmp_path)

        mock_cfg = _build_config({
            "sakura": {},
            "hinata": {"supervisor": "sakura"},
            "natsume": {"supervisor": "hinata"},
        })

        messenger = MagicMock()
        msg_mock = MagicMock()
        msg_mock.id = "m1"
        msg_mock.thread_id = "t1"
        msg_mock.type = "message"
        messenger.send.return_value = msg_mock
        messenger.anima_name = "sakura"

        with (
            patch("core.config.models.load_config", return_value=mock_cfg),
            patch("core.paths.get_animas_dir", return_value=animas_dir),
        ):
            memory = MemoryManager(dirs["sakura"])
            handler = ToolHandler(
                anima_dir=dirs["sakura"],
                memory=memory,
                messenger=messenger,
                tool_registry=[],
            )

            # 1. Check org dashboard
            result = handler.handle("org_dashboard", {})
            assert "hinata" in result
            assert "natsume" in result

            # 2. Ping all descendants
            result = handler.handle("ping_subordinate", {})
            parsed = json.loads(result)
            names = {r["name"] for r in parsed}
            assert "hinata" in names
            assert "natsume" in names

            # 3. Read grandchild's state (initially empty)
            result = handler.handle("read_subordinate_state", {"name": "natsume"})
            assert "(なし)" in result

            # 4. Delegate task to hinata
            result = handler.handle("delegate_task", {
                "name": "hinata",
                "instruction": "natsume にデータ収集を指示して結果をまとめてください",
                "summary": "データ収集まとめ",
                "deadline": "3h",
            })
            assert "委譲しました" in result
            assert "hinata" in result

            # Verify DM was sent
            messenger.send.assert_called()

            # 5. Track the delegated task
            result = handler.handle("task_tracker", {"status": "all"})
            parsed = json.loads(result)
            assert len(parsed) == 1
            assert parsed[0]["delegated_to"] == "hinata"
            assert parsed[0]["subordinate_status"] == "pending"

            # 6. Simulate subordinate completing the task
            from core.memory.task_queue import TaskQueueManager
            sub_tqm = TaskQueueManager(dirs["hinata"])
            tasks = sub_tqm.list_tasks()
            assert len(tasks) == 1
            sub_tqm.update_status(tasks[0].task_id, "done")

            # 7. Re-track — should show completed
            result = handler.handle("task_tracker", {"status": "all"})
            parsed = json.loads(result)
            assert parsed[0]["subordinate_status"] == "done"

            # 8. Check permissions
            result = handler.handle("check_permissions", {})
            parsed = json.loads(result)
            assert "delegate_task" in parsed["internal_tools"]
            assert "org_dashboard" in parsed["internal_tools"]


class TestPermissionBoundariesE2E:
    """Verify permission boundaries are enforced correctly."""

    def test_non_supervisor_cannot_use_supervisor_tools(self, tmp_path):
        """An anima with no subordinates gets rejected by descendant checks."""
        animas_dir, dirs = _setup_org(tmp_path)

        mock_cfg = _build_config({
            "sakura": {},
            "hinata": {"supervisor": "sakura"},
            "natsume": {"supervisor": "hinata"},
        })

        with (
            patch("core.config.models.load_config", return_value=mock_cfg),
            patch("core.paths.get_animas_dir", return_value=animas_dir),
        ):
            memory = MemoryManager(dirs["natsume"])
            handler = ToolHandler(
                anima_dir=dirs["natsume"],
                memory=memory,
                tool_registry=[],
            )

            # natsume has no subordinates
            result = handler.handle("org_dashboard", {})
            assert "配下の Anima はいません" in result

            # Cannot delegate to sakura (not a subordinate)
            result = handler.handle("delegate_task", {
                "name": "sakura",
                "instruction": "test",
                "deadline": "1h",
            })
            assert "PermissionDenied" in result

    def test_delegate_only_to_direct_subordinate(self, tmp_path):
        """Cannot delegate to grandchild — only direct subordinates."""
        animas_dir, dirs = _setup_org(tmp_path)

        mock_cfg = _build_config({
            "sakura": {},
            "hinata": {"supervisor": "sakura"},
            "natsume": {"supervisor": "hinata"},
        })

        with (
            patch("core.config.models.load_config", return_value=mock_cfg),
            patch("core.paths.get_animas_dir", return_value=animas_dir),
        ):
            memory = MemoryManager(dirs["sakura"])
            handler = ToolHandler(
                anima_dir=dirs["sakura"],
                memory=memory,
                tool_registry=[],
            )

            result = handler.handle("delegate_task", {
                "name": "natsume",
                "instruction": "test",
                "deadline": "1h",
            })
            assert "PermissionDenied" in result
            assert "直属部下ではありません" in result
