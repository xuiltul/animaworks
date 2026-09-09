from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Delegation proxies one atomic publication when local SQLite is denied."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from core.exceptions import TaskPersistenceError
from core.memory.task_queue import TaskQueueManager
from core.tooling.handler import ToolHandler


def _make_handler(tmp_path: Path, anima_name: str = "rin") -> ToolHandler:
    anima_dir = tmp_path / "animas" / anima_name
    anima_dir.mkdir(parents=True, exist_ok=True)
    (anima_dir / "permissions.md").write_text("", encoding="utf-8")
    (anima_dir / "state").mkdir(exist_ok=True)
    (anima_dir / "status.json").write_text("{}", encoding="utf-8")

    memory = MagicMock()
    memory.read_permissions.return_value = ""
    messenger = MagicMock()
    msg = MagicMock()
    msg.id = "m1"
    msg.thread_id = "t1"
    messenger.send.return_value = msg

    return ToolHandler(
        anima_dir=anima_dir,
        memory=memory,
        messenger=messenger,
    )


def _setup_target(tmp_path: Path, name: str = "natsume") -> Path:
    target = tmp_path / "animas" / name
    target.mkdir(parents=True, exist_ok=True)
    (target / "state").mkdir(exist_ok=True)
    (target / "status.json").write_text("{}", encoding="utf-8")
    return target


def _delegate_args() -> dict:
    return {
        "name": "natsume",
        "instruction": "resolve PR conflicts",
        "summary": "PR conflicts",
        "deadline": "2h",
    }


class TestDelegateTaskErofsFallback:
    @pytest.mark.parametrize("wrapped", [False, True])
    def test_permission_failure_proxies_whole_transaction(self, tmp_path, monkeypatch, wrapped):
        handler = _make_handler(tmp_path)
        target = _setup_target(tmp_path)
        monkeypatch.setenv("ANIMAWORKS_SERVER_URL", "http://server.test:18500")
        denied = OSError(30, "Read-only file system")
        if wrapped:
            error = TaskPersistenceError("persistence failed")
            error.__cause__ = denied
        else:
            error = denied
        response = httpx.Response(200, json={"ok": True}, request=httpx.Request("POST", "http://server.test"))
        with (
            patch.object(handler, "_check_subordinate", return_value=None),
            patch("core.paths.get_animas_dir", return_value=tmp_path / "animas"),
            patch.object(TaskQueueManager, "submit", side_effect=error),
            patch("httpx.post", return_value=response) as post,
            patch("core.tooling.handler_delegation._record_taskboard_delegation") as board,
        ):
            result = handler.handle("delegate_task", _delegate_args())
        assert not result.strip().startswith("{")
        sent = post.call_args.kwargs["json"]
        assert sent["delegator"] == "rin" and sent["target"] == "natsume"
        assert sent["instruction"] == "resolve PR conflicts"
        assert sent["sub_task_id"] in result
        assert sent["tracking_task_id"] in result
        assert not any(key.startswith("persist_") for key in sent)
        board.assert_not_called()
        assert TaskQueueManager(target).list_tasks() == []

    def test_alias_write_denied_rolls_back_subordinate_before_proxy(self, tmp_path):
        from core.taskboard.tasks import TaskStore

        handler = _make_handler(tmp_path)
        target = _setup_target(tmp_path)
        response = httpx.Response(200, json={"ok": True}, request=httpx.Request("POST", "http://server.test"))
        with (
            patch.object(handler, "_check_subordinate", return_value=None),
            patch("core.paths.get_animas_dir", return_value=tmp_path / "animas"),
            patch.object(TaskStore, "alias", side_effect=OSError(30, "Read-only file system")),
            patch("httpx.post", return_value=response) as post,
        ):
            result = handler.handle("delegate_task", _delegate_args())
        assert "PersistenceFailed" not in result
        post.assert_called_once()
        assert TaskQueueManager(target).list_tasks() == []

    @pytest.mark.parametrize("failure", ["transport", "server"])
    def test_host_failure_reports_persistence_failed(self, tmp_path, failure):
        handler = _make_handler(tmp_path)
        _setup_target(tmp_path)
        response = httpx.Response(
            500, json={"detail": "disk full"}, request=httpx.Request("POST", "http://server.test")
        )
        with (
            patch.object(handler, "_check_subordinate", return_value=None),
            patch("core.paths.get_animas_dir", return_value=tmp_path / "animas"),
            patch.object(TaskQueueManager, "submit", side_effect=OSError(30, "Read-only file system")),
            patch(
                "httpx.post",
                side_effect=httpx.ConnectError("down") if failure == "transport" else None,
                return_value=response,
            ) as post,
        ):
            result = handler.handle("delegate_task", _delegate_args())
        assert json.loads(result)["error_type"] == "PersistenceFailed"
        assert post.call_count == 2
        assert post.call_args_list[0].kwargs["json"] == post.call_args_list[1].kwargs["json"]

    def test_lost_response_retries_same_task_identity(self, tmp_path):
        handler = _make_handler(tmp_path)
        _setup_target(tmp_path)
        response = httpx.Response(200, json={"ok": True}, request=httpx.Request("POST", "http://server.test"))
        with (
            patch.object(handler, "_check_subordinate", return_value=None),
            patch("core.paths.get_animas_dir", return_value=tmp_path / "animas"),
            patch.object(TaskQueueManager, "submit", side_effect=OSError(30, "Read-only file system")),
            patch("httpx.post", side_effect=[httpx.ReadTimeout("response lost"), response]) as post,
        ):
            result = handler.handle("delegate_task", _delegate_args())
        assert "PersistenceFailed" not in result
        assert post.call_count == 2
        assert post.call_args_list[0].kwargs["json"] == post.call_args_list[1].kwargs["json"]

    def test_direct_success_needs_no_http_or_pending_file(self, tmp_path):
        handler = _make_handler(tmp_path)
        target = _setup_target(tmp_path)
        with (
            patch.object(handler, "_check_subordinate", return_value=None),
            patch("core.paths.get_animas_dir", return_value=tmp_path / "animas"),
            patch("httpx.post") as post,
        ):
            result = handler.handle("delegate_task", _delegate_args())
        assert "PersistenceFailed" not in result
        post.assert_not_called()
        assert len(TaskQueueManager(target).list_tasks()) == 1
        assert not list((target / "state" / "pending").glob("*.json"))

    def test_mcp_env_includes_server_url(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Codex MCP env must inject ANIMAWORKS_SERVER_URL (contract for EROFS fallback)."""
        from core.execution.codex_sdk import CodexSDKExecutor
        from core.schemas import ModelConfig

        anima_dir = tmp_path / "animas" / "rin"
        anima_dir.mkdir(parents=True)
        model = ModelConfig(model="codex/gpt-5.6-sol", api_key="sk-test")
        executor = CodexSDKExecutor(model_config=model, anima_dir=anima_dir)
        monkeypatch.delenv("ANIMAWORKS_SERVER_URL", raising=False)
        env = executor._build_mcp_env()
        assert "ANIMAWORKS_SERVER_URL" in env
        assert env["ANIMAWORKS_SERVER_URL"].startswith("http")
        assert "18500" in env["ANIMAWORKS_SERVER_URL"]
