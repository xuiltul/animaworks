"""Unit tests for Task tool delegation + SDK subagent branching."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ── Mock claude_agent_sdk before importing ────────────────────
_mock_sdk = MagicMock()
_mock_types = MagicMock()


def _sync_hook_json_output(**kwargs: Any) -> dict[str, Any]:
    return dict(kwargs)


_mock_types.SyncHookJSONOutput = _sync_hook_json_output
_mock_types.PreToolUseHookSpecificOutput = dict
_mock_types.HookInput = dict
_mock_types.HookContext = dict

sys.modules.setdefault("claude_agent_sdk", _mock_sdk)
sys.modules.setdefault("claude_agent_sdk.types", _mock_types)

from core.execution._sdk_hooks import (  # noqa: E402
    _count_active_tasks,
    _intercept_task_to_delegation,
    _read_status_json,
    _role_matches,
    _select_subordinate,
)


# ── _read_status_json ─────────────────────────────────────────


class TestReadStatusJson:

    def test_reads_valid_json(self, tmp_path: Path) -> None:
        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()
        (anima_dir / "status.json").write_text(
            json.dumps({"enabled": True, "role": "engineer"}),
            encoding="utf-8",
        )
        result = _read_status_json(anima_dir)
        assert result["enabled"] is True
        assert result["role"] == "engineer"

    def test_returns_empty_on_missing(self, tmp_path: Path) -> None:
        assert _read_status_json(tmp_path / "nonexistent") == {}

    def test_returns_empty_on_invalid_json(self, tmp_path: Path) -> None:
        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()
        (anima_dir / "status.json").write_text("not json", encoding="utf-8")
        assert _read_status_json(anima_dir) == {}


# ── _count_active_tasks ───────────────────────────────────────


class TestCountActiveTasks:

    def test_counts_pending_and_in_progress(self, tmp_path: Path) -> None:
        anima_dir = tmp_path / "anima"
        state_dir = anima_dir / "state"
        state_dir.mkdir(parents=True)

        entries = [
            {"task_id": "t1", "status": "pending"},
            {"task_id": "t2", "status": "in_progress"},
            {"task_id": "t3", "status": "done"},
        ]
        (state_dir / "task_queue.jsonl").write_text(
            "\n".join(json.dumps(e) for e in entries) + "\n",
            encoding="utf-8",
        )
        assert _count_active_tasks(anima_dir) == 2

    def test_handles_status_updates(self, tmp_path: Path) -> None:
        anima_dir = tmp_path / "anima"
        state_dir = anima_dir / "state"
        state_dir.mkdir(parents=True)

        lines = [
            json.dumps({"task_id": "t1", "status": "pending"}),
            json.dumps({"task_id": "t1", "status": "done", "_event": "update"}),
        ]
        (state_dir / "task_queue.jsonl").write_text(
            "\n".join(lines) + "\n", encoding="utf-8",
        )
        assert _count_active_tasks(anima_dir) == 0

    def test_returns_zero_on_missing_file(self, tmp_path: Path) -> None:
        assert _count_active_tasks(tmp_path / "anima") == 0


# ── _role_matches ─────────────────────────────────────────────


class TestRoleMatches:

    def test_matches_role(self) -> None:
        assert _role_matches({"role": "engineer"}, "Need an engineer to fix this")

    def test_matches_specialty(self) -> None:
        assert _role_matches({"specialty": "frontend"}, "Build frontend component")

    def test_no_match(self) -> None:
        assert not _role_matches({"role": "writer"}, "Deploy the server")

    def test_empty_status(self) -> None:
        assert not _role_matches({}, "any description")


# ── _select_subordinate ──────────────────────────────────────


class TestSelectSubordinate:

    def _make_config(self, animas: dict[str, dict]) -> SimpleNamespace:
        cfg = SimpleNamespace()
        cfg.animas = {
            name: SimpleNamespace(**data)
            for name, data in animas.items()
        }
        return cfg

    @patch("core.config.models.load_config")
    @patch("core.paths.get_animas_dir")
    def test_explicit_naming(self, mock_animas_dir, mock_load, tmp_path: Path) -> None:
        """Explicit subordinate name in description takes priority."""
        animas_dir = tmp_path / "animas"
        alice_dir = animas_dir / "alice"
        alice_dir.mkdir(parents=True)
        (alice_dir / "status.json").write_text(
            json.dumps({"enabled": True}), encoding="utf-8",
        )

        mock_animas_dir.return_value = animas_dir
        mock_load.return_value = self._make_config({
            "boss": {"supervisor": None},
            "alice": {"supervisor": "boss"},
            "bob": {"supervisor": "boss"},
        })

        boss_dir = animas_dir / "boss"
        boss_dir.mkdir(parents=True)

        result = _select_subordinate(boss_dir, "Ask alice to handle this")
        assert result == "alice"

    @patch("core.config.models.load_config")
    @patch("core.paths.get_animas_dir")
    def test_workload_selection(self, mock_animas_dir, mock_load, tmp_path: Path) -> None:
        """Selects subordinate with fewer active tasks."""
        animas_dir = tmp_path / "animas"

        alice_dir = animas_dir / "alice"
        alice_dir.mkdir(parents=True)
        (alice_dir / "status.json").write_text(
            json.dumps({"enabled": True}), encoding="utf-8",
        )
        alice_state = alice_dir / "state"
        alice_state.mkdir()
        (alice_state / "task_queue.jsonl").write_text(
            json.dumps({"task_id": "t1", "status": "pending"}) + "\n"
            + json.dumps({"task_id": "t2", "status": "in_progress"}) + "\n",
            encoding="utf-8",
        )

        bob_dir = animas_dir / "bob"
        bob_dir.mkdir(parents=True)
        (bob_dir / "status.json").write_text(
            json.dumps({"enabled": True}), encoding="utf-8",
        )

        mock_animas_dir.return_value = animas_dir
        mock_load.return_value = self._make_config({
            "boss": {"supervisor": None},
            "alice": {"supervisor": "boss"},
            "bob": {"supervisor": "boss"},
        })

        boss_dir = animas_dir / "boss"
        boss_dir.mkdir(parents=True)

        result = _select_subordinate(boss_dir, "do something generic")
        assert result == "bob"

    @patch("core.config.models.load_config")
    @patch("core.paths.get_animas_dir")
    def test_all_disabled_returns_none(self, mock_animas_dir, mock_load, tmp_path: Path) -> None:
        """Returns None when all subordinates are disabled."""
        animas_dir = tmp_path / "animas"
        alice_dir = animas_dir / "alice"
        alice_dir.mkdir(parents=True)
        (alice_dir / "status.json").write_text(
            json.dumps({"enabled": False}), encoding="utf-8",
        )

        mock_animas_dir.return_value = animas_dir
        mock_load.return_value = self._make_config({
            "boss": {"supervisor": None},
            "alice": {"supervisor": "boss"},
        })

        boss_dir = animas_dir / "boss"
        boss_dir.mkdir(parents=True)

        result = _select_subordinate(boss_dir, "any task")
        assert result is None

    @patch("core.config.models.load_config")
    @patch("core.paths.get_animas_dir")
    def test_role_bonus(self, mock_animas_dir, mock_load, tmp_path: Path) -> None:
        """Role-matching subordinate is preferred over lower-workload one."""
        animas_dir = tmp_path / "animas"

        alice_dir = animas_dir / "alice"
        alice_dir.mkdir(parents=True)
        (alice_dir / "status.json").write_text(
            json.dumps({"enabled": True, "role": "engineer"}), encoding="utf-8",
        )
        alice_state = alice_dir / "state"
        alice_state.mkdir()
        (alice_state / "task_queue.jsonl").write_text(
            json.dumps({"task_id": "t1", "status": "pending"}) + "\n",
            encoding="utf-8",
        )

        bob_dir = animas_dir / "bob"
        bob_dir.mkdir(parents=True)
        (bob_dir / "status.json").write_text(
            json.dumps({"enabled": True, "role": "writer"}), encoding="utf-8",
        )

        mock_animas_dir.return_value = animas_dir
        mock_load.return_value = self._make_config({
            "boss": {"supervisor": None},
            "alice": {"supervisor": "boss"},
            "bob": {"supervisor": "boss"},
        })

        boss_dir = animas_dir / "boss"
        boss_dir.mkdir(parents=True)

        result = _select_subordinate(boss_dir, "need an engineer to fix the bug")
        assert result == "alice"

    @patch("core.config.models.load_config")
    @patch("core.paths.get_animas_dir")
    def test_no_subordinates(self, mock_animas_dir, mock_load, tmp_path: Path) -> None:
        """Returns None when anima has no subordinates."""
        animas_dir = tmp_path / "animas"
        mock_animas_dir.return_value = animas_dir
        mock_load.return_value = self._make_config({
            "worker": {"supervisor": "boss"},
        })

        worker_dir = animas_dir / "worker"
        worker_dir.mkdir(parents=True)

        result = _select_subordinate(worker_dir, "any task")
        assert result is None


# ── _intercept_task_to_delegation ─────────────────────────────


class TestInterceptTaskToDelegation:

    @patch("core.execution._sdk_hooks._select_subordinate", return_value=None)
    def test_returns_none_when_no_subordinate(self, mock_select, tmp_path: Path) -> None:
        anima_dir = tmp_path / "animas" / "boss"
        anima_dir.mkdir(parents=True)

        result = _intercept_task_to_delegation(
            anima_dir, {"description": "test", "prompt": "do it"}, None,
        )
        assert result is None

    @patch("core.execution._sdk_hooks._log_tool_use")
    @patch("core.execution._sdk_hooks._select_subordinate", return_value="alice")
    def test_delegation_creates_queue_entries(self, mock_select, mock_log, tmp_path: Path) -> None:
        animas_dir = tmp_path / "animas"
        boss_dir = animas_dir / "boss"
        boss_dir.mkdir(parents=True)
        (boss_dir / "state").mkdir()

        alice_dir = animas_dir / "alice"
        alice_dir.mkdir(parents=True)
        (alice_dir / "state").mkdir()

        shared_dir = tmp_path / "shared"
        inbox_dir = shared_dir / "inbox" / "alice"
        inbox_dir.mkdir(parents=True)

        data_dir = tmp_path / "data"
        run_dir = data_dir / "run" / "inbox_wake"
        run_dir.mkdir(parents=True)

        with patch("core.paths.get_animas_dir", return_value=animas_dir), \
             patch("core.paths.get_shared_dir", return_value=shared_dir), \
             patch("core.paths.get_data_dir", return_value=data_dir):
            result = _intercept_task_to_delegation(
                boss_dir,
                {"description": "Build feature", "prompt": "Implement the login form"},
                "tool-123",
            )

        assert result is not None
        assert "DELEGATION_OK" in result["reason"]
        assert "alice" in result["reason"]
        assert result["task_id"]

        # Verify subordinate queue entry
        alice_queue = alice_dir / "state" / "task_queue.jsonl"
        assert alice_queue.exists()
        entries = [json.loads(l) for l in alice_queue.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(entries) >= 1
        assert entries[0]["assignee"] == "alice"

        # Verify own tracking entry
        boss_queue = boss_dir / "state" / "task_queue.jsonl"
        assert boss_queue.exists()
        own_entries = [json.loads(l) for l in boss_queue.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert any(e.get("status") == "delegated" for e in own_entries)

        # Verify wake file
        assert (run_dir / "alice").exists()


# ── PreToolUse hook branching ─────────────────────────────────


class TestPreToolHookTaskBranching:
    """Verify the has_subordinates branching in PreToolUse hook."""

    @pytest.fixture()
    def hook_no_subs(self, tmp_path: Path):
        """Build hook with has_subordinates=False."""
        try:
            import claude_agent_sdk.types  # noqa: F401
        except ImportError:
            pytest.skip("claude_agent_sdk not installed")

        from core.execution._sdk_hooks import _build_pre_tool_hook

        anima_dir = tmp_path / "animas" / "worker"
        anima_dir.mkdir(parents=True)
        (anima_dir / "state" / "pending").mkdir(parents=True)

        with patch("core.execution._sdk_hooks._cache_subordinate_paths", return_value=([], [], [])):
            return _build_pre_tool_hook(anima_dir, has_subordinates=False)

    @pytest.fixture()
    def hook_with_subs(self, tmp_path: Path):
        """Build hook with has_subordinates=True."""
        try:
            import claude_agent_sdk.types  # noqa: F401
        except ImportError:
            pytest.skip("claude_agent_sdk not installed")

        from core.execution._sdk_hooks import _build_pre_tool_hook

        anima_dir = tmp_path / "animas" / "boss"
        anima_dir.mkdir(parents=True)
        (anima_dir / "state" / "pending").mkdir(parents=True)

        with patch("core.execution._sdk_hooks._cache_subordinate_paths", return_value=([], [], [])):
            return _build_pre_tool_hook(anima_dir, has_subordinates=True)

    async def test_no_subs_passes_through(self, hook_no_subs) -> None:
        """Non-supervisor: Task tool passes through (empty SyncHookJSONOutput)."""
        input_data = {
            "tool_name": "Task",
            "tool_input": {"description": "test", "prompt": "do it"},
        }
        with patch("core.execution._sdk_hooks._log_tool_use"):
            result = await hook_no_subs(input_data, "tool-1", {})

        output = result.get("hookSpecificOutput", {})
        assert output.get("permissionDecision") != "deny"

    async def test_with_subs_delegation_attempted(self, hook_with_subs) -> None:
        """Supervisor: delegation is attempted, falls back to pending on failure."""
        input_data = {
            "tool_name": "Task",
            "tool_input": {"description": "test", "prompt": "do it"},
        }
        with patch("core.execution._sdk_hooks._intercept_task_to_delegation", return_value=None), \
             patch("core.execution._sdk_hooks._log_tool_use"):
            result = await hook_with_subs(input_data, "tool-1", {})

        output = result.get("hookSpecificOutput", {})
        assert output.get("permissionDecision") == "deny"
        assert "INTERCEPT_OK" in output.get("permissionDecisionReason", "")

    async def test_with_subs_delegation_success(self, hook_with_subs) -> None:
        """Supervisor: successful delegation returns DELEGATION_OK."""
        input_data = {
            "tool_name": "Task",
            "tool_input": {"description": "test", "prompt": "do it"},
        }
        delegation_result = {
            "task_id": "abc123",
            "reason": "DELEGATION_OK: delegated to alice (sub_task_id: x, own_tracking_id: abc123). DM sent.",
        }
        with patch("core.execution._sdk_hooks._intercept_task_to_delegation", return_value=delegation_result), \
             patch("core.execution._sdk_hooks._log_tool_use"):
            result = await hook_with_subs(input_data, "tool-1", {})

        output = result.get("hookSpecificOutput", {})
        assert output.get("permissionDecision") == "deny"
        assert "DELEGATION_OK" in output.get("permissionDecisionReason", "")
