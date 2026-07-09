"""Tests for the machine tool module (core/tools/machine.py)."""

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import json
import os
import subprocess
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from core.tools._base import ToolResult
from core.tools.machine import (
    _MAX_CALLS_PER_HEARTBEAT,
    _MAX_CALLS_PER_SESSION,
    _VALID_ENGINES,
    EXECUTION_PROFILE,
    _build_command,
    _build_env,
    _build_instruction,
    _get_available_engines,
    _validate_working_directory,
    dispatch,
    get_tool_schemas,
    reset_call_counts,
)


@pytest.fixture(autouse=True)
def _isolate_machine_config():
    config = SimpleNamespace(
        machine=SimpleNamespace(engine_priority=[], default_models={}),
        credentials={},
        workspaces={},
    )
    with patch("core.config.models.load_config", return_value=config):
        yield


def _set_pipe_output(mock_proc: MagicMock, stdout_text: str, stderr_text: str = "") -> None:
    """Provide file-like stdout/stderr mocks compatible with _stream_to_file()."""
    mock_proc.stdout = io.StringIO(stdout_text)
    mock_proc.stderr = io.StringIO(stderr_text)


# ── Schema Tests ──────────────────────────────────────────


class TestToolSchemas:
    """Schema tests — mock all engines as available for deterministic results."""

    def _schemas_with_all_engines(self):
        with patch("core.tools.machine.shutil.which", return_value="/usr/bin/fake"):
            return get_tool_schemas()

    def test_get_tool_schemas_returns_list(self):
        schemas = self._schemas_with_all_engines()
        assert isinstance(schemas, list)
        assert len(schemas) == 1

    def test_schema_name(self):
        schema = self._schemas_with_all_engines()[0]
        assert schema["name"] == "machine_run"

    def test_schema_required_params(self):
        schema = self._schemas_with_all_engines()[0]
        required = schema["parameters"]["required"]
        assert "engine" in required
        assert "instruction" in required
        assert "working_directory" in required

    def test_schema_engine_no_enum(self):
        schema = self._schemas_with_all_engines()[0]
        engine_prop = schema["parameters"]["properties"]["engine"]
        assert "enum" not in engine_prop
        assert "type" in engine_prop

    def test_schema_has_background_param(self):
        schema = self._schemas_with_all_engines()[0]
        props = schema["parameters"]["properties"]
        assert "background" in props
        assert props["background"]["type"] == "boolean"

    def test_schema_description_shows_recommended_engine(self):
        schema = self._schemas_with_all_engines()[0]
        desc = schema["description"]
        assert "cursor-agent" in desc  # default top-priority engine
        assert "__list__" in desc


# ── Engine Availability Tests ─────────────────────────────


class TestEngineAvailability:
    """Dynamic engine availability probing via shutil.which."""

    def test_all_engines_available(self):
        with patch("core.tools.machine.shutil.which", return_value="/usr/bin/fake"):
            available = _get_available_engines()
            assert set(available) == _VALID_ENGINES

    def test_no_engines_available(self):
        with patch("core.tools.machine.shutil.which", return_value=None):
            available = _get_available_engines()
            assert available == []

    def test_partial_engines_available(self):
        def selective_which(name, path=None):
            return "/usr/bin/fake" if name in ("claude", "cursor-agent") else None

        with patch("core.tools.machine.shutil.which", side_effect=selective_which):
            available = _get_available_engines()
            assert set(available) == {"claude", "cursor-agent"}
            assert available[0] == "cursor-agent"  # priority order

    def test_schemas_empty_when_no_engines(self):
        with patch("core.tools.machine.shutil.which", return_value=None):
            schemas = get_tool_schemas()
            assert schemas == []

    def test_schemas_description_shows_recommended(self):
        def selective_which(name, path=None):
            return "/usr/bin/fake" if name == "cursor-agent" else None

        with patch("core.tools.machine.shutil.which", side_effect=selective_which):
            schemas = get_tool_schemas()
            assert len(schemas) == 1
            desc = schemas[0]["description"]
            assert "cursor-agent" in desc

    def test_schema_description_reflects_available_engines(self):
        def selective_which(name, path=None):
            return "/usr/bin/fake" if name in ("codex", "gemini") else None

        with patch("core.tools.machine.shutil.which", side_effect=selective_which):
            schemas = get_tool_schemas()
            desc = schemas[0]["description"]
            engine_desc = schemas[0]["parameters"]["properties"]["engine"]["description"]
            assert "codex" in engine_desc  # codex is higher priority than gemini
            assert "__list__" in desc

    def test_engine_description_reflects_available(self):
        def selective_which(name, path=None):
            return "/usr/bin/fake" if name == "claude" else None

        with patch("core.tools.machine.shutil.which", side_effect=selective_which):
            schemas = get_tool_schemas()
            engine_desc = schemas[0]["parameters"]["properties"]["engine"]["description"]
            assert "claude" in engine_desc


# ── Execution Profile Tests ───────────────────────────────


class TestExecutionProfile:
    def test_profile_has_run(self):
        assert "run" in EXECUTION_PROFILE

    def test_run_is_background_eligible(self):
        assert EXECUTION_PROFILE["run"]["background_eligible"] is True

    def test_run_expected_seconds(self):
        assert EXECUTION_PROFILE["run"]["expected_seconds"] == 600


# ── Environment Sanitization Tests ────────────────────────


class TestBuildEnv:
    def test_allows_path(self):
        with patch.dict(os.environ, {"PATH": "/usr/bin", "HOME": "/home/test"}, clear=True):
            env = _build_env("claude")
            assert "PATH" in env
            assert "HOME" in env

    def test_blocks_animaworks_vars(self):
        with patch.dict(
            os.environ,
            {
                "PATH": "/usr/bin",
                "ANIMAWORKS_HOME": "/secret",
                "ANIMAWORKS_ANIMA_DIR": "/anima",
                "ANIMAWORKS_SOCKET": "/sock",
            },
            clear=True,
        ):
            env = _build_env("claude")
            assert "ANIMAWORKS_HOME" not in env
            assert "ANIMAWORKS_ANIMA_DIR" not in env
            assert "ANIMAWORKS_SOCKET" not in env

    def test_claude_strips_parent_anthropic_key_without_engine_credentials(self):
        with patch.dict(
            os.environ,
            {
                "PATH": "/usr/bin",
                "ANTHROPIC_API_KEY": "sk-test",
                "OPENAI_API_KEY": "sk-oai",
            },
            clear=True,
        ):
            with patch("core.tools.machine._resolve_engine_credentials", return_value={}):
                env = _build_env("claude")
            assert "ANTHROPIC_API_KEY" not in env
            assert env.get("OPENAI_API_KEY") == "sk-oai"

    def test_blocks_random_vars(self):
        with patch.dict(
            os.environ,
            {"PATH": "/usr/bin", "SECRET_TOKEN": "abc", "MY_VAR": "xyz"},
            clear=True,
        ):
            env = _build_env("claude")
            assert "SECRET_TOKEN" not in env
            assert "MY_VAR" not in env


# ── Instruction Prefix Tests ──────────────────────────────


class TestBuildInstruction:
    def test_adds_workspace_constraint(self):
        result = _build_instruction("Do something", "/tmp/work")
        assert "WORKSPACE CONSTRAINT" in result
        assert "/tmp/work" in result
        assert "Do something" in result

    def test_warns_about_animaworks(self):
        result = _build_instruction("test", "/tmp/work")
        assert "~/.animaworks/" in result


# ── Command Builder Tests ─────────────────────────────────


class TestBuildCommand:
    def test_claude_command(self):
        cmd = _build_command("claude", "/tmp/work")
        assert cmd[0] == "claude"
        assert "-p" in cmd
        assert "--output-format" in cmd

    def test_codex_command(self):
        cmd = _build_command("codex", "/tmp/work")
        assert cmd[0] == "codex"
        assert "exec" in cmd
        assert "-C" in cmd
        idx = cmd.index("-C")
        assert cmd[idx + 1] == "/tmp/work"

    def test_gemini_command(self):
        cmd = _build_command("gemini", "/tmp/work")
        assert cmd[0] == "gemini"
        assert "--approval-mode" in cmd

    def test_cursor_agent_command(self):
        cmd = _build_command("cursor-agent", "/tmp/work")
        assert cmd[0] == "cursor-agent"
        assert "--trust" in cmd
        assert "--force" in cmd
        assert "--workspace" in cmd

    def test_instruction_not_in_command(self):
        """Instruction should not be in the command — it's passed via stdin."""
        cmd = _build_command("claude", "/tmp/work")
        for arg in cmd:
            assert "instruction" not in arg.lower() or arg == "--output-format"

    def test_model_override(self):
        cmd = _build_command("claude", "/tmp/work", model="haiku")
        assert "--model" in cmd
        idx = cmd.index("--model")
        assert cmd[idx + 1] == "haiku"

    def test_codex_model_override(self):
        cmd = _build_command("codex", "/tmp/work", model="o3-mini")
        assert "-m" in cmd
        idx = cmd.index("-m")
        assert cmd[idx + 1] == "o3-mini"

    def test_default_model_from_config(self):
        with patch(
            "core.tools.machine._get_default_model",
            return_value="claude-4.6-opus-high-thinking",
        ):
            cmd = _build_command("cursor-agent", "/tmp/work")
            assert "--model" in cmd
            idx = cmd.index("--model")
            assert cmd[idx + 1] == "claude-4.6-opus-high-thinking"

    def test_explicit_model_overrides_config_default(self):
        with patch(
            "core.tools.machine._get_default_model",
            return_value="claude-4.6-opus-high-thinking",
        ):
            cmd = _build_command("cursor-agent", "/tmp/work", model="sonnet-4")
            assert "--model" in cmd
            idx = cmd.index("--model")
            assert cmd[idx + 1] == "sonnet-4"

    def test_no_model_flag_when_no_config_default(self):
        with patch("core.tools.machine._get_default_model", return_value=None):
            cmd = _build_command("cursor-agent", "/tmp/work")
            assert "--model" not in cmd

    def test_codex_command_includes_reasoning_effort(self):
        cmd = _build_command("codex", "/tmp/work")
        assert 'model_reasoning_effort="high"' in cmd

    def test_codex_command_sandboxed_disables_inner_sandbox(self):
        cmd = _build_command("codex", "/tmp/work", fs_sandboxed=True)
        assert "danger-full-access" in cmd
        assert "workspace-write" not in cmd

    def test_codex_command_not_sandboxed_keeps_workspace_write(self):
        cmd = _build_command("codex", "/tmp/work", fs_sandboxed=False)
        assert "workspace-write" in cmd
        assert "danger-full-access" not in cmd


# ── Working Directory Validation Tests ────────────────────


class TestValidateWorkingDirectory:
    def test_allows_normal_directory(self, tmp_path):
        anima_dir = str(tmp_path / "animas" / "test")
        result = _validate_working_directory("/tmp/project", anima_dir)
        assert result is None

    def test_blocks_memory_directory(self, tmp_path):
        anima_dir = tmp_path / "animas" / "test"
        anima_dir.mkdir(parents=True)
        wd = str(anima_dir / "memory")
        result = _validate_working_directory(wd, str(anima_dir))
        assert result is not None

    def test_blocks_episodes_directory(self, tmp_path):
        anima_dir = tmp_path / "animas" / "test"
        anima_dir.mkdir(parents=True)
        wd = str(anima_dir / "episodes")
        result = _validate_working_directory(wd, str(anima_dir))
        assert result is not None

    def test_blocks_knowledge_directory(self, tmp_path):
        anima_dir = tmp_path / "animas" / "test"
        anima_dir.mkdir(parents=True)
        wd = str(anima_dir / "knowledge")
        result = _validate_working_directory(wd, str(anima_dir))
        assert result is not None

    def test_blocks_state_directory(self, tmp_path):
        anima_dir = tmp_path / "animas" / "test"
        anima_dir.mkdir(parents=True)
        wd = str(anima_dir / "state")
        result = _validate_working_directory(wd, str(anima_dir))
        assert result is not None

    def test_no_anima_dir_allows_all(self):
        result = _validate_working_directory("/tmp/project", None)
        assert result is None


# ── Rate Limiting Tests ───────────────────────────────────


class TestRateLimiting:
    def setup_method(self):
        reset_call_counts()

    def test_dispatch_increments_counter(self, tmp_path):
        wd = tmp_path / "workspace"
        wd.mkdir()
        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()
        with patch("core.tools.machine.shutil.which", return_value="/usr/bin/claude"):
            mock_proc = MagicMock()
            _set_pipe_output(mock_proc, "ok\n")
            mock_proc.stdin = MagicMock()
            mock_proc.returncode = 0
            mock_proc.pid = 99999
            mock_proc.wait = MagicMock(return_value=None)
            with patch("core.tools.machine.subprocess.Popen", return_value=mock_proc):
                dispatch(
                    "machine_run",
                    {
                        "engine": "claude",
                        "instruction": "test",
                        "working_directory": str(wd),
                        "anima_dir": str(anima_dir),
                    },
                )

    def test_rate_limit_exceeded_in_session(self, tmp_path):
        wd = tmp_path / "workspace"
        wd.mkdir()
        anima_dir = tmp_path / "test_anima_rate"
        anima_dir.mkdir()

        with patch("core.tools.machine.shutil.which", return_value="/usr/bin/claude"):
            mock_proc = MagicMock()
            _set_pipe_output(mock_proc, "ok\n")
            mock_proc.stdin = MagicMock()
            mock_proc.returncode = 0
            mock_proc.pid = 99999
            mock_proc.wait = MagicMock(return_value=None)
            with patch("core.tools.machine.subprocess.Popen", return_value=mock_proc):
                for _ in range(_MAX_CALLS_PER_SESSION):
                    dispatch(
                        "machine_run",
                        {
                            "engine": "claude",
                            "instruction": "test",
                            "working_directory": str(wd),
                            "anima_dir": str(anima_dir),
                        },
                    )

                result = json.loads(
                    dispatch(
                        "machine_run",
                        {
                            "engine": "claude",
                            "instruction": "test",
                            "working_directory": str(wd),
                            "anima_dir": str(anima_dir),
                        },
                    )
                )
                assert "error" in result

    def test_rate_limit_heartbeat(self, tmp_path):
        wd = tmp_path / "workspace"
        wd.mkdir()
        anima_dir = tmp_path / "test_anima_hb"
        anima_dir.mkdir()

        with patch("core.tools.machine.shutil.which", return_value="/usr/bin/claude"):
            mock_proc = MagicMock()
            _set_pipe_output(mock_proc, "ok\n")
            mock_proc.stdin = MagicMock()
            mock_proc.returncode = 0
            mock_proc.pid = 99999
            mock_proc.wait = MagicMock(return_value=None)
            with patch("core.tools.machine.subprocess.Popen", return_value=mock_proc):
                for _ in range(_MAX_CALLS_PER_HEARTBEAT):
                    dispatch(
                        "machine_run",
                        {
                            "engine": "claude",
                            "instruction": "test",
                            "working_directory": str(wd),
                            "anima_dir": str(anima_dir),
                            "trigger": "heartbeat",
                        },
                    )

                result = json.loads(
                    dispatch(
                        "machine_run",
                        {
                            "engine": "claude",
                            "instruction": "test",
                            "working_directory": str(wd),
                            "anima_dir": str(anima_dir),
                            "trigger": "heartbeat",
                        },
                    )
                )
                assert "error" in result

    def test_rate_limit_heartbeat_via_handler_injected_trigger(self, tmp_path):
        """ToolHandler injects the trigger as "_trigger"; the heartbeat limit
        must apply to that key as well (native MCP path)."""
        wd = tmp_path / "workspace"
        wd.mkdir()
        anima_dir = tmp_path / "test_anima_hb_native"
        anima_dir.mkdir()

        with patch("core.tools.machine.shutil.which", return_value="/usr/bin/claude"):
            mock_proc = MagicMock()
            _set_pipe_output(mock_proc, "ok\n")
            mock_proc.stdin = MagicMock()
            mock_proc.returncode = 0
            mock_proc.pid = 99999
            mock_proc.wait = MagicMock(return_value=None)
            with patch("core.tools.machine.subprocess.Popen", return_value=mock_proc):
                for _ in range(_MAX_CALLS_PER_HEARTBEAT):
                    dispatch(
                        "machine_run",
                        {
                            "engine": "claude",
                            "instruction": "test",
                            "working_directory": str(wd),
                            "anima_dir": str(anima_dir),
                            "_trigger": "heartbeat",
                        },
                    )

                result = json.loads(
                    dispatch(
                        "machine_run",
                        {
                            "engine": "claude",
                            "instruction": "test",
                            "working_directory": str(wd),
                            "anima_dir": str(anima_dir),
                            "_trigger": "heartbeat",
                        },
                    )
                )
                assert "error" in result


# ── Dispatch Tests ────────────────────────────────────────


class TestDispatch:
    def setup_method(self):
        reset_call_counts()

    def test_invalid_engine(self, tmp_path):
        result = json.loads(
            dispatch(
                "machine_run",
                {
                    "engine": "invalid",
                    "instruction": "test",
                    "working_directory": str(tmp_path),
                },
            )
        )
        assert "error" in result

    def test_empty_instruction(self, tmp_path):
        result = json.loads(
            dispatch(
                "machine_run",
                {
                    "engine": "claude",
                    "instruction": "",
                    "working_directory": str(tmp_path),
                },
            )
        )
        assert "error" in result

    def test_missing_working_directory(self):
        result = json.loads(
            dispatch(
                "machine_run",
                {
                    "engine": "claude",
                    "instruction": "test",
                    "working_directory": "",
                },
            )
        )
        assert "error" in result

    def test_nonexistent_directory(self):
        result = json.loads(
            dispatch(
                "machine_run",
                {
                    "engine": "claude",
                    "instruction": "test",
                    "working_directory": "/nonexistent/path/xyz",
                    "anima_dir": "/tmp/anima",
                },
            )
        )
        assert "error" in result

    def test_engine_not_found(self, tmp_path):
        with patch("core.tools.machine.shutil.which", return_value=None):
            result = json.loads(
                dispatch(
                    "machine_run",
                    {
                        "engine": "claude",
                        "instruction": "test",
                        "working_directory": str(tmp_path),
                        "anima_dir": "/tmp/anima",
                    },
                )
            )
            assert "error" in result

    def test_successful_execution(self, tmp_path):
        wd = tmp_path / "workspace"
        wd.mkdir()
        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()
        with patch("core.tools.machine.shutil.which", return_value="/usr/bin/claude"):
            mock_proc = MagicMock()
            _set_pipe_output(mock_proc, "Implementation complete.\nFiles modified: 3\n")
            mock_proc.stdin = MagicMock()
            mock_proc.returncode = 0
            mock_proc.pid = 99999
            mock_proc.wait = MagicMock(return_value=None)
            with patch("core.tools.machine.subprocess.Popen", return_value=mock_proc):
                result = json.loads(
                    dispatch(
                        "machine_run",
                        {
                            "engine": "claude",
                            "instruction": "Refactor the module",
                            "working_directory": str(wd),
                            "anima_dir": str(anima_dir),
                        },
                    )
                )
                assert result["success"] is True
                assert "Implementation complete" in result["output"]
                assert result["engine"] == "claude"

    def test_execution_with_nonzero_exit(self, tmp_path):
        wd = tmp_path / "workspace"
        wd.mkdir()
        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()
        with patch("core.tools.machine.shutil.which", return_value="/usr/bin/codex"):
            mock_proc = MagicMock()
            _set_pipe_output(mock_proc, "partial output\n", "error occurred\n")
            mock_proc.stdin = MagicMock()
            mock_proc.returncode = 1
            mock_proc.pid = 99999
            mock_proc.wait = MagicMock(return_value=None)
            with patch("core.tools.machine.subprocess.Popen", return_value=mock_proc):
                result = json.loads(
                    dispatch(
                        "machine_run",
                        {
                            "engine": "codex",
                            "instruction": "bad instruction",
                            "working_directory": str(wd),
                            "anima_dir": str(anima_dir),
                        },
                    )
                )
                assert result["success"] is False
                assert result["exit_code"] == 1

    def test_timeout_handling(self, tmp_path):
        wd = tmp_path / "workspace"
        wd.mkdir()
        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()
        with patch("core.tools.machine.shutil.which", return_value="/usr/bin/claude"):
            mock_proc = MagicMock()
            _set_pipe_output(mock_proc, "partial output\n")
            mock_proc.stdin = MagicMock()
            mock_proc.pid = 12345
            mock_proc.returncode = -1
            mock_proc.wait = MagicMock(side_effect=subprocess.TimeoutExpired(cmd=["claude"], timeout=10))
            with (
                patch("core.tools.machine.subprocess.Popen", return_value=mock_proc),
                patch("core.tools.machine.terminate_subprocess") as mock_terminate,
            ):
                result = json.loads(
                    dispatch(
                        "machine_run",
                        {
                            "engine": "claude",
                            "instruction": "long task",
                            "working_directory": str(wd),
                            "anima_dir": str(anima_dir),
                            "timeout": 10,
                        },
                    )
                )
                assert result["success"] is False
                assert result.get("timed_out") is True
                assert mock_terminate.call_args_list[0].args == (mock_proc,)
                assert mock_terminate.call_args_list[0].kwargs == {"force": False}
                assert mock_terminate.call_args_list[1].args == (mock_proc,)
                assert mock_terminate.call_args_list[1].kwargs == {"force": True}

    def test_unknown_action(self):
        result = json.loads(dispatch("unknown_action", {}))
        assert "error" in result

    def test_subprocess_called_with_sanitized_env(self, tmp_path):
        wd = tmp_path / "workspace"
        wd.mkdir()
        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()
        with patch("core.tools.machine.shutil.which", return_value="/usr/bin/claude"):
            mock_proc = MagicMock()
            _set_pipe_output(mock_proc, "ok\n")
            mock_proc.stdin = MagicMock()
            mock_proc.returncode = 0
            mock_proc.pid = 99999
            mock_proc.wait = MagicMock(return_value=None)
            with (
                patch("core.tools.machine.subprocess.Popen", return_value=mock_proc) as mock_popen,
                patch.dict(
                    os.environ,
                    {"ANIMAWORKS_HOME": "/secret", "PATH": "/usr/bin"},
                ),
            ):
                dispatch(
                    "machine_run",
                    {
                        "engine": "claude",
                        "instruction": "test",
                        "working_directory": str(wd),
                        "anima_dir": str(anima_dir),
                    },
                )
                call_kwargs = mock_popen.call_args
                assert call_kwargs is not None
                env = call_kwargs.kwargs.get("env") or call_kwargs[1].get("env")
                assert "ANIMAWORKS_HOME" not in env

    def test_subprocess_runs_in_working_directory(self, tmp_path):
        wd = tmp_path / "workspace"
        wd.mkdir()
        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()
        with patch("core.tools.machine.shutil.which", return_value="/usr/bin/claude"):
            mock_proc = MagicMock()
            _set_pipe_output(mock_proc, "ok\n")
            mock_proc.stdin = MagicMock()
            mock_proc.returncode = 0
            mock_proc.pid = 99999
            mock_proc.wait = MagicMock(return_value=None)
            with patch("core.tools.machine.subprocess.Popen", return_value=mock_proc) as mock_popen:
                dispatch(
                    "machine_run",
                    {
                        "engine": "claude",
                        "instruction": "test",
                        "working_directory": str(wd),
                        "anima_dir": str(anima_dir),
                    },
                )
                call_kwargs = mock_popen.call_args
                assert call_kwargs is not None
                cwd = call_kwargs.kwargs.get("cwd") or call_kwargs[1].get("cwd")
                assert cwd == str(wd)


# ── Sandboxed Execution Tests ─────────────────────────────


class TestSandboxedExecution:
    """Mode C shells run inside a Codex sandbox (HOME read-only).

    Only the codex engine can run there (ephemeral CODEX_HOME + inner
    sandbox disabled, confined by the inherited outer sandbox); other
    engines are rejected with a hint to use engine=codex.
    """

    def setup_method(self):
        reset_call_counts()

    def _mock_proc(self, output: str = "done\n") -> MagicMock:
        mock_proc = MagicMock()
        _set_pipe_output(mock_proc, output)
        mock_proc.stdin = MagicMock()
        mock_proc.returncode = 0
        mock_proc.pid = 99999
        mock_proc.wait = MagicMock(return_value=None)
        return mock_proc

    def test_sandboxed_non_codex_rejected(self, tmp_path):
        wd = tmp_path / "workspace"
        wd.mkdir()
        with (
            patch("core.tools.machine._is_fs_sandboxed", return_value=True),
            patch("core.tools.machine.shutil.which", return_value="/usr/bin/claude"),
            patch("core.tools.machine.subprocess.Popen") as mock_popen,
        ):
            result = json.loads(
                dispatch(
                    "machine_run",
                    {
                        "engine": "claude",
                        "instruction": "test",
                        "working_directory": str(wd),
                        "anima_dir": str(tmp_path / "anima"),
                    },
                )
            )
            assert result["success"] is False
            assert "codex" in result["error"]
            mock_popen.assert_not_called()

    def test_sandboxed_codex_executes_with_ephemeral_home(self, tmp_path):
        wd = tmp_path / "workspace"
        wd.mkdir()
        with (
            patch("core.tools.machine._is_fs_sandboxed", return_value=True),
            patch("core.tools.machine.shutil.which", return_value="/usr/bin/codex"),
            patch(
                "core.tools.machine.subprocess.Popen",
                return_value=self._mock_proc(),
            ) as mock_popen,
        ):
            result = json.loads(
                dispatch(
                    "machine_run",
                    {
                        "engine": "codex",
                        "instruction": "test",
                        "working_directory": str(wd),
                        "anima_dir": str(tmp_path / "anima"),
                    },
                )
            )
        assert result["success"] is True
        cmd = mock_popen.call_args[0][0]
        assert "danger-full-access" in cmd
        assert "workspace-write" not in cmd
        env = mock_popen.call_args.kwargs["env"]
        assert "CODEX_HOME" in env
        assert not os.path.exists(env["CODEX_HOME"])  # cleaned up after run

    def test_not_sandboxed_codex_uses_default_home(self, tmp_path):
        wd = tmp_path / "workspace"
        wd.mkdir()
        with (
            patch("core.tools.machine._is_fs_sandboxed", return_value=False),
            patch("core.tools.machine.shutil.which", return_value="/usr/bin/codex"),
            patch(
                "core.tools.machine.subprocess.Popen",
                return_value=self._mock_proc(),
            ) as mock_popen,
        ):
            result = json.loads(
                dispatch(
                    "machine_run",
                    {
                        "engine": "codex",
                        "instruction": "test",
                        "working_directory": str(wd),
                        "anima_dir": str(tmp_path / "anima"),
                    },
                )
            )
        assert result["success"] is True
        cmd = mock_popen.call_args[0][0]
        assert "workspace-write" in cmd
        env = mock_popen.call_args.kwargs["env"]
        assert "CODEX_HOME" not in env


# ── Auto-Discovery Test ───────────────────────────────────


# ── Output Truncation Tests ───────────────────────────────


class TestOutputTruncation:
    def setup_method(self):
        reset_call_counts()

    def test_large_output_truncated(self, tmp_path):
        wd = tmp_path / "workspace"
        wd.mkdir()
        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()
        large_output = "x" * 60_000
        with patch("core.tools.machine.shutil.which", return_value="/usr/bin/claude"):
            mock_proc = MagicMock()
            _set_pipe_output(mock_proc, large_output)
            mock_proc.stdin = MagicMock()
            mock_proc.returncode = 0
            mock_proc.pid = 99999
            mock_proc.wait = MagicMock(return_value=None)
            with patch("core.tools.machine.subprocess.Popen", return_value=mock_proc):
                result = json.loads(
                    dispatch(
                        "machine_run",
                        {
                            "engine": "claude",
                            "instruction": "generate large output",
                            "working_directory": str(wd),
                            "anima_dir": str(anima_dir),
                        },
                    )
                )
                assert result["success"] is True
                assert "truncated" in result["output"]
                assert len(result["output"]) < 60_000


# ── CLI Tests ─────────────────────────────────────────────


class TestCliMain:
    def test_no_subcommand_prints_help(self, capsys):
        from core.tools.machine import cli_main

        cli_main([])
        captured = capsys.readouterr()
        assert "machine tool" in captured.out.lower() or "usage" in captured.out.lower()

    def test_run_success_text_output(self, tmp_path, capsys):
        from core.tools.machine import cli_main

        with patch("core.tools.machine.shutil.which", return_value="/usr/bin/claude"):
            mock_proc = MagicMock()
            _set_pipe_output(mock_proc, "Hello from CLI\n")
            mock_proc.stdin = MagicMock()
            mock_proc.returncode = 0
            mock_proc.pid = 99999
            mock_proc.wait = MagicMock(return_value=None)
            with patch("core.tools.machine.subprocess.Popen", return_value=mock_proc):
                cli_main(["run", "test instruction", "-d", str(tmp_path)])
                captured = capsys.readouterr()
                assert "Hello from CLI" in captured.out

    def test_run_success_json_output(self, tmp_path, capsys):
        from core.tools.machine import cli_main

        with patch("core.tools.machine.shutil.which", return_value="/usr/bin/claude"):
            mock_proc = MagicMock()
            _set_pipe_output(mock_proc, "JSON result\n")
            mock_proc.stdin = MagicMock()
            mock_proc.returncode = 0
            mock_proc.pid = 99999
            mock_proc.wait = MagicMock(return_value=None)
            with patch("core.tools.machine.subprocess.Popen", return_value=mock_proc):
                cli_main(["run", "test", "-d", str(tmp_path), "-j"])
                captured = capsys.readouterr()
                data = json.loads(captured.out)
                assert data["success"] is True
                assert "JSON result" in data["output"]

    def test_run_failure_exits(self, tmp_path):
        from core.tools.machine import cli_main

        with patch("core.tools.machine.shutil.which", return_value=None), pytest.raises(SystemExit):
            cli_main(["run", "test", "-d", str(tmp_path)])

    def test_run_background_flag_sets_async_timeout(self, tmp_path, capsys):
        from core.tools.machine import _DEFAULT_TIMEOUT_ASYNC, cli_main

        with patch("core.tools.machine.shutil.which", return_value="/usr/bin/claude"):
            mock_proc = MagicMock()
            _set_pipe_output(mock_proc, "bg output\n")
            mock_proc.stdin = MagicMock()
            mock_proc.returncode = 0
            mock_proc.pid = 99999
            mock_proc.wait = MagicMock(return_value=None)
            with (
                patch("core.tools.machine.subprocess.Popen", return_value=mock_proc),
                patch("core.tools.machine._execute", wraps=None) as mock_exec,
            ):
                mock_exec.return_value = ToolResult(success=True, text="bg done")
                cli_main(["run", "--background", "test bg", "-d", str(tmp_path)])
                mock_exec.assert_called_once()
                call_kwargs = mock_exec.call_args
                assert (
                    call_kwargs.kwargs.get("timeout") == _DEFAULT_TIMEOUT_ASYNC
                    or call_kwargs[1].get("timeout") == _DEFAULT_TIMEOUT_ASYNC
                )

    def test_run_background_appears_in_help(self, capsys):
        from core.tools.machine import cli_main

        with pytest.raises(SystemExit):
            cli_main(["run", "--help"])
        captured = capsys.readouterr()
        assert "--background" in captured.out


# ── Auto-Discovery Test ───────────────────────────────────


class TestAutoDiscovery:
    def test_machine_in_tool_modules(self):
        from core.tools import TOOL_MODULES

        assert "machine" in TOOL_MODULES


# ── Engine Priority Tests ─────────────────────────────────


class TestEnginePriority:
    """Tests for engine priority ordering and __list__ dispatch."""

    def test_default_priority_order(self):
        from core.tools.machine import _DEFAULT_ENGINE_PRIORITY

        assert _DEFAULT_ENGINE_PRIORITY[0] == "cursor-agent"
        assert set(_DEFAULT_ENGINE_PRIORITY) == _VALID_ENGINES

    def test_available_engines_respect_priority(self):
        with patch("core.tools.machine.shutil.which", return_value="/usr/bin/fake"):
            available = _get_available_engines()
            assert available[0] == "cursor-agent"

    def test_config_priority_override(self):
        with (
            patch("core.tools.machine.shutil.which", return_value="/usr/bin/fake"),
            patch("core.tools.machine._get_engine_priority", return_value=["gemini", "claude"]),
        ):
            available = _get_available_engines()
            assert available[0] == "gemini"
            assert available[1] == "claude"

    def test_list_sentinel_dispatch(self):
        with patch("core.tools.machine.shutil.which", return_value="/usr/bin/fake"):
            result_str = dispatch("machine_run", {"engine": "__list__"})
            result = json.loads(result_str)
            assert "engines" in result
            assert result["total"] == len(_VALID_ENGINES)
            assert result["recommended"] == "cursor-agent"
            assert result["engines"][0]["rank"] == 1
            assert result["engines"][0]["recommended"] is True

    def test_list_sentinel_does_not_count_rate_limit(self):
        reset_call_counts()
        with patch("core.tools.machine.shutil.which", return_value="/usr/bin/fake"):
            dispatch("machine_run", {"engine": "__list__"})
            from core.tools.machine import _session_call_counts

            assert sum(_session_call_counts.values()) == 0

    def test_single_engine_no_list_mention(self):
        def selective_which(name, path=None):
            return "/usr/bin/fake" if name == "claude" else None

        with patch("core.tools.machine.shutil.which", side_effect=selective_which):
            schemas = get_tool_schemas()
            desc = schemas[0]["description"]
            assert "__list__" not in desc
