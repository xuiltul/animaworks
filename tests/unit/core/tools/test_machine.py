"""Tests for the machine tool module (core/tools/machine.py)."""

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.tools.machine import (
    EXECUTION_PROFILE,
    _MAX_CALLS_PER_HEARTBEAT,
    _MAX_CALLS_PER_SESSION,
    _VALID_ENGINES,
    _build_command,
    _build_env,
    _build_instruction,
    _validate_working_directory,
    dispatch,
    get_tool_schemas,
    reset_call_counts,
)


# ── Schema Tests ──────────────────────────────────────────


class TestToolSchemas:
    def test_get_tool_schemas_returns_list(self):
        schemas = get_tool_schemas()
        assert isinstance(schemas, list)
        assert len(schemas) == 1

    def test_schema_name(self):
        schema = get_tool_schemas()[0]
        assert schema["name"] == "machine_run"

    def test_schema_required_params(self):
        schema = get_tool_schemas()[0]
        required = schema["parameters"]["required"]
        assert "engine" in required
        assert "instruction" in required
        assert "working_directory" in required

    def test_schema_engine_enum(self):
        schema = get_tool_schemas()[0]
        engine_enum = schema["parameters"]["properties"]["engine"]["enum"]
        assert set(engine_enum) == _VALID_ENGINES

    def test_schema_has_background_param(self):
        schema = get_tool_schemas()[0]
        props = schema["parameters"]["properties"]
        assert "background" in props
        assert props["background"]["type"] == "boolean"


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

    def test_allows_api_keys(self):
        with patch.dict(
            os.environ,
            {
                "PATH": "/usr/bin",
                "ANTHROPIC_API_KEY": "sk-test",
                "OPENAI_API_KEY": "sk-oai",
            },
            clear=True,
        ):
            env = _build_env("claude")
            assert env.get("ANTHROPIC_API_KEY") == "sk-test"
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
        with patch("core.tools.machine.shutil.which", return_value="/usr/bin/claude"):
            mock_proc = MagicMock()
            mock_proc.communicate.return_value = ("ok", "")
            mock_proc.returncode = 0
            mock_proc.pid = 99999
            with patch("core.tools.machine.subprocess.Popen", return_value=mock_proc):
                dispatch(
                    "machine_run",
                    {
                        "engine": "claude",
                        "instruction": "test",
                        "working_directory": str(wd),
                        "anima_dir": "/tmp/anima",
                    },
                )

    def test_rate_limit_exceeded_in_session(self, tmp_path):
        wd = tmp_path / "workspace"
        wd.mkdir()
        anima_dir = "/tmp/test_anima_rate"

        with patch("core.tools.machine.shutil.which", return_value="/usr/bin/claude"):
            mock_proc = MagicMock()
            mock_proc.communicate.return_value = ("ok", "")
            mock_proc.returncode = 0
            mock_proc.pid = 99999
            with patch("core.tools.machine.subprocess.Popen", return_value=mock_proc):
                for _ in range(_MAX_CALLS_PER_SESSION):
                    dispatch(
                        "machine_run",
                        {
                            "engine": "claude",
                            "instruction": "test",
                            "working_directory": str(wd),
                            "anima_dir": anima_dir,
                        },
                    )

                result = json.loads(
                    dispatch(
                        "machine_run",
                        {
                            "engine": "claude",
                            "instruction": "test",
                            "working_directory": str(wd),
                            "anima_dir": anima_dir,
                        },
                    )
                )
                assert "error" in result

    def test_rate_limit_heartbeat(self, tmp_path):
        wd = tmp_path / "workspace"
        wd.mkdir()
        anima_dir = "/tmp/test_anima_hb"

        with patch("core.tools.machine.shutil.which", return_value="/usr/bin/claude"):
            mock_proc = MagicMock()
            mock_proc.communicate.return_value = ("ok", "")
            mock_proc.returncode = 0
            mock_proc.pid = 99999
            with patch("core.tools.machine.subprocess.Popen", return_value=mock_proc):
                for _ in range(_MAX_CALLS_PER_HEARTBEAT):
                    dispatch(
                        "machine_run",
                        {
                            "engine": "claude",
                            "instruction": "test",
                            "working_directory": str(wd),
                            "anima_dir": anima_dir,
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
                            "anima_dir": anima_dir,
                            "trigger": "heartbeat",
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
        assert result["success"] is False

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
            assert result["success"] is False
            assert "error" in result

    def test_successful_execution(self, tmp_path):
        with patch("core.tools.machine.shutil.which", return_value="/usr/bin/claude"):
            mock_proc = MagicMock()
            mock_proc.communicate.return_value = (
                "Implementation complete.\nFiles modified: 3",
                "",
            )
            mock_proc.returncode = 0
            mock_proc.pid = 99999
            with patch("core.tools.machine.subprocess.Popen", return_value=mock_proc):
                result = json.loads(
                    dispatch(
                        "machine_run",
                        {
                            "engine": "claude",
                            "instruction": "Refactor the module",
                            "working_directory": str(tmp_path),
                            "anima_dir": "/tmp/anima",
                        },
                    )
                )
                assert result["success"] is True
                assert "Implementation complete" in result["output"]
                assert result["engine"] == "claude"

    def test_execution_with_nonzero_exit(self, tmp_path):
        with patch("core.tools.machine.shutil.which", return_value="/usr/bin/codex"):
            mock_proc = MagicMock()
            mock_proc.communicate.return_value = ("partial output", "error occurred")
            mock_proc.returncode = 1
            mock_proc.pid = 99999
            with patch("core.tools.machine.subprocess.Popen", return_value=mock_proc):
                result = json.loads(
                    dispatch(
                        "machine_run",
                        {
                            "engine": "codex",
                            "instruction": "bad instruction",
                            "working_directory": str(tmp_path),
                            "anima_dir": "/tmp/anima",
                        },
                    )
                )
                assert result["success"] is False
                assert result["exit_code"] == 1

    def test_timeout_handling(self, tmp_path):
        with patch("core.tools.machine.shutil.which", return_value="/usr/bin/claude"):
            mock_proc = MagicMock()
            mock_proc.communicate.side_effect = [
                subprocess.TimeoutExpired(cmd=["claude"], timeout=10),
                ("partial", ""),
            ]
            mock_proc.pid = 12345
            mock_proc.kill = MagicMock()
            with patch("core.tools.machine.subprocess.Popen", return_value=mock_proc):
                with patch("core.tools.machine.os.killpg"):
                    with patch("core.tools.machine.os.getpgid", return_value=12345):
                        result = json.loads(
                            dispatch(
                                "machine_run",
                                {
                                    "engine": "claude",
                                    "instruction": "long task",
                                    "working_directory": str(tmp_path),
                                    "anima_dir": "/tmp/anima",
                                    "timeout": 10,
                                },
                            )
                        )
                        assert result["success"] is False
                        assert result.get("timed_out") is True

    def test_unknown_action(self):
        result = json.loads(dispatch("unknown_action", {}))
        assert "error" in result

    def test_subprocess_called_with_sanitized_env(self, tmp_path):
        with patch("core.tools.machine.shutil.which", return_value="/usr/bin/claude"):
            mock_proc = MagicMock()
            mock_proc.communicate.return_value = ("ok", "")
            mock_proc.returncode = 0
            mock_proc.pid = 99999
            with patch("core.tools.machine.subprocess.Popen", return_value=mock_proc) as mock_popen:
                with patch.dict(
                    os.environ,
                    {"ANIMAWORKS_HOME": "/secret", "PATH": "/usr/bin"},
                ):
                    dispatch(
                        "machine_run",
                        {
                            "engine": "claude",
                            "instruction": "test",
                            "working_directory": str(tmp_path),
                            "anima_dir": "/tmp/anima",
                        },
                    )
                    call_kwargs = mock_popen.call_args
                    env = call_kwargs.kwargs.get("env") or call_kwargs[1].get("env")
                    assert "ANIMAWORKS_HOME" not in env

    def test_subprocess_runs_in_working_directory(self, tmp_path):
        with patch("core.tools.machine.shutil.which", return_value="/usr/bin/claude"):
            mock_proc = MagicMock()
            mock_proc.communicate.return_value = ("ok", "")
            mock_proc.returncode = 0
            mock_proc.pid = 99999
            with patch("core.tools.machine.subprocess.Popen", return_value=mock_proc) as mock_popen:
                dispatch(
                    "machine_run",
                    {
                        "engine": "claude",
                        "instruction": "test",
                        "working_directory": str(tmp_path),
                        "anima_dir": "/tmp/anima",
                    },
                )
                call_kwargs = mock_popen.call_args
                cwd = call_kwargs.kwargs.get("cwd") or call_kwargs[1].get("cwd")
                assert cwd == str(tmp_path)


# ── Auto-Discovery Test ───────────────────────────────────


# ── Output Truncation Tests ───────────────────────────────


class TestOutputTruncation:
    def setup_method(self):
        reset_call_counts()

    def test_large_output_truncated(self, tmp_path):
        wd = tmp_path / "workspace"
        wd.mkdir()
        large_output = "x" * 60_000
        with patch("core.tools.machine.shutil.which", return_value="/usr/bin/claude"):
            mock_proc = MagicMock()
            mock_proc.communicate.return_value = (large_output, "")
            mock_proc.returncode = 0
            mock_proc.pid = 99999
            with patch("core.tools.machine.subprocess.Popen", return_value=mock_proc):
                result = json.loads(
                    dispatch(
                        "machine_run",
                        {
                            "engine": "claude",
                            "instruction": "generate large output",
                            "working_directory": str(wd),
                            "anima_dir": "/tmp/anima",
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
            mock_proc.communicate.return_value = ("Hello from CLI", "")
            mock_proc.returncode = 0
            mock_proc.pid = 99999
            with patch("core.tools.machine.subprocess.Popen", return_value=mock_proc):
                cli_main(["run", "test instruction", "-d", str(tmp_path)])
                captured = capsys.readouterr()
                assert "Hello from CLI" in captured.out

    def test_run_success_json_output(self, tmp_path, capsys):
        from core.tools.machine import cli_main

        with patch("core.tools.machine.shutil.which", return_value="/usr/bin/claude"):
            mock_proc = MagicMock()
            mock_proc.communicate.return_value = ("JSON result", "")
            mock_proc.returncode = 0
            mock_proc.pid = 99999
            with patch("core.tools.machine.subprocess.Popen", return_value=mock_proc):
                cli_main(["run", "test", "-d", str(tmp_path), "-j"])
                captured = capsys.readouterr()
                data = json.loads(captured.out)
                assert data["success"] is True
                assert data["output"] == "JSON result"

    def test_run_failure_exits(self, tmp_path):
        from core.tools.machine import cli_main

        with patch("core.tools.machine.shutil.which", return_value=None):
            with pytest.raises(SystemExit):
                cli_main(["run", "test", "-d", str(tmp_path)])


# ── Auto-Discovery Test ───────────────────────────────────


class TestAutoDiscovery:
    def test_machine_in_tool_modules(self):
        from core.tools import TOOL_MODULES

        assert "machine" in TOOL_MODULES
