from __future__ import annotations

"""Tests for CommandRunner and Bash background execution."""

import json
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from core.tooling.handler_files import CommandRunner


@pytest.fixture(autouse=True)
def _reset_counter():
    """Reset CommandRunner counter between tests."""
    original = CommandRunner._counter
    CommandRunner._counter = 0
    CommandRunner._active.clear()
    yield
    CommandRunner._counter = original
    CommandRunner._active.clear()


@pytest.fixture()
def output_dir(tmp_path: Path) -> Path:
    d = tmp_path / "state" / "cmd_output"
    d.mkdir(parents=True)
    return d


class TestCommandRunnerIdGeneration:
    def test_sequential_ids(self):
        assert CommandRunner._next_id() == "cmd_1"
        assert CommandRunner._next_id() == "cmd_2"
        assert CommandRunner._next_id() == "cmd_3"

    def test_custom_prefix(self):
        assert CommandRunner._next_id("machine") == "machine_1"
        assert CommandRunner._next_id("machine") == "machine_2"


class TestCommandRunnerStart:
    def test_background_creates_output_file(self, tmp_path: Path):
        runner = CommandRunner("echo hello", tmp_path, timeout=10)
        output_dir = tmp_path / "state" / "cmd_output"
        cmd_id = runner.start(output_dir)

        assert cmd_id.startswith("cmd_")
        output_file = output_dir / f"{cmd_id}.txt"
        assert output_file.exists()

        # Wait for completion
        time.sleep(1)
        content = output_file.read_text()
        assert f"--- {cmd_id} ---" in content
        assert "pid:" in content
        assert "command: echo hello" in content
        assert "status: running" in content

    def test_background_writes_footer_on_completion(self, tmp_path: Path):
        runner = CommandRunner("echo done", tmp_path, timeout=10)
        output_dir = tmp_path / "state" / "cmd_output"
        cmd_id = runner.start(output_dir)

        # Wait for process to complete
        time.sleep(2)
        content = (output_dir / f"{cmd_id}.txt").read_text()
        assert "--- FINISHED ---" in content
        assert "exit_code: 0" in content
        assert "elapsed_seconds:" in content

    def test_background_captures_stdout(self, tmp_path: Path):
        runner = CommandRunner("echo hello_world", tmp_path, timeout=10)
        output_dir = tmp_path / "state" / "cmd_output"
        cmd_id = runner.start(output_dir)

        time.sleep(2)
        content = (output_dir / f"{cmd_id}.txt").read_text()
        assert "hello_world" in content

    def test_background_captures_stderr(self, tmp_path: Path):
        runner = CommandRunner("echo errormsg >&2", tmp_path, timeout=10)
        output_dir = tmp_path / "state" / "cmd_output"
        cmd_id = runner.start(output_dir)

        time.sleep(2)
        content = (output_dir / f"{cmd_id}.txt").read_text()
        assert "[stderr] errormsg" in content

    def test_background_nonzero_exit_code(self, tmp_path: Path):
        runner = CommandRunner("bash -c 'exit 42'", tmp_path, timeout=10)
        output_dir = tmp_path / "state" / "cmd_output"
        cmd_id = runner.start(output_dir)

        time.sleep(2)
        content = (output_dir / f"{cmd_id}.txt").read_text()
        assert "exit_code: 42" in content

    def test_background_removed_from_active_on_completion(self, tmp_path: Path):
        runner = CommandRunner("echo fast", tmp_path, timeout=10)
        output_dir = tmp_path / "state" / "cmd_output"
        cmd_id = runner.start(output_dir)
        assert cmd_id in CommandRunner._active

        time.sleep(2)
        assert cmd_id not in CommandRunner._active

    def test_creates_output_dir_if_missing(self, tmp_path: Path):
        runner = CommandRunner("echo test", tmp_path, timeout=10)
        output_dir = tmp_path / "new_dir" / "cmd_output"
        assert not output_dir.exists()

        cmd_id = runner.start(output_dir)
        assert output_dir.exists()
        assert (output_dir / f"{cmd_id}.txt").exists()
        time.sleep(1)


class TestCommandRunnerTimeout:
    def test_timeout_kills_process_and_records(self, tmp_path: Path):
        runner = CommandRunner("sleep 60", tmp_path, timeout=2)
        output_dir = tmp_path / "state" / "cmd_output"
        cmd_id = runner.start(output_dir)

        time.sleep(10)
        content = (output_dir / f"{cmd_id}.txt").read_text()
        assert "--- FINISHED ---" in content
        assert "timed_out: true" in content


class TestHandleExecuteCommandBackground:
    """Test _handle_execute_command with background=True via ToolHandler."""

    def _make_handler(self, tmp_path: Path):
        from core.memory.manager import MemoryManager
        from core.tooling.handler import ToolHandler

        anima_dir = tmp_path / "animas" / "test"
        anima_dir.mkdir(parents=True, exist_ok=True)
        permissions_content = "# Permissions\n\n## コマンド実行\n(unrestricted)\n"
        (anima_dir / "permissions.md").write_text(permissions_content, encoding="utf-8")

        memory = MagicMock(spec=MemoryManager)
        memory.base_dir = anima_dir
        memory.read_permissions.return_value = permissions_content

        handler = ToolHandler(
            anima_dir=anima_dir,
            memory=memory,
            context_window=32_000,
        )
        return handler, anima_dir

    def test_background_true_returns_immediately(self, tmp_path: Path):
        handler, anima_dir = self._make_handler(tmp_path)

        result = handler.handle("Bash", {"command": "sleep 5 && echo bg_done", "background": True})
        parsed = json.loads(result)

        assert parsed["status"] == "background"
        assert parsed["cmd_id"].startswith("cmd_")
        assert "output_file" in parsed
        assert Path(parsed["output_file"]).exists()

    def test_background_false_is_synchronous(self, tmp_path: Path):
        handler, anima_dir = self._make_handler(tmp_path)

        result = handler.handle("Bash", {"command": "echo sync_test", "timeout": 10})

        assert "sync_test" in result
        assert "background" not in result

    def test_background_respects_permissions(self, tmp_path: Path):
        handler, anima_dir = self._make_handler(tmp_path)
        restricted_perms = "# Permissions\n\n## コマンド実行\n- echo\n- ls\n"
        handler._memory.read_permissions.return_value = restricted_perms

        result = handler.handle("Bash", {"command": "rm -rf /tmp/test", "background": True})
        assert "PermissionDenied" in result or "Blocked" in result

    def test_background_custom_timeout(self, tmp_path: Path):
        handler, anima_dir = self._make_handler(tmp_path)

        result = handler.handle("Bash", {"command": "echo quick", "background": True, "timeout": 60})
        parsed = json.loads(result)
        assert parsed["status"] == "background"

    def test_execute_command_alias_works(self, tmp_path: Path):
        handler, anima_dir = self._make_handler(tmp_path)

        result = handler.handle("execute_command", {"command": "echo alias_test", "background": True})
        parsed = json.loads(result)
        assert parsed["status"] == "background"

    def test_background_output_file_has_correct_content(self, tmp_path: Path):
        handler, anima_dir = self._make_handler(tmp_path)

        result = handler.handle("Bash", {"command": "echo file_content_test", "background": True})
        parsed = json.loads(result)

        time.sleep(2)
        content = Path(parsed["output_file"]).read_text()
        assert "file_content_test" in content
        assert "--- FINISHED ---" in content
        assert "exit_code: 0" in content
