"""Unit tests for cli/commands/server.py — Server startup/stop commands."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.platform.process import subprocess_session_kwargs


# ── PID helpers ──────────────────────────────────────────


class TestPidHelpers:
    @patch("cli.commands.server._get_pid_file")
    def test_write_pid_file(self, mock_get_pid, tmp_path):
        from cli.commands.server import _write_pid_file

        pid_file = tmp_path / "server.pid"
        mock_get_pid.return_value = pid_file

        _write_pid_file()

        assert pid_file.exists()
        assert pid_file.read_text().strip() == str(os.getpid())

    @patch("cli.commands.server._get_pid_file")
    def test_remove_pid_file(self, mock_get_pid, tmp_path):
        from cli.commands.server import _remove_pid_file

        pid_file = tmp_path / "server.pid"
        pid_file.write_text("12345", encoding="utf-8")
        mock_get_pid.return_value = pid_file

        _remove_pid_file()

        assert not pid_file.exists()

    @patch("cli.commands.server._get_pid_file")
    def test_remove_pid_file_missing(self, mock_get_pid, tmp_path):
        from cli.commands.server import _remove_pid_file

        pid_file = tmp_path / "server.pid"
        mock_get_pid.return_value = pid_file

        # Should not raise
        _remove_pid_file()

    @patch("cli.commands.server._get_pid_file")
    def test_read_pid_valid(self, mock_get_pid, tmp_path):
        from cli.commands.server import _read_pid

        pid_file = tmp_path / "server.pid"
        pid_file.write_text("12345", encoding="utf-8")
        mock_get_pid.return_value = pid_file

        assert _read_pid() == 12345

    @patch("cli.commands.server._get_pid_file")
    def test_read_pid_missing(self, mock_get_pid, tmp_path):
        from cli.commands.server import _read_pid

        pid_file = tmp_path / "nonexistent.pid"
        mock_get_pid.return_value = pid_file

        assert _read_pid() is None

    @patch("cli.commands.server._get_pid_file")
    def test_read_pid_invalid(self, mock_get_pid, tmp_path):
        from cli.commands.server import _read_pid

        pid_file = tmp_path / "server.pid"
        pid_file.write_text("not_a_number", encoding="utf-8")
        mock_get_pid.return_value = pid_file

        assert _read_pid() is None

    def test_is_process_alive_current(self):
        from cli.commands.server import _is_process_alive

        # Current process should be alive
        assert _is_process_alive(os.getpid()) is True

    def test_is_process_alive_nonexistent(self):
        from cli.commands.server import _is_process_alive

        # Very high PID that likely doesn't exist
        assert _is_process_alive(999999999) is False


class TestFindServerPidByProcess:
    @patch("cli.commands.server.find_first_matching_pid", return_value=None)
    def test_returns_none_when_no_match(self, mock_find):
        """Returns None when no matching process is found."""
        from cli.commands.server import _find_server_pid_by_process

        result = _find_server_pid_by_process()

        assert result is None
        mock_find.assert_called_once()

    @patch("cli.commands.server.find_first_matching_pid", return_value=12345)
    def test_finds_matching_process(self, mock_find):
        """Delegates process lookup to the adapter."""
        from cli.commands.server import _find_server_pid_by_process

        assert _find_server_pid_by_process() == 12345
        mock_find.assert_called_once()


# ── _stop_server ─────────────────────────────────────────


class TestStopServer:
    @patch("cli.commands.server._find_server_pid_by_process", return_value=None)
    @patch("cli.commands.server._read_pid", return_value=None)
    def test_no_pid_file_no_process(self, mock_pid, mock_find, capsys):
        from cli.commands.server import _stop_server

        result = _stop_server()
        assert result is True
        assert "not running" in capsys.readouterr().out

    @patch("cli.commands.server._remove_pid_file")
    @patch("cli.commands.server._is_process_alive", return_value=False)
    @patch("cli.commands.server._read_pid", return_value=12345)
    def test_stale_pid(self, mock_pid, mock_alive, mock_remove, capsys):
        from cli.commands.server import _stop_server

        result = _stop_server()
        assert result is True
        assert "Stale" in capsys.readouterr().out

    @patch("cli.commands.server._remove_pid_file")
    @patch("cli.commands.server.terminate_pid")
    @patch("cli.commands.server._is_process_alive", side_effect=[True, False])
    @patch("cli.commands.server._read_pid", return_value=12345)
    def test_successful_stop(self, mock_pid, mock_alive, mock_terminate, mock_remove, capsys):
        from cli.commands.server import _stop_server

        result = _stop_server()
        assert result is True
        mock_terminate.assert_called_once_with(12345, force=False, include_children=False)

    @patch("cli.commands.server.terminate_pid", side_effect=ProcessLookupError)
    @patch("cli.commands.server._is_process_alive", return_value=True)
    @patch("cli.commands.server._read_pid", return_value=12345)
    def test_process_already_exited_on_kill(self, mock_pid, mock_alive, mock_terminate, capsys):
        from cli.commands.server import _stop_server

        result = _stop_server()
        assert result is True
        assert "already exited" in capsys.readouterr().out

    @patch("cli.commands.server.terminate_pid", side_effect=PermissionError)
    @patch("cli.commands.server._is_process_alive", return_value=True)
    @patch("cli.commands.server._read_pid", return_value=12345)
    def test_permission_error(self, mock_pid, mock_alive, mock_terminate, capsys):
        from cli.commands.server import _stop_server

        result = _stop_server()
        assert result is False
        assert "Permission denied" in capsys.readouterr().out

    @patch("cli.commands.server._remove_pid_file")
    @patch("cli.commands.server.terminate_pid")
    @patch("cli.commands.server._is_process_alive", side_effect=[True, False])
    @patch("cli.commands.server._find_server_pid_by_process", return_value=54321)
    @patch("cli.commands.server._read_pid", return_value=None)
    def test_fallback_to_process_scan(
        self,
        mock_pid,
        mock_find,
        mock_alive,
        mock_terminate,
        mock_remove,
        capsys,
    ):
        """When PID file is missing, fall back to process scan."""
        from cli.commands.server import _stop_server

        result = _stop_server()
        assert result is True
        out = capsys.readouterr().out
        assert "PID file missing" in out
        assert "54321" in out
        mock_terminate.assert_called_once_with(54321, force=False, include_children=False)

    # ── Force mode tests ─────────────────────────────────

    @patch("cli.commands.server._kill_orphan_runners", return_value=0)
    @patch("cli.commands.server._remove_pid_file")
    @patch("cli.commands.server.terminate_pid")
    @patch("time.sleep")
    @patch("time.monotonic")
    @patch("cli.commands.server._is_process_alive")
    @patch("cli.commands.server._read_pid", return_value=12345)
    def test_force_sigkill_after_timeout(
        self,
        mock_pid,
        mock_alive,
        mock_monotonic,
        mock_sleep,
        mock_terminate,
        mock_remove,
        mock_orphans,
        capsys,
    ):
        """Force mode escalates to SIGKILL when SIGTERM times out."""
        from cli.commands.server import _stop_server

        fake_time = [0.0]

        def advance_sleep(seconds):
            fake_time[0] += seconds

        mock_monotonic.side_effect = lambda: fake_time[0]
        mock_sleep.side_effect = advance_sleep

        call_count = [0]

        def alive_side_effect(pid):
            call_count[0] += 1
            return call_count[0] <= 10

        mock_alive.side_effect = alive_side_effect

        result = _stop_server(timeout=1, force=True)
        assert result is True
        out = capsys.readouterr().out
        assert "SIGKILL" in out
        assert "force-killed" in out
        assert mock_terminate.call_count == 2
        assert mock_terminate.call_args_list[0].args == (12345,)
        assert mock_terminate.call_args_list[0].kwargs == {
            "force": False,
            "include_children": False,
        }
        assert mock_terminate.call_args_list[1].args == (12345,)
        assert mock_terminate.call_args_list[1].kwargs == {
            "force": True,
            "include_children": True,
        }

    @patch("cli.commands.server._kill_orphan_runners", return_value=3)
    @patch("cli.commands.server._find_server_pid_by_process", return_value=None)
    @patch("cli.commands.server._read_pid", return_value=None)
    def test_force_kills_orphans_when_no_server(
        self,
        mock_pid,
        mock_find,
        mock_orphans,
        capsys,
    ):
        """Force mode cleans up orphan runners even when server is not running."""
        from cli.commands.server import _stop_server

        result = _stop_server(force=True)
        assert result is True
        out = capsys.readouterr().out
        assert "3 orphan" in out
        mock_orphans.assert_called_once()

    @patch("cli.commands.server._is_process_alive", return_value=True)
    @patch("cli.commands.server.terminate_pid")
    @patch("cli.commands.server._read_pid", return_value=12345)
    def test_non_force_timeout_returns_false(self, mock_pid, mock_terminate, mock_alive, capsys):
        """Without --force, timeout returns False without SIGKILL."""
        from cli.commands.server import _stop_server

        result = _stop_server(timeout=1, force=False)
        assert result is False
        out = capsys.readouterr().out
        assert "did not stop" in out
        assert "SIGKILL" not in out


# ── cmd_start ────────────────────────────────────────────


class TestCmdStart:
    @patch("cli.commands.server._is_process_alive", return_value=True)
    @patch("cli.commands.server._read_pid", return_value=999)
    def test_already_running(self, mock_pid, mock_alive):
        from cli.commands.server import cmd_start

        args = argparse.Namespace(host="0.0.0.0", port=18500)
        with pytest.raises(SystemExit):
            cmd_start(args)

    @patch("cli.commands.server._find_server_pid_by_process", return_value=777)
    @patch("cli.commands.server._is_process_alive", side_effect=lambda pid: pid == 777)
    @patch("cli.commands.server._read_pid", return_value=None)
    def test_already_running_orphan(self, mock_pid, mock_alive, mock_find):
        """Detect running orphan process (no PID file) and refuse to start."""
        from cli.commands.server import cmd_start

        args = argparse.Namespace(host="0.0.0.0", port=18500)
        with pytest.raises(SystemExit):
            cmd_start(args)

    @patch("cli.commands.server._remove_pid_file")
    @patch("cli.commands.server._start_pid_watchdog")
    @patch("uvicorn.run")
    @patch("server.app.create_app")
    @patch("core.paths.get_shared_dir", return_value=Path("/tmp/shared"))
    @patch("core.paths.get_animas_dir", return_value=Path("/tmp/animas"))
    @patch("core.init.ensure_runtime_dir")
    @patch("cli.commands.server._write_pid_file")
    @patch("cli.commands.server._kill_orphan_runners", return_value=0)
    @patch("cli.commands.server._find_server_pid_by_process", return_value=None)
    @patch("cli.commands.server._is_process_alive", return_value=False)
    @patch("cli.commands.server._read_pid", return_value=999)
    def test_stale_pid_cleanup_and_start(
        self,
        mock_pid,
        mock_alive,
        mock_find,
        mock_kill,
        mock_write_pid,
        mock_ensure,
        mock_animas,
        mock_shared,
        mock_create,
        mock_uvicorn,
        mock_watchdog,
        mock_remove,
    ):
        from cli.commands.server import cmd_start

        mock_app = MagicMock()
        mock_create.return_value = mock_app

        args = argparse.Namespace(host="0.0.0.0", port=18500, foreground=True)
        cmd_start(args)

        mock_uvicorn.assert_called_once_with(
            mock_app,
            host="0.0.0.0",
            port=18500,
            log_level="info",
            timeout_keep_alive=65,
            ws_ping_interval=25,
            ws_ping_timeout=5,
        )

    @patch("cli.commands.server._remove_pid_file")
    @patch("cli.commands.server._start_pid_watchdog")
    @patch("uvicorn.run")
    @patch("server.app.create_app")
    @patch("core.paths.get_shared_dir", return_value=Path("/tmp/shared"))
    @patch("core.paths.get_animas_dir", return_value=Path("/tmp/animas"))
    @patch("core.init.ensure_runtime_dir")
    @patch("cli.commands.server._write_pid_file")
    @patch("cli.commands.server._kill_orphan_runners", return_value=0)
    @patch("cli.commands.server._find_server_pid_by_process", return_value=None)
    @patch("cli.commands.server._read_pid", return_value=None)
    def test_uvicorn_timeout_keep_alive(
        self,
        mock_pid,
        mock_find,
        mock_kill,
        mock_write_pid,
        mock_ensure,
        mock_animas,
        mock_shared,
        mock_create,
        mock_uvicorn,
        mock_watchdog,
        mock_remove,
    ):
        from cli.commands.server import cmd_start

        mock_app = MagicMock()
        mock_create.return_value = mock_app

        args = argparse.Namespace(host="0.0.0.0", port=18500, foreground=True)
        cmd_start(args)

        call_kwargs = mock_uvicorn.call_args
        assert call_kwargs.kwargs.get("timeout_keep_alive") == 65

    @patch("cli.commands.server._remove_pid_file")
    @patch("cli.commands.server._start_pid_watchdog")
    @patch("uvicorn.run")
    @patch("server.app.create_app")
    @patch("core.paths.get_shared_dir", return_value=Path("/tmp/shared"))
    @patch("core.paths.get_animas_dir", return_value=Path("/tmp/animas"))
    @patch("core.init.ensure_runtime_dir")
    @patch("cli.commands.server._write_pid_file")
    @patch("cli.commands.server._kill_orphan_runners", return_value=0)
    @patch("cli.commands.server._find_server_pid_by_process", return_value=None)
    @patch("cli.commands.server._read_pid", return_value=None)
    def test_uvicorn_ws_ping_settings(
        self,
        mock_pid,
        mock_find,
        mock_kill,
        mock_write_pid,
        mock_ensure,
        mock_animas,
        mock_shared,
        mock_create,
        mock_uvicorn,
        mock_watchdog,
        mock_remove,
    ):
        from cli.commands.server import cmd_start

        mock_app = MagicMock()
        mock_create.return_value = mock_app

        args = argparse.Namespace(host="0.0.0.0", port=18500, foreground=True)
        cmd_start(args)

        call_kwargs = mock_uvicorn.call_args
        assert call_kwargs.kwargs.get("ws_ping_interval") == 25
        assert call_kwargs.kwargs.get("ws_ping_timeout") == 5


# ── cmd_serve ────────────────────────────────────────────


class TestCmdServe:
    @patch("cli.commands.server.cmd_start")
    def test_serve_delegates_to_start(self, mock_start):
        from cli.commands.server import cmd_serve

        args = argparse.Namespace(host="0.0.0.0", port=18500)
        cmd_serve(args)
        mock_start.assert_called_once_with(args)


# ── cmd_stop ─────────────────────────────────────────────


class TestCmdStop:
    @patch("cli.commands.server._stop_server", return_value=True)
    def test_stop_success(self, mock_stop):
        from cli.commands.server import cmd_stop

        args = argparse.Namespace(force=False)
        cmd_stop(args)
        mock_stop.assert_called_once_with(force=False)

    @patch("cli.commands.server._stop_server", return_value=False)
    def test_stop_failure(self, mock_stop):
        from cli.commands.server import cmd_stop

        args = argparse.Namespace(force=False)
        with pytest.raises(SystemExit):
            cmd_stop(args)

    @patch("cli.commands.server._stop_server", return_value=True)
    def test_stop_force(self, mock_stop):
        from cli.commands.server import cmd_stop

        args = argparse.Namespace(force=True)
        cmd_stop(args)
        mock_stop.assert_called_once_with(force=True)


# ── cmd_restart ──────────────────────────────────────────


class TestCmdRestart:
    @patch("cli.commands.server._clear_pycache", return_value=0)
    @patch("cli.commands.server._stop_server", return_value=True)
    @patch("cli.commands.server._spawn_restart_helper", return_value=99999)
    @patch("cli.commands.server._is_process_alive", return_value=True)
    @patch("cli.commands.server._read_pid", return_value=12345)
    def test_restart_spawns_helper_then_stops(
        self,
        mock_pid,
        mock_alive,
        mock_helper,
        mock_stop,
        mock_clear,
        capsys,
    ):
        from cli.commands.server import cmd_restart

        args = argparse.Namespace(host="0.0.0.0", port=18500, force=False)
        cmd_restart(args)

        mock_helper.assert_called_once_with(args, 12345)
        mock_stop.assert_called_once_with(force=False, extra_exclude_pids={99999})
        out = capsys.readouterr().out
        assert "99999" in out

    @patch("cli.commands.server._clear_pycache", return_value=0)
    @patch("cli.commands.server._stop_server", return_value=True)
    @patch("cli.commands.server._spawn_restart_helper", return_value=99999)
    @patch("cli.commands.server._is_process_alive", return_value=True)
    @patch("cli.commands.server._read_pid", return_value=12345)
    def test_restart_with_force(
        self,
        mock_pid,
        mock_alive,
        mock_helper,
        mock_stop,
        mock_clear,
    ):
        from cli.commands.server import cmd_restart

        args = argparse.Namespace(host="0.0.0.0", port=18500, force=True)
        cmd_restart(args)

        mock_stop.assert_called_once_with(force=True, extra_exclude_pids={99999})

    @patch("cli.commands.server._clear_pycache", return_value=0)
    @patch("cli.commands.server._stop_server", return_value=True)
    @patch("cli.commands.server._spawn_restart_helper", return_value=99999)
    @patch("cli.commands.server._find_server_pid_by_process", return_value=None)
    @patch("cli.commands.server._is_process_alive", return_value=False)
    @patch("cli.commands.server._read_pid", return_value=12345)
    def test_restart_stale_pid_falls_back_to_scan(
        self,
        mock_pid,
        mock_alive,
        mock_find,
        mock_helper,
        mock_stop,
        mock_clear,
    ):
        """When PID exists but process is dead, falls back to process scan."""
        from cli.commands.server import cmd_restart

        args = argparse.Namespace(host="0.0.0.0", port=18500, force=False)
        cmd_restart(args)

        mock_find.assert_called_once()
        mock_helper.assert_called_once_with(args, None)

    @patch("cli.commands.server._clear_pycache", return_value=0)
    @patch("cli.commands.server._stop_server", return_value=True)
    @patch("cli.commands.server._spawn_restart_helper", return_value=99999)
    @patch("cli.commands.server._find_server_pid_by_process", return_value=None)
    @patch("cli.commands.server._read_pid", return_value=None)
    def test_restart_no_pid_no_process(self, mock_pid, mock_find, mock_helper, mock_stop, mock_clear):
        """When no PID file and no process found, old_pid=None."""
        from cli.commands.server import cmd_restart

        args = argparse.Namespace(host="0.0.0.0", port=18500, force=False)
        cmd_restart(args)

        mock_find.assert_called_once()
        mock_helper.assert_called_once_with(args, None)

    @patch("cli.commands.server._clear_pycache", return_value=0)
    @patch("cli.commands.server._stop_server", return_value=True)
    @patch("cli.commands.server._spawn_restart_helper", return_value=99999)
    @patch("cli.commands.server._find_server_pid_by_process", return_value=54321)
    @patch("cli.commands.server._read_pid", return_value=None)
    def test_restart_no_pid_file_finds_by_scan(
        self,
        mock_pid,
        mock_find,
        mock_helper,
        mock_stop,
        mock_clear,
    ):
        """When PID file is missing but process scan finds server, passes scanned PID."""
        from cli.commands.server import cmd_restart

        args = argparse.Namespace(host="0.0.0.0", port=18500, force=False)
        cmd_restart(args)

        mock_find.assert_called_once()
        mock_helper.assert_called_once_with(args, 54321)


class TestSpawnRestartHelper:
    @patch("cli.commands.server._get_daemon_log_path")
    def test_helper_starts_detached_process(self, mock_log_path, tmp_path):
        from cli.commands.server import _spawn_restart_helper

        log_file = tmp_path / "daemon.log"
        mock_log_path.return_value = log_file

        args = argparse.Namespace(host="0.0.0.0", port=18500)

        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.pid = 77777
            mock_popen.return_value = mock_proc

            pid = _spawn_restart_helper(args, old_pid=12345)

        assert pid == 77777
        call_kwargs = mock_popen.call_args
        for key, value in subprocess_session_kwargs().items():
            assert call_kwargs.kwargs[key] == value
        helper_code = mock_popen.call_args.args[0][2]
        assert "find_first_matching_pid" in helper_code
        assert "terminate_pid" in helper_code
        assert "/proc" not in helper_code
        assert "os.killpg" not in helper_code

    @patch("cli.commands.server._get_daemon_log_path")
    def test_helper_accepts_none_old_pid(self, mock_log_path, tmp_path):
        from cli.commands.server import _spawn_restart_helper

        log_file = tmp_path / "daemon.log"
        mock_log_path.return_value = log_file

        args = argparse.Namespace(host="0.0.0.0", port=18500)

        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.pid = 88888
            mock_popen.return_value = mock_proc

            pid = _spawn_restart_helper(args, old_pid=None)

        assert pid == 88888


# ── Deprecated commands ──────────────────────────────────


class TestDeprecatedCommands:
    def test_gateway_deprecated(self):
        from cli.commands.server import cmd_gateway

        args = argparse.Namespace()
        with pytest.raises(SystemExit):
            cmd_gateway(args)

    def test_worker_deprecated(self):
        from cli.commands.server import cmd_worker

        args = argparse.Namespace()
        with pytest.raises(SystemExit):
            cmd_worker(args)


# ── _clear_pycache ───────────────────────────────────────


class TestClearPycache:
    def test_clear_pycache(self, tmp_path):
        """Verify _clear_pycache removes __pycache__ directories."""

        from cli.commands.server import _clear_pycache

        # _clear_pycache uses Path(__file__) to find the project root.
        # We patch __file__ at the module level to point into tmp_path.
        fake_server_py = tmp_path / "cli" / "commands" / "server.py"
        fake_server_py.parent.mkdir(parents=True, exist_ok=True)
        fake_server_py.touch()

        # Create __pycache__ dirs under tmp_path (the "project root")
        cache1 = tmp_path / "src" / "__pycache__"
        cache1.mkdir(parents=True)
        cache2 = tmp_path / "lib" / "__pycache__"
        cache2.mkdir(parents=True)

        import cli.commands.server as server_mod

        original = server_mod.__file__
        try:
            server_mod.__file__ = str(fake_server_py)
            count = _clear_pycache()
            assert count == 2
            assert not cache1.exists()
            assert not cache2.exists()
        finally:
            server_mod.__file__ = original
