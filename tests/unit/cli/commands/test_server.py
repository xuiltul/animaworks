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

# ── Public PID lifecycle and process discovery ────────────


class TestPidLifecycle:
    def test_foreground_start_writes_and_cleans_pid_file(
        self,
        tmp_path,
        data_dir_at_tmp_path,
        monkeypatch,
    ):
        """Exercise PID persistence through the public foreground-start command."""
        from cli.commands.server import cmd_start
        from core.platform.env import SERVER_URL_ENV
        from core.platform.pid import read_server_pid
        from core.runtime.process_role import get_process_role

        monkeypatch.delenv("ANIMAWORKS_PROCESS_ROLE", raising=False)
        monkeypatch.setenv(SERVER_URL_ENV, "http://127.0.0.1:18500")
        args = argparse.Namespace(host="127.0.0.1", port=18500, foreground=True)
        app = MagicMock()

        def assert_pid_written(*_args, **_kwargs):
            assert get_process_role() == "root"
            assert read_server_pid() == os.getpid()

        with (
            patch("core.infra.runtime_init.ensure_runtime_dir"),
            patch("core.platform.fd_limits.raise_fd_soft_limit"),
            patch("server.app.create_app", return_value=app),
            patch("cli.commands.server.find_first_matching_pid", return_value=None),
            patch("cli.commands.server.terminate_matching_processes", return_value=0),
            patch("uvicorn.run", side_effect=assert_pid_written) as run_server,
            patch("atexit.register"),
            patch("threading.Thread") as thread_class,
        ):
            thread_class.return_value.start.return_value = None
            cmd_start(args)

        run_server.assert_called_once()
        assert read_server_pid() is None


class TestFindServerThroughStopCommand:
    """Verify process discovery through the user-facing ``cmd_stop`` command."""

    @pytest.fixture(autouse=True)
    def _runtime_data_dir(self, data_dir_at_tmp_path):
        pass

    def test_no_matching_process_reports_not_running(self, capsys):
        from cli.commands.server import cmd_stop

        with (
            patch("cli.commands.server.find_first_matching_pid", return_value=None) as find_process,
            patch("cli.commands.server.terminate_matching_processes", return_value=0),
        ):
            cmd_stop(argparse.Namespace(force=False))

        assert "not running" in capsys.readouterr().out
        find_process.assert_called_once()

    def test_process_env_data_dir_is_used(self, tmp_path):
        from cli.commands.server import cmd_stop

        process = MagicMock()
        process.cmdline.return_value = ["python", "-m", "cli", "start"]
        with (
            patch("cli.commands.server.find_first_matching_pid", return_value=12345) as find_process,
            patch("cli.commands.server.psutil.Process", return_value=process),
            patch("cli.commands.server.read_process_env", return_value=(True, str(tmp_path))) as read_env,
            patch("cli.commands.server.is_pid_alive", return_value=False),
            patch("cli.commands.server.terminate_pid") as terminate,
            patch("cli.commands.server.terminate_matching_processes", return_value=0),
        ):
            cmd_stop(argparse.Namespace(force=False))

        read_env.assert_called_once()
        find_process.assert_called_once()
        terminate.assert_called_once_with(12345, force=False, include_children=False)

    def test_explicit_different_data_dir_is_ignored(self, tmp_path, capsys):
        from cli.commands.server import cmd_stop

        process = MagicMock()
        process.cmdline.return_value = ["python", "-m", "cli", "start", "--data-dir=/tmp/other-data"]
        with (
            patch("cli.commands.server.find_first_matching_pid", side_effect=[12345, None]) as find_process,
            patch("cli.commands.server.psutil.Process", return_value=process),
            patch("cli.commands.server.read_process_env") as read_env,
            patch("cli.commands.server.terminate_matching_processes", return_value=0),
        ):
            cmd_stop(argparse.Namespace(force=False))

        read_env.assert_not_called()
        assert find_process.call_count == 2
        assert 12345 in find_process.call_args_list[1].kwargs["exclude_pids"]
        assert "not running" in capsys.readouterr().out

    def test_relative_data_dir_resolves_against_process_cwd(self, tmp_path, monkeypatch):
        from cli.commands.server import cmd_stop

        runtime_dir = tmp_path / "runtime"
        monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(runtime_dir))
        process = MagicMock()
        process.cmdline.return_value = ["animaworks", "start", "--data-dir", "runtime"]
        process.cwd.return_value = str(tmp_path)
        with (
            patch("cli.commands.server.find_first_matching_pid", return_value=12345),
            patch("cli.commands.server.psutil.Process", return_value=process),
            patch("cli.commands.server.is_pid_alive", return_value=False),
            patch("cli.commands.server.terminate_pid") as terminate,
            patch("cli.commands.server.terminate_matching_processes", return_value=0),
        ):
            cmd_stop(argparse.Namespace(force=False))

        process.cwd.assert_called_once()
        terminate.assert_called_once_with(12345, force=False, include_children=False)


# ── Public stop command ───────────────────────────────────


class TestStopServer:
    @pytest.fixture(autouse=True)
    def _runtime_data_dir(self, data_dir_at_tmp_path):
        pass

    @pytest.mark.parametrize("exit_after", [12.0, 89.0])
    def test_default_waits_for_whole_server_shutdown_without_force_kill(self, exit_after):
        from cli.commands.server import cmd_stop

        clock = [0.0]

        def sleep(seconds):
            clock[0] += seconds

        with (
            patch("cli.commands.server.time.monotonic", side_effect=lambda: clock[0]),
            patch("cli.commands.server.time.sleep", side_effect=sleep),
            patch("cli.commands.server.read_server_pid", return_value=12345),
            patch("cli.commands.server.is_pid_alive", side_effect=lambda _pid: clock[0] < exit_after),
            patch("cli.commands.server.terminate_pid") as terminate,
            patch("cli.commands.server.terminate_matching_processes", return_value=0),
            patch("cli.commands.server._restart_drain_timeout_seconds", return_value=30),
        ):
            cmd_stop(argparse.Namespace(force=False))

        assert exit_after <= clock[0] < 90
        terminate.assert_called_once_with(12345, force=False, include_children=False)

    def test_default_still_reports_failure_at_ninety_seconds(self):
        from cli.commands.server import cmd_stop

        clock = [0.0]

        def sleep(seconds):
            clock[0] += seconds

        with (
            patch("cli.commands.server.time.monotonic", side_effect=lambda: clock[0]),
            patch("cli.commands.server.time.sleep", side_effect=sleep),
            patch("cli.commands.server.read_server_pid", return_value=12345),
            patch("cli.commands.server.is_pid_alive", return_value=True),
            patch("cli.commands.server.terminate_pid") as terminate,
            patch("cli.commands.server._restart_drain_timeout_seconds", return_value=30),
            pytest.raises(SystemExit) as stopped,
        ):
            cmd_stop(argparse.Namespace(force=False))

        assert stopped.value.code == 1
        assert 90 <= clock[0] < 90.3
        terminate.assert_called_once_with(12345, force=False, include_children=False)

    def test_no_pid_file_no_process(self, capsys):
        from cli.commands.server import cmd_stop

        with (
            patch("cli.commands.server.read_server_pid", return_value=None),
            patch("cli.commands.server.find_first_matching_pid", return_value=None) as find_process,
            patch("cli.commands.server.terminate_matching_processes", return_value=0) as clean_orphans,
        ):
            cmd_stop(argparse.Namespace(force=False))

        assert "not running" in capsys.readouterr().out
        find_process.assert_called_once()
        clean_orphans.assert_called_once()

    def test_stale_pid(self, capsys):
        from cli.commands.server import cmd_stop

        with (
            patch("cli.commands.server.read_server_pid", return_value=12345),
            patch("cli.commands.server.is_pid_alive", return_value=False),
            patch("cli.commands.server.terminate_matching_processes", return_value=0) as clean_orphans,
        ):
            cmd_stop(argparse.Namespace(force=False))

        assert "Stale" in capsys.readouterr().out
        clean_orphans.assert_called_once()

    def test_successful_stop(self):
        from cli.commands.server import cmd_stop

        with (
            patch("cli.commands.server.read_server_pid", return_value=12345),
            patch("cli.commands.server.is_pid_alive", side_effect=[True, False]),
            patch("cli.commands.server.terminate_pid") as terminate,
            patch("cli.commands.server.terminate_matching_processes", return_value=0),
        ):
            cmd_stop(argparse.Namespace(force=False))

        terminate.assert_called_once_with(12345, force=False, include_children=False)

    def test_process_already_exited_on_kill(self, capsys):
        from cli.commands.server import cmd_stop

        with (
            patch("cli.commands.server.read_server_pid", return_value=12345),
            patch("cli.commands.server.is_pid_alive", return_value=True),
            patch("cli.commands.server.terminate_pid", side_effect=ProcessLookupError),
            patch("cli.commands.server.terminate_matching_processes", return_value=0),
        ):
            cmd_stop(argparse.Namespace(force=False))

        assert "already exited" in capsys.readouterr().out

    def test_permission_error(self, capsys):
        from cli.commands.server import cmd_stop

        with (
            patch("cli.commands.server.read_server_pid", return_value=12345),
            patch("cli.commands.server.is_pid_alive", return_value=True),
            patch("cli.commands.server.terminate_pid", side_effect=PermissionError),
            pytest.raises(SystemExit) as stopped,
        ):
            cmd_stop(argparse.Namespace(force=False))

        assert stopped.value.code == 1
        assert "Permission denied" in capsys.readouterr().out

    def test_fallback_to_process_scan(self, tmp_path: Path, capsys):
        from cli.commands.server import cmd_stop

        process = MagicMock()
        process.cmdline.return_value = ["python", "-m", "cli", "start"]
        with (
            patch("cli.commands.server.read_server_pid", return_value=None),
            patch("cli.commands.server.find_first_matching_pid", return_value=54321) as find_process,
            patch("cli.commands.server.psutil.Process", return_value=process),
            patch("cli.commands.server.read_process_env", return_value=(True, str(tmp_path))),
            patch("cli.commands.server.is_pid_alive", return_value=False),
            patch("cli.commands.server.terminate_pid") as terminate,
            patch("cli.commands.server.terminate_matching_processes", return_value=0),
        ):
            cmd_stop(argparse.Namespace(force=False))

        output = capsys.readouterr().out
        assert "PID file missing" in output
        assert "54321" in output
        find_process.assert_called_once()
        terminate.assert_called_once_with(54321, force=False, include_children=False)

    def test_force_sigkill_after_timeout(self, capsys):
        """Force mode escalates to SIGKILL when SIGTERM times out."""
        from cli.commands.server import cmd_stop

        clock = [0.0]
        alive = iter([True, True, False, False])

        def sleep(_seconds):
            clock[0] += 100.0

        with (
            patch("cli.commands.server.time.monotonic", side_effect=lambda: clock[0]),
            patch("cli.commands.server.time.sleep", side_effect=sleep),
            patch("cli.commands.server.read_server_pid", return_value=12345),
            patch("cli.commands.server.is_pid_alive", side_effect=lambda _pid: next(alive)),
            patch("cli.commands.server.terminate_pid") as terminate,
            patch("cli.commands.server.terminate_matching_processes", return_value=0),
        ):
            cmd_stop(argparse.Namespace(force=True))

        output = capsys.readouterr().out
        assert "SIGKILL" in output
        assert "force-killed" in output
        assert terminate.call_count == 2
        assert terminate.call_args_list[0].args == (12345,)
        assert terminate.call_args_list[0].kwargs == {"force": False, "include_children": False}
        assert terminate.call_args_list[1].args == (12345,)
        assert terminate.call_args_list[1].kwargs == {"force": True, "include_children": True}

    def test_non_force_kills_orphans_when_no_server(self, capsys):
        """Normal stop also cleans up orphan runners when server is not running."""
        from cli.commands.server import cmd_stop

        with (
            patch("cli.commands.server.read_server_pid", return_value=None),
            patch("cli.commands.server.find_first_matching_pid", return_value=None),
            patch("cli.commands.server.terminate_matching_processes", return_value=3),
        ):
            cmd_stop(argparse.Namespace(force=False))

        output = capsys.readouterr().out
        assert "3 orphan" in output
        assert "not running" in output

    def test_non_force_timeout_returns_failure(self, capsys):
        """Without --force, timeout exits with failure and does not send SIGKILL."""
        from cli.commands.server import cmd_stop

        clock = [0.0]

        def sleep(_seconds):
            clock[0] += 100.0

        with (
            patch("cli.commands.server.time.monotonic", side_effect=lambda: clock[0]),
            patch("cli.commands.server.time.sleep", side_effect=sleep),
            patch("cli.commands.server.read_server_pid", return_value=12345),
            patch("cli.commands.server.is_pid_alive", return_value=True),
            patch("cli.commands.server.terminate_pid") as terminate,
            pytest.raises(SystemExit) as stopped,
        ):
            cmd_stop(argparse.Namespace(force=False))

        assert stopped.value.code == 1
        output = capsys.readouterr().out
        assert "did not stop" in output
        assert "SIGKILL" not in output
        terminate.assert_called_once_with(12345, force=False, include_children=False)


# ── Public cmd_start behavior ─────────────────────────────


@pytest.fixture(autouse=True)
def _server_test_data_dir(data_dir_at_tmp_path):
    """Keep server command paths inside each test's temporary root."""


@pytest.fixture(autouse=True)
def _pid_owner_check_off(monkeypatch):
    """Fake PIDs (999, 12345) may be real processes on the test host; keep them "possibly ours".

    Tests of the reused-PID handling switch the check on explicitly.
    """
    monkeypatch.setattr("cli.commands.server.is_unrelated_process", lambda _pid: False)


class TestReusedPidFile:
    """A server.pid left from before a reboot can name a live, unrelated process."""

    def test_foreground_start_ignores_pid_reused_by_unrelated_process(self, foreground_server_mocks, monkeypatch):
        from cli.commands.server import cmd_start

        (foreground_server_mocks["data_dir"] / "server.pid").write_text("599\n", encoding="utf-8")
        monkeypatch.setattr("cli.commands.server.is_pid_alive", lambda pid: pid == 599)
        monkeypatch.setattr("cli.commands.server.is_unrelated_process", lambda pid: pid == 599)

        cmd_start(argparse.Namespace(host="127.0.0.1", port=18500, foreground=True))

        foreground_server_mocks["run_server"].assert_called_once()

    def test_foreground_start_ignores_pid_file_naming_itself(self, foreground_server_mocks, monkeypatch):
        """After a reboot the server can get the same PID it had before."""
        from cli.commands.server import cmd_start

        (foreground_server_mocks["data_dir"] / "server.pid").write_text(f"{os.getpid()}\n", encoding="utf-8")

        cmd_start(argparse.Namespace(host="127.0.0.1", port=18500, foreground=True))

        foreground_server_mocks["run_server"].assert_called_once()

    def test_foreground_start_still_refuses_possible_server(self, foreground_server_mocks, monkeypatch):
        from cli.commands.server import EXIT_ALREADY_RUNNING, cmd_start

        (foreground_server_mocks["data_dir"] / "server.pid").write_text("599\n", encoding="utf-8")
        monkeypatch.setattr("cli.commands.server.is_pid_alive", lambda pid: pid == 599)

        with pytest.raises(SystemExit) as exc:
            cmd_start(argparse.Namespace(host="127.0.0.1", port=18500, foreground=True))

        assert exc.value.code == EXIT_ALREADY_RUNNING
        foreground_server_mocks["run_server"].assert_not_called()

    def test_stop_does_not_signal_unrelated_process(self, capsys, monkeypatch):
        from cli.commands.server import cmd_stop

        monkeypatch.setattr("cli.commands.server.read_server_pid", lambda: 599)
        monkeypatch.setattr("cli.commands.server.is_pid_alive", lambda pid: pid == 599)
        monkeypatch.setattr("cli.commands.server.is_unrelated_process", lambda pid: pid == 599)
        with (
            patch("cli.commands.server.terminate_pid") as terminate,
            patch("cli.commands.server.terminate_matching_processes", return_value=0),
            patch("cli.commands.server._remove_pid_file") as remove_pid_file,
        ):
            cmd_stop(argparse.Namespace(force=False))

        terminate.assert_not_called()
        remove_pid_file.assert_called_once()
        assert "Stale PID file (pid=599)" in capsys.readouterr().out


@pytest.fixture
def foreground_server_mocks(monkeypatch, data_dir_at_tmp_path):
    from core.platform.pid import read_server_pid

    app = MagicMock()
    create_app = MagicMock(return_value=app)

    def _run_server(*_args, **_kwargs):
        assert read_server_pid() == os.getpid()

    run_server = MagicMock(side_effect=_run_server)
    thread_factory = MagicMock()
    find_process = MagicMock(return_value=None)
    clean_orphans = MagicMock(return_value=0)
    monkeypatch.setattr("server.app.create_app", create_app)
    monkeypatch.setattr("uvicorn.run", run_server)
    monkeypatch.setattr("core.infra.runtime_init.ensure_runtime_dir", lambda: None)
    monkeypatch.setattr("core.platform.fd_limits.raise_fd_soft_limit", lambda **_kwargs: None)
    monkeypatch.setattr("cli.commands.server.find_first_matching_pid", find_process)
    monkeypatch.setattr("cli.commands.server.terminate_matching_processes", clean_orphans)
    monkeypatch.setattr("threading.Thread", thread_factory)
    monkeypatch.setattr("atexit.register", lambda *_args, **_kwargs: None)
    from core.platform.env import SERVER_URL_ENV

    monkeypatch.setenv(SERVER_URL_ENV, "")
    return {
        "app": app,
        "create_app": create_app,
        "run_server": run_server,
        "data_dir": data_dir_at_tmp_path,
        "find_process": find_process,
    }


class TestCmdStart:
    def test_already_running(self):
        from cli.commands.server import EXIT_ALREADY_RUNNING, cmd_start

        args = argparse.Namespace(host="0.0.0.0", port=18500)
        with (
            patch("cli.commands.server.is_pid_alive", return_value=True),
            patch("cli.commands.server.read_server_pid", return_value=999),
            pytest.raises(SystemExit) as exc,
        ):
            cmd_start(args)
        assert exc.value.code == EXIT_ALREADY_RUNNING

    def test_already_running_orphan(self, tmp_path):
        """A process found by the shared process adapter prevents daemon startup."""
        from cli.commands.server import EXIT_ALREADY_RUNNING, cmd_start

        process = MagicMock()
        process.cmdline.return_value = ["python", "-m", "cli", "start", "--data-dir", str(tmp_path)]
        args = argparse.Namespace(host="0.0.0.0", port=18500)
        with (
            patch("cli.commands.server.psutil.Process", return_value=process),
            patch("cli.commands.server.find_first_matching_pid", return_value=777) as find_process,
            patch("cli.commands.server.is_pid_alive", side_effect=lambda pid: pid == 777),
            patch("cli.commands.server.read_server_pid", return_value=None),
            pytest.raises(SystemExit) as exc,
        ):
            cmd_start(args)
        assert exc.value.code == EXIT_ALREADY_RUNNING
        find_process.assert_called_once()

    def test_foreground_already_running(self):
        from cli.commands.server import EXIT_ALREADY_RUNNING, cmd_start

        args = argparse.Namespace(host="0.0.0.0", port=18500, foreground=True)
        with (
            patch("cli.commands.server.is_pid_alive", return_value=True),
            patch("cli.commands.server.read_server_pid", return_value=999),
            pytest.raises(SystemExit) as exc,
        ):
            cmd_start(args)
        assert exc.value.code == EXIT_ALREADY_RUNNING

    def test_foreground_already_running_orphan(self, tmp_path):
        from cli.commands.server import EXIT_ALREADY_RUNNING, cmd_start

        process = MagicMock()
        process.cmdline.return_value = ["python", "-m", "cli", "start", "--data-dir", str(tmp_path)]
        args = argparse.Namespace(host="0.0.0.0", port=18500, foreground=True)
        with (
            patch("cli.commands.server.psutil.Process", return_value=process),
            patch("cli.commands.server.find_first_matching_pid", return_value=777) as find_process,
            patch("cli.commands.server.is_pid_alive", side_effect=lambda pid: pid == 777),
            patch("cli.commands.server.read_server_pid", return_value=None),
            pytest.raises(SystemExit) as exc,
        ):
            cmd_start(args)
        assert exc.value.code == EXIT_ALREADY_RUNNING
        find_process.assert_called_once()

    def test_pid1_runs_foreground(self, foreground_server_mocks, monkeypatch):
        """As PID 1 (container) daemonize is impossible, so the public command serves foreground."""
        from cli.commands.server import cmd_start

        monkeypatch.setattr(os, "getpid", lambda: 1)
        cmd_start(argparse.Namespace(host="0.0.0.0", port=18500))

        foreground_server_mocks["create_app"].assert_called_once()
        foreground_server_mocks["run_server"].assert_called_once()

    def test_normal_pid_daemonizes(self, tmp_path, monkeypatch):
        """A normal process starts a detached child through the public subprocess boundary."""
        from cli.commands.server import cmd_start

        process = MagicMock()
        process.pid = 23456
        process.poll.return_value = None
        popen = MagicMock(return_value=process)
        monkeypatch.setattr("cli.commands.server.subprocess.Popen", popen)
        monkeypatch.setattr("cli.commands.server.find_first_matching_pid", MagicMock(return_value=None))
        monkeypatch.setattr("cli.commands.server.socket.create_connection", MagicMock(return_value=MagicMock()))

        cmd_start(argparse.Namespace(host="0.0.0.0", port=18500))

        popen.assert_called_once()

    def test_stale_pid_cleanup_and_start(self, foreground_server_mocks, monkeypatch):
        from cli.commands.server import cmd_start
        from core.platform.pid import read_server_pid

        pid_file = foreground_server_mocks["data_dir"] / "server.pid"
        pid_file.write_text("999", encoding="utf-8")
        monkeypatch.setattr("cli.commands.server.is_pid_alive", lambda _pid: False)

        cmd_start(argparse.Namespace(host="0.0.0.0", port=18507, foreground=True))

        assert foreground_server_mocks["run_server"].call_args.kwargs["port"] == 18507
        assert read_server_pid() is None

    def test_uvicorn_timeout_keep_alive(self, foreground_server_mocks):
        from cli.commands.server import cmd_start

        cmd_start(argparse.Namespace(host="0.0.0.0", port=18500, foreground=True))

        assert foreground_server_mocks["run_server"].call_args.kwargs["timeout_keep_alive"] == 65

    def test_uvicorn_ws_ping_settings(self, foreground_server_mocks):
        from cli.commands.server import cmd_start

        cmd_start(argparse.Namespace(host="0.0.0.0", port=18500, foreground=True))

        kwargs = foreground_server_mocks["run_server"].call_args.kwargs
        assert kwargs["ws_ping_interval"] == 25
        assert kwargs["ws_ping_timeout"] == 5


# ── cmd_serve ────────────────────────────────────────────


class TestCmdServe:
    @patch("cli.commands.server.cmd_start")
    def test_serve_delegates_to_start(self, mock_start):
        from cli.commands.server import cmd_serve

        args = argparse.Namespace(host="0.0.0.0", port=18500)
        cmd_serve(args)
        mock_start.assert_called_once_with(args)


# cmd_stop behavior is covered end-to-end by TestStopServer above.


# ── Public cmd_restart behavior ──────────────────────────


@pytest.fixture
def restart_command_mocks(monkeypatch, data_dir_at_tmp_path):
    """Stub OS process/socket boundaries while using real restart helpers."""
    helper = MagicMock()
    helper.pid = 99999
    popen = MagicMock(return_value=helper)
    socket_connection = MagicMock()
    clean_orphans = MagicMock(return_value=0)
    monkeypatch.setattr("cli.commands.server.subprocess.Popen", popen)
    monkeypatch.setattr("cli.commands.server.socket.create_connection", lambda *_args, **_kwargs: socket_connection)
    monkeypatch.setattr("cli.commands.server.terminate_matching_processes", clean_orphans)
    monkeypatch.setattr("shutil.rmtree", lambda _path: None)
    return {
        "data_dir": data_dir_at_tmp_path,
        "popen": popen,
        "helper": helper,
        "clean_orphans": clean_orphans,
    }


class TestCmdRestart:
    def test_restart_spawns_helper_before_stopping(self, restart_command_mocks, monkeypatch, capsys):
        from cli.commands.server import cmd_restart

        data_dir = restart_command_mocks["data_dir"]
        (data_dir / "server.pid").write_text("12345", encoding="utf-8")
        monkeypatch.setattr("cli.commands.server.find_first_matching_pid", MagicMock(return_value=None))
        monkeypatch.setattr("cli.commands.server.is_pid_alive", MagicMock(side_effect=[True, True, False]))
        terminate = MagicMock()
        monkeypatch.setattr("cli.commands.server.terminate_pid", terminate)

        cmd_restart(argparse.Namespace(host="0.0.0.0", port=18500, force=False))

        helper_code = restart_command_mocks["popen"].call_args.args[0][2]
        assert "old_pid = 12345" in helper_code
        assert terminate.call_args.kwargs == {"force": False, "include_children": False}
        assert "99999" in capsys.readouterr().out

    def test_restart_force_escalates_through_stop_api(self, restart_command_mocks, monkeypatch, capsys):
        from cli.commands.server import cmd_restart

        data_dir = restart_command_mocks["data_dir"]
        (data_dir / "server.pid").write_text("12345", encoding="utf-8")
        monkeypatch.setattr("cli.commands.server.find_first_matching_pid", MagicMock(return_value=None))
        alive = MagicMock(side_effect=[True, True, True, False, False])
        monkeypatch.setattr("cli.commands.server.is_pid_alive", alive)
        clock = [0.0]
        monkeypatch.setattr("cli.commands.server.time.monotonic", lambda: clock[0])
        monkeypatch.setattr("cli.commands.server.time.sleep", lambda _seconds: clock.__setitem__(0, clock[0] + 100.0))
        terminate = MagicMock()
        monkeypatch.setattr("cli.commands.server.terminate_pid", terminate)

        cmd_restart(argparse.Namespace(host="0.0.0.0", port=18500, force=True))

        assert terminate.call_count == 2
        assert terminate.call_args_list[0].kwargs == {"force": False, "include_children": False}
        assert terminate.call_args_list[1].kwargs == {"force": True, "include_children": True}
        assert "force-killed" in capsys.readouterr().out

    def test_restart_stale_pid_falls_back_to_scan(self, restart_command_mocks, monkeypatch):
        from cli.commands.server import cmd_restart

        data_dir = restart_command_mocks["data_dir"]
        (data_dir / "server.pid").write_text("12345", encoding="utf-8")
        find_process = MagicMock(return_value=None)
        monkeypatch.setattr("cli.commands.server.find_first_matching_pid", find_process)
        monkeypatch.setattr("cli.commands.server.is_pid_alive", MagicMock(side_effect=[False, False]))

        cmd_restart(argparse.Namespace(host="0.0.0.0", port=18500, force=False))

        helper_code = restart_command_mocks["popen"].call_args.args[0][2]
        assert "old_pid = None" in helper_code
        find_process.assert_called_once()

    def test_restart_without_pid_uses_public_process_scan(self, restart_command_mocks, monkeypatch):
        from cli.commands.server import cmd_restart

        data_dir = restart_command_mocks["data_dir"]
        process = MagicMock()
        process.cmdline.return_value = ["python", "-m", "cli", "start", "--data-dir", str(data_dir)]
        monkeypatch.setattr("cli.commands.server.psutil.Process", lambda _pid: process)
        find_process = MagicMock(return_value=54321)
        monkeypatch.setattr("cli.commands.server.find_first_matching_pid", find_process)
        monkeypatch.setattr("cli.commands.server.is_pid_alive", lambda _pid: False)
        terminate = MagicMock()
        monkeypatch.setattr("cli.commands.server.terminate_pid", terminate)

        cmd_restart(argparse.Namespace(host="0.0.0.0", port=18500, force=False))

        helper_code = restart_command_mocks["popen"].call_args.args[0][2]
        assert "old_pid = 54321" in helper_code
        assert find_process.call_count == 2
        terminate.assert_called_once_with(54321, force=False, include_children=False)


class TestSpawnRestartHelper:
    def test_helper_starts_detached_process(self):
        from cli.commands.server import _spawn_restart_helper

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
        assert "Lingering server process still detected" in helper_code
        assert "include_children=True" in helper_code
        assert "/proc/" not in helper_code
        assert "os.killpg" not in helper_code

    def test_helper_accepts_none_old_pid(self):
        from cli.commands.server import _spawn_restart_helper

        args = argparse.Namespace(host="0.0.0.0", port=18500)

        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.pid = 88888
            mock_popen.return_value = mock_proc

            pid = _spawn_restart_helper(args, old_pid=None)

        assert pid == 88888

    def test_old_server_kill_grace_uses_drain_timeout_plus_margin(self, monkeypatch):
        from cli.commands.server import _spawn_restart_helper

        monkeypatch.setattr("cli.commands.server._restart_drain_timeout_seconds", lambda: 123)
        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value.pid = 999
            _spawn_restart_helper(argparse.Namespace(host="127.0.0.1", port=18500), old_pid=111)

        helper_code = mock_popen.call_args.args[0][2]
        assert "OLD_SERVER_WAIT_SECONDS = 183" in helper_code
        assert "time.monotonic() + OLD_SERVER_WAIT_SECONDS" in helper_code


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


class TestKillOrphanRunnersScope:
    """Orphan cleanup must stay inside its own data dir."""

    def test_sibling_data_dir_is_not_matched(self, tmp_path):
        from cli.commands.server import _kill_orphan_runners

        data_dir = tmp_path / ".animaworks"
        with (
            patch("core.paths.get_data_dir", return_value=data_dir),
            patch("cli.commands.server.terminate_matching_processes", return_value=0) as terminate,
        ):
            _kill_orphan_runners()

        path_filter = terminate.call_args.kwargs["path_contains"]
        own = f"--socket-path {data_dir}/run/sockets/a.sock"
        sibling = f"--socket-path {tmp_path}/.animaworks-demo/run/sockets/a.sock"
        assert path_filter in own
        assert path_filter not in sibling
