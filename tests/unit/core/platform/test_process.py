"""Unit tests for core.platform.process."""

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import signal
from unittest.mock import MagicMock, patch

import psutil
import pytest

from core.platform import process


class TestSubprocessSessionKwargs:
    def test_windows_returns_creationflags(self):
        with (
            patch("core.platform.process.os.name", "nt"),
            patch("core.platform.process.subprocess.CREATE_NEW_PROCESS_GROUP", 512, create=True),
        ):
            assert process.subprocess_session_kwargs() == {"creationflags": 512}

    def test_posix_returns_start_new_session(self):
        with patch("core.platform.process.os.name", "posix"):
            assert process.subprocess_session_kwargs() == {"start_new_session": True}


class TestTerminatePid:
    def test_windows_terminate_pid_kills_children_without_killpg(self):
        child = MagicMock()
        proc_obj = MagicMock()
        proc_obj.children.return_value = [child]

        with (
            patch("core.platform.process.os.name", "nt"),
            patch("core.platform.process.psutil.Process", return_value=proc_obj),
            patch("core.platform.process._terminate_psutil_process") as mock_terminate,
        ):
            process.terminate_pid(12345, force=True, include_children=True)

        assert mock_terminate.call_args_list[0].args == (child,)
        assert mock_terminate.call_args_list[0].kwargs == {"force": True}
        assert mock_terminate.call_args_list[1].args == (proc_obj,)
        assert mock_terminate.call_args_list[1].kwargs == {"force": True}

    def test_missing_pid_is_ignored(self):
        with patch(
            "core.platform.process.psutil.Process",
            side_effect=psutil.NoSuchProcess(pid=99999),
        ):
            process.terminate_pid(99999)

    def test_include_children_snapshots_before_group_signal(self):
        calls = []
        child = MagicMock()
        child.terminate.side_effect = lambda: calls.append("child")
        root = MagicMock()
        root.children.side_effect = lambda recursive: (calls.append("snapshot"), [child])[1]

        def killpg(_pgid, _sig):
            calls.append("killpg")

        with (
            patch("core.platform.process.os.name", "posix"),
            patch("core.platform.process.os.getpgid", return_value=123),
            patch("core.platform.process.os.killpg", side_effect=killpg),
            patch("core.platform.process.psutil.Process", return_value=root),
        ):
            process.terminate_pid(123, include_children=True)

        assert calls == ["snapshot", "killpg", "child"]


class TestProcessTreeHelpers:
    def test_signal_tree_kills_group_and_snapshotted_descendants(self):
        child = MagicMock()
        with (
            patch("core.platform.process.os.name", "posix"),
            patch("core.platform.process.os.killpg") as killpg,
        ):
            process.signal_tree(123, signal.SIGKILL, pgid=456, descendants=[child])

        killpg.assert_called_once_with(456, signal.SIGKILL)
        child.send_signal.assert_called_once_with(signal.SIGKILL)

    @pytest.mark.parametrize(
        ("sig", "root_method", "child_method"),
        [(signal.SIGTERM, "terminate", "terminate"), (signal.SIGKILL, "kill", "kill")],
    )
    def test_signal_tree_maps_windows_signals_without_killpg(self, sig, root_method, child_method):
        root = MagicMock()
        child = MagicMock()
        with (
            patch("core.platform.process.os.name", "nt"),
            patch("core.platform.process.os.killpg") as killpg,
            patch("core.platform.process.psutil.Process", return_value=root),
        ):
            process.signal_tree(123, sig, pgid=456, descendants=[child])

        killpg.assert_not_called()
        getattr(root, root_method).assert_called_once_with()
        getattr(child, child_method).assert_called_once_with()

    def test_kill_tree_orders_deepest_first_and_ignores_psutil_errors(self):
        killed = []
        deep = MagicMock()
        deep.parents.return_value = [MagicMock(), MagicMock()]
        deep.kill.side_effect = lambda: killed.append("deep")
        shallow = MagicMock()
        shallow.parents.return_value = [MagicMock()]
        shallow.kill.side_effect = lambda: killed.append("shallow")
        missing = MagicMock()
        missing.parents.return_value = []
        missing.kill.side_effect = psutil.NoSuchProcess(pid=1)
        denied = MagicMock()
        denied.parents.side_effect = psutil.AccessDenied(pid=2)
        denied.kill.side_effect = psutil.AccessDenied(pid=2)
        root = MagicMock()
        root.kill.side_effect = lambda: killed.append("root")

        with patch("core.platform.process.psutil.Process", return_value=root):
            count = process.kill_tree(123, descendants=[shallow, missing, denied, deep])

        assert killed == ["deep", "shallow", "root"]
        assert count == 3

    def test_process_group_exists_checks_posix_group_and_missing_group(self):
        with (
            patch("core.platform.process.os.name", "posix"),
            patch("core.platform.process.os.killpg") as killpg,
        ):
            assert process.process_group_exists(456, fallback_alive=False)
            killpg.side_effect = ProcessLookupError
            assert not process.process_group_exists(456, fallback_alive=True)

        assert killpg.call_count == 2
        killpg.assert_called_with(456, 0)

    def test_process_group_exists_uses_fallback_without_posix_group(self):
        with (
            patch("core.platform.process.os.name", "nt"),
            patch("core.platform.process.os.killpg") as killpg,
        ):
            assert process.process_group_exists(456, fallback_alive=True)
            assert not process.process_group_exists(None, fallback_alive=False)

        killpg.assert_not_called()

    def test_terminate_tree_terms_then_kills_only_wait_procs_survivors(self):
        exited = MagicMock()
        survivor = MagicMock()
        with patch("core.platform.process.psutil.wait_procs", return_value=([exited], [survivor])) as wait_procs:
            alive = process.terminate_tree(123, descendants=[exited, survivor], grace_sec=2.0, include_root=False)

        exited.terminate.assert_called_once_with()
        survivor.terminate.assert_called_once_with()
        wait_procs.assert_called_once_with([exited, survivor], timeout=2.0)
        exited.kill.assert_not_called()
        survivor.kill.assert_called_once_with()
        assert alive == [survivor]

    @pytest.mark.parametrize(
        "module_name",
        [
            "core.runtime.task_runner",
            "core.supervisor.task_runner",  # legacy name until 2026-11 (S3a)
        ],
    )
    def test_task_runner_subtree_pids_includes_job_and_marker_subtrees(self, module_name: str):
        job_descendant = MagicMock(pid=11)
        marker_descendant = MagicMock(pid=21)
        job_root = MagicMock()
        job_root.children.return_value = [job_descendant]
        marker_root = MagicMock(pid=20)
        marker_root.cmdline.return_value = ["python", "-m", module_name]
        marker_root.children.return_value = [marker_descendant]
        unrelated = MagicMock(pid=30)
        unrelated.cmdline.return_value = ["python", "-m", "other"]
        root = MagicMock()
        root.children.return_value = [marker_root, unrelated]

        def process_for(pid):
            return {10: job_root, 20: marker_root}[pid]

        with patch("core.platform.process.psutil.Process", side_effect=process_for):
            pids = process.task_runner_subtree_pids(root, {10})

        assert pids == {10, 11, 20, 21}


class TestFindMatchingPids:
    def test_filters_by_marker_user_and_python(self):
        current_proc = MagicMock()
        current_proc.username.return_value = "me"

        matching = MagicMock()
        matching.info = {
            "pid": 101,
            "cmdline": ["python", "-m", "cli", "start"],
            "exe": r"C:\Python312\python.exe",
            "name": "python.exe",
            "username": "me",
        }
        wrong_user = MagicMock()
        wrong_user.info = {
            "pid": 202,
            "cmdline": ["python", "-m", "cli", "start"],
            "exe": r"C:\Python312\python.exe",
            "name": "python.exe",
            "username": "other",
        }
        wrong_exe = MagicMock()
        wrong_exe.info = {
            "pid": 303,
            "cmdline": ["node", "main.py", "start"],
            "exe": r"C:\node.exe",
            "name": "node.exe",
            "username": "me",
        }

        with (
            patch("core.platform.process.psutil.Process", return_value=current_proc),
            patch("core.platform.process.psutil.process_iter", return_value=[matching, wrong_user, wrong_exe]),
        ):
            matches = process.find_matching_pids(("main.py start", "cli start"))

        assert matches == [101]

    def test_terminate_matching_processes_returns_count(self):
        with (
            patch("core.platform.process.find_matching_pids", return_value=[1, 2, 3]),
            patch("core.platform.process.terminate_pid") as mock_terminate,
        ):
            count = process.terminate_matching_processes(("runner",), force=True)

        assert count == 3
        assert mock_terminate.call_count == 3


class TestIsUnrelatedProcess:
    """A PID file can name a live process that reused the server's old PID."""

    @staticmethod
    def _patch(target: MagicMock, *, me: str = "alice", root: bool = False):
        current = MagicMock()
        current.username.return_value = me

        def _process(*args):
            return target if args else current

        return (
            patch("core.platform.process.psutil.Process", side_effect=_process),
            patch("core.platform.process._running_as_root", return_value=root),
        )

    @staticmethod
    def _proc(*, user: str = "alice", exe: str | Exception = "/usr/bin/python3.12", name: str = "python3.12"):
        proc = MagicMock()
        proc.username.return_value = user
        if isinstance(exe, Exception):
            proc.exe.side_effect = exe
        else:
            proc.exe.return_value = exe
        proc.name.return_value = name
        return proc

    def _check(self, proc: MagicMock, **kwargs) -> bool:
        process_patch, root_patch = self._patch(proc, **kwargs)
        with process_patch, root_patch:
            return process.is_unrelated_process(599)

    def test_other_users_process_is_unrelated(self):
        """The macOS case: a root daemon took the PID an enclave server had before the reboot."""
        assert self._check(self._proc(user="root", exe=psutil.AccessDenied(599), name="AppleCredentialM")) is True

    def test_same_user_non_python_process_is_unrelated(self):
        assert self._check(self._proc(exe="/Applications/Slack.app/Contents/MacOS/Slack", name="Slack")) is True

    @pytest.mark.parametrize(
        "exe",
        [
            "/usr/bin/python3.12",
            "/opt/homebrew/Frameworks/Python.framework/Versions/3.12/Resources/Python.app/Contents/MacOS/Python",
            r"C:\Python312\python.exe",
        ],
    )
    def test_same_user_python_process_may_be_the_server(self, exe):
        assert self._check(self._proc(exe=exe)) is False

    def test_falls_back_to_process_name_when_exe_is_unreadable(self):
        assert self._check(self._proc(exe=psutil.AccessDenied(599), name="python3.12")) is False
        assert self._check(self._proc(exe=psutil.AccessDenied(599), name="Slack")) is True

    def test_unreadable_owner_is_treated_as_possible_server(self):
        proc = self._proc()
        proc.username.side_effect = psutil.AccessDenied(599)
        assert self._check(proc) is False

    def test_missing_process_is_not_reported_as_unrelated(self):
        with patch("core.platform.process.psutil.Process", side_effect=psutil.NoSuchProcess(599)):
            assert process.is_unrelated_process(599) is False

    def test_root_caller_judges_by_executable_only(self):
        """``sudo animaworks stop`` must still find a server owned by another user."""
        assert self._check(self._proc(user="alice"), me="root", root=True) is False
        assert self._check(self._proc(user="root", exe="/usr/sbin/cron", name="cron"), me="root", root=True) is True

    def test_current_test_process_is_not_unrelated(self):
        import os

        assert process.is_unrelated_process(os.getpid()) is False
