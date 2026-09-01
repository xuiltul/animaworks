"""Unit tests for the dead command reaper (orphan sandboxes + broad searches).

psutil is mocked throughout; no real process is touched.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import core.supervisor.dead_command_reaper as reaper

NOW = 1_000_000_000.0
DATA_DIR = Path("/data")


class FakeProc:
    """Duck-typed stand-in for a psutil.Process."""

    def __init__(self, *, pid: int, cmdline: list[str], create_time: float, parent=None, cwd="/"):
        self.pid = pid
        self._cmdline = cmdline
        self._create_time = create_time
        self._parent = parent
        self._cwd = cwd

    def cmdline(self):
        return self._cmdline

    def create_time(self):
        return self._create_time

    def parent(self):
        return self._parent

    def name(self):
        return self._cmdline[0].rsplit("/", 1)[-1] if self._cmdline else ""

    def cwd(self):
        return self._cwd

    def children(self, recursive=True):
        return []

    def __repr__(self):  # pragma: no cover
        return f"FakeProc(pid={self.pid}, cmd={self._cmdline})"


def _systemd() -> FakeProc:
    return FakeProc(pid=1, cmdline=["/lib/systemd/systemd", "--user"], create_time=0.0)


def _codex() -> FakeProc:
    return FakeProc(pid=500, cmdline=["bash", "-c", "codex exec ..."], create_time=0.0)


@pytest.fixture
def fake_now(monkeypatch: pytest.MonkeyPatch) -> float:
    monkeypatch.setattr(reaper.time, "time", lambda: NOW)
    return NOW


@pytest.fixture
def fake_data_dir(monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(DATA_DIR))
    return DATA_DIR


def _run_collect(procs: list[FakeProc], *, fn) -> list:
    with patch.object(reaper.psutil, "process_iter", return_value=procs):
        return fn()


class TestOrphanSandbox:
    def test_sandbox_reparented_to_systemd_is_targeted(self, fake_now, fake_data_dir) -> None:
        sandbox = FakeProc(
            pid=100,
            cmdline=["codex-linux-sandbox", "--", "/bin/bash"],
            create_time=NOW - 600,
            parent=_systemd(),
        )
        targets = _run_collect([sandbox], fn=reaper.collect_orphan_sandboxes)
        assert len(targets) == 1
        assert targets[0]["pid"] == 100
        assert targets[0]["kind"] == "orphan-sandbox"
        assert targets[0]["age"] == 600

    def test_sandbox_with_no_parent_is_targeted(self, fake_now, fake_data_dir) -> None:
        sandbox = FakeProc(
            pid=101,
            cmdline=["codex-linux-sandbox", "run"],
            create_time=NOW - 400,
            parent=None,
        )
        targets = _run_collect([sandbox], fn=reaper.collect_orphan_sandboxes)
        assert len(targets) == 1
        assert targets[0]["pid"] == 101

    def test_sandbox_still_owned_by_codex_is_not_targeted(self, fake_now, fake_data_dir) -> None:
        sandbox = FakeProc(
            pid=102,
            cmdline=["codex-linux-sandbox", "--", "/bin/sh"],
            create_time=NOW - 600,
            parent=_codex(),
        )
        targets = _run_collect([sandbox], fn=reaper.collect_orphan_sandboxes)
        assert targets == []

    def test_young_sandbox_is_not_targeted(self, fake_now, fake_data_dir) -> None:
        sandbox = FakeProc(
            pid=103,
            cmdline=["codex-linux-sandbox", "--", "/bin/sh"],
            create_time=NOW - 60,  # below 5 min
            parent=_systemd(),
        )
        targets = _run_collect([sandbox], fn=reaper.collect_orphan_sandboxes)
        assert targets == []

    def test_non_sandbox_is_ignored(self, fake_now, fake_data_dir) -> None:
        proc = FakeProc(pid=104, cmdline=["python", "-m", "http.server"], create_time=NOW - 600, parent=None)
        targets = _run_collect([proc], fn=reaper.collect_orphan_sandboxes)
        assert targets == []

    def test_sessions_are_deduplicated(self, fake_now, fake_data_dir) -> None:
        sandbox_a = FakeProc(
            pid=110,
            cmdline=["codex-linux-sandbox", "--", "/bin/sh"],
            create_time=NOW - 600,
            parent=_systemd(),
        )
        sandbox_b = FakeProc(
            pid=111,
            cmdline=["codex-linux-sandbox", "--", "/bin/sh", "-c", "sleep"],
            create_time=NOW - 700,
            parent=None,
        )
        with patch.object(reaper.os, "getsid", side_effect=lambda pid: 99):
            targets = _run_collect([sandbox_a, sandbox_b], fn=reaper.collect_orphan_sandboxes)
        assert len(targets) == 1


class TestBroadSearch:
    def test_recursive_grep_over_data_root_is_targeted(self, fake_now, fake_data_dir) -> None:
        proc = FakeProc(
            pid=200,
            cmdline=["grep", "-R", "needle", str(DATA_DIR / "animas")],
            create_time=NOW - 900,  # above 10 min
            cwd="/somewhere",
        )
        targets = _run_collect([proc], fn=reaper.collect_broad_searches)
        assert len(targets) == 1
        assert targets[0]["pid"] == 200
        assert targets[0]["kind"] == "broad-search"

    def test_young_broad_search_is_not_targeted(self, fake_now, fake_data_dir) -> None:
        proc = FakeProc(
            pid=201,
            cmdline=["rg", str(DATA_DIR / "state")],
            create_time=NOW - 300,  # below 10 min
            cwd="/",
        )
        targets = _run_collect([proc], fn=reaper.collect_broad_searches)
        assert targets == []

    def test_narrow_search_is_not_targeted(self, fake_now, fake_data_dir) -> None:
        proc = FakeProc(
            pid=202,
            cmdline=["grep", "-R", "needle", str(DATA_DIR / "not-a-broad-root")],
            create_time=NOW - 900,
            cwd="/",
        )
        targets = _run_collect([proc], fn=reaper.collect_broad_searches)
        assert targets == []

    def test_relative_path_resolved_against_cwd(self, fake_now, fake_data_dir) -> None:
        proc = FakeProc(
            pid=203,
            cmdline=["grep", "-R", "needle", "animas"],
            create_time=NOW - 900,
            cwd=str(DATA_DIR),
        )
        targets = _run_collect([proc], fn=reaper.collect_broad_searches)
        assert len(targets) == 1
        assert targets[0]["pid"] == 203

    def test_unreadable_cwd_defers_to_miss(self, fake_now, fake_data_dir) -> None:
        class NoCwdProcess(FakeProc):
            def cwd(self):
                raise LookupError("no cwd")

        proc = NoCwdProcess(
            pid=204,
            cmdline=["grep", "-R", "needle", "animas"],
            create_time=NOW - 900,
            cwd="/",
        )
        targets = _run_collect([proc], fn=reaper.collect_broad_searches)
        assert targets == []

    def test_non_recursive_grep_is_not_targeted(self, fake_now, fake_data_dir) -> None:
        proc = FakeProc(
            pid=205,
            cmdline=["grep", "needle", str(DATA_DIR / "animas")],
            create_time=NOW - 900,
            cwd="/",
        )
        targets = _run_collect([proc], fn=reaper.collect_broad_searches)
        assert targets == []


class TestKillExecution:
    def test_orphan_sandbox_kills_session_and_descendants(self, fake_now, fake_data_dir) -> None:
        child = FakeProc(pid=302, cmdline=[], create_time=0.0)
        child_proc = MagicMock()
        child_proc.pid = 302
        target = {"pid": 301, "age": 100, "kind": "orphan-sandbox", "cmd": "codex-linux-sandbox"}
        with (
            patch.object(reaper.os, "getsid", return_value=999),
            patch.object(reaper.os, "killpg") as killpg,
            patch.object(reaper.psutil, "Process") as ps_proc,
            patch.object(reaper.os, "kill") as kill,
        ):
            ps_proc.return_value.children.return_value = [child_proc]
            result = reaper.kill_target(target)
        assert result is True
        killpg.assert_called_once_with(999, 9)
        called = sorted(int(c.args[0]) for c in kill.call_args_list)
        assert called == [301, 302]

    def test_broad_search_skips_session_kill(self, fake_now, fake_data_dir) -> None:
        """broad-search must NOT killpg (caller shares the session); pid+descendants only."""
        child = FakeProc(pid=312, cmdline=[], create_time=0.0)
        child_proc = MagicMock()
        child_proc.pid = 312
        target = {"pid": 311, "age": 100, "kind": "broad-search", "cmd": "grep -R foo /data/animas"}
        with (
            patch.object(reaper.os, "getsid", return_value=999),
            patch.object(reaper.os, "killpg") as killpg,
            patch.object(reaper.psutil, "Process") as ps_proc,
            patch.object(reaper.os, "kill") as kill,
        ):
            ps_proc.return_value.children.return_value = [child_proc]
            result = reaper.kill_target(target)
        assert result is True
        killpg.assert_not_called()
        called = sorted(int(c.args[0]) for c in kill.call_args_list)
        assert called == [311, 312]

    def test_kill_targets_logs_fixed_line_and_counts(self, fake_now, fake_data_dir) -> None:
        target = {"pid": 401, "age": 55, "kind": "broad-search", "cmd": "grep -R foo /data/animas"}
        logs: list[str] = []
        with (
            patch.object(reaper.os, "getsid", return_value=401),
            patch.object(reaper.os, "killpg"),
            patch.object(reaper.psutil, "Process") as ps_proc,
            patch.object(reaper.os, "kill"),
        ):
            ps_proc.return_value.children.return_value = []
            count = reaper.kill_targets([target, {"pid": 402, "age": 1, "kind": "broad-search", "cmd": "x"}], log=logs.append)
        assert count == 2
        assert len(logs) == 2
        assert "reaped dead command: pid=401 age=55 kind=broad-search cmd=grep -R foo /data/animas" in logs[0]

    def test_kill_target_missing_process_is_harmless(self, fake_now, fake_data_dir) -> None:
        target = {"pid": 501, "age": 10, "kind": "orphan-sandbox", "cmd": "codex-linux-sandbox"}
        with (
            patch.object(reaper.os, "getsid", side_effect=ProcessLookupError),
            patch.object(reaper.psutil, "Process") as ps_proc,
            patch.object(reaper.os, "kill", side_effect=ProcessLookupError),
        ):
            ps_proc.return_value.children.side_effect = ProcessLookupError
            assert reaper.kill_target(target) is False


def test_collect_dead_commands_combines_both_kinds(fake_now, fake_data_dir) -> None:
    sandbox = FakeProc(
        pid=600,
        cmdline=["codex-linux-sandbox", "run"],
        create_time=NOW - 600,
        parent=None,
    )
    search = FakeProc(
        pid=601,
        cmdline=["grep", "-R", "x", str(DATA_DIR / "animas")],
        create_time=NOW - 900,
        cwd="/",
    )
    with patch.object(reaper.psutil, "process_iter", return_value=[sandbox, search]):
        targets = reaper.collect_dead_commands()
    kinds = {t["kind"] for t in targets}
    assert kinds == {"orphan-sandbox", "broad-search"}
