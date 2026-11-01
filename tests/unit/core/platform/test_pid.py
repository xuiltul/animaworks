from __future__ import annotations

from pathlib import Path

from core.platform.pid import read_server_pid


def test_read_server_pid_valid(tmp_path: Path) -> None:
    (tmp_path / "server.pid").write_text("12345\n", encoding="utf-8")

    assert read_server_pid(tmp_path) == 12345


def test_read_server_pid_missing(tmp_path: Path) -> None:
    assert read_server_pid(tmp_path) is None


def test_read_server_pid_invalid(tmp_path: Path) -> None:
    (tmp_path / "server.pid").write_text("not_a_number", encoding="utf-8")

    assert read_server_pid(tmp_path) is None


def test_is_server_running_ignores_pid_reused_by_unrelated_process(tmp_path: Path, monkeypatch) -> None:
    from core.platform import pid as pid_module

    (tmp_path / "server.pid").write_text("599\n", encoding="utf-8")
    monkeypatch.setattr("core.platform.process.is_process_alive", lambda pid: pid == 599)
    monkeypatch.setattr("core.platform.process.is_unrelated_process", lambda pid: pid == 599)

    assert pid_module.is_server_running(tmp_path) is False


def test_is_server_running_accepts_live_python_process(tmp_path: Path) -> None:
    import os

    from core.platform.pid import is_server_running

    (tmp_path / "server.pid").write_text(f"{os.getpid()}\n", encoding="utf-8")

    assert is_server_running(tmp_path) is True
