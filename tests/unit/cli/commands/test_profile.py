# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── Helpers ──────────────────────────────────────────────────


@pytest.fixture()
def profiles_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Redirect _PROFILES_FILE to a temp location."""
    pf = tmp_path / "profiles.json"
    monkeypatch.setattr("cli.commands.profile._PROFILES_FILE", pf)
    return pf


def _write_profiles(pf: Path, profiles: dict) -> None:
    pf.write_text(json.dumps({"version": 1, "profiles": profiles}, indent=2), encoding="utf-8")


def _read_profiles(pf: Path) -> dict:
    return json.loads(pf.read_text(encoding="utf-8"))["profiles"]


# ── Storage helpers ──────────────────────────────────────────


class TestLoadProfiles:
    def test_missing_file(self, profiles_file: Path) -> None:
        from cli.commands.profile import _load_profiles

        assert _load_profiles() == {}

    def test_valid_file(self, profiles_file: Path) -> None:
        from cli.commands.profile import _load_profiles

        _write_profiles(profiles_file, {"test": {"data_dir": "/tmp/d", "port": 18500}})
        result = _load_profiles()
        assert "test" in result
        assert result["test"]["port"] == 18500

    def test_corrupt_json(self, profiles_file: Path, capsys: pytest.CaptureFixture) -> None:
        from cli.commands.profile import _load_profiles

        profiles_file.write_text("{invalid", encoding="utf-8")
        result = _load_profiles()
        assert result == {}
        captured = capsys.readouterr()
        assert captured.err != ""


class TestSaveProfiles:
    def test_atomic_write(self, profiles_file: Path) -> None:
        from cli.commands.profile import _save_profiles

        _save_profiles({"p1": {"data_dir": "/d", "port": 18500}})
        data = json.loads(profiles_file.read_text(encoding="utf-8"))
        assert data["version"] == 1
        assert "p1" in data["profiles"]
        assert not profiles_file.with_suffix(".json.tmp").exists()


class TestNextAvailablePort:
    def test_empty(self) -> None:
        from cli.commands.profile import _next_available_port

        assert _next_available_port({}) == 18500

    def test_first_taken(self) -> None:
        from cli.commands.profile import _next_available_port

        profiles = {"a": {"port": 18500}}
        assert _next_available_port(profiles) == 18510

    def test_gap_reuse(self) -> None:
        from cli.commands.profile import _next_available_port

        profiles = {"a": {"port": 18500}, "b": {"port": 18520}}
        assert _next_available_port(profiles) == 18510


# ── PID/status helpers ───────────────────────────────────────


class TestReadPidFor:
    def test_no_pid_file(self, tmp_path: Path) -> None:
        from cli.commands.profile import _read_pid_for

        assert _read_pid_for(tmp_path) is None

    def test_valid_pid(self, tmp_path: Path) -> None:
        from cli.commands.profile import _read_pid_for

        (tmp_path / "server.pid").write_text("12345", encoding="utf-8")
        assert _read_pid_for(tmp_path) == 12345

    def test_invalid_pid(self, tmp_path: Path) -> None:
        from cli.commands.profile import _read_pid_for

        (tmp_path / "server.pid").write_text("notanumber", encoding="utf-8")
        assert _read_pid_for(tmp_path) is None


class TestIsProcessAlive:
    def test_current_process(self) -> None:
        from cli.commands.profile import _is_process_alive

        assert _is_process_alive(os.getpid()) is True

    def test_nonexistent_pid(self) -> None:
        from cli.commands.profile import _is_process_alive

        assert _is_process_alive(99999999) is False


class TestProfileStatus:
    def test_stopped(self, tmp_path: Path) -> None:
        from cli.commands.profile import _profile_status

        result = _profile_status({"data_dir": str(tmp_path)})
        assert "stopped" in result.lower() or "停止" in result

    def test_running(self, tmp_path: Path) -> None:
        from cli.commands.profile import _profile_status

        (tmp_path / "server.pid").write_text(str(os.getpid()), encoding="utf-8")
        result = _profile_status({"data_dir": str(tmp_path)})
        assert str(os.getpid()) in result

    def test_stale_pid(self, tmp_path: Path) -> None:
        from cli.commands.profile import _profile_status

        (tmp_path / "server.pid").write_text("99999999", encoding="utf-8")
        result = _profile_status({"data_dir": str(tmp_path)})
        assert "stale" in result.lower() or "古い" in result


# ── Commands ─────────────────────────────────────────────────


class TestCmdProfileList:
    def test_empty(self, profiles_file: Path, capsys: pytest.CaptureFixture) -> None:
        from cli.commands.profile import cmd_profile_list

        cmd_profile_list(argparse.Namespace())
        out = capsys.readouterr().out
        assert "animaworks profile add" in out.lower() or "登録されていません" in out

    def test_with_profiles(self, profiles_file: Path, capsys: pytest.CaptureFixture, tmp_path: Path) -> None:
        from cli.commands.profile import cmd_profile_list

        _write_profiles(profiles_file, {"demo": {"data_dir": str(tmp_path), "port": 18500}})
        cmd_profile_list(argparse.Namespace())
        out = capsys.readouterr().out
        assert "demo" in out
        assert "18500" in out


class TestCmdProfileAdd:
    def test_add_default(self, profiles_file: Path, capsys: pytest.CaptureFixture) -> None:
        from cli.commands.profile import cmd_profile_add

        args = argparse.Namespace(name="proj", data_dir=None, port=None)
        cmd_profile_add(args)
        profiles = _read_profiles(profiles_file)
        assert "proj" in profiles
        assert profiles["proj"]["port"] == 18500

    def test_add_explicit(self, profiles_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        from cli.commands.profile import cmd_profile_add

        dd = str(tmp_path / "mydata")
        args = argparse.Namespace(name="custom", data_dir=dd, port=19000)
        cmd_profile_add(args)
        profiles = _read_profiles(profiles_file)
        assert profiles["custom"]["port"] == 19000

    def test_add_duplicate(self, profiles_file: Path) -> None:
        from cli.commands.profile import cmd_profile_add

        _write_profiles(profiles_file, {"existing": {"data_dir": "/d", "port": 18500}})
        args = argparse.Namespace(name="existing", data_dir=None, port=None)
        with pytest.raises(SystemExit, match="1"):
            cmd_profile_add(args)

    def test_add_shows_init_hint_when_dir_missing(self, profiles_file: Path, capsys: pytest.CaptureFixture) -> None:
        from cli.commands.profile import cmd_profile_add

        args = argparse.Namespace(name="new", data_dir="/nonexistent/path", port=None)
        cmd_profile_add(args)
        out = capsys.readouterr().out
        assert "init" in out.lower() or "初期化" in out


class TestCmdProfileRemove:
    def test_remove(self, profiles_file: Path, capsys: pytest.CaptureFixture, tmp_path: Path) -> None:
        from cli.commands.profile import cmd_profile_remove

        _write_profiles(profiles_file, {"rm_me": {"data_dir": str(tmp_path), "port": 18500}})
        args = argparse.Namespace(name="rm_me")
        cmd_profile_remove(args)
        profiles = _read_profiles(profiles_file)
        assert "rm_me" not in profiles
        out = capsys.readouterr().out
        assert str(tmp_path) in out

    def test_remove_not_found(self, profiles_file: Path) -> None:
        from cli.commands.profile import cmd_profile_remove

        args = argparse.Namespace(name="ghost")
        with pytest.raises(SystemExit, match="1"):
            cmd_profile_remove(args)

    def test_remove_running_rejected(self, profiles_file: Path, tmp_path: Path) -> None:
        from cli.commands.profile import cmd_profile_remove

        (tmp_path / "server.pid").write_text(str(os.getpid()), encoding="utf-8")
        _write_profiles(profiles_file, {"active": {"data_dir": str(tmp_path), "port": 18500}})
        args = argparse.Namespace(name="active")
        with pytest.raises(SystemExit, match="1"):
            cmd_profile_remove(args)


class TestCmdProfileStart:
    @patch("cli.commands.server.cmd_start")
    def test_start(
        self, mock_start: MagicMock, profiles_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from cli.commands.profile import cmd_profile_start

        _write_profiles(profiles_file, {"s": {"data_dir": str(tmp_path), "port": 18510}})
        args = argparse.Namespace(name="s", host=None)
        old_env = os.environ.get("ANIMAWORKS_DATA_DIR")
        cmd_profile_start(args)
        mock_start.assert_called_once()
        call_args = mock_start.call_args[0][0]
        assert call_args.port == 18510
        assert call_args.foreground is False
        if old_env is not None:
            os.environ["ANIMAWORKS_DATA_DIR"] = old_env
        else:
            os.environ.pop("ANIMAWORKS_DATA_DIR", None)

    def test_start_not_found(self, profiles_file: Path) -> None:
        from cli.commands.profile import cmd_profile_start

        args = argparse.Namespace(name="missing", host=None)
        with pytest.raises(SystemExit, match="1"):
            cmd_profile_start(args)

    def test_start_already_running(self, profiles_file: Path, tmp_path: Path) -> None:
        from cli.commands.profile import cmd_profile_start

        (tmp_path / "server.pid").write_text(str(os.getpid()), encoding="utf-8")
        _write_profiles(profiles_file, {"run": {"data_dir": str(tmp_path), "port": 18500}})
        args = argparse.Namespace(name="run", host=None)
        with pytest.raises(SystemExit, match="1"):
            cmd_profile_start(args)


class TestCmdProfileStop:
    @patch("cli.commands.server.cmd_stop")
    def test_stop(self, mock_stop: MagicMock, profiles_file: Path, tmp_path: Path) -> None:
        from cli.commands.profile import cmd_profile_stop

        (tmp_path / "server.pid").write_text(str(os.getpid()), encoding="utf-8")
        _write_profiles(profiles_file, {"st": {"data_dir": str(tmp_path), "port": 18500}})
        args = argparse.Namespace(name="st", force=False)
        old_env = os.environ.get("ANIMAWORKS_DATA_DIR")
        cmd_profile_stop(args)
        mock_stop.assert_called_once()
        if old_env is not None:
            os.environ["ANIMAWORKS_DATA_DIR"] = old_env
        else:
            os.environ.pop("ANIMAWORKS_DATA_DIR", None)

    def test_stop_not_running(self, profiles_file: Path, tmp_path: Path) -> None:
        from cli.commands.profile import cmd_profile_stop

        _write_profiles(profiles_file, {"idle": {"data_dir": str(tmp_path), "port": 18500}})
        args = argparse.Namespace(name="idle", force=False)
        with pytest.raises(SystemExit, match="1"):
            cmd_profile_stop(args)


class TestCmdProfileStartAll:
    @patch("cli.commands.server.cmd_start")
    def test_start_all(self, mock_start: MagicMock, profiles_file: Path, tmp_path: Path) -> None:
        from cli.commands.profile import cmd_profile_start_all

        d1 = tmp_path / "d1"
        d2 = tmp_path / "d2"
        d1.mkdir()
        d2.mkdir()
        _write_profiles(
            profiles_file,
            {
                "a": {"data_dir": str(d1), "port": 18500},
                "b": {"data_dir": str(d2), "port": 18510},
            },
        )
        args = argparse.Namespace(host=None)
        old_env = os.environ.get("ANIMAWORKS_DATA_DIR")
        cmd_profile_start_all(args)
        assert mock_start.call_count == 2
        if old_env is not None:
            os.environ["ANIMAWORKS_DATA_DIR"] = old_env
        else:
            os.environ.pop("ANIMAWORKS_DATA_DIR", None)


class TestCmdProfileStopAll:
    @patch("cli.commands.server.cmd_stop")
    def test_stop_all(self, mock_stop: MagicMock, profiles_file: Path, tmp_path: Path) -> None:
        from cli.commands.profile import cmd_profile_stop_all

        d1 = tmp_path / "d1"
        d1.mkdir()
        (d1 / "server.pid").write_text(str(os.getpid()), encoding="utf-8")
        d2 = tmp_path / "d2"
        d2.mkdir()
        _write_profiles(
            profiles_file,
            {
                "a": {"data_dir": str(d1), "port": 18500},
                "b": {"data_dir": str(d2), "port": 18510},
            },
        )
        args = argparse.Namespace(force=False)
        old_env = os.environ.get("ANIMAWORKS_DATA_DIR")
        cmd_profile_stop_all(args)
        assert mock_stop.call_count == 1
        if old_env is not None:
            os.environ["ANIMAWORKS_DATA_DIR"] = old_env
        else:
            os.environ.pop("ANIMAWORKS_DATA_DIR", None)


# ── Parser registration ──────────────────────────────────────


class TestRegisterProfileCommand:
    def test_registration(self) -> None:
        from cli.commands.profile import register_profile_command

        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers()
        register_profile_command(sub)
        args = parser.parse_args(["profile", "list"])
        assert hasattr(args, "func")

    def test_add_args(self) -> None:
        from cli.commands.profile import register_profile_command

        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers()
        register_profile_command(sub)
        args = parser.parse_args(["profile", "add", "myproj", "--data-dir", "/tmp/d", "--port", "19000"])
        assert args.name == "myproj"
        assert args.data_dir == "/tmp/d"
        assert args.port == 19000

    def test_start_args(self) -> None:
        from cli.commands.profile import register_profile_command

        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers()
        register_profile_command(sub)
        args = parser.parse_args(["profile", "start", "p1", "--host", "127.0.0.1"])
        assert args.name == "p1"
        assert args.host == "127.0.0.1"

    def test_stop_force(self) -> None:
        from cli.commands.profile import register_profile_command

        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers()
        register_profile_command(sub)
        args = parser.parse_args(["profile", "stop", "p1", "--force"])
        assert args.force is True
