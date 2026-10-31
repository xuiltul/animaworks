"""Tests for the animaworks-tool supervisor CLI (ToolHandler-backed)."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from cli.commands.supervisor_cmd import cmd_supervisor, register_supervisor_command


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="animaworks")
    subparsers = parser.add_subparsers(dest="command")
    register_supervisor_command(subparsers)
    return parser


@pytest.fixture
def runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Set up boss+sub minimal runtime and point the supervisor at boss."""
    data_dir = tmp_path / "data"
    animas_dir = data_dir / "animas"

    for name in ("boss", "sub"):
        d = animas_dir / name
        (d / "state").mkdir(parents=True)
        (d / "activity_log").mkdir(exist_ok=True)
        (d / "status.json").write_text('{"enabled": true}', encoding="utf-8")
        (d / "IDENTITY.md").write_text(f"# {name}", encoding="utf-8")

    (animas_dir / "sub" / "state" / "current_state.md").write_text(
        "working on sub task",
        encoding="utf-8",
    )

    (data_dir / "config.json").write_text(
        json.dumps({"animas": {"boss": {}, "sub": {"supervisor": "boss"}}}),
        encoding="utf-8",
    )

    monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(data_dir))
    monkeypatch.setenv("ANIMAWORKS_ANIMA_DIR", str(animas_dir / "boss"))
    return data_dir


def _run(argv: list[str]) -> str:
    from contextlib import redirect_stdout
    from io import StringIO

    args = _parser().parse_args(argv)
    buf = StringIO()
    with redirect_stdout(buf):
        args.func(args)
    return buf.getvalue()


def test_org_dashboard_includes_subordinate(runtime: Path) -> None:
    out = _run(["supervisor", "org-dashboard"])
    assert "sub" in out


def test_read_state_returns_subordinate_state(runtime: Path) -> None:
    out = _run(["supervisor", "read-state", "sub"])
    assert "working on sub task" in out


def test_read_state_non_descendant_returns_tool_handler_error(
    runtime: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = _parser().parse_args(["supervisor", "read-state", "other"])
    with pytest.raises(SystemExit) as exc_info:
        args.func(args)
    assert exc_info.value.code == 1
    out = capsys.readouterr().out
    assert "PermissionDenied" in out or "error" in out


def test_missing_anima_dir_exits_1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANIMAWORKS_ANIMA_DIR", raising=False)
    with pytest.raises(SystemExit) as exc_info:
        cmd_supervisor(SimpleNamespace(supervisor_command="org-dashboard"))
    assert exc_info.value.code == 1
