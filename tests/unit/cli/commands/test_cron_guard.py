"""Tests for cron-guard CLI registration and edge cases."""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import patch

from cli.commands.cron_guard import cmd_cron_guard_enable, register_cron_guard_command


def test_parser_registers_list_and_enable() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    register_cron_guard_command(subparsers)

    listed = parser.parse_args(["cron-guard", "list", "alice"])
    enabled = parser.parse_args(["cron-guard", "enable", "alice", "daily report"])

    assert listed.anima == "alice"
    assert enabled.task == "daily report"


def test_enable_missing_task_keeps_sidecar_unchanged(tmp_path: Path, capsys) -> None:
    anima_dir = tmp_path / "animas" / "alice"
    anima_dir.mkdir(parents=True)
    with patch("core.paths.get_animas_dir", return_value=tmp_path / "animas"):
        cmd_cron_guard_enable(argparse.Namespace(anima="alice", task="missing"))

    assert "is not disabled" in capsys.readouterr().out
    assert not (anima_dir / "state" / "cron_disabled.json").exists()
