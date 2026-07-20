"""Tests for splitting the legacy board into company-scoped channels."""

from __future__ import annotations

import json
from pathlib import Path

from core.messenger import is_channel_member
from core.migrations.registry import MigrationRunner
from core.migrations.steps import register_all_steps, step_split_board_by_company


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _create_anima(data_dir: Path, name: str, company: str | None) -> None:
    status = {"enabled": True}
    if company is not None:
        status["company"] = company
    _write_json(data_dir / "animas" / name / "status.json", status)


def _create_legacy_board(data_dir: Path) -> tuple[Path, Path, str]:
    channels_dir = data_dir / "shared" / "channels"
    channels_dir.mkdir(parents=True)
    board_path = channels_dir / "board.jsonl"
    history = '{"from":"quartz","content":"retained"}\n'
    board_path.write_text(history, encoding="utf-8")
    meta_path = channels_dir / "board.meta.json"
    _write_json(
        meta_path,
        {
            "members": ["quartz", "ember", "willow"],
            "created_by": "quartz",
            "created_at": "2026-07-01T00:00:00+00:00",
            "description": "Legacy board",
            "custom_field": "preserved",
        },
    )
    return board_path, meta_path, history


def test_split_board_groups_members_from_runtime_status_and_closes_legacy(tmp_path: Path) -> None:
    board_path, meta_path, history = _create_legacy_board(tmp_path)
    _create_anima(tmp_path, "quartz", "north")
    _create_anima(tmp_path, "ember", "south")
    _create_anima(tmp_path, "willow", "north")

    # A stale aggregate config must not affect the migration grouping.
    _write_json(
        tmp_path / "config.json",
        {"animas": {"quartz": {"company": "south"}, "ember": {"company": "north"}}},
    )

    result = step_split_board_by_company(tmp_path, dry_run=False, verbose=True)

    assert result.error is None
    assert result.changed == 5
    assert json.loads((meta_path.parent / "board-north.meta.json").read_text())["members"] == [
        "quartz",
        "willow",
    ]
    assert json.loads((meta_path.parent / "board-south.meta.json").read_text())["members"] == ["ember"]
    assert (meta_path.parent / "board-north.jsonl").read_text(encoding="utf-8") == ""
    assert (meta_path.parent / "board-south.jsonl").read_text(encoding="utf-8") == ""

    legacy_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert legacy_meta["members"] == []
    assert legacy_meta["closed"] is True
    assert legacy_meta["custom_field"] == "preserved"
    assert board_path.read_text(encoding="utf-8") == history
    assert not is_channel_member(tmp_path / "shared", "board", "quartz")


def test_split_board_is_idempotent_and_preserves_successor_history(tmp_path: Path) -> None:
    _create_legacy_board(tmp_path)
    _create_anima(tmp_path, "quartz", "north")
    _create_anima(tmp_path, "ember", "south")
    _create_anima(tmp_path, "willow", "north")

    first = step_split_board_by_company(tmp_path, dry_run=False, verbose=False)
    north_history = tmp_path / "shared" / "channels" / "board-north.jsonl"
    north_history.write_text('{"content":"new history"}\n', encoding="utf-8")
    before = {path.name: path.read_bytes() for path in (tmp_path / "shared" / "channels").iterdir()}

    second = step_split_board_by_company(tmp_path, dry_run=False, verbose=False)
    after = {path.name: path.read_bytes() for path in (tmp_path / "shared" / "channels").iterdir()}

    assert first.error is None
    assert second.error is None
    assert second.changed == 0
    assert second.skipped == 1
    assert after == before


def test_split_board_dry_run_reports_without_writing(tmp_path: Path) -> None:
    _create_legacy_board(tmp_path)
    _create_anima(tmp_path, "quartz", "north")
    _create_anima(tmp_path, "ember", "south")
    _create_anima(tmp_path, "willow", "north")
    channels_dir = tmp_path / "shared" / "channels"
    before = {path.name: path.read_bytes() for path in channels_dir.iterdir()}

    result = step_split_board_by_company(tmp_path, dry_run=True, verbose=False)

    assert result.error is None
    assert result.changed == 5
    assert {path.name: path.read_bytes() for path in channels_dir.iterdir()} == before
    assert not (channels_dir / "board-north.jsonl").exists()
    assert not (channels_dir / "board-south.meta.json").exists()


def test_split_board_step_is_registered_before_version(tmp_path: Path) -> None:
    runner = MigrationRunner(tmp_path)
    register_all_steps(runner)
    ids = [step["id"] for step in runner.list_steps()]

    assert ids.index("split_board_by_company_20260720") < ids.index("update_version")
