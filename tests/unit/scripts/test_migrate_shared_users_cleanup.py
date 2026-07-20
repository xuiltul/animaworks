"""Unit tests for scripts/migrate_shared_users_cleanup.py (temp dirs only)."""

from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

import json
from datetime import date
from pathlib import Path

from scripts.migrate_shared_users_cleanup import execute_cleanup, plan_moves, build_match_names


def _layout(tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    animas = data_dir / "animas"
    users = data_dir / "shared" / "users"
    animas.mkdir(parents=True)
    users.mkdir(parents=True)

    # Active anima
    (animas / "sakura").mkdir()
    (animas / "sakura" / "identity.md").write_text("# sakura\n", encoding="utf-8")
    # Tombstoned anima still on disk
    (animas / "oldbot").mkdir()
    (animas / "oldbot" / "status.json").write_text(
        json.dumps({"enabled": False}),
        encoding="utf-8",
    )

    # Contaminated user dirs (anima names)
    (users / "sakura").mkdir()
    (users / "sakura" / "conversations").mkdir()
    (users / "sakura" / "conversations" / "2026-01-01.jsonl").write_text("{}\n", encoding="utf-8")
    (users / "oldbot").mkdir()
    # Human user — must stay
    (users / "alice").mkdir()
    (users / "alice" / "index.md").write_text("# Alice\n", encoding="utf-8")
    # Retired anima no longer on disk
    (users / "legacy_bot").mkdir()

    return data_dir


def test_dry_run_lists_without_moving(tmp_path: Path) -> None:
    data_dir = _layout(tmp_path)
    result = execute_cleanup(
        data_dir,
        dry_run=True,
        backup_date=date(2026, 7, 20),
    )
    assert result.dry_run is True
    assert {p.name for p in result.planned} == {"sakura", "oldbot"}
    assert result.moved == ()
    assert (data_dir / "shared" / "users" / "sakura").is_dir()
    assert (data_dir / "shared" / "users" / "alice").is_dir()
    assert not (data_dir / "shared" / "users_backup_20260720").exists()


def test_execute_moves_matching_dirs(tmp_path: Path) -> None:
    data_dir = _layout(tmp_path)
    result = execute_cleanup(
        data_dir,
        dry_run=False,
        backup_date=date(2026, 7, 20),
    )
    assert result.dry_run is False
    assert {p.name for p in result.moved} == {"sakura", "oldbot"}
    backup = data_dir / "shared" / "users_backup_20260720"
    assert (backup / "sakura" / "conversations" / "2026-01-01.jsonl").is_file()
    assert (backup / "oldbot").is_dir()
    assert not (data_dir / "shared" / "users" / "sakura").exists()
    assert not (data_dir / "shared" / "users" / "oldbot").exists()
    # Human remains
    assert (data_dir / "shared" / "users" / "alice" / "index.md").is_file()
    # Retired without --extra-names stays
    assert (data_dir / "shared" / "users" / "legacy_bot").is_dir()


def test_extra_names_moves_retired(tmp_path: Path) -> None:
    data_dir = _layout(tmp_path)
    result = execute_cleanup(
        data_dir,
        dry_run=False,
        extra_names=["legacy_bot"],
        backup_date=date(2026, 7, 20),
    )
    assert "legacy_bot" in {p.name for p in result.moved}
    assert not (data_dir / "shared" / "users" / "legacy_bot").exists()
    assert (data_dir / "shared" / "users_backup_20260720" / "legacy_bot").is_dir()
    assert (data_dir / "shared" / "users" / "alice").is_dir()


def test_build_match_names_includes_extra(tmp_path: Path) -> None:
    data_dir = _layout(tmp_path)
    names = build_match_names(data_dir, extra_names=["legacy_bot", "  spaced  "])
    assert "sakura" in names
    assert "oldbot" in names
    assert "legacy_bot" in names
    assert "spaced" in names


def test_plan_moves_empty_when_no_users_dir(tmp_path: Path) -> None:
    data_dir = tmp_path / "empty"
    data_dir.mkdir()
    backup, plans = plan_moves(data_dir, match_names={"sakura"}, backup_date=date(2026, 1, 1))
    assert plans == []
    assert backup.name == "users_backup_20260101"
