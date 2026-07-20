from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
import sqlite3
from pathlib import Path
from unittest.mock import patch

from core.migrations.registry import MigrationRunner
from core.migrations.steps import register_all_steps, step_tool_prompts_db_to_md


def _create_db(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE tool_descriptions (name TEXT PRIMARY KEY, description TEXT)")
    connection.execute("CREATE TABLE tool_guides (key TEXT PRIMARY KEY, content TEXT)")
    connection.execute("CREATE TABLE system_sections (key TEXT PRIMARY KEY, content TEXT)")
    connection.execute("INSERT INTO tool_descriptions VALUES (?, ?)", ("Read", "Custom description"))
    connection.execute("INSERT INTO tool_guides VALUES (?, ?)", ("non_s", "Custom guide"))
    connection.execute("INSERT INTO system_sections VALUES (?, ?)", ("behavior_rules", "Custom rules"))
    connection.commit()
    connection.close()


def test_db_with_differences_writes_markdown(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    templates_dir = tmp_path / "templates"
    data_dir.mkdir()
    db_path = data_dir / "tool_prompts.sqlite3"
    _create_db(db_path)
    target = templates_dir / "ja/prompts/tool_descriptions/Read.md"
    target.parent.mkdir(parents=True)
    target.write_text("Old description\n", encoding="utf-8")
    original_db = db_path.read_bytes()

    with patch("core.paths.TEMPLATES_DIR", templates_dir):
        result = step_tool_prompts_db_to_md(data_dir, dry_run=False, verbose=True)

    assert result.error is None
    assert result.changed == 3
    assert target.read_text(encoding="utf-8") == "Custom description\n"
    assert db_path.read_bytes() == original_db


def test_missing_db_is_skipped(tmp_path: Path) -> None:
    result = step_tool_prompts_db_to_md(tmp_path, dry_run=False, verbose=True)

    assert result.error is None
    assert result.changed == 0
    assert result.skipped == 1


def test_empty_db_is_skipped(tmp_path: Path) -> None:
    db_path = tmp_path / "tool_prompts.sqlite3"
    sqlite3.connect(db_path).close()

    with patch("core.paths.TEMPLATES_DIR", tmp_path / "templates"):
        result = step_tool_prompts_db_to_md(tmp_path, dry_run=False, verbose=True)

    assert result.error is None
    assert result.changed == 0
    assert result.skipped == 1


def test_second_run_has_no_changes(tmp_path: Path) -> None:
    db_path = tmp_path / "tool_prompts.sqlite3"
    templates_dir = tmp_path / "templates"
    _create_db(db_path)

    with patch("core.paths.TEMPLATES_DIR", templates_dir):
        first = step_tool_prompts_db_to_md(tmp_path, dry_run=False, verbose=True)
        second = step_tool_prompts_db_to_md(tmp_path, dry_run=False, verbose=True)

    assert first.changed == 3
    assert second.error is None
    assert second.changed == 0
    assert second.skipped == 3


def test_read_only_destination_returns_error(tmp_path: Path) -> None:
    db_path = tmp_path / "tool_prompts.sqlite3"
    templates_dir = tmp_path / "templates"
    _create_db(db_path)

    with (
        patch("core.paths.TEMPLATES_DIR", templates_dir),
        patch.object(Path, "write_text", side_effect=PermissionError("read-only destination")),
    ):
        result = step_tool_prompts_db_to_md(tmp_path, dry_run=False, verbose=True)

    assert result.changed == 0
    assert result.error is not None
    assert "read-only destination" in result.error


def test_dry_run_does_not_change_files(tmp_path: Path) -> None:
    db_path = tmp_path / "tool_prompts.sqlite3"
    templates_dir = tmp_path / "templates"
    _create_db(db_path)
    target = templates_dir / "ja/prompts/tool_descriptions/Read.md"
    target.parent.mkdir(parents=True)
    target.write_text("Old description\n", encoding="utf-8")

    with patch("core.paths.TEMPLATES_DIR", templates_dir):
        result = step_tool_prompts_db_to_md(tmp_path, dry_run=True, verbose=True)

    assert result.error is None
    assert result.changed == 3
    assert target.read_text(encoding="utf-8") == "Old description\n"
    assert not (templates_dir / "ja/prompts/tool_guides/non_s.md").exists()


def test_step_is_registered_as_db_sync_before_version(tmp_path: Path) -> None:
    runner = MigrationRunner(tmp_path)
    register_all_steps(runner)
    steps = runner.list_steps()
    ids = [step["id"] for step in steps]
    registered = next(step for step in steps if step["id"] == "tool_prompts_db_to_md")

    assert registered["category"] == "db_sync"
    assert ids.index("tool_prompts_db_to_md") < ids.index("update_version")
