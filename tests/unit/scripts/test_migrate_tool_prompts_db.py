from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import patch

from core.paths import load_prompt, load_prompt_text
from core.prompt.tool_content import apply_prompt_descriptions, load_guide
from scripts.migrate_tool_prompts_db import main, migrate


def _create_db(path: Path, *, with_locale: bool = False) -> None:
    connection = sqlite3.connect(path)
    locale_column = ", locale TEXT NOT NULL" if with_locale else ""
    primary_key = "PRIMARY KEY (name, locale)" if with_locale else "PRIMARY KEY (name)"
    connection.execute(
        f"CREATE TABLE tool_descriptions "
        f"(name TEXT NOT NULL, description TEXT, updated_at TEXT{locale_column}, {primary_key})"
    )
    guide_primary_key = "PRIMARY KEY (key, locale)" if with_locale else "PRIMARY KEY (key)"
    connection.execute(
        f"CREATE TABLE tool_guides "
        f"(key TEXT NOT NULL, content TEXT, updated_at TEXT{locale_column}, {guide_primary_key})"
    )
    section_primary_key = "PRIMARY KEY (key, locale)" if with_locale else "PRIMARY KEY (key)"
    connection.execute(
        f"CREATE TABLE system_sections "
        f"(key TEXT NOT NULL, content TEXT, condition TEXT, updated_at TEXT{locale_column}, {section_primary_key})"
    )
    if with_locale:
        connection.execute("INSERT INTO tool_descriptions VALUES (?, ?, ?, ?)", ("Read", "説明", "now", "ja"))
        connection.execute("INSERT INTO tool_descriptions VALUES (?, ?, ?, ?)", ("Read", "Description", "now", "en"))
    else:
        connection.execute("INSERT INTO tool_descriptions VALUES (?, ?, ?)", ("Read", "Read description", "now"))
        connection.execute("INSERT INTO tool_guides VALUES (?, ?, ?)", ("non_s", "Tool guide", "now"))
        connection.execute(
            "INSERT INTO system_sections VALUES (?, ?, ?, ?)",
            ("behavior_rules", "Rules", None, "now"),
        )
        connection.execute("INSERT INTO tool_guides VALUES (?, ?, ?)", ("empty", "", "now"))
    connection.commit()
    connection.close()


def test_missing_db_is_successful_skip(tmp_path: Path, capsys) -> None:
    result = main(["--db", str(tmp_path / "missing.sqlite3"), "--templates", str(tmp_path / "templates")])
    assert result == 0
    assert "DB無し" in capsys.readouterr().out


def test_dry_run_lists_changes_without_writing(tmp_path: Path, capsys) -> None:
    db_path = tmp_path / "prompts.sqlite3"
    templates = tmp_path / "templates"
    _create_db(db_path)

    written, skipped = migrate(db_path, templates, locale="en", dry_run=True)

    assert (written, skipped) == (3, 1)
    assert not templates.exists()
    output = capsys.readouterr().out
    assert "WOULD WRITE" in output
    assert "書き出し=3件、スキップ=1件" in output


def test_writes_all_table_types_and_is_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "prompts.sqlite3"
    templates = tmp_path / "templates"
    _create_db(db_path)

    assert migrate(db_path, templates, locale="en") == (3, 1)
    assert (templates / "en/prompts/tool_descriptions/Read.md").read_text() == "Read description\n"
    assert (templates / "en/prompts/tool_guides/non_s.md").read_text() == "Tool guide\n"
    assert (templates / "en/prompts/behavior_rules.md").read_text() == "Rules\n"
    assert migrate(db_path, templates, locale="en") == (0, 4)


def test_locale_column_routes_rows_to_locale_directories(tmp_path: Path) -> None:
    db_path = tmp_path / "localized.sqlite3"
    templates = tmp_path / "templates"
    _create_db(db_path, with_locale=True)

    assert migrate(db_path, templates) == (2, 0)
    assert (templates / "ja/prompts/tool_descriptions/Read.md").read_text() == "説明\n"
    assert (templates / "en/prompts/tool_descriptions/Read.md").read_text() == "Description\n"


def test_existing_difference_is_overwritten_but_db_is_unchanged(tmp_path: Path) -> None:
    db_path = tmp_path / "prompts.sqlite3"
    templates = tmp_path / "templates"
    _create_db(db_path)
    target = templates / "ja/prompts/behavior_rules.md"
    target.parent.mkdir(parents=True)
    target.write_text("old\n", encoding="utf-8")
    before = db_path.read_bytes()

    migrate(db_path, templates)

    assert target.read_text(encoding="utf-8") == "Rules\n"
    assert db_path.read_bytes() == before


def test_rendered_emotion_json_is_escaped_for_template_loading(tmp_path: Path) -> None:
    db_path = tmp_path / "prompts.sqlite3"
    templates = tmp_path / "templates"
    _create_db(db_path)
    connection = sqlite3.connect(db_path)
    rendered = '<!-- emotion: {"emotion": "<name>"} -->'
    connection.execute(
        "INSERT INTO system_sections VALUES (?, ?, ?, ?)",
        ("emotion_instruction", rendered, None, "now"),
    )
    connection.commit()
    connection.close()

    migrate(db_path, templates, locale="en")

    target = templates / "en/prompts/builder/emotion_instruction.md"
    assert '{{"emotion": "<name>"}}' in target.read_text(encoding="utf-8")
    with patch("core.paths.TEMPLATES_DIR", templates):
        assert load_prompt("builder/emotion_instruction", locale="en").strip() == rendered


def test_reflection_json_examples_are_preserved_by_raw_loading(tmp_path: Path) -> None:
    db_path = tmp_path / "prompts.sqlite3"
    templates = tmp_path / "templates"
    _create_db(db_path)
    connection = sqlite3.connect(db_path)
    rendered = 'Never expose {"status": "raw"}'
    connection.execute(
        "INSERT INTO system_sections VALUES (?, ?, ?, ?)",
        ("a_reflection", rendered, None, "now"),
    )
    connection.commit()
    connection.close()

    migrate(db_path, templates, locale="en")

    target = templates / "en/prompts/a_reflection.md"
    assert '{"status": "raw"}' in target.read_text(encoding="utf-8")
    with patch("core.paths.TEMPLATES_DIR", templates):
        assert load_prompt_text("a_reflection", locale="en").strip() == rendered


def test_custom_json_in_descriptions_and_guides_survives_migration(tmp_path: Path) -> None:
    db_path = tmp_path / "prompts.sqlite3"
    templates = tmp_path / "templates"
    _create_db(db_path)
    connection = sqlite3.connect(db_path)
    description = 'Return JSON like {"status": "ok"}'
    guide = 'Example: {"tool": "Read", "arguments": {"path": "x"}}'
    connection.execute("INSERT INTO tool_descriptions VALUES (?, ?, ?)", ("JsonTool", description, "now"))
    connection.execute("INSERT INTO tool_guides VALUES (?, ?, ?)", ("json_guide", guide, "now"))
    connection.commit()
    connection.close()

    migrate(db_path, templates, locale="en")

    with (
        patch("core.paths.TEMPLATES_DIR", templates),
        patch("core.paths._get_locale", return_value="en"),
    ):
        tools = [{"name": "JsonTool", "description": "fallback", "parameters": {}}]
        assert apply_prompt_descriptions(tools)[0]["description"] == description
        assert load_guide("json_guide", locale="en").strip() == guide
