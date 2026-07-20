from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Convert the legacy tool prompt SQLite database to Markdown templates."""

import json
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

SUPPORTED_LOCALES = {"ja", "en", "ko"}


@dataclass(frozen=True)
class TableSpec:
    """Map a legacy database table to its Markdown destination."""

    name: str
    key_column: str
    value_column: str
    subdirectory: str | None


@dataclass(frozen=True)
class MigrationMessages:
    """User-facing progress message templates for a migration frontend."""

    missing_db: str
    default_locale: str
    missing_table: str
    empty_value: str
    unsupported_locale: str
    unchanged: str
    created: str
    updated: str
    summary: str
    database_unchanged: str
    cleanup_hint: str


TABLE_SPECS = (
    TableSpec("tool_descriptions", "name", "description", "tool_descriptions"),
    TableSpec("tool_guides", "key", "content", "tool_guides"),
    TableSpec("system_sections", "key", "content", None),
)

DEFAULT_MESSAGES = MigrationMessages(
    missing_db="Database not found: {db_path}; skipping migration.",
    default_locale="Destination for rows without a locale: {locale}",
    missing_table="SKIP table={table}: table or required columns are missing",
    empty_value="SKIP {table}/{key}: empty value",
    unsupported_locale="SKIP {table}/{key}: unsupported locale {locale!r}",
    unchanged="SKIP {target}: no differences",
    created="new",
    updated="updated",
    summary="Result: written={written}, skipped={skipped}",
    database_unchanged="Source database was not modified or deleted: {db_path}",
    cleanup_hint="Delete the obsolete database manually after verifying the result.",
)


def _default_locale(db_path: Path, requested: str | None) -> str:
    if requested:
        return requested
    config_path = db_path.parent / "config.json"
    try:
        locale = json.loads(config_path.read_text(encoding="utf-8")).get("locale")
    except (OSError, json.JSONDecodeError, AttributeError):
        locale = None
    return locale if locale in SUPPORTED_LOCALES else "ja"


def _read_only_connection(db_path: Path) -> sqlite3.Connection:
    uri = f"file:{quote(str(db_path.resolve()))}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _table_columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in connection.execute(f'PRAGMA table_info("{table}")')}


def _safe_key(value: object) -> str:
    key = str(value)
    if not key or key in {".", ".."} or "/" in key or "\\" in key or "\x00" in key:
        raise ValueError(f"unsafe prompt key: {key!r}")
    return key


def _target_path(templates_dir: Path, locale: str, spec: TableSpec, key: str) -> Path:
    prompts_dir = templates_dir / locale / "prompts"
    if spec.name == "system_sections" and key == "emotion_instruction":
        return prompts_dir / "builder" / "emotion_instruction.md"
    if spec.subdirectory:
        prompts_dir /= spec.subdirectory
    return prompts_dir / f"{key}.md"


def _markdown_content(spec: TableSpec, key: str, raw_content: object) -> str:
    content = str(raw_content).rstrip()
    if spec.name == "system_sections" and key == "emotion_instruction":
        # This DB row is already rendered text, while load_prompt applies
        # str.format_map. Escape its literal JSON object for safe reloading.
        content = content.replace("{", "{{").replace("}", "}}")
    return content + "\n"


def migrate_tool_prompts(
    db_path: Path,
    templates_dir: Path,
    *,
    locale: str | None = None,
    dry_run: bool = False,
    output: Callable[[str], None] = print,
    messages: MigrationMessages = DEFAULT_MESSAGES,
) -> tuple[int, int]:
    """Migrate DB rows and return ``(written, skipped)`` counts.

    The database is always opened in SQLite read-only mode. ``output`` allows
    the CLI to print progress while migration steps collect it as details.
    """
    db_path = db_path.expanduser()
    templates_dir = templates_dir.expanduser()
    if not db_path.is_file():
        output(messages.missing_db.format(db_path=db_path))
        return 0, 0

    default_locale = _default_locale(db_path, locale)
    written = 0
    skipped = 0
    mode_label = "DRY-RUN" if dry_run else "WRITE"
    output(f"[{mode_label}] DB: {db_path}")
    output(messages.default_locale.format(locale=default_locale))

    with _read_only_connection(db_path) as connection:
        for spec in TABLE_SPECS:
            columns = _table_columns(connection, spec.name)
            required = {spec.key_column, spec.value_column}
            if not required.issubset(columns):
                output(messages.missing_table.format(table=spec.name))
                continue

            locale_column = next((name for name in ("locale", "language", "lang") if name in columns), None)
            selected = [spec.key_column, spec.value_column]
            if locale_column:
                selected.append(locale_column)
            quoted = ", ".join(f'"{name}"' for name in selected)
            rows = connection.execute(f'SELECT {quoted} FROM "{spec.name}" ORDER BY "{spec.key_column}"')

            for row in rows:
                key = _safe_key(row[spec.key_column])
                raw_content = row[spec.value_column]
                if raw_content is None or not str(raw_content).strip():
                    skipped += 1
                    output(messages.empty_value.format(table=spec.name, key=key))
                    continue
                row_locale = str(row[locale_column]) if locale_column and row[locale_column] else default_locale
                if row_locale not in SUPPORTED_LOCALES:
                    skipped += 1
                    output(messages.unsupported_locale.format(table=spec.name, key=key, locale=row_locale))
                    continue

                content = _markdown_content(spec, key, raw_content)
                target = _target_path(templates_dir, row_locale, spec, key)
                try:
                    current = target.read_text(encoding="utf-8")
                except FileNotFoundError:
                    current = None
                if current == content:
                    skipped += 1
                    output(messages.unchanged.format(target=target))
                    continue

                written += 1
                action = "WOULD WRITE" if dry_run else "WRITE"
                state = messages.created if current is None else messages.updated
                output(f"{action} {target} ({state})")
                if not dry_run:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(content, encoding="utf-8")

    output(messages.summary.format(written=written, skipped=skipped))
    output(messages.database_unchanged.format(db_path=db_path))
    output(messages.cleanup_hint)
    return written, skipped
