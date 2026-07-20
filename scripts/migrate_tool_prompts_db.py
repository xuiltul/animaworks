#!/usr/bin/env python3
"""Migrate the legacy tool prompt SQLite database to Markdown templates."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

SUPPORTED_LOCALES = {"ja", "en", "ko"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = Path.home() / ".animaworks" / "tool_prompts.sqlite3"
DEFAULT_TEMPLATES_DIR = PROJECT_ROOT / "templates"


@dataclass(frozen=True)
class TableSpec:
    name: str
    key_column: str
    value_column: str
    subdirectory: str | None


TABLE_SPECS = (
    TableSpec("tool_descriptions", "name", "description", "tool_descriptions"),
    TableSpec("tool_guides", "key", "content", "tool_guides"),
    TableSpec("system_sections", "key", "content", None),
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH, help="legacy SQLite DB path")
    parser.add_argument(
        "--templates",
        type=Path,
        default=DEFAULT_TEMPLATES_DIR,
        help="templates directory to update",
    )
    parser.add_argument("--locale", choices=sorted(SUPPORTED_LOCALES), help="locale for rows without a locale column")
    parser.add_argument("--dry-run", action="store_true", help="show changes without writing files")
    return parser.parse_args(argv)


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


def migrate(
    db_path: Path,
    templates_dir: Path,
    *,
    locale: str | None = None,
    dry_run: bool = False,
) -> tuple[int, int]:
    """Migrate DB rows and return ``(written, skipped)`` counts."""
    db_path = db_path.expanduser()
    templates_dir = templates_dir.expanduser()
    if not db_path.is_file():
        print(f"DB無し: {db_path}。マイグレーションをスキップします。")
        return 0, 0

    default_locale = _default_locale(db_path, locale)
    written = 0
    skipped = 0
    mode_label = "DRY-RUN" if dry_run else "WRITE"
    print(f"[{mode_label}] DB: {db_path}")
    print(f"ロケール列がない行の書き出し先: {default_locale}")

    with _read_only_connection(db_path) as connection:
        for spec in TABLE_SPECS:
            columns = _table_columns(connection, spec.name)
            required = {spec.key_column, spec.value_column}
            if not required.issubset(columns):
                print(f"SKIP table={spec.name}: テーブルまたは必須列がありません")
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
                    print(f"SKIP {spec.name}/{key}: 空の値")
                    continue
                row_locale = str(row[locale_column]) if locale_column and row[locale_column] else default_locale
                if row_locale not in SUPPORTED_LOCALES:
                    skipped += 1
                    print(f"SKIP {spec.name}/{key}: 未対応ロケール {row_locale!r}")
                    continue

                content = _markdown_content(spec, key, raw_content)
                target = _target_path(templates_dir, row_locale, spec, key)
                try:
                    current = target.read_text(encoding="utf-8")
                except FileNotFoundError:
                    current = None
                if current == content:
                    skipped += 1
                    print(f"SKIP {target}: 差分なし")
                    continue

                written += 1
                action = "WOULD WRITE" if dry_run else "WRITE"
                print(f"{action} {target} ({'新規' if current is None else '更新'})")
                if not dry_run:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(content, encoding="utf-8")

    print(f"結果: 書き出し={written}件、スキップ={skipped}件")
    print(f"元DBは変更・削除していません: {db_path}")
    print("不要になったDBは、動作確認後に手動削除できます。")
    return written, skipped


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        migrate(args.db, args.templates, locale=args.locale, dry_run=args.dry_run)
    except (sqlite3.Error, OSError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
