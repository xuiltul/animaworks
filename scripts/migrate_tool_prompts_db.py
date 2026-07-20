#!/usr/bin/env python3
"""Migrate the legacy tool prompt SQLite database to Markdown templates."""

from __future__ import annotations

import argparse
import sqlite3
from collections.abc import Sequence
from pathlib import Path

from core.migrations.tool_prompts import SUPPORTED_LOCALES, MigrationMessages, migrate_tool_prompts
from core.paths import TEMPLATES_DIR

DEFAULT_DB_PATH = Path.home() / ".animaworks" / "tool_prompts.sqlite3"
DEFAULT_TEMPLATES_DIR = TEMPLATES_DIR
CLI_MESSAGES = MigrationMessages(
    missing_db="DB無し: {db_path}。マイグレーションをスキップします。",
    default_locale="ロケール列がない行の書き出し先: {locale}",
    missing_table="SKIP table={table}: テーブルまたは必須列がありません",
    empty_value="SKIP {table}/{key}: 空の値",
    unsupported_locale="SKIP {table}/{key}: 未対応ロケール {locale!r}",
    unchanged="SKIP {target}: 差分なし",
    created="新規",
    updated="更新",
    summary="結果: 書き出し={written}件、スキップ={skipped}件",
    database_unchanged="元DBは変更・削除していません: {db_path}",
    cleanup_hint="不要になったDBは、動作確認後に手動削除できます。",
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


def migrate(
    db_path: Path,
    templates_dir: Path,
    *,
    locale: str | None = None,
    dry_run: bool = False,
) -> tuple[int, int]:
    """Migrate DB rows and return ``(written, skipped)`` counts."""
    return migrate_tool_prompts(
        db_path,
        templates_dir,
        locale=locale,
        dry_run=dry_run,
        messages=CLI_MESSAGES,
    )


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
