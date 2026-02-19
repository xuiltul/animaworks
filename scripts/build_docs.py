#!/usr/bin/env python3

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""docs/ ディレクトリを走査して mkdocs.yml の nav を自動生成し、mkdocs build を実行する。

Usage:
    ./scripts/build_docs.py             # nav更新 + ビルド
    ./scripts/build_docs.py --dry-run   # nav更新のみ（ビルドしない）
    ./scripts/build_docs.py --diff      # 差分表示のみ（変更しない）
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"
MKDOCS_YML = PROJECT_ROOT / "mkdocs.yml"


# ── Title Extraction ──────────────────────────────────────

def _extract_title(md_path: Path) -> str:
    """Markdown ファイルから最初の # 見出しを抽出する。"""
    try:
        with open(md_path, encoding="utf-8") as f:
            for line in f:
                m = re.match(r"^#\s+(.+)", line)
                if m:
                    return m.group(1).strip()
    except (OSError, UnicodeDecodeError):
        pass
    # フォールバック: ファイル名からタイトル生成
    stem = md_path.stem
    stem = re.sub(r"^\d{8}_", "", stem)
    stem = re.sub(r"^\d{4}-\d{2}-\d{2}_", "", stem)
    return stem.replace("_", " ").replace("-", " ")


# ── Nav Tree Builder ──────────────────────────────────────

def _build_nav_tree(base_dir: Path, rel_to: Path) -> list[dict]:
    """ディレクトリを再帰的に走査して nav ツリーを構築する。"""
    if not base_dir.is_dir():
        return []

    entries: list[dict] = []
    files = sorted(
        [f for f in base_dir.iterdir() if f.is_file() and f.suffix == ".md"],
        key=lambda p: p.name.lower(),
    )
    dirs = sorted(
        [d for d in base_dir.iterdir() if d.is_dir()],
        key=lambda p: p.name.lower(),
    )

    # index.md を先頭に配置
    for f in files:
        if f.name == "index.md":
            title = _extract_title(f)
            entries.append({title: str(f.relative_to(rel_to))})
            break

    # その他の .md ファイル
    for f in files:
        if f.name == "index.md":
            continue
        title = _extract_title(f)
        entries.append({title: str(f.relative_to(rel_to))})

    # サブディレクトリ
    for d in dirs:
        children = _build_nav_tree(d, rel_to)
        if children:
            entries.append({d.name + "/": children})

    return entries


# ── YAML Generation ───────────────────────────────────────

def _yaml_quote(s: str) -> str:
    """YAML 特殊文字を含む文字列をクォートする。"""
    if re.search(r"[:\[\]{}#&*!|>'\"%@`]", s):
        escaped = s.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return s


def _nav_to_yaml_lines(nav: list[dict], indent: int = 2) -> list[str]:
    """nav ツリーを YAML テキスト行に変換する。"""
    lines: list[str] = []
    prefix = " " * indent

    for entry in nav:
        for title, value in entry.items():
            quoted_title = _yaml_quote(title)
            if isinstance(value, str):
                lines.append(f"{prefix}- {quoted_title}: {value}")
            elif isinstance(value, list):
                lines.append(f"{prefix}- {quoted_title}:")
                lines.extend(_nav_to_yaml_lines(value, indent + 4))

    return lines


# ── mkdocs.yml Update ────────────────────────────────────

def _replace_nav_section(original: str, nav_lines: list[str]) -> str:
    """mkdocs.yml テキスト内の nav セクションのみを置換する。"""
    nav_text = "nav:\n" + "\n".join(nav_lines) + "\n"

    # nav: セクション開始位置
    nav_match = re.search(r"^nav:\s*\n", original, re.MULTILINE)
    if not nav_match:
        # nav がなければ先頭に追加
        return nav_text + "\n" + original

    # 次のトップレベルキー (インデントなしの非空行) を探す
    after_nav = original[nav_match.end():]
    next_key = re.search(r"^\S", after_nav, re.MULTILINE)

    if next_key:
        end_pos = nav_match.end() + next_key.start()
    else:
        end_pos = len(original)

    return original[: nav_match.start()] + nav_text + "\n" + original[end_pos:]


# ── Main ─────────────────────────────────────────────────

def main() -> int:
    diff_only = "--diff" in sys.argv
    dry_run = "--dry-run" in sys.argv or diff_only

    # nav ツリー構築
    nav = _build_nav_tree(DOCS_DIR, DOCS_DIR)
    nav_lines = _nav_to_yaml_lines(nav)

    original = MKDOCS_YML.read_text(encoding="utf-8")
    updated = _replace_nav_section(original, nav_lines)

    if diff_only:
        if original == updated:
            print("nav is up to date.")
        else:
            # 簡易 diff 表示
            old_lines = original.splitlines()
            new_lines = updated.splitlines()
            import difflib

            diff = difflib.unified_diff(
                old_lines, new_lines, fromfile="mkdocs.yml (current)", tofile="mkdocs.yml (generated)", lineterm=""
            )
            for line in diff:
                print(line)
        return 0

    # mkdocs.yml 書き込み
    MKDOCS_YML.write_text(updated, encoding="utf-8")
    file_count = sum(1 for line in nav_lines if not line.rstrip().endswith(":"))
    print(f"nav updated: {file_count} files")

    if dry_run:
        print("(dry-run: skipping mkdocs build)")
        return 0

    # mkdocs build
    result = subprocess.run(["mkdocs", "build"], cwd=PROJECT_ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"mkdocs build failed:\n{result.stderr}", file=sys.stderr)
        return 1

    print("mkdocs build completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
