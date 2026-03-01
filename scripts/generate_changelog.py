#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""generate_changelog.py — Git commits から CHANGELOG.md を自動更新する。

Usage:
    # [Unreleased] セクションをコミットから自動生成
    python scripts/generate_changelog.py

    # [Unreleased] を新バージョンとして確定 + pyproject.toml 更新
    python scripts/generate_changelog.py --release 0.4.0

    # 確認のみ（ファイル変更なし）
    python scripts/generate_changelog.py --dry-run
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHANGELOG = ROOT / "CHANGELOG.md"
PYPROJECT = ROOT / "pyproject.toml"

# ── Commit prefix → changelog category mapping ──────────
PREFIX_MAP: dict[str, str] = {
    "feat": "Added",
    "fix": "Fixed",
    "refactor": "Changed",
    "perf": "Performance",
    "breaking": "Breaking",
}

SKIP_PATTERNS: list[str] = [
    r"^docs:\s.*implementedに移動",
    r"^docs:\s.*作業記録",
    r"^chore:",
    r"^test:",
    r"^ci:",
    r"^style:",
]


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(ROOT), *args],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def get_last_version_date() -> str | None:
    """CHANGELOG.md から最新バージョンの日付を取得する。"""
    if not CHANGELOG.exists():
        return None
    for line in CHANGELOG.read_text().splitlines():
        m = re.match(r"^## \[\d+\.\d+\.\d+\]\s*-\s*(\d{4}-\d{2}-\d{2})", line)
        if m:
            return m.group(1)
    return None


def get_last_version() -> str | None:
    """CHANGELOG.md から最新バージョン番号を取得する。"""
    if not CHANGELOG.exists():
        return None
    for line in CHANGELOG.read_text().splitlines():
        m = re.match(r"^## \[(\d+\.\d+\.\d+)\]", line)
        if m:
            return m.group(1)
    return None


def collect_commits(since_date: str | None) -> list[tuple[str, str]]:
    """指定日以降のコミットを (hash, subject) のリストで返す。"""
    args = ["log", "--oneline", "--no-merges", "--format=%h %s"]
    if since_date:
        args.extend(["--since", f"{since_date} 23:59:59"])
    return [
        (parts[0], " ".join(parts[1:]))
        for line in git(*args).splitlines()
        if line.strip()
        for parts in [line.split(None, 1)]
        if len(parts) == 2
    ]


def should_skip(subject: str) -> bool:
    return any(re.search(p, subject) for p in SKIP_PATTERNS)


def categorize(commits: list[tuple[str, str]]) -> dict[str, list[str]]:
    """コミットをカテゴリ別に分類する。"""
    result: dict[str, list[str]] = defaultdict(list)
    for _hash, subject in commits:
        if should_skip(subject):
            continue
        matched = False
        for prefix, category in PREFIX_MAP.items():
            pattern = rf"^{prefix}(?:\(.+?\))?:\s*(.+)"
            m = re.match(pattern, subject)
            if m:
                result[category].append(m.group(1))
                matched = True
                break
        if not matched and not subject.startswith("docs:"):
            result["Other"].append(subject)
    return dict(result)


def format_unreleased(categories: dict[str, list[str]]) -> str:
    """カテゴリ分類済みコミットを Markdown セクションに整形する。"""
    order = ["Breaking", "Added", "Fixed", "Changed", "Performance", "Other"]
    lines: list[str] = []
    for cat in order:
        entries = categories.get(cat, [])
        if not entries:
            continue
        lines.append(f"### {cat}")
        for entry in entries:
            lines.append(f"- {entry}")
        lines.append("")
    return "\n".join(lines)


def update_changelog(new_unreleased: str, release: str | None = None) -> str:
    """CHANGELOG.md の [Unreleased] セクションを更新する。--release 時はバージョン確定。"""
    if not CHANGELOG.exists():
        raise FileNotFoundError(f"{CHANGELOG} not found")

    text = CHANGELOG.read_text()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    unreleased_re = re.compile(
        r"(## \[Unreleased\]\n)(.*?)(?=\n## \[|\n\[Unreleased\]:|\Z)",
        re.DOTALL,
    )
    m = unreleased_re.search(text)
    if not m:
        raise ValueError("Could not find [Unreleased] section in CHANGELOG.md")

    if release:
        last_ver = get_last_version()
        content = new_unreleased.strip()
        if not content:
            content = m.group(2).strip()

        replacement = (
            f"## [Unreleased]\n\n"
            f"## [{release}] - {today}\n\n"
            f"{content}\n\n"
        )
        text = unreleased_re.sub(replacement, text)

        repo_url = "https://github.com/xuiltul/animaworks"
        links_section = re.compile(
            r"(\[Unreleased\]:.*\n)(\[" + re.escape(last_ver or "") + r"\]:.*)?",
        )
        new_links = (
            f"[Unreleased]: {repo_url}/compare/v{release}...HEAD\n"
            f"[{release}]: {repo_url}/compare/v{last_ver}...v{release}"
            if last_ver
            else (
                f"[Unreleased]: {repo_url}/compare/v{release}...HEAD\n"
                f"[{release}]: {repo_url}/releases/tag/v{release}"
            )
        )
        m_links = links_section.search(text)
        if m_links:
            text = text[: m_links.start()] + new_links + text[m_links.end() :]
    else:
        replacement = f"## [Unreleased]\n\n{new_unreleased}"
        text = unreleased_re.sub(replacement, text)

    return text


def update_pyproject_version(version: str) -> None:
    """pyproject.toml の version を更新する。"""
    text = PYPROJECT.read_text()
    text = re.sub(
        r'^(version\s*=\s*")[^"]*(")', rf"\g<1>{version}\2",
        text, count=1, flags=re.MULTILINE,
    )
    PYPROJECT.write_text(text)


def get_current_version() -> str:
    """pyproject.toml から現在のバージョンを取得する。"""
    text = PYPROJECT.read_text()
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not m:
        raise ValueError("Could not find version in pyproject.toml")
    return m.group(1)


def bump_version(current: str, bump: str = "patch") -> str:
    """SemVer バージョンをインクリメントする。"""
    parts = current.split(".")
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        raise ValueError(f"Invalid semver: {current}")
    major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
    if bump == "major":
        return f"{major + 1}.0.0"
    if bump == "minor":
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


def resolve_release_version(release_arg: str | None) -> str | None:
    """--release 引数をバージョン文字列に解決する。

    patch/minor/major → 自動インクリメント、数値指定 → そのまま。
    """
    if not release_arg:
        return None
    if release_arg in ("patch", "minor", "major"):
        current = get_current_version()
        version = bump_version(current, release_arg)
        print(f"Auto-increment: {current} → {version} ({release_arg})")
        return version
    return release_arg


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CHANGELOG.md from git commits")
    parser.add_argument(
        "--release", metavar="VERSION_OR_BUMP", nargs="?", const="patch",
        help="Cut a release. 'patch' (default if no value), 'minor', 'major', or explicit version (e.g. 0.4.7)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print output without modifying files")
    args = parser.parse_args()

    release_version = resolve_release_version(args.release)

    last_date = get_last_version_date()
    commits = collect_commits(last_date)

    if not commits and not release_version:
        print("No new commits since last version. Nothing to update.")
        return

    categories = categorize(commits)
    unreleased_text = format_unreleased(categories)

    if args.dry_run:
        print("=== Unreleased entries ===")
        print(unreleased_text or "(empty)")
        if release_version:
            print(f"\nWould release as v{release_version}")
        return

    new_text = update_changelog(unreleased_text, release=release_version)
    CHANGELOG.write_text(new_text)
    print(f"Updated {CHANGELOG}")

    if release_version:
        update_pyproject_version(release_version)
        print(f"Updated {PYPROJECT} version to {release_version}")
        print(f"\nNext steps:")
        print(f"  git add CHANGELOG.md pyproject.toml")
        print(f"  git commit -m 'release: v{release_version}'")
        print(f"  git tag v{release_version}")

    total = sum(len(v) for v in categories.values())
    print(f"\n{total} entries across {len(categories)} categories:")
    for cat, entries in sorted(categories.items()):
        print(f"  {cat}: {len(entries)}")


if __name__ == "__main__":
    main()
