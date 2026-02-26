#!/usr/bin/env python3
"""Verify that English template placeholders match Japanese originals.

Lists files in templates/ja/, finds English counterparts in templates/en/,
extracts {placeholder} patterns (single-brace, not {{), and reports mismatches.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Root: repo root (parent of templates/)
REPO_ROOT = Path(__file__).resolve().parent.parent
JA_BASE = REPO_ROOT / "templates" / "ja"
EN_BASE = REPO_ROOT / "templates" / "en"

# Single-brace placeholder: {xyz} but NOT {{xyz}} (escaped)
# Match { when not preceded by {, capture content, match } when not followed by }
_PLACEHOLDER_RE = re.compile(r"(?<!\{)\{([^{}]+)\}(?!\})")


def extract_placeholders(text: str) -> set[str]:
    """Extract {placeholder} patterns from text, skipping {{ and }}."""
    return set(_PLACEHOLDER_RE.findall(text))


def get_file_pairs() -> list[tuple[Path, Path]]:
    """Get (ja_path, en_path) pairs for verification.

    Covers:
    - templates/ja/prompts/ (recursive)
    - bootstrap.md
    - company/vision.md
    - anima_templates/_blank/ (recursive)
    """
    pairs: list[tuple[Path, Path]] = []

    # 1. prompts/ recursive
    prompts_ja = JA_BASE / "prompts"
    prompts_en = EN_BASE / "prompts"
    if prompts_ja.exists():
        for ja_path in prompts_ja.rglob("*"):
            if ja_path.is_file():
                rel = ja_path.relative_to(prompts_ja)
                en_path = prompts_en / rel
                if en_path.exists():
                    pairs.append((ja_path, en_path))

    # 2. bootstrap.md
    for name in ("bootstrap.md",):
        ja_p = JA_BASE / name
        en_p = EN_BASE / name
        if ja_p.exists() and en_p.exists():
            pairs.append((ja_p, en_p))

    # 3. company/vision.md
    ja_vision = JA_BASE / "company" / "vision.md"
    en_vision = EN_BASE / "company" / "vision.md"
    if ja_vision.exists() and en_vision.exists():
        pairs.append((ja_vision, en_vision))

    # 4. anima_templates/_blank/ recursive
    blank_ja = JA_BASE / "anima_templates" / "_blank"
    blank_en = EN_BASE / "anima_templates" / "_blank"
    if blank_ja.exists():
        for ja_path in blank_ja.rglob("*"):
            if ja_path.is_file():
                rel = ja_path.relative_to(blank_ja)
                en_path = blank_en / rel
                if en_path.exists():
                    pairs.append((ja_path, en_path))

    return pairs


def get_ja_only_files() -> list[Path]:
    """List files in ja/ that have no en/ counterpart."""
    ja_only: list[Path] = []

    def collect_ja_without_en(ja_dir: Path, en_dir: Path) -> None:
        if not ja_dir.exists():
            return
        for p in ja_dir.rglob("*"):
            if p.is_file():
                rel = p.relative_to(ja_dir)
                en_p = en_dir / rel
                if not en_p.exists():
                    ja_only.append(p)

    collect_ja_without_en(JA_BASE / "prompts", EN_BASE / "prompts")
    for ja_p, en_p in [
        (JA_BASE / "bootstrap.md", EN_BASE / "bootstrap.md"),
        (JA_BASE / "company" / "vision.md", EN_BASE / "company" / "vision.md"),
    ]:
        if ja_p.exists() and not en_p.exists():
            ja_only.append(ja_p)
    collect_ja_without_en(
        JA_BASE / "anima_templates" / "_blank",
        EN_BASE / "anima_templates" / "_blank",
    )

    return ja_only


def main() -> int:
    """Run verification and report results."""
    pairs = get_file_pairs()
    ja_only = get_ja_only_files()

    errors: list[str] = []
    mismatches: list[tuple[Path, Path, set[str], set[str]]] = []

    for ja_path, en_path in pairs:
        ja_text = ja_path.read_text(encoding="utf-8")
        en_text = en_path.read_text(encoding="utf-8")

        ja_placeholders = extract_placeholders(ja_text)
        en_placeholders = extract_placeholders(en_text)

        in_ja_not_en = ja_placeholders - en_placeholders
        in_en_not_ja = en_placeholders - ja_placeholders

        if in_ja_not_en or in_en_not_ja:
            rel_ja = ja_path.relative_to(REPO_ROOT)
            mismatches.append(
                (ja_path, en_path, in_ja_not_en, in_en_not_ja)
            )

    # Report
    if ja_only:
        errors.append(
            f"Files in ja/ but not in en/ ({len(ja_only)}):"
        )
        for p in sorted(ja_only):
            rel = p.relative_to(REPO_ROOT)
            errors.append(f"  - {rel}")

    if mismatches:
        errors.append(f"\nPlaceholder mismatches ({len(mismatches)} file pairs):")
        for ja_path, en_path, in_ja_not_en, in_en_not_ja in mismatches:
            rel = ja_path.relative_to(REPO_ROOT)
            errors.append(f"\n  {rel}")
            if in_ja_not_en:
                errors.append(f"    In JA but not EN: {sorted(in_ja_not_en)}")
            if in_en_not_ja:
                errors.append(f"    In EN but not JA: {sorted(in_en_not_ja)}")

    if errors:
        print("\n".join(errors))
        return 1

    print("All placeholders match.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
