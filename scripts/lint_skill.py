#!/usr/bin/env python3
"""Skill SKILL.md linter — validates format compliance with Agent Skills standard."""

from __future__ import annotations

import re
import sys
from pathlib import Path


def lint_skill(path: Path) -> tuple[list[str], list[str]]:
    """Lint a SKILL.md file. Returns (errors, warnings)."""
    errors: list[str] = []
    warnings: list[str] = []

    text = path.read_text(encoding="utf-8")

    # Check frontmatter exists
    if not text.startswith("---"):
        errors.append("Missing YAML frontmatter (must start with ---)")
        return errors, warnings

    # Parse frontmatter
    parts = text.split("---", 2)
    if len(parts) < 3:
        errors.append("Malformed YAML frontmatter (missing closing ---)")
        return errors, warnings

    fm_text = parts[1]
    body = parts[2]

    # Extract name
    name_match = re.search(r"^name:\s*(.+)$", fm_text, re.MULTILINE)
    if not name_match:
        errors.append("Missing required field: name")
    else:
        name = name_match.group(1).strip()
        if not re.match(r"^[a-z0-9-]+$", name):
            errors.append(f"name must be lowercase letters, numbers, and hyphens only: '{name}'")
        if len(name) > 64:
            errors.append(f"name must be 64 characters or fewer: {len(name)} chars")

    # Extract description (handle >- folded block scalar)
    desc_match = re.search(r"^description:\s*>-?\s*\n((?:\s+.+\n?)*)", fm_text, re.MULTILINE)
    if not desc_match:
        desc_match = re.search(r"^description:\s*(.+)$", fm_text, re.MULTILINE)

    if not desc_match:
        errors.append("Missing required field: description")
    else:
        desc_raw = desc_match.group(1)
        # Fold multi-line: join with spaces, strip
        desc = " ".join(line.strip() for line in desc_raw.strip().splitlines())

        if not desc:
            errors.append("description must not be empty")
        else:
            # Length check
            if len(desc) > 250:
                warnings.append(f"description exceeds 250 chars ({len(desc)} chars) — will be truncated in catalog")

            # 「」 bracket keyword detection
            brackets = re.findall(r"「.+?」", desc)
            if brackets:
                warnings.append(
                    f"「」bracket keywords detected ({len(brackets)} found) — migrate to 'Use when:' pattern"
                )

            # Use when: pattern check
            if "use when:" not in desc.lower() and "use for:" not in desc.lower():
                warnings.append("No 'Use when:' pattern found — recommended for skill discovery")

            # XML tag check
            if re.search(r"<[a-zA-Z][^>]*>", desc):
                errors.append("description must not contain XML tags")

    # Body length check
    body_lines = body.strip().splitlines()
    if len(body_lines) > 500:
        warnings.append(
            f"SKILL.md body exceeds 500 lines ({len(body_lines)} lines) — consider splitting into references/"
        )

    return errors, warnings


def lint_directory(dir_path: Path) -> dict[str, tuple[list[str], list[str]]]:
    """Lint all SKILL.md files in a skills directory."""
    results: dict[str, tuple[list[str], list[str]]] = {}
    for skill_md in sorted(dir_path.rglob("SKILL.md")):
        results[str(skill_md)] = lint_skill(skill_md)
    return results


def main() -> int:
    """CLI entry point."""
    if len(sys.argv) < 2:
        print("Usage: lint_skill.py <path_to_SKILL.md_or_directory>")
        print("  Validates SKILL.md files against Agent Skills standard.")
        return 1

    target = Path(sys.argv[1])

    if target.is_file():
        errors, warnings = lint_skill(target)
        _print_results(str(target), errors, warnings)
        return 1 if errors else (2 if warnings else 0)

    if target.is_dir():
        results = lint_directory(target)
        if not results:
            print(f"No SKILL.md files found in {target}")
            return 1

        has_errors = False
        has_warnings = False
        for path, (errors, warnings) in results.items():
            _print_results(path, errors, warnings)
            if errors:
                has_errors = True
            if warnings:
                has_warnings = True

        total = len(results)
        err_count = sum(1 for _, (e, _) in results.items() if e)
        warn_count = sum(1 for _, (_, w) in results.items() if w)
        print(f"\n{'=' * 60}")
        print(f"Total: {total} skills, {err_count} with errors, {warn_count} with warnings")
        return 1 if has_errors else (2 if has_warnings else 0)

    print(f"Path not found: {target}")
    return 1


def _print_results(path: str, errors: list[str], warnings: list[str]) -> None:
    """Print lint results for a single file."""
    if not errors and not warnings:
        print(f"✓ {path}")
        return

    print(f"\n{'─' * 60}")
    print(f"  {path}")
    for e in errors:
        print(f"  ERROR: {e}")
    for w in warnings:
        print(f"  WARN:  {w}")


if __name__ == "__main__":
    sys.exit(main())
