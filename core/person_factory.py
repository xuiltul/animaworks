# AnimaWorks - Digital Person Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# This file is part of AnimaWorks core/server, licensed under AGPL-3.0.
# See LICENSES/AGPL-3.0.txt for the full license text.

"""Person creation factory: create new Digital Persons from templates, blank, or MD files."""

from __future__ import annotations

import json
import logging
import re
import shutil
from pathlib import Path

from core.paths import TEMPLATES_DIR

logger = logging.getLogger("animaworks.person_factory")

PERSON_TEMPLATES_DIR = TEMPLATES_DIR / "person_templates"
BLANK_TEMPLATE_DIR = PERSON_TEMPLATES_DIR / "_blank"
BOOTSTRAP_TEMPLATE = TEMPLATES_DIR / "bootstrap.md"

# Subdirectories every person needs at runtime
_RUNTIME_SUBDIRS = [
    "episodes",
    "knowledge",
    "procedures",
    "skills",
    "state",
    "shortterm",
    "shortterm/archive",
]


def list_person_templates() -> list[str]:
    """List available person templates (excluding _blank)."""
    if not PERSON_TEMPLATES_DIR.exists():
        return []
    return [
        d.name
        for d in sorted(PERSON_TEMPLATES_DIR.iterdir())
        if d.is_dir() and not d.name.startswith("_")
    ]


def create_from_template(
    persons_dir: Path, template_name: str, *, person_name: str | None = None
) -> Path:
    """Create a person by copying a named template.

    Args:
        persons_dir: Runtime persons directory (~/.animaworks/persons/).
        template_name: Template directory name under person_templates/.
        person_name: Override the directory name.  Defaults to template_name.

    Returns:
        Path to the created person directory.
    """
    template_dir = PERSON_TEMPLATES_DIR / template_name
    if not template_dir.exists():
        raise FileNotFoundError(f"Template not found: {template_name}")

    name = person_name or template_name
    person_dir = persons_dir / name
    if person_dir.exists():
        raise FileExistsError(f"Person already exists: {name}")

    shutil.copytree(template_dir, person_dir)
    _ensure_runtime_subdirs(person_dir)
    _init_state_files(person_dir)
    _place_bootstrap(person_dir)
    _place_send_script(person_dir)

    logger.info("Created person '%s' from template '%s'", name, template_name)
    return person_dir


def create_blank(persons_dir: Path, name: str) -> Path:
    """Create a blank person with skeleton files.

    The {name} placeholder in skeleton files is replaced with the actual name.
    Copies the entire _blank template tree (including subdirectories like skills/).

    Args:
        persons_dir: Runtime persons directory.
        name: Person name (lowercase alphanumeric).

    Returns:
        Path to the created person directory.
    """
    person_dir = persons_dir / name
    if person_dir.exists():
        raise FileExistsError(f"Person already exists: {name}")

    try:
        if BLANK_TEMPLATE_DIR.exists():
            shutil.copytree(BLANK_TEMPLATE_DIR, person_dir)
            # Replace {name} placeholder in all markdown files
            for md_file in person_dir.rglob("*.md"):
                content = md_file.read_text(encoding="utf-8")
                if "{name}" in content:
                    md_file.write_text(
                        content.replace("{name}", name), encoding="utf-8"
                    )
        else:
            person_dir.mkdir(parents=True, exist_ok=True)

        _ensure_runtime_subdirs(person_dir)
        _init_state_files(person_dir)
        _place_bootstrap(person_dir)
        _place_send_script(person_dir)
    except Exception:
        logger.error("Failed to create blank person '%s'; rolling back", name)
        shutil.rmtree(person_dir, ignore_errors=True)
        raise

    logger.info("Created blank person '%s'", name)
    return person_dir


def create_from_md(persons_dir: Path, md_path: Path, name: str | None = None) -> Path:
    """Create a person from an MD character-sheet file.

    The MD file is validated, then placed as ``character_sheet.md`` in the new
    person directory.  Sections in the sheet are applied to identity.md,
    injection.md, permissions.md, and status.json.

    On any failure after the person directory has been created the directory
    is rolled back (removed) so no partial state is left behind.

    Args:
        persons_dir: Runtime persons directory.
        md_path: Path to the source MD file.
        name: Person name.  If None, extracted from MD content.

    Returns:
        Path to the created person directory.

    Raises:
        FileNotFoundError: If *md_path* does not exist.
        ValueError: If the character sheet is invalid or the name cannot be
            determined.
    """
    if not md_path.exists():
        raise FileNotFoundError(f"MD file not found: {md_path}")

    md_content = md_path.read_text(encoding="utf-8")

    # Validate before creating anything on disk
    _validate_character_sheet(md_content)

    if not name:
        name = _extract_name_from_md(md_content)
    if not name:
        raise ValueError(
            "Could not extract person name from MD file. "
            "Add a '# Character: name' heading or specify --name."
        )

    # Create blank skeleton first, then layer character sheet on top
    person_dir = create_blank(persons_dir, name)
    try:
        (person_dir / "character_sheet.md").write_text(md_content, encoding="utf-8")
        _apply_defaults_from_sheet(person_dir, md_content)
        _create_status_json(person_dir, _parse_character_sheet_info(md_content))
    except Exception:
        logger.error(
            "Failed to set up person '%s' from MD file; rolling back", name
        )
        shutil.rmtree(person_dir, ignore_errors=True)
        raise

    logger.info("Created person '%s' from MD file '%s'", name, md_path)
    return person_dir


def _extract_name_from_md(content: str) -> str | None:
    """Try to extract a person name from MD content.

    Looks for patterns like:
        # Character: Hinata
        # {name}
        英名 Hinata
    """
    # Try "# Character: Name" or "# Name"
    m = re.search(r"^#\s+(?:Character:\s*)?(\w+)", content, re.MULTILINE)
    if m:
        return m.group(1).lower()

    # Try "| 英名 | Name |" table row format
    m = re.search(r"\|\s*英名\s*\|\s*(\w+)\s*\|", content)
    if m:
        return m.group(1).lower()

    # Try "英名 Name"
    m = re.search(r"英名\s+(\w+)", content)
    if m:
        return m.group(1).lower()

    return None


# ── CharacterSheet Helpers ──────────


def _parse_character_sheet_info(content: str) -> dict[str, str]:
    """Parse the markdown table in the '基本情報' section of a character sheet.

    Expects rows like ``| 英名 | sakura |`` and returns a dict mapping
    the first column to the second, e.g. ``{"英名": "sakura", ...}``.

    Args:
        content: Full markdown content of the character sheet.

    Returns:
        Mapping of item name to value parsed from the table.
    """
    info: dict[str, str] = {}
    in_section = False
    for line in content.splitlines():
        if re.match(r"^##\s+基本情報", line):
            in_section = True
            continue
        if in_section and line.startswith("## "):
            break
        if not in_section:
            continue
        # Match table data rows (skip header / separator rows)
        m = re.match(r"\|\s*(.+?)\s*\|\s*(.+?)\s*\|", line)
        if m:
            key = m.group(1).strip()
            value = m.group(2).strip()
            # Skip header row and separator row
            if key in ("項目", "---", "------") or value in ("設定", "---", "------"):
                continue
            if set(key) <= {"-", " "} or set(value) <= {"-", " "}:
                continue
            info[key] = value
    return info


def _validate_character_sheet(content: str) -> None:
    """Validate that a character sheet contains all required sections.

    Required sections:
        - ``## 基本情報``
        - ``## 人格`` (or ``→ identity.md``)
        - ``## 役割・行動方針`` (or ``→ injection.md``)

    Args:
        content: Full markdown content of the character sheet.

    Raises:
        ValueError: If any required section is missing.
    """
    missing: list[str] = []

    if not re.search(r"^##\s+基本情報", content, re.MULTILINE):
        missing.append("基本情報")

    has_personality = (
        re.search(r"^##\s+人格", content, re.MULTILINE) is not None
        or "→ identity.md" in content
    )
    if not has_personality:
        missing.append("人格 (or '→ identity.md')")

    has_injection = (
        re.search(r"^##\s+役割・行動方針", content, re.MULTILINE) is not None
        or "→ injection.md" in content
    )
    if not has_injection:
        missing.append("役割・行動方針 (or '→ injection.md')")

    if missing:
        raise ValueError(
            "Character sheet is missing required sections: "
            + ", ".join(missing)
        )


def _create_status_json(person_dir: Path, info: dict[str, str]) -> None:
    """Create status.json in *person_dir* from parsed character-sheet info.

    The JSON contains supervisor, role, execution mode, model, and credential
    extracted from the ``基本情報`` table.

    Args:
        person_dir: Target person directory.
        info: Dict returned by :func:`_parse_character_sheet_info`.
    """
    supervisor_raw = info.get("上司", "")
    supervisor = "" if supervisor_raw in ("(なし)", "なし", "-", "") else supervisor_raw

    status = {
        "supervisor": supervisor,
        "role": info.get("役割", "worker"),
        "execution_mode": info.get("実行モード", "autonomous"),
        "model": info.get("モデル", ""),
        "credential": info.get("credential", ""),
    }
    (person_dir / "status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.debug("Created status.json in %s", person_dir)


def _extract_section_content(md_content: str, heading: str) -> str | None:
    """Extract the body text under a ``## heading`` section.

    Returns the content between the heading and the next ``##`` heading
    (or end-of-file), stripped.  Returns ``None`` if the heading is not
    found or the section body is empty / only contains a redirect marker
    like ``→ identity.md``.

    Args:
        md_content: Full markdown text.
        heading: Section heading text (without ``## ``).

    Returns:
        Section body text, or ``None``.
    """
    pattern = rf"^##\s+{re.escape(heading)}\s*\n(.*?)(?=^##\s|\Z)"
    m = re.search(pattern, md_content, re.MULTILINE | re.DOTALL)
    if not m:
        return None
    body = m.group(1).strip()
    # Treat redirect markers as "no content"
    if not body or body.startswith("→"):
        return None
    return body


def _apply_defaults_from_sheet(person_dir: Path, md_content: str) -> None:
    """Apply character-sheet sections to person files, keeping template defaults.

    For sections marked ``[省略可]`` in the sheet, if they are omitted or
    empty the existing template files (heartbeat.md, cron.md, permissions.md)
    are left untouched.

    Sections that **are** present overwrite the corresponding file:
        - ``## 人格``          → ``identity.md``
        - ``## 役割・行動方針`` → ``injection.md``
        - ``## 権限``          → ``permissions.md``

    Args:
        person_dir: Target person directory (already populated by create_blank).
        md_content: Full markdown content of the character sheet.
    """
    # Identity
    personality = _extract_section_content(md_content, "人格")
    if personality:
        (person_dir / "identity.md").write_text(
            personality + "\n", encoding="utf-8"
        )
        logger.debug("Wrote identity.md from character sheet for %s", person_dir.name)

    # Injection
    injection = _extract_section_content(md_content, "役割・行動方針")
    if injection:
        (person_dir / "injection.md").write_text(
            injection + "\n", encoding="utf-8"
        )
        logger.debug("Wrote injection.md from character sheet for %s", person_dir.name)

    # Permissions
    permissions = _extract_section_content(md_content, "権限")
    if permissions:
        (person_dir / "permissions.md").write_text(
            permissions + "\n", encoding="utf-8"
        )
        logger.debug("Wrote permissions.md from character sheet for %s", person_dir.name)


# ── Runtime Helpers ──────────


def _ensure_runtime_subdirs(person_dir: Path) -> None:
    """Create runtime-only subdirectories."""
    for subdir in _RUNTIME_SUBDIRS:
        (person_dir / subdir).mkdir(parents=True, exist_ok=True)


def _init_state_files(person_dir: Path) -> None:
    """Create initial state files if they don't exist."""
    current_task = person_dir / "state" / "current_task.md"
    if not current_task.exists():
        current_task.write_text("status: idle\n", encoding="utf-8")

    pending = person_dir / "state" / "pending.md"
    if not pending.exists():
        pending.write_text("", encoding="utf-8")


def _should_create_bootstrap(person_dir: Path) -> bool:
    """Check if bootstrap.md should be created for this person.

    Bootstrap is needed when:
    - identity.md doesn't exist
    - identity.md exists but is empty or contains "未定義"
    - character_sheet.md exists (will be processed by bootstrap)

    Bootstrap is NOT needed when:
    - identity.md exists and is fully defined (e.g., from template)

    Returns:
        True if bootstrap.md should be created, False otherwise.
    """
    identity = person_dir / "identity.md"
    if not identity.exists():
        return True

    content = identity.read_text(encoding="utf-8")
    if not content.strip() or "未定義" in content:
        return True

    if (person_dir / "character_sheet.md").exists():
        return True

    return False


def _place_bootstrap(person_dir: Path) -> None:
    """Copy the bootstrap template into the person directory if needed."""
    if not _should_create_bootstrap(person_dir):
        logger.debug("Skipping bootstrap for %s (identity already defined)", person_dir)
        return

    if BOOTSTRAP_TEMPLATE.exists():
        shutil.copy2(BOOTSTRAP_TEMPLATE, person_dir / "bootstrap.md")
        logger.debug("Placed bootstrap.md in %s", person_dir)


def _place_send_script(person_dir: Path) -> None:
    """Place the send wrapper script in person_dir if not already present."""
    src = BLANK_TEMPLATE_DIR / "send"
    dst = person_dir / "send"
    if src.exists() and not dst.exists():
        shutil.copy2(src, dst)
        dst.chmod(0o755)
        logger.debug("Placed send script in %s", person_dir)


def ensure_send_scripts(persons_dir: Path) -> None:
    """Ensure every person directory has the send wrapper script.

    Iterates all subdirectories under *persons_dir* that contain an
    ``identity.md`` file (i.e. valid person directories) and calls
    :func:`_place_send_script` for each.  Existing send scripts are
    never overwritten.

    This should be called during server startup so that persons created
    before the send script feature was added also get the script.

    Args:
        persons_dir: Runtime persons directory (e.g. ``~/.animaworks/persons/``).
    """
    if not persons_dir.exists():
        return
    for person_dir in sorted(persons_dir.iterdir()):
        if person_dir.is_dir() and (person_dir / "identity.md").exists():
            _place_send_script(person_dir)


def validate_person_name(name: str) -> str | None:
    """Validate a person name.  Returns error message or None if valid."""
    if not name:
        return "Name cannot be empty"
    if not re.match(r"^[a-z][a-z0-9_-]*$", name):
        return "Name must be lowercase alphanumeric (a-z, 0-9, -, _), starting with a letter"
    if name.startswith("_"):
        return "Name cannot start with underscore"
    return None