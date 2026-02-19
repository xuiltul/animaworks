# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Anima creation factory: create new Digital Animas from templates, blank, or MD files."""

from __future__ import annotations

import json
import logging
import re
import shutil
from pathlib import Path
from typing import Any

from core.paths import TEMPLATES_DIR

logger = logging.getLogger("animaworks.anima_factory")

ANIMA_TEMPLATES_DIR = TEMPLATES_DIR / "anima_templates"
BLANK_TEMPLATE_DIR = ANIMA_TEMPLATES_DIR / "_blank"
BOOTSTRAP_TEMPLATE = TEMPLATES_DIR / "bootstrap.md"
ROLES_DIR = TEMPLATES_DIR / "roles"
VALID_ROLES = frozenset({"engineer", "researcher", "manager", "writer", "ops", "general"})

# Subdirectories every anima needs at runtime
_RUNTIME_SUBDIRS = [
    "episodes",
    "knowledge",
    "procedures",
    "skills",
    "state",
    "shortterm",
    "shortterm/archive",
]


def list_anima_templates() -> list[str]:
    """List available anima templates (excluding _blank)."""
    if not ANIMA_TEMPLATES_DIR.exists():
        return []
    return [
        d.name
        for d in sorted(ANIMA_TEMPLATES_DIR.iterdir())
        if d.is_dir() and not d.name.startswith("_")
    ]


def create_from_template(
    animas_dir: Path, template_name: str, *, anima_name: str | None = None
) -> Path:
    """Create an anima by copying a named template.

    Args:
        animas_dir: Runtime animas directory (~/.animaworks/animas/).
        template_name: Template directory name under anima_templates/.
        anima_name: Override the directory name.  Defaults to template_name.

    Returns:
        Path to the created anima directory.
    """
    template_dir = ANIMA_TEMPLATES_DIR / template_name
    if not template_dir.exists():
        raise FileNotFoundError(f"Template not found: {template_name}")

    name = anima_name or template_name
    anima_dir = animas_dir / name
    if anima_dir.exists():
        raise FileExistsError(f"Anima already exists: {name}")

    shutil.copytree(template_dir, anima_dir)
    _ensure_runtime_subdirs(anima_dir)
    _init_state_files(anima_dir)
    _place_bootstrap(anima_dir)
    _place_send_script(anima_dir)
    _place_board_script(anima_dir)
    _ensure_status_json(anima_dir)

    logger.info("Created anima '%s' from template '%s'", name, template_name)
    return anima_dir


def create_blank(animas_dir: Path, name: str) -> Path:
    """Create a blank anima with skeleton files.

    The {name} placeholder in skeleton files is replaced with the actual name.
    Copies the entire _blank template tree (including subdirectories like skills/).

    Args:
        animas_dir: Runtime animas directory.
        name: Anima name (lowercase alphanumeric).

    Returns:
        Path to the created anima directory.
    """
    anima_dir = animas_dir / name
    if anima_dir.exists():
        raise FileExistsError(f"Anima already exists: {name}")

    try:
        if BLANK_TEMPLATE_DIR.exists():
            shutil.copytree(BLANK_TEMPLATE_DIR, anima_dir)
            # Replace {name} placeholder in all markdown files
            for md_file in anima_dir.rglob("*.md"):
                content = md_file.read_text(encoding="utf-8")
                if "{name}" in content:
                    md_file.write_text(
                        content.replace("{name}", name), encoding="utf-8"
                    )
        else:
            anima_dir.mkdir(parents=True, exist_ok=True)

        _ensure_runtime_subdirs(anima_dir)
        _init_state_files(anima_dir)
        _place_bootstrap(anima_dir)
        _place_send_script(anima_dir)
        _place_board_script(anima_dir)
        # Create a minimal status.json so the reconciliation loop
        # recognises this anima as a valid on-disk entry.
        _ensure_status_json(anima_dir)
    except Exception:
        logger.error("Failed to create blank anima '%s'; rolling back", name)
        shutil.rmtree(anima_dir, ignore_errors=True)
        raise

    logger.info("Created blank anima '%s'", name)
    return anima_dir


def create_from_md(
    animas_dir: Path,
    md_path: Path | None = None,
    name: str | None = None,
    *,
    content: str | None = None,
    supervisor: str | None = None,
    role: str | None = None,
) -> Path:
    """Create an anima from an MD character-sheet file or content string.

    The MD content is validated, then placed as ``character_sheet.md`` in the
    new anima directory.  Sections in the sheet are applied to identity.md,
    injection.md, permissions.md, and status.json.

    On any failure after the anima directory has been created the directory
    is rolled back (removed) so no partial state is left behind.

    Args:
        animas_dir: Runtime animas directory.
        md_path: Path to the source MD file.  Either *md_path* or *content*
            must be provided.
        name: Anima name.  If None, extracted from MD content.
        content: Character sheet content as a string.  If provided, *md_path*
            is ignored.
        supervisor: Explicit supervisor name.  If given, overrides the value
            parsed from the character sheet's ``基本情報`` table.

    Returns:
        Path to the created anima directory.

    Raises:
        FileNotFoundError: If *md_path* is given but does not exist.
        ValueError: If no content source is provided, the character sheet is
            invalid, or the name cannot be determined.
    """
    if content is not None:
        md_content = content
    elif md_path is not None:
        if not md_path.exists():
            raise FileNotFoundError(f"MD file not found: {md_path}")
        md_content = md_path.read_text(encoding="utf-8")
    else:
        raise ValueError("Either md_path or content must be provided")

    # Validate before creating anything on disk
    _validate_character_sheet(md_content)

    if not name:
        name = _extract_name_from_md(md_content)
    if not name:
        raise ValueError(
            "Could not extract anima name from MD file. "
            "Add a '# Character: name' heading or specify --name."
        )

    # Create blank skeleton first, then layer character sheet on top
    anima_dir = create_blank(animas_dir, name)
    try:
        (anima_dir / "character_sheet.md").write_text(md_content, encoding="utf-8")
        _apply_defaults_from_sheet(anima_dir, md_content)
        # Apply role template defaults
        resolved_role = role or "general"
        _apply_role_defaults(anima_dir, resolved_role)
        _create_status_json(
            anima_dir,
            _parse_character_sheet_info(md_content),
            supervisor_override=supervisor,
            role=resolved_role,
        )
    except Exception:
        logger.error(
            "Failed to set up anima '%s' from MD file; rolling back", name
        )
        shutil.rmtree(anima_dir, ignore_errors=True)
        raise

    logger.info("Created anima '%s' from MD file '%s'", name, md_path)
    return anima_dir


def _extract_name_from_md(content: str) -> str | None:
    """Try to extract an anima name from MD content.

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


def _ensure_status_json(anima_dir: Path) -> None:
    """Create a minimal status.json if one does not already exist.

    This is called by :func:`create_blank` so that every anima directory
    has a valid status.json from the start.  :func:`_create_status_json`
    (used by :func:`create_from_md`) may overwrite it later with richer
    metadata parsed from the character sheet.
    """
    status_path = anima_dir / "status.json"
    if status_path.exists():
        return
    status = {"enabled": True}
    status_path.write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.debug("Created minimal status.json in %s", anima_dir)


def _create_status_json(
    anima_dir: Path,
    info: dict[str, str],
    *,
    supervisor_override: str | None = None,
    role: str = "general",
) -> None:
    """Create status.json in *anima_dir* from parsed character-sheet info.

    The JSON contains supervisor, role, execution mode, model, and credential
    extracted from the ``基本情報`` table.  Role defaults from
    ``templates/roles/<role>/defaults.json`` are merged in for model config
    fields; character sheet values take priority.

    Args:
        anima_dir: Target anima directory.
        info: Dict returned by :func:`_parse_character_sheet_info`.
        supervisor_override: If given, takes priority over the value in
            *info* (the ``上司`` field from the character sheet).
        role: Role name used to load defaults.json.
    """
    # Load role defaults
    role_defaults: dict[str, Any] = {}
    defaults_path = ROLES_DIR / role / "defaults.json"
    if defaults_path.is_file():
        try:
            role_defaults = json.loads(defaults_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.debug("Failed to load role defaults for %s", role)

    if supervisor_override:
        supervisor = supervisor_override
    else:
        supervisor_raw = info.get("上司", "")
        supervisor = "" if supervisor_raw in ("(なし)", "なし", "-", "") else supervisor_raw

    status: dict[str, object] = {
        "supervisor": supervisor,
        "role": role,
        "enabled": True,
    }
    # Only write execution_mode when explicitly specified in the character sheet.
    # When omitted, resolve_execution_mode() falls through to
    # DEFAULT_MODEL_MODE_PATTERNS (e.g. "claude-*" → "A1").
    explicit_mode = info.get("実行モード")
    if explicit_mode:
        status["execution_mode"] = explicit_mode

    # Merge role defaults (model config fields)
    for key in ("model", "context_threshold", "max_turns", "max_chains",
                "conversation_history_threshold"):
        if key in role_defaults:
            status[key] = role_defaults[key]

    # Character sheet model/credential override role defaults if specified
    sheet_model = info.get("モデル", "")
    if sheet_model:
        status["model"] = sheet_model
    sheet_cred = info.get("credential", "")
    if sheet_cred:
        status["credential"] = sheet_cred

    (anima_dir / "status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.debug("Created status.json in %s (role=%s)", anima_dir, role)


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


def _apply_defaults_from_sheet(anima_dir: Path, md_content: str) -> None:
    """Apply character-sheet sections to anima files, keeping template defaults.

    For sections marked ``[省略可]`` in the sheet, if they are omitted or
    empty the existing template files (heartbeat.md, cron.md, permissions.md)
    are left untouched.

    Sections that **are** present overwrite the corresponding file:
        - ``## 人格``          → ``identity.md``
        - ``## 役割・行動方針`` → ``injection.md``
        - ``## 権限``          → ``permissions.md``

    Args:
        anima_dir: Target anima directory (already populated by create_blank).
        md_content: Full markdown content of the character sheet.
    """
    # Identity
    personality = _extract_section_content(md_content, "人格")
    if personality:
        (anima_dir / "identity.md").write_text(
            personality + "\n", encoding="utf-8"
        )
        logger.debug("Wrote identity.md from character sheet for %s", anima_dir.name)

    # Injection
    injection = _extract_section_content(md_content, "役割・行動方針")
    if injection:
        (anima_dir / "injection.md").write_text(
            injection + "\n", encoding="utf-8"
        )
        logger.debug("Wrote injection.md from character sheet for %s", anima_dir.name)

    # Permissions
    permissions = _extract_section_content(md_content, "権限")
    if permissions:
        (anima_dir / "permissions.md").write_text(
            permissions + "\n", encoding="utf-8"
        )
        logger.debug("Wrote permissions.md from character sheet for %s", anima_dir.name)


def _apply_role_defaults(anima_dir: Path, role: str) -> None:
    """Apply role template files to the anima directory.

    Copies permissions.md and specialty_prompt.md from the role template,
    and merges defaults.json values into status.json.

    Args:
        anima_dir: Target anima directory.
        role: Role name (must be in VALID_ROLES).
    """
    if role not in VALID_ROLES:
        logger.warning("Unknown role '%s', falling back to 'general'", role)
        role = "general"

    role_dir = ROLES_DIR / role
    if not role_dir.exists():
        logger.warning("Role template directory not found: %s", role_dir)
        return

    # Copy permissions.md (overwrite blank template)
    perm_src = role_dir / "permissions.md"
    if perm_src.exists():
        perm_content = perm_src.read_text(encoding="utf-8")
        # Replace {name} placeholder
        if "{name}" in perm_content:
            perm_content = perm_content.replace("{name}", anima_dir.name)
        (anima_dir / "permissions.md").write_text(perm_content, encoding="utf-8")

    # Copy specialty_prompt.md
    spec_src = role_dir / "specialty_prompt.md"
    if spec_src.exists():
        (anima_dir / "specialty_prompt.md").write_text(
            spec_src.read_text(encoding="utf-8"), encoding="utf-8"
        )

    logger.debug("Applied role '%s' defaults to %s", role, anima_dir.name)


# ── Runtime Helpers ──────────


def _ensure_runtime_subdirs(anima_dir: Path) -> None:
    """Create runtime-only subdirectories."""
    for subdir in _RUNTIME_SUBDIRS:
        (anima_dir / subdir).mkdir(parents=True, exist_ok=True)


def _init_state_files(anima_dir: Path) -> None:
    """Create initial state files if they don't exist."""
    current_task = anima_dir / "state" / "current_task.md"
    if not current_task.exists():
        current_task.write_text("status: idle\n", encoding="utf-8")

    pending = anima_dir / "state" / "pending.md"
    if not pending.exists():
        pending.write_text("", encoding="utf-8")


def _should_create_bootstrap(anima_dir: Path) -> bool:
    """Check if bootstrap.md should be created for this anima.

    Bootstrap is needed when:
    - identity.md doesn't exist
    - identity.md exists but is empty or contains "未定義"
    - character_sheet.md exists (will be processed by bootstrap)

    Bootstrap is NOT needed when:
    - identity.md exists and is fully defined (e.g., from template)

    Returns:
        True if bootstrap.md should be created, False otherwise.
    """
    identity = anima_dir / "identity.md"
    if not identity.exists():
        return True

    content = identity.read_text(encoding="utf-8")
    if not content.strip() or "未定義" in content:
        return True

    if (anima_dir / "character_sheet.md").exists():
        return True

    return False


def _place_bootstrap(anima_dir: Path) -> None:
    """Copy the bootstrap template into the anima directory if needed."""
    if not _should_create_bootstrap(anima_dir):
        logger.debug("Skipping bootstrap for %s (identity already defined)", anima_dir)
        return

    if BOOTSTRAP_TEMPLATE.exists():
        shutil.copy2(BOOTSTRAP_TEMPLATE, anima_dir / "bootstrap.md")
        logger.debug("Placed bootstrap.md in %s", anima_dir)


def _place_send_script(anima_dir: Path) -> None:
    """Place (or update) the send wrapper script in anima_dir from template."""
    src = BLANK_TEMPLATE_DIR / "send"
    dst = anima_dir / "send"
    if not src.exists():
        return
    shutil.copy2(src, dst)
    dst.chmod(0o755)
    logger.debug("Updated send script in %s", anima_dir)


def _place_board_script(anima_dir: Path) -> None:
    """Place (or update) the board wrapper script in anima_dir from template."""
    src = BLANK_TEMPLATE_DIR / "board"
    dst = anima_dir / "board"
    if not src.exists():
        return
    shutil.copy2(src, dst)
    dst.chmod(0o755)
    logger.debug("Updated board script in %s", anima_dir)


def ensure_send_scripts(animas_dir: Path) -> None:
    """Ensure every anima directory has the send wrapper script.

    Iterates all subdirectories under *animas_dir* that contain an
    ``identity.md`` file (i.e. valid anima directories) and calls
    :func:`_place_send_script` for each.  Existing send scripts are overwritten to ensure
    they match the latest template.

    This should be called during server startup so that animas created
    before the send script feature was added also get the script.

    Args:
        animas_dir: Runtime animas directory (e.g. ``~/.animaworks/animas/``).
    """
    if not animas_dir.exists():
        return
    for anima_dir in sorted(animas_dir.iterdir()):
        if anima_dir.is_dir() and (anima_dir / "identity.md").exists():
            _place_send_script(anima_dir)


def ensure_board_scripts(animas_dir: Path) -> None:
    """Ensure every anima directory has the board wrapper script.

    Iterates all subdirectories under *animas_dir* that contain an
    ``identity.md`` file (i.e. valid anima directories) and calls
    :func:`_place_board_script` for each.  Existing board scripts are
    overwritten to ensure they match the latest template.

    This should be called during server startup so that animas created
    before the board script feature was added also get the script.

    Args:
        animas_dir: Runtime animas directory (e.g. ``~/.animaworks/animas/``).
    """
    if not animas_dir.exists():
        return
    for anima_dir in sorted(animas_dir.iterdir()):
        if anima_dir.is_dir() and (anima_dir / "identity.md").exists():
            _place_board_script(anima_dir)


def validate_anima_name(name: str) -> str | None:
    """Validate an anima name.  Returns error message or None if valid."""
    if not name:
        return "Name cannot be empty"
    if not re.match(r"^[a-z][a-z0-9_-]*$", name):
        return "Name must be lowercase alphanumeric (a-z, 0-9, -, _), starting with a letter"
    if name.startswith("_"):
        return "Name cannot start with underscore"
    return None