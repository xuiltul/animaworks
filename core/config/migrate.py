# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Migrate legacy config.md files to unified config.json.

Also provides cron.md migration from Japanese text schedules to standard
5-field cron expressions, and Person→Anima rename migration for runtime data.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

logger = logging.getLogger("animaworks.config_migrate")

# ── Cron migration constants ─────────────────────────────

# Japanese day-of-week to cron numeric (0=Sun, 1=Mon, ... 6=Sat)
_JP_DAY_TO_CRON: dict[str, str] = {
    "月": "1",
    "火": "2",
    "水": "3",
    "木": "4",
    "金": "5",
    "土": "6",
    "日": "0",
    "月曜": "1",
    "火曜": "2",
    "水曜": "3",
    "木曜": "4",
    "金曜": "5",
    "土曜": "6",
    "日曜": "0",
}


def _parse_config_md(path: Path) -> dict[str, str]:
    """Parse a legacy config.md file and return key-value pairs."""
    raw = path.read_text(encoding="utf-8")
    # Ignore 備考/設定例 sections
    for marker in ("## 備考", "### 設定例"):
        idx = raw.find(marker)
        if idx != -1:
            raw = raw[:idx]

    result = {}
    for m in re.finditer(r"^-\s*(\w+)\s*:\s*(.+)$", raw, re.MULTILINE):
        result[m.group(1).strip()] = m.group(2).strip()
    return result


def _env_name_to_credential_name(env_name: str) -> str:
    """Derive a credential name from an env var name.

    ANTHROPIC_API_KEY -> anthropic
    ANTHROPIC_API_KEY_MYNAME -> anthropic_myname
    OLLAMA_API_KEY -> ollama
    """
    name = env_name.lower()
    # Remove _api_key suffix/infix
    name = re.sub(r"_api_key$", "", name)
    name = re.sub(r"_api_key_", "_", name)
    return name or "default"


def migrate_to_config_json(data_dir: Path) -> None:
    """Build config.json from existing config.md files and environment variables.

    Scans animas_dir for config.md files, parses them, collects credentials,
    and writes a unified config.json.
    """
    from core.config.models import (
        AnimaWorksConfig,
        CredentialConfig,
        AnimaModelConfig,
        save_config,
    )

    animas_dir = data_dir / "animas"
    config = AnimaWorksConfig()

    if not animas_dir.exists():
        save_config(config, data_dir / "config.json")
        return

    seen_credentials: dict[str, CredentialConfig] = {}

    for anima_dir in sorted(animas_dir.iterdir()):
        if not anima_dir.is_dir():
            continue
        config_md = anima_dir / "config.md"
        if not config_md.exists():
            continue

        logger.info("Migrating config.md for anima: %s", anima_dir.name)
        parsed = _parse_config_md(config_md)

        # Determine credential
        api_key_env = parsed.get("api_key_env", "ANTHROPIC_API_KEY")
        base_url = parsed.get("api_base_url", "")
        cred_name = _env_name_to_credential_name(api_key_env)

        if cred_name not in seen_credentials:
            api_key_value = os.environ.get(api_key_env, "")
            seen_credentials[cred_name] = CredentialConfig(
                api_key=api_key_value,
                base_url=base_url or None,
            )

        # Build anima config (only override non-default values)
        anima_cfg = AnimaModelConfig(
            model=parsed.get("model") or None,
            fallback_model=parsed.get("fallback_model") or None,
            max_tokens=int(parsed["max_tokens"]) if "max_tokens" in parsed else None,
            max_turns=int(parsed["max_turns"]) if "max_turns" in parsed else None,
            credential=cred_name,
        )
        config.animas[anima_dir.name] = anima_cfg

    config.credentials = seen_credentials

    # Ensure at least an "anthropic" credential exists
    if "anthropic" not in config.credentials:
        config.credentials["anthropic"] = CredentialConfig(
            api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
        )

    save_config(config, data_dir / "config.json")
    logger.info(
        "Migration complete: %d animas, %d credentials -> %s",
        len(config.animas),
        len(config.credentials),
        data_dir / "config.json",
    )


# ── Cron format migration ────────────────────────────────


def _convert_jp_schedule_to_cron(schedule_text: str) -> str | None:
    """Convert a Japanese schedule string to a 5-field cron expression.

    Args:
        schedule_text: Japanese schedule like ``"毎日 9:00 JST"``

    Returns:
        Cron expression string, or None if the pattern cannot be converted
        automatically (e.g. bi-weekly, nth weekday, last day of month).
    """
    s = schedule_text.strip()
    # Remove trailing timezone markers (JST, UTC, etc.)
    s = re.sub(r"\s+[A-Z]{2,4}$", "", s)

    # X分毎 → */X * * * *
    m = re.match(r"(\d+)分毎", s)
    if m:
        return f"*/{m.group(1)} * * * *"

    # X時間毎 → 0 */X * * *
    m = re.match(r"(\d+)時間毎", s)
    if m:
        return f"0 */{m.group(1)} * * *"

    # 毎日 HH:MM → MM HH * * *
    m = re.match(r"毎日\s+(\d{1,2}):(\d{2})", s)
    if m:
        return f"{int(m.group(2))} {int(m.group(1))} * * *"

    # 平日 HH:MM → MM HH * * 1-5
    m = re.match(r"平日\s+(\d{1,2}):(\d{2})", s)
    if m:
        return f"{int(m.group(2))} {int(m.group(1))} * * 1-5"

    # 毎週X曜 HH:MM → MM HH * * N
    m = re.match(r"毎週(.+?曜?)\s+(\d{1,2}):(\d{2})", s)
    if m:
        day_key = m.group(1)
        day_num = _JP_DAY_TO_CRON.get(day_key)
        if day_num:
            return f"{int(m.group(3))} {int(m.group(2))} * * {day_num}"

    # 毎月DD日 HH:MM → MM HH DD * *
    m = re.match(r"毎月(\d{1,2})日\s+(\d{1,2}):(\d{2})", s)
    if m:
        return f"{int(m.group(3))} {int(m.group(2))} {int(m.group(1))} * *"

    # Unconvertible patterns: 隔週, 毎月最終日, 第NX曜
    # Return None — caller should handle gracefully
    return None


def _is_already_migrated(content: str) -> bool:
    """Check whether a cron.md file already uses the new ``schedule:`` format.

    Returns True if any ``schedule:`` directive is found in the body
    (not inside an HTML comment).
    """
    stripped = re.sub(r"<!--.*?-->", "", content, flags=re.DOTALL)
    return bool(re.search(r"^\s*schedule:\s*", stripped, re.MULTILINE))


def migrate_cron_format(anima_dir: Path) -> bool:
    """Migrate an anima's cron.md from Japanese text schedules to cron expressions.

    Reads the existing cron.md, converts each ``## Title（Schedule）`` section
    to the new format with ``schedule: <cron-expression>``, and writes it back.

    Args:
        anima_dir: Path to the anima directory containing cron.md.

    Returns:
        True if migration was performed, False if no migration was needed
        (file missing, already migrated, or no convertible schedules).
    """
    cron_md = anima_dir / "cron.md"
    if not cron_md.exists():
        return False

    content = cron_md.read_text(encoding="utf-8")
    if not content.strip():
        return False

    # Skip if already migrated
    if _is_already_migrated(content):
        logger.info("cron.md already migrated for %s, skipping", anima_dir.name)
        return False

    # Preserve HTML comments by extracting and restoring them
    # We process the raw content line-by-line, handling comment blocks
    output_lines: list[str] = []
    migrated_any = False
    in_comment = False
    section_buffer: list[str] = []
    section_title = ""
    section_schedule = ""

    def _flush_section() -> None:
        """Flush the accumulated section buffer to output_lines."""
        nonlocal migrated_any
        if not section_title:
            output_lines.extend(section_buffer)
            return

        # Try to convert the schedule
        cron_expr = _convert_jp_schedule_to_cron(section_schedule) if section_schedule else None

        if cron_expr:
            # Write new format: ## Title (without schedule in parens)
            output_lines.append(f"## {section_title}")
            output_lines.append(f"schedule: {cron_expr}")
            # Copy body lines (skip empty leading lines)
            for bline in section_buffer:
                output_lines.append(bline)
            migrated_any = True
        elif section_schedule:
            # Unconvertible schedule — keep original title with comment
            output_lines.append(f"## {section_title}")
            output_lines.append(f"<!-- MIGRATION NOTE: could not auto-convert '{section_schedule}' to cron expression -->")
            # Copy body lines as-is
            for bline in section_buffer:
                output_lines.append(bline)
            logger.warning(
                "Could not auto-convert schedule '%s' for task '%s' in %s",
                section_schedule, section_title, anima_dir.name,
            )
        else:
            # No schedule at all — keep as-is
            output_lines.append(f"## {section_title}")
            for bline in section_buffer:
                output_lines.append(bline)

    for line in content.splitlines():
        # Track HTML comment state (simple: single-line open/close)
        if "<!--" in line and "-->" not in line:
            in_comment = True
            output_lines.append(line)
            continue
        if in_comment:
            output_lines.append(line)
            if "-->" in line:
                in_comment = False
            continue
        # Single-line comment — pass through
        if "<!--" in line and "-->" in line:
            # Could be a full line comment — check if it's wrapping a section
            # Just pass through
            if not section_title:
                output_lines.append(line)
            else:
                section_buffer.append(line)
            continue

        if line.startswith("## "):
            # Flush previous section
            _flush_section()
            section_buffer = []

            header = line[3:].strip()
            sm = re.search(r"[（(](.+?)[）)]", header)
            if sm:
                section_schedule = sm.group(1)
                section_title = header[: header.find("（" if "（" in header else "(")].strip()
            else:
                section_title = header
                section_schedule = ""
        elif section_title:
            section_buffer.append(line)
        else:
            output_lines.append(line)

    # Flush last section
    _flush_section()

    if not migrated_any:
        return False

    cron_md.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    logger.info("Migrated cron.md for %s", anima_dir.name)
    return True


def migrate_all_cron(animas_dir: Path) -> int:
    """Migrate cron.md for all animas in the given directory.

    Args:
        animas_dir: Path to the ``animas/`` directory containing
            per-anima subdirectories.

    Returns:
        Number of animas whose cron.md was successfully migrated.
    """
    if not animas_dir.exists():
        logger.info("Animas directory does not exist: %s", animas_dir)
        return 0

    count = 0
    for anima_dir in sorted(animas_dir.iterdir()):
        if not anima_dir.is_dir():
            continue
        try:
            if migrate_cron_format(anima_dir):
                count += 1
        except Exception:
            logger.exception("Failed to migrate cron.md for %s", anima_dir.name)

    logger.info("Cron migration complete: %d animas migrated", count)
    return count


# ── Person -> Anima rename migration ──────────────────────

# Old directory/key names (pre-rename). Stored as constants so that
# automated rename tools do not accidentally rewrite them.
_OLD_DIR_NAME = "persons"
_NEW_DIR_NAME = "animas"
_OLD_DEFAULTS_KEY = "person_defaults"
_NEW_DEFAULTS_KEY = "anima_defaults"
_OLD_COLLECTION_KEY = "persons"
_NEW_COLLECTION_KEY = "animas"
_OLD_TOOL_NAME = "create_" + "person"  # noqa: intentionally split
_NEW_TOOL_NAME = "create_anima"
_OLD_ENV_VAR = "ANIMAWORKS_" + "PERSON_DIR"  # noqa: intentionally split
_NEW_ENV_VAR = "ANIMAWORKS_ANIMA_DIR"


def _needs_person_to_anima_migration(data_dir: Path) -> bool:
    """Check whether the Person-to-Anima rename migration is needed.

    Returns True if the old directory exists (and the new one does not),
    or if config.json contains old key names.
    """
    old_dir = data_dir / _OLD_DIR_NAME
    new_dir = data_dir / _NEW_DIR_NAME

    # Directory-level check
    if old_dir.exists() and not new_dir.exists():
        return True

    # Config key check
    config_path = data_dir / "config.json"
    if config_path.is_file():
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
            if _OLD_DEFAULTS_KEY in data or _OLD_COLLECTION_KEY in data:
                return True
        except (json.JSONDecodeError, OSError):
            pass

    return False


def _migrate_config_keys(config_path: Path) -> bool:
    """Rename old config keys to new ones in config.json.

    Renames ``person_defaults`` to ``anima_defaults`` and
    ``persons`` to ``animas``.

    Args:
        config_path: Path to config.json.

    Returns:
        True if any keys were renamed, False otherwise.
    """
    if not config_path.is_file():
        return False

    try:
        text = config_path.read_text(encoding="utf-8")
        data = json.loads(text)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read config.json for key migration: %s", exc)
        return False

    changed = False

    if _OLD_DEFAULTS_KEY in data and _NEW_DEFAULTS_KEY not in data:
        data[_NEW_DEFAULTS_KEY] = data.pop(_OLD_DEFAULTS_KEY)
        changed = True
        logger.info(
            "Renamed config key: %s -> %s",
            _OLD_DEFAULTS_KEY,
            _NEW_DEFAULTS_KEY,
        )

    if _OLD_COLLECTION_KEY in data and _NEW_COLLECTION_KEY not in data:
        data[_NEW_COLLECTION_KEY] = data.pop(_OLD_COLLECTION_KEY)
        changed = True
        logger.info(
            "Renamed config key: %s -> %s",
            _OLD_COLLECTION_KEY,
            _NEW_COLLECTION_KEY,
        )

    if changed:
        new_text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
        config_path.write_text(new_text, encoding="utf-8")
        logger.info("Updated config.json with renamed keys")

    return changed


def _migrate_skill_references(animas_dir: Path) -> int:
    """Replace old tool name with new one in skill files.

    Searches all ``*.md`` files under ``animas/*/skills/`` and performs
    the text replacement (``create_person`` -> ``create_anima``).

    Args:
        animas_dir: Path to the ``animas/`` directory.

    Returns:
        Number of files updated.
    """
    if not animas_dir.exists():
        return 0

    count = 0
    for skill_file in animas_dir.glob("*/skills/*.md"):
        try:
            content = skill_file.read_text(encoding="utf-8")
            if _OLD_TOOL_NAME in content:
                updated = content.replace(_OLD_TOOL_NAME, _NEW_TOOL_NAME)
                skill_file.write_text(updated, encoding="utf-8")
                count += 1
                logger.info(
                    "Updated skill file: %s (%s -> %s)",
                    skill_file,
                    _OLD_TOOL_NAME,
                    _NEW_TOOL_NAME,
                )
        except OSError as exc:
            logger.warning("Failed to update skill file %s: %s", skill_file, exc)

    return count


def _migrate_env_var_references(animas_dir: Path) -> int:
    """Replace old env var name with new one in runtime files.

    Searches Markdown and shell script files under the animas directory
    for the old environment variable name and replaces with the new one.

    Args:
        animas_dir: Path to the ``animas/`` directory.

    Returns:
        Number of files updated.
    """
    if not animas_dir.exists():
        return 0

    count = 0
    patterns = ["**/*.md", "**/*.sh"]
    seen: set[Path] = set()

    for pattern in patterns:
        for fpath in animas_dir.glob(pattern):
            if fpath in seen:
                continue
            seen.add(fpath)
            try:
                content = fpath.read_text(encoding="utf-8")
                if _OLD_ENV_VAR in content:
                    updated = content.replace(_OLD_ENV_VAR, _NEW_ENV_VAR)
                    fpath.write_text(updated, encoding="utf-8")
                    count += 1
                    logger.info(
                        "Updated env var reference: %s (%s -> %s)",
                        fpath,
                        _OLD_ENV_VAR,
                        _NEW_ENV_VAR,
                    )
            except OSError as exc:
                logger.warning("Failed to update file %s: %s", fpath, exc)

    return count


def migrate_person_to_anima(data_dir: Path) -> None:
    """Migrate runtime data from Person naming to Anima naming.

    Performs the following steps (all idempotent):

    1. Rename ``persons/`` directory to ``animas/``
    2. Rename ``logs/persons/`` directory to ``logs/animas/``
    3. Rename config.json keys (old defaults/collection keys to new ones)
    4. Update tool name references in skill files
    5. Update environment variable references in runtime files

    Args:
        data_dir: The AnimaWorks runtime data directory
            (e.g. ``~/.animaworks``).
    """
    if not _needs_person_to_anima_migration(data_dir):
        logger.debug("Person-to-Anima migration not needed")
        return

    logger.info("Starting Person -> Anima rename migration in %s", data_dir)

    # Step 1: Rename old dir -> new dir
    old_dir = data_dir / _OLD_DIR_NAME
    new_dir = data_dir / _NEW_DIR_NAME
    if old_dir.exists() and not new_dir.exists():
        try:
            old_dir.rename(new_dir)
            logger.info("Renamed directory: %s -> %s", old_dir, new_dir)
        except OSError as exc:
            logger.warning(
                "Failed to rename %s -> %s: %s", old_dir, new_dir, exc
            )

    # Step 2: Rename logs/old -> logs/new
    log_old_dir = data_dir / "logs" / _OLD_DIR_NAME
    log_new_dir = data_dir / "logs" / _NEW_DIR_NAME
    if log_old_dir.exists() and not log_new_dir.exists():
        try:
            log_old_dir.rename(log_new_dir)
            logger.info(
                "Renamed log directory: %s -> %s",
                log_old_dir,
                log_new_dir,
            )
        except OSError as exc:
            logger.warning(
                "Failed to rename %s -> %s: %s",
                log_old_dir,
                log_new_dir,
                exc,
            )

    # Step 3: Rename config.json keys
    config_path = data_dir / "config.json"
    try:
        _migrate_config_keys(config_path)
    except Exception:
        logger.exception("Failed to migrate config.json keys")

    # Step 4: Update skill file references
    try:
        skill_count = _migrate_skill_references(new_dir)
        if skill_count:
            logger.info("Updated %d skill file(s)", skill_count)
    except Exception:
        logger.exception("Failed to update skill file references")

    # Step 5: Update environment variable references
    try:
        env_count = _migrate_env_var_references(new_dir)
        if env_count:
            logger.info("Updated %d file(s) with env var references", env_count)
    except Exception:
        logger.exception("Failed to update env var references")

    logger.info("Person -> Anima rename migration complete")
