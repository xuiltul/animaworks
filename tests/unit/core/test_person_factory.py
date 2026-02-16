"""Unit tests for core/person_factory.py — person creation factory."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from core.person_factory import (
    BLANK_TEMPLATE_DIR,
    BOOTSTRAP_TEMPLATE,
    PERSON_TEMPLATES_DIR,
    _RUNTIME_SUBDIRS,
    _apply_defaults_from_sheet,
    _create_status_json,
    _ensure_runtime_subdirs,
    _extract_name_from_md,
    _extract_section_content,
    _init_state_files,
    _parse_character_sheet_info,
    _place_bootstrap,
    _place_send_script,
    _should_create_bootstrap,
    _validate_character_sheet,
    create_blank,
    create_from_md,
    create_from_template,
    list_person_templates,
    validate_person_name,
)


# ── validate_person_name ──────────────────────────────────


class TestValidatePersonName:
    def test_valid_names(self):
        assert validate_person_name("alice") is None
        assert validate_person_name("bob-smith") is None
        assert validate_person_name("charlie_01") is None
        assert validate_person_name("a") is None

    def test_empty_name(self):
        assert validate_person_name("") is not None

    def test_uppercase_rejected(self):
        assert validate_person_name("Alice") is not None

    def test_starts_with_number(self):
        assert validate_person_name("123abc") is not None

    def test_starts_with_underscore(self):
        assert validate_person_name("_test") is not None

    def test_special_chars(self):
        assert validate_person_name("a.b") is not None
        assert validate_person_name("a b") is not None
        assert validate_person_name("a@b") is not None


# ── _extract_name_from_md ─────────────────────────────────


class TestExtractNameFromMd:
    def test_character_heading(self):
        assert _extract_name_from_md("# Character: Hinata") == "hinata"

    def test_simple_heading(self):
        assert _extract_name_from_md("# Sakura") == "sakura"

    def test_eimei_pattern(self):
        assert _extract_name_from_md("英名 Hinata") == "hinata"

    def test_no_match(self):
        assert _extract_name_from_md("No heading here") is None

    def test_multiline(self):
        content = "Some intro\n# Character: Alice\nMore text"
        assert _extract_name_from_md(content) == "alice"


# ── _ensure_runtime_subdirs ───────────────────────────────


class TestEnsureRuntimeSubdirs:
    def test_creates_all_subdirs(self, tmp_path):
        person_dir = tmp_path / "person"
        person_dir.mkdir()
        _ensure_runtime_subdirs(person_dir)
        for subdir in _RUNTIME_SUBDIRS:
            assert (person_dir / subdir).is_dir()


# ── _init_state_files ─────────────────────────────────────


class TestInitStateFiles:
    def test_creates_state_files(self, tmp_path):
        person_dir = tmp_path / "person"
        (person_dir / "state").mkdir(parents=True)
        _init_state_files(person_dir)
        ct = person_dir / "state" / "current_task.md"
        assert ct.exists()
        assert ct.read_text(encoding="utf-8") == "status: idle\n"
        pending = person_dir / "state" / "pending.md"
        assert pending.exists()
        assert pending.read_text(encoding="utf-8") == ""

    def test_does_not_overwrite_existing(self, tmp_path):
        person_dir = tmp_path / "person"
        (person_dir / "state").mkdir(parents=True)
        ct = person_dir / "state" / "current_task.md"
        ct.write_text("status: busy\n", encoding="utf-8")
        _init_state_files(person_dir)
        assert ct.read_text(encoding="utf-8") == "status: busy\n"


# ── _should_create_bootstrap ──────────────────────────────


class TestShouldCreateBootstrap:
    def test_no_identity(self, tmp_path):
        """Bootstrap needed when identity.md doesn't exist."""
        person_dir = tmp_path / "person"
        person_dir.mkdir()
        assert _should_create_bootstrap(person_dir) is True

    def test_empty_identity(self, tmp_path):
        """Bootstrap needed when identity.md is empty."""
        person_dir = tmp_path / "person"
        person_dir.mkdir()
        (person_dir / "identity.md").write_text("", encoding="utf-8")
        assert _should_create_bootstrap(person_dir) is True

    def test_identity_with_undefined(self, tmp_path):
        """Bootstrap needed when identity.md contains '未定義'."""
        person_dir = tmp_path / "person"
        person_dir.mkdir()
        (person_dir / "identity.md").write_text("名前: 未定義\n職業: 未定義", encoding="utf-8")
        assert _should_create_bootstrap(person_dir) is True

    def test_character_sheet_exists(self, tmp_path):
        """Bootstrap needed when character_sheet.md exists."""
        person_dir = tmp_path / "person"
        person_dir.mkdir()
        (person_dir / "identity.md").write_text("# Defined identity", encoding="utf-8")
        (person_dir / "character_sheet.md").write_text("# Character details", encoding="utf-8")
        assert _should_create_bootstrap(person_dir) is True

    def test_defined_identity_no_bootstrap(self, tmp_path):
        """Bootstrap NOT needed when identity.md is fully defined."""
        person_dir = tmp_path / "person"
        person_dir.mkdir()
        (person_dir / "identity.md").write_text(
            "# Person Identity\n\nName: Alice\nRole: Developer",
            encoding="utf-8"
        )
        assert _should_create_bootstrap(person_dir) is False


# ── _place_bootstrap ──────────────────────────────────────


class TestPlaceBootstrap:
    def test_copies_bootstrap(self, tmp_path):
        """Bootstrap is copied when needed (no identity.md)."""
        person_dir = tmp_path / "person"
        person_dir.mkdir()
        bootstrap = tmp_path / "bootstrap.md"
        bootstrap.write_text("Bootstrap content", encoding="utf-8")
        with patch("core.person_factory.BOOTSTRAP_TEMPLATE", bootstrap):
            _place_bootstrap(person_dir)
        assert (person_dir / "bootstrap.md").exists()
        assert (person_dir / "bootstrap.md").read_text(encoding="utf-8") == "Bootstrap content"

    def test_no_bootstrap_template(self, tmp_path):
        person_dir = tmp_path / "person"
        person_dir.mkdir()
        fake = tmp_path / "nonexistent_bootstrap.md"
        with patch("core.person_factory.BOOTSTRAP_TEMPLATE", fake):
            _place_bootstrap(person_dir)
        assert not (person_dir / "bootstrap.md").exists()

    def test_skips_bootstrap_when_not_needed(self, tmp_path):
        """Bootstrap is NOT copied when identity.md is fully defined."""
        person_dir = tmp_path / "person"
        person_dir.mkdir()
        (person_dir / "identity.md").write_text(
            "# Fully Defined\n\nName: Alice\nRole: Dev",
            encoding="utf-8"
        )
        bootstrap = tmp_path / "bootstrap.md"
        bootstrap.write_text("Bootstrap content", encoding="utf-8")
        with patch("core.person_factory.BOOTSTRAP_TEMPLATE", bootstrap):
            _place_bootstrap(person_dir)
        assert not (person_dir / "bootstrap.md").exists()


# ── list_person_templates ─────────────────────────────────


class TestListPersonTemplates:
    def test_no_templates_dir(self, tmp_path):
        with patch("core.person_factory.PERSON_TEMPLATES_DIR", tmp_path / "no"):
            assert list_person_templates() == []

    def test_lists_non_underscore_dirs(self, tmp_path):
        tpl_dir = tmp_path / "templates"
        tpl_dir.mkdir()
        (tpl_dir / "_blank").mkdir()
        (tpl_dir / "dev").mkdir()
        (tpl_dir / "sales").mkdir()
        (tpl_dir / "not_a_dir.txt").write_text("file", encoding="utf-8")
        with patch("core.person_factory.PERSON_TEMPLATES_DIR", tpl_dir):
            result = list_person_templates()
            assert "dev" in result
            assert "sales" in result
            assert "_blank" not in result
            assert "not_a_dir.txt" not in result


# ── create_from_template ──────────────────────────────────


class TestCreateFromTemplate:
    def test_creates_from_template(self, tmp_path):
        tpl_dir = tmp_path / "tpl"
        tpl_dir.mkdir()
        (tpl_dir / "dev").mkdir()
        (tpl_dir / "dev" / "identity.md").write_text("I am dev", encoding="utf-8")

        persons_dir = tmp_path / "persons"
        persons_dir.mkdir()

        with patch("core.person_factory.PERSON_TEMPLATES_DIR", tpl_dir), \
             patch("core.person_factory.BOOTSTRAP_TEMPLATE", tmp_path / "no"):
            person_dir = create_from_template(persons_dir, "dev")
            assert person_dir.exists()
            assert (person_dir / "identity.md").read_text(encoding="utf-8") == "I am dev"
            # Runtime subdirs should be created
            assert (person_dir / "episodes").is_dir()

    def test_raises_for_missing_template(self, tmp_path):
        tpl_dir = tmp_path / "tpl"
        tpl_dir.mkdir()
        persons_dir = tmp_path / "persons"
        persons_dir.mkdir()
        with patch("core.person_factory.PERSON_TEMPLATES_DIR", tpl_dir):
            with pytest.raises(FileNotFoundError):
                create_from_template(persons_dir, "nonexistent")

    def test_raises_for_existing_person(self, tmp_path):
        tpl_dir = tmp_path / "tpl"
        (tpl_dir / "dev").mkdir(parents=True)
        persons_dir = tmp_path / "persons"
        (persons_dir / "dev").mkdir(parents=True)
        with patch("core.person_factory.PERSON_TEMPLATES_DIR", tpl_dir):
            with pytest.raises(FileExistsError):
                create_from_template(persons_dir, "dev")

    def test_custom_name(self, tmp_path):
        tpl_dir = tmp_path / "tpl"
        (tpl_dir / "dev").mkdir(parents=True)
        (tpl_dir / "dev" / "identity.md").write_text("dev id", encoding="utf-8")
        persons_dir = tmp_path / "persons"
        persons_dir.mkdir()
        with patch("core.person_factory.PERSON_TEMPLATES_DIR", tpl_dir), \
             patch("core.person_factory.BOOTSTRAP_TEMPLATE", tmp_path / "no"):
            person_dir = create_from_template(persons_dir, "dev", person_name="alice")
            assert person_dir.name == "alice"


# ── create_blank ──────────────────────────────────────────


class TestCreateBlank:
    def test_creates_blank_person(self, tmp_path):
        persons_dir = tmp_path / "persons"
        persons_dir.mkdir()
        with patch("core.person_factory.BLANK_TEMPLATE_DIR", tmp_path / "no_blank"), \
             patch("core.person_factory.BOOTSTRAP_TEMPLATE", tmp_path / "no"):
            person_dir = create_blank(persons_dir, "alice")
            assert person_dir.exists()
            assert (person_dir / "episodes").is_dir()
            assert (person_dir / "state" / "current_task.md").exists()

    def test_blank_with_template(self, tmp_path):
        blank_dir = tmp_path / "blank"
        blank_dir.mkdir()
        (blank_dir / "identity.md").write_text("I am {name}", encoding="utf-8")

        persons_dir = tmp_path / "persons"
        persons_dir.mkdir()
        with patch("core.person_factory.BLANK_TEMPLATE_DIR", blank_dir), \
             patch("core.person_factory.BOOTSTRAP_TEMPLATE", tmp_path / "no"):
            person_dir = create_blank(persons_dir, "bob")
            content = (person_dir / "identity.md").read_text(encoding="utf-8")
            assert content == "I am bob"

    def test_raises_for_existing(self, tmp_path):
        persons_dir = tmp_path / "persons"
        (persons_dir / "alice").mkdir(parents=True)
        with pytest.raises(FileExistsError):
            create_blank(persons_dir, "alice")


# ── create_from_md ────────────────────────────────────────


class TestCreateFromMd:
    _VALID_SHEET = (
        "# Character: Alice\n\n"
        "## 基本情報\n\n"
        "| 項目 | 設定 |\n|------|------|\n| 英名 | alice |\n\n"
        "## 人格\n\nDetails here\n\n"
        "## 役割・行動方針\n\nRole details\n"
    )

    def test_creates_from_md(self, tmp_path):
        persons_dir = tmp_path / "persons"
        persons_dir.mkdir()
        md_file = tmp_path / "char.md"
        md_file.write_text(self._VALID_SHEET, encoding="utf-8")

        with patch("core.person_factory.BLANK_TEMPLATE_DIR", tmp_path / "no_blank"), \
             patch("core.person_factory.BOOTSTRAP_TEMPLATE", tmp_path / "no"):
            person_dir = create_from_md(persons_dir, md_file)
            assert person_dir.name == "alice"
            assert (person_dir / "character_sheet.md").exists()
            assert "Details here" in (person_dir / "character_sheet.md").read_text(encoding="utf-8")

    def test_explicit_name(self, tmp_path):
        persons_dir = tmp_path / "persons"
        persons_dir.mkdir()
        md_file = tmp_path / "char.md"
        md_file.write_text(self._VALID_SHEET, encoding="utf-8")

        with patch("core.person_factory.BLANK_TEMPLATE_DIR", tmp_path / "no_blank"), \
             patch("core.person_factory.BOOTSTRAP_TEMPLATE", tmp_path / "no"):
            person_dir = create_from_md(persons_dir, md_file, name="bob")
            assert person_dir.name == "bob"

    def test_raises_for_missing_md(self, tmp_path):
        persons_dir = tmp_path / "persons"
        persons_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            create_from_md(persons_dir, tmp_path / "nonexistent.md")

    def test_raises_for_unextractable_name(self, tmp_path):
        """Sheet with required sections but no extractable name raises ValueError."""
        persons_dir = tmp_path / "persons"
        persons_dir.mkdir()
        md_file = tmp_path / "char.md"
        # Valid sections but no name pattern anywhere
        md_file.write_text(
            "## 基本情報\n\n| 項目 | 設定 |\n|------|------|\n\n"
            "## 人格\n\nSome personality\n\n"
            "## 役割・行動方針\n\nSome role\n",
            encoding="utf-8",
        )

        with patch("core.person_factory.BLANK_TEMPLATE_DIR", tmp_path / "no_blank"), \
             patch("core.person_factory.BOOTSTRAP_TEMPLATE", tmp_path / "no"):
            with pytest.raises(ValueError, match="Could not extract"):
                create_from_md(persons_dir, md_file)


# ── _place_send_script ───────────────────────────────────


class TestPlaceSendScript:
    def test_copies_send_script(self, tmp_path):
        """Send script is copied from blank template to person dir."""
        blank_dir = tmp_path / "blank"
        blank_dir.mkdir()
        send_src = blank_dir / "send"
        send_src.write_text("#!/bin/bash\necho send", encoding="utf-8")

        person_dir = tmp_path / "person"
        person_dir.mkdir()

        with patch("core.person_factory.BLANK_TEMPLATE_DIR", blank_dir):
            _place_send_script(person_dir)

        dst = person_dir / "send"
        assert dst.exists()
        assert dst.read_text(encoding="utf-8") == "#!/bin/bash\necho send"
        # Check executable permission
        assert dst.stat().st_mode & 0o755

    def test_does_not_overwrite_existing(self, tmp_path):
        """If send script already exists in person_dir, don't overwrite."""
        blank_dir = tmp_path / "blank"
        blank_dir.mkdir()
        (blank_dir / "send").write_text("#!/bin/bash\nnew", encoding="utf-8")

        person_dir = tmp_path / "person"
        person_dir.mkdir()
        (person_dir / "send").write_text("#!/bin/bash\nold", encoding="utf-8")

        with patch("core.person_factory.BLANK_TEMPLATE_DIR", blank_dir):
            _place_send_script(person_dir)

        # Should keep the old content
        assert (person_dir / "send").read_text(encoding="utf-8") == "#!/bin/bash\nold"

    def test_no_source_script(self, tmp_path):
        """If blank template has no send script, do nothing."""
        blank_dir = tmp_path / "blank"
        blank_dir.mkdir()
        # No send script in blank_dir

        person_dir = tmp_path / "person"
        person_dir.mkdir()

        with patch("core.person_factory.BLANK_TEMPLATE_DIR", blank_dir):
            _place_send_script(person_dir)

        assert not (person_dir / "send").exists()

    def test_create_blank_includes_send_script(self, tmp_path):
        """create_blank() should call _place_send_script."""
        blank_dir = tmp_path / "blank"
        blank_dir.mkdir()
        (blank_dir / "send").write_text("#!/bin/bash\ntest", encoding="utf-8")
        (blank_dir / "identity.md").write_text("{name}", encoding="utf-8")

        persons_dir = tmp_path / "persons"
        persons_dir.mkdir()

        with patch("core.person_factory.BLANK_TEMPLATE_DIR", blank_dir), \
             patch("core.person_factory.BOOTSTRAP_TEMPLATE", tmp_path / "no"):
            person_dir = create_blank(persons_dir, "alice")

        assert (person_dir / "send").exists()

    def test_create_from_template_includes_send_script(self, tmp_path):
        """create_from_template() should also include send script."""
        tpl_dir = tmp_path / "tpl"
        (tpl_dir / "dev").mkdir(parents=True)
        (tpl_dir / "dev" / "identity.md").write_text("dev", encoding="utf-8")

        blank_dir = tmp_path / "blank"
        blank_dir.mkdir()
        (blank_dir / "send").write_text("#!/bin/bash\ntest", encoding="utf-8")

        persons_dir = tmp_path / "persons"
        persons_dir.mkdir()

        with patch("core.person_factory.PERSON_TEMPLATES_DIR", tpl_dir), \
             patch("core.person_factory.BLANK_TEMPLATE_DIR", blank_dir), \
             patch("core.person_factory.BOOTSTRAP_TEMPLATE", tmp_path / "no"):
            person_dir = create_from_template(persons_dir, "dev")

        assert (person_dir / "send").exists()


# ── _parse_character_sheet_info ──────────────────────────


class TestParseCharacterSheetInfo:
    def test_valid_basic_info_table(self):
        content = """\
## 基本情報

| 項目 | 設定 |
|------|------|
| 英名 | sakura |
| 役割 | developer |
| 上司 | tanaka |
| モデル | claude-sonnet |

## 人格
"""
        info = _parse_character_sheet_info(content)
        assert info["英名"] == "sakura"
        assert info["役割"] == "developer"
        assert info["上司"] == "tanaka"
        assert info["モデル"] == "claude-sonnet"

    def test_empty_content(self):
        info = _parse_character_sheet_info("No 基本情報 section here")
        assert info == {}

    def test_skips_header_and_separator_rows(self):
        content = """\
## 基本情報

| 項目 | 設定 |
|------|------|
| 英名 | hinata |
"""
        info = _parse_character_sheet_info(content)
        assert "項目" not in info
        assert "---" not in info
        assert info["英名"] == "hinata"


# ── _validate_character_sheet ────────────────────────────


class TestValidateCharacterSheet:
    def test_valid_sheet(self):
        content = """\
## 基本情報

| 英名 | sakura |

## 人格

Friendly and thoughtful.

## 役割・行動方針

Backend development.
"""
        # Should not raise
        _validate_character_sheet(content)

    def test_missing_basic_info(self):
        content = """\
## 人格

Friendly.

## 役割・行動方針

Development.
"""
        with pytest.raises(ValueError, match="基本情報"):
            _validate_character_sheet(content)

    def test_missing_personality(self):
        content = """\
## 基本情報

| 英名 | sakura |

## 役割・行動方針

Development.
"""
        with pytest.raises(ValueError, match="人格"):
            _validate_character_sheet(content)

    def test_missing_injection(self):
        content = """\
## 基本情報

| 英名 | sakura |

## 人格

Friendly.
"""
        with pytest.raises(ValueError, match="役割・行動方針"):
            _validate_character_sheet(content)

    def test_identity_md_redirect_counts_as_valid(self):
        content = """\
## 基本情報

| 英名 | sakura |

→ identity.md

→ injection.md
"""
        # Both 人格 and 役割・行動方針 are satisfied by redirect markers
        _validate_character_sheet(content)


# ── _create_status_json ──────────────────────────────────


class TestCreateStatusJson:
    def test_creates_status_json_with_correct_fields(self, tmp_path):
        person_dir = tmp_path / "person"
        person_dir.mkdir()
        info = {
            "英名": "sakura",
            "役割": "developer",
            "上司": "tanaka",
            "実行モード": "assisted",
            "モデル": "openai/gpt-4o",
            "credential": "openai_key",
        }
        _create_status_json(person_dir, info)
        status = json.loads((person_dir / "status.json").read_text(encoding="utf-8"))
        assert status["supervisor"] == "tanaka"
        assert status["role"] == "developer"
        assert status["execution_mode"] == "assisted"
        assert status["model"] == "openai/gpt-4o"
        assert status["credential"] == "openai_key"

    def test_supervisor_nashi_normalized(self, tmp_path):
        person_dir = tmp_path / "person"
        person_dir.mkdir()
        info = {"上司": "(なし)"}
        _create_status_json(person_dir, info)
        status = json.loads((person_dir / "status.json").read_text(encoding="utf-8"))
        assert status["supervisor"] == ""


# ── _extract_section_content ─────────────────────────────


class TestExtractSectionContent:
    def test_extracts_content(self):
        md = """\
## 人格

Friendly and thoughtful.
Loves coding.

## 役割・行動方針

Backend development.
"""
        result = _extract_section_content(md, "人格")
        assert result is not None
        assert "Friendly and thoughtful." in result
        assert "Loves coding." in result

    def test_returns_none_for_missing_section(self):
        md = """\
## 基本情報

| 英名 | sakura |
"""
        result = _extract_section_content(md, "人格")
        assert result is None

    def test_returns_none_for_redirect_only(self):
        md = """\
## 人格

→ identity.md

## 役割・行動方針

Development.
"""
        result = _extract_section_content(md, "人格")
        assert result is None


# ── _apply_defaults_from_sheet ───────────────────────────


class TestApplyDefaultsFromSheet:
    def test_writes_identity_md(self, tmp_path):
        person_dir = tmp_path / "person"
        person_dir.mkdir()
        md = """\
## 人格

Friendly and thoughtful.

## 役割・行動方針

Backend development.
"""
        _apply_defaults_from_sheet(person_dir, md)
        identity = person_dir / "identity.md"
        assert identity.exists()
        assert "Friendly and thoughtful." in identity.read_text(encoding="utf-8")

    def test_writes_injection_md(self, tmp_path):
        person_dir = tmp_path / "person"
        person_dir.mkdir()
        md = """\
## 人格

Personality here.

## 役割・行動方針

Backend development focus.
"""
        _apply_defaults_from_sheet(person_dir, md)
        injection = person_dir / "injection.md"
        assert injection.exists()
        assert "Backend development focus." in injection.read_text(encoding="utf-8")

    def test_writes_permissions_md(self, tmp_path):
        person_dir = tmp_path / "person"
        person_dir.mkdir()
        md = """\
## 人格

Personality.

## 役割・行動方針

Development.

## 権限

- read_file: OK
- write_file: OK
"""
        _apply_defaults_from_sheet(person_dir, md)
        permissions = person_dir / "permissions.md"
        assert permissions.exists()
        content = permissions.read_text(encoding="utf-8")
        assert "read_file" in content
        assert "write_file" in content

    def test_does_not_overwrite_when_section_missing(self, tmp_path):
        person_dir = tmp_path / "person"
        person_dir.mkdir()
        (person_dir / "identity.md").write_text("Original identity", encoding="utf-8")
        # MD with no 人格 section
        md = """\
## 役割・行動方針

Development.
"""
        _apply_defaults_from_sheet(person_dir, md)
        assert (person_dir / "identity.md").read_text(encoding="utf-8") == "Original identity"


# ── create_from_md rollback ──────────────────────────────


class TestCreateFromMdRollback:
    def test_rollback_on_failure(self, tmp_path):
        """On failure after directory creation, the directory is cleaned up."""
        persons_dir = tmp_path / "persons"
        persons_dir.mkdir()
        md_file = tmp_path / "char.md"
        md_file.write_text(
            """\
# Character: Sakura

## 基本情報

| 英名 | sakura |

## 人格

Friendly.

## 役割・行動方針

Development.
""",
            encoding="utf-8",
        )

        with patch("core.person_factory.BLANK_TEMPLATE_DIR", tmp_path / "no_blank"), \
             patch("core.person_factory.BOOTSTRAP_TEMPLATE", tmp_path / "no"), \
             patch(
                 "core.person_factory._apply_defaults_from_sheet",
                 side_effect=RuntimeError("simulated failure"),
             ):
            with pytest.raises(RuntimeError, match="simulated failure"):
                create_from_md(persons_dir, md_file)

        # Directory should be cleaned up after rollback
        assert not (persons_dir / "sakura").exists()


# ── _extract_name_from_md (table format) ─────────────────


class TestExtractNameFromMdTable:
    def test_table_format(self):
        content = """\
## 基本情報

| 項目 | 設定 |
|------|------|
| 英名 | hinata |
| 役割 | developer |
"""
        assert _extract_name_from_md(content) == "hinata"
