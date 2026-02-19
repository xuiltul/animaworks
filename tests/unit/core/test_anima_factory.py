"""Unit tests for core/anima_factory.py — anima creation factory."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from core.anima_factory import (
    BLANK_TEMPLATE_DIR,
    BOOTSTRAP_TEMPLATE,
    ANIMA_TEMPLATES_DIR,
    _RUNTIME_SUBDIRS,
    _apply_defaults_from_sheet,
    _create_status_json,
    _ensure_runtime_subdirs,
    _ensure_status_json,
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
    list_anima_templates,
    validate_anima_name,
)


# ── validate_anima_name ──────────────────────────────────


class TestValidateAnimaName:
    def test_valid_names(self):
        assert validate_anima_name("alice") is None
        assert validate_anima_name("bob-smith") is None
        assert validate_anima_name("charlie_01") is None
        assert validate_anima_name("a") is None

    def test_empty_name(self):
        assert validate_anima_name("") is not None

    def test_uppercase_rejected(self):
        assert validate_anima_name("Alice") is not None

    def test_starts_with_number(self):
        assert validate_anima_name("123abc") is not None

    def test_starts_with_underscore(self):
        assert validate_anima_name("_test") is not None

    def test_special_chars(self):
        assert validate_anima_name("a.b") is not None
        assert validate_anima_name("a b") is not None
        assert validate_anima_name("a@b") is not None


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
        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()
        _ensure_runtime_subdirs(anima_dir)
        for subdir in _RUNTIME_SUBDIRS:
            assert (anima_dir / subdir).is_dir()


# ── _init_state_files ─────────────────────────────────────


class TestInitStateFiles:
    def test_creates_state_files(self, tmp_path):
        anima_dir = tmp_path / "anima"
        (anima_dir / "state").mkdir(parents=True)
        _init_state_files(anima_dir)
        ct = anima_dir / "state" / "current_task.md"
        assert ct.exists()
        assert ct.read_text(encoding="utf-8") == "status: idle\n"
        pending = anima_dir / "state" / "pending.md"
        assert pending.exists()
        assert pending.read_text(encoding="utf-8") == ""

    def test_does_not_overwrite_existing(self, tmp_path):
        anima_dir = tmp_path / "anima"
        (anima_dir / "state").mkdir(parents=True)
        ct = anima_dir / "state" / "current_task.md"
        ct.write_text("status: busy\n", encoding="utf-8")
        _init_state_files(anima_dir)
        assert ct.read_text(encoding="utf-8") == "status: busy\n"


# ── _should_create_bootstrap ──────────────────────────────


class TestShouldCreateBootstrap:
    def test_no_identity(self, tmp_path):
        """Bootstrap needed when identity.md doesn't exist."""
        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()
        assert _should_create_bootstrap(anima_dir) is True

    def test_empty_identity(self, tmp_path):
        """Bootstrap needed when identity.md is empty."""
        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()
        (anima_dir / "identity.md").write_text("", encoding="utf-8")
        assert _should_create_bootstrap(anima_dir) is True

    def test_identity_with_undefined(self, tmp_path):
        """Bootstrap needed when identity.md contains '未定義'."""
        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()
        (anima_dir / "identity.md").write_text("名前: 未定義\n職業: 未定義", encoding="utf-8")
        assert _should_create_bootstrap(anima_dir) is True

    def test_character_sheet_exists(self, tmp_path):
        """Bootstrap needed when character_sheet.md exists."""
        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()
        (anima_dir / "identity.md").write_text("# Defined identity", encoding="utf-8")
        (anima_dir / "character_sheet.md").write_text("# Character details", encoding="utf-8")
        assert _should_create_bootstrap(anima_dir) is True

    def test_defined_identity_no_bootstrap(self, tmp_path):
        """Bootstrap NOT needed when identity.md is fully defined."""
        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()
        (anima_dir / "identity.md").write_text(
            "# Anima Identity\n\nName: Alice\nRole: Developer",
            encoding="utf-8"
        )
        assert _should_create_bootstrap(anima_dir) is False


# ── _place_bootstrap ──────────────────────────────────────


class TestPlaceBootstrap:
    def test_copies_bootstrap(self, tmp_path):
        """Bootstrap is copied when needed (no identity.md)."""
        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()
        bootstrap = tmp_path / "bootstrap.md"
        bootstrap.write_text("Bootstrap content", encoding="utf-8")
        with patch("core.anima_factory.BOOTSTRAP_TEMPLATE", bootstrap):
            _place_bootstrap(anima_dir)
        assert (anima_dir / "bootstrap.md").exists()
        assert (anima_dir / "bootstrap.md").read_text(encoding="utf-8") == "Bootstrap content"

    def test_no_bootstrap_template(self, tmp_path):
        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()
        fake = tmp_path / "nonexistent_bootstrap.md"
        with patch("core.anima_factory.BOOTSTRAP_TEMPLATE", fake):
            _place_bootstrap(anima_dir)
        assert not (anima_dir / "bootstrap.md").exists()

    def test_skips_bootstrap_when_not_needed(self, tmp_path):
        """Bootstrap is NOT copied when identity.md is fully defined."""
        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()
        (anima_dir / "identity.md").write_text(
            "# Fully Defined\n\nName: Alice\nRole: Dev",
            encoding="utf-8"
        )
        bootstrap = tmp_path / "bootstrap.md"
        bootstrap.write_text("Bootstrap content", encoding="utf-8")
        with patch("core.anima_factory.BOOTSTRAP_TEMPLATE", bootstrap):
            _place_bootstrap(anima_dir)
        assert not (anima_dir / "bootstrap.md").exists()


# ── list_anima_templates ─────────────────────────────────


class TestListAnimaTemplates:
    def test_no_templates_dir(self, tmp_path):
        with patch("core.anima_factory.ANIMA_TEMPLATES_DIR", tmp_path / "no"):
            assert list_anima_templates() == []

    def test_lists_non_underscore_dirs(self, tmp_path):
        tpl_dir = tmp_path / "templates"
        tpl_dir.mkdir()
        (tpl_dir / "_blank").mkdir()
        (tpl_dir / "dev").mkdir()
        (tpl_dir / "sales").mkdir()
        (tpl_dir / "not_a_dir.txt").write_text("file", encoding="utf-8")
        with patch("core.anima_factory.ANIMA_TEMPLATES_DIR", tpl_dir):
            result = list_anima_templates()
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

        animas_dir = tmp_path / "animas"
        animas_dir.mkdir()

        with patch("core.anima_factory.ANIMA_TEMPLATES_DIR", tpl_dir), \
             patch("core.anima_factory.BOOTSTRAP_TEMPLATE", tmp_path / "no"):
            anima_dir = create_from_template(animas_dir, "dev")
            assert anima_dir.exists()
            assert (anima_dir / "identity.md").read_text(encoding="utf-8") == "I am dev"
            # Runtime subdirs should be created
            assert (anima_dir / "episodes").is_dir()

    def test_raises_for_missing_template(self, tmp_path):
        tpl_dir = tmp_path / "tpl"
        tpl_dir.mkdir()
        animas_dir = tmp_path / "animas"
        animas_dir.mkdir()
        with patch("core.anima_factory.ANIMA_TEMPLATES_DIR", tpl_dir):
            with pytest.raises(FileNotFoundError):
                create_from_template(animas_dir, "nonexistent")

    def test_raises_for_existing_anima(self, tmp_path):
        tpl_dir = tmp_path / "tpl"
        (tpl_dir / "dev").mkdir(parents=True)
        animas_dir = tmp_path / "animas"
        (animas_dir / "dev").mkdir(parents=True)
        with patch("core.anima_factory.ANIMA_TEMPLATES_DIR", tpl_dir):
            with pytest.raises(FileExistsError):
                create_from_template(animas_dir, "dev")

    def test_custom_name(self, tmp_path):
        tpl_dir = tmp_path / "tpl"
        (tpl_dir / "dev").mkdir(parents=True)
        (tpl_dir / "dev" / "identity.md").write_text("dev id", encoding="utf-8")
        animas_dir = tmp_path / "animas"
        animas_dir.mkdir()
        with patch("core.anima_factory.ANIMA_TEMPLATES_DIR", tpl_dir), \
             patch("core.anima_factory.BOOTSTRAP_TEMPLATE", tmp_path / "no"):
            anima_dir = create_from_template(animas_dir, "dev", anima_name="alice")
            assert anima_dir.name == "alice"

    def test_creates_status_json(self, tmp_path):
        """create_from_template() creates status.json with enabled=true."""
        tpl_dir = tmp_path / "tpl"
        (tpl_dir / "dev").mkdir(parents=True)
        (tpl_dir / "dev" / "identity.md").write_text("dev id", encoding="utf-8")
        animas_dir = tmp_path / "animas"
        animas_dir.mkdir()
        with patch("core.anima_factory.ANIMA_TEMPLATES_DIR", tpl_dir), \
             patch("core.anima_factory.BOOTSTRAP_TEMPLATE", tmp_path / "no"):
            anima_dir = create_from_template(animas_dir, "dev")
            status_path = anima_dir / "status.json"
            assert status_path.exists()
            data = json.loads(status_path.read_text(encoding="utf-8"))
            assert data["enabled"] is True


# ── create_blank ──────────────────────────────────────────


class TestCreateBlank:
    def test_creates_blank_anima(self, tmp_path):
        animas_dir = tmp_path / "animas"
        animas_dir.mkdir()
        with patch("core.anima_factory.BLANK_TEMPLATE_DIR", tmp_path / "no_blank"), \
             patch("core.anima_factory.BOOTSTRAP_TEMPLATE", tmp_path / "no"):
            anima_dir = create_blank(animas_dir, "alice")
            assert anima_dir.exists()
            assert (anima_dir / "episodes").is_dir()
            assert (anima_dir / "state" / "current_task.md").exists()

    def test_blank_with_template(self, tmp_path):
        blank_dir = tmp_path / "blank"
        blank_dir.mkdir()
        (blank_dir / "identity.md").write_text("I am {name}", encoding="utf-8")

        animas_dir = tmp_path / "animas"
        animas_dir.mkdir()
        with patch("core.anima_factory.BLANK_TEMPLATE_DIR", blank_dir), \
             patch("core.anima_factory.BOOTSTRAP_TEMPLATE", tmp_path / "no"):
            anima_dir = create_blank(animas_dir, "bob")
            content = (anima_dir / "identity.md").read_text(encoding="utf-8")
            assert content == "I am bob"

    def test_raises_for_existing(self, tmp_path):
        animas_dir = tmp_path / "animas"
        (animas_dir / "alice").mkdir(parents=True)
        with pytest.raises(FileExistsError):
            create_blank(animas_dir, "alice")

    def test_creates_status_json(self, tmp_path):
        """create_blank() creates status.json with enabled=true."""
        animas_dir = tmp_path / "animas"
        animas_dir.mkdir()
        with patch("core.anima_factory.BLANK_TEMPLATE_DIR", tmp_path / "no_blank"), \
             patch("core.anima_factory.BOOTSTRAP_TEMPLATE", tmp_path / "no"):
            anima_dir = create_blank(animas_dir, "alice")
            status_path = anima_dir / "status.json"
            assert status_path.exists()
            data = json.loads(status_path.read_text(encoding="utf-8"))
            assert data["enabled"] is True


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
        animas_dir = tmp_path / "animas"
        animas_dir.mkdir()
        md_file = tmp_path / "char.md"
        md_file.write_text(self._VALID_SHEET, encoding="utf-8")

        with patch("core.anima_factory.BLANK_TEMPLATE_DIR", tmp_path / "no_blank"), \
             patch("core.anima_factory.BOOTSTRAP_TEMPLATE", tmp_path / "no"):
            anima_dir = create_from_md(animas_dir, md_file)
            assert anima_dir.name == "alice"
            assert (anima_dir / "character_sheet.md").exists()
            assert "Details here" in (anima_dir / "character_sheet.md").read_text(encoding="utf-8")

    def test_explicit_name(self, tmp_path):
        animas_dir = tmp_path / "animas"
        animas_dir.mkdir()
        md_file = tmp_path / "char.md"
        md_file.write_text(self._VALID_SHEET, encoding="utf-8")

        with patch("core.anima_factory.BLANK_TEMPLATE_DIR", tmp_path / "no_blank"), \
             patch("core.anima_factory.BOOTSTRAP_TEMPLATE", tmp_path / "no"):
            anima_dir = create_from_md(animas_dir, md_file, name="bob")
            assert anima_dir.name == "bob"

    def test_raises_for_missing_md(self, tmp_path):
        animas_dir = tmp_path / "animas"
        animas_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            create_from_md(animas_dir, tmp_path / "nonexistent.md")

    def test_raises_for_unextractable_name(self, tmp_path):
        """Sheet with required sections but no extractable name raises ValueError."""
        animas_dir = tmp_path / "animas"
        animas_dir.mkdir()
        md_file = tmp_path / "char.md"
        # Valid sections but no name pattern anywhere
        md_file.write_text(
            "## 基本情報\n\n| 項目 | 設定 |\n|------|------|\n\n"
            "## 人格\n\nSome personality\n\n"
            "## 役割・行動方針\n\nSome role\n",
            encoding="utf-8",
        )

        with patch("core.anima_factory.BLANK_TEMPLATE_DIR", tmp_path / "no_blank"), \
             patch("core.anima_factory.BOOTSTRAP_TEMPLATE", tmp_path / "no"):
            with pytest.raises(ValueError, match="Could not extract"):
                create_from_md(animas_dir, md_file)

    def test_creates_from_content_string(self, tmp_path):
        """create_from_md with content= creates anima without a file."""
        animas_dir = tmp_path / "animas"
        animas_dir.mkdir()

        with patch("core.anima_factory.BLANK_TEMPLATE_DIR", tmp_path / "no_blank"), \
             patch("core.anima_factory.BOOTSTRAP_TEMPLATE", tmp_path / "no"):
            anima_dir = create_from_md(animas_dir, content=self._VALID_SHEET)
            assert anima_dir.name == "alice"
            assert (anima_dir / "character_sheet.md").exists()

    def test_content_takes_priority_over_path(self, tmp_path):
        """When both content and md_path are given, content wins."""
        animas_dir = tmp_path / "animas"
        animas_dir.mkdir()
        md_file = tmp_path / "char.md"
        md_file.write_text("invalid content", encoding="utf-8")

        with patch("core.anima_factory.BLANK_TEMPLATE_DIR", tmp_path / "no_blank"), \
             patch("core.anima_factory.BOOTSTRAP_TEMPLATE", tmp_path / "no"):
            anima_dir = create_from_md(animas_dir, md_file, content=self._VALID_SHEET)
            assert anima_dir.name == "alice"

    def test_raises_when_neither_path_nor_content(self, tmp_path):
        """Must provide either md_path or content."""
        animas_dir = tmp_path / "animas"
        animas_dir.mkdir()
        with pytest.raises(ValueError, match="Either md_path or content"):
            create_from_md(animas_dir)

    def test_supervisor_override(self, tmp_path):
        """Explicit supervisor parameter overrides sheet value."""
        animas_dir = tmp_path / "animas"
        animas_dir.mkdir()

        sheet_with_supervisor = (
            "# Character: Bob\n\n"
            "## 基本情報\n\n"
            "| 項目 | 設定 |\n|------|------|\n"
            "| 英名 | bob |\n| 上司 | tanaka |\n\n"
            "## 人格\n\nDetails\n\n"
            "## 役割・行動方針\n\nRole\n"
        )
        with patch("core.anima_factory.BLANK_TEMPLATE_DIR", tmp_path / "no_blank"), \
             patch("core.anima_factory.BOOTSTRAP_TEMPLATE", tmp_path / "no"):
            anima_dir = create_from_md(
                animas_dir, content=sheet_with_supervisor, supervisor="rin"
            )
            status = json.loads(
                (anima_dir / "status.json").read_text(encoding="utf-8")
            )
            assert status["supervisor"] == "rin"

    def test_supervisor_from_sheet_when_no_override(self, tmp_path):
        """Without explicit supervisor, sheet value is used."""
        animas_dir = tmp_path / "animas"
        animas_dir.mkdir()

        sheet_with_supervisor = (
            "# Character: Carol\n\n"
            "## 基本情報\n\n"
            "| 項目 | 設定 |\n|------|------|\n"
            "| 英名 | carol |\n| 上司 | tanaka |\n\n"
            "## 人格\n\nDetails\n\n"
            "## 役割・行動方針\n\nRole\n"
        )
        with patch("core.anima_factory.BLANK_TEMPLATE_DIR", tmp_path / "no_blank"), \
             patch("core.anima_factory.BOOTSTRAP_TEMPLATE", tmp_path / "no"):
            anima_dir = create_from_md(
                animas_dir, content=sheet_with_supervisor
            )
            status = json.loads(
                (anima_dir / "status.json").read_text(encoding="utf-8")
            )
            assert status["supervisor"] == "tanaka"


# ── _place_send_script ───────────────────────────────────


class TestPlaceSendScript:
    def test_copies_send_script(self, tmp_path):
        """Send script is copied from blank template to anima dir."""
        blank_dir = tmp_path / "blank"
        blank_dir.mkdir()
        send_src = blank_dir / "send"
        send_src.write_text("#!/bin/bash\necho send", encoding="utf-8")

        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()

        with patch("core.anima_factory.BLANK_TEMPLATE_DIR", blank_dir):
            _place_send_script(anima_dir)

        dst = anima_dir / "send"
        assert dst.exists()
        assert dst.read_text(encoding="utf-8") == "#!/bin/bash\necho send"
        # Check executable permission
        assert dst.stat().st_mode & 0o755

    def test_overwrites_existing(self, tmp_path):
        """Send script is always overwritten to match the latest template."""
        blank_dir = tmp_path / "blank"
        blank_dir.mkdir()
        (blank_dir / "send").write_text("#!/bin/bash\nnew", encoding="utf-8")

        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()
        (anima_dir / "send").write_text("#!/bin/bash\nold", encoding="utf-8")

        with patch("core.anima_factory.BLANK_TEMPLATE_DIR", blank_dir):
            _place_send_script(anima_dir)

        # Should be updated to match the template
        assert (anima_dir / "send").read_text(encoding="utf-8") == "#!/bin/bash\nnew"

    def test_no_source_script(self, tmp_path):
        """If blank template has no send script, do nothing."""
        blank_dir = tmp_path / "blank"
        blank_dir.mkdir()
        # No send script in blank_dir

        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()

        with patch("core.anima_factory.BLANK_TEMPLATE_DIR", blank_dir):
            _place_send_script(anima_dir)

        assert not (anima_dir / "send").exists()

    def test_create_blank_includes_send_script(self, tmp_path):
        """create_blank() should call _place_send_script."""
        blank_dir = tmp_path / "blank"
        blank_dir.mkdir()
        (blank_dir / "send").write_text("#!/bin/bash\ntest", encoding="utf-8")
        (blank_dir / "identity.md").write_text("{name}", encoding="utf-8")

        animas_dir = tmp_path / "animas"
        animas_dir.mkdir()

        with patch("core.anima_factory.BLANK_TEMPLATE_DIR", blank_dir), \
             patch("core.anima_factory.BOOTSTRAP_TEMPLATE", tmp_path / "no"):
            anima_dir = create_blank(animas_dir, "alice")

        assert (anima_dir / "send").exists()

    def test_create_from_template_includes_send_script(self, tmp_path):
        """create_from_template() should also include send script."""
        tpl_dir = tmp_path / "tpl"
        (tpl_dir / "dev").mkdir(parents=True)
        (tpl_dir / "dev" / "identity.md").write_text("dev", encoding="utf-8")

        blank_dir = tmp_path / "blank"
        blank_dir.mkdir()
        (blank_dir / "send").write_text("#!/bin/bash\ntest", encoding="utf-8")

        animas_dir = tmp_path / "animas"
        animas_dir.mkdir()

        with patch("core.anima_factory.ANIMA_TEMPLATES_DIR", tpl_dir), \
             patch("core.anima_factory.BLANK_TEMPLATE_DIR", blank_dir), \
             patch("core.anima_factory.BOOTSTRAP_TEMPLATE", tmp_path / "no"):
            anima_dir = create_from_template(animas_dir, "dev")

        assert (anima_dir / "send").exists()


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
        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()
        info = {
            "英名": "sakura",
            "役割": "developer",
            "上司": "tanaka",
            "実行モード": "assisted",
            "モデル": "openai/gpt-4o",
            "credential": "openai_key",
        }
        _create_status_json(anima_dir, info)
        status = json.loads((anima_dir / "status.json").read_text(encoding="utf-8"))
        assert status["supervisor"] == "tanaka"
        assert status["role"] == "general"
        assert status["execution_mode"] == "assisted"
        assert status["model"] == "openai/gpt-4o"
        assert status["credential"] == "openai_key"

    def test_supervisor_nashi_normalized(self, tmp_path):
        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()
        info = {"上司": "(なし)"}
        _create_status_json(anima_dir, info)
        status = json.loads((anima_dir / "status.json").read_text(encoding="utf-8"))
        assert status["supervisor"] == ""

    def test_supervisor_override(self, tmp_path):
        """Explicit supervisor_override takes priority over sheet value."""
        person_dir = tmp_path / "person"
        person_dir.mkdir()
        info = {"上司": "tanaka"}
        _create_status_json(person_dir, info, supervisor_override="yamada")
        status = json.loads((person_dir / "status.json").read_text(encoding="utf-8"))
        assert status["supervisor"] == "yamada"

    def test_supervisor_override_with_empty_sheet(self, tmp_path):
        """supervisor_override works even when sheet has no supervisor."""
        person_dir = tmp_path / "person"
        person_dir.mkdir()
        info = {}
        _create_status_json(person_dir, info, supervisor_override="rin")
        status = json.loads((person_dir / "status.json").read_text(encoding="utf-8"))
        assert status["supervisor"] == "rin"


# ── _ensure_status_json ──────────────────────────────────


class TestEnsureStatusJson:
    def test_creates_minimal_status_json(self, tmp_path):
        """Creates {"enabled": true} when no status.json exists."""
        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()
        _ensure_status_json(anima_dir)
        status_path = anima_dir / "status.json"
        assert status_path.exists()
        data = json.loads(status_path.read_text(encoding="utf-8"))
        assert data == {"enabled": True}

    def test_does_not_overwrite_existing(self, tmp_path):
        """Preserves existing status.json with richer fields."""
        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()
        existing = {"enabled": True, "supervisor": "tanaka", "role": "developer"}
        (anima_dir / "status.json").write_text(
            json.dumps(existing, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        _ensure_status_json(anima_dir)
        data = json.loads((anima_dir / "status.json").read_text(encoding="utf-8"))
        assert data == existing  # unchanged

    def test_does_not_overwrite_disabled(self, tmp_path):
        """Preserves disabled status.json."""
        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()
        existing = {"enabled": False}
        (anima_dir / "status.json").write_text(
            json.dumps(existing) + "\n", encoding="utf-8",
        )
        _ensure_status_json(anima_dir)
        data = json.loads((anima_dir / "status.json").read_text(encoding="utf-8"))
        assert data["enabled"] is False


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
        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()
        md = """\
## 人格

Friendly and thoughtful.

## 役割・行動方針

Backend development.
"""
        _apply_defaults_from_sheet(anima_dir, md)
        identity = anima_dir / "identity.md"
        assert identity.exists()
        assert "Friendly and thoughtful." in identity.read_text(encoding="utf-8")

    def test_writes_injection_md(self, tmp_path):
        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()
        md = """\
## 人格

Personality here.

## 役割・行動方針

Backend development focus.
"""
        _apply_defaults_from_sheet(anima_dir, md)
        injection = anima_dir / "injection.md"
        assert injection.exists()
        assert "Backend development focus." in injection.read_text(encoding="utf-8")

    def test_writes_permissions_md(self, tmp_path):
        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()
        md = """\
## 人格

Personality.

## 役割・行動方針

Development.

## 権限

- read_file: OK
- write_file: OK
"""
        _apply_defaults_from_sheet(anima_dir, md)
        permissions = anima_dir / "permissions.md"
        assert permissions.exists()
        content = permissions.read_text(encoding="utf-8")
        assert "read_file" in content
        assert "write_file" in content

    def test_does_not_overwrite_when_section_missing(self, tmp_path):
        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()
        (anima_dir / "identity.md").write_text("Original identity", encoding="utf-8")
        # MD with no 人格 section
        md = """\
## 役割・行動方針

Development.
"""
        _apply_defaults_from_sheet(anima_dir, md)
        assert (anima_dir / "identity.md").read_text(encoding="utf-8") == "Original identity"


# ── create_from_md rollback ──────────────────────────────


class TestCreateFromMdRollback:
    def test_rollback_on_failure(self, tmp_path):
        """On failure after directory creation, the directory is cleaned up."""
        animas_dir = tmp_path / "animas"
        animas_dir.mkdir()
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

        with patch("core.anima_factory.BLANK_TEMPLATE_DIR", tmp_path / "no_blank"), \
             patch("core.anima_factory.BOOTSTRAP_TEMPLATE", tmp_path / "no"), \
             patch(
                 "core.anima_factory._apply_defaults_from_sheet",
                 side_effect=RuntimeError("simulated failure"),
             ):
            with pytest.raises(RuntimeError, match="simulated failure"):
                create_from_md(animas_dir, md_file)

        # Directory should be cleaned up after rollback
        assert not (animas_dir / "sakura").exists()


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
