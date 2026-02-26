"""Unit tests for i18n / multi-locale support."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import patch

import pytest

from core.anima_factory import (
    _detect_sheet_locale,
    _extract_name_from_md,
    _parse_character_sheet_info,
    _validate_character_sheet,
)
from core.paths import _prompt_cache, load_prompt, resolve_template_path
from core.prompt.builder import _load_fallback_strings, _load_section_strings
from core.tooling.prompt_db import get_default_description, get_default_guide


# ── TestLoadPromptLocale ─────────────────────────────────────


class TestLoadPromptLocale:
    """Test load_prompt() locale-aware resolution."""

    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        _prompt_cache.clear()
        yield
        _prompt_cache.clear()

    def test_default_locale_loads_ja(self, tmp_path):
        """load_prompt() with default locale=ja loads Japanese template."""
        ja_prompts = tmp_path / "ja" / "prompts"
        en_prompts = tmp_path / "en" / "prompts"
        ja_prompts.mkdir(parents=True)
        en_prompts.mkdir(parents=True)
        (ja_prompts / "test.md").write_text("日本語テンプレート", encoding="utf-8")
        (en_prompts / "test.md").write_text("English template", encoding="utf-8")

        with patch("core.paths.TEMPLATES_DIR", tmp_path):
            result = load_prompt("test", locale="ja")
        assert result == "日本語テンプレート"

    def test_explicit_locale_en(self, tmp_path):
        """load_prompt(locale="en") loads English template."""
        ja_prompts = tmp_path / "ja" / "prompts"
        en_prompts = tmp_path / "en" / "prompts"
        ja_prompts.mkdir(parents=True)
        en_prompts.mkdir(parents=True)
        (ja_prompts / "test.md").write_text("日本語", encoding="utf-8")
        (en_prompts / "test.md").write_text("English", encoding="utf-8")

        with patch("core.paths.TEMPLATES_DIR", tmp_path):
            result = load_prompt("test", locale="en")
        assert result == "English"

    def test_fallback_en_to_ja(self, tmp_path):
        """When en template doesn't exist, falls back to ja."""
        ja_prompts = tmp_path / "ja" / "prompts"
        en_prompts = tmp_path / "en" / "prompts"
        ja_prompts.mkdir(parents=True)
        en_prompts.mkdir(parents=True)
        (ja_prompts / "test.md").write_text("日本語フォールバック", encoding="utf-8")
        # No en/test.md

        with patch("core.paths.TEMPLATES_DIR", tmp_path):
            result = load_prompt("test", locale="en")
        assert result == "日本語フォールバック"

    def test_fallback_unknown_locale_to_en_to_ja(self, tmp_path):
        """When unknown locale (e.g. "fr") requested, falls back to en then ja."""
        ja_prompts = tmp_path / "ja" / "prompts"
        en_prompts = tmp_path / "en" / "prompts"
        ja_prompts.mkdir(parents=True)
        en_prompts.mkdir(parents=True)
        (ja_prompts / "test.md").write_text("ja content", encoding="utf-8")
        (en_prompts / "test.md").write_text("en content", encoding="utf-8")

        with patch("core.paths.TEMPLATES_DIR", tmp_path):
            result = load_prompt("test", locale="fr")
        # fr -> en (exists) -> returns en content
        assert result == "en content"

    def test_fallback_unknown_locale_to_ja_when_no_en(self, tmp_path):
        """When fr requested and en missing, falls back to ja."""
        ja_prompts = tmp_path / "ja" / "prompts"
        en_prompts = tmp_path / "en" / "prompts"
        ja_prompts.mkdir(parents=True)
        en_prompts.mkdir(parents=True)
        (ja_prompts / "test.md").write_text("ja content", encoding="utf-8")
        # No en/test.md

        with patch("core.paths.TEMPLATES_DIR", tmp_path):
            result = load_prompt("test", locale="fr")
        assert result == "ja content"

    def test_cache_key_includes_locale(self, tmp_path):
        """Different locales use different cache entries."""
        ja_prompts = tmp_path / "ja" / "prompts"
        en_prompts = tmp_path / "en" / "prompts"
        ja_prompts.mkdir(parents=True)
        en_prompts.mkdir(parents=True)
        (ja_prompts / "test.md").write_text("ja", encoding="utf-8")
        (en_prompts / "test.md").write_text("en", encoding="utf-8")

        with patch("core.paths.TEMPLATES_DIR", tmp_path):
            load_prompt("test", locale="ja")
            load_prompt("test", locale="en")
        assert ("ja", "test") in _prompt_cache
        assert ("en", "test") in _prompt_cache
        assert _prompt_cache[("ja", "test")] == "ja"
        assert _prompt_cache[("en", "test")] == "en"


# ── TestResolveTemplatePath ───────────────────────────────────


class TestResolveTemplatePath:
    def test_resolves_locale_path(self, tmp_path):
        """Resolves to locale/category/filename."""
        ja_prompts = tmp_path / "ja" / "prompts"
        ja_prompts.mkdir(parents=True)
        (ja_prompts / "env.md").write_text("content", encoding="utf-8")

        with patch("core.paths.TEMPLATES_DIR", tmp_path):
            path = resolve_template_path("prompts", "env.md", locale="ja")
        assert path == tmp_path / "ja" / "prompts" / "env.md"

    def test_fallback_to_shared(self, tmp_path):
        """Falls back to _shared/ when not in any locale dir."""
        shared_prompts = tmp_path / "_shared" / "prompts"
        shared_prompts.mkdir(parents=True)
        (shared_prompts / "shared_only.md").write_text("shared", encoding="utf-8")
        # No ja/en for this file

        with patch("core.paths.TEMPLATES_DIR", tmp_path):
            path = resolve_template_path("prompts", "shared_only.md", locale="ja")
        assert path == tmp_path / "_shared" / "prompts" / "shared_only.md"
        assert path.read_text(encoding="utf-8") == "shared"

    def test_raises_file_not_found(self, tmp_path):
        """Raises FileNotFoundError when template not found anywhere."""
        with patch("core.paths.TEMPLATES_DIR", tmp_path):
            with pytest.raises(FileNotFoundError) as excinfo:
                resolve_template_path("prompts", "nonexistent_xyz.md", locale="ja")
        assert "nonexistent_xyz" in str(excinfo.value)

    def test_fallback_chain_locale_en_ja(self, tmp_path):
        """Fallback chain: requested locale -> en -> ja."""
        ja_prompts = tmp_path / "ja" / "prompts"
        en_prompts = tmp_path / "en" / "prompts"
        ja_prompts.mkdir(parents=True)
        en_prompts.mkdir(parents=True)
        (ja_prompts / "chain.md").write_text("ja", encoding="utf-8")
        (en_prompts / "chain.md").write_text("en", encoding="utf-8")

        with patch("core.paths.TEMPLATES_DIR", tmp_path):
            path_fr = resolve_template_path("prompts", "chain.md", locale="fr")
            path_xx = resolve_template_path("prompts", "chain.md", locale="xx")
        assert path_fr.read_text(encoding="utf-8") == "en"
        assert path_xx.read_text(encoding="utf-8") == "en"


# ── TestBuilderTemplateParse ───────────────────────────────────


class TestBuilderTemplateParse:
    def test_load_section_strings_returns_dict(self):
        """_load_section_strings() returns non-empty dict."""
        result = _load_section_strings(locale="ja")
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_load_section_strings_has_required_keys(self):
        """Section strings contain all expected keys."""
        result = _load_section_strings(locale="ja")
        expected = {
            "group1_header",
            "group2_header",
            "group3_header",
            "current_state_header",
            "pending_tasks_header",
            "available_tools_header",
        }
        for key in expected:
            assert key in result

    def test_load_fallback_strings_returns_dict(self):
        """_load_fallback_strings() returns non-empty dict."""
        result = _load_fallback_strings(locale="ja")
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_load_fallback_strings_has_required_keys(self):
        """Fallback strings contain all expected keys."""
        result = _load_fallback_strings(locale="ja")
        expected = {"unset", "none", "truncated", "summary"}
        for key in expected:
            assert key in result


# ── TestToolDescriptionLocale ──────────────────────────────────


class TestToolDescriptionLocale:
    def test_get_default_description_ja(self):
        """get_default_description returns Japanese for ja locale."""
        result = get_default_description("search_memory", locale="ja")
        assert "長期記憶" in result or "キーワード検索" in result

    def test_get_default_description_en(self):
        """get_default_description returns English for en locale."""
        result = get_default_description("search_memory", locale="en")
        assert "long-term memory" in result or "keyword" in result.lower()

    def test_get_default_description_fallback(self):
        """get_default_description falls back from unknown locale to en."""
        result = get_default_description("search_memory", locale="fr")
        assert len(result) > 0
        assert "long-term memory" in result or "keyword" in result.lower()

    def test_get_default_guide_ja(self):
        """get_default_guide returns Japanese for ja locale."""
        result = get_default_guide("s_mcp", locale="ja")
        assert "MCPツール" in result or "タスク" in result

    def test_get_default_guide_en(self):
        """get_default_guide returns English for en locale."""
        result = get_default_guide("s_mcp", locale="en")
        assert "MCP" in result or "task" in result.lower()


# ── TestCharacterSheetMultilingual ─────────────────────────────


class TestCharacterSheetMultilingual:
    def test_detect_sheet_locale_ja(self):
        """Detects Japanese character sheet."""
        content = """# Character: ひなた

## 基本情報

| 項目 | 設定 |
| 英名 | hinata |
"""
        assert _detect_sheet_locale(content) == "ja"

    def test_detect_sheet_locale_en(self):
        """Detects English character sheet."""
        content = """# Character: Alice

## Basic Information

| Field | Value |
| Name | alice |
"""
        assert _detect_sheet_locale(content) == "en"

    def test_parse_japanese_sheet(self):
        """Parses Japanese character sheet correctly."""
        content = """# Character: さくら

## 基本情報

| 項目 | 設定 |
|------|------|
| 英名 | sakura |
| 上司 | manager |

## 人格

Personality content.

## 役割・行動方針

Role content.
"""
        info = _parse_character_sheet_info(content)
        assert info.get("name") == "sakura"
        assert info.get("supervisor") == "manager"

    def test_parse_english_sheet(self):
        """Parses English character sheet correctly."""
        content = """# Character: Bob

## Basic Information

| Field | Value |
|-------|------|
| Name | bob |
| Supervisor | alice |

## Personality

Personality content.

## Role & Guidelines

Role content.
"""
        info = _parse_character_sheet_info(content)
        assert info.get("name") == "bob"
        assert info.get("supervisor") == "alice"

    def test_validate_english_sheet(self):
        """Validates English character sheet without errors."""
        content = """# Character: Charlie

## Basic Information

| Field | Value |
| Name | charlie |

## Personality

Content.

## Role & Guidelines

Content.
"""
        _validate_character_sheet(content)

    def test_extract_name_from_english_sheet(self):
        """Extracts name from English character sheet."""
        content = """# Character: Dave

## Basic Information

| Field | Value |
| Name | dave |
"""
        assert _extract_name_from_md(content) == "dave"
