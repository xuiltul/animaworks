"""Unit tests for core/paths.py — path resolution and prompt loading."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from core.paths import (
    PROJECT_DIR,
    TEMPLATES_DIR,
    PROMPTS_DIR,
    _prompt_cache,
    get_common_knowledge_dir,
    get_common_skills_dir,
    get_company_dir,
    get_data_dir,
    get_animas_dir,
    get_shared_dir,
    get_tmp_dir,
    load_prompt,
)


# ── Constants ─────────────────────────────────────────────


class TestConstants:
    def test_project_dir_exists(self):
        assert PROJECT_DIR.is_dir()

    def test_templates_dir_is_under_project(self):
        assert TEMPLATES_DIR == PROJECT_DIR / "templates"

    def test_prompts_dir_is_under_templates(self):
        assert PROMPTS_DIR == TEMPLATES_DIR / "prompts"


# ── get_data_dir ──────────────────────────────────────────


class TestGetDataDir:
    def test_default_is_home_animaworks(self):
        with patch.dict("os.environ", {}, clear=False):
            # Remove ANIMAWORKS_DATA_DIR if set
            import os
            old = os.environ.pop("ANIMAWORKS_DATA_DIR", None)
            try:
                result = get_data_dir()
                assert result == Path.home() / ".animaworks"
            finally:
                if old is not None:
                    os.environ["ANIMAWORKS_DATA_DIR"] = old

    def test_env_override(self, tmp_path):
        with patch.dict("os.environ", {"ANIMAWORKS_DATA_DIR": str(tmp_path)}):
            result = get_data_dir()
            assert result == tmp_path.resolve()

    def test_env_override_with_tilde(self):
        with patch.dict("os.environ", {"ANIMAWORKS_DATA_DIR": "~/custom_dir"}):
            result = get_data_dir()
            assert str(result).startswith(str(Path.home()))
            assert "custom_dir" in str(result)


# ── Subdirectory helpers ──────────────────────────────────


class TestSubdirHelpers:
    def test_get_animas_dir(self, tmp_path):
        with patch.dict("os.environ", {"ANIMAWORKS_DATA_DIR": str(tmp_path)}):
            assert get_animas_dir() == tmp_path.resolve() / "animas"

    def test_get_shared_dir(self, tmp_path):
        with patch.dict("os.environ", {"ANIMAWORKS_DATA_DIR": str(tmp_path)}):
            assert get_shared_dir() == tmp_path.resolve() / "shared"

    def test_get_company_dir(self, tmp_path):
        with patch.dict("os.environ", {"ANIMAWORKS_DATA_DIR": str(tmp_path)}):
            assert get_company_dir() == tmp_path.resolve() / "company"

    def test_get_common_skills_dir(self, tmp_path):
        with patch.dict("os.environ", {"ANIMAWORKS_DATA_DIR": str(tmp_path)}):
            assert get_common_skills_dir() == tmp_path.resolve() / "common_skills"

    def test_get_common_knowledge_dir(self, tmp_path):
        with patch.dict("os.environ", {"ANIMAWORKS_DATA_DIR": str(tmp_path)}):
            assert get_common_knowledge_dir() == tmp_path.resolve() / "common_knowledge"

    def test_get_tmp_dir(self, tmp_path):
        with patch.dict("os.environ", {"ANIMAWORKS_DATA_DIR": str(tmp_path)}):
            assert get_tmp_dir() == tmp_path.resolve() / "tmp"


# ── load_prompt ───────────────────────────────────────────


class TestLoadPrompt:
    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        _prompt_cache.clear()
        yield
        _prompt_cache.clear()

    def test_load_simple_template(self, tmp_path):
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "test_tpl.md").write_text(
            "Hello {anima_name}!", encoding="utf-8"
        )
        with patch("core.paths.PROMPTS_DIR", prompts_dir):
            result = load_prompt("test_tpl", anima_name="World")
            assert result == "Hello World!"

    def test_load_without_kwargs(self, tmp_path):
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "plain.md").write_text(
            "No placeholders here.", encoding="utf-8"
        )
        with patch("core.paths.PROMPTS_DIR", prompts_dir):
            result = load_prompt("plain")
            assert result == "No placeholders here."

    def test_caching(self, tmp_path):
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "cached.md").write_text("Content", encoding="utf-8")
        with patch("core.paths.PROMPTS_DIR", prompts_dir):
            load_prompt("cached")
            assert "cached" in _prompt_cache
            # Modify file; cached version should be returned
            (prompts_dir / "cached.md").write_text("Modified", encoding="utf-8")
            result = load_prompt("cached")
            assert result == "Content"  # still cached

    def test_missing_template_raises(self, tmp_path):
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        with patch("core.paths.PROMPTS_DIR", prompts_dir):
            with pytest.raises(FileNotFoundError):
                load_prompt("nonexistent")

    def test_format_map_with_multiple_kwargs(self, tmp_path):
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "multi.md").write_text(
            "{a} and {b}", encoding="utf-8"
        )
        with patch("core.paths.PROMPTS_DIR", prompts_dir):
            result = load_prompt("multi", a="X", b="Y")
            assert result == "X and Y"
