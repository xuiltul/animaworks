from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for FileToolsMixin._try_write_with_frontmatter.

Verifies that writing .md files inside knowledge/ or procedures/ under
anima_dir automatically injects YAML frontmatter, while files elsewhere
or with existing frontmatter are left untouched.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from core.memory.frontmatter import parse_frontmatter
from core.tooling.handler_files import FileToolsMixin


def _make_mixin(tmp_path: Path) -> FileToolsMixin:
    """Build a minimal FileToolsMixin with stubbed attributes."""
    mixin = FileToolsMixin.__new__(FileToolsMixin)

    anima_dir = tmp_path / "anima"
    anima_dir.mkdir()
    (anima_dir / "procedures").mkdir()
    (anima_dir / "knowledge").mkdir()

    mixin._anima_dir = anima_dir

    memory = MagicMock()

    def _write_procedure(path, content, metadata):
        import yaml
        from core.memory.frontmatter import strip_content_frontmatter
        content = strip_content_frontmatter(content)
        fm = yaml.dump(metadata, default_flow_style=False, allow_unicode=True).rstrip()
        path.write_text(f"---\n{fm}\n---\n\n{content}", encoding="utf-8")

    def _write_knowledge(path, content, metadata):
        import yaml
        from core.memory.frontmatter import strip_content_frontmatter
        content = strip_content_frontmatter(content)
        fm = yaml.dump(metadata, default_flow_style=False, allow_unicode=True)
        path.write_text(f"---\n{fm}---\n\n{content}", encoding="utf-8")

    memory.write_procedure_with_meta = MagicMock(side_effect=_write_procedure)
    memory.write_knowledge_with_meta = MagicMock(side_effect=_write_knowledge)
    mixin._memory = memory

    return mixin


class TestWriteFileFrontmatterHook:
    """Tests for FileToolsMixin._try_write_with_frontmatter."""

    def test_procedures_auto_frontmatter(self, tmp_path) -> None:
        """procedures/ への Write で FM が自動付与される。"""
        mixin = _make_mixin(tmp_path)
        target = mixin._anima_dir / "procedures" / "deploy.md"

        handled = mixin._try_write_with_frontmatter(target, "# Deploy\n\nStep 1")
        assert handled is True

        mixin._memory.write_procedure_with_meta.assert_called_once()
        text = target.read_text(encoding="utf-8")
        assert text.startswith("---")
        meta, body = parse_frontmatter(text)
        assert "description" in meta
        assert meta["confidence"] == pytest.approx(0.5)
        assert "Step 1" in body

    def test_knowledge_auto_frontmatter(self, tmp_path) -> None:
        """knowledge/ への Write で FM が自動付与される。"""
        mixin = _make_mixin(tmp_path)
        target = mixin._anima_dir / "knowledge" / "insight.md"

        handled = mixin._try_write_with_frontmatter(target, "# Insight\n\nDetail")
        assert handled is True

        mixin._memory.write_knowledge_with_meta.assert_called_once()
        text = target.read_text(encoding="utf-8")
        assert text.startswith("---")
        meta, body = parse_frontmatter(text)
        assert meta["confidence"] == pytest.approx(0.5)
        assert "created_at" in meta

    def test_skip_if_already_has_frontmatter(self, tmp_path) -> None:
        """既に FM があるコンテンツはスキップされる。"""
        mixin = _make_mixin(tmp_path)
        target = mixin._anima_dir / "knowledge" / "existing.md"

        content = "---\nconfidence: 0.9\n---\n\nBody"
        handled = mixin._try_write_with_frontmatter(target, content)
        assert handled is False

    def test_non_memory_path_unchanged(self, tmp_path) -> None:
        """memory 外のパスは通常書き込み（handled=False）。"""
        mixin = _make_mixin(tmp_path)
        target = mixin._anima_dir / "state" / "notes.md"
        target.parent.mkdir(parents=True, exist_ok=True)

        handled = mixin._try_write_with_frontmatter(target, "# Notes\n\nPlain")
        assert handled is False

    def test_non_md_file_unchanged(self, tmp_path) -> None:
        """non-.md ファイルは handled=False。"""
        mixin = _make_mixin(tmp_path)
        target = mixin._anima_dir / "knowledge" / "data.json"

        handled = mixin._try_write_with_frontmatter(target, '{"key": "value"}')
        assert handled is False

    def test_no_anima_dir_returns_false(self, tmp_path) -> None:
        """_anima_dir が None の場合は handled=False。"""
        mixin = FileToolsMixin.__new__(FileToolsMixin)
        mixin._anima_dir = None
        mixin._memory = MagicMock()

        target = tmp_path / "knowledge" / "test.md"
        handled = mixin._try_write_with_frontmatter(target, "content")
        assert handled is False
