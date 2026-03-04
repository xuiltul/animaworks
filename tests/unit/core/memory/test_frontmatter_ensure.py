from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for FrontmatterService ensure_*_frontmatter and _extract_description.

Tests cover:
- ensure_knowledge_frontmatter: auto-add FM to bare knowledge files
- ensure_procedure_frontmatter: auto-add FM to bare procedure files
- _extract_description: heading extraction + fallback
"""

import pytest

from core.memory.frontmatter import FrontmatterService, parse_frontmatter


# ── ensure_knowledge_frontmatter ─────────────────────────


class TestEnsureKnowledgeFrontmatter:
    """Tests for FrontmatterService.ensure_knowledge_frontmatter()."""

    def _make_service(self, tmp_path):
        kd = tmp_path / "knowledge"
        kd.mkdir()
        pd = tmp_path / "procedures"
        pd.mkdir()
        return FrontmatterService(tmp_path, kd, pd), kd

    def test_adds_frontmatter_to_files_without(self, tmp_path) -> None:
        """FM なしの knowledge ファイルに FM が付与される。"""
        svc, kd = self._make_service(tmp_path)

        (kd / "topic_a.md").write_text("# Topic A\n\nSome insight.", encoding="utf-8")
        (kd / "topic_b.md").write_text("Plain text content", encoding="utf-8")

        result = svc.ensure_knowledge_frontmatter()
        assert result == 2

        for name in ("topic_a.md", "topic_b.md"):
            text = (kd / name).read_text(encoding="utf-8")
            assert text.startswith("---")
            meta, _ = parse_frontmatter(text)
            assert meta["confidence"] == pytest.approx(0.5)
            assert "created_at" in meta
            assert "updated_at" in meta

    def test_skips_files_with_existing_frontmatter(self, tmp_path) -> None:
        """既存 FM ありファイルはスキップされる。"""
        svc, kd = self._make_service(tmp_path)

        original = "---\nconfidence: 0.9\n---\n\nExisting content"
        (kd / "existing.md").write_text(original, encoding="utf-8")

        result = svc.ensure_knowledge_frontmatter()
        assert result == 0
        assert (kd / "existing.md").read_text(encoding="utf-8") == original

    def test_empty_directory(self, tmp_path) -> None:
        """空ディレクトリでは 0 を返す。"""
        svc, _ = self._make_service(tmp_path)
        assert svc.ensure_knowledge_frontmatter() == 0

    def test_nonexistent_directory(self, tmp_path) -> None:
        """存在しないディレクトリでは 0 を返す。"""
        svc = FrontmatterService(
            tmp_path,
            tmp_path / "nonexistent_knowledge",
            tmp_path / "procedures",
        )
        assert svc.ensure_knowledge_frontmatter() == 0


# ── ensure_procedure_frontmatter / _extract_description ──


class TestEnsureProcedureFrontmatter:
    """Tests for FrontmatterService.ensure_procedure_frontmatter()."""

    def _make_service(self, tmp_path):
        kd = tmp_path / "knowledge"
        kd.mkdir()
        pd = tmp_path / "procedures"
        pd.mkdir()
        return FrontmatterService(tmp_path, kd, pd), pd

    def test_adds_frontmatter_to_bare_procedures(self, tmp_path) -> None:
        """FM なしの procedure ファイルに FM が付与される。"""
        svc, pd = self._make_service(tmp_path)

        (pd / "deploy.md").write_text("# Deploy to production\n\nStep 1…", encoding="utf-8")

        result = svc.ensure_procedure_frontmatter()
        assert result == 1

        text = (pd / "deploy.md").read_text(encoding="utf-8")
        meta, body = parse_frontmatter(text)
        assert meta["description"] == "Deploy to production"
        assert meta["confidence"] == pytest.approx(0.5)
        assert "Step 1" in body

    def test_skips_existing_frontmatter(self, tmp_path) -> None:
        svc, pd = self._make_service(tmp_path)

        original = "---\ndescription: existing\n---\n\nContent"
        (pd / "existing.md").write_text(original, encoding="utf-8")

        assert svc.ensure_procedure_frontmatter() == 0
        assert (pd / "existing.md").read_text(encoding="utf-8") == original


class TestExtractDescription:
    """Tests for FrontmatterService._extract_description()."""

    def test_extracts_heading(self) -> None:
        """# 見出しから description を抽出する。"""
        text = "# Deploy Workflow\n\nStep 1: build\nStep 2: push"
        result = FrontmatterService._extract_description(text, "fallback_name")
        assert result == "Deploy Workflow"

    def test_fallback_to_name(self) -> None:
        """見出しなしではファイル名にフォールバックする。"""
        text = "No heading here, just steps.\nStep 1: build"
        result = FrontmatterService._extract_description(text, "my_deploy-process")
        assert result == "my deploy process"
