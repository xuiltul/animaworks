"""Unit tests for core/memory/frontmatter.py — FrontmatterService."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest

from core.memory.frontmatter import FrontmatterService
from core.schemas import SkillMeta


# ── Fixtures ─────────────────────────────────────────────


@pytest.fixture
def anima_dir(tmp_path: Path) -> Path:
    d = tmp_path / "anima"
    d.mkdir()
    return d


@pytest.fixture
def knowledge_dir(anima_dir: Path) -> Path:
    d = anima_dir / "knowledge"
    d.mkdir()
    return d


@pytest.fixture
def procedures_dir(anima_dir: Path) -> Path:
    d = anima_dir / "procedures"
    d.mkdir()
    return d


@pytest.fixture
def svc(anima_dir: Path, knowledge_dir: Path, procedures_dir: Path) -> FrontmatterService:
    return FrontmatterService(
        anima_dir=anima_dir,
        knowledge_dir=knowledge_dir,
        procedures_dir=procedures_dir,
    )


# ── Knowledge frontmatter ────────────────────────────────


class TestWriteKnowledgeWithMeta:
    def test_writes_correct_yaml_frontmatter_and_body(
        self, svc: FrontmatterService, knowledge_dir: Path, monkeypatch,
    ) -> None:
        """write_knowledge_with_meta writes YAML frontmatter delimiters + body."""
        target = knowledge_dir / "topic.md"

        # monkeypatch atomic_write_text to use plain write for simplicity
        def _simple_write(path: Path, content: str, encoding: str = "utf-8") -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding=encoding)

        monkeypatch.setattr("core.memory.frontmatter.atomic_write_text", _simple_write)

        metadata = {"source": "test", "version": 1}
        svc.write_knowledge_with_meta(target, "Body text here", metadata)

        text = target.read_text(encoding="utf-8")
        assert text.startswith("---\n")
        assert "\n---\n" in text
        assert "source: test" in text
        assert "version: 1" in text
        assert "Body text here" in text


class TestReadKnowledgeContent:
    def test_strips_frontmatter_returns_body(
        self, svc: FrontmatterService, knowledge_dir: Path,
    ) -> None:
        """read_knowledge_content strips frontmatter and returns the body."""
        target = knowledge_dir / "topic.md"
        target.write_text(
            "---\nsource: test\n---\n\nBody text here",
            encoding="utf-8",
        )

        result = svc.read_knowledge_content(target)
        assert result == "Body text here"
        assert "source: test" not in result

    def test_returns_full_text_when_no_frontmatter(
        self, svc: FrontmatterService, knowledge_dir: Path,
    ) -> None:
        """read_knowledge_content returns full text when no frontmatter exists."""
        target = knowledge_dir / "plain.md"
        target.write_text("Just plain content", encoding="utf-8")

        result = svc.read_knowledge_content(target)
        assert result == "Just plain content"


class TestReadKnowledgeMetadata:
    def test_parses_yaml_frontmatter(
        self, svc: FrontmatterService, knowledge_dir: Path,
    ) -> None:
        """read_knowledge_metadata parses YAML frontmatter into a dict."""
        target = knowledge_dir / "topic.md"
        target.write_text(
            "---\nsource: test\nversion: 2\n---\n\nBody",
            encoding="utf-8",
        )

        meta = svc.read_knowledge_metadata(target)
        assert meta["source"] == "test"
        assert meta["version"] == 2

    def test_legacy_migration_superseded_at_to_valid_until(
        self, svc: FrontmatterService, knowledge_dir: Path,
    ) -> None:
        """read_knowledge_metadata renames superseded_at to valid_until."""
        target = knowledge_dir / "legacy.md"
        target.write_text(
            "---\nsuperseded_at: '2026-01-15'\n---\n\nOld content",
            encoding="utf-8",
        )

        meta = svc.read_knowledge_metadata(target)
        assert "valid_until" in meta
        assert meta["valid_until"] == "2026-01-15"
        assert "superseded_at" not in meta

    def test_returns_empty_dict_when_no_frontmatter(
        self, svc: FrontmatterService, knowledge_dir: Path,
    ) -> None:
        """read_knowledge_metadata returns empty dict when no frontmatter."""
        target = knowledge_dir / "no_meta.md"
        target.write_text("No frontmatter here", encoding="utf-8")

        meta = svc.read_knowledge_metadata(target)
        assert meta == {}


class TestUpdateKnowledgeMetadata:
    def test_merges_updates_into_existing_metadata(
        self, svc: FrontmatterService, knowledge_dir: Path, monkeypatch,
    ) -> None:
        """update_knowledge_metadata merges new keys into existing metadata."""
        target = knowledge_dir / "topic.md"

        # monkeypatch atomic_write_text so write_knowledge_with_meta works
        def _simple_write(path: Path, content: str, encoding: str = "utf-8") -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding=encoding)

        monkeypatch.setattr("core.memory.frontmatter.atomic_write_text", _simple_write)

        # Seed the file with initial metadata
        target.write_text(
            "---\nsource: original\nversion: 1\n---\n\nBody content",
            encoding="utf-8",
        )

        svc.update_knowledge_metadata(target, {"version": 2, "reviewed": True})

        meta = svc.read_knowledge_metadata(target)
        assert meta["source"] == "original"  # preserved
        assert meta["version"] == 2          # updated
        assert meta["reviewed"] is True      # added

        body = svc.read_knowledge_content(target)
        assert body == "Body content"


# ── Procedure frontmatter ────────────────────────────────


class TestWriteProcedureWithMeta:
    def test_writes_correct_format(
        self, svc: FrontmatterService, procedures_dir: Path,
    ) -> None:
        """write_procedure_with_meta writes YAML frontmatter + body."""
        target = procedures_dir / "deploy.md"
        metadata = {"description": "deploy procedure", "tags": ["deploy"]}
        svc.write_procedure_with_meta(target, "Step 1: deploy", metadata)

        text = target.read_text(encoding="utf-8")
        assert text.startswith("---\n")
        assert "\n---\n" in text
        assert "description: deploy procedure" in text
        assert "Step 1: deploy" in text


class TestReadProcedureContent:
    def test_strips_frontmatter(
        self, svc: FrontmatterService, procedures_dir: Path,
    ) -> None:
        """read_procedure_content strips frontmatter and returns body."""
        target = procedures_dir / "deploy.md"
        target.write_text(
            "---\ndescription: test\n---\n\nProcedure body",
            encoding="utf-8",
        )

        result = svc.read_procedure_content(target)
        assert result == "Procedure body"
        assert "description:" not in result

    def test_returns_empty_for_missing_file(
        self, svc: FrontmatterService, procedures_dir: Path,
    ) -> None:
        """read_procedure_content returns empty string for missing file."""
        result = svc.read_procedure_content(procedures_dir / "nonexistent.md")
        assert result == ""


class TestReadProcedureMetadata:
    def test_parses_yaml(
        self, svc: FrontmatterService, procedures_dir: Path,
    ) -> None:
        """read_procedure_metadata parses YAML frontmatter into a dict."""
        target = procedures_dir / "deploy.md"
        target.write_text(
            "---\ndescription: deploy\nconfidence: 0.9\n---\n\nBody",
            encoding="utf-8",
        )

        meta = svc.read_procedure_metadata(target)
        assert meta["description"] == "deploy"
        assert meta["confidence"] == 0.9

    def test_returns_empty_dict_for_missing_file(
        self, svc: FrontmatterService, procedures_dir: Path,
    ) -> None:
        """read_procedure_metadata returns empty dict for missing file."""
        meta = svc.read_procedure_metadata(procedures_dir / "nonexistent.md")
        assert meta == {}


class TestListProcedureMetas:
    def test_returns_skill_meta_list(
        self, svc: FrontmatterService, procedures_dir: Path,
    ) -> None:
        """list_procedure_metas returns SkillMeta list via extract function."""
        (procedures_dir / "alpha.md").write_text("alpha", encoding="utf-8")
        (procedures_dir / "beta.md").write_text("beta", encoding="utf-8")

        def fake_extract(f: Path, is_common: bool) -> SkillMeta:
            return SkillMeta(
                name=f.stem,
                description=f"desc of {f.stem}",
                path=f,
                is_common=is_common,
            )

        result = svc.list_procedure_metas(fake_extract)
        assert len(result) == 2
        names = [m.name for m in result]
        assert "alpha" in names
        assert "beta" in names
        assert all(isinstance(m, SkillMeta) for m in result)
        assert all(m.is_common is False for m in result)


class TestEnsureProcedureFrontmatter:
    def test_adds_frontmatter_to_bare_files(
        self, svc: FrontmatterService, anima_dir: Path, procedures_dir: Path,
    ) -> None:
        """ensure_procedure_frontmatter adds YAML frontmatter to bare files."""
        bare = procedures_dir / "my_procedure.md"
        bare.write_text("Bare procedure content", encoding="utf-8")

        count = svc.ensure_procedure_frontmatter()
        assert count == 1

        text = bare.read_text(encoding="utf-8")
        assert text.startswith("---\n")
        # No heading in content, so description falls back to filename stem
        assert "description: my procedure" in text
        assert "Bare procedure content" in text

    def test_description_from_heading(
        self, svc: FrontmatterService, procedures_dir: Path,
    ) -> None:
        """ensure_procedure_frontmatter extracts description from first heading."""
        f = procedures_dir / "deploy_app.md"
        f.write_text("# Deploy Application\n\n1. Pull\n2. Build", encoding="utf-8")

        count = svc.ensure_procedure_frontmatter()
        assert count == 1

        text = f.read_text(encoding="utf-8")
        assert "description: Deploy Application" in text

    def test_idempotent_per_file(
        self, svc: FrontmatterService, procedures_dir: Path,
    ) -> None:
        """ensure_procedure_frontmatter is idempotent per-file (checks frontmatter existence)."""
        bare = procedures_dir / "task.md"
        bare.write_text("Task content", encoding="utf-8")

        # First run
        count1 = svc.ensure_procedure_frontmatter()
        assert count1 == 1

        # Second run should be a no-op because frontmatter already exists
        count2 = svc.ensure_procedure_frontmatter()
        assert count2 == 0

    def test_skips_files_with_existing_frontmatter(
        self, svc: FrontmatterService, anima_dir: Path, procedures_dir: Path,
    ) -> None:
        """ensure_procedure_frontmatter skips files that already have frontmatter."""
        already_migrated = procedures_dir / "existing.md"
        already_migrated.write_text(
            "---\ndescription: already done\n---\n\nContent",
            encoding="utf-8",
        )
        bare = procedures_dir / "bare.md"
        bare.write_text("Bare content", encoding="utf-8")

        count = svc.ensure_procedure_frontmatter()
        assert count == 1  # only the bare file

        # The file with existing frontmatter should be unchanged
        text = already_migrated.read_text(encoding="utf-8")
        assert text == "---\ndescription: already done\n---\n\nContent"
