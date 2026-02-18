from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Unit tests for Phase 1: Procedure frontmatter foundation."""

import tempfile
from pathlib import Path

import pytest


# ── Fixtures ──────────────────────────────────────────────


@pytest.fixture
def anima_dir(tmp_path: Path) -> Path:
    """Create a minimal anima directory for MemoryManager."""
    d = tmp_path / "animas" / "test-anima"
    for sub in ("episodes", "knowledge", "procedures", "skills", "state"):
        (d / sub).mkdir(parents=True)
    return d


@pytest.fixture
def memory(anima_dir: Path, monkeypatch: pytest.MonkeyPatch):
    """Create a MemoryManager that skips RAG initialization."""
    monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(anima_dir.parent.parent))

    # Create required directories that MemoryManager expects
    data_dir = anima_dir.parent.parent
    (data_dir / "company").mkdir(parents=True, exist_ok=True)
    (data_dir / "common_skills").mkdir(parents=True, exist_ok=True)
    (data_dir / "common_knowledge").mkdir(parents=True, exist_ok=True)
    (data_dir / "shared" / "users").mkdir(parents=True, exist_ok=True)

    from core.memory.manager import MemoryManager

    return MemoryManager(anima_dir)


# ── 1-1: write_procedure_with_meta ────────────────────────


class TestWriteProcedureWithMeta:
    def test_writes_frontmatter_and_body(self, memory, anima_dir: Path) -> None:
        metadata = {
            "description": "Deploy procedure",
            "tags": ["deploy", "ops"],
            "success_count": 0,
            "failure_count": 0,
            "confidence": 0.5,
            "version": 1,
        }
        body = "# Deploy\n\n1. Pull latest\n2. Run deploy script"

        memory.write_procedure_with_meta(
            Path("deploy.md"), body, metadata,
        )

        path = anima_dir / "procedures" / "deploy.md"
        assert path.exists()
        text = path.read_text(encoding="utf-8")
        assert text.startswith("---\n")
        assert "description: Deploy procedure" in text
        assert "# Deploy" in text

    def test_writes_absolute_path(self, memory, tmp_path: Path) -> None:
        target = tmp_path / "custom" / "proc.md"
        memory.write_procedure_with_meta(
            target, "Body content", {"description": "test"},
        )
        assert target.exists()
        assert "description: test" in target.read_text(encoding="utf-8")


# ── 1-1: read_procedure_content ──────────────────────────


class TestReadProcedureContent:
    def test_strips_frontmatter(self, memory, anima_dir: Path) -> None:
        proc = anima_dir / "procedures" / "test.md"
        proc.write_text(
            "---\ndescription: test\n---\n\n# Body\n\nContent here",
            encoding="utf-8",
        )
        body = memory.read_procedure_content(Path("test.md"))
        assert body == "# Body\n\nContent here"

    def test_no_frontmatter(self, memory, anima_dir: Path) -> None:
        proc = anima_dir / "procedures" / "plain.md"
        proc.write_text("# Just content\n\nNo frontmatter", encoding="utf-8")
        body = memory.read_procedure_content(Path("plain.md"))
        assert body == "# Just content\n\nNo frontmatter"

    def test_missing_file(self, memory) -> None:
        assert memory.read_procedure_content(Path("nonexistent.md")) == ""


# ── 1-1: read_procedure_metadata ─────────────────────────


class TestReadProcedureMetadata:
    def test_parses_yaml(self, memory, anima_dir: Path) -> None:
        proc = anima_dir / "procedures" / "deploy.md"
        proc.write_text(
            "---\ndescription: deploy proc\ntags:\n- ops\nsuccess_count: 5\n---\n\n# Body",
            encoding="utf-8",
        )
        meta = memory.read_procedure_metadata(Path("deploy.md"))
        assert meta["description"] == "deploy proc"
        assert meta["tags"] == ["ops"]
        assert meta["success_count"] == 5

    def test_no_frontmatter_returns_empty(self, memory, anima_dir: Path) -> None:
        proc = anima_dir / "procedures" / "plain.md"
        proc.write_text("# Plain content", encoding="utf-8")
        assert memory.read_procedure_metadata(Path("plain.md")) == {}

    def test_missing_file_returns_empty(self, memory) -> None:
        assert memory.read_procedure_metadata(Path("missing.md")) == {}

    def test_malformed_yaml(self, memory, anima_dir: Path) -> None:
        proc = anima_dir / "procedures" / "bad.md"
        proc.write_text("---\n: invalid: yaml: [[[]\n---\n\nBody", encoding="utf-8")
        # Should not raise, returns empty dict
        meta = memory.read_procedure_metadata(Path("bad.md"))
        assert isinstance(meta, dict)


# ── 1-1: list_procedure_metas ────────────────────────────


class TestListProcedureMetas:
    def test_returns_skill_metas(self, memory, anima_dir: Path) -> None:
        for name, desc in [("deploy", "Deploy steps"), ("backup", "Backup procedure")]:
            (anima_dir / "procedures" / f"{name}.md").write_text(
                f"---\ndescription: {desc}\n---\n\n# {name}",
                encoding="utf-8",
            )

        metas = memory.list_procedure_metas()
        assert len(metas) == 2
        names = {m.name for m in metas}
        assert names == {"deploy", "backup"}
        assert all(m.description for m in metas)

    def test_empty_procedures(self, memory) -> None:
        assert memory.list_procedure_metas() == []


# ── 1-2: _validate_procedure_format ──────────────────────


class TestValidateProcedureFormat:
    def test_valid_procedure(self) -> None:
        from core.tooling.handler import _validate_procedure_format

        content = "---\ndescription: Good procedure\ntags: [ops]\n---\n\n# Steps\n1. Do stuff"
        assert _validate_procedure_format(content) == ""

    def test_missing_frontmatter(self) -> None:
        from core.tooling.handler import _validate_procedure_format

        content = "# No frontmatter\n\nJust content"
        result = _validate_procedure_format(content)
        assert "フロントマター" in result

    def test_missing_description(self) -> None:
        from core.tooling.handler import _validate_procedure_format

        content = "---\ntags: [test]\n---\n\n# Content"
        result = _validate_procedure_format(content)
        assert "description" in result

    def test_incomplete_frontmatter(self) -> None:
        from core.tooling.handler import _validate_procedure_format

        content = "---\nno closing"
        result = _validate_procedure_format(content)
        assert "フロントマター" in result


# ── 1-3: indexer _strip_frontmatter ──────────────────────


class TestStripFrontmatter:
    def test_strips_yaml_block(self) -> None:
        from core.memory.rag.indexer import MemoryIndexer

        content = "---\ndescription: test\ntags: [a]\n---\n\n# Body content"
        result = MemoryIndexer._strip_frontmatter(content)
        assert result == "# Body content"

    def test_no_frontmatter(self) -> None:
        from core.memory.rag.indexer import MemoryIndexer

        content = "# No frontmatter\nJust content"
        assert MemoryIndexer._strip_frontmatter(content) == content

    def test_incomplete_frontmatter(self) -> None:
        from core.memory.rag.indexer import MemoryIndexer

        content = "---\nonly opening"
        assert MemoryIndexer._strip_frontmatter(content) == content


# ── 1-4: migrate_legacy_procedures ───────────────────────


class TestMigrateLegacyProcedures:
    def test_migrates_plain_files(self, memory, anima_dir: Path) -> None:
        proc_dir = anima_dir / "procedures"
        (proc_dir / "deploy_app.md").write_text(
            "# Deploy App\n\n1. Pull\n2. Build\n3. Deploy",
            encoding="utf-8",
        )
        (proc_dir / "backup-db.md").write_text(
            "# Backup DB\n\n1. Dump\n2. Upload",
            encoding="utf-8",
        )

        count = memory.migrate_legacy_procedures()
        assert count == 2

        # Verify frontmatter was added
        for name in ("deploy_app.md", "backup-db.md"):
            text = (proc_dir / name).read_text(encoding="utf-8")
            assert text.startswith("---\n")
            assert "description:" in text

        # Verify backups
        backup_dir = anima_dir / "archive" / "pre_migration_procedures"
        assert (backup_dir / "deploy_app.md").exists()
        assert (backup_dir / "backup-db.md").exists()

        # Verify marker
        assert (proc_dir / ".migrated").exists()

    def test_skips_already_migrated(self, memory, anima_dir: Path) -> None:
        (anima_dir / "procedures" / ".migrated").write_text("done", encoding="utf-8")
        (anima_dir / "procedures" / "test.md").write_text("# Plain", encoding="utf-8")

        count = memory.migrate_legacy_procedures()
        assert count == 0  # marker blocks re-migration
        # File should remain unmodified
        text = (anima_dir / "procedures" / "test.md").read_text(encoding="utf-8")
        assert not text.startswith("---")

    def test_skips_files_with_frontmatter(self, memory, anima_dir: Path) -> None:
        proc_dir = anima_dir / "procedures"
        (proc_dir / "modern.md").write_text(
            "---\ndescription: Already modern\n---\n\n# Modern",
            encoding="utf-8",
        )
        (proc_dir / "legacy.md").write_text("# Legacy content", encoding="utf-8")

        count = memory.migrate_legacy_procedures()
        assert count == 1  # only legacy.md migrated

    def test_empty_directory(self, memory, anima_dir: Path) -> None:
        count = memory.migrate_legacy_procedures()
        assert count == 0
        assert (anima_dir / "procedures" / ".migrated").exists()

    def test_description_from_filename(self, memory, anima_dir: Path) -> None:
        proc_dir = anima_dir / "procedures"
        (proc_dir / "run_daily_backup.md").write_text("# Steps", encoding="utf-8")

        memory.migrate_legacy_procedures()
        meta = memory.read_procedure_metadata(Path("run_daily_backup.md"))
        assert meta["description"] == "run daily backup"
        assert meta["confidence"] == 0.5
        assert meta["version"] == 1
