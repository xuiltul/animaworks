from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for frontmatter validation, repair, and service repair methods."""

from pathlib import Path

import pytest

from core.memory.frontmatter import (
    FrontmatterService,
    parse_frontmatter,
    repair_double_frontmatter,
    validate_and_complete_frontmatter,
)


# ── validate_and_complete_frontmatter ────────────────────────────


class TestValidateAndCompleteFrontmatter:
    """Tests for validate_and_complete_frontmatter()."""

    def test_missing_created_at_filled(self) -> None:
        meta: dict = {"confidence": 0.7}
        result = validate_and_complete_frontmatter(meta)
        assert "created_at" in result
        assert result["confidence"] == 0.7

    def test_missing_confidence_filled(self) -> None:
        meta: dict = {"created_at": "2026-03-05T10:00:00"}
        result = validate_and_complete_frontmatter(meta)
        assert result["confidence"] == 0.5
        assert result["created_at"] == "2026-03-05T10:00:00"

    def test_both_present_no_overwrite(self) -> None:
        meta: dict = {"created_at": "2026-01-01T00:00:00", "confidence": 0.9}
        result = validate_and_complete_frontmatter(meta)
        assert result["created_at"] == "2026-01-01T00:00:00"
        assert result["confidence"] == 0.9

    def test_both_missing_filled(self) -> None:
        meta: dict = {"version": 1}
        result = validate_and_complete_frontmatter(meta)
        assert "created_at" in result
        assert result["confidence"] == 0.5
        assert result["version"] == 1

    def test_uses_file_mtime_when_path_exists(self, tmp_path: Path) -> None:
        f = tmp_path / "test.md"
        f.write_text("content", encoding="utf-8")
        meta: dict = {"version": 1}
        result = validate_and_complete_frontmatter(meta, f)
        assert "created_at" in result
        # Should be from file mtime, not now_iso()
        assert "T" in result["created_at"]

    def test_nonexistent_path_uses_now(self, tmp_path: Path) -> None:
        f = tmp_path / "nonexistent.md"
        meta: dict = {}
        result = validate_and_complete_frontmatter(meta, f)
        assert "created_at" in result


# ── repair_double_frontmatter ──────────────────────────────────


class TestRepairDoubleFrontmatter:
    """Tests for repair_double_frontmatter()."""

    def test_double_frontmatter_repaired(self, tmp_path: Path) -> None:
        f = tmp_path / "double.md"
        f.write_text(
            "---\nconfidence: 0.7\n---\n---\ncreated_at: '2026-01-01'\ntitle: test\n---\n\nBody content\n",
            encoding="utf-8",
        )
        assert repair_double_frontmatter(f) is True

        meta, body = parse_frontmatter(f.read_text(encoding="utf-8"))
        assert meta["confidence"] == 0.7
        assert meta["created_at"] == "2026-01-01"
        assert meta["title"] == "test"
        assert "Body content" in body

    def test_normal_file_no_op(self, tmp_path: Path) -> None:
        f = tmp_path / "normal.md"
        f.write_text(
            "---\nconfidence: 0.7\ncreated_at: '2026-01-01'\n---\n\nBody content\n",
            encoding="utf-8",
        )
        assert repair_double_frontmatter(f) is False

    def test_no_frontmatter_no_op(self, tmp_path: Path) -> None:
        f = tmp_path / "plain.md"
        f.write_text("Just plain text\n", encoding="utf-8")
        assert repair_double_frontmatter(f) is False

    def test_nonexistent_file(self, tmp_path: Path) -> None:
        f = tmp_path / "nonexistent.md"
        assert repair_double_frontmatter(f) is False

    def test_outer_takes_precedence(self, tmp_path: Path) -> None:
        f = tmp_path / "conflict.md"
        f.write_text(
            "---\nconfidence: 0.9\n---\n---\nconfidence: 0.3\nextra: value\n---\n\nBody\n",
            encoding="utf-8",
        )
        repair_double_frontmatter(f)
        meta, _ = parse_frontmatter(f.read_text(encoding="utf-8"))
        assert meta["confidence"] == 0.9
        assert meta["extra"] == "value"


# ── FrontmatterService.repair_knowledge_frontmatter ─────────────


class TestRepairKnowledgeFrontmatter:
    """Integration tests for repair_knowledge_frontmatter()."""

    def test_repairs_double_and_missing_fields(self, tmp_path: Path) -> None:
        knowledge_dir = tmp_path / "knowledge"
        procedures_dir = tmp_path / "procedures"
        knowledge_dir.mkdir()
        procedures_dir.mkdir()

        # File with double frontmatter and missing confidence
        (knowledge_dir / "double.md").write_text(
            "---\ncreated_at: '2026-01-01'\n---\n---\ntitle: test\n---\n\nContent\n",
            encoding="utf-8",
        )
        # File with missing created_at
        (knowledge_dir / "incomplete.md").write_text(
            "---\nversion: 1\n---\n\nContent here\n",
            encoding="utf-8",
        )
        # File that is already correct — should not be counted
        (knowledge_dir / "ok.md").write_text(
            "---\nconfidence: 0.8\ncreated_at: '2026-02-01'\n---\n\nOK\n",
            encoding="utf-8",
        )

        svc = FrontmatterService(tmp_path, knowledge_dir, procedures_dir)
        count = svc.repair_knowledge_frontmatter()

        assert count == 2

        # Verify double was merged
        meta1, body1 = parse_frontmatter(
            (knowledge_dir / "double.md").read_text(encoding="utf-8"),
        )
        assert meta1["created_at"] == "2026-01-01"
        assert "confidence" in meta1

        # Verify incomplete got missing fields
        meta2, _ = parse_frontmatter(
            (knowledge_dir / "incomplete.md").read_text(encoding="utf-8"),
        )
        assert "created_at" in meta2
        assert "confidence" in meta2

    def test_repairs_unparseable_frontmatter(self, tmp_path: Path) -> None:
        knowledge_dir = tmp_path / "knowledge"
        procedures_dir = tmp_path / "procedures"
        knowledge_dir.mkdir()
        procedures_dir.mkdir()

        (knowledge_dir / "broken.md").write_text(
            "---\n{invalid yaml: [unclosed\n---\n\nActual body content.\n",
            encoding="utf-8",
        )

        svc = FrontmatterService(tmp_path, knowledge_dir, procedures_dir)
        count = svc.repair_knowledge_frontmatter()

        assert count == 1
        text = (knowledge_dir / "broken.md").read_text(encoding="utf-8")
        meta, body = parse_frontmatter(text)
        assert meta, "Repaired file must have parseable frontmatter"
        assert meta["confidence"] == 0.5
        assert "created_at" in meta
        assert "Actual body content" in body

    def test_unparseable_without_dash_prefix_not_touched(self, tmp_path: Path) -> None:
        knowledge_dir = tmp_path / "knowledge"
        procedures_dir = tmp_path / "procedures"
        knowledge_dir.mkdir()
        procedures_dir.mkdir()

        (knowledge_dir / "plain.md").write_text(
            "Plain text without any frontmatter.\n",
            encoding="utf-8",
        )

        svc = FrontmatterService(tmp_path, knowledge_dir, procedures_dir)
        count = svc.repair_knowledge_frontmatter()
        assert count == 0

    def test_unparseable_body_preserved(self, tmp_path: Path) -> None:
        knowledge_dir = tmp_path / "knowledge"
        procedures_dir = tmp_path / "procedures"
        knowledge_dir.mkdir()
        procedures_dir.mkdir()

        (knowledge_dir / "garbled.md").write_text(
            "---\n!!!not-yaml!!!\n---\n\n# Important Knowledge\n\nThis must be preserved.\n",
            encoding="utf-8",
        )

        svc = FrontmatterService(tmp_path, knowledge_dir, procedures_dir)
        count = svc.repair_knowledge_frontmatter()
        assert count == 1

        text = (knowledge_dir / "garbled.md").read_text(encoding="utf-8")
        meta, body = parse_frontmatter(text)
        assert "Important Knowledge" in body
        assert "This must be preserved" in body

    def test_empty_directory(self, tmp_path: Path) -> None:
        knowledge_dir = tmp_path / "knowledge"
        procedures_dir = tmp_path / "procedures"
        knowledge_dir.mkdir()
        procedures_dir.mkdir()

        svc = FrontmatterService(tmp_path, knowledge_dir, procedures_dir)
        assert svc.repair_knowledge_frontmatter() == 0

    def test_nonexistent_directory(self, tmp_path: Path) -> None:
        svc = FrontmatterService(
            tmp_path, tmp_path / "knowledge", tmp_path / "procedures",
        )
        assert svc.repair_knowledge_frontmatter() == 0


class TestRepairProcedureFrontmatter:
    """Integration tests for repair_procedure_frontmatter()."""

    def test_repairs_missing_fields(self, tmp_path: Path) -> None:
        knowledge_dir = tmp_path / "knowledge"
        procedures_dir = tmp_path / "procedures"
        knowledge_dir.mkdir()
        procedures_dir.mkdir()

        (procedures_dir / "proc.md").write_text(
            "---\ndescription: test proc\n---\n\n# Steps\n1. Do thing\n",
            encoding="utf-8",
        )

        svc = FrontmatterService(tmp_path, knowledge_dir, procedures_dir)
        count = svc.repair_procedure_frontmatter()

        assert count == 1
        meta, _ = parse_frontmatter(
            (procedures_dir / "proc.md").read_text(encoding="utf-8"),
        )
        assert "confidence" in meta
        assert "created_at" in meta
