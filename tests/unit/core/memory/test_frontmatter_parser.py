from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the canonical line-based frontmatter parser.

Tests cover:
- split_frontmatter: line-based ``---`` detection
- parse_frontmatter: YAML parsing + body extraction
- strip_frontmatter: body-only extraction
- strip_content_frontmatter: double-frontmatter prevention
- Edge cases: inline ``---``, double frontmatter, BOM, empty YAML,
  no frontmatter, Windows line endings
"""

import pytest

from core.memory.frontmatter import (
    parse_frontmatter,
    split_frontmatter,
    strip_content_frontmatter,
    strip_frontmatter,
)


# ── split_frontmatter ────────────────────────────────────


class TestSplitFrontmatter:
    """Tests for split_frontmatter()."""

    def test_standard_frontmatter(self) -> None:
        text = "---\nkey: value\n---\n\nBody content"
        yaml_str, body = split_frontmatter(text)
        assert yaml_str == "key: value\n"
        assert body == "Body content"

    def test_no_frontmatter(self) -> None:
        text = "Just plain text"
        yaml_str, body = split_frontmatter(text)
        assert yaml_str == ""
        assert body == "Just plain text"

    def test_no_frontmatter_starts_with_hash(self) -> None:
        text = "# Heading\n\nBody"
        yaml_str, body = split_frontmatter(text)
        assert yaml_str == ""
        assert body == text

    def test_inline_dashes_in_yaml_value(self) -> None:
        """The core bug: ``---`` inside a YAML value must not split."""
        text = "---\ndescription: Before---After comparison\nconfidence: 0.8\n---\n\nBody content"
        yaml_str, body = split_frontmatter(text)
        assert "Before---After comparison" in yaml_str
        assert "confidence: 0.8" in yaml_str
        assert body == "Body content"

    def test_dashes_in_body(self) -> None:
        """Horizontal rules in body must not confuse the parser."""
        text = "---\nkey: value\n---\n\nBody\n\n---\n\nMore body"
        yaml_str, body = split_frontmatter(text)
        assert yaml_str == "key: value\n"
        assert "---" in body
        assert "More body" in body

    def test_double_frontmatter(self) -> None:
        """Double frontmatter: outer block is parsed, inner stays in body."""
        text = "---\nconfidence: 0.5\n---\n\n---\ntopic: something\n---\n\nActual content"
        yaml_str, body = split_frontmatter(text)
        assert "confidence: 0.5" in yaml_str
        assert "---" in body
        assert "Actual content" in body

    def test_empty_yaml_block(self) -> None:
        text = "---\n---\n\nBody"
        yaml_str, body = split_frontmatter(text)
        assert yaml_str == ""
        assert body == "Body"

    def test_closing_fence_with_trailing_spaces(self) -> None:
        text = "---\nkey: val\n---   \n\nBody"
        yaml_str, body = split_frontmatter(text)
        assert "key: val" in yaml_str
        assert body == "Body"

    def test_only_opening_fence(self) -> None:
        """File starts with ``---`` but no closing fence -> no frontmatter."""
        text = "---\nkey: value\nno closing fence"
        yaml_str, body = split_frontmatter(text)
        assert yaml_str == ""
        assert body == text

    def test_windows_line_endings(self) -> None:
        text = "---\r\nkey: value\r\n---\r\n\r\nBody"
        yaml_str, body = split_frontmatter(text)
        assert "key: value" in yaml_str
        assert "Body" in body.strip()

    def test_single_newline_after_closing_fence(self) -> None:
        text = "---\nkey: val\n---\nBody"
        yaml_str, body = split_frontmatter(text)
        assert yaml_str == "key: val\n"
        assert body == "Body"

    def test_multiline_yaml(self) -> None:
        text = (
            "---\n"
            "tags:\n"
            "  - python\n"
            "  - testing\n"
            "description: A test file\n"
            "---\n\n"
            "Body content"
        )
        yaml_str, body = split_frontmatter(text)
        assert "tags:" in yaml_str
        assert "  - python" in yaml_str
        assert "description: A test file" in yaml_str
        assert body == "Body content"


# ── parse_frontmatter ────────────────────────────────────


class TestParseFrontmatter:
    """Tests for parse_frontmatter()."""

    def test_parses_yaml_dict(self) -> None:
        text = "---\nconfidence: 0.8\ntags: [a, b]\n---\n\nBody"
        meta, body = parse_frontmatter(text)
        assert meta["confidence"] == pytest.approx(0.8)
        assert meta["tags"] == ["a", "b"]
        assert body == "Body"

    def test_no_frontmatter_returns_empty_dict(self) -> None:
        text = "No frontmatter"
        meta, body = parse_frontmatter(text)
        assert meta == {}
        assert body == "No frontmatter"

    def test_invalid_yaml_returns_empty_dict(self) -> None:
        text = "---\n: : : invalid\n---\n\nBody"
        meta, body = parse_frontmatter(text)
        assert meta == {}
        assert body == "Body"

    def test_yaml_non_dict_returns_empty_dict(self) -> None:
        text = "---\n- item1\n- item2\n---\n\nBody"
        meta, body = parse_frontmatter(text)
        assert meta == {}
        assert body == "Body"

    def test_inline_dashes_parsed_correctly(self) -> None:
        """YAML with ``---`` in value must parse all fields."""
        text = "---\ndescription: Phase 1---Phase 2\nconfidence: 0.9\n---\n\nContent"
        meta, body = parse_frontmatter(text)
        assert meta["description"] == "Phase 1---Phase 2"
        assert meta["confidence"] == pytest.approx(0.9)
        assert body == "Content"


# ── strip_frontmatter ────────────────────────────────────


class TestStripFrontmatter:
    """Tests for strip_frontmatter()."""

    def test_strips_frontmatter(self) -> None:
        text = "---\nkey: val\n---\n\nBody only"
        assert strip_frontmatter(text) == "Body only"

    def test_no_frontmatter_returns_original(self) -> None:
        text = "No frontmatter here"
        assert strip_frontmatter(text) == text


# ── strip_content_frontmatter ────────────────────────────


class TestStripContentFrontmatter:
    """Tests for strip_content_frontmatter()."""

    def test_strips_accidental_frontmatter(self) -> None:
        """LLM-generated content with frontmatter should be stripped."""
        content = "---\ntopic: something\n---\n\nActual content"
        result = strip_content_frontmatter(content)
        assert result == "Actual content"
        assert "---" not in result
        assert "topic" not in result

    def test_leading_whitespace_with_frontmatter(self) -> None:
        content = "  \n---\ntopic: x\n---\n\nContent"
        result = strip_content_frontmatter(content)
        assert result == "Content"

    def test_no_frontmatter_unchanged(self) -> None:
        content = "# Heading\n\nPlain content"
        result = strip_content_frontmatter(content)
        assert result == content

    def test_empty_content(self) -> None:
        assert strip_content_frontmatter("") == ""

    def test_dashes_in_middle_not_stripped(self) -> None:
        """Horizontal rules within content must not be treated as frontmatter."""
        content = "# Title\n\n---\n\nSection 2"
        result = strip_content_frontmatter(content)
        assert result == content


# ── Round-trip with write helpers ─────────────────────────


class TestWriteRoundTrip:
    """Verify write-then-read round-trip with double-frontmatter prevention."""

    def test_knowledge_write_strips_double_frontmatter(self, tmp_path) -> None:
        from core.memory.frontmatter import FrontmatterService

        kd = tmp_path / "knowledge"
        kd.mkdir()
        svc = FrontmatterService(tmp_path, kd, tmp_path / "procedures")

        content_with_fm = "---\nold_key: old_val\n---\n\nActual content"
        svc.write_knowledge_with_meta(
            kd / "test.md", content_with_fm, {"confidence": 0.8},
        )

        meta = svc.read_knowledge_metadata(kd / "test.md")
        body = svc.read_knowledge_content(kd / "test.md")

        assert meta["confidence"] == pytest.approx(0.8)
        assert "old_key" not in meta
        assert body == "Actual content"

    def test_procedure_write_strips_double_frontmatter(self, tmp_path) -> None:
        from core.memory.frontmatter import FrontmatterService

        pd = tmp_path / "procedures"
        pd.mkdir()
        svc = FrontmatterService(tmp_path, tmp_path / "knowledge", pd)

        content_with_fm = "---\nstale: true\n---\n\nStep 1: deploy"
        svc.write_procedure_with_meta(
            pd / "deploy.md", content_with_fm, {"description": "deploy"},
        )

        meta = svc.read_procedure_metadata(pd / "deploy.md")
        body = svc.read_procedure_content(pd / "deploy.md")

        assert meta["description"] == "deploy"
        assert "stale" not in meta
        assert body == "Step 1: deploy"

    def test_inline_dashes_survive_roundtrip(self, tmp_path) -> None:
        """Values containing ``---`` survive write -> read cycle."""
        from core.memory.frontmatter import FrontmatterService

        kd = tmp_path / "knowledge"
        kd.mkdir()
        svc = FrontmatterService(tmp_path, kd, tmp_path / "procedures")

        svc.write_knowledge_with_meta(
            kd / "test.md",
            "Body text here",
            {"description": "Before---After comparison", "confidence": 0.7},
        )

        meta = svc.read_knowledge_metadata(kd / "test.md")
        body = svc.read_knowledge_content(kd / "test.md")

        assert meta["description"] == "Before---After comparison"
        assert meta["confidence"] == pytest.approx(0.7)
        assert body == "Body text here"
