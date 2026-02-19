# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for _build_full_org_tree and top-level org context.

Verifies the indented tree formatting, marker placement, and
speciality annotations in the organization tree builder.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock


def _make_animas_dict(specs: list[tuple[str, str | None, str | None]]) -> dict:
    """Build a mock animas dict from (name, supervisor, speciality) tuples."""
    animas = {}
    for name, sup, spec in specs:
        m = MagicMock()
        m.supervisor = sup
        m.speciality = spec
        animas[name] = m
    return animas


class TestBuildFullOrgTree:
    """Tests for _build_full_org_tree formatting."""

    def test_simple_hierarchy(self):
        from core.prompt.builder import _build_full_org_tree

        animas = _make_animas_dict([
            ("sakura", None, "経営"),
            ("rin", "sakura", "開発"),
            ("kotoha", "sakura", "広報"),
            ("aoi", "rin", None),
        ])

        tree = _build_full_org_tree("sakura", animas)
        assert "sakura" in tree
        assert "rin" in tree
        assert "kotoha" in tree
        assert "aoi" in tree
        assert "← あなた" in tree

    def test_single_anima(self):
        from core.prompt.builder import _build_full_org_tree

        animas = _make_animas_dict([("sakura", None, None)])

        tree = _build_full_org_tree("sakura", animas)
        assert "sakura" in tree
        assert "← あなた" in tree

    def test_tree_formatting_branch_markers(self):
        """Multiple children should use tree branch markers."""
        from core.prompt.builder import _build_full_org_tree

        animas = _make_animas_dict([
            ("sakura", None, None),
            ("a", "sakura", None),
            ("b", "sakura", None),
        ])

        tree = _build_full_org_tree("sakura", animas)
        # First child uses intermediate branch, last uses terminal branch
        assert "├── " in tree
        assert "└── " in tree

    def test_you_marker_on_non_root(self):
        """The 'you' marker should appear next to the target anima even if non-root."""
        from core.prompt.builder import _build_full_org_tree

        animas = _make_animas_dict([
            ("sakura", None, "経営"),
            ("rin", "sakura", "開発"),
        ])

        tree = _build_full_org_tree("rin", animas)
        # rin should have the marker, sakura should not
        for line in tree.splitlines():
            if "rin" in line:
                assert "← あなた" in line
            if "sakura" in line and "rin" not in line:
                assert "← あなた" not in line

    def test_speciality_annotation(self):
        """Anima entries should include speciality in parentheses."""
        from core.prompt.builder import _build_full_org_tree

        animas = _make_animas_dict([
            ("sakura", None, "経営"),
            ("rin", "sakura", "開発"),
        ])

        tree = _build_full_org_tree("sakura", animas)
        assert "sakura (経営)" in tree
        assert "rin (開発)" in tree

    def test_no_speciality_no_parens(self):
        """Anima entries without speciality should show just the name."""
        from core.prompt.builder import _build_full_org_tree

        animas = _make_animas_dict([
            ("sakura", None, None),
            ("rin", "sakura", None),
        ])

        tree = _build_full_org_tree("sakura", animas)
        # "sakura" should appear without parens (but may have arrow marker)
        for line in tree.splitlines():
            if "sakura" in line:
                assert "()" not in line

    def test_deep_hierarchy(self):
        """Three levels of hierarchy should produce proper indentation."""
        from core.prompt.builder import _build_full_org_tree

        animas = _make_animas_dict([
            ("sakura", None, "CEO"),
            ("rin", "sakura", "CTO"),
            ("aoi", "rin", "Dev"),
            ("kotoha", "sakura", "PR"),
        ])

        tree = _build_full_org_tree("sakura", animas)
        lines = tree.splitlines()

        # All four names must appear
        all_text = tree
        assert "sakura" in all_text
        assert "rin" in all_text
        assert "aoi" in all_text
        assert "kotoha" in all_text

        # aoi should be indented more than rin (child of rin)
        rin_line = next(l for l in lines if "rin" in l)
        aoi_line = next(l for l in lines if "aoi" in l)
        # aoi has deeper indentation (more leading chars before name)
        rin_indent = len(rin_line) - len(rin_line.lstrip())
        aoi_indent = len(aoi_line) - len(aoi_line.lstrip())
        # aoi's prefix (including tree chars) should be longer than rin's
        assert len(aoi_line.split("aoi")[0]) > len(rin_line.split("rin")[0])

    def test_multiple_roots(self):
        """When there are multiple root animas (supervisor=None), all appear."""
        from core.prompt.builder import _build_full_org_tree

        animas = _make_animas_dict([
            ("sakura", None, None),
            ("rin", None, None),
            ("aoi", "sakura", None),
        ])

        tree = _build_full_org_tree("sakura", animas)
        assert "sakura" in tree
        assert "rin" in tree
        assert "aoi" in tree


class TestFormatAnimaEntry:
    """Tests for _format_anima_entry helper."""

    def test_with_speciality(self):
        from core.prompt.builder import _format_anima_entry

        result = _format_anima_entry("rin", "開発")
        assert result == "rin (開発)"

    def test_without_speciality(self):
        from core.prompt.builder import _format_anima_entry

        result = _format_anima_entry("rin", None)
        assert result == "rin"

    def test_empty_speciality(self):
        from core.prompt.builder import _format_anima_entry

        # None means no speciality
        result = _format_anima_entry("rin", None)
        assert "(" not in result
