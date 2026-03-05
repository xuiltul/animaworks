from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for handler_memory.py — parse-failed frontmatter fallback."""

import threading
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from core.memory.frontmatter import parse_frontmatter
from core.tooling.handler_memory import MemoryToolsMixin


# ── Fixture ──────────────────────────────────────────────


class _FakeWriteHandler(MemoryToolsMixin):
    """Minimal stub exercising _handle_write_memory_file."""

    def __init__(self, anima_dir: Path) -> None:
        self._anima_dir = anima_dir
        self._anima_name = "test"
        self._superuser = True
        self._memory = MagicMock()
        self._memory._get_indexer.return_value = None
        self._activity = MagicMock()
        self._subordinate_activity_dirs: list[Path] = []
        self._subordinate_management_files: list[Path] = []
        self._descendant_activity_dirs: list[Path] = []
        self._descendant_state_files: list[Path] = []
        self._descendant_state_dirs: list[Path] = []
        self._peer_activity_dirs: list[Path] = []
        self._state_file_lock: threading.Lock | None = None
        self._on_schedule_changed = None
        self._min_trust_seen = 2

    def _is_state_file(self, path: Path) -> bool:
        return False

    def _check_tool_creation_permission(self, kind: str) -> bool:
        return True


@pytest.fixture()
def handler(tmp_path: Path) -> _FakeWriteHandler:
    anima_dir = tmp_path / "anima"
    anima_dir.mkdir()
    (anima_dir / "knowledge").mkdir()
    return _FakeWriteHandler(anima_dir)


# ── Tests: parse-failed frontmatter fallback ─────────────


class TestParsefailedFrontmatterFallback:
    """When LLM writes broken YAML frontmatter, fallback FM is generated."""

    def test_broken_yaml_gets_valid_frontmatter(self, handler: _FakeWriteHandler) -> None:
        broken_content = "---\n{invalid yaml: [unclosed\n---\n\nActual knowledge body here."
        handler._handle_write_memory_file({
            "path": "knowledge/test.md",
            "content": broken_content,
            "mode": "overwrite",
        })

        path = handler._anima_dir / "knowledge" / "test.md"
        text = path.read_text(encoding="utf-8")
        meta, body = parse_frontmatter(text)

        assert meta, "Saved file must have parseable frontmatter"
        assert meta["confidence"] == 0.5
        assert "created_at" in meta
        assert "updated_at" in meta
        assert "Actual knowledge body here" in body

    def test_auto_frontmatter_applied_prevents_origin_double_injection(
        self, handler: _FakeWriteHandler,
    ) -> None:
        handler._min_trust_seen = 0
        broken_content = "---\n{broken: yaml\n---\n\nBody text."
        handler._handle_write_memory_file({
            "path": "knowledge/trust.md",
            "content": broken_content,
            "mode": "overwrite",
        })

        path = handler._anima_dir / "knowledge" / "trust.md"
        text = path.read_text(encoding="utf-8")
        assert text.count("---") == 2, "Should have exactly one frontmatter block (2 fences)"

    def test_preserves_created_at_on_overwrite(self, handler: _FakeWriteHandler) -> None:
        path = handler._anima_dir / "knowledge" / "existing.md"
        path.write_text(
            "---\nconfidence: 0.8\ncreated_at: '2026-01-15T10:00:00+09:00'\n---\n\nOld content\n",
            encoding="utf-8",
        )

        broken_content = "---\n!!!broken!!!\n---\n\nNew content here."
        handler._handle_write_memory_file({
            "path": "knowledge/existing.md",
            "content": broken_content,
            "mode": "overwrite",
        })

        text = path.read_text(encoding="utf-8")
        meta, body = parse_frontmatter(text)
        assert meta["created_at"] == "2026-01-15T10:00:00+09:00"
        assert "New content here" in body

    def test_valid_yaml_frontmatter_still_works(self, handler: _FakeWriteHandler) -> None:
        valid_content = "---\nconfidence: 0.9\ntitle: my knowledge\n---\n\nGood body."
        handler._handle_write_memory_file({
            "path": "knowledge/valid.md",
            "content": valid_content,
            "mode": "overwrite",
        })

        path = handler._anima_dir / "knowledge" / "valid.md"
        text = path.read_text(encoding="utf-8")
        meta, body = parse_frontmatter(text)
        assert meta["confidence"] == 0.9
        assert "Good body" in body

    def test_content_without_frontmatter_not_affected(self, handler: _FakeWriteHandler) -> None:
        plain_content = "# Knowledge Title\n\nSome knowledge without frontmatter."
        handler._handle_write_memory_file({
            "path": "knowledge/plain.md",
            "content": plain_content,
            "mode": "overwrite",
        })

        path = handler._anima_dir / "knowledge" / "plain.md"
        text = path.read_text(encoding="utf-8")
        meta, body = parse_frontmatter(text)
        assert meta, "Framework should add frontmatter for plain knowledge"
        assert meta["confidence"] == 0.5
