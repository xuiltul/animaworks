from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for handler_memory.py — parse-failed frontmatter fallback."""

import threading
from pathlib import Path
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
        self._runtime_session_context = None
        self._read_paths: set[str] = set()

    def _is_state_file(self, path: Path) -> bool:
        return False

    def _check_tool_creation_permission(self, kind: str) -> bool:
        return True

    def _check_file_permission(self, path: str, write: bool = False) -> None:
        return None


@pytest.fixture()
def handler(tmp_path: Path) -> _FakeWriteHandler:
    anima_dir = tmp_path / "anima"
    anima_dir.mkdir()
    (anima_dir / "knowledge").mkdir()
    return _FakeWriteHandler(anima_dir)


# ── Tests: parse-failed frontmatter fallback ─────────────


def test_case_record_knowledge_write_completes_with_episodes_hint(handler: _FakeWriteHandler) -> None:
    content = "Release follow-up for PR #1842, commit deadbee, on 2026-09-28."

    result = handler._handle_write_memory_file(
        {"path": "knowledge/release-note.md", "content": content, "mode": "overwrite"}
    )

    assert "Written to knowledge/release-note.md" in result
    assert "episodes/" in result
    assert (handler._anima_dir / "knowledge" / "release-note.md").is_file()
    assert "Release follow-up" in (handler._anima_dir / "knowledge" / "release-note.md").read_text()


def test_regular_knowledge_write_has_no_case_record_hint(handler: _FakeWriteHandler) -> None:
    result = handler._handle_write_memory_file(
        {
            "path": "knowledge/stable-setting.md",
            "content": "The stable retry delay is 30 seconds.",
            "mode": "overwrite",
        }
    )

    assert "Written to knowledge/stable-setting.md" in result
    assert "episodes/" not in result


class TestParsefailedFrontmatterFallback:
    """When LLM writes broken YAML frontmatter, fallback FM is generated."""

    def test_broken_yaml_gets_valid_frontmatter(self, handler: _FakeWriteHandler) -> None:
        broken_content = "---\n{invalid yaml: [unclosed\n---\n\nActual knowledge body here."
        handler._handle_write_memory_file(
            {
                "path": "knowledge/test.md",
                "content": broken_content,
                "mode": "overwrite",
            }
        )

        path = handler._anima_dir / "knowledge" / "test.md"
        text = path.read_text(encoding="utf-8")
        meta, body = parse_frontmatter(text)

        assert meta, "Saved file must have parseable frontmatter"
        assert meta["confidence"] == 0.5
        assert "created_at" in meta
        assert "updated_at" in meta
        assert "Actual knowledge body here" in body

    def test_auto_frontmatter_applied_prevents_origin_double_injection(
        self,
        handler: _FakeWriteHandler,
    ) -> None:
        handler._min_trust_seen = 0
        broken_content = "---\n{broken: yaml\n---\n\nBody text."
        handler._handle_write_memory_file(
            {
                "path": "knowledge/trust.md",
                "content": broken_content,
                "mode": "overwrite",
            }
        )

        path = handler._anima_dir / "knowledge" / "trust.md"
        text = path.read_text(encoding="utf-8")
        assert text.count("---") == 2, "Should have exactly one frontmatter block (2 fences)"

    def test_preserves_created_at_on_overwrite(self, handler: _FakeWriteHandler) -> None:
        path = handler._anima_dir / "knowledge" / "existing.md"
        path.write_text(
            "---\nconfidence: 0.8\ncreated_at: '2026-01-15T10:00:00+09:00'\n---\n\nOld content\n",
            encoding="utf-8",
        )

        handler._read_paths.add("knowledge/existing.md")
        broken_content = "---\n!!!broken!!!\n---\n\nNew content here."
        handler._handle_write_memory_file(
            {
                "path": "knowledge/existing.md",
                "content": broken_content,
                "mode": "overwrite",
            }
        )

        text = path.read_text(encoding="utf-8")
        meta, body = parse_frontmatter(text)
        assert meta["created_at"] == "2026-01-15T10:00:00+09:00"
        assert "New content here" in body

    def test_valid_yaml_frontmatter_still_works(self, handler: _FakeWriteHandler) -> None:
        valid_content = "---\nconfidence: 0.9\ntitle: my knowledge\n---\n\nGood body."
        handler._handle_write_memory_file(
            {
                "path": "knowledge/valid.md",
                "content": valid_content,
                "mode": "overwrite",
            }
        )

        path = handler._anima_dir / "knowledge" / "valid.md"
        text = path.read_text(encoding="utf-8")
        meta, body = parse_frontmatter(text)
        assert meta["confidence"] == 0.9
        assert "Good body" in body

    def test_plain_knowledge_write_applies_mixed_origin_once(self, handler: _FakeWriteHandler) -> None:
        handler._min_trust_seen = 1
        handler._handle_write_memory_file(
            {"path": "knowledge/mixed.md", "content": "# Mixed knowledge", "mode": "overwrite"}
        )

        text = (handler._anima_dir / "knowledge" / "mixed.md").read_text(encoding="utf-8")
        meta, body = parse_frontmatter(text)
        assert meta["origin"] == "mixed"
        assert text.count("---") == 2
        assert "# Mixed knowledge" in body

    def test_llm_frontmatter_origin_is_downgraded_in_place(self, handler: _FakeWriteHandler) -> None:
        handler._min_trust_seen = 0
        handler._handle_write_memory_file(
            {
                "path": "knowledge/llm-frontmatter.md",
                "content": "---\ntitle: Source\norigin: human\n---\n\n# Knowledge body",
                "mode": "overwrite",
            }
        )

        text = (handler._anima_dir / "knowledge" / "llm-frontmatter.md").read_text(encoding="utf-8")
        meta, body = parse_frontmatter(text)
        assert meta["origin"] == "external_web"
        assert text.count("---") == 2
        assert "# Knowledge body" in body

    def test_legacy_shared_trust_file_is_ignored_without_runtime_context(self, handler: _FakeWriteHandler) -> None:
        (handler._anima_dir / "run").mkdir()
        (handler._anima_dir / "run" / "min_trust_seen").write_text("0", encoding="utf-8")
        handler._handle_write_memory_file(
            {"path": "knowledge/no-context.md", "content": "# Local knowledge", "mode": "overwrite"}
        )

        meta, _ = parse_frontmatter((handler._anima_dir / "knowledge" / "no-context.md").read_text(encoding="utf-8"))
        assert "origin" not in meta

    def test_other_session_trust_file_is_not_used(self, handler: _FakeWriteHandler) -> None:
        from core.execution.session.session_context import RuntimeSessionContext
        from core.trust import record_session_trust

        own_context = RuntimeSessionContext.create(session_type="chat", thread_id="t", trigger="chat")
        other_context = RuntimeSessionContext.create(session_type="chat", thread_id="t", trigger="chat")
        handler._runtime_session_context = own_context
        record_session_trust(handler._anima_dir, other_context.tool_session_id, 0)
        handler._handle_write_memory_file(
            {"path": "knowledge/isolated.md", "content": "# Isolated knowledge", "mode": "overwrite"}
        )

        meta, _ = parse_frontmatter((handler._anima_dir / "knowledge" / "isolated.md").read_text(encoding="utf-8"))
        assert "origin" not in meta

    def test_append_downgrades_existing_origin_only_when_needed(self, handler: _FakeWriteHandler) -> None:
        path = handler._anima_dir / "knowledge" / "append.md"
        path.write_text("---\norigin: mixed\n---\n\nExisting body\n", encoding="utf-8")
        handler._min_trust_seen = 0
        handler._handle_write_memory_file(
            {"path": "knowledge/append.md", "content": "Appended external data", "mode": "append"}
        )
        meta, body = parse_frontmatter(path.read_text(encoding="utf-8"))
        assert meta["origin"] == "external_web"
        assert "Appended external data" in body

        path.write_text("---\norigin: mixed\n---\n\nExisting body\n", encoding="utf-8")
        handler._min_trust_seen = 1
        handler._handle_write_memory_file(
            {"path": "knowledge/append.md", "content": "Appended mixed data", "mode": "append"}
        )
        meta, body = parse_frontmatter(path.read_text(encoding="utf-8"))
        assert meta["origin"] == "mixed"
        assert "Appended mixed data" in body

    def test_content_without_frontmatter_not_affected(self, handler: _FakeWriteHandler) -> None:
        plain_content = "# Knowledge Title\n\nSome knowledge without frontmatter."
        handler._handle_write_memory_file(
            {
                "path": "knowledge/plain.md",
                "content": plain_content,
                "mode": "overwrite",
            }
        )

        path = handler._anima_dir / "knowledge" / "plain.md"
        text = path.read_text(encoding="utf-8")
        meta, body = parse_frontmatter(text)
        assert meta, "Framework should add frontmatter for plain knowledge"
        assert meta["confidence"] == 0.5
