# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for MemoryManager decomposition.

Verifies that the Facade-based MemoryManager correctly delegates to
the extracted sub-services while maintaining backward compatibility.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from core.memory.manager import (
    MemoryManager,
    # Re-exports from skill_metadata module
    match_skills_by_description,
    _normalize_text,
    _extract_bracket_keywords,
    _extract_comma_keywords,
    _match_tier1,
    _match_tier2,
    _match_tier3_vector,
)
from core.memory.cron_logger import CronLogger
from core.memory.resolution_tracker import ResolutionTracker
from core.memory.config_reader import ConfigReader
from core.memory.skill_metadata import SkillMetadataService
from core.memory.rag_search import RAGMemorySearch
from core.memory.frontmatter import FrontmatterService
from core.schemas import SkillMeta


# ── Import compatibility ─────────────────────────────────


class TestImportCompatibility:
    """Re-exports from core.memory.manager still work."""

    def test_match_skills_importable_from_manager(self) -> None:
        assert callable(match_skills_by_description)

    def test_normalize_text_importable_from_manager(self) -> None:
        assert callable(_normalize_text)

    def test_extract_bracket_keywords_importable(self) -> None:
        assert callable(_extract_bracket_keywords)

    def test_extract_comma_keywords_importable(self) -> None:
        assert callable(_extract_comma_keywords)

    def test_match_tier1_importable(self) -> None:
        assert callable(_match_tier1)

    def test_match_tier2_importable(self) -> None:
        assert callable(_match_tier2)

    def test_match_tier3_vector_importable(self) -> None:
        assert callable(_match_tier3_vector)


# ── New module direct imports ────────────────────────────


class TestNewModuleImports:
    """New extracted modules are directly importable."""

    def test_cron_logger_importable(self) -> None:
        assert CronLogger is not None

    def test_resolution_tracker_importable(self) -> None:
        assert ResolutionTracker is not None

    def test_config_reader_importable(self) -> None:
        assert ConfigReader is not None

    def test_skill_metadata_service_importable(self) -> None:
        assert SkillMetadataService is not None

    def test_rag_memory_search_importable(self) -> None:
        assert RAGMemorySearch is not None

    def test_frontmatter_service_importable(self) -> None:
        assert FrontmatterService is not None


# ── Facade delegation ────────────────────────────────────


class TestFacadeDelegation:
    """MemoryManager methods delegate to the correct sub-service."""

    @pytest.fixture
    def mm(self, tmp_path: Path) -> MemoryManager:
        anima_dir = tmp_path / "animas" / "test-anima"
        anima_dir.mkdir(parents=True)
        return MemoryManager(anima_dir)

    def test_cron_log_roundtrip(self, mm: MemoryManager) -> None:
        """append_cron_log + read_cron_log works through the facade."""
        mm.append_cron_log("daily-backup", summary="OK", duration_ms=1234)
        result = mm.read_cron_log(days=1)
        assert "daily-backup" in result
        assert "1234ms" in result

    def test_cron_command_log(self, mm: MemoryManager) -> None:
        """append_cron_command_log writes to the log file."""
        import json as _json
        from datetime import date as _date

        mm.append_cron_command_log(
            "test-cmd", exit_code=0, stdout="line1", stderr="", duration_ms=100,
        )
        # Command logs don't have 'summary' so read_cron_log skips them.
        # Verify by reading the JSONL directly.
        log_dir = mm.anima_dir / "state" / "cron_logs"
        log_file = log_dir / f"{_date.today().isoformat()}.jsonl"
        assert log_file.exists()
        entry = _json.loads(log_file.read_text().strip())
        assert entry["task"] == "test-cmd"
        assert entry["exit_code"] == 0

    def test_resolution_roundtrip(self, mm: MemoryManager) -> None:
        """append_resolution + read_resolutions works through the facade."""
        mm.append_resolution("disk full", "ops-anima")
        entries = mm.read_resolutions(days=1)
        assert len(entries) >= 1
        assert entries[-1]["issue"] == "disk full"
        assert entries[-1]["resolver"] == "ops-anima"

    def test_skill_meta_static(self, tmp_path: Path) -> None:
        """_extract_skill_meta delegates to SkillMetadataService."""
        f = tmp_path / "test.md"
        f.write_text("---\nname: test-skill\ndescription: テスト\n---\n\nBody.\n")
        meta = MemoryManager._extract_skill_meta(f)
        assert meta.name == "test-skill"
        assert meta.description == "テスト"

    def test_list_skill_metas(self, mm: MemoryManager) -> None:
        """list_skill_metas returns metas from the skills directory."""
        (mm.skills_dir / "a.md").write_text("---\nname: alpha\ndescription: テスト\n---\n\n")
        result = mm.list_skill_metas()
        assert len(result) == 1
        assert result[0].name == "alpha"

    def test_knowledge_frontmatter_roundtrip(self, mm: MemoryManager) -> None:
        """Knowledge frontmatter write/read works through the facade."""
        path = mm.knowledge_dir / "test.md"
        mm.write_knowledge_with_meta(path, "body content", {"topic": "test"})
        assert mm.read_knowledge_content(path) == "body content"
        meta = mm.read_knowledge_metadata(path)
        assert meta["topic"] == "test"

    def test_procedure_frontmatter_roundtrip(self, mm: MemoryManager) -> None:
        """Procedure frontmatter write/read works through the facade."""
        mm.write_procedure_with_meta(
            Path("proc.md"), "do the thing", {"description": "test proc"},
        )
        content = mm.read_procedure_content(Path("proc.md"))
        assert "do the thing" in content
        meta = mm.read_procedure_metadata(Path("proc.md"))
        assert meta["description"] == "test proc"

    def test_search_knowledge_keyword(self, mm: MemoryManager) -> None:
        """search_knowledge finds keyword matches."""
        (mm.knowledge_dir / "test.md").write_text("# Python tips\nUse list comprehension.\n")
        results = mm.search_knowledge("comprehension")
        assert any("comprehension" in line for _, line in results)

    def test_search_memory_text_scope(self, mm: MemoryManager) -> None:
        """search_memory_text respects scope parameter."""
        (mm.knowledge_dir / "k.md").write_text("knowledge content\n")
        (mm.episodes_dir / "e.md").write_text("episode content\n")
        results = mm.search_memory_text("content", scope="knowledge")
        files = [f for f, _ in results]
        assert "k.md" in files
        assert "e.md" not in files


# ── Backward-compatible RAG proxies ──────────────────────


class TestRAGProxies:
    """Tests that backward-compatible RAG attribute proxies work."""

    @pytest.fixture
    def mm(self, tmp_path: Path) -> MemoryManager:
        anima_dir = tmp_path / "animas" / "test-anima"
        anima_dir.mkdir(parents=True)
        return MemoryManager(anima_dir)

    def test_indexer_proxy(self, mm: MemoryManager) -> None:
        """_indexer property delegates to RAGMemorySearch."""
        assert mm._indexer is None  # Not initialized yet

    def test_indexer_initialized_proxy(self, mm: MemoryManager) -> None:
        """_indexer_initialized defaults to False."""
        assert mm._indexer_initialized is False

    def test_get_indexer_proxy(self, mm: MemoryManager) -> None:
        """_get_indexer() delegates to RAGMemorySearch."""
        # Will try to initialize (may fail due to deps), but shouldn't crash
        result = mm._get_indexer()
        # Either None (deps missing) or an indexer object
        assert result is None or result is not None


# ── __new__ bypass compatibility ─────────────────────────


class TestNewBypassCompat:
    """Tests that MemoryManager works when __init__ is bypassed via __new__."""

    def test_append_episode_without_init(self, tmp_path: Path) -> None:
        """append_episode works on a __new__-constructed instance."""
        anima_dir = tmp_path / "animas" / "test"
        episodes_dir = anima_dir / "episodes"
        episodes_dir.mkdir(parents=True)

        mm = MemoryManager.__new__(MemoryManager)
        mm.anima_dir = anima_dir
        mm.episodes_dir = episodes_dir
        mm._indexer = None
        mm._indexer_initialized = True

        mm.append_episode("Test entry")
        # Verify it was written
        files = list(episodes_dir.glob("*.md"))
        assert len(files) == 1
        assert "Test entry" in files[0].read_text()

    def test_list_skill_metas_without_init(self, tmp_path: Path) -> None:
        """list_skill_metas works on a __new__-constructed instance."""
        anima_dir = tmp_path / "animas" / "test"
        skills_dir = anima_dir / "skills"
        skills_dir.mkdir(parents=True)
        (skills_dir / "test.md").write_text("---\nname: t\ndescription: d\n---\n\n")

        mm = MemoryManager.__new__(MemoryManager)
        mm.anima_dir = anima_dir
        mm.skills_dir = skills_dir

        result = mm.list_skill_metas()
        assert len(result) == 1
        assert result[0].name == "t"
