"""Unit tests for core/memory/manager.py — MemoryManager."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from core.memory.manager import MemoryManager
from core.schemas import ModelConfig
from core.time_utils import today_local


@pytest.fixture
def anima_dir(tmp_path: Path) -> Path:
    d = tmp_path / "anima"
    d.mkdir()
    return d


@pytest.fixture
def mm(anima_dir: Path, data_dir: Path) -> MemoryManager:
    return MemoryManager(anima_dir)


# ── Initialization ────────────────────────────────────────


class TestInit:
    def test_creates_subdirs(self, anima_dir, data_dir):
        mm = MemoryManager(anima_dir)
        assert mm.episodes_dir.is_dir()
        assert mm.knowledge_dir.is_dir()
        assert mm.procedures_dir.is_dir()
        assert mm.skills_dir.is_dir()
        assert mm.state_dir.is_dir()

    def test_anima_dir(self, mm, anima_dir):
        assert mm.anima_dir == anima_dir


# ── Read helpers ──────────────────────────────────────────


class TestRead:
    def test_read_nonexistent(self, mm):
        assert mm._read(mm.anima_dir / "nonexistent.md") == ""

    def test_read_existing(self, mm, anima_dir):
        (anima_dir / "test.md").write_text("content", encoding="utf-8")
        assert mm._read(anima_dir / "test.md") == "content"


class TestReadIdentity:
    def test_no_file(self, mm):
        assert mm.read_identity() == ""

    def test_with_file(self, mm, anima_dir):
        (anima_dir / "IDENTITY.md").write_text("I am Alice", encoding="utf-8")
        assert mm.read_identity() == "I am Alice"

    def test_strips_frontmatter(self, mm, anima_dir):
        (anima_dir / "IDENTITY.md").write_text(
            "---\nname: Alice\nrole: engineer\n---\n\n# Identity\nI am Alice",
            encoding="utf-8",
        )
        assert mm.read_identity() == "# Identity\nI am Alice"


class TestReadSoul:
    def test_no_file(self, mm):
        assert mm.read_soul() == ""

    def test_with_file(self, mm, anima_dir):
        (anima_dir / "SOUL.md").write_text("Curious and kind", encoding="utf-8")
        assert mm.read_soul() == "Curious and kind"

    def test_strips_frontmatter(self, mm, anima_dir):
        (anima_dir / "SOUL.md").write_text(
            "---\nversion: 1\n---\n\n# Soul: Alice\nCurious",
            encoding="utf-8",
        )
        assert mm.read_soul() == "# Soul: Alice\nCurious"


class TestReadInjection:
    def test_no_file(self, mm):
        assert mm.read_injection() == ""

    def test_with_file(self, mm, anima_dir):
        (anima_dir / "AGENTS.md").write_text("Role info", encoding="utf-8")
        assert mm.read_injection() == "Role info"


class TestReadPermissions:
    def test_no_file(self, mm):
        """The default open policy adds no prompt section."""
        assert mm.read_permissions() == ""

    def test_with_file(self, mm, anima_dir):
        """Legacy open permissions migrate but add no default-only prompt section."""
        (anima_dir / "permissions.md").write_text("- web_search: OK", encoding="utf-8")
        result = mm.read_permissions()
        assert result == ""
        assert (anima_dir / "permissions.json").is_file()


class TestReadCurrentState:
    def test_default(self, mm):
        assert mm.read_current_state() == "status: idle"

    def test_with_file(self, mm, anima_dir):
        (anima_dir / "state").mkdir(parents=True, exist_ok=True)
        (anima_dir / "state" / "current_state.md").write_text("status: busy\ntask: testing", encoding="utf-8")
        assert "busy" in mm.read_current_state()


class TestCurrentStateMigration:
    """Tests for _migrate_current_task_to_state() — current_task.md → current_state.md."""

    def test_migrates_old_file_when_new_absent(self, anima_dir, data_dir):
        """When current_task.md exists and current_state.md does not → file is renamed."""
        (anima_dir / "state").mkdir(parents=True, exist_ok=True)
        old_path = anima_dir / "state" / "current_task.md"
        old_path.write_text("status: working\ntask: migrate me", encoding="utf-8")

        mm = MemoryManager(anima_dir)

        assert not old_path.exists()
        assert (anima_dir / "state" / "current_state.md").exists()
        assert "working" in mm.read_current_state()

    def test_removes_old_file_when_both_exist(self, anima_dir, data_dir):
        """When both exist → current_task.md is auto-deleted, current_state.md is used."""
        (anima_dir / "state").mkdir(parents=True, exist_ok=True)
        old_path = anima_dir / "state" / "current_task.md"
        old_path.write_text("status: old", encoding="utf-8")
        (anima_dir / "state" / "current_state.md").write_text("status: new\ntask: preferred", encoding="utf-8")

        mm = MemoryManager(anima_dir)

        assert not old_path.exists(), "stale current_task.md should have been deleted"
        assert "new" in mm.read_current_state()
        assert "preferred" in mm.read_current_state()

    def test_returns_idle_when_neither_exists(self, anima_dir, data_dir):
        """When neither exists → read_current_state() returns 'status: idle'."""
        (anima_dir / "state").mkdir(parents=True, exist_ok=True)

        mm = MemoryManager(anima_dir)

        assert mm.read_current_state() == "status: idle"


class TestReadHeartbeatConfig:
    def test_no_file(self, mm):
        assert mm.read_heartbeat_config() == ""

    def test_with_file(self, mm, anima_dir):
        (anima_dir / "HEARTBEAT.md").write_text("巡回間隔: 15分", encoding="utf-8")
        assert "15分" in mm.read_heartbeat_config()


class TestReadCronConfig:
    def test_no_file(self, mm):
        assert mm.read_cron_config() == ""


class TestReadBootstrap:
    def test_no_file(self, mm):
        assert mm.read_bootstrap() == ""

    def test_with_file(self, mm, anima_dir):
        (anima_dir / "BOOTSTRAP.md").write_text("Bootstrap", encoding="utf-8")
        assert mm.read_bootstrap() == "Bootstrap"


class TestReadCompanyVision:
    def test_reads_vision(self, mm, data_dir):
        assert "Company Vision" in mm.read_company_vision() or mm.read_company_vision() == ""

    def test_reads_company_vision_summary_and_memory_pointer(self, mm, anima_dir, data_dir):
        (anima_dir / "status.json").write_text(json.dumps({"company": "alpha"}), encoding="utf-8")
        company_dir = data_dir / "companies" / "alpha"
        company_dir.mkdir(parents=True)
        (company_dir / "vision.md").write_text("Full company vision", encoding="utf-8")
        (company_dir / "vision_summary.md").write_text("  One-line summary.  \n", encoding="utf-8")

        assert mm.read_company_vision_summary() == ("One-line summary.", str(company_dir / "vision.md"))

    def test_prefers_assigned_company_vision_then_falls_back(self, mm, anima_dir, data_dir):
        (anima_dir / "status.json").write_text(json.dumps({"company": "alpha"}), encoding="utf-8")
        company_dir = data_dir / "companies" / "alpha"
        company_dir.mkdir(parents=True)
        company_vision = company_dir / "vision.md"
        company_vision.write_text("Alpha-specific vision", encoding="utf-8")

        assert mm.read_company_vision() == "Alpha-specific vision"

        company_vision.write_text("", encoding="utf-8")
        assert mm.read_company_vision() == ""

        company_vision.unlink()
        assert "Company Vision" in mm.read_company_vision()

    def test_unassigned_anima_uses_legacy_company_vision(self, mm):
        assert "Company Vision" in mm.read_company_vision()


# ── List helpers ──────────────────────────────────────────


class TestResolutionCompanyBoundary:
    def test_keeps_same_company_and_unassigned_resolvers_only(self, mm, anima_dir, tmp_path):
        (anima_dir / "status.json").write_text(json.dumps({"company": "alpha"}), encoding="utf-8")
        root = anima_dir.parent
        for name, company in (("colleague", "alpha"), ("outsider", "beta")):
            anima = root / name
            anima.mkdir()
            (anima / "status.json").write_text(json.dumps({"company": company}), encoding="utf-8")
        (root / "unassigned").mkdir()

        resolutions = [
            {"resolver": "colleague", "issue": "same company"},
            {"resolver": "outsider", "issue": "other company"},
            {"resolver": "unassigned", "issue": "unassigned resolver"},
            {"resolver": "human operator", "issue": "human resolver"},
        ]

        assert mm.filter_resolutions_by_company(resolutions) == [resolutions[0]]

    def test_unassigned_anima_keeps_only_unassigned_resolvers(self, mm, anima_dir, tmp_path):
        root = anima_dir.parent
        (root / "unassigned").mkdir()
        company_anima = root / "assigned"
        company_anima.mkdir()
        (company_anima / "status.json").write_text(json.dumps({"company": "alpha"}), encoding="utf-8")

        resolutions = [
            {"resolver": "unassigned", "issue": "no company"},
            {"resolver": "assigned", "issue": "company assigned"},
            {"resolver": "unknown", "issue": "unknown resolver"},
        ]

        assert mm.filter_resolutions_by_company(resolutions) == [resolutions[0], resolutions[2]]


class TestListFiles:
    def test_list_knowledge_files(self, mm, anima_dir):
        (anima_dir / "knowledge" / "topic1.md").write_text("k1", encoding="utf-8")
        (anima_dir / "knowledge" / "topic2.md").write_text("k2", encoding="utf-8")
        result = mm.list_knowledge_files()
        assert "topic1" in result
        assert "topic2" in result

    def test_list_knowledge_empty(self, mm):
        assert mm.list_knowledge_files() == []

    def test_list_episode_files(self, mm, anima_dir):
        (anima_dir / "episodes" / "2026-01-01.md").write_text("ep", encoding="utf-8")
        result = mm.list_episode_files()
        assert "2026-01-01" in result

    def test_list_procedure_files(self, mm, anima_dir):
        (anima_dir / "procedures" / "deploy.md").write_text("proc", encoding="utf-8")
        result = mm.list_procedure_files()
        assert "deploy" in result


# ── Write helpers ─────────────────────────────────────────


class TestAppendEpisode:
    def test_creates_file(self, mm):
        mm.append_episode("Did something")
        path = mm.episodes_dir / f"{today_local().isoformat()}.md"
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert "Did something" in content
        assert today_local().isoformat() in content

    def test_appends_to_existing(self, mm):
        mm.append_episode("First")
        mm.append_episode("Second")
        path = mm.episodes_dir / f"{today_local().isoformat()}.md"
        content = path.read_text(encoding="utf-8")
        assert "First" in content
        assert "Second" in content


class TestUpdateState:
    def test_writes_state(self, mm):
        mm.update_state("status: busy\ntask: writing tests")
        content = mm.read_current_state()
        assert "busy" in content


class TestSearchMemoryText:
    """Tests for keyword fallback (vector unavailable)."""

    def test_search_all(self, mm, anima_dir):
        mm._rag._indexer = None
        mm._rag._indexer_initialized = True
        (anima_dir / "knowledge" / "python.md").write_text("Python is great\nJava is OK", encoding="utf-8")
        (anima_dir / "episodes" / "2026-01-01.md").write_text("Learned Python today", encoding="utf-8")
        results = mm.search_memory_text("python")
        assert len(results) >= 2

    def test_search_knowledge_scope(self, mm, anima_dir):
        mm._rag._indexer = None
        mm._rag._indexer_initialized = True
        (anima_dir / "knowledge" / "test.md").write_text("keyword here", encoding="utf-8")
        (anima_dir / "episodes" / "2026-01-01.md").write_text("keyword in episode", encoding="utf-8")
        results = mm.search_memory_text("keyword", scope="knowledge")
        assert all("knowledge" in r["source_file"] or r["source_file"].endswith("test.md") for r in results)

    def test_case_insensitive(self, mm, anima_dir):
        mm._rag._indexer = None
        mm._rag._indexer_initialized = True
        (anima_dir / "knowledge" / "test.md").write_text("UPPERCASE content", encoding="utf-8")
        results = mm.search_memory_text("uppercase")
        assert len(results) == 1

    def test_no_results(self, mm):
        mm._rag._indexer = None
        mm._rag._indexer_initialized = True
        results = mm.search_memory_text("nonexistent_query_xyz")
        assert results == []


class TestSearchMemoryTextCommonKnowledge:
    """Tests for keyword fallback with common_knowledge."""

    def _force_keyword_fallback(self, mm):
        mm._rag._indexer = None
        mm._rag._indexer_initialized = True

    def test_search_common_knowledge_scope(self, mm, data_dir):
        """search_memory_text with scope='common_knowledge' searches the shared dir."""
        self._force_keyword_fallback(mm)
        ck_dir = data_dir / "common_knowledge"
        ck_dir.mkdir(parents=True, exist_ok=True)
        (ck_dir / "shared_policy.md").write_text("Company-wide shared policy document", encoding="utf-8")
        results = mm.search_memory_text("shared policy", scope="common_knowledge")
        assert len(results) >= 1
        assert any("shared_policy.md" in r["source_file"] for r in results)

    def test_search_common_knowledge_scope_no_personal(self, mm, anima_dir, data_dir):
        """scope='common_knowledge' does NOT search personal knowledge."""
        self._force_keyword_fallback(mm)
        (anima_dir / "knowledge" / "personal.md").write_text("Personal knowledge only", encoding="utf-8")
        ck_dir = data_dir / "common_knowledge"
        ck_dir.mkdir(parents=True, exist_ok=True)
        results = mm.search_memory_text("Personal knowledge", scope="common_knowledge")
        assert all("personal.md" not in r["source_file"] for r in results)

    def test_search_all_includes_common_knowledge(self, mm, anima_dir, data_dir):
        """scope='all' includes common_knowledge dir in search."""
        self._force_keyword_fallback(mm)
        ck_dir = data_dir / "common_knowledge"
        ck_dir.mkdir(parents=True, exist_ok=True)
        (ck_dir / "global_info.md").write_text("Global information for everyone", encoding="utf-8")
        (anima_dir / "knowledge" / "local.md").write_text("Local knowledge for anima", encoding="utf-8")
        results = mm.search_memory_text("information", scope="all")
        sources = [r["source_file"] for r in results]
        assert any("global_info.md" in s for s in sources)

    def test_search_common_knowledge_empty_dir(self, mm, data_dir):
        """scope='common_knowledge' with empty dir returns no results."""
        self._force_keyword_fallback(mm)
        ck_dir = data_dir / "common_knowledge"
        ck_dir.mkdir(parents=True, exist_ok=True)
        results = mm.search_memory_text("anything", scope="common_knowledge")
        assert results == []

    def test_common_knowledge_dir_attribute(self, mm, data_dir):
        """MemoryManager has common_knowledge_dir attribute pointing to shared dir."""
        expected = data_dir / "common_knowledge"
        assert mm.common_knowledge_dir == expected


# ── read_model_config ─────────────────────────────────────


class TestReadModelConfig:
    def test_from_config_json(self, data_dir, make_anima):
        anima_dir = make_anima("test-anima", model="gpt-4o")
        mm = MemoryManager(anima_dir)
        mc = mm.read_model_config()
        assert isinstance(mc, ModelConfig)
        assert mc.model == "gpt-4o"

    def test_missing_config_returns_default(self, anima_dir, data_dir, tmp_path):
        from core.config.models import invalidate_cache

        config_path = tmp_path / "missing-config.json"
        with patch("core.config.models.get_config_path", return_value=config_path):
            invalidate_cache()
            mc = MemoryManager(anima_dir).read_model_config()
            invalidate_cache()

        assert mc == ModelConfig()
