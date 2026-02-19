"""Unit tests for core/memory/manager.py — MemoryManager."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from core.memory.manager import MemoryManager
from core.schemas import ModelConfig


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
        (anima_dir / "identity.md").write_text("I am Alice", encoding="utf-8")
        assert mm.read_identity() == "I am Alice"


class TestReadInjection:
    def test_no_file(self, mm):
        assert mm.read_injection() == ""

    def test_with_file(self, mm, anima_dir):
        (anima_dir / "injection.md").write_text("Role info", encoding="utf-8")
        assert mm.read_injection() == "Role info"


class TestReadPermissions:
    def test_no_file(self, mm):
        assert mm.read_permissions() == ""

    def test_with_file(self, mm, anima_dir):
        (anima_dir / "permissions.md").write_text("- web_search: OK", encoding="utf-8")
        assert mm.read_permissions() == "- web_search: OK"


class TestReadCurrentState:
    def test_default(self, mm):
        assert mm.read_current_state() == "status: idle"

    def test_with_file(self, mm, anima_dir):
        (anima_dir / "state").mkdir(parents=True, exist_ok=True)
        (anima_dir / "state" / "current_task.md").write_text(
            "status: busy\ntask: testing", encoding="utf-8"
        )
        assert "busy" in mm.read_current_state()


class TestReadPending:
    def test_no_file(self, mm):
        assert mm.read_pending() == ""

    def test_with_file(self, mm, anima_dir):
        (anima_dir / "state").mkdir(parents=True, exist_ok=True)
        (anima_dir / "state" / "pending.md").write_text("- task 1", encoding="utf-8")
        assert mm.read_pending() == "- task 1"


class TestReadHeartbeatConfig:
    def test_no_file(self, mm):
        assert mm.read_heartbeat_config() == ""

    def test_with_file(self, mm, anima_dir):
        (anima_dir / "heartbeat.md").write_text("巡回間隔: 15分", encoding="utf-8")
        assert "15分" in mm.read_heartbeat_config()


class TestReadCronConfig:
    def test_no_file(self, mm):
        assert mm.read_cron_config() == ""


class TestReadBootstrap:
    def test_no_file(self, mm):
        assert mm.read_bootstrap() == ""

    def test_with_file(self, mm, anima_dir):
        (anima_dir / "bootstrap.md").write_text("Bootstrap", encoding="utf-8")
        assert mm.read_bootstrap() == "Bootstrap"


class TestReadFile:
    def test_read_relative(self, mm, anima_dir):
        (anima_dir / "custom.md").write_text("custom content", encoding="utf-8")
        assert mm.read_file("custom.md") == "custom content"

    def test_read_nonexistent_relative(self, mm):
        assert mm.read_file("nonexistent.md") == ""


class TestReadCompanyVision:
    def test_reads_vision(self, mm, data_dir):
        assert "Company Vision" in mm.read_company_vision() or mm.read_company_vision() == ""


# ── List helpers ──────────────────────────────────────────


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

    def test_list_skill_files(self, mm, anima_dir):
        (anima_dir / "skills" / "coding.md").write_text("skill", encoding="utf-8")
        result = mm.list_skill_files()
        assert "coding" in result


class TestExtractSkillSummary:
    def test_extracts_summary(self, tmp_path):
        skill = tmp_path / "skill.md"
        skill.write_text("# Skill\n## 概要\nFirst line summary\nSecond line", encoding="utf-8")
        meta = MemoryManager._extract_skill_meta(skill)
        assert meta.description == "First line summary"

    def test_empty_overview(self, tmp_path):
        skill = tmp_path / "skill.md"
        skill.write_text("# Skill\n## 概要\n## 手順", encoding="utf-8")
        meta = MemoryManager._extract_skill_meta(skill)
        assert meta.description == ""

    def test_no_overview_section(self, tmp_path):
        skill = tmp_path / "skill.md"
        skill.write_text("# Skill\nJust content", encoding="utf-8")
        meta = MemoryManager._extract_skill_meta(skill)
        assert meta.description == ""


class TestListSkillSummaries:
    def test_summaries(self, mm, anima_dir):
        (anima_dir / "skills" / "coding.md").write_text(
            "# Coding\n## 概要\nWrite code efficiently\n## 手順\n1. Plan",
            encoding="utf-8",
        )
        result = mm.list_skill_summaries()
        assert len(result) == 1
        assert result[0][0] == "coding"
        assert result[0][1] == "Write code efficiently"


class TestListCommonSkillSummaries:
    def test_summaries(self, mm, data_dir):
        common_dir = data_dir / "common_skills"
        common_dir.mkdir(exist_ok=True)
        (common_dir / "cron-management.md").write_text(
            "# cron-management\n## 概要\nManage cron.md format\n## 手順\n1. Read",
            encoding="utf-8",
        )
        result = mm.list_common_skill_summaries()
        assert len(result) >= 1
        names = [r[0] for r in result]
        assert "cron-management" in names

    def test_empty_common_skills(self, mm):
        result = mm.list_common_skill_summaries()
        assert isinstance(result, list)


class TestListSharedUsers:
    def test_no_users(self, mm, data_dir):
        result = mm.list_shared_users()
        assert isinstance(result, list)

    def test_with_users(self, mm, data_dir):
        users_dir = data_dir / "shared" / "users"
        users_dir.mkdir(parents=True, exist_ok=True)
        (users_dir / "john").mkdir()
        (users_dir / "jane").mkdir()
        result = mm.list_shared_users()
        assert "john" in result
        assert "jane" in result


# ── Write helpers ─────────────────────────────────────────


class TestAppendEpisode:
    def test_creates_file(self, mm):
        mm.append_episode("Did something")
        path = mm.episodes_dir / f"{date.today().isoformat()}.md"
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert "Did something" in content
        assert date.today().isoformat() in content

    def test_appends_to_existing(self, mm):
        mm.append_episode("First")
        mm.append_episode("Second")
        path = mm.episodes_dir / f"{date.today().isoformat()}.md"
        content = path.read_text(encoding="utf-8")
        assert "First" in content
        assert "Second" in content


class TestUpdateState:
    def test_writes_state(self, mm):
        mm.update_state("status: busy\ntask: writing tests")
        content = mm.read_current_state()
        assert "busy" in content


class TestUpdatePending:
    def test_writes_pending(self, mm):
        mm.update_pending("- task 1\n- task 2")
        content = mm.read_pending()
        assert "task 1" in content


class TestWriteKnowledge:
    def test_writes_knowledge(self, mm):
        mm.write_knowledge("python", "Python is a programming language")
        result = mm.list_knowledge_files()
        assert "python" in result

    def test_sanitizes_filename(self, mm):
        mm.write_knowledge("topic/with:special chars", "content")
        # Should create file with sanitized name
        files = list(mm.knowledge_dir.glob("*.md"))
        assert len(files) == 1


# ── Read helpers for Mode B ───────────────────────────────


class TestReadRecentEpisodes:
    def test_reads_recent(self, mm):
        today = date.today()
        for i in range(3):
            d = today - timedelta(days=i)
            (mm.episodes_dir / f"{d.isoformat()}.md").write_text(
                f"Day {i}", encoding="utf-8"
            )
        result = mm.read_recent_episodes(days=3)
        assert "Day 0" in result
        assert "Day 1" in result
        assert "Day 2" in result

    def test_empty_when_no_episodes(self, mm):
        result = mm.read_recent_episodes(days=7)
        assert result == ""


class TestSearchMemoryText:
    def test_search_all(self, mm, anima_dir):
        (anima_dir / "knowledge" / "python.md").write_text(
            "Python is great\nJava is OK", encoding="utf-8"
        )
        (anima_dir / "episodes" / "2026-01-01.md").write_text(
            "Learned Python today", encoding="utf-8"
        )
        results = mm.search_memory_text("python")
        assert len(results) >= 2

    def test_search_knowledge_scope(self, mm, anima_dir):
        (anima_dir / "knowledge" / "test.md").write_text(
            "keyword here", encoding="utf-8"
        )
        (anima_dir / "episodes" / "2026-01-01.md").write_text(
            "keyword in episode", encoding="utf-8"
        )
        results = mm.search_memory_text("keyword", scope="knowledge")
        assert all("knowledge" in r[0] or r[0] == "test.md" for r in results)

    def test_case_insensitive(self, mm, anima_dir):
        (anima_dir / "knowledge" / "test.md").write_text(
            "UPPERCASE content", encoding="utf-8"
        )
        results = mm.search_memory_text("uppercase")
        assert len(results) == 1

    def test_no_results(self, mm):
        results = mm.search_memory_text("nonexistent_query_xyz")
        assert results == []


class TestSearchMemoryTextCommonKnowledge:
    def test_search_common_knowledge_scope(self, mm, data_dir):
        """search_memory_text with scope='common_knowledge' searches the shared dir."""
        ck_dir = data_dir / "common_knowledge"
        ck_dir.mkdir(parents=True, exist_ok=True)
        (ck_dir / "shared_policy.md").write_text(
            "Company-wide shared policy document", encoding="utf-8"
        )
        results = mm.search_memory_text("shared policy", scope="common_knowledge")
        assert len(results) >= 1
        assert any("shared_policy.md" in r[0] for r in results)

    def test_search_common_knowledge_scope_no_personal(self, mm, anima_dir, data_dir):
        """scope='common_knowledge' does NOT search personal knowledge."""
        (anima_dir / "knowledge" / "personal.md").write_text(
            "Personal knowledge only", encoding="utf-8"
        )
        ck_dir = data_dir / "common_knowledge"
        ck_dir.mkdir(parents=True, exist_ok=True)
        results = mm.search_memory_text("Personal knowledge", scope="common_knowledge")
        # Should NOT find the personal knowledge file
        assert all("personal.md" not in r[0] for r in results)

    def test_search_all_includes_common_knowledge(self, mm, anima_dir, data_dir):
        """scope='all' includes common_knowledge dir in search."""
        ck_dir = data_dir / "common_knowledge"
        ck_dir.mkdir(parents=True, exist_ok=True)
        (ck_dir / "global_info.md").write_text(
            "Global information for everyone", encoding="utf-8"
        )
        (anima_dir / "knowledge" / "local.md").write_text(
            "Local knowledge for anima", encoding="utf-8"
        )
        results = mm.search_memory_text("information", scope="all")
        filenames = [r[0] for r in results]
        assert any("global_info.md" in f for f in filenames)

    def test_search_common_knowledge_empty_dir(self, mm, data_dir):
        """scope='common_knowledge' with empty dir returns no results."""
        ck_dir = data_dir / "common_knowledge"
        ck_dir.mkdir(parents=True, exist_ok=True)
        results = mm.search_memory_text("anything", scope="common_knowledge")
        assert results == []

    def test_common_knowledge_dir_attribute(self, mm, data_dir):
        """MemoryManager has common_knowledge_dir attribute pointing to shared dir."""
        expected = data_dir / "common_knowledge"
        assert mm.common_knowledge_dir == expected


class TestSearchKnowledge:
    def test_search(self, mm, anima_dir):
        (anima_dir / "knowledge" / "topic.md").write_text(
            "Important info here", encoding="utf-8"
        )
        results = mm.search_knowledge("important")
        assert len(results) == 1
        assert "topic.md" in results[0][0]


class TestSearchProcedures:
    def test_search(self, mm, anima_dir):
        (anima_dir / "procedures" / "deploy.md").write_text(
            "Deploy to production", encoding="utf-8"
        )
        results = mm.search_procedures("deploy")
        assert len(results) == 1


# ── read_model_config ─────────────────────────────────────


class TestReadModelConfig:
    def test_from_config_json(self, data_dir, make_anima):
        anima_dir = make_anima("test-anima", model="gpt-4o")
        mm = MemoryManager(anima_dir)
        mc = mm.read_model_config()
        assert isinstance(mc, ModelConfig)
        assert mc.model == "gpt-4o"

    def test_legacy_fallback(self, tmp_path, monkeypatch):
        # Set up isolated environment where config.json does NOT exist
        fake_data = tmp_path / "fake_data"
        fake_data.mkdir()
        anima_dir = tmp_path / "anima_legacy"
        anima_dir.mkdir()
        (anima_dir / "config.md").write_text(
            "- model: custom-model\n- max_tokens: 2048\n",
            encoding="utf-8",
        )
        # Redirect ANIMAWORKS_DATA_DIR to a dir without config.json
        monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(fake_data))
        with patch("core.memory.manager.get_company_dir", return_value=fake_data / "co"), \
             patch("core.memory.manager.get_common_skills_dir", return_value=fake_data / "cs"), \
             patch("core.memory.manager.get_shared_dir", return_value=fake_data / "sh"):
            from core.config.models import invalidate_cache
            invalidate_cache()
            mm = MemoryManager(anima_dir)
            mc = mm.read_model_config()
            invalidate_cache()
            assert mc.model == "custom-model"
            assert mc.max_tokens == 2048


class TestResolveApiKey:
    def test_direct_key(self, data_dir, make_anima):
        anima_dir = make_anima("test-anima", api_key="sk-direct")
        mm = MemoryManager(anima_dir)
        mc = mm.read_model_config()
        assert mm.resolve_api_key(mc) == "sk-direct"

    def test_env_fallback(self, data_dir, make_anima):
        anima_dir = make_anima("test-anima")
        mm = MemoryManager(anima_dir)
        mc = mm.read_model_config()
        mc.api_key = None
        mc.api_key_env = "TEST_API_KEY_RESOLVE"
        with patch.dict("os.environ", {"TEST_API_KEY_RESOLVE": "sk-env"}):
            assert mm.resolve_api_key(mc) == "sk-env"


# ── _read_model_config_from_md ────────────────────────────


class TestReadModelConfigFromMd:
    def test_empty(self, anima_dir, data_dir):
        mm = MemoryManager(anima_dir)
        mc = mm._read_model_config_from_md()
        assert isinstance(mc, ModelConfig)
        assert mc.model == "claude-sonnet-4-20250514"

    def test_parses_fields(self, anima_dir, data_dir):
        (anima_dir / "config.md").write_text(
            "# Config\n- model: gpt-4o\n- max_tokens: 8192\n- max_turns: 10\n- api_base_url: http://localhost:8000\n",
            encoding="utf-8",
        )
        mm = MemoryManager(anima_dir)
        mc = mm._read_model_config_from_md()
        assert mc.model == "gpt-4o"
        assert mc.max_tokens == 8192
        assert mc.max_turns == 10
        assert mc.api_base_url == "http://localhost:8000"

    def test_ignores_biko_section(self, anima_dir, data_dir):
        (anima_dir / "config.md").write_text(
            "- model: real\n\n## 備考\n- model: fake\n",
            encoding="utf-8",
        )
        mm = MemoryManager(anima_dir)
        mc = mm._read_model_config_from_md()
        assert mc.model == "real"


class TestReadTodayEpisodes:
    def test_reads_today(self, mm):
        today = date.today().isoformat()
        (mm.episodes_dir / f"{today}.md").write_text(
            "Today's log", encoding="utf-8"
        )
        result = mm.read_today_episodes()
        assert "Today's log" in result

    def test_empty_when_no_today(self, mm):
        assert mm.read_today_episodes() == ""
