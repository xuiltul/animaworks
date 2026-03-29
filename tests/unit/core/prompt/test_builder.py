"""Unit tests for core/prompt/builder.py — system prompt construction."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from core.prompt.builder import (
    BuildResult,
    _build_messaging_section,
    _build_org_context,
    _discover_other_animas,
    _format_anima_entry,
    _normalize_headings,
    build_system_prompt,
    inject_shortterm,
)
from core.schemas import SkillMeta

_MOCK_SECTIONS = (
    "[group1_header]: 1. 動作環境と行動ルール\n"
    "[current_time_label]: **現在時刻**:\n"
    "[group2_header]: 2. あなた自身\n"
    "[group3_header]: 3. 現在の状況\n"
    "[current_state_header]: ## 現在の状態\n"
    "[pending_tasks_header]: ## 未完了タスク\n"
    "[procedures_header]: ## Procedures（手順書）\n"
    "[distilled_knowledge_header]: ## Distilled Knowledge\n"
    "[group4_header]: 4. 記憶と能力\n"
    "[group5_header]: 5. 組織とコミュニケーション\n"
    "[group6_header]: 6. メタ設定\n"
    "[you_marker]:   ← あなた\n"
    "[common_label]: (共通スキル)\n"
    "[recent_tool_results_header]: ## Recent Tool Results\n"
)

_MOCK_FALLBACKS = (
    "[unset]: (未設定)\n"
    "[none]: (なし)\n"
    "[none_top_level]: (なし — あなたがトップです)\n"
    "[no_other_animas]: (まだ他の社員はいません)\n"
    "[truncated]: （前半省略）\n"
    "[summary]: （要約）\n"
)


def _mock_load_prompt_with_builder(default: str = "section"):
    """Return a side_effect function that renders builder/* templates minimally."""

    def _mock(name: str, **kwargs) -> str:
        if name == "builder/sections":
            return _MOCK_SECTIONS
        if name == "builder/fallbacks":
            return _MOCK_FALLBACKS
        if name == "builder/task_in_progress":
            return f"## ⚠️ 進行中タスク\n\n{kwargs.get('state', '')}"
        if name == "builder/task_queue":
            return f"## 未完了タスク\n\n{kwargs.get('task_summary', '')}"
        if name == "builder/external_tools_guide":
            cats = kwargs.get("categories", "")
            return f"外部ツールを使うには `discover_tools` を呼んでください。\nカテゴリ: {cats}"
        if name == "skills_guide":
            return "## スキルと手順書\n\nスキルと手順書はあなたが持つ能力・作業手順です。\n使用する際はskillツールで読み込んでから実行してください。"
        return default

    return _mock


# ── _discover_other_animas ───────────────────────────────


class TestDiscoverOtherAnimas:
    def test_finds_siblings(self, tmp_path):
        animas_root = tmp_path / "animas"
        animas_root.mkdir()
        alice = animas_root / "alice"
        alice.mkdir()
        (alice / "identity.md").write_text("I am Alice", encoding="utf-8")
        bob = animas_root / "bob"
        bob.mkdir()
        (bob / "identity.md").write_text("I am Bob", encoding="utf-8")

        result = _discover_other_animas(alice)
        assert result == ["bob"]

    def test_excludes_self(self, tmp_path):
        animas_root = tmp_path / "animas"
        animas_root.mkdir()
        alice = animas_root / "alice"
        alice.mkdir()
        (alice / "identity.md").write_text("I am Alice", encoding="utf-8")

        result = _discover_other_animas(alice)
        assert "alice" not in result

    def test_excludes_dirs_without_identity(self, tmp_path):
        animas_root = tmp_path / "animas"
        animas_root.mkdir()
        alice = animas_root / "alice"
        alice.mkdir()
        (alice / "identity.md").write_text("I am Alice", encoding="utf-8")
        noident = animas_root / "noident"
        noident.mkdir()
        # no identity.md

        result = _discover_other_animas(alice)
        assert "noident" not in result

    def test_no_siblings(self, tmp_path):
        animas_root = tmp_path / "animas"
        animas_root.mkdir()
        alice = animas_root / "alice"
        alice.mkdir()
        (alice / "identity.md").write_text("I am Alice", encoding="utf-8")

        result = _discover_other_animas(alice)
        assert result == []


# ── _build_messaging_section ──────────────────────────────


class TestBuildMessagingSection:
    def test_with_animas(self, tmp_path):
        anima_dir = tmp_path / "alice"
        anima_dir.mkdir()
        with (
            patch("core.tooling.prompt_db.get_prompt_store", return_value=None),
            patch("core.prompt.messaging.load_prompt", return_value="messaging section"),
        ):
            result = _build_messaging_section(anima_dir, ["bob", "charlie"])
            assert result == "messaging section"

    def test_no_animas(self, tmp_path):
        anima_dir = tmp_path / "alice"
        anima_dir.mkdir()

        def _mock_lp(name: str, **kwargs) -> str:
            if name == "builder/fallbacks":
                return _MOCK_FALLBACKS
            return "messaging section"

        with (
            patch("core.tooling.prompt_db.get_prompt_store", return_value=None),
            patch("core.prompt.sections.load_prompt", side_effect=_mock_lp),
            patch("core.prompt.messaging.load_prompt", side_effect=_mock_lp) as mock_lp,
        ):
            _build_messaging_section(anima_dir, [])
            call_kwargs = mock_lp.call_args[1]
            assert "(まだ他の社員はいません)" in call_kwargs["animas_line"]

    def test_s_mode_uses_messaging_s_template(self, tmp_path):
        """S mode should load the messaging_s template."""
        anima_dir = tmp_path / "alice"
        anima_dir.mkdir()
        with (
            patch("core.tooling.prompt_db.get_prompt_store", return_value=None),
            patch("core.prompt.messaging.load_prompt", return_value="s messaging") as mock_lp,
        ):
            result = _build_messaging_section(anima_dir, ["bob"], execution_mode="s")
            assert result == "s messaging"
            template_names = [c[0][0] for c in mock_lp.call_args_list]
            assert "messaging_s" in template_names

    def test_a_mode_uses_messaging_template(self, tmp_path):
        """A mode should load the standard messaging template."""
        anima_dir = tmp_path / "alice"
        anima_dir.mkdir()
        with (
            patch("core.tooling.prompt_db.get_prompt_store", return_value=None),
            patch("core.prompt.messaging.load_prompt", return_value="a messaging") as mock_lp,
        ):
            result = _build_messaging_section(anima_dir, ["bob"], execution_mode="a")
            assert result == "a messaging"
            template_names = [c[0][0] for c in mock_lp.call_args_list]
            assert "messaging" in template_names

    def test_default_mode_uses_s_template(self, tmp_path):
        """Default execution_mode should be s, using messaging_s template."""
        anima_dir = tmp_path / "alice"
        anima_dir.mkdir()
        with (
            patch("core.tooling.prompt_db.get_prompt_store", return_value=None),
            patch("core.prompt.messaging.load_prompt", return_value="section") as mock_lp,
        ):
            _build_messaging_section(anima_dir, ["bob"])
            template_names = [c[0][0] for c in mock_lp.call_args_list]
            assert "messaging_s" in template_names


# ── build_system_prompt ───────────────────────────────────


class TestBuildSystemPrompt:
    def test_builds_prompt(self, tmp_path, data_dir):
        anima_dir = tmp_path / "animas" / "alice"
        anima_dir.mkdir(parents=True)
        (anima_dir / "identity.md").write_text("I am Alice", encoding="utf-8")

        memory = MagicMock()
        memory.anima_dir = anima_dir
        memory.read_company_vision.return_value = "Company Vision"
        memory.read_identity.return_value = "I am Alice"
        memory.read_injection.return_value = ""
        memory.read_permissions.return_value = ""
        memory.read_specialty_prompt.return_value = ""
        memory.read_current_state.return_value = "status: idle"
        memory.read_pending.return_value = ""
        memory.read_bootstrap.return_value = ""
        memory.list_knowledge_files.return_value = []
        memory.list_episode_files.return_value = []
        memory.list_procedure_files.return_value = []
        memory.list_skill_summaries.return_value = []
        memory.list_common_skill_summaries.return_value = []
        memory.list_skill_metas.return_value = []
        memory.list_common_skill_metas.return_value = []
        memory.collect_distilled_knowledge_separated.return_value = ([], [])
        memory.common_skills_dir = data_dir / "common_skills"
        memory.list_shared_users.return_value = []
        memory.collect_distilled_knowledge_separated.return_value = ([], [])

        with patch("core.prompt.builder.load_prompt", return_value="prompt section"):
            result = build_system_prompt(memory)
            assert isinstance(result, BuildResult)
            assert isinstance(result.system_prompt, str)
            assert len(result) > 0

    def test_includes_identity(self, tmp_path, data_dir):
        anima_dir = tmp_path / "animas" / "alice"
        anima_dir.mkdir(parents=True)
        (anima_dir / "identity.md").write_text("I am Alice", encoding="utf-8")

        memory = MagicMock()
        memory.anima_dir = anima_dir
        memory.read_company_vision.return_value = ""
        memory.read_identity.return_value = "I am Alice"
        memory.read_injection.return_value = ""
        memory.read_permissions.return_value = ""
        memory.read_specialty_prompt.return_value = ""
        memory.read_current_state.return_value = ""
        memory.read_pending.return_value = ""
        memory.read_bootstrap.return_value = ""
        memory.list_knowledge_files.return_value = []
        memory.list_episode_files.return_value = []
        memory.list_procedure_files.return_value = []
        memory.list_skill_summaries.return_value = []
        memory.list_common_skill_summaries.return_value = []
        memory.list_skill_metas.return_value = []
        memory.list_common_skill_metas.return_value = []
        memory.collect_distilled_knowledge_separated.return_value = ([], [])
        memory.common_skills_dir = data_dir / "common_skills"
        memory.list_shared_users.return_value = []
        memory.collect_distilled_knowledge_separated.return_value = ([], [])

        with patch("core.prompt.builder.load_prompt", return_value="prompt"):
            result = build_system_prompt(memory)
            assert "I am Alice" in result

    def test_skills_not_in_system_prompt(self, tmp_path, data_dir):
        """Skills are not listed in memory_guide — catalog is in Group 4 ``<available_skills>``."""
        anima_dir = tmp_path / "animas" / "alice"
        anima_dir.mkdir(parents=True)
        (anima_dir / "identity.md").write_text("I am Alice", encoding="utf-8")

        memory = MagicMock()
        memory.anima_dir = anima_dir
        memory.read_company_vision.return_value = ""
        memory.read_identity.return_value = ""
        memory.read_injection.return_value = ""
        memory.read_permissions.return_value = ""
        memory.read_specialty_prompt.return_value = ""
        memory.read_current_state.return_value = ""
        memory.read_pending.return_value = ""
        memory.read_bootstrap.return_value = ""
        memory.list_knowledge_files.return_value = ["topic-a", "topic-b"]
        memory.list_episode_files.return_value = []
        memory.list_procedure_files.return_value = ["proc-x"]
        memory.list_skill_summaries.return_value = [("coding", "Write code")]
        memory.list_common_skill_summaries.return_value = [("deploy", "Deploy apps")]
        memory.list_skill_metas.return_value = [
            SkillMeta(name="coding", description="Write code", path=Path("/tmp/test/skills/coding.md"), is_common=False)
        ]
        memory.list_common_skill_metas.return_value = [
            SkillMeta(
                name="deploy", description="Deploy apps", path=Path("/tmp/test/common_skills/deploy.md"), is_common=True
            )
        ]
        memory.list_procedure_metas.return_value = []
        memory.collect_distilled_knowledge_separated.return_value = ([], [])
        memory.common_skills_dir = data_dir / "common_skills"
        memory.list_shared_users.return_value = []
        memory.collect_distilled_knowledge_separated.return_value = ([], [])

        with patch("core.prompt.builder.load_prompt", side_effect=_mock_load_prompt_with_builder()):
            result = build_system_prompt(memory)
            prompt = result.system_prompt
            assert "スキルと手順書" not in prompt
            assert "- coding: Write code" not in prompt
            assert "- deploy" not in prompt

    def test_memory_guide_uses_counts(self, tmp_path, data_dir):
        """memory_guide receives knowledge/procedure counts, not file name lists."""
        anima_dir = tmp_path / "animas" / "alice"
        anima_dir.mkdir(parents=True)
        (anima_dir / "identity.md").write_text("I am Alice", encoding="utf-8")

        memory = MagicMock()
        memory.anima_dir = anima_dir
        memory.read_company_vision.return_value = ""
        memory.read_identity.return_value = ""
        memory.read_injection.return_value = ""
        memory.read_permissions.return_value = ""
        memory.read_specialty_prompt.return_value = ""
        memory.read_current_state.return_value = ""
        memory.read_pending.return_value = ""
        memory.read_bootstrap.return_value = ""
        memory.list_knowledge_files.return_value = ["a", "b", "c"]
        memory.list_episode_files.return_value = []
        memory.list_procedure_files.return_value = ["p1"]
        memory.list_skill_summaries.return_value = []
        memory.list_common_skill_summaries.return_value = []
        memory.list_skill_metas.return_value = []
        memory.list_common_skill_metas.return_value = []
        memory.list_procedure_metas.return_value = []
        memory.collect_distilled_knowledge_separated.return_value = ([], [])
        memory.common_skills_dir = data_dir / "common_skills"
        memory.list_shared_users.return_value = ["taka"]
        memory.collect_distilled_knowledge_separated.return_value = ([], [])

        captured_calls: list[dict] = []

        def _capture_load_prompt(name: str, **kwargs) -> str:
            captured_calls.append({"name": name, "kwargs": kwargs})
            base = _mock_load_prompt_with_builder()
            return base(name, **kwargs)

        with patch("core.prompt.builder.load_prompt", side_effect=_capture_load_prompt):
            build_system_prompt(memory)

        mg_calls = [c for c in captured_calls if c["name"] == "memory_guide"]
        assert len(mg_calls) == 1
        kw = mg_calls[0]["kwargs"]
        assert kw["knowledge_count"] == 3
        assert kw["procedure_count"] == 1
        assert "skill_names" not in kw
        assert "episode_list" not in kw
        assert "knowledge_list" not in kw

    def test_includes_bootstrap(self, tmp_path, data_dir):
        anima_dir = tmp_path / "animas" / "alice"
        anima_dir.mkdir(parents=True)
        (anima_dir / "identity.md").write_text("I am Alice", encoding="utf-8")

        memory = MagicMock()
        memory.anima_dir = anima_dir
        memory.read_company_vision.return_value = ""
        memory.read_identity.return_value = ""
        memory.read_injection.return_value = ""
        memory.read_permissions.return_value = ""
        memory.read_specialty_prompt.return_value = ""
        memory.read_current_state.return_value = ""
        memory.read_pending.return_value = ""
        memory.read_bootstrap.return_value = "Bootstrap instructions"
        memory.list_knowledge_files.return_value = []
        memory.list_episode_files.return_value = []
        memory.list_procedure_files.return_value = []
        memory.list_skill_summaries.return_value = []
        memory.list_common_skill_summaries.return_value = []
        memory.list_skill_metas.return_value = []
        memory.list_common_skill_metas.return_value = []
        memory.collect_distilled_knowledge_separated.return_value = ([], [])
        memory.common_skills_dir = data_dir / "common_skills"
        memory.list_shared_users.return_value = []
        memory.collect_distilled_knowledge_separated.return_value = ([], [])

        with patch("core.prompt.builder.load_prompt", return_value="section"):
            result = build_system_prompt(memory)
            assert "Bootstrap instructions" in result

    def test_includes_state_and_pending(self, tmp_path, data_dir):
        anima_dir = tmp_path / "animas" / "alice"
        anima_dir.mkdir(parents=True)
        (anima_dir / "identity.md").write_text("I am Alice", encoding="utf-8")

        memory = MagicMock()
        memory.anima_dir = anima_dir
        memory.read_company_vision.return_value = ""
        memory.read_identity.return_value = ""
        memory.read_injection.return_value = ""
        memory.read_permissions.return_value = ""
        memory.read_specialty_prompt.return_value = ""
        memory.read_current_state.return_value = "status: working"
        memory.read_pending.return_value = "- task 1"
        memory.read_bootstrap.return_value = ""
        memory.list_knowledge_files.return_value = []
        memory.list_episode_files.return_value = []
        memory.list_procedure_files.return_value = []
        memory.list_skill_summaries.return_value = []
        memory.list_common_skill_summaries.return_value = []
        memory.list_skill_metas.return_value = []
        memory.list_common_skill_metas.return_value = []
        memory.collect_distilled_knowledge_separated.return_value = ([], [])
        memory.common_skills_dir = data_dir / "common_skills"
        memory.list_shared_users.return_value = []
        memory.collect_distilled_knowledge_separated.return_value = ([], [])

        with patch("core.prompt.builder.load_prompt", side_effect=_mock_load_prompt_with_builder()):
            result = build_system_prompt(memory)
            # Non-idle state goes to "進行中タスク" branch (Issue #114: pending removed)
            assert "進行中タスク" in result
            assert "status: working" in result

    def test_a_mode_injects_external_tools_hint_with_bash(self, tmp_path, data_dir):
        """Mode A hints should mention Bash and animaworks-tool."""
        anima_dir = tmp_path / "animas" / "alice"
        anima_dir.mkdir(parents=True)
        (anima_dir / "identity.md").write_text("I am Alice", encoding="utf-8")

        memory = MagicMock()
        memory.anima_dir = anima_dir
        memory.read_company_vision.return_value = ""
        memory.read_identity.return_value = ""
        memory.read_injection.return_value = ""
        memory.read_permissions.return_value = "## 外部ツール\n- chatwork: OK"
        memory.read_specialty_prompt.return_value = ""
        memory.read_current_state.return_value = ""
        memory.read_pending.return_value = ""
        memory.read_bootstrap.return_value = ""
        memory.list_knowledge_files.return_value = []
        memory.list_episode_files.return_value = []
        memory.list_procedure_files.return_value = []
        memory.list_skill_summaries.return_value = []
        memory.list_common_skill_summaries.return_value = []
        memory.list_skill_metas.return_value = []
        memory.list_common_skill_metas.return_value = []
        memory.collect_distilled_knowledge_separated.return_value = ([], [])
        memory.common_skills_dir = data_dir / "common_skills"
        memory.list_shared_users.return_value = []

        with patch("core.prompt.builder.load_prompt", side_effect=_mock_load_prompt_with_builder()):
            result = build_system_prompt(
                memory,
                tool_registry=["chatwork", "slack"],
                execution_mode="a",
            )
            assert "Bash" in result
            assert "animaworks-tool" in result

    def test_s_mode_injects_external_tools_hint_with_bash(self, tmp_path, data_dir):
        """S mode injects External Tools hint mentioning Bash."""
        anima_dir = tmp_path / "animas" / "alice"
        anima_dir.mkdir(parents=True)
        (anima_dir / "identity.md").write_text("I am Alice", encoding="utf-8")

        memory = MagicMock()
        memory.anima_dir = anima_dir
        memory.read_company_vision.return_value = ""
        memory.read_identity.return_value = ""
        memory.read_injection.return_value = ""
        memory.read_permissions.return_value = "## 外部ツール\n- chatwork: OK"
        memory.read_specialty_prompt.return_value = ""
        memory.read_current_state.return_value = ""
        memory.read_pending.return_value = ""
        memory.read_bootstrap.return_value = ""
        memory.list_knowledge_files.return_value = []
        memory.list_episode_files.return_value = []
        memory.list_procedure_files.return_value = []
        memory.list_skill_summaries.return_value = []
        memory.list_common_skill_summaries.return_value = []
        memory.list_skill_metas.return_value = []
        memory.list_common_skill_metas.return_value = []
        memory.collect_distilled_knowledge_separated.return_value = ([], [])
        memory.common_skills_dir = data_dir / "common_skills"
        memory.list_shared_users.return_value = []

        with patch("core.prompt.builder.load_prompt", return_value="section"):
            result = build_system_prompt(
                memory,
                tool_registry=["chatwork"],
                execution_mode="s",
            )
            assert "External Tools" in result
            assert "Bash" in result
            assert "animaworks-tool" in result

    def test_b_mode_injects_external_tools_hint_with_bash_cli(self, tmp_path, data_dir):
        """B mode injects External Tools hint mentioning Bash + animaworks-tool."""
        anima_dir = tmp_path / "animas" / "alice"
        anima_dir.mkdir(parents=True)
        (anima_dir / "identity.md").write_text("I am Alice", encoding="utf-8")

        memory = MagicMock()
        memory.anima_dir = anima_dir
        memory.read_company_vision.return_value = ""
        memory.read_identity.return_value = ""
        memory.read_injection.return_value = ""
        memory.read_permissions.return_value = "## 外部ツール\n- chatwork: OK"
        memory.read_specialty_prompt.return_value = ""
        memory.read_current_state.return_value = ""
        memory.read_pending.return_value = ""
        memory.read_bootstrap.return_value = ""
        memory.list_knowledge_files.return_value = []
        memory.list_episode_files.return_value = []
        memory.list_procedure_files.return_value = []
        memory.list_skill_summaries.return_value = []
        memory.list_common_skill_summaries.return_value = []
        memory.list_skill_metas.return_value = []
        memory.list_common_skill_metas.return_value = []
        memory.collect_distilled_knowledge_separated.return_value = ([], [])
        memory.common_skills_dir = data_dir / "common_skills"
        memory.list_shared_users.return_value = []

        with patch("core.prompt.builder.load_prompt", return_value="section"):
            result = build_system_prompt(
                memory,
                tool_registry=["chatwork"],
                execution_mode="b",
            )
            assert "animaworks-tool" in result


# ── _format_anima_entry ──────────────────────────────────


class TestFormatAnimaEntry:
    def test_with_speciality(self):
        assert _format_anima_entry("alice", "frontend") == "alice (frontend)"

    def test_without_speciality(self):
        assert _format_anima_entry("alice", None) == "alice"

    def test_empty_speciality(self):
        assert _format_anima_entry("alice", "") == "alice"

    def test_with_speciality_and_model(self):
        assert _format_anima_entry("alice", "frontend", "claude-opus-4-6") == "alice (frontend, Opus)"

    def test_with_model_only(self):
        assert _format_anima_entry("alice", None, "bedrock/jp.anthropic.claude-sonnet-4-6") == "alice (Sonnet)"


# ── _build_org_context ───────────────────────────────────


class TestBuildOrgContext:
    """Test organisation context derivation from supervisor chain."""

    def test_top_level_anima(self, data_dir, make_anima):
        """Top-level anima (no supervisor) sees full org tree."""
        make_anima("sakura")
        make_anima("rin", supervisor="sakura", speciality="development")
        make_anima("kotoha", supervisor="sakura", speciality="communication")

        result = _build_org_context("sakura", ["rin", "kotoha"])
        assert "あなたはトップレベルです" in result
        assert "rin (development, Sonnet)" in result
        assert "kotoha (communication, Sonnet)" in result

    def test_middle_manager(self, data_dir, make_anima):
        """Middle manager sees supervisor, subordinates, and peers."""
        make_anima("sakura")
        make_anima("rin", supervisor="sakura", speciality="development")
        make_anima("kotoha", supervisor="sakura", speciality="communication")
        make_anima("alice", supervisor="rin", speciality="frontend")

        result = _build_org_context("rin", ["sakura", "kotoha", "alice"])
        # Supervisor
        assert "sakura (Sonnet)" in result
        # Subordinate
        assert "alice (frontend, Sonnet)" in result
        # Peer
        assert "kotoha (communication, Sonnet)" in result

    def test_leaf_worker(self, data_dir, make_anima):
        """Leaf worker sees supervisor and peers but no subordinates."""
        make_anima("sakura")
        make_anima("rin", supervisor="sakura", speciality="development")
        make_anima("alice", supervisor="rin", speciality="frontend")
        make_anima("bob", supervisor="rin", speciality="backend")

        result = _build_org_context("alice", ["sakura", "rin", "bob"])
        # Supervisor
        assert "rin (development, Sonnet)" in result
        # No subordinates
        assert "部下" in result
        assert "(なし)" in result
        # Peer
        assert "bob (backend, Sonnet)" in result

    def test_solo_anima(self, data_dir, make_anima):
        """Solo anima with no relationships."""
        make_anima("sakura")

        result = _build_org_context("sakura", [])
        assert "あなたがトップです" in result
        # No communication rules when alone
        assert "コミュニケーションルール" not in result

    def test_communication_rules_injected_when_others_exist(self, data_dir, make_anima):
        """Communication rules are included when other animas exist."""
        make_anima("sakura")
        make_anima("rin", supervisor="sakura", speciality="development")

        result = _build_org_context("sakura", ["rin"])
        assert "コミュニケーションルール" in result

    def test_speciality_not_set(self, data_dir, make_anima):
        """Handles animas without speciality gracefully."""
        make_anima("sakura")
        make_anima("rin", supervisor="sakura")

        result = _build_org_context("sakura", ["rin"])
        # rin should appear without parenthetical speciality
        assert "rin" in result
        # anima_speciality should show (未設定) for sakura
        assert "(未設定)" in result

    def test_config_load_failure_returns_empty(self, data_dir):
        """Returns empty string when config cannot be loaded."""
        with patch("core.config.load_config", side_effect=RuntimeError):
            result = _build_org_context("sakura", ["rin"])
            assert result == ""

    def test_anima_not_in_config(self, data_dir, make_anima):
        """Handles gracefully when anima_name is not in config.animas."""
        make_anima("rin", supervisor="sakura", speciality="development")
        # sakura has no entry in config but is referenced as supervisor

        result = _build_org_context("unknown", ["rin"])
        assert "あなたがトップです" in result


# ── inject_shortterm ──────────────────────────────────────


class TestInjectShortterm:
    def test_no_shortterm(self):
        shortterm = MagicMock()
        shortterm.load_markdown.return_value = ""
        result = inject_shortterm("base prompt", shortterm)
        assert result == "base prompt"

    def test_with_shortterm(self):
        shortterm = MagicMock()
        shortterm.load_markdown.return_value = "# Short-term memory\nContent"
        result = inject_shortterm("base prompt", shortterm)
        assert "base prompt" in result
        assert "Short-term memory" in result
        assert "---" in result


# ── _normalize_headings ──────────────────────────────────


class TestNormalizeHeadings:
    def test_shifts_h1_to_h2(self):
        content = "# Title\nSome text\n## Subtitle\nMore"
        result = _normalize_headings(content)
        assert result.startswith("## Title")
        assert "## Subtitle" in result

    def test_preserves_h2_and_below(self):
        content = "## Already H2\n### H3\n#### H4"
        assert _normalize_headings(content) == content

    def test_no_headings(self):
        content = "Just text\nMore text"
        assert _normalize_headings(content) == content

    def test_skips_code_blocks(self):
        content = "## Intro\n```\n# this is a comment\n```\n# Real H1"
        result = _normalize_headings(content)
        assert "# this is a comment" in result
        assert "## Real H1" in result

    def test_no_space_after_hash_not_heading(self):
        content = "#hashtag is not a heading"
        assert _normalize_headings(content) == content

    def test_multiple_h1(self):
        content = "# First\ntext\n# Second\ntext"
        result = _normalize_headings(content)
        lines = result.split("\n")
        assert lines[0] == "## First"
        assert lines[2] == "## Second"


# ── XML tag assembly ─────────────────────────────────────


class TestAssemblyWithTags:
    def test_prompt_has_group_tags(self, tmp_path, data_dir):
        """Generated prompt must contain <group_N> tags."""
        anima_dir = tmp_path / "animas" / "alice"
        anima_dir.mkdir(parents=True)
        (anima_dir / "identity.md").write_text("I am Alice", encoding="utf-8")

        memory = MagicMock()
        memory.anima_dir = anima_dir
        memory.read_company_vision.return_value = ""
        memory.read_identity.return_value = "I am Alice"
        memory.read_injection.return_value = ""
        memory.read_permissions.return_value = ""
        memory.read_specialty_prompt.return_value = ""
        memory.read_current_state.return_value = ""
        memory.read_pending.return_value = ""
        memory.read_bootstrap.return_value = ""
        memory.list_knowledge_files.return_value = []
        memory.list_episode_files.return_value = []
        memory.list_procedure_files.return_value = []
        memory.list_skill_metas.return_value = []
        memory.list_common_skill_metas.return_value = []
        memory.collect_distilled_knowledge_separated.return_value = ([], [])
        memory.common_skills_dir = data_dir / "common_skills"
        memory.list_shared_users.return_value = []

        with patch("core.prompt.builder.load_prompt", side_effect=_mock_load_prompt_with_builder()):
            result = build_system_prompt(memory)
            prompt = result.system_prompt

        assert "<group_1" in prompt
        assert "</group_1>" in prompt
        assert "<group_2" in prompt
        assert "</group_2>" in prompt

    def test_prompt_has_section_tags(self, tmp_path, data_dir):
        """Non-header sections wrapped in <section> tags."""
        anima_dir = tmp_path / "animas" / "alice"
        anima_dir.mkdir(parents=True)
        (anima_dir / "identity.md").write_text("I am Alice", encoding="utf-8")

        memory = MagicMock()
        memory.anima_dir = anima_dir
        memory.read_company_vision.return_value = ""
        memory.read_identity.return_value = "I am Alice"
        memory.read_injection.return_value = ""
        memory.read_permissions.return_value = ""
        memory.read_specialty_prompt.return_value = ""
        memory.read_current_state.return_value = ""
        memory.read_pending.return_value = ""
        memory.read_bootstrap.return_value = ""
        memory.list_knowledge_files.return_value = []
        memory.list_episode_files.return_value = []
        memory.list_procedure_files.return_value = []
        memory.list_skill_metas.return_value = []
        memory.list_common_skill_metas.return_value = []
        memory.collect_distilled_knowledge_separated.return_value = ([], [])
        memory.common_skills_dir = data_dir / "common_skills"
        memory.list_shared_users.return_value = []

        with patch("core.prompt.builder.load_prompt", side_effect=_mock_load_prompt_with_builder()):
            result = build_system_prompt(memory)
            prompt = result.system_prompt

        assert '<section name="identity">' in prompt
        assert "</section>" in prompt

    def test_no_h1_in_output(self, tmp_path, data_dir):
        """No H1 headings should appear in the final prompt."""
        anima_dir = tmp_path / "animas" / "alice"
        anima_dir.mkdir(parents=True)
        (anima_dir / "identity.md").write_text("# Identity H1\nContent", encoding="utf-8")

        memory = MagicMock()
        memory.anima_dir = anima_dir
        memory.read_company_vision.return_value = ""
        memory.read_identity.return_value = "# Identity H1\nContent"
        memory.read_injection.return_value = "# Injection H1\n## Sub"
        memory.read_permissions.return_value = ""
        memory.read_specialty_prompt.return_value = ""
        memory.read_current_state.return_value = ""
        memory.read_pending.return_value = ""
        memory.read_bootstrap.return_value = ""
        memory.list_knowledge_files.return_value = []
        memory.list_episode_files.return_value = []
        memory.list_procedure_files.return_value = []
        memory.list_skill_metas.return_value = []
        memory.list_common_skill_metas.return_value = []
        memory.collect_distilled_knowledge_separated.return_value = ([], [])
        memory.common_skills_dir = data_dir / "common_skills"
        memory.list_shared_users.return_value = []

        with patch("core.prompt.builder.load_prompt", side_effect=_mock_load_prompt_with_builder()):
            result = build_system_prompt(memory)
            prompt = result.system_prompt

        for line in prompt.split("\n"):
            stripped = line.lstrip()
            if stripped.startswith("```"):
                continue
            if stripped.startswith("# ") and not stripped.startswith("## "):
                raise AssertionError(f"H1 heading found in output: {line!r}")

    def test_no_triple_dash_separator(self, tmp_path, data_dir):
        """Sections should not be joined by --- separators."""
        anima_dir = tmp_path / "animas" / "alice"
        anima_dir.mkdir(parents=True)
        (anima_dir / "identity.md").write_text("I am Alice", encoding="utf-8")

        memory = MagicMock()
        memory.anima_dir = anima_dir
        memory.read_company_vision.return_value = ""
        memory.read_identity.return_value = "I am Alice"
        memory.read_injection.return_value = ""
        memory.read_permissions.return_value = ""
        memory.read_specialty_prompt.return_value = ""
        memory.read_current_state.return_value = ""
        memory.read_pending.return_value = ""
        memory.read_bootstrap.return_value = ""
        memory.list_knowledge_files.return_value = []
        memory.list_episode_files.return_value = []
        memory.list_procedure_files.return_value = []
        memory.list_skill_metas.return_value = []
        memory.list_common_skill_metas.return_value = []
        memory.collect_distilled_knowledge_separated.return_value = ([], [])
        memory.common_skills_dir = data_dir / "common_skills"
        memory.list_shared_users.return_value = []

        with patch("core.prompt.builder.load_prompt", side_effect=_mock_load_prompt_with_builder()):
            result = build_system_prompt(memory)

        assert "\n\n---\n\n" not in result.system_prompt
