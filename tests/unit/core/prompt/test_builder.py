"""Unit tests for core/prompt/builder.py — system prompt construction."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.prompt.builder import (
    BuildResult,
    _build_messaging_section,
    _build_org_context,
    _discover_other_animas,
    _format_anima_entry,
    build_system_prompt,
    inject_shortterm,
)
from core.schemas import SkillMeta


def _mock_load_prompt_with_builder(default: str = "section"):
    """Return a side_effect function that renders builder/* templates minimally."""

    def _mock(name: str, **kwargs) -> str:
        if name == "builder/common_skills_header":
            lines = kwargs.get("common_skill_lines", "")
            return f"## 共通スキル\n\n| スキル名 | 概要 |\n|---------|------|\n{lines}"
        if name == "builder/task_in_progress":
            return f"## ⚠️ 進行中タスク\n\n{kwargs.get('state', '')}"
        if name == "builder/task_queue":
            return f"## 未完了タスク\n\n{kwargs.get('task_summary', '')}"
        if name == "builder/external_tools_guide":
            cats = kwargs.get("categories", "")
            return (
                f"外部ツールを使うには `discover_tools` を呼んでください。\n"
                f"カテゴリ: {cats}"
            )
        if name == "builder/hiring_rules_a1":
            return "## 雇用ルール\n\ncreate-anima"
        if name == "builder/hiring_rules_other":
            return "## 雇用ルール\n\ncreate-anima"
        if name == "builder/skill_injection":
            return (
                f"## {kwargs.get('section_title', '')}: "
                f"{kwargs.get('skill_name', '')} {kwargs.get('label', '')}\n\n"
                f"{kwargs.get('body', '')}"
            )
        if name == "builder/procedures_header":
            return f"## 手順書\n\n{kwargs.get('proc_lines', '')}"
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
        with patch("core.prompt.builder.load_prompt", return_value="messaging section"):
            result = _build_messaging_section(anima_dir, ["bob", "charlie"])
            assert result == "messaging section"

    def test_no_animas(self, tmp_path):
        anima_dir = tmp_path / "alice"
        anima_dir.mkdir()
        with patch("core.prompt.builder.load_prompt", return_value="messaging section") as mock_lp:
            _build_messaging_section(anima_dir, [])
            call_kwargs = mock_lp.call_args[1]
            assert "(まだ他の社員はいません)" in call_kwargs["animas_line"]

    def test_a1_mode_uses_messaging_a1_template(self, tmp_path):
        """A1 mode should load the messaging_a1 template."""
        anima_dir = tmp_path / "alice"
        anima_dir.mkdir()
        with patch("core.prompt.builder.load_prompt", return_value="a1 messaging") as mock_lp:
            result = _build_messaging_section(anima_dir, ["bob"], execution_mode="a1")
            assert result == "a1 messaging"
            mock_lp.assert_called_once()
            assert mock_lp.call_args[0][0] == "messaging_a1"

    def test_a2_mode_uses_messaging_template(self, tmp_path):
        """A2 mode should load the standard messaging template."""
        anima_dir = tmp_path / "alice"
        anima_dir.mkdir()
        with patch("core.prompt.builder.load_prompt", return_value="a2 messaging") as mock_lp:
            result = _build_messaging_section(anima_dir, ["bob"], execution_mode="a2")
            assert result == "a2 messaging"
            mock_lp.assert_called_once()
            assert mock_lp.call_args[0][0] == "messaging"

    def test_default_mode_uses_a1_template(self, tmp_path):
        """Default execution_mode should be a1, using messaging_a1 template."""
        anima_dir = tmp_path / "alice"
        anima_dir.mkdir()
        with patch("core.prompt.builder.load_prompt", return_value="section") as mock_lp:
            _build_messaging_section(anima_dir, ["bob"])
            assert mock_lp.call_args[0][0] == "messaging_a1"


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
        memory.common_skills_dir = data_dir / "common_skills"
        memory.list_shared_users.return_value = []

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
        memory.common_skills_dir = data_dir / "common_skills"
        memory.list_shared_users.return_value = []

        with patch("core.prompt.builder.load_prompt", return_value="prompt"):
            result = build_system_prompt(memory)
            assert "I am Alice" in result

    def test_includes_skills(self, tmp_path, data_dir):
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
        memory.list_knowledge_files.return_value = []
        memory.list_episode_files.return_value = []
        memory.list_procedure_files.return_value = []
        memory.list_skill_summaries.return_value = [("coding", "Write code")]
        memory.list_common_skill_summaries.return_value = [("deploy", "Deploy apps")]
        memory.list_skill_metas.return_value = [SkillMeta(name="coding", description="Write code", path=Path("/tmp/test/skills/coding.md"), is_common=False)]
        memory.list_common_skill_metas.return_value = [SkillMeta(name="deploy", description="Deploy apps", path=Path("/tmp/test/common_skills/deploy.md"), is_common=True)]
        memory.common_skills_dir = data_dir / "common_skills"
        memory.list_shared_users.return_value = []

        with patch("core.prompt.builder.load_prompt", side_effect=_mock_load_prompt_with_builder()):
            result = build_system_prompt(memory)
            # Common skills section is built inline (not via load_prompt)
            assert "共通スキル" in result
            assert "deploy" in result
            # Personal skills are built via load_prompt("skills_guide", ...)
            # which returns "section" in mock; verify it was called
            # The memory_guide template gets skill names as kwargs
            assert "coding" in result or "section" in result

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
        memory.common_skills_dir = data_dir / "common_skills"
        memory.list_shared_users.return_value = []

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
        memory.common_skills_dir = data_dir / "common_skills"
        memory.list_shared_users.return_value = []

        with patch("core.prompt.builder.load_prompt", side_effect=_mock_load_prompt_with_builder()):
            result = build_system_prompt(memory)
            # Non-idle state goes to "進行中タスク" branch
            assert "進行中タスク" in result
            assert "status: working" in result
            assert "未完了タスク" in result
            assert "task 1" in result

    def test_a2_mode_injects_discover_tools_guide(self, tmp_path, data_dir):
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
        memory.common_skills_dir = data_dir / "common_skills"
        memory.list_shared_users.return_value = []

        with patch("core.prompt.builder.load_prompt", side_effect=_mock_load_prompt_with_builder()):
            result = build_system_prompt(
                memory,
                tool_registry=["chatwork", "slack"],
                execution_mode="a2",
            )
            assert "discover_tools" in result
            assert "chatwork" in result

    def test_a1_mode_uses_cli_guide(self, tmp_path, data_dir):
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
        memory.common_skills_dir = data_dir / "common_skills"
        memory.list_shared_users.return_value = []

        with patch("core.prompt.builder.load_prompt", return_value="section"), \
             patch("core.tooling.guide.build_tools_guide", return_value="CLI guide") as mock_guide:
            result = build_system_prompt(
                memory,
                tool_registry=["chatwork"],
                execution_mode="a1",
            )
            # A1 mode should call the CLI guide builder
            mock_guide.assert_called_once()


# ── _format_anima_entry ──────────────────────────────────


class TestFormatAnimaEntry:
    def test_with_speciality(self):
        assert _format_anima_entry("alice", "frontend") == "alice (frontend)"

    def test_without_speciality(self):
        assert _format_anima_entry("alice", None) == "alice"

    def test_empty_speciality(self):
        assert _format_anima_entry("alice", "") == "alice"


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
        assert "rin (development)" in result
        assert "kotoha (communication)" in result

    def test_middle_manager(self, data_dir, make_anima):
        """Middle manager sees supervisor, subordinates, and peers."""
        make_anima("sakura")
        make_anima("rin", supervisor="sakura", speciality="development")
        make_anima("kotoha", supervisor="sakura", speciality="communication")
        make_anima("alice", supervisor="rin", speciality="frontend")

        result = _build_org_context("rin", ["sakura", "kotoha", "alice"])
        # Supervisor
        assert "sakura" in result
        # Subordinate
        assert "alice (frontend)" in result
        # Peer
        assert "kotoha (communication)" in result

    def test_leaf_worker(self, data_dir, make_anima):
        """Leaf worker sees supervisor and peers but no subordinates."""
        make_anima("sakura")
        make_anima("rin", supervisor="sakura", speciality="development")
        make_anima("alice", supervisor="rin", speciality="frontend")
        make_anima("bob", supervisor="rin", speciality="backend")

        result = _build_org_context("alice", ["sakura", "rin", "bob"])
        # Supervisor
        assert "rin (development)" in result
        # No subordinates
        assert "部下" in result
        assert "(なし)" in result
        # Peer
        assert "bob (backend)" in result

    def test_solo_anima(self, data_dir, make_anima):
        """Solo anima with no relationships."""
        make_anima("sakura")

        result = _build_org_context("sakura", [])
        assert "あなたがトップです" in result
        # No communication rules when alone
        assert "コミュニケーションルール" not in result

    def test_communication_rules_injected_when_others_exist(
        self, data_dir, make_anima
    ):
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


# ── hiring_context placement ─────────────────────────────


class TestHiringContextPlacement:
    """Verify hiring_context is injected before behavior_rules."""

    def _build_solo_prompt(self, tmp_path, data_dir):
        """Helper: build system prompt for a solo anima with no supervisor."""
        anima_dir = tmp_path / "animas" / "solo"
        anima_dir.mkdir(parents=True)
        (anima_dir / "identity.md").write_text("I am Solo", encoding="utf-8")

        memory = MagicMock()
        memory.anima_dir = anima_dir
        memory.read_company_vision.return_value = ""
        memory.read_identity.return_value = "I am Solo"
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
        memory.common_skills_dir = data_dir / "common_skills"
        memory.list_shared_users.return_value = []

        # read_model_config returns a ModelConfig with supervisor=None
        from core.schemas import ModelConfig
        memory.read_model_config.return_value = ModelConfig()

        return memory

    def test_hiring_context_before_behavior_rules(self, tmp_path, data_dir):
        """hiring_context must appear before behavior_rules in the prompt."""
        memory = self._build_solo_prompt(tmp_path, data_dir)

        # Use real load_prompt so both templates are loaded with real content
        result = build_system_prompt(memory)

        # hiring_context contains "チーム構成について"
        # behavior_rules contains "行動ルール"
        assert "チーム構成について" in result
        assert "行動ルール" in result
        assert result.index("チーム構成について") < result.index("行動ルール")

    def test_hiring_context_not_injected_with_peers(self, tmp_path, data_dir):
        """hiring_context must NOT be injected when other animas exist."""
        animas_root = tmp_path / "animas"
        animas_root.mkdir(parents=True, exist_ok=True)
        solo = animas_root / "solo"
        solo.mkdir()
        (solo / "identity.md").write_text("I am Solo", encoding="utf-8")
        peer = animas_root / "peer"
        peer.mkdir()
        (peer / "identity.md").write_text("I am Peer", encoding="utf-8")

        memory = MagicMock()
        memory.anima_dir = solo
        memory.read_company_vision.return_value = ""
        memory.read_identity.return_value = "I am Solo"
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
        memory.common_skills_dir = data_dir / "common_skills"
        memory.list_shared_users.return_value = []

        result = build_system_prompt(memory)

        assert "チーム構成について" not in result

    def test_hiring_context_not_injected_with_supervisor(self, tmp_path, data_dir):
        """hiring_context must NOT be injected when anima has a supervisor."""
        memory = self._build_solo_prompt(tmp_path, data_dir)

        from core.schemas import ModelConfig
        memory.read_model_config.return_value = ModelConfig(supervisor="boss")

        result = build_system_prompt(memory)

        assert "チーム構成について" not in result


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
