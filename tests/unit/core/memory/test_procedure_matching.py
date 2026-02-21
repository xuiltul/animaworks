from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Unit tests for Phase 2: 3-tier matching + auto-injection for procedures."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.schemas import SkillMeta


# ── Fixtures ──────────────────────────────────────────────


@pytest.fixture
def anima_dir(tmp_path: Path) -> Path:
    """Create a minimal anima directory."""
    d = tmp_path / "animas" / "test-anima"
    for sub in ("episodes", "knowledge", "procedures", "skills", "state"):
        (d / sub).mkdir(parents=True)
    return d


@pytest.fixture
def memory(anima_dir: Path, monkeypatch: pytest.MonkeyPatch):
    """Create a MemoryManager that skips RAG initialization."""
    monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(anima_dir.parent.parent))

    data_dir = anima_dir.parent.parent
    (data_dir / "company").mkdir(parents=True, exist_ok=True)
    (data_dir / "common_skills").mkdir(parents=True, exist_ok=True)
    (data_dir / "common_knowledge").mkdir(parents=True, exist_ok=True)
    (data_dir / "shared" / "users").mkdir(parents=True, exist_ok=True)

    from core.memory.manager import MemoryManager

    return MemoryManager(anima_dir)


# ── 2-1: match_skills_by_description with procedures ─────


class TestProcedureSkillMatching:
    """Test that procedures can be matched via the existing 3-tier engine."""

    def test_procedure_meta_matches_tier1(self, anima_dir: Path) -> None:
        """Procedure with bracket keywords should match Tier 1."""
        from core.memory.manager import match_skills_by_description

        procedure_meta = SkillMeta(
            name="deploy",
            description="「デプロイ」手順、「リリース」時に使用",
            path=anima_dir / "procedures" / "deploy.md",
            is_common=False,
        )
        matched = match_skills_by_description(
            "デプロイをお願いします", [procedure_meta],
        )
        assert len(matched) == 1
        assert matched[0].name == "deploy"

    def test_procedure_meta_matches_tier2(self, anima_dir: Path) -> None:
        """Procedure with vocabulary keywords should match Tier 2."""
        from core.memory.manager import match_skills_by_description

        procedure_meta = SkillMeta(
            name="backup",
            description="database backup procedure with verification steps",
            path=anima_dir / "procedures" / "backup.md",
            is_common=False,
        )
        matched = match_skills_by_description(
            "I need to run the database backup and verify it",
            [procedure_meta],
        )
        assert len(matched) == 1
        assert matched[0].name == "backup"

    def test_mixed_skills_and_procedures(self, anima_dir: Path) -> None:
        """Skills and procedures can both be matched in a single call."""
        from core.memory.manager import match_skills_by_description

        skill = SkillMeta(
            name="git-flow",
            description="「git」ブランチ戦略、「マージ」手順",
            path=anima_dir / "skills" / "git-flow.md",
            is_common=False,
        )
        procedure = SkillMeta(
            name="release-checklist",
            description="「リリース」前のチェックリスト、「デプロイ」手順",
            path=anima_dir / "procedures" / "release-checklist.md",
            is_common=False,
        )
        matched = match_skills_by_description(
            "リリースの準備をお願いします",
            [skill, procedure],
        )
        assert len(matched) == 1
        assert matched[0].name == "release-checklist"


# ── 2-2: builder.py procedures injection ─────────────────


class TestBuilderProcedureInjection:
    """Test that build_system_prompt includes procedure matching."""

    def test_procedure_appears_in_unified_table(self, memory, anima_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Procedures should appear in the unified skills/procedures table."""
        # Write a procedure
        (anima_dir / "procedures" / "deploy.md").write_text(
            "---\ndescription: \"「デプロイ」手順\"\n---\n\n# Deploy Steps\n\n1. Pull\n2. Build\n3. Deploy",
            encoding="utf-8",
        )

        # Mock load_prompt to avoid needing template files
        def mock_load_prompt(name, **kwargs):
            if name == "skills_guide":
                return (
                    "## スキルと手順書\n\n"
                    "スキルと手順書はあなたが持つ能力・作業手順です。\n"
                    "使用する際は該当ファイルをReadで読んでから実行してください。"
                )
            return f"[{name}]"

        monkeypatch.setattr("core.prompt.builder.load_prompt", mock_load_prompt)

        # Mock load_config to avoid needing real config
        mock_config = MagicMock()
        mock_config.animas = {}
        mock_config.human_notification.enabled = False
        monkeypatch.setattr("core.prompt.builder._build_org_context", lambda *a, **kw: "")

        from core.prompt.builder import build_system_prompt

        result = build_system_prompt(
            memory,
            message="デプロイをお願いします",
        )
        prompt = result.system_prompt

        assert "| deploy | 手順 |" in prompt

    def test_all_procedures_in_unified_table(self, memory, anima_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """ALL procedures always appear in the unified table regardless of message content."""
        (anima_dir / "procedures" / "backup.md").write_text(
            "---\ndescription: backup procedure\n---\n\n# Backup",
            encoding="utf-8",
        )

        def mock_load_prompt(name, **kwargs):
            if name == "skills_guide":
                return (
                    "## スキルと手順書\n\n"
                    "スキルと手順書はあなたが持つ能力・作業手順です。\n"
                    "使用する際は該当ファイルをReadで読んでから実行してください。"
                )
            return f"[{name}]"

        monkeypatch.setattr("core.prompt.builder.load_prompt", mock_load_prompt)
        monkeypatch.setattr("core.prompt.builder._build_org_context", lambda *a, **kw: "")

        mock_config = MagicMock()
        mock_config.animas = {}
        mock_config.human_notification.enabled = False

        from core.prompt.builder import build_system_prompt

        result = build_system_prompt(
            memory,
            message="こんにちは",  # Unrelated message — procedure still appears
        )
        prompt = result.system_prompt

        assert "| backup | 手順 |" in prompt

    def test_injected_procedures_always_empty(self, memory, anima_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """BuildResult.injected_procedures is always an empty list."""
        (anima_dir / "procedures" / "deploy.md").write_text(
            "---\ndescription: \"「デプロイ」手順\"\n---\n\n# Deploy",
            encoding="utf-8",
        )

        def mock_load_prompt(name, **kwargs):
            if name == "skills_guide":
                return (
                    "## スキルと手順書\n\n"
                    "スキルと手順書はあなたが持つ能力・作業手順です。\n"
                    "使用する際は該当ファイルをReadで読んでから実行してください。"
                )
            return f"[{name}]"

        monkeypatch.setattr("core.prompt.builder.load_prompt", mock_load_prompt)
        monkeypatch.setattr("core.prompt.builder._build_org_context", lambda *a, **kw: "")

        from core.prompt.builder import build_system_prompt

        result = build_system_prompt(memory, message="デプロイをお願いします")

        assert result.injected_procedures == []


# ── 2-3: Priming Channel D with procedures ──────────────


class TestPrimingChannelDProcedures:
    """Test that Channel D searches procedures/ in addition to skills/."""

    async def test_channel_d_matches_procedure_filename(self, anima_dir: Path) -> None:
        """Procedure filenames should be matched in Channel D."""
        # Create a procedure file
        proc_dir = anima_dir / "procedures"
        proc_dir.mkdir(parents=True, exist_ok=True)
        (proc_dir / "deploy-pipeline.md").write_text(
            "---\ndescription: deploy pipeline\n---\n\n# Deploy Pipeline",
            encoding="utf-8",
        )

        from core.memory.priming import PrimingEngine

        engine = PrimingEngine(anima_dir)

        result = await engine._channel_d_skill_match("deploy pipeline", ["deploy"])
        assert "deploy-pipeline" in result

    async def test_channel_d_matches_procedure_content(self, anima_dir: Path) -> None:
        """Procedure file content should be matched in Channel D."""
        proc_dir = anima_dir / "procedures"
        proc_dir.mkdir(parents=True, exist_ok=True)
        (proc_dir / "ops-runbook.md").write_text(
            "---\ndescription: incident response runbook\n---\n\n# Incident Response\n\nWhen an alert fires...",
            encoding="utf-8",
        )

        from core.memory.priming import PrimingEngine

        engine = PrimingEngine(anima_dir)

        result = await engine._channel_d_skill_match("incident response", ["incident"])
        assert "ops-runbook" in result

    async def test_channel_d_no_double_count(self, anima_dir: Path) -> None:
        """Same-named skill and procedure should not produce duplicates."""
        skills_dir = anima_dir / "skills"
        proc_dir = anima_dir / "procedures"
        skills_dir.mkdir(parents=True, exist_ok=True)
        proc_dir.mkdir(parents=True, exist_ok=True)

        (skills_dir / "deploy.md").write_text(
            "---\ndescription: \"「deploy」アプリケーションのデプロイ手順\"\n---\n\n# Deploy Skill",
            encoding="utf-8",
        )
        (proc_dir / "deploy.md").write_text(
            "---\ndescription: \"「deploy」デプロイ手続き\"\n---\n\n# Deploy Procedure",
            encoding="utf-8",
        )

        from core.memory.priming import PrimingEngine

        engine = PrimingEngine(anima_dir)

        result = await engine._channel_d_skill_match("deploy the application", ["deploy"])
        # "deploy" should appear only once due to dedup (personal skill takes precedence)
        assert result.count("deploy") == 1
