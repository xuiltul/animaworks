"""Unit tests for DK prompt-injection removal in builder.py."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from core.prompt.builder import BuildResult, build_system_prompt


def _make_mock_memory(anima_dir: Path, data_dir: Path) -> MagicMock:
    """Create a mock MemoryManager with standard stubs."""
    memory = MagicMock()
    memory.anima_dir = anima_dir
    memory.read_company_vision.return_value = ""
    memory.read_identity.return_value = "I am Test Anima"
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
    memory.list_procedure_metas.return_value = []
    memory.common_skills_dir = data_dir / "common_skills"
    memory.list_shared_users.return_value = []
    model_cfg = MagicMock()
    model_cfg.model = "claude-sonnet-4-6"
    model_cfg.supervisor = None
    memory.read_model_config.return_value = model_cfg
    return memory


class TestDKPromptInjectionRemoved:
    def test_knowledge_and_procedure_summaries_not_injected(
        self,
        tmp_path: Path,
        data_dir: Path,
    ) -> None:
        """DK sections are no longer part of the fixed system prompt."""
        anima_dir = tmp_path / "animas" / "alice"
        anima_dir.mkdir(parents=True)
        (anima_dir / "identity.md").write_text("I am Alice", encoding="utf-8")

        memory = _make_mock_memory(anima_dir, data_dir)
        memory.list_knowledge_files.return_value = [anima_dir / "knowledge" / "python-basics.md"]
        memory.list_procedure_files.return_value = [anima_dir / "procedures" / "deploy-procedure.md"]
        memory.collect_distilled_knowledge_separated = MagicMock(
            return_value=(
                [
                    {
                        "name": "deploy-procedure",
                        "content": "FULL_PROC_BODY",
                        "description": "Docker deploy steps",
                        "confidence": 0.7,
                        "path": "/tmp/test/deploy-procedure.md",
                        "mtime": 0.0,
                    },
                ],
                [
                    {
                        "name": "python-basics",
                        "content": "FULL_KNOWLEDGE_BODY",
                        "description": "Python language overview",
                        "confidence": 0.9,
                        "path": "/tmp/test/python-basics.md",
                        "mtime": 0.0,
                    },
                ],
            )
        )

        with patch("core.prompt.builder._build_org_context", return_value=""):
            result = build_system_prompt(memory)

        assert isinstance(result, BuildResult)
        assert "dk_procedures" not in result.system_prompt
        assert "dk_knowledge" not in result.system_prompt
        assert "Docker deploy steps" not in result.system_prompt
        assert "Python language overview" not in result.system_prompt
        assert "FULL_PROC_BODY" not in result.system_prompt
        assert "FULL_KNOWLEDGE_BODY" not in result.system_prompt
        memory.collect_distilled_knowledge_separated.assert_not_called()

    def test_build_result_contains_only_system_prompt(self) -> None:
        result = BuildResult(system_prompt="hello")

        assert result.system_prompt == "hello"
        assert not hasattr(result, "injected_procedures")
        assert not hasattr(result, "injected_knowledge_files")
        assert not hasattr(result, "overflow_files")
