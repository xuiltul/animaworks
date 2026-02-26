from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""E2E integration tests for prompt injection defense boundary labeling."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.execution._sanitize import wrap_tool_result
from core.memory.priming import PrimingResult, format_priming_section
from core.prompt.builder import build_system_prompt


# ── Helpers ────────────────────────────────────────────────────


def _make_mock_memory(tmp_path: Path) -> MagicMock:
    """Build a minimal MemoryManager mock for build_system_prompt."""
    memory = MagicMock()
    memory.anima_dir = tmp_path / "animas" / "test_anima"
    memory.anima_dir.mkdir(parents=True, exist_ok=True)
    memory.read_identity.return_value = "test identity"
    memory.read_injection.return_value = ""
    memory.read_permissions.return_value = ""
    memory.read_specialty_prompt.return_value = ""
    memory.read_bootstrap.return_value = ""
    memory.read_company_vision.return_value = ""
    memory.read_current_state.return_value = ""
    memory.read_pending.return_value = ""
    memory.read_resolutions.return_value = []
    memory.read_model_config.return_value = MagicMock(
        model="claude-sonnet-4-20250514", supervisor=None, max_chains=3
    )
    memory.list_knowledge_files.return_value = []
    memory.list_episode_files.return_value = []
    memory.list_procedure_files.return_value = []
    memory.list_skill_metas.return_value = []
    memory.list_common_skill_metas.return_value = []
    memory.list_procedure_metas.return_value = []
    memory.list_shared_users.return_value = []
    memory.collect_distilled_knowledge.return_value = []
    memory.collect_distilled_knowledge_separated.return_value = ([], [])
    memory.common_skills_dir = tmp_path / "common_skills"
    return memory


# ── format_priming_section tests ────────────────────────────────


def test_format_priming_section_wraps_sender_profile() -> None:
    """Create a PrimingResult with sender_profile, call format_priming_section, verify output."""
    result = PrimingResult(sender_profile="User profile data")
    formatted = format_priming_section(result, sender_name="human")
    assert '<priming source="sender_profile" trust="medium">' in formatted
    assert "User profile data" in formatted


def test_format_priming_section_wraps_recent_activity() -> None:
    """PrimingResult with recent_activity -> <priming source="recent_activity" trust="untrusted">."""
    result = PrimingResult(recent_activity="Recent activity timeline")
    formatted = format_priming_section(result, sender_name="human")
    assert '<priming source="recent_activity" trust="untrusted">' in formatted
    assert "Recent activity timeline" in formatted


def test_format_priming_section_wraps_related_knowledge() -> None:
    """PrimingResult with related_knowledge -> <priming source="related_knowledge" trust="medium">."""
    result = PrimingResult(related_knowledge="Related knowledge from RAG")
    formatted = format_priming_section(result, sender_name="human")
    assert '<priming source="related_knowledge" trust="medium">' in formatted
    assert "Related knowledge from RAG" in formatted


def test_format_priming_section_wraps_pending_tasks() -> None:
    """PrimingResult with pending_tasks -> <priming source="pending_tasks" trust="medium">."""
    result = PrimingResult(pending_tasks="Task 1, Task 2")
    formatted = format_priming_section(result, sender_name="human")
    assert '<priming source="pending_tasks" trust="medium">' in formatted
    assert "Task 1, Task 2" in formatted


def test_format_priming_section_skills_not_wrapped() -> None:
    """PrimingResult with matched_skills -> output does NOT contain <priming> for skills."""
    result = PrimingResult(matched_skills=["skill1", "skill2"])
    formatted = format_priming_section(result, sender_name="human")
    assert "skill1" in formatted
    assert "skill2" in formatted
    assert "使えそうなスキル" in formatted
    assert "あなたが持っているスキル" in formatted
    # Skills are plain text only - no <priming> wrapper
    assert "<priming" not in formatted


def test_format_priming_section_empty_result() -> None:
    """Empty PrimingResult returns empty string."""
    result = PrimingResult()
    formatted = format_priming_section(result, sender_name="human")
    assert formatted == ""


def test_wrap_tool_result_preserves_json_content() -> None:
    """JSON result content is preserved intact inside tags."""
    json_content = '{"items": [{"id": 1, "name": "a"}, {"id": 2}], "total": 2}'
    wrapped = wrap_tool_result("web_search", json_content)
    assert json_content in wrapped
    assert wrapped.startswith('<tool_result tool="web_search" trust="untrusted">')
    assert wrapped.endswith("</tool_result>")
    # Verify the JSON is exactly as supplied (no escaping/alteration)
    assert '{"items": [{"id": 1, "name": "a"}, {"id": 2}], "total": 2}' in wrapped


# ── builder interpretation rules ─────────────────────────────────


def test_builder_includes_interpretation_rules(tmp_path: Path) -> None:
    """Build system prompt and verify it contains interpretation rules."""
    memory = _make_mock_memory(tmp_path)

    def _mock_load_prompt(name: str, **kwargs: object) -> str:
        if name == "tool_data_interpretation":
            return "## ツール結果・外部データの解釈ルール\n\n- Content here"
        return f"[{name}]"

    with patch("core.prompt.builder.load_prompt", side_effect=_mock_load_prompt):
        result = build_system_prompt(memory)

    assert "ツール結果・外部データの解釈ルール" in result.system_prompt
