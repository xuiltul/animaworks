# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Engine-neutral system prompt tests.

Verifies that the prompt builder:
- prepends the per-mode host built-in tool line (tool_guide.host_tools.{s,c,x,a})
- does not emit Claude-specific tool names for non-S modes (c/x/a)
- no longer includes the retired "タスク期限" section or the AI-speed table
Implemented per issue 20260829_prompt-deadline-and-engine-neutral;
task deadlines removed per the A1 task-model teardown plan (2026-09).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from core.prompt.builder import build_system_prompt

_NEUTRAL_GUIDE = "### Tools\nUser guide content."  # intentionally free of Claude tool names

_MODES = ["s", "c", "x", "a"]

_HOST_LINE_MARKERS = {
    "s": "Read / Write / Edit / Bash / Grep / Glob / WebSearch / WebFetch",
    "c": "exec_command",  # host built-in tools for Codex
    "x": "Grok CLI",  # host built-in tools for Grok
    "a": "Read / Write / Edit / Bash / Grep / Glob / WebSearch / WebFetch",  # A exposes Claude-compatible names
}


def _make_memory(anima_dir, data_dir) -> MagicMock:
    """Standard MemoryManager mock matching existing builder tests."""
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
    memory.list_skill_summaries.return_value = []
    memory.list_common_skill_summaries.return_value = []
    memory.list_skill_metas.return_value = []
    memory.list_common_skill_metas.return_value = []
    memory.common_skills_dir = data_dir / "common_skills"
    memory.list_shared_users.return_value = []
    return memory


def _build(memory, mode: str, anima_dir) -> str:
    """Build the system prompt for a given execution mode using a neutral tool guide."""
    with patch("core.prompt.builder.load_guide", return_value=_NEUTRAL_GUIDE):
        result = build_system_prompt(memory, execution_mode=mode)
    return result.system_prompt


@pytest.fixture
def prompt_factory(tmp_path, data_dir):
    anima_dir = tmp_path / "animas" / "alice"
    anima_dir.mkdir(parents=True)
    (anima_dir / "identity.md").write_text("I am Alice", encoding="utf-8")
    (anima_dir / "state").mkdir()

    memory = _make_memory(anima_dir, data_dir)

    def _factory(mode: str) -> str:
        return _build(memory, mode, anima_dir)

    return _factory


@pytest.mark.parametrize("mode", _MODES)
def test_host_tool_line_present(mode: str, prompt_factory) -> None:
    """Every mode gets its §4 host built-in tool line."""
    prompt = prompt_factory(mode)
    assert "ホストの組込みツール" in prompt
    assert _HOST_LINE_MARKERS[mode] in prompt


@pytest.mark.parametrize("mode", ["c", "x"])
def test_non_s_modes_are_engine_neutral(mode: str, prompt_factory) -> None:
    """Codex/Grok modes must not leak Claude-specific tool names."""
    prompt = prompt_factory(mode)
    assert "Read / Write / Edit" not in prompt
    assert "Glob" not in prompt
    assert "Grep" not in prompt


def test_a_mode_uses_claude_compatible_names(prompt_factory) -> None:
    """A (LiteLLM) mode exposes Claude-compatible host tool names."""
    prompt = prompt_factory("a")
    assert "ホストの組込みツール" in prompt
    assert "Glob" in prompt
    assert "Grep" in prompt


def test_s_mode_retains_host_tool_names(prompt_factory) -> None:
    """S (Claude SDK) mode keeps the Claude tool names in its host line."""
    prompt = prompt_factory("s")
    assert "Read / Write / Edit" in prompt
    assert "Glob" in prompt


def test_no_task_deadline_section_or_ai_speed_table(prompt_factory) -> None:
    """Task deadlines were torn out (A1 task-model teardown); AI-speed table stays gone too."""
    prompt = prompt_factory("s")
    assert "タスク期限" not in prompt
    assert "AI-speed" not in prompt
    assert "| New implementation" not in prompt
