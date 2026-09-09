"""Tests for system prompt file fallback (ARG_MAX workaround).

When the system prompt exceeds _PROMPT_FILE_THRESHOLD, it should be written
to a temp file and passed via --system-prompt-file instead of --system-prompt.
"""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.asyncio

from core.schemas import ModelConfig
from tests.helpers.mocks import (
    patch_agent_sdk,
    patch_agent_sdk_streaming,
)

# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def model_config() -> ModelConfig:
    return ModelConfig(
        model="claude-sonnet-4-6",
        api_key="sk-test",
        context_threshold=0.50,
    )


@pytest.fixture
def anima_dir(tmp_path: Path) -> Path:
    d = tmp_path / "animas" / "test"
    d.mkdir(parents=True)
    return d


def _make_executor(model_config: ModelConfig, anima_dir: Path):
    with patch_agent_sdk():
        from core.execution.agent_sdk import AgentSDKExecutor

        return AgentSDKExecutor(
            model_config=model_config,
            anima_dir=anima_dir,
        )


# ── Tests: _build_sdk_options prompt file fallback ────────────


class TestPromptFileFallback:
    """Verify --system-prompt-file is used when prompt exceeds threshold."""

    def test_small_prompt_no_file(self, model_config, anima_dir):
        """Prompts under threshold pass directly as system_prompt."""
        executor = _make_executor(model_config, anima_dir)
        small_prompt = "You are a helpful assistant."
        with patch_agent_sdk():
            options, temp_files = executor._build_sdk_options(
                small_prompt,
                200000,
                {},
            )
        prompt_files = [f for f in temp_files if f.suffix == ".txt"]
        assert not prompt_files
        # ClaudeAgentOptions was called with system_prompt=small_prompt
        call_kwargs = options.call_args if hasattr(options, "call_args") else None
        # MagicMock records the kwargs; verify system_prompt was passed
        if call_kwargs:
            assert call_kwargs.kwargs.get("system_prompt") == small_prompt
        for f in temp_files:
            f.unlink(missing_ok=True)

    def test_large_prompt_creates_file(self, model_config, anima_dir):
        """Prompts over threshold are written to a temp file."""
        executor = _make_executor(model_config, anima_dir)
        # 120KB prompt (over 100KB threshold)
        large_prompt = "A" * 120_000
        with patch_agent_sdk():
            options, temp_files = executor._build_sdk_options(
                large_prompt,
                200000,
                {},
            )
        prompt_files = [f for f in temp_files if f.suffix == ".txt"]
        assert len(prompt_files) == 1
        prompt_file = prompt_files[0]
        assert prompt_file.exists()
        assert prompt_file.read_text(encoding="utf-8") == large_prompt
        # Verify system_prompt is None (SDK will emit --system-prompt "")
        if hasattr(options, "call_args") and options.call_args:
            assert options.call_args.kwargs.get("system_prompt") is None
            assert "system-prompt-file" in options.call_args.kwargs.get("extra_args", {})
        # Cleanup
        for f in temp_files:
            f.unlink(missing_ok=True)

    def test_large_prompt_file_content_utf8(self, model_config, anima_dir):
        """Non-ASCII prompts are correctly written as UTF-8."""
        executor = _make_executor(model_config, anima_dir)
        # Japanese text that exceeds threshold (each char ~3 bytes in UTF-8)
        large_prompt = "あ" * 40_000  # 40K chars × 3 bytes = 120KB
        with patch_agent_sdk():
            options, temp_files = executor._build_sdk_options(
                large_prompt,
                200000,
                {},
            )
        prompt_files = [f for f in temp_files if f.suffix == ".txt"]
        assert len(prompt_files) == 1
        prompt_file = prompt_files[0]
        assert prompt_file.exists()
        content = prompt_file.read_text(encoding="utf-8")
        assert content == large_prompt
        for f in temp_files:
            f.unlink(missing_ok=True)

    def test_threshold_boundary_no_file(self, model_config, anima_dir):
        """Prompt at exactly the threshold does not trigger file fallback."""
        executor = _make_executor(model_config, anima_dir)
        from core.execution.agent_sdk import _PROMPT_FILE_THRESHOLD

        # ASCII: 1 byte per char
        boundary_prompt = "X" * _PROMPT_FILE_THRESHOLD
        with patch_agent_sdk():
            options, temp_files = executor._build_sdk_options(
                boundary_prompt,
                200000,
                {},
            )
        prompt_files = [f for f in temp_files if f.suffix == ".txt"]
        assert not prompt_files
        for f in temp_files:
            f.unlink(missing_ok=True)

    def test_threshold_boundary_plus_one_creates_file(self, model_config, anima_dir):
        """Prompt one byte over the threshold triggers file fallback."""
        executor = _make_executor(model_config, anima_dir)
        from core.execution.agent_sdk import _PROMPT_FILE_THRESHOLD

        over_prompt = "X" * (_PROMPT_FILE_THRESHOLD + 1)
        with patch_agent_sdk():
            options, temp_files = executor._build_sdk_options(
                over_prompt,
                200000,
                {},
            )
        prompt_files = [f for f in temp_files if f.suffix == ".txt"]
        assert len(prompt_files) == 1
        for f in temp_files:
            f.unlink(missing_ok=True)


# ── Tests: cleanup ────────────────────────────────────────────


class TestPromptFileCleanup:
    """Verify temp prompt files are cleaned up after execution."""

    def test_cleanup_prompt_files(self, tmp_path):
        """_cleanup_prompt_files removes all listed files."""
        from core.execution.agent_sdk import _cleanup_prompt_files

        f1 = tmp_path / "prompt1.txt"
        f2 = tmp_path / "prompt2.txt"
        f1.write_text("test1")
        f2.write_text("test2")

        _cleanup_prompt_files([f1, f2])

        assert not f1.exists()
        assert not f2.exists()

    def test_cleanup_prompt_files_missing_ok(self, tmp_path):
        """_cleanup_prompt_files silently handles already-deleted files."""
        from core.execution.agent_sdk import _cleanup_prompt_files

        missing = tmp_path / "nonexistent.txt"
        _cleanup_prompt_files([missing])  # Should not raise

    def test_cleanup_prompt_files_empty_list(self):
        """_cleanup_prompt_files with empty list is a no-op."""
        from core.execution.agent_sdk import _cleanup_prompt_files

        _cleanup_prompt_files([])  # Should not raise

    async def test_execute_cleans_up_prompt_file(self, model_config, anima_dir):
        """execute() removes the temp prompt file after completion."""
        executor = _make_executor(model_config, anima_dir)
        large_prompt = "B" * 120_000

        created_files: list[Path] = []
        original_build = executor._build_sdk_options

        def _tracking_build(*args, **kwargs):
            opts, tfs = original_build(*args, **kwargs)
            created_files.extend(tfs)
            return opts, tfs

        with patch_agent_sdk():
            executor._build_sdk_options = _tracking_build
            await executor.execute(
                prompt="hello",
                system_prompt=large_prompt,
            )

        # Temp file should have been created and cleaned up
        assert len(created_files) >= 1
        for f in created_files:
            assert not f.exists(), f"Temp file was not cleaned up: {f}"

    async def test_execute_streaming_cleans_up_prompt_file(
        self,
        model_config,
        anima_dir,
    ):
        """execute_streaming() removes the temp prompt file after completion."""
        from core.prompt.context import ContextTracker

        executor = _make_executor(model_config, anima_dir)
        large_prompt = "C" * 120_000
        tracker = ContextTracker(model="claude-sonnet-4-6")

        created_files: list[Path] = []
        original_build = executor._build_sdk_options

        def _tracking_build(*args, **kwargs):
            opts, tfs = original_build(*args, **kwargs)
            created_files.extend(tfs)
            return opts, tfs

        with patch_agent_sdk_streaming():
            executor._build_sdk_options = _tracking_build
            events = []
            async for event in executor.execute_streaming(
                system_prompt=large_prompt,
                prompt="hello",
                tracker=tracker,
            ):
                events.append(event)

        # Temp file should have been created and cleaned up
        assert len(created_files) >= 1
        for f in created_files:
            assert not f.exists(), f"Temp file was not cleaned up: {f}"


class TestMcpIsolation:
    """Only the servers animaworks builds may attach to an anima's session."""

    @pytest.mark.filterwarnings("ignore::pytest.PytestWarning")
    def test_strict_mcp_config_is_always_passed(self, model_config, anima_dir):
        """Account-level claude.ai connectors must not ride along.

        Without ``--strict-mcp-config`` the CLI also loads whatever the
        logged-in claude.ai account has connected — for this fleet Gmail /
        Calendar / Drive, 40 tools whose schemas every anima paid for on
        every single request.
        """
        executor = _make_executor(model_config, anima_dir)
        with patch_agent_sdk():
            options, temp_files = executor._build_sdk_options(
                "You are a helpful assistant.",
                200000,
                {},
            )
        # ``ClaudeAgentOptions`` is mocked as ``MagicMock``, so the kwargs
        # land on the returned instance as attributes.
        assert "strict-mcp-config" in options.extra_args
        assert options.extra_args["strict-mcp-config"] is None  # boolean flag
        assert set(options.mcp_servers) == {"aw"}
        for f in temp_files:
            f.unlink(missing_ok=True)

    @pytest.mark.filterwarnings("ignore::pytest.PytestWarning")
    def test_builtin_tool_surface_is_pinned(self, model_config, anima_dir):
        """The CLI sends every built-in schema; only the used ones are worth it.

        25 built-in tools plus the account connectors measured ~69K tokens
        of a 200K window on this fleet, before any conversation.
        """
        from core.execution._sdk_options import BUILTIN_TOOLS

        executor = _make_executor(model_config, anima_dir)
        with patch_agent_sdk():
            options, temp_files = executor._build_sdk_options(
                "You are a helpful assistant.",
                200000,
                {},
            )
        assert options.extra_args["tools"] == ",".join(BUILTIN_TOOLS)
        # Bash carries almost all real tool traffic; Skill is what the
        # skill loader needs.  Losing either would be silent.
        assert "Bash" in BUILTIN_TOOLS
        assert "Skill" in BUILTIN_TOOLS
        # Losing ToolSearch re-sends every schema in full (+11K measured).
        assert "ToolSearch" in BUILTIN_TOOLS
        # Subagents are off for animas; delegation goes through aw MCP.
        assert not [t for t in BUILTIN_TOOLS if t.startswith("Task")]
        for f in temp_files:
            f.unlink(missing_ok=True)
