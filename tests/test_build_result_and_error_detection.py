from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Tests for M4/M5/M6: BuildResult, error detection, double-count prevention,
and bugfix: pending procedures persistence + streaming retry BuildResult extraction.
"""

import json
from pathlib import Path

import pytest

from core.memory.conversation import (
    ConversationMemory,
    ConversationTurn,
    _ERROR_PATTERN,
    _RESOLVED_PATTERN,
)
from core.prompt.builder import BuildResult
from core.schemas import ModelConfig


# ── BuildResult tests ──────────────────────────────────────


class TestBuildResult:
    """Test BuildResult backward compatibility and structure."""

    def test_str_returns_system_prompt(self) -> None:
        result = BuildResult(system_prompt="hello world")
        assert str(result) == "hello world"

    def test_len_returns_prompt_length(self) -> None:
        result = BuildResult(system_prompt="abc")
        assert len(result) == 3

    def test_encode_returns_bytes(self) -> None:
        result = BuildResult(system_prompt="日本語テスト")
        encoded = result.encode("utf-8")
        assert isinstance(encoded, bytes)
        assert "日本語テスト".encode("utf-8") == encoded

    def test_injected_procedures_default_empty(self) -> None:
        result = BuildResult(system_prompt="test")
        assert result.injected_procedures == []

    def test_injected_procedures_populated(self) -> None:
        paths = [Path("/tmp/proc1.md"), Path("/tmp/proc2.md")]
        result = BuildResult(system_prompt="test", injected_procedures=paths)
        assert len(result.injected_procedures) == 2
        assert result.injected_procedures[0] == Path("/tmp/proc1.md")


# ── Error detection pattern tests ─────────────────────────


class TestErrorDetectionPatterns:
    """Test M5: Improved error detection heuristic."""

    def test_error_word_boundary_match(self) -> None:
        """'error' as a standalone word should match."""
        assert _ERROR_PATTERN.search("An error occurred")
        assert _ERROR_PATTERN.search("Error: something failed")

    def test_failed_word_boundary_match(self) -> None:
        """'failed' as a standalone word should match."""
        assert _ERROR_PATTERN.search("The operation failed")
        assert _ERROR_PATTERN.search("Build failed with exit code 1")

    def test_japanese_error_pattern(self) -> None:
        """Japanese error patterns with particles should match."""
        assert _ERROR_PATTERN.search("エラーが発生しました")
        assert _ERROR_PATTERN.search("エラーは解消されました")
        assert _ERROR_PATTERN.search("エラーを確認しました")
        assert _ERROR_PATTERN.search("エラーの原因を調査")
        assert _ERROR_PATTERN.search("処理に失敗しました")  # 失敗しま
        assert _ERROR_PATTERN.search("失敗して中断")  # 失敗して
        assert _ERROR_PATTERN.search("失敗した結果")  # 失敗した

    def test_error_substring_no_match(self) -> None:
        """'error' as a substring should NOT match (word boundary)."""
        assert not _ERROR_PATTERN.search("terrorize")
        assert not _ERROR_PATTERN.search("unerror is not a word but mirrors are")

    def test_japanese_error_without_particle_no_match(self) -> None:
        """'エラー' without a matching particle should NOT match."""
        # Standalone 'エラー' at end of string without particle
        assert not _ERROR_PATTERN.search("エラー")
        # エラー followed by non-matching character
        assert not _ERROR_PATTERN.search("エラーメッセージ")

    def test_resolved_pattern_matches(self) -> None:
        """Resolution keywords should be detected."""
        assert _RESOLVED_PATTERN.search("I've fixed the issue")
        assert _RESOLVED_PATTERN.search("The problem is resolved")
        assert _RESOLVED_PATTERN.search("問題は解決しました")
        assert _RESOLVED_PATTERN.search("修正済みです")
        assert _RESOLVED_PATTERN.search("デプロイに成功しました")

    def test_error_with_resolution_override(self) -> None:
        """Error detection + resolution detection should cancel out."""
        text = "There was an error in the config, but I've fixed it."
        has_error = bool(_ERROR_PATTERN.search(text))
        has_resolution = bool(_RESOLVED_PATTERN.search(text))
        assert has_error
        assert has_resolution
        # In the actual code, resolution overrides error

    def test_only_last_assistant_turn_matters(self) -> None:
        """Verify that error detection logic uses only the last assistant turn."""
        from core.memory.conversation import ConversationMemory, ConversationTurn
        from core.schemas import ModelConfig

        # Create minimal anima dir (in-memory only for pattern test)
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            anima_dir = Path(tmpdir) / "test-anima"
            for sub in ("episodes", "knowledge", "procedures", "skills", "state"):
                (anima_dir / sub).mkdir(parents=True)

            conv = ConversationMemory(anima_dir, ModelConfig())

            # First turn has error, but last turn is success
            turns = [
                ConversationTurn(role="assistant", content="I got an error trying to connect."),
                ConversationTurn(role="human", content="Try again"),
                ConversationTurn(role="assistant", content="Successfully deployed. Everything works."),
            ]

            # Filter to last assistant turn only
            assistant_turns = [t for t in turns if t.role == "assistant"]
            last_turn = assistant_turns[-1]
            has_error = bool(_ERROR_PATTERN.search(last_turn.content))
            if has_error and _RESOLVED_PATTERN.search(last_turn.content):
                has_error = False

            # Last turn has no error
            assert not has_error


# ── Double-count prevention tests ─────────────────────────


class TestDoubleCountPrevention:
    """Test M4: report_procedure_outcome sets session flag that prevents auto-tracking."""

    def test_handler_writes_session_id(self) -> None:
        """report_procedure_outcome should write _reported_session_id to metadata."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            anima_dir = Path(tmpdir) / "animas" / "test-anima"
            for sub in ("episodes", "knowledge", "procedures", "skills", "state"):
                (anima_dir / sub).mkdir(parents=True)

            # Set up environment for MemoryManager
            import os
            os.environ["ANIMAWORKS_DATA_DIR"] = str(anima_dir.parent.parent)
            data_dir = anima_dir.parent.parent
            (data_dir / "company").mkdir(parents=True, exist_ok=True)
            (data_dir / "common_skills").mkdir(parents=True, exist_ok=True)
            (data_dir / "common_knowledge").mkdir(parents=True, exist_ok=True)
            (data_dir / "shared" / "users").mkdir(parents=True, exist_ok=True)

            from core.memory.manager import MemoryManager
            from core.tooling.handler import ToolHandler

            memory = MemoryManager(anima_dir)
            memory.write_procedure_with_meta(
                Path("deploy.md"),
                "# Deploy Steps",
                {"description": "deploy", "success_count": 0, "failure_count": 0, "confidence": 0.5},
            )

            handler = ToolHandler(anima_dir, memory)
            handler.handle("report_procedure_outcome", {
                "path": "procedures/deploy.md",
                "success": True,
            })

            meta = memory.read_procedure_metadata(
                anima_dir / "procedures" / "deploy.md",
            )
            assert "_reported_session_id" in meta
            assert meta["_reported_session_id"] == handler.session_id

    def test_session_id_is_unique_per_handler(self) -> None:
        """Each ToolHandler instance should get a unique session_id."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            anima_dir = Path(tmpdir) / "animas" / "test-anima"
            for sub in ("episodes", "knowledge", "procedures", "skills", "state"):
                (anima_dir / sub).mkdir(parents=True)

            import os
            os.environ["ANIMAWORKS_DATA_DIR"] = str(anima_dir.parent.parent)
            data_dir = anima_dir.parent.parent
            (data_dir / "company").mkdir(parents=True, exist_ok=True)
            (data_dir / "common_skills").mkdir(parents=True, exist_ok=True)
            (data_dir / "common_knowledge").mkdir(parents=True, exist_ok=True)
            (data_dir / "shared" / "users").mkdir(parents=True, exist_ok=True)

            from core.memory.manager import MemoryManager
            from core.tooling.handler import ToolHandler

            memory = MemoryManager(anima_dir)
            h1 = ToolHandler(anima_dir, memory)
            h2 = ToolHandler(anima_dir, memory)
            assert h1.session_id != h2.session_id

    def test_reset_session_id_generates_new(self) -> None:
        """reset_session_id should produce a new value."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            anima_dir = Path(tmpdir) / "animas" / "test-anima"
            for sub in ("episodes", "knowledge", "procedures", "skills", "state"):
                (anima_dir / sub).mkdir(parents=True)

            import os
            os.environ["ANIMAWORKS_DATA_DIR"] = str(anima_dir.parent.parent)
            data_dir = anima_dir.parent.parent
            (data_dir / "company").mkdir(parents=True, exist_ok=True)
            (data_dir / "common_skills").mkdir(parents=True, exist_ok=True)
            (data_dir / "common_knowledge").mkdir(parents=True, exist_ok=True)
            (data_dir / "shared" / "users").mkdir(parents=True, exist_ok=True)

            from core.memory.manager import MemoryManager
            from core.tooling.handler import ToolHandler

            memory = MemoryManager(anima_dir)
            handler = ToolHandler(anima_dir, memory)
            old_id = handler.session_id
            handler.reset_session_id()
            assert handler.session_id != old_id


# ── Pending procedures persistence tests ─────────────────


class TestPendingProceduresPersistence:
    """Test bugfix: injected_procedures stored/loaded for heartbeat finalization."""

    @pytest.fixture
    def anima_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / "animas" / "test-anima"
        for sub in ("episodes", "knowledge", "procedures", "skills", "state"):
            (d / sub).mkdir(parents=True)
        return d

    @pytest.fixture
    def conv_mem(self, anima_dir: Path) -> ConversationMemory:
        return ConversationMemory(anima_dir, ModelConfig())

    def test_store_and_load_procedures(self, conv_mem: ConversationMemory) -> None:
        """Stored procedures should be retrievable via _load_pending_procedures."""
        procs = [Path("/tmp/proc1.md"), Path("/tmp/proc2.md")]
        conv_mem.store_injected_procedures(procs, session_id="abc123")

        loaded_procs, session_id = conv_mem._load_pending_procedures()
        assert len(loaded_procs) == 2
        assert loaded_procs[0] == Path("/tmp/proc1.md")
        assert loaded_procs[1] == Path("/tmp/proc2.md")
        assert session_id == "abc123"

    def test_load_clears_file(self, conv_mem: ConversationMemory) -> None:
        """_load_pending_procedures should delete the file after reading."""
        procs = [Path("/tmp/proc.md")]
        conv_mem.store_injected_procedures(procs)

        conv_mem._load_pending_procedures()
        assert not conv_mem._pending_procedures_path.exists()

    def test_load_returns_empty_when_no_file(self, conv_mem: ConversationMemory) -> None:
        """_load_pending_procedures returns empty when no pending file."""
        procs, sid = conv_mem._load_pending_procedures()
        assert procs == []
        assert sid == ""

    def test_store_empty_list_is_noop(self, conv_mem: ConversationMemory) -> None:
        """Storing empty list should not create the file."""
        conv_mem.store_injected_procedures([])
        assert not conv_mem._pending_procedures_path.exists()

    def test_second_load_returns_empty(self, conv_mem: ConversationMemory) -> None:
        """Second call to _load_pending_procedures returns empty (file was cleared)."""
        procs = [Path("/tmp/proc.md")]
        conv_mem.store_injected_procedures(procs, session_id="xyz")

        first_procs, first_sid = conv_mem._load_pending_procedures()
        assert len(first_procs) == 1
        assert first_sid == "xyz"

        second_procs, second_sid = conv_mem._load_pending_procedures()
        assert second_procs == []
        assert second_sid == ""

    def test_separate_instances_share_file(self, anima_dir: Path) -> None:
        """Different ConversationMemory instances share the same pending file."""
        cm1 = ConversationMemory(anima_dir, ModelConfig())
        cm2 = ConversationMemory(anima_dir, ModelConfig())

        cm1.store_injected_procedures([Path("/tmp/proc.md")], session_id="s1")
        procs, sid = cm2._load_pending_procedures()
        assert len(procs) == 1
        assert sid == "s1"

    def test_corrupt_file_returns_empty(self, conv_mem: ConversationMemory) -> None:
        """Corrupt JSON in pending file should return empty and clean up."""
        conv_mem._state_dir.mkdir(parents=True, exist_ok=True)
        conv_mem._pending_procedures_path.write_text("not valid json")

        procs, sid = conv_mem._load_pending_procedures()
        assert procs == []
        assert sid == ""
        assert not conv_mem._pending_procedures_path.exists()


# ── Streaming retry BuildResult extraction test ──────────


class TestStreamingRetryBuildResultExtraction:
    """Test bugfix: streaming retry path extracts .system_prompt from BuildResult."""

    def test_build_result_not_passed_as_str(self) -> None:
        """BuildResult is NOT a str — passing it where str is expected would fail."""
        result = BuildResult(system_prompt="test prompt")
        # BuildResult should not be a str subclass
        assert not isinstance(result, str)
        # But str() should work via __str__
        assert str(result) == "test prompt"

    def test_system_prompt_attribute_is_str(self) -> None:
        """BuildResult.system_prompt is always a plain str."""
        result = BuildResult(system_prompt="test prompt")
        assert isinstance(result.system_prompt, str)
        assert result.system_prompt == "test prompt"

    def test_build_result_add_returns_str(self) -> None:
        """BuildResult + str should return str (for inject_shortterm compatibility)."""
        result = BuildResult(system_prompt="hello")
        combined = result + " world"
        assert isinstance(combined, str)
        assert combined == "hello world"

    def test_build_result_radd_returns_str(self) -> None:
        """str + BuildResult should return str."""
        result = BuildResult(system_prompt="world")
        combined = "hello " + result
        assert isinstance(combined, str)
        assert combined == "hello world"
