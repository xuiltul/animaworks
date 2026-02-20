from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Tests for replied_to file persistence across execution modes.

Covers Change 2: replied_to.jsonl write/read/corruption handling.

- ToolHandler._persist_replied_to() writes entries to run/replied_to.jsonl
- BaseExecutor._read_replied_to_file() reads and filters entries
- Corrupted lines are gracefully skipped
- Legacy _SEND_PATTERNS and _parse_replied_to are removed from agent_sdk
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest


class TestPersistRepliedTo:
    """Test ToolHandler._persist_replied_to() writes to JSONL file."""

    def _make_handler(self, tmp_path: Path):
        from core.tooling.handler import ToolHandler

        memory = MagicMock()
        return ToolHandler(anima_dir=tmp_path, memory=memory)

    def test_persist_creates_file_and_appends(self, tmp_path):
        handler = self._make_handler(tmp_path)
        handler._persist_replied_to("alice", success=True)
        handler._persist_replied_to("bob", success=True)
        handler._persist_replied_to("charlie", success=False)

        path = tmp_path / "run" / "replied_to.jsonl"
        assert path.exists()
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3

        entry0 = json.loads(lines[0])
        assert entry0 == {"to": "alice", "success": True}

        entry2 = json.loads(lines[2])
        assert entry2 == {"to": "charlie", "success": False}

    def test_persist_creates_run_directory(self, tmp_path):
        handler = self._make_handler(tmp_path)
        # run/ directory does not exist yet
        assert not (tmp_path / "run").exists()
        handler._persist_replied_to("alice", success=True)
        assert (tmp_path / "run" / "replied_to.jsonl").exists()


class TestReadRepliedToFile:
    """Test BaseExecutor._read_replied_to_file() reads from JSONL file."""

    def _make_executor(self, tmp_path: Path):
        """Create a concrete executor for testing the base method."""
        from core.execution.base import BaseExecutor
        from core.schemas import ModelConfig

        class _TestExecutor(BaseExecutor):
            async def execute(self, prompt, system_prompt="", tracker=None,
                            shortterm=None, trigger="", images=None):
                raise NotImplementedError

        config = ModelConfig(model="test-model")
        return _TestExecutor(config, tmp_path)

    def test_read_nonexistent_file_returns_empty_set(self, tmp_path):
        executor = self._make_executor(tmp_path)
        assert executor._read_replied_to_file() == set()

    def test_read_valid_entries(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir(parents=True)
        path = run_dir / "replied_to.jsonl"
        path.write_text(
            '{"to": "alice", "success": true}\n'
            '{"to": "bob", "success": false}\n'
            '{"to": "charlie", "success": true}\n',
            encoding="utf-8",
        )
        executor = self._make_executor(tmp_path)
        result = executor._read_replied_to_file()
        assert result == {"alice", "charlie"}  # bob excluded (success=false)

    def test_read_corrupted_lines_are_skipped(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir(parents=True)
        path = run_dir / "replied_to.jsonl"
        path.write_text(
            '{"to": "alice", "success": true}\n'
            'NOT VALID JSON\n'
            '{"to": "charlie", "success": true}\n',
            encoding="utf-8",
        )
        executor = self._make_executor(tmp_path)
        result = executor._read_replied_to_file()
        assert result == {"alice", "charlie"}

    def test_read_empty_file_returns_empty_set(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir(parents=True)
        path = run_dir / "replied_to.jsonl"
        path.write_text("", encoding="utf-8")
        executor = self._make_executor(tmp_path)
        assert executor._read_replied_to_file() == set()

    def test_read_empty_lines_ignored(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir(parents=True)
        path = run_dir / "replied_to.jsonl"
        path.write_text(
            '\n\n{"to": "alice", "success": true}\n\n',
            encoding="utf-8",
        )
        executor = self._make_executor(tmp_path)
        assert executor._read_replied_to_file() == {"alice"}


class TestSendPatternsRemoved:
    """Verify _SEND_PATTERNS and _parse_replied_to are removed from agent_sdk."""

    def test_send_patterns_not_in_module(self):
        import core.execution.agent_sdk as mod
        assert not hasattr(mod, "_SEND_PATTERNS")

    def test_parse_replied_to_not_in_executor(self):
        from core.execution.agent_sdk import AgentSDKExecutor
        assert not hasattr(AgentSDKExecutor, "_parse_replied_to")
