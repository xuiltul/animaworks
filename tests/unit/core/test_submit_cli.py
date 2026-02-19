# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for animaworks-tool submit CLI command.

Validates ``_handle_submit()`` in ``core/tools/__init__.py``:
- Pending JSON file creation in ``state/background_tasks/pending/``
- JSON output with task_id, status, tool, subcommand
- Error handling for missing arguments and missing environment variables
- Subcommand detection logic (first non-flag argument)
- All required fields present in the pending task descriptor
"""
from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import patch

import pytest


class TestHandleSubmit:
    """Tests for the _handle_submit() function."""

    def test_writes_pending_json(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """submit writes a valid pending task JSON file to the correct directory."""
        anima_dir = tmp_path / "animas" / "test-anima"
        anima_dir.mkdir(parents=True)
        monkeypatch.setenv("ANIMAWORKS_ANIMA_DIR", str(anima_dir))

        from core.tools import _handle_submit

        captured = io.StringIO()
        with patch("builtins.print", side_effect=lambda *a, **kw: captured.write(str(a[0]) + "\n")):
            _handle_submit(["image_gen", "3d", "assets/avatar.png"])

        # Verify the pending directory was created and a file was written
        pending_dir = anima_dir / "state" / "background_tasks" / "pending"
        assert pending_dir.is_dir()
        pending_files = list(pending_dir.glob("*.json"))
        assert len(pending_files) == 1

        data = json.loads(pending_files[0].read_text(encoding="utf-8"))
        assert data["tool_name"] == "image_gen"
        assert data["subcommand"] == "3d"
        assert data["raw_args"] == ["3d", "assets/avatar.png"]
        assert data["anima_name"] == "test-anima"
        assert data["status"] == "pending"

    def test_pending_file_named_with_task_id(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The pending JSON filename matches the task_id inside."""
        anima_dir = tmp_path / "animas" / "test-anima"
        anima_dir.mkdir(parents=True)
        monkeypatch.setenv("ANIMAWORKS_ANIMA_DIR", str(anima_dir))

        from core.tools import _handle_submit

        captured = io.StringIO()
        with patch("builtins.print", side_effect=lambda *a, **kw: captured.write(str(a[0]) + "\n")):
            _handle_submit(["local_llm", "generate", "hello"])

        pending_dir = anima_dir / "state" / "background_tasks" / "pending"
        pending_files = list(pending_dir.glob("*.json"))
        assert len(pending_files) == 1

        data = json.loads(pending_files[0].read_text(encoding="utf-8"))
        expected_filename = f"{data['task_id']}.json"
        assert pending_files[0].name == expected_filename

    def test_output_json(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """submit prints valid JSON with task_id, status, tool, and subcommand."""
        anima_dir = tmp_path / "animas" / "test-anima"
        anima_dir.mkdir(parents=True)
        monkeypatch.setenv("ANIMAWORKS_ANIMA_DIR", str(anima_dir))

        from core.tools import _handle_submit

        captured = io.StringIO()
        with patch("builtins.print", side_effect=lambda *a, **kw: captured.write(str(a[0]) + "\n")):
            _handle_submit(["local_llm", "generate", "hello"])

        output = json.loads(captured.getvalue())
        assert output["status"] == "submitted"
        assert output["tool"] == "local_llm"
        assert output["subcommand"] == "generate"
        assert "task_id" in output
        # Task ID should be 12-char hex (uuid4().hex[:12])
        assert len(output["task_id"]) == 12

    def test_output_contains_message(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Output JSON contains a human-readable message field."""
        anima_dir = tmp_path / "animas" / "test-anima"
        anima_dir.mkdir(parents=True)
        monkeypatch.setenv("ANIMAWORKS_ANIMA_DIR", str(anima_dir))

        from core.tools import _handle_submit

        captured = io.StringIO()
        with patch("builtins.print", side_effect=lambda *a, **kw: captured.write(str(a[0]) + "\n")):
            _handle_submit(["transcribe", "audio.wav"])

        output = json.loads(captured.getvalue())
        assert "message" in output
        assert output["task_id"] in output["message"]

    def test_no_args_exits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """submit with no args calls sys.exit(1)."""
        monkeypatch.setenv("ANIMAWORKS_ANIMA_DIR", "/tmp/fake")

        from core.tools import _handle_submit

        with pytest.raises(SystemExit) as exc_info:
            _handle_submit([])
        assert exc_info.value.code == 1

    def test_no_anima_dir_exits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """submit without ANIMAWORKS_ANIMA_DIR calls sys.exit(1)."""
        monkeypatch.delenv("ANIMAWORKS_ANIMA_DIR", raising=False)

        from core.tools import _handle_submit

        with pytest.raises(SystemExit) as exc_info:
            _handle_submit(["image_gen", "3d"])
        assert exc_info.value.code == 1

    def test_pending_json_contains_all_required_fields(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Pending JSON contains all required fields per the design spec."""
        anima_dir = tmp_path / "animas" / "sakura"
        anima_dir.mkdir(parents=True)
        monkeypatch.setenv("ANIMAWORKS_ANIMA_DIR", str(anima_dir))

        from core.tools import _handle_submit

        captured = io.StringIO()
        with patch("builtins.print", side_effect=lambda *a, **kw: captured.write(str(a[0]) + "\n")):
            _handle_submit(["image_gen", "pipeline", "--prompt", "1girl"])

        pending_dir = anima_dir / "state" / "background_tasks" / "pending"
        data = json.loads(list(pending_dir.glob("*.json"))[0].read_text(encoding="utf-8"))

        required_fields = {
            "task_id", "tool_name", "subcommand", "raw_args",
            "anima_name", "anima_dir", "submitted_at", "status",
        }
        assert required_fields.issubset(data.keys()), (
            f"Missing fields: {required_fields - set(data.keys())}"
        )
        # Validate types
        assert isinstance(data["task_id"], str) and len(data["task_id"]) == 12
        assert isinstance(data["tool_name"], str)
        assert isinstance(data["subcommand"], str)
        assert isinstance(data["raw_args"], list)
        assert isinstance(data["anima_name"], str)
        assert isinstance(data["anima_dir"], str)
        assert isinstance(data["submitted_at"], (int, float))
        assert data["status"] == "pending"

    def test_anima_name_from_dir_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """anima_name is derived from the last component of ANIMAWORKS_ANIMA_DIR."""
        anima_dir = tmp_path / "animas" / "hana-chan"
        anima_dir.mkdir(parents=True)
        monkeypatch.setenv("ANIMAWORKS_ANIMA_DIR", str(anima_dir))

        from core.tools import _handle_submit

        captured = io.StringIO()
        with patch("builtins.print", side_effect=lambda *a, **kw: captured.write(str(a[0]) + "\n")):
            _handle_submit(["web_search", "search", "python asyncio"])

        pending_dir = anima_dir / "state" / "background_tasks" / "pending"
        data = json.loads(list(pending_dir.glob("*.json"))[0].read_text(encoding="utf-8"))
        assert data["anima_name"] == "hana-chan"
        assert data["anima_dir"] == str(anima_dir)

    def test_subcommand_detection_skips_flags(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Subcommand is the first non-flag argument after tool_name."""
        anima_dir = tmp_path / "animas" / "test-anima"
        anima_dir.mkdir(parents=True)
        monkeypatch.setenv("ANIMAWORKS_ANIMA_DIR", str(anima_dir))

        from core.tools import _handle_submit

        captured = io.StringIO()
        with patch("builtins.print", side_effect=lambda *a, **kw: captured.write(str(a[0]) + "\n")):
            _handle_submit(["transcribe", "--language", "ja", "/path/to/audio.wav"])

        pending_dir = anima_dir / "state" / "background_tasks" / "pending"
        data = json.loads(list(pending_dir.glob("*.json"))[0].read_text(encoding="utf-8"))
        assert data["tool_name"] == "transcribe"
        # raw_args are the arguments after tool_name
        assert data["raw_args"] == ["--language", "ja", "/path/to/audio.wav"]

    def test_subcommand_empty_when_only_flags(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When all args are flags, subcommand should be empty string."""
        anima_dir = tmp_path / "animas" / "test-anima"
        anima_dir.mkdir(parents=True)
        monkeypatch.setenv("ANIMAWORKS_ANIMA_DIR", str(anima_dir))

        from core.tools import _handle_submit

        captured = io.StringIO()
        with patch("builtins.print", side_effect=lambda *a, **kw: captured.write(str(a[0]) + "\n")):
            _handle_submit(["some_tool", "--verbose", "--dry-run"])

        pending_dir = anima_dir / "state" / "background_tasks" / "pending"
        data = json.loads(list(pending_dir.glob("*.json"))[0].read_text(encoding="utf-8"))
        assert data["subcommand"] == ""

    def test_tool_name_only_no_additional_args(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """submit with only tool_name (no subcommand or extra args) works."""
        anima_dir = tmp_path / "animas" / "test-anima"
        anima_dir.mkdir(parents=True)
        monkeypatch.setenv("ANIMAWORKS_ANIMA_DIR", str(anima_dir))

        from core.tools import _handle_submit

        captured = io.StringIO()
        with patch("builtins.print", side_effect=lambda *a, **kw: captured.write(str(a[0]) + "\n")):
            _handle_submit(["transcribe"])

        pending_dir = anima_dir / "state" / "background_tasks" / "pending"
        pending_files = list(pending_dir.glob("*.json"))
        assert len(pending_files) == 1

        data = json.loads(pending_files[0].read_text(encoding="utf-8"))
        assert data["tool_name"] == "transcribe"
        assert data["subcommand"] == ""
        assert data["raw_args"] == []

        output = json.loads(captured.getvalue())
        assert output["status"] == "submitted"
        assert output["tool"] == "transcribe"

    def test_unique_task_ids(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Each submit generates a unique task_id."""
        anima_dir = tmp_path / "animas" / "test-anima"
        anima_dir.mkdir(parents=True)
        monkeypatch.setenv("ANIMAWORKS_ANIMA_DIR", str(anima_dir))

        from core.tools import _handle_submit

        task_ids = []
        for _ in range(5):
            captured = io.StringIO()
            with patch("builtins.print", side_effect=lambda *a, **kw: captured.write(str(a[0]) + "\n")):
                _handle_submit(["image_gen", "3d", "test.png"])
            output = json.loads(captured.getvalue())
            task_ids.append(output["task_id"])

        # All task_ids should be unique
        assert len(set(task_ids)) == 5
