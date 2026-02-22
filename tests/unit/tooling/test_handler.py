"""Tests for core.tooling.handler — ToolHandler permission and dispatch."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.tooling.handler import (
    ToolHandler,
    _SHELL_METACHAR_RE,
    _error_result,
    _EPISODE_FILENAME_RE,
    _PROTECTED_FILES,
    _is_protected_write,
    _validate_episode_path,
)


# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def anima_dir(tmp_path: Path) -> Path:
    d = tmp_path / "animas" / "test-anima"
    d.mkdir(parents=True)
    (d / "permissions.md").write_text("", encoding="utf-8")
    return d


@pytest.fixture
def memory(anima_dir: Path) -> MagicMock:
    m = MagicMock()
    m.read_permissions.return_value = ""
    m.search_memory_text.return_value = []
    return m


@pytest.fixture
def messenger() -> MagicMock:
    m = MagicMock()
    m.anima_name = "test-anima"
    msg = MagicMock()
    msg.id = "msg_001"
    msg.thread_id = "thread_001"
    m.send.return_value = msg
    return m


@pytest.fixture
def handler(anima_dir: Path, memory: MagicMock) -> ToolHandler:
    return ToolHandler(
        anima_dir=anima_dir,
        memory=memory,
        messenger=None,
        tool_registry=[],
    )


@pytest.fixture
def handler_with_messenger(
    anima_dir: Path, memory: MagicMock, messenger: MagicMock,
) -> ToolHandler:
    return ToolHandler(
        anima_dir=anima_dir,
        memory=memory,
        messenger=messenger,
        tool_registry=[],
    )


# ── Properties ────────────────────────────────────────────────


class TestProperties:
    def test_on_message_sent_property(self, handler: ToolHandler):
        assert handler.on_message_sent is None
        fn = MagicMock()
        handler.on_message_sent = fn
        assert handler.on_message_sent is fn

    def test_replied_to(self, handler: ToolHandler):
        assert handler.replied_to == set()

    def test_reset_replied_to(self, handler_with_messenger: ToolHandler, anima_dir: Path):
        h = handler_with_messenger
        # Create "alice" directory so outbound resolver recognises it as internal
        alice_dir = anima_dir.parent / "alice"
        alice_dir.mkdir(exist_ok=True)
        with patch("core.paths.get_animas_dir", return_value=anima_dir.parent):
            h.handle("send_message", {"to": "alice", "content": "hi", "intent": "report"})
        assert "alice" in h.replied_to
        h.reset_replied_to()
        assert h.replied_to == set()


# ── Main dispatch routing ─────────────────────────────────────


class TestHandleRouting:
    def test_search_memory(self, handler: ToolHandler, memory: MagicMock):
        memory.search_memory_text.return_value = [
            ("knowledge/k1.md", "some result"),
        ]
        result = handler.handle("search_memory", {"query": "test", "scope": "all"})
        assert "knowledge/k1.md" in result
        assert "some result" in result

    def test_search_memory_no_results(self, handler: ToolHandler, memory: MagicMock):
        memory.search_memory_text.return_value = []
        result = handler.handle("search_memory", {"query": "nothing"})
        assert "No results" in result

    def test_read_memory_file(self, handler: ToolHandler, anima_dir: Path):
        (anima_dir / "knowledge").mkdir(exist_ok=True)
        (anima_dir / "knowledge" / "test.md").write_text("content here", encoding="utf-8")
        result = handler.handle("read_memory_file", {"path": "knowledge/test.md"})
        assert result == "content here"

    def test_read_memory_file_not_found(self, handler: ToolHandler):
        result = handler.handle("read_memory_file", {"path": "nonexistent.md"})
        assert "File not found" in result

    def test_read_memory_file_common_knowledge_prefix(
        self, handler: ToolHandler, tmp_path: Path,
    ):
        """read_memory_file with common_knowledge/ prefix resolves to shared dir."""
        ck_dir = tmp_path / "ck"
        ck_dir.mkdir()
        (ck_dir / "policy.md").write_text("shared content", encoding="utf-8")
        with patch(
            "core.paths.get_common_knowledge_dir",
            return_value=ck_dir,
        ):
            result = handler.handle(
                "read_memory_file", {"path": "common_knowledge/policy.md"},
            )
        assert result == "shared content"

    def test_read_memory_file_common_knowledge_not_found(
        self, handler: ToolHandler, tmp_path: Path,
    ):
        """read_memory_file with common_knowledge/ prefix for missing file."""
        ck_dir = tmp_path / "ck_empty"
        ck_dir.mkdir()
        with patch(
            "core.paths.get_common_knowledge_dir",
            return_value=ck_dir,
        ):
            result = handler.handle(
                "read_memory_file", {"path": "common_knowledge/missing.md"},
            )
        assert "File not found" in result

    def test_write_memory_file_overwrite(self, handler: ToolHandler, anima_dir: Path):
        result = handler.handle(
            "write_memory_file",
            {"path": "knowledge/new.md", "content": "new content"},
        )
        assert "Written to" in result
        assert (anima_dir / "knowledge" / "new.md").read_text(encoding="utf-8") == "new content"

    def test_write_memory_file_append(self, handler: ToolHandler, anima_dir: Path):
        (anima_dir / "knowledge").mkdir(exist_ok=True)
        (anima_dir / "knowledge" / "log.md").write_text("line1\n", encoding="utf-8")
        handler.handle(
            "write_memory_file",
            {"path": "knowledge/log.md", "content": "line2\n", "mode": "append"},
        )
        content = (anima_dir / "knowledge" / "log.md").read_text(encoding="utf-8")
        assert content == "line1\nline2\n"

    def test_send_message_without_messenger(self, handler: ToolHandler):
        result = handler.handle("send_message", {"to": "alice", "content": "hi"})
        assert "Error" in result

    def test_send_message_with_messenger(
        self, handler_with_messenger: ToolHandler, anima_dir: Path,
    ):
        alice_dir = anima_dir.parent / "alice"
        alice_dir.mkdir(exist_ok=True)
        with patch("core.paths.get_animas_dir", return_value=anima_dir.parent):
            result = handler_with_messenger.handle(
                "send_message", {"to": "alice", "content": "hello", "intent": "report"},
            )
        assert "Message sent to alice" in result
        assert "alice" in handler_with_messenger.replied_to

    def test_send_message_with_intent(
        self, handler_with_messenger: ToolHandler, anima_dir: Path,
    ):
        alice_dir = anima_dir.parent / "alice"
        alice_dir.mkdir(exist_ok=True)
        with patch("core.paths.get_animas_dir", return_value=anima_dir.parent):
            result = handler_with_messenger.handle(
                "send_message", {"to": "alice", "content": "hello", "intent": "delegation"},
            )
        assert "Message sent to alice" in result
        handler_with_messenger._messenger.send.assert_called_once_with(
            to="alice", content="hello", thread_id="", reply_to="", intent="delegation",
        )

    def test_send_message_intent_empty_returns_error(
        self, handler_with_messenger: ToolHandler, anima_dir: Path,
    ):
        """Empty intent (no intent provided) is rejected by the new DM restriction."""
        alice_dir = anima_dir.parent / "alice"
        alice_dir.mkdir(exist_ok=True)
        with patch("core.paths.get_animas_dir", return_value=anima_dir.parent):
            result = handler_with_messenger.handle(
                "send_message", {"to": "alice", "content": "hello"},
            )
        assert "Error" in result
        assert "intent" in result
        handler_with_messenger._messenger.send.assert_not_called()

    def test_send_message_invalid_intent_returns_error(
        self, handler_with_messenger: ToolHandler, anima_dir: Path,
    ):
        """Intent that is not 'report' or 'delegation' is rejected."""
        alice_dir = anima_dir.parent / "alice"
        alice_dir.mkdir(exist_ok=True)
        long_intent = "x" * 100
        with patch("core.paths.get_animas_dir", return_value=anima_dir.parent):
            result = handler_with_messenger.handle(
                "send_message", {"to": "alice", "content": "hello", "intent": long_intent},
            )
        assert "Error" in result
        assert "intent" in result
        handler_with_messenger._messenger.send.assert_not_called()

    def test_send_message_calls_on_message_sent(
        self, handler_with_messenger: ToolHandler, anima_dir: Path,
    ):
        callback = MagicMock()
        handler_with_messenger.on_message_sent = callback
        alice_dir = anima_dir.parent / "alice"
        alice_dir.mkdir(exist_ok=True)
        with patch("core.paths.get_animas_dir", return_value=anima_dir.parent):
            handler_with_messenger.handle(
                "send_message", {"to": "alice", "content": "hello", "intent": "report"},
            )
        callback.assert_called_once_with("test-anima", "alice", "hello")

    def test_send_message_on_message_sent_error_swallowed(
        self, handler_with_messenger: ToolHandler, anima_dir: Path,
    ):
        callback = MagicMock(side_effect=RuntimeError("boom"))
        handler_with_messenger.on_message_sent = callback
        alice_dir = anima_dir.parent / "alice"
        alice_dir.mkdir(exist_ok=True)
        with patch("core.paths.get_animas_dir", return_value=anima_dir.parent):
            # Should not raise
            result = handler_with_messenger.handle(
                "send_message", {"to": "alice", "content": "hello", "intent": "report"},
            )
        assert "Message sent" in result

    def test_send_message_duplicate_recipient_returns_error(
        self, handler_with_messenger: ToolHandler, anima_dir: Path,
    ):
        """Sending a second message to the same recipient returns an error."""
        alice_dir = anima_dir.parent / "alice"
        alice_dir.mkdir(exist_ok=True)
        with patch("core.paths.get_animas_dir", return_value=anima_dir.parent):
            result1 = handler_with_messenger.handle(
                "send_message", {"to": "alice", "content": "first", "intent": "report"},
            )
            assert "Message sent to alice" in result1
            result2 = handler_with_messenger.handle(
                "send_message", {"to": "alice", "content": "second", "intent": "report"},
            )
        assert "Error" in result2
        assert "alice" in result2

    def test_send_message_max_recipients_returns_error(
        self, handler_with_messenger: ToolHandler, anima_dir: Path,
    ):
        """After sending to 2 recipients, a 3rd recipient is rejected."""
        alice_dir = anima_dir.parent / "alice"
        alice_dir.mkdir(exist_ok=True)
        bob_dir = anima_dir.parent / "bob"
        bob_dir.mkdir(exist_ok=True)
        charlie_dir = anima_dir.parent / "charlie"
        charlie_dir.mkdir(exist_ok=True)
        with patch("core.paths.get_animas_dir", return_value=anima_dir.parent):
            result1 = handler_with_messenger.handle(
                "send_message", {"to": "alice", "content": "hi", "intent": "report"},
            )
            assert "Message sent to alice" in result1
            result2 = handler_with_messenger.handle(
                "send_message", {"to": "bob", "content": "hi", "intent": "report"},
            )
            assert "Message sent to bob" in result2
            result3 = handler_with_messenger.handle(
                "send_message", {"to": "charlie", "content": "hi", "intent": "report"},
            )
        assert "Error" in result3
        assert "2" in result3

    def test_send_message_two_recipients_allowed(
        self, handler_with_messenger: ToolHandler, anima_dir: Path,
    ):
        """Sending to two different recipients should both succeed."""
        alice_dir = anima_dir.parent / "alice"
        alice_dir.mkdir(exist_ok=True)
        bob_dir = anima_dir.parent / "bob"
        bob_dir.mkdir(exist_ok=True)
        with patch("core.paths.get_animas_dir", return_value=anima_dir.parent):
            result1 = handler_with_messenger.handle(
                "send_message", {"to": "alice", "content": "hi", "intent": "report"},
            )
            result2 = handler_with_messenger.handle(
                "send_message", {"to": "bob", "content": "hi", "intent": "report"},
            )
        assert "Message sent to alice" in result1
        assert "Message sent to bob" in result2

    def test_unknown_tool(self, handler: ToolHandler):
        result = handler.handle("totally_unknown_tool", {})
        assert "Unknown tool" in result

    def test_external_dispatch(self, handler: ToolHandler):
        handler._external = MagicMock()
        handler._external.dispatch.return_value = "external result"
        result = handler.handle("some_external_tool", {"arg": "val"})
        assert result == "external result"

    def test_external_dispatch_returns_none_falls_to_unknown(self, handler: ToolHandler):
        handler._external = MagicMock()
        handler._external.dispatch.return_value = None
        result = handler.handle("some_external_tool", {"arg": "val"})
        assert "Unknown tool" in result


# ── File operation handlers ───────────────────────────────────


class TestFileOperations:
    def test_read_file_in_anima_dir(self, handler: ToolHandler, anima_dir: Path):
        (anima_dir / "test.txt").write_text("hello", encoding="utf-8")
        result = handler.handle("read_file", {"path": str(anima_dir / "test.txt")})
        assert result == "hello"

    def test_read_file_not_found(self, handler: ToolHandler, anima_dir: Path):
        result = handler.handle("read_file", {"path": str(anima_dir / "missing.txt")})
        parsed = json.loads(result)
        assert parsed["error_type"] == "FileNotFound"

    def test_read_file_not_a_file(self, handler: ToolHandler, anima_dir: Path):
        result = handler.handle("read_file", {"path": str(anima_dir)})
        parsed = json.loads(result)
        assert parsed["error_type"] == "InvalidArguments"

    def test_read_file_truncated_at_100k(self, handler: ToolHandler, anima_dir: Path):
        big_content = "x" * 200_000
        (anima_dir / "big.txt").write_text(big_content, encoding="utf-8")
        result = handler.handle("read_file", {"path": str(anima_dir / "big.txt")})
        assert len(result) == 100_000

    def test_read_file_permission_denied(self, handler: ToolHandler):
        result = handler.handle("read_file", {"path": "/etc/passwd"})
        parsed = json.loads(result)
        assert parsed["error_type"] == "PermissionDenied"

    def test_write_file_in_anima_dir(self, handler: ToolHandler, anima_dir: Path):
        path = anima_dir / "output.txt"
        result = handler.handle("write_file", {"path": str(path), "content": "data"})
        assert "Written to" in result
        assert path.read_text(encoding="utf-8") == "data"

    def test_write_file_permission_denied(self, handler: ToolHandler):
        result = handler.handle("write_file", {"path": "/etc/no", "content": "data"})
        parsed = json.loads(result)
        assert parsed["error_type"] == "PermissionDenied"

    def test_edit_file_success(self, handler: ToolHandler, anima_dir: Path):
        path = anima_dir / "code.py"
        path.write_text("def foo():\n    pass\n", encoding="utf-8")
        result = handler.handle(
            "edit_file",
            {"path": str(path), "old_string": "pass", "new_string": "return 42"},
        )
        assert "Edited" in result
        assert "return 42" in path.read_text(encoding="utf-8")

    def test_edit_file_old_string_not_found(self, handler: ToolHandler, anima_dir: Path):
        path = anima_dir / "code.py"
        path.write_text("def foo():\n    pass\n", encoding="utf-8")
        result = handler.handle(
            "edit_file",
            {"path": str(path), "old_string": "NOTEXIST", "new_string": "new"},
        )
        parsed = json.loads(result)
        assert parsed["error_type"] == "StringNotFound"

    def test_edit_file_ambiguous_match(self, handler: ToolHandler, anima_dir: Path):
        path = anima_dir / "code.py"
        path.write_text("pass\npass\n", encoding="utf-8")
        result = handler.handle(
            "edit_file",
            {"path": str(path), "old_string": "pass", "new_string": "new"},
        )
        parsed = json.loads(result)
        assert parsed["error_type"] == "AmbiguousMatch"
        assert parsed["context"]["match_count"] == 2

    def test_edit_file_not_found(self, handler: ToolHandler, anima_dir: Path):
        result = handler.handle(
            "edit_file",
            {"path": str(anima_dir / "missing.py"), "old_string": "x", "new_string": "y"},
        )
        parsed = json.loads(result)
        assert parsed["error_type"] == "FileNotFound"


# ── Command execution ─────────────────────────────────────────


class TestExecuteCommand:
    def test_command_denied_without_permission(self, handler: ToolHandler):
        result = handler.handle("execute_command", {"command": "ls"})
        parsed = json.loads(result)
        assert parsed["error_type"] == "PermissionDenied"

    def test_empty_command_denied(self, handler: ToolHandler):
        result = handler.handle("execute_command", {"command": ""})
        parsed = json.loads(result)
        assert parsed["error_type"] == "PermissionDenied"

    def test_shell_metachar_rejected(self, handler: ToolHandler, memory: MagicMock):
        memory.read_permissions.return_value = "## コマンド実行\n- ls: OK"
        result = handler.handle("execute_command", {"command": "ls; rm -rf /"})
        parsed = json.loads(result)
        assert parsed["error_type"] == "PermissionDenied"
        assert "metacharacters" in parsed["message"]

    def test_command_allowed(self, handler: ToolHandler, memory: MagicMock):
        memory.read_permissions.return_value = "## コマンド実行\n- echo: OK"
        result = handler.handle("execute_command", {"command": "echo hello"})
        assert "hello" in result

    def test_command_not_in_allowed_list(self, handler: ToolHandler, memory: MagicMock):
        memory.read_permissions.return_value = "## コマンド実行\n- git: OK"
        result = handler.handle("execute_command", {"command": "rm -rf /"})
        parsed = json.loads(result)
        assert parsed["error_type"] == "PermissionDenied"

    def test_no_explicit_command_list_allows_all(self, handler: ToolHandler, memory: MagicMock):
        # Section exists but no command entries
        memory.read_permissions.return_value = "## コマンド実行\nany command is fine"
        result = handler.handle("execute_command", {"command": "echo hi"})
        assert "hi" in result

    def test_command_timeout(self, handler: ToolHandler, memory: MagicMock):
        memory.read_permissions.return_value = "## コマンド実行\n- sleep: OK"
        result = handler.handle(
            "execute_command", {"command": "sleep 999", "timeout": 1},
        )
        parsed = json.loads(result)
        assert parsed["error_type"] == "Timeout"


# ── File permission checks ────────────────────────────────────


class TestFilePermissions:
    def test_own_anima_dir_always_allowed(self, handler: ToolHandler, anima_dir: Path):
        result = handler._check_file_permission(str(anima_dir / "any_file.md"))
        assert result is None

    def test_denied_without_file_section(self, handler: ToolHandler, memory: MagicMock):
        memory.read_permissions.return_value = "# Some other section"
        result = handler._check_file_permission("/tmp/outside.txt")
        parsed = json.loads(result)
        assert parsed["error_type"] == "PermissionDenied"

    def test_denied_empty_allowed_dirs(self, handler: ToolHandler, memory: MagicMock):
        memory.read_permissions.return_value = "## ファイル操作\nno paths listed"
        result = handler._check_file_permission("/tmp/outside.txt")
        parsed = json.loads(result)
        assert parsed["error_type"] == "PermissionDenied"

    def test_allowed_path_in_whitelist(
        self, handler: ToolHandler, memory: MagicMock, tmp_path: Path,
    ):
        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()
        memory.read_permissions.return_value = f"## ファイル操作\n- {allowed_dir}: OK"
        result = handler._check_file_permission(str(allowed_dir / "file.txt"))
        assert result is None

    def test_denied_path_not_in_whitelist(self, handler: ToolHandler, memory: MagicMock):
        memory.read_permissions.return_value = "## ファイル操作\n- /opt/safe: OK"
        result = handler._check_file_permission("/tmp/not_safe/file.txt")
        parsed = json.loads(result)
        assert parsed["error_type"] == "PermissionDenied"
        assert "not under any allowed" in parsed["message"]

    def test_file_section_ends_at_next_header(self, handler: ToolHandler, memory: MagicMock):
        memory.read_permissions.return_value = (
            "## ファイル操作\n- /opt/safe: OK\n## コマンド実行\n- /opt/also: not a path"
        )
        result = handler._check_file_permission("/opt/also/file.txt")
        parsed = json.loads(result)
        assert parsed["error_type"] == "PermissionDenied"


# ── Command permission checks ─────────────────────────────────


class TestCommandPermissions:
    def test_empty_command(self, handler: ToolHandler):
        result = handler._check_command_permission("")
        parsed = json.loads(result)
        assert parsed["error_type"] == "PermissionDenied"
        assert "Empty" in parsed["message"]

    def test_whitespace_only(self, handler: ToolHandler):
        result = handler._check_command_permission("   ")
        parsed = json.loads(result)
        assert parsed["error_type"] == "PermissionDenied"
        assert "Empty" in parsed["message"]

    def test_metachar_semicolon(self, handler: ToolHandler):
        result = handler._check_command_permission("ls; echo hi")
        parsed = json.loads(result)
        assert parsed["error_type"] == "PermissionDenied"
        assert "metacharacters" in parsed["message"]

    def test_metachar_pipe(self, handler: ToolHandler):
        result = handler._check_command_permission("ls | grep foo")
        parsed = json.loads(result)
        assert parsed["error_type"] == "PermissionDenied"
        assert "metacharacters" in parsed["message"]

    def test_metachar_backtick(self, handler: ToolHandler):
        result = handler._check_command_permission("echo `whoami`")
        parsed = json.loads(result)
        assert parsed["error_type"] == "PermissionDenied"
        assert "metacharacters" in parsed["message"]

    def test_metachar_dollar(self, handler: ToolHandler):
        result = handler._check_command_permission("echo $HOME")
        parsed = json.loads(result)
        assert parsed["error_type"] == "PermissionDenied"
        assert "metacharacters" in parsed["message"]

    def test_no_command_section(self, handler: ToolHandler, memory: MagicMock):
        memory.read_permissions.return_value = "nothing relevant"
        result = handler._check_command_permission("git status")
        parsed = json.loads(result)
        assert parsed["error_type"] == "PermissionDenied"
        assert "not enabled" in parsed["message"]

    def test_invalid_syntax(self, handler: ToolHandler, memory: MagicMock):
        memory.read_permissions.return_value = "## コマンド実行\n- git: OK"
        result = handler._check_command_permission("git 'unclosed")
        parsed = json.loads(result)
        assert parsed["error_type"] == "PermissionDenied"
        assert "Invalid command syntax" in parsed["message"]


# ── Shell metachar regex ──────────────────────────────────────


class TestShellMetacharRe:
    @pytest.mark.parametrize("char", [";", "&", "|", "`", "$", "(", ")", "{", "}"])
    def test_detects_metachar(self, char: str):
        assert _SHELL_METACHAR_RE.search(f"cmd {char} other")

    def test_safe_command(self):
        assert _SHELL_METACHAR_RE.search("git status --short") is None

    def test_safe_command_with_quotes(self):
        assert _SHELL_METACHAR_RE.search("echo 'hello world'") is None


# ── _error_result ────────────────────────────────────────────


class TestErrorResult:
    def test_basic_error(self):
        result = _error_result("TestError", "Something went wrong")
        parsed = json.loads(result)
        assert parsed["status"] == "error"
        assert parsed["error_type"] == "TestError"
        assert parsed["message"] == "Something went wrong"
        assert "context" not in parsed
        assert "suggestion" not in parsed

    def test_with_suggestion(self):
        result = _error_result("FileNotFound", "not found", suggestion="Use list_directory")
        parsed = json.loads(result)
        assert parsed["suggestion"] == "Use list_directory"

    def test_with_context(self):
        result = _error_result("AmbiguousMatch", "matches 3", context={"match_count": 3})
        parsed = json.loads(result)
        assert parsed["context"]["match_count"] == 3

    def test_with_all_fields(self):
        result = _error_result(
            "PermissionDenied", "denied",
            context={"allowed_dirs": ["/tmp"]},
            suggestion="Check permissions",
        )
        parsed = json.loads(result)
        assert parsed["status"] == "error"
        assert parsed["context"]["allowed_dirs"] == ["/tmp"]
        assert parsed["suggestion"] == "Check permissions"


# ── search_code handler ──────────────────────────────────────


class TestSearchCode:
    def test_search_code_basic(self, handler: ToolHandler, anima_dir: Path):
        (anima_dir / "test.py").write_text("def hello():\n    return 42\n", encoding="utf-8")
        result = handler.handle("search_code", {"pattern": "hello"})
        assert "test.py:1" in result
        assert "def hello" in result

    def test_search_code_no_matches(self, handler: ToolHandler, anima_dir: Path):
        (anima_dir / "test.py").write_text("def foo():\n    pass\n", encoding="utf-8")
        result = handler.handle("search_code", {"pattern": "nonexistent"})
        assert "No matches" in result

    def test_search_code_with_glob(self, handler: ToolHandler, anima_dir: Path):
        (anima_dir / "test.py").write_text("hello\n", encoding="utf-8")
        (anima_dir / "test.md").write_text("hello\n", encoding="utf-8")
        result = handler.handle("search_code", {"pattern": "hello", "glob": "*.py"})
        assert "test.py" in result
        # md file should not be included
        assert "test.md" not in result

    def test_search_code_invalid_regex(self, handler: ToolHandler):
        result = handler.handle("search_code", {"pattern": "[invalid"})
        parsed = json.loads(result)
        assert parsed["error_type"] == "InvalidArguments"

    def test_search_code_empty_pattern(self, handler: ToolHandler):
        result = handler.handle("search_code", {"pattern": ""})
        parsed = json.loads(result)
        assert parsed["error_type"] == "InvalidArguments"

    def test_search_code_permission_denied(self, handler: ToolHandler):
        result = handler.handle("search_code", {"pattern": "test", "path": "/etc"})
        assert "error" in result.lower() or "permission" in result.lower()


# ── list_directory handler ───────────────────────────────────


class TestListDirectory:
    def test_list_directory_basic(self, handler: ToolHandler, anima_dir: Path):
        (anima_dir / "file1.txt").write_text("a", encoding="utf-8")
        (anima_dir / "file2.txt").write_text("b", encoding="utf-8")
        (anima_dir / "subdir").mkdir()
        result = handler.handle("list_directory", {})
        assert "file1.txt" in result
        assert "file2.txt" in result
        assert "subdir/" in result

    def test_list_directory_with_pattern(self, handler: ToolHandler, anima_dir: Path):
        (anima_dir / "test.py").write_text("", encoding="utf-8")
        (anima_dir / "test.md").write_text("", encoding="utf-8")
        result = handler.handle("list_directory", {"pattern": "*.py"})
        assert "test.py" in result
        assert "test.md" not in result

    def test_list_directory_not_found(self, handler: ToolHandler, anima_dir: Path):
        result = handler.handle("list_directory", {"path": str(anima_dir / "nonexistent")})
        parsed = json.loads(result)
        assert parsed["error_type"] == "FileNotFound"

    def test_list_directory_not_a_dir(self, handler: ToolHandler, anima_dir: Path):
        (anima_dir / "file.txt").write_text("x", encoding="utf-8")
        result = handler.handle("list_directory", {"path": str(anima_dir / "file.txt")})
        parsed = json.loads(result)
        assert parsed["error_type"] == "InvalidArguments"

    def test_list_directory_empty(self, handler: ToolHandler, anima_dir: Path):
        empty_dir = anima_dir / "empty"
        empty_dir.mkdir()
        result = handler.handle("list_directory", {"path": str(empty_dir)})
        assert "empty" in result.lower()


# ── Structured errors in existing handlers ───────────────────


class TestStructuredErrors:
    def test_read_file_not_found_structured(self, handler: ToolHandler, anima_dir: Path):
        result = handler.handle("read_file", {"path": str(anima_dir / "missing.txt")})
        parsed = json.loads(result)
        assert parsed["status"] == "error"
        assert parsed["error_type"] == "FileNotFound"

    def test_edit_file_string_not_found_structured(self, handler: ToolHandler, anima_dir: Path):
        path = anima_dir / "code.py"
        path.write_text("def foo():\n    pass\n", encoding="utf-8")
        result = handler.handle(
            "edit_file",
            {"path": str(path), "old_string": "NOTEXIST", "new_string": "new"},
        )
        parsed = json.loads(result)
        assert parsed["error_type"] == "StringNotFound"
        assert "suggestion" in parsed

    def test_edit_file_ambiguous_structured(self, handler: ToolHandler, anima_dir: Path):
        path = anima_dir / "code.py"
        path.write_text("pass\npass\n", encoding="utf-8")
        result = handler.handle(
            "edit_file",
            {"path": str(path), "old_string": "pass", "new_string": "new"},
        )
        parsed = json.loads(result)
        assert parsed["error_type"] == "AmbiguousMatch"
        assert parsed["context"]["match_count"] == 2

    def test_command_timeout_structured(self, handler: ToolHandler, memory: MagicMock):
        memory.read_permissions.return_value = "## コマンド実行\n- sleep: OK"
        result = handler.handle(
            "execute_command", {"command": "sleep 999", "timeout": 1},
        )
        parsed = json.loads(result)
        assert parsed["error_type"] == "Timeout"

    def test_permission_denied_structured(self, handler: ToolHandler):
        result = handler.handle("read_file", {"path": "/etc/passwd"})
        parsed = json.loads(result)
        assert parsed["error_type"] == "PermissionDenied"


# ── Schedule changed callback ────────────────────────────────


class TestScheduleChangedCallback:
    def test_on_schedule_changed_property(self, handler: ToolHandler):
        assert handler.on_schedule_changed is None
        fn = MagicMock()
        handler.on_schedule_changed = fn
        assert handler.on_schedule_changed is fn

    def test_write_heartbeat_triggers_callback(
        self, anima_dir: Path, memory: MagicMock,
    ):
        callback = MagicMock()
        h = ToolHandler(
            anima_dir=anima_dir,
            memory=memory,
            on_schedule_changed=callback,
        )
        h.handle("write_memory_file", {"path": "heartbeat.md", "content": "new config"})
        callback.assert_called_once_with("test-anima")

    def test_write_cron_triggers_callback(
        self, anima_dir: Path, memory: MagicMock,
    ):
        callback = MagicMock()
        h = ToolHandler(
            anima_dir=anima_dir,
            memory=memory,
            on_schedule_changed=callback,
        )
        h.handle("write_memory_file", {"path": "cron.md", "content": "new cron"})
        callback.assert_called_once_with("test-anima")

    def test_write_other_file_does_not_trigger_callback(
        self, anima_dir: Path, memory: MagicMock,
    ):
        callback = MagicMock()
        h = ToolHandler(
            anima_dir=anima_dir,
            memory=memory,
            on_schedule_changed=callback,
        )
        h.handle("write_memory_file", {"path": "knowledge/note.md", "content": "note"})
        callback.assert_not_called()

    def test_callback_error_does_not_break_write(
        self, anima_dir: Path, memory: MagicMock,
    ):
        callback = MagicMock(side_effect=RuntimeError("reload failed"))
        h = ToolHandler(
            anima_dir=anima_dir,
            memory=memory,
            on_schedule_changed=callback,
        )
        result = h.handle("write_memory_file", {"path": "heartbeat.md", "content": "cfg"})
        assert "Written to" in result
        # File should still be written despite callback error
        assert (anima_dir / "heartbeat.md").read_text(encoding="utf-8") == "cfg"

    def test_no_callback_set_does_not_error(self, handler: ToolHandler, anima_dir: Path):
        # handler has no on_schedule_changed set (default None)
        result = handler.handle("write_memory_file", {"path": "heartbeat.md", "content": "cfg"})
        assert "Written to" in result


# ── Memory write security ─────────────────────────────────────


class TestMemoryWriteSecurity:
    """Tests for protected file and path traversal checks in memory tools."""

    @pytest.mark.parametrize("protected_file", [
        "permissions.md",
        "identity.md",
        "bootstrap.md",
    ])
    def test_write_memory_file_blocked_for_protected(
        self, handler: ToolHandler, protected_file: str,
    ):
        result = handler.handle(
            "write_memory_file",
            {"path": protected_file, "content": "malicious"},
        )
        parsed = json.loads(result)
        assert parsed["error_type"] == "PermissionDenied"
        assert "protected file" in parsed["message"]

    def test_write_memory_file_allowed_for_non_protected(
        self, handler: ToolHandler, anima_dir: Path,
    ):
        result = handler.handle(
            "write_memory_file",
            {"path": "knowledge/safe.md", "content": "safe content"},
        )
        assert "Written to" in result
        assert (anima_dir / "knowledge" / "safe.md").read_text(encoding="utf-8") == "safe content"

    def test_write_memory_file_path_traversal_blocked(
        self, handler: ToolHandler, tmp_path: Path,
    ):
        # Create another anima's directory
        other = tmp_path / "animas" / "other-anima" / "knowledge"
        other.mkdir(parents=True)
        result = handler.handle(
            "write_memory_file",
            {"path": "../other-anima/knowledge/stolen.md", "content": "hacked"},
        )
        parsed = json.loads(result)
        assert parsed["error_type"] == "PermissionDenied"
        assert "outside anima directory" in parsed["message"]
        # Verify file was NOT created
        assert not (other / "stolen.md").exists()

    def test_read_memory_file_path_traversal_blocked(
        self, handler: ToolHandler, tmp_path: Path,
    ):
        # Create another anima's directory with a file
        other = tmp_path / "animas" / "other-anima"
        other.mkdir(parents=True)
        (other / "identity.md").write_text("secret identity", encoding="utf-8")
        result = handler.handle(
            "read_memory_file",
            {"path": "../other-anima/identity.md"},
        )
        parsed = json.loads(result)
        assert parsed["error_type"] == "PermissionDenied"
        assert "outside anima directory" in parsed["message"]

    def test_read_memory_file_normal_access_allowed(
        self, handler: ToolHandler, anima_dir: Path,
    ):
        (anima_dir / "episodes").mkdir(exist_ok=True)
        (anima_dir / "episodes" / "2026-02-15.md").write_text("daily log", encoding="utf-8")
        result = handler.handle(
            "read_memory_file",
            {"path": "episodes/2026-02-15.md"},
        )
        assert result == "daily log"

    def test_write_memory_file_heartbeat_still_allowed(
        self, handler: ToolHandler, anima_dir: Path,
    ):
        """heartbeat.md is NOT in the protected list."""
        result = handler.handle(
            "write_memory_file",
            {"path": "heartbeat.md", "content": "new heartbeat config"},
        )
        assert "Written to" in result

    def test_write_memory_file_cron_still_allowed(
        self, handler: ToolHandler, anima_dir: Path,
    ):
        """cron.md is NOT in the protected list."""
        result = handler.handle(
            "write_memory_file",
            {"path": "cron.md", "content": "new cron config"},
        )
        assert "Written to" in result


# ── File permission write protection ─────────────────────────


class TestFilePermissionWriteProtection:
    """Tests for _check_file_permission with write=True."""

    def test_write_to_own_permissions_md_blocked(
        self, handler: ToolHandler, anima_dir: Path,
    ):
        result = handler._check_file_permission(
            str(anima_dir / "permissions.md"), write=True,
        )
        assert result is not None
        parsed = json.loads(result)
        assert parsed["error_type"] == "PermissionDenied"
        assert "protected file" in parsed["message"]

    def test_write_to_own_identity_md_blocked(
        self, handler: ToolHandler, anima_dir: Path,
    ):
        result = handler._check_file_permission(
            str(anima_dir / "identity.md"), write=True,
        )
        assert result is not None
        parsed = json.loads(result)
        assert parsed["error_type"] == "PermissionDenied"

    def test_read_permissions_md_allowed(
        self, handler: ToolHandler, anima_dir: Path,
    ):
        """Reading protected files is allowed (write=False)."""
        result = handler._check_file_permission(
            str(anima_dir / "permissions.md"), write=False,
        )
        assert result is None

    def test_read_permissions_md_allowed_default(
        self, handler: ToolHandler, anima_dir: Path,
    ):
        """Default write=False should allow reading."""
        result = handler._check_file_permission(
            str(anima_dir / "permissions.md"),
        )
        assert result is None

    def test_write_to_knowledge_allowed(
        self, handler: ToolHandler, anima_dir: Path,
    ):
        result = handler._check_file_permission(
            str(anima_dir / "knowledge" / "note.md"), write=True,
        )
        assert result is None

    def test_write_file_handler_blocks_protected(
        self, handler: ToolHandler, anima_dir: Path,
    ):
        """write_file tool should block writes to permissions.md."""
        result = handler.handle(
            "write_file",
            {"path": str(anima_dir / "permissions.md"), "content": "hacked"},
        )
        parsed = json.loads(result)
        assert parsed["error_type"] == "PermissionDenied"

    def test_edit_file_handler_blocks_protected(
        self, handler: ToolHandler, anima_dir: Path,
    ):
        """edit_file tool should block edits to identity.md."""
        (anima_dir / "identity.md").write_text("original identity", encoding="utf-8")
        result = handler.handle(
            "edit_file",
            {
                "path": str(anima_dir / "identity.md"),
                "old_string": "original",
                "new_string": "hacked",
            },
        )
        parsed = json.loads(result)
        assert parsed["error_type"] == "PermissionDenied"
        # Verify file was NOT modified
        assert "original identity" == (anima_dir / "identity.md").read_text(encoding="utf-8")


# ── Command path traversal ───────────────────────────────────


class TestCommandPathTraversal:
    """Tests for path traversal detection in execute_command."""

    def test_command_with_path_traversal_blocked(
        self, handler: ToolHandler, memory: MagicMock,
    ):
        memory.read_permissions.return_value = "## コマンド実行\n- cp: OK"
        result = handler.handle(
            "execute_command",
            {"command": "cp ../other-anima/secrets.md ./stolen.md"},
        )
        parsed = json.loads(result)
        assert parsed["error_type"] == "PermissionDenied"
        assert "outside anima directory" in parsed["message"]

    def test_command_without_traversal_allowed(
        self, handler: ToolHandler, memory: MagicMock,
    ):
        memory.read_permissions.return_value = "## コマンド実行\n- echo: OK"
        result = handler.handle(
            "execute_command",
            {"command": "echo hello"},
        )
        assert "hello" in result

    def test_command_with_safe_dotdot_in_own_dir(
        self, handler: ToolHandler, memory: MagicMock, anima_dir: Path,
    ):
        """Path with .. that still resolves within anima_dir should be allowed."""
        (anima_dir / "subdir").mkdir(exist_ok=True)
        memory.read_permissions.return_value = "## コマンド実行\n- ls: OK"
        result = handler.handle(
            "execute_command",
            {"command": "ls subdir/.."},
        )
        # Should NOT be blocked — resolves within anima_dir
        assert "PermissionDenied" not in result or "outside anima" not in result


# ── _is_protected_write unit tests ───────────────────────────


class TestIsProtectedWrite:
    """Direct unit tests for the _is_protected_write function."""

    def test_protected_file_blocked(self, anima_dir: Path):
        result = _is_protected_write(anima_dir, anima_dir / "permissions.md")
        assert result is not None
        parsed = json.loads(result)
        assert parsed["error_type"] == "PermissionDenied"

    def test_non_protected_file_allowed(self, anima_dir: Path):
        result = _is_protected_write(anima_dir, anima_dir / "knowledge" / "note.md")
        assert result is None

    def test_path_traversal_blocked(self, anima_dir: Path):
        target = anima_dir / ".." / "other-anima" / "file.md"
        result = _is_protected_write(anima_dir, target)
        assert result is not None
        parsed = json.loads(result)
        assert parsed["error_type"] == "PermissionDenied"

    def test_within_anima_dir_allowed(self, anima_dir: Path):
        result = _is_protected_write(anima_dir, anima_dir / "episodes" / "log.md")
        assert result is None


# ── _check_tool_creation_permission ──────────────────────────


class TestToolCreationPermission:
    """Tests for _check_tool_creation_permission()."""

    def test_no_memory_returns_false(self, anima_dir: Path):
        h = ToolHandler(anima_dir=anima_dir, memory=None, tool_registry=[])
        assert h._check_tool_creation_permission("個人ツール") is False

    def test_no_tool_creation_section_returns_false(
        self, handler: ToolHandler, memory: MagicMock,
    ):
        memory.read_permissions.return_value = "## その他\n- something: yes"
        assert handler._check_tool_creation_permission("個人ツール") is False

    def test_personal_tool_yes(self, handler: ToolHandler, memory: MagicMock):
        memory.read_permissions.return_value = (
            "## ツール作成\n- 個人ツール: yes"
        )
        assert handler._check_tool_creation_permission("個人ツール") is True

    def test_personal_tool_ok(self, handler: ToolHandler, memory: MagicMock):
        memory.read_permissions.return_value = (
            "## ツール作成\n- 個人ツール: OK"
        )
        assert handler._check_tool_creation_permission("個人ツール") is True

    def test_shared_tool_yes(self, handler: ToolHandler, memory: MagicMock):
        memory.read_permissions.return_value = (
            "## ツール作成\n- 共有ツール: yes"
        )
        assert handler._check_tool_creation_permission("共有ツール") is True

    @pytest.mark.parametrize("value", ["YES", "True", "ENABLED", "true", "Yes"])
    def test_case_insensitive(
        self, handler: ToolHandler, memory: MagicMock, value: str,
    ):
        memory.read_permissions.return_value = (
            f"## ツール作成\n- 個人ツール: {value}"
        )
        assert handler._check_tool_creation_permission("個人ツール") is True

    def test_different_kind_not_matching(
        self, handler: ToolHandler, memory: MagicMock,
    ):
        memory.read_permissions.return_value = (
            "## ツール作成\n- 共有ツール: yes"
        )
        assert handler._check_tool_creation_permission("個人ツール") is False

    def test_bullet_with_asterisk(self, handler: ToolHandler, memory: MagicMock):
        memory.read_permissions.return_value = (
            "## ツール作成\n* 個人ツール: yes"
        )
        assert handler._check_tool_creation_permission("個人ツール") is True

    def test_bullet_with_dash(self, handler: ToolHandler, memory: MagicMock):
        memory.read_permissions.return_value = (
            "## ツール作成\n- 個人ツール: enabled"
        )
        assert handler._check_tool_creation_permission("個人ツール") is True


# ── write_memory_file tool creation permission ───────────────


class TestWriteMemoryFileToolCreation:
    """Tests for tool creation permission check in _handle_write_memory_file()."""

    def test_writing_tool_py_without_permission_denied(
        self, handler: ToolHandler, memory: MagicMock,
    ):
        memory.read_permissions.return_value = "## その他\n- nothing"
        result = handler.handle(
            "write_memory_file",
            {"path": "tools/my_tool.py", "content": "print('hi')"},
        )
        parsed = json.loads(result)
        assert parsed["error_type"] == "PermissionDenied"
        assert "ツール作成" in parsed["message"]

    def test_writing_tool_py_with_permission_succeeds(
        self, handler: ToolHandler, memory: MagicMock, anima_dir: Path,
    ):
        memory.read_permissions.return_value = (
            "## ツール作成\n- 個人ツール: yes"
        )
        result = handler.handle(
            "write_memory_file",
            {"path": "tools/my_tool.py", "content": "print('hi')"},
        )
        assert "Written to" in result
        assert (anima_dir / "tools" / "my_tool.py").read_text(encoding="utf-8") == "print('hi')"

    def test_writing_non_tool_file_skips_permission_check(
        self, handler: ToolHandler, memory: MagicMock, anima_dir: Path,
    ):
        """Writing knowledge/note.md should not check tool creation permission."""
        memory.read_permissions.return_value = ""  # No permissions at all
        result = handler.handle(
            "write_memory_file",
            {"path": "knowledge/note.md", "content": "just a note"},
        )
        assert "Written to" in result
        assert (anima_dir / "knowledge" / "note.md").read_text(encoding="utf-8") == "just a note"

    def test_writing_tools_readme_not_py_skips_permission(
        self, handler: ToolHandler, memory: MagicMock, anima_dir: Path,
    ):
        """Writing tools/readme.md (not .py) should not require tool creation permission."""
        memory.read_permissions.return_value = ""  # No permissions at all
        result = handler.handle(
            "write_memory_file",
            {"path": "tools/readme.md", "content": "tool docs"},
        )
        assert "Written to" in result
        assert (anima_dir / "tools" / "readme.md").read_text(encoding="utf-8") == "tool docs"


# ── refresh_tools handler ────────────────────────────────────


class TestRefreshTools:
    """Tests for _handle_refresh_tools()."""

    @patch("core.tooling.handler.ExternalToolDispatcher")
    def test_no_tools_found(
        self, _mock_cls: MagicMock, handler: ToolHandler,
    ):
        with patch(
            "core.tools.discover_personal_tools", return_value={},
        ), patch(
            "core.tools.discover_common_tools", return_value={},
        ):
            result = handler.handle("refresh_tools", {})
        assert "No personal or common tools found" in result

    @patch("core.tooling.handler.ExternalToolDispatcher")
    def test_discovered_tools_returned(
        self, _mock_cls: MagicMock, handler: ToolHandler,
    ):
        with patch(
            "core.tools.discover_personal_tools",
            return_value={"my_tool": "/path/to/my_tool.py"},
        ), patch(
            "core.tools.discover_common_tools",
            return_value={"shared_util": "/path/to/shared_util.py"},
        ):
            result = handler.handle("refresh_tools", {})
        assert "my_tool" in result
        assert "shared_util" in result
        assert "2 discovered" in result

    def test_calls_update_personal_tools(self, handler: ToolHandler):
        mock_external = MagicMock()
        handler._external = mock_external
        with patch(
            "core.tools.discover_personal_tools",
            return_value={"tool_a": "/a.py"},
        ), patch(
            "core.tools.discover_common_tools",
            return_value={"tool_b": "/b.py"},
        ):
            handler.handle("refresh_tools", {})
        mock_external.update_personal_tools.assert_called_once_with(
            {"tool_b": "/b.py", "tool_a": "/a.py"},
        )


# ── share_tool handler ───────────────────────────────────────


class TestShareTool:
    """Tests for _handle_share_tool()."""

    def test_personal_tool_not_found(
        self, handler: ToolHandler, anima_dir: Path,
    ):
        result = handler.handle("share_tool", {"tool_name": "nonexistent"})
        parsed = json.loads(result)
        assert parsed["error_type"] == "FileNotFound"
        assert "nonexistent" in parsed["message"]

    def test_permission_denied_without_shared_tool_permission(
        self, handler: ToolHandler, memory: MagicMock, anima_dir: Path,
    ):
        # Create the personal tool file so it passes the existence check
        tools_dir = anima_dir / "tools"
        tools_dir.mkdir(parents=True, exist_ok=True)
        (tools_dir / "my_tool.py").write_text("print('hi')", encoding="utf-8")

        memory.read_permissions.return_value = ""  # No permission
        result = handler.handle("share_tool", {"tool_name": "my_tool"})
        parsed = json.loads(result)
        assert parsed["error_type"] == "PermissionDenied"
        assert "共有ツール" in parsed["message"]

    def test_copies_file_when_permitted(
        self, handler: ToolHandler, memory: MagicMock, anima_dir: Path, tmp_path: Path,
    ):
        # Create the personal tool file
        tools_dir = anima_dir / "tools"
        tools_dir.mkdir(parents=True, exist_ok=True)
        (tools_dir / "my_tool.py").write_text("print('shared')", encoding="utf-8")

        memory.read_permissions.return_value = (
            "## ツール作成\n- 共有ツール: yes"
        )

        common_dir = tmp_path / "common_tools"
        with patch("core.paths.get_data_dir", return_value=tmp_path):
            result = handler.handle("share_tool", {"tool_name": "my_tool"})

        assert "Shared tool" in result
        assert (common_dir / "my_tool.py").read_text(encoding="utf-8") == "print('shared')"

    def test_error_when_common_tool_already_exists(
        self, handler: ToolHandler, memory: MagicMock, anima_dir: Path, tmp_path: Path,
    ):
        # Create the personal tool file
        tools_dir = anima_dir / "tools"
        tools_dir.mkdir(parents=True, exist_ok=True)
        (tools_dir / "my_tool.py").write_text("print('new')", encoding="utf-8")

        # Create existing common tool
        common_dir = tmp_path / "common_tools"
        common_dir.mkdir(parents=True, exist_ok=True)
        (common_dir / "my_tool.py").write_text("print('old')", encoding="utf-8")

        memory.read_permissions.return_value = (
            "## ツール作成\n- 共有ツール: yes"
        )

        with patch("core.paths.get_data_dir", return_value=tmp_path):
            result = handler.handle("share_tool", {"tool_name": "my_tool"})

        parsed = json.loads(result)
        assert parsed["error_type"] == "FileExists"
        assert "already exists" in parsed["message"]
        # Verify original was NOT overwritten
        assert (common_dir / "my_tool.py").read_text(encoding="utf-8") == "print('old')"

    def test_path_traversal_in_tool_name_blocked(
        self, handler: ToolHandler, anima_dir: Path,
    ):
        result = handler.handle("share_tool", {"tool_name": "../../identity"})
        parsed = json.loads(result)
        assert parsed["error_type"] == "InvalidArguments"
        assert "valid Python identifier" in parsed["message"]

    def test_slash_in_tool_name_blocked(
        self, handler: ToolHandler, anima_dir: Path,
    ):
        result = handler.handle("share_tool", {"tool_name": "foo/bar"})
        parsed = json.loads(result)
        assert parsed["error_type"] == "InvalidArguments"


# ── _validate_episode_path unit tests ────────────────────────


class TestValidateEpisodePath:
    """Direct unit tests for _validate_episode_path()."""

    def test_standard_pattern_no_warning(self):
        assert _validate_episode_path("episodes/2026-02-17.md") is None

    def test_suffixed_pattern_no_warning(self):
        assert _validate_episode_path("episodes/2026-02-17_heartbeat.md") is None

    def test_suffixed_long_name_no_warning(self):
        assert _validate_episode_path("episodes/2026-02-17_heartbeat_emergency_response.md") is None

    def test_non_standard_name_returns_warning(self):
        result = _validate_episode_path("episodes/random_notes.md")
        assert result is not None
        assert "WARNING" in result
        assert "random_notes.md" in result
        assert "YYYY-MM-DD.md" in result

    def test_invalid_date_digits_accepted(self):
        """Regex validates format (YYYY-MM-DD) not date validity — by design."""
        assert _validate_episode_path("episodes/2026-99-99.md") is None

    def test_no_extension_returns_warning(self):
        result = _validate_episode_path("episodes/2026-02-17")
        assert result is not None
        assert "WARNING" in result

    def test_txt_extension_returns_warning(self):
        result = _validate_episode_path("episodes/notes.txt")
        assert result is not None
        assert "WARNING" in result

    def test_non_episode_path_no_validation(self):
        assert _validate_episode_path("knowledge/topic.md") is None

    def test_state_path_no_validation(self):
        assert _validate_episode_path("state/current_task.md") is None

    def test_episode_subdirectory_no_validation(self):
        """Subdirectories under episodes/ are not validated."""
        assert _validate_episode_path("episodes/2026-02/17.md") is None


class TestEpisodeFilenameRegex:
    """Tests for _EPISODE_FILENAME_RE pattern."""

    @pytest.mark.parametrize("filename", [
        "2026-02-17.md",
        "2026-01-01.md",
        "2026-12-31_heartbeat.md",
        "2026-02-17_cron_batch.md",
        "2026-02-17_a.md",
    ])
    def test_valid_patterns(self, filename: str):
        assert _EPISODE_FILENAME_RE.match(filename)

    @pytest.mark.parametrize("filename", [
        "random_notes.md",
        "2026-02-17",
        "notes.txt",
        "20260217.md",
        "2026-2-17.md",
    ])
    def test_invalid_patterns(self, filename: str):
        assert not _EPISODE_FILENAME_RE.match(filename)


# ── write_memory_file episode path warning (integration) ─────


class TestWriteMemoryFileEpisodeWarning:
    """Tests for episode path warning in _handle_write_memory_file()."""

    def test_standard_episode_no_warning(
        self, handler: ToolHandler, anima_dir: Path,
    ):
        result = handler.handle(
            "write_memory_file",
            {"path": "episodes/2026-02-17.md", "content": "## 10:00 — テスト\n"},
        )
        assert "Written to" in result
        assert "WARNING" not in result

    def test_suffixed_episode_no_warning(
        self, handler: ToolHandler, anima_dir: Path,
    ):
        result = handler.handle(
            "write_memory_file",
            {"path": "episodes/2026-02-17_heartbeat.md", "content": "check"},
        )
        assert "Written to" in result
        assert "WARNING" not in result

    def test_non_standard_episode_returns_warning(
        self, handler: ToolHandler, anima_dir: Path,
    ):
        result = handler.handle(
            "write_memory_file",
            {"path": "episodes/random_notes.md", "content": "some notes"},
        )
        assert "Written to" in result
        assert "WARNING" in result
        assert "YYYY-MM-DD.md" in result
        # File should still be written
        assert (anima_dir / "episodes" / "random_notes.md").exists()

    def test_non_episode_path_no_warning(
        self, handler: ToolHandler, anima_dir: Path,
    ):
        result = handler.handle(
            "write_memory_file",
            {"path": "knowledge/note.md", "content": "content"},
        )
        assert "Written to" in result
        assert "WARNING" not in result

    def test_warning_does_not_block_write(
        self, handler: ToolHandler, anima_dir: Path,
    ):
        result = handler.handle(
            "write_memory_file",
            {"path": "episodes/bad_name.md", "content": "important data"},
        )
        assert "Written to" in result
        assert "WARNING" in result
        content = (anima_dir / "episodes" / "bad_name.md").read_text(encoding="utf-8")
        assert content == "important data"

    def test_warning_with_append_mode(
        self, handler: ToolHandler, anima_dir: Path,
    ):
        (anima_dir / "episodes").mkdir(exist_ok=True)
        (anima_dir / "episodes" / "my_log.md").write_text("line1\n", encoding="utf-8")
        result = handler.handle(
            "write_memory_file",
            {"path": "episodes/my_log.md", "content": "line2\n", "mode": "append"},
        )
        assert "Written to" in result
        assert "WARNING" in result
        content = (anima_dir / "episodes" / "my_log.md").read_text(encoding="utf-8")
        assert content == "line1\nline2\n"
