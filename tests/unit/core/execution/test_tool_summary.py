from __future__ import annotations

"""Tests for core.execution._tool_summary — human-readable tool arg summaries."""

from core.execution._tool_summary import make_tool_detail_chunk, summarize_tool_args


class TestSummarizeToolArgs:

    def test_bash_truncates_command(self):
        cmd = "x" * 200
        result = summarize_tool_args("Bash", {"command": cmd})
        assert len(result) == 120
        assert result == cmd[:120]

    def test_read_returns_file_path(self):
        assert summarize_tool_args("Read", {"file_path": "/foo/bar.py"}) == "/foo/bar.py"

    def test_write_returns_file_path(self):
        assert summarize_tool_args("Write", {"file_path": "/a/b.txt"}) == "/a/b.txt"

    def test_edit_returns_file_path(self):
        assert summarize_tool_args("Edit", {"file_path": "/edit.rs"}) == "/edit.rs"

    def test_grep_pattern_and_path(self):
        result = summarize_tool_args("Grep", {"pattern": "foo", "path": "/src"})
        assert result == "foo in /src"

    def test_grep_default_path(self):
        result = summarize_tool_args("Grep", {"pattern": "bar"})
        assert result == "bar in ."

    def test_glob_pattern(self):
        assert summarize_tool_args("Glob", {"pattern": "*.py"}) == "*.py"

    def test_task_description(self):
        desc = "a" * 100
        result = summarize_tool_args("Task", {"description": desc})
        assert len(result) == 80

    def test_send_message(self):
        assert summarize_tool_args("send_message", {"to": "alice"}) == "→ alice"

    def test_delegate_task(self):
        result = summarize_tool_args("delegate_task", {
            "name": "bob", "instruction": "do stuff",
        })
        assert result == "→ bob: do stuff"

    def test_web_search(self):
        assert summarize_tool_args("web_search", {"query": "test"}) == "test"

    def test_x_search(self):
        assert summarize_tool_args("x_search", {"query": "tweet"}) == "tweet"

    def test_skill(self):
        assert summarize_tool_args("skill", {"name": "greet"}) == "greet"

    def test_search_memory(self):
        assert summarize_tool_args("search_memory", {"query": "hello"}) == "hello"

    def test_save_memory(self):
        assert summarize_tool_args("save_memory", {"category": "episodes"}) == "episodes"

    def test_read_channel(self):
        assert summarize_tool_args("read_channel", {"channel": "general"}) == "#general"

    def test_post_channel(self):
        assert summarize_tool_args("post_channel", {"channel": "ops"}) == "#ops"

    def test_manage_channel(self):
        result = summarize_tool_args("manage_channel", {"action": "create", "channel": "eng"})
        assert result == "create #eng"

    def test_read_file_mode_a(self):
        assert summarize_tool_args("read_file", {"path": "/src/main.py"}) == "/src/main.py"

    def test_write_file_mode_a(self):
        assert summarize_tool_args("write_file", {"path": "/out.txt"}) == "/out.txt"

    def test_edit_file_mode_a(self):
        assert summarize_tool_args("edit_file", {"path": "/fix.py"}) == "/fix.py"

    def test_execute_command_mode_a(self):
        assert summarize_tool_args("execute_command", {"command": "ls -la"}) == "ls -la"

    def test_search_files_mode_a(self):
        result = summarize_tool_args("search_files", {"pattern": "TODO", "path": "/src"})
        assert result == "TODO in /src"

    def test_glob_files_mode_a(self):
        assert summarize_tool_args("glob_files", {"glob_pattern": "**/*.py"}) == "**/*.py"

    def test_unknown_tool_returns_empty(self):
        assert summarize_tool_args("SomeUnknownTool", {"arg": "value"}) == ""

    def test_empty_args(self):
        assert summarize_tool_args("Read", {}) == ""

    def test_bash_empty_command(self):
        assert summarize_tool_args("Bash", {}) == ""


class TestMakeToolDetailChunk:

    def test_returns_chunk_for_known_tool(self):
        chunk = make_tool_detail_chunk("Read", "tool_1", {"file_path": "/x.py"})
        assert chunk is not None
        assert chunk["type"] == "tool_detail"
        assert chunk["tool_id"] == "tool_1"
        assert chunk["tool_name"] == "Read"
        assert chunk["detail"] == "/x.py"

    def test_returns_none_for_unknown_tool(self):
        chunk = make_tool_detail_chunk("UnknownTool", "tool_2", {"a": "b"})
        assert chunk is None

    def test_returns_none_for_empty_summary(self):
        chunk = make_tool_detail_chunk("Read", "tool_3", {})
        assert chunk is None
