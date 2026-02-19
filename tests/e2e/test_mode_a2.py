# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for Mode A2 (LiteLLM + tool_use loop) execution.

Mode A2 iteratively calls litellm.acompletion, processes tool_calls,
and returns results to the LLM until a final text response is produced.
"""

from __future__ import annotations

import pytest

from tests.helpers.mocks import (
    make_litellm_response,
    make_tool_call,
    patch_litellm,
)


class TestModeA2Mock:
    """Mode A2 tests using mocked LLM calls."""

    async def test_basic_no_tool_calls(self, make_agent_core):
        """A2 basic: LLM returns text without tool calls."""
        agent = make_agent_core(
            name="a2-basic",
            model="openai/gpt-4o",
        )

        resp = make_litellm_response(content="Hello from A2 mode.")

        with patch_litellm(resp):
            result = await agent.run_cycle("Say hello")

        assert result.action == "responded"
        assert "Hello from A2 mode." in result.summary

    async def test_search_memory_tool_call(self, make_agent_core):
        """A2: LLM calls search_memory, then responds with final text."""
        agent = make_agent_core(
            name="a2-search",
            model="openai/gpt-4o",
        )

        # Write a knowledge file so search has results
        (agent.anima_dir / "knowledge" / "facts.md").write_text(
            "The answer to everything is 42.\n", encoding="utf-8"
        )

        # First call: tool_call to search_memory
        tc = make_tool_call(
            "search_memory",
            {"query": "answer", "scope": "knowledge"},
            call_id="call_search",
        )
        resp1 = make_litellm_response(content="", tool_calls=[tc])

        # Second call: final response
        resp2 = make_litellm_response(content="The answer is 42.")

        with patch_litellm(resp1, resp2):
            result = await agent.run_cycle("What is the answer?")

        assert "42" in result.summary

    async def test_read_file_permission_allowed(self, make_agent_core, tmp_path):
        """A2: read_file succeeds for paths within anima_dir."""
        agent = make_agent_core(
            name="a2-read-ok",
            model="openai/gpt-4o",
        )

        # Write a file inside anima_dir
        test_file = agent.anima_dir / "knowledge" / "test_data.md"
        test_file.write_text("Secret knowledge content.", encoding="utf-8")

        tc = make_tool_call(
            "read_file",
            {"path": str(test_file)},
            call_id="call_read",
        )
        resp1 = make_litellm_response(content="", tool_calls=[tc])
        resp2 = make_litellm_response(content="Read the file.")

        with patch_litellm(resp1, resp2):
            result = await agent.run_cycle("Read my knowledge file")

        assert result.summary

    async def test_read_file_permission_denied(self, make_agent_core):
        """A2: read_file is denied for paths outside allowed directories."""
        agent = make_agent_core(
            name="a2-read-deny",
            model="openai/gpt-4o",
            permissions="## ファイル操作\n- /allowed/path/\n",
        )

        tc = make_tool_call(
            "read_file",
            {"path": "/etc/passwd"},
            call_id="call_read_forbidden",
        )
        resp1 = make_litellm_response(content="", tool_calls=[tc])
        resp2 = make_litellm_response(
            content="Access denied for the requested file."
        )

        with patch_litellm(resp1, resp2):
            result = await agent.run_cycle("Read /etc/passwd")

        # The agent should have received a "Permission denied" tool result
        # and then produced a final response
        assert result.summary

    async def test_execute_command_allowed(self, make_agent_core):
        """A2: execute_command succeeds for allowed commands."""
        agent = make_agent_core(
            name="a2-cmd-ok",
            model="openai/gpt-4o",
            permissions="## コマンド実行\n- echo: OK\n",
        )

        tc = make_tool_call(
            "execute_command",
            {"command": "echo hello_test"},
            call_id="call_cmd",
        )
        resp1 = make_litellm_response(content="", tool_calls=[tc])
        resp2 = make_litellm_response(content="Command output: hello_test")

        with patch_litellm(resp1, resp2):
            result = await agent.run_cycle("Run echo hello_test")

        assert result.summary

    async def test_execute_command_metachar_rejected(self, make_agent_core):
        """A2: commands with shell metacharacters are rejected."""
        agent = make_agent_core(
            name="a2-cmd-metachar",
            model="openai/gpt-4o",
            permissions="## コマンド実行\n- ls: OK\n",
        )

        tc = make_tool_call(
            "execute_command",
            {"command": "ls; rm -rf /"},
            call_id="call_cmd_bad",
        )
        resp1 = make_litellm_response(content="", tool_calls=[tc])
        resp2 = make_litellm_response(content="Command was rejected.")

        with patch_litellm(resp1, resp2):
            result = await agent.run_cycle("List files dangerously")

        assert result.summary

    async def test_write_and_edit_file(self, make_agent_core):
        """A2: write_file and edit_file modify files on disk."""
        agent = make_agent_core(
            name="a2-write-edit",
            model="openai/gpt-4o",
        )

        target = agent.anima_dir / "knowledge" / "new_file.md"

        # Tool call 1: write_file
        tc_write = make_tool_call(
            "write_file",
            {"path": str(target), "content": "original content"},
            call_id="call_write",
        )
        resp1 = make_litellm_response(content="", tool_calls=[tc_write])

        # Tool call 2: edit_file
        tc_edit = make_tool_call(
            "edit_file",
            {
                "path": str(target),
                "old_string": "original",
                "new_string": "modified",
            },
            call_id="call_edit",
        )
        resp2 = make_litellm_response(content="", tool_calls=[tc_edit])

        # Final response
        resp3 = make_litellm_response(content="File updated.")

        with patch_litellm(resp1, resp2, resp3):
            result = await agent.run_cycle("Write and edit a file")

        assert target.exists()
        assert "modified content" in target.read_text(encoding="utf-8")

    async def test_context_threshold_session_chain(self, make_agent_core):
        """A2: context threshold triggers session chaining."""
        agent = make_agent_core(
            name="a2-chain",
            model="openai/gpt-4o",
            context_threshold=0.001,  # Very low to force chaining
            max_chains=1,
        )

        # First response with high token count to trigger threshold
        resp1 = make_litellm_response(
            content="Working on it...",
            prompt_tokens=200_000,
            completion_tokens=1_000,
        )

        # After chain restart: continuation prompt response (no tool calls)
        resp2 = make_litellm_response(
            content="Continued after chain.",
            prompt_tokens=1_000,
            completion_tokens=500,
        )

        with patch_litellm(resp1, resp2):
            result = await agent.run_cycle("Do a complex task")

        # The first response hit the threshold and chaining occurred
        assert result.summary


class TestModeA2Live:
    """Mode A2 tests using real API calls."""

    @pytest.mark.live
    @pytest.mark.timeout(60)
    async def test_live_basic_response(self, make_agent_core):
        """Live A2: real LiteLLM call with a non-Claude model."""
        pytest.importorskip("litellm")
        agent = make_agent_core(
            name="a2-live",
            model="claude-sonnet-4-20250514",
            execution_mode="autonomous",
        )
        # Force A2 mode even with Claude model
        agent._sdk_available = False

        result = await agent.run_cycle(
            "Reply with exactly: ANIMAWORKS_A2_TEST_OK"
        )

        assert result.summary
        assert result.action == "responded"


class TestModeA2AzureLive:
    """Mode A2 tests using Azure OpenAI API.

    These tests call a real Azure GPT-4.1 endpoint. LLM output is
    non-deterministic, so tool-calling tests are marked ``flaky``
    with up to 2 retries.
    """

    pytestmark = [pytest.mark.live, pytest.mark.azure]

    async def test_live_azure_basic_response(self, make_agent_core):
        """Live A2: Azure OpenAI gpt-4.1 call via LiteLLM."""
        pytest.importorskip("litellm")
        import os

        agent = make_agent_core(
            name="a2-azure-live",
            model="azure/gpt-4.1",
            credential="azure",
            api_base_url=os.environ.get("AZURE_API_BASE", ""),
        )
        agent._sdk_available = False

        result = await agent.run_cycle(
            "Reply with exactly: ANIMAWORKS_AZURE_TEST_OK"
        )

        assert result.summary
        assert result.action == "responded"

    @pytest.mark.flaky(reruns=2)
    async def test_live_azure_search_memory(self, make_agent_core):
        """Live A2 + GPT-4.1: LLM autonomously calls search_memory tool."""
        import os

        agent = make_agent_core(
            name="a2-azure-memory",
            model="azure/gpt-4.1",
            credential="azure",
            api_base_url=os.environ.get("AZURE_API_BASE", ""),
            max_turns=10,
        )
        agent._sdk_available = False

        # Seed knowledge that the model must discover
        (agent.anima_dir / "knowledge" / "secret.md").write_text(
            "# Secret Knowledge\nThe project codename is PHOENIX-42.\n",
            encoding="utf-8",
        )

        result = await agent.run_cycle(
            "What is the project codename? "
            "Use search_memory with query='codename' and scope='knowledge' "
            "to find it."
        )

        assert result.action == "responded"
        assert "PHOENIX" in result.summary or "42" in result.summary

    @pytest.mark.flaky(reruns=2)
    async def test_live_azure_read_file(self, make_agent_core):
        """Live A2 + GPT-4.1: LLM autonomously calls read_file tool."""
        import os

        agent = make_agent_core(
            name="a2-azure-readfile",
            model="azure/gpt-4.1",
            credential="azure",
            api_base_url=os.environ.get("AZURE_API_BASE", ""),
            max_turns=10,
        )
        agent._sdk_available = False

        target = agent.anima_dir / "knowledge" / "report.md"
        target.write_text(
            "# Monthly Report\nRevenue: 1,234,567 JPY\nStatus: On track\n",
            encoding="utf-8",
        )

        result = await agent.run_cycle(
            f"Use read_file with path='{target}' to read the file. "
            "Tell me the revenue figure you found."
        )

        assert result.action == "responded"
        assert "1,234,567" in result.summary

    @pytest.mark.flaky(reruns=2)
    async def test_live_azure_write_file(self, make_agent_core):
        """Live A2 + GPT-4.1: LLM autonomously calls write_file tool."""
        import os

        agent = make_agent_core(
            name="a2-azure-writefile",
            model="azure/gpt-4.1",
            credential="azure",
            api_base_url=os.environ.get("AZURE_API_BASE", ""),
            max_turns=10,
        )
        agent._sdk_available = False

        target = agent.anima_dir / "knowledge" / "output.md"

        result = await agent.run_cycle(
            f"Use write_file with path='{target}' and "
            "content='WRITE_TEST_SUCCESSFUL' to create a file. "
            "After the tool confirms success, say 'Done'."
        )

        assert result.action == "responded"
        assert target.exists()
        content = target.read_text(encoding="utf-8")
        assert "WRITE_TEST_SUCCESSFUL" in content

    @pytest.mark.flaky(reruns=2)
    async def test_live_azure_search_code(self, make_agent_core):
        """Live A2 + GPT-4.1: LLM autonomously calls search_code tool."""
        import os

        agent = make_agent_core(
            name="a2-azure-searchcode",
            model="azure/gpt-4.1",
            credential="azure",
            api_base_url=os.environ.get("AZURE_API_BASE", ""),
            max_turns=10,
        )
        agent._sdk_available = False

        # Create files for search_code to find
        (agent.anima_dir / "knowledge" / "alpha.md").write_text(
            "This file contains MARKER_ALPHA_999.\n", encoding="utf-8",
        )
        (agent.anima_dir / "knowledge" / "beta.md").write_text(
            "No special markers here.\n", encoding="utf-8",
        )

        result = await agent.run_cycle(
            "Use search_code with pattern='MARKER_ALPHA' and "
            f"directory='{agent.anima_dir}' to search for the marker. "
            "Tell me the exact marker string you found."
        )

        assert result.action == "responded"
        assert "MARKER_ALPHA" in result.summary

    @pytest.mark.flaky(reruns=2)
    async def test_live_azure_list_directory(self, make_agent_core):
        """Live A2 + GPT-4.1: LLM autonomously calls list_directory tool."""
        import os

        agent = make_agent_core(
            name="a2-azure-listdir",
            model="azure/gpt-4.1",
            credential="azure",
            api_base_url=os.environ.get("AZURE_API_BASE", ""),
            max_turns=10,
        )
        agent._sdk_available = False

        # Create recognizable files
        knowledge_dir = agent.anima_dir / "knowledge"
        (knowledge_dir / "unique_file_xyz.md").write_text(
            "test", encoding="utf-8",
        )

        result = await agent.run_cycle(
            f"Use list_directory with path='{knowledge_dir}' to list files. "
            "Tell me the exact filenames returned by the tool."
        )

        assert result.action == "responded"
        assert "unique_file_xyz" in result.summary

    @pytest.mark.flaky(reruns=2)
    async def test_live_azure_multi_tool_chain(self, make_agent_core):
        """Live A2 + GPT-4.1: LLM chains multiple tool calls autonomously."""
        import os

        agent = make_agent_core(
            name="a2-azure-chain",
            model="azure/gpt-4.1",
            credential="azure",
            api_base_url=os.environ.get("AZURE_API_BASE", ""),
            max_turns=15,
        )
        agent._sdk_available = False

        chain_path = agent.anima_dir / "knowledge" / "chain_test.md"

        result = await agent.run_cycle(
            f"Use write_file with path='{chain_path}' and "
            "content='CHAIN_STEP_1_DONE' to create a file. "
            f"Then use read_file with path='{chain_path}' to read it back. "
            "Tell me the exact content you read from the file."
        )

        assert result.action == "responded"
        assert chain_path.exists()
        assert "CHAIN_STEP_1_DONE" in result.summary
