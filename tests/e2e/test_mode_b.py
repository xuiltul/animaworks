"""Tests for Mode B (assisted) execution.

Mode B is a 1-shot LLM call where the framework handles memory I/O.
Post-call: episode recording and knowledge extraction.
"""

from __future__ import annotations

from datetime import date

import pytest

from tests.helpers.mocks import make_litellm_response, patch_litellm


class TestModeBMock:
    """Mode B tests using mocked LLM calls."""

    async def test_basic_oneshot_response(self, make_agent_core):
        """Basic 1-shot response returns correct CycleResult."""
        agent = make_agent_core(
            name="b-basic",
            model="ollama/gemma3:27b",
            execution_mode="assisted",
        )

        main_resp = make_litellm_response(content="Assisted response from LLM.")
        extract_resp = make_litellm_response(content="なし")

        with patch_litellm(main_resp, extract_resp):
            result = await agent.run_cycle("Hello, tell me something.")

        assert result.action == "responded"
        assert result.summary == "Assisted response from LLM."

    async def test_episode_recorded_after_response(self, make_agent_core):
        """Episode file is written after a Mode B response."""
        agent = make_agent_core(
            name="b-episode",
            model="ollama/gemma3:27b",
            execution_mode="assisted",
        )

        main_resp = make_litellm_response(content="I think about things.")
        extract_resp = make_litellm_response(content="なし")

        with patch_litellm(main_resp, extract_resp):
            await agent.run_cycle("What do you think?")

        # Check episode file was written
        today = date.today().isoformat()
        episode_path = agent.person_dir / "episodes" / f"{today}.md"
        assert episode_path.exists()
        content = episode_path.read_text(encoding="utf-8")
        assert "[assisted]" in content
        assert "What do you think?" in content

    async def test_knowledge_extracted_after_response(self, make_agent_core):
        """Knowledge file is created when LLM extracts useful information."""
        agent = make_agent_core(
            name="b-knowledge",
            model="ollama/gemma3:27b",
            execution_mode="assisted",
        )

        main_resp = make_litellm_response(
            content="The capital of France is Paris."
        )
        extract_resp = make_litellm_response(
            content="## 知識抽出\nFrance's capital is Paris — useful geographic fact.\n\n## 返信判定\n返信不要"
        )

        with patch_litellm(main_resp, extract_resp):
            await agent.run_cycle("What is the capital of France?")

        # Check knowledge file was created
        knowledge_files = list(
            agent.person_dir.glob("knowledge/learned_*.md")
        )
        assert len(knowledge_files) >= 1
        content = knowledge_files[0].read_text(encoding="utf-8")
        assert "Paris" in content

    async def test_knowledge_extraction_skipped_when_nashi(self, make_agent_core):
        """No knowledge file when LLM returns 'なし'."""
        agent = make_agent_core(
            name="b-no-knowledge",
            model="ollama/gemma3:27b",
            execution_mode="assisted",
        )

        main_resp = make_litellm_response(content="Hello!")
        extract_resp = make_litellm_response(content="なし")

        # Snapshot existing knowledge files
        existing = set(agent.person_dir.glob("knowledge/*.md"))

        with patch_litellm(main_resp, extract_resp):
            await agent.run_cycle("Hi there")

        new_files = set(agent.person_dir.glob("knowledge/*.md")) - existing
        assert len(new_files) == 0


class TestModeBLive:
    """Mode B tests using real API calls."""

    @pytest.mark.live
    @pytest.mark.timeout(60)
    async def test_live_basic_response(self, make_agent_core):
        """Live Mode B: real LLM call produces a response and records an episode."""
        agent = make_agent_core(
            name="b-live",
            model="claude-sonnet-4-20250514",
            execution_mode="assisted",
        )

        result = await agent.run_cycle(
            "Reply with exactly: ANIMAWORKS_B_TEST_OK"
        )

        assert result.summary
        assert result.action == "responded"
        # Episode should be recorded
        today = date.today().isoformat()
        episode_path = agent.person_dir / "episodes" / f"{today}.md"
        assert episode_path.exists()


class TestModeBOllamaLive:
    """Mode B tests using Ollama on remote GPU server."""

    @pytest.mark.live
    @pytest.mark.ollama
    @pytest.mark.timeout(120)
    async def test_live_ollama_basic_response(self, make_agent_core):
        """Live Mode B: Ollama model produces an assisted response."""
        import os

        agent = make_agent_core(
            name="b-ollama-live",
            model="ollama/glm-flash-q8:16k",
            credential="ollama",
            execution_mode="assisted",
            api_base_url=os.environ.get("OLLAMA_API_BASE", ""),
        )

        result = await agent.run_cycle(
            "Reply with exactly: ANIMAWORKS_OLLAMA_TEST_OK"
        )

        assert result.summary
        assert result.action == "responded"


class TestModeBAzureLive:
    """Mode B tests using Azure OpenAI API."""

    @pytest.mark.live
    @pytest.mark.azure
    @pytest.mark.timeout(60)
    async def test_live_azure_assisted_response(self, make_agent_core):
        """Live Mode B: Azure OpenAI gpt-4.1 in assisted mode."""
        import os

        agent = make_agent_core(
            name="b-azure-live",
            model="azure/gpt-4.1",
            credential="azure",
            execution_mode="assisted",
            api_base_url=os.environ.get("AZURE_API_BASE", ""),
        )

        result = await agent.run_cycle(
            "Reply with exactly: ANIMAWORKS_AZURE_B_TEST_OK"
        )

        assert result.summary
        assert result.action == "responded"
