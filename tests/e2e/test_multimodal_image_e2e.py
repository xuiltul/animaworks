# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for multimodal image input across execution modes.

Verifies that process_message, run_cycle, and process_message_stream
correctly accept, forward, and persist image data through the
Mode B (assisted), Mode A2 (LiteLLM), and AgentCore layers.
"""

from __future__ import annotations

import json

from tests.helpers.mocks import make_litellm_response, patch_litellm


# ── Mode B (assisted) ────────────────────────────────────────


class TestMultimodalModeB:
    """Mode B image handling: warns and ignores images."""

    async def test_process_message_with_images_mode_b(self, make_digital_anima):
        """Mode B (assisted) process_message accepts images but warns and ignores them."""
        dp = make_digital_anima(
            name="multimodal-b",
            model="ollama/gemma3:27b",
            execution_mode="assisted",
        )

        images = [{"data": "dGVzdA==", "media_type": "image/png"}]

        main_resp = make_litellm_response(content="I see your message.")
        extract_resp = make_litellm_response(content="\u306a\u3057")

        with patch_litellm(main_resp, extract_resp):
            result = await dp.process_message(
                "check this image",
                from_person="human",
                images=images,
            )

        assert "I see your message" in result


# ── Mode A2 (LiteLLM) ───────────────────────────────────────


class TestMultimodalModeA2:
    """Mode A2 image handling: passes images as OpenAI vision format."""

    async def test_process_message_with_images_a2(self, make_digital_anima):
        """Mode A2 (LiteLLM) passes images as OpenAI vision format."""
        dp = make_digital_anima(
            name="multimodal-a2",
            model="openai/gpt-4o",
        )

        images = [{"data": "dGVzdA==", "media_type": "image/png"}]

        resp = make_litellm_response(content="I can see a test image.")

        with patch_litellm(resp) as mock_fn:
            result = await dp.process_message(
                "describe this",
                from_person="human",
                images=images,
            )

        assert "test image" in result

        # Verify LiteLLM was called with image content blocks
        call_args = mock_fn.call_args
        messages = call_args.kwargs.get("messages", [])
        # Find the user message (after system message)
        user_msg = next((m for m in messages if m["role"] == "user"), None)
        assert user_msg is not None
        # When images are present, content should be a list
        assert isinstance(user_msg["content"], list)
        # Check for image_url block
        image_blocks = [
            b for b in user_msg["content"] if b.get("type") == "image_url"
        ]
        assert len(image_blocks) == 1
        assert "data:image/png;base64," in image_blocks[0]["image_url"]["url"]


# ── Conversation memory persistence ─────────────────────────


class TestMultimodalConversationMemory:
    """Verify attachments are persisted in conversation.json."""

    async def test_attachments_persisted(self, make_digital_anima):
        """process_message with attachment_paths stores them in conversation."""
        dp = make_digital_anima(
            name="multimodal-conv",
            model="ollama/gemma3:27b",
            execution_mode="assisted",
        )

        main_resp = make_litellm_response(content="Noted.")
        extract_resp = make_litellm_response(content="\u306a\u3057")

        with patch_litellm(main_resp, extract_resp):
            await dp.process_message(
                "here is a photo",
                from_person="human",
                attachment_paths=["attachments/20260217_120000_0.png"],
            )

        # Check conversation.json
        conv_path = dp.anima_dir / "state" / "conversation.json"
        assert conv_path.exists()
        data = json.loads(conv_path.read_text(encoding="utf-8"))
        turns = data.get("turns", [])

        # First turn should be human with attachments
        human_turn = turns[0]
        assert human_turn["role"] == "human"
        assert human_turn.get("attachments") == [
            "attachments/20260217_120000_0.png"
        ]


# ── Backward compatibility ───────────────────────────────────


class TestMultimodalBackwardCompat:
    """Verify that omitting images does not break existing behaviour."""

    async def test_process_message_without_images(self, make_digital_anima):
        """process_message without images works as before."""
        dp = make_digital_anima(
            name="multimodal-compat",
            model="ollama/gemma3:27b",
            execution_mode="assisted",
        )

        main_resp = make_litellm_response(content="Hello!")
        extract_resp = make_litellm_response(content="\u306a\u3057")

        with patch_litellm(main_resp, extract_resp):
            result = await dp.process_message("Hi", from_person="human")

        assert "Hello" in result


# ── AgentCore.run_cycle ──────────────────────────────────────


class TestMultimodalAgentCore:
    """AgentCore.run_cycle forwards images to the executor."""

    async def test_run_cycle_with_images_a2(self, make_agent_core):
        """AgentCore.run_cycle passes images to A2 executor."""
        agent = make_agent_core(
            name="agent-img-a2",
            model="openai/gpt-4o",
        )

        images = [{"data": "dGVzdA==", "media_type": "image/jpeg"}]

        resp = make_litellm_response(content="Image analyzed.")

        with patch_litellm(resp):
            result = await agent.run_cycle(
                "analyze this image",
                trigger="message:human",
                images=images,
            )

        assert result.summary == "Image analyzed."

    async def test_run_cycle_without_images(self, make_agent_core):
        """AgentCore.run_cycle works without images (backward compat)."""
        agent = make_agent_core(
            name="agent-noimg",
            model="openai/gpt-4o",
        )

        resp = make_litellm_response(content="OK.")

        with patch_litellm(resp):
            result = await agent.run_cycle("hello", trigger="message:human")

        assert result.summary == "OK."


# ── Streaming ────────────────────────────────────────────────


class TestMultimodalStreaming:
    """Streaming path accepts images and yields expected event types."""

    async def test_process_message_stream_with_images(self, make_digital_anima):
        """process_message_stream accepts images parameter."""
        dp = make_digital_anima(
            name="multimodal-stream",
            model="ollama/gemma3:27b",
            execution_mode="assisted",
        )

        images = [{"data": "dGVzdA==", "media_type": "image/png"}]

        main_resp = make_litellm_response(content="I see it.")
        extract_resp = make_litellm_response(content="\u306a\u3057")

        chunks = []
        with patch_litellm(main_resp, extract_resp):
            async for chunk in dp.process_message_stream(
                "look at this",
                from_person="human",
                images=images,
            ):
                chunks.append(chunk)

        # Should have text_delta and cycle_done events
        types = [c.get("type") for c in chunks]
        assert "text_delta" in types
        assert "cycle_done" in types
