"""Tests for StreamingThinkFilter and strip_thinking_tags."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from core.execution.base import StreamingThinkFilter, strip_thinking_tags, strip_untagged_thinking

# ── strip_thinking_tags ──────────────────────────────────────


class TestStripThinkingTags:
    def test_no_think_tags(self):
        thinking, response = strip_thinking_tags("Hello world")
        assert thinking == ""
        assert response == "Hello world"

    def test_with_think_tags(self):
        text = "<think>reasoning here</think>actual response"
        thinking, response = strip_thinking_tags(text)
        assert thinking == "reasoning here"
        assert response == "actual response"

    def test_think_tag_stripped_from_thinking(self):
        text = "<think>some thinking</think>response"
        thinking, response = strip_thinking_tags(text)
        assert "<think>" not in thinking
        assert thinking == "some thinking"

    def test_empty_think_block(self):
        text = "<think></think>response"
        thinking, response = strip_thinking_tags(text)
        assert thinking == ""
        assert response == "response"

    def test_multiline_thinking(self):
        text = "<think>line1\nline2\nline3</think>response"
        thinking, response = strip_thinking_tags(text)
        assert thinking == "line1\nline2\nline3"
        assert response == "response"

    def test_no_closing_tag(self):
        text = "<think>unclosed thinking"
        thinking, response = strip_thinking_tags(text)
        assert thinking == ""
        assert response == text

    def test_whitespace_after_closing_tag(self):
        text = "<think>thought</think>  \nresponse"
        thinking, response = strip_thinking_tags(text)
        assert thinking == "thought"

    def test_missing_open_tag(self):
        """vLLM reasoning parser may strip <think> but leave </think>."""
        text = "reasoning here</think>actual response"
        thinking, response = strip_thinking_tags(text)
        assert thinking == "reasoning here"
        assert response == "actual response"

    def test_multiple_think_blocks_no_leak(self):
        """Qwen models may emit multiple </think> tags; none should leak."""
        text = "<think>thinking1</think>response1\nmore thinking\n</think>response2"
        thinking, response = strip_thinking_tags(text)
        assert thinking == "thinking1"
        assert "</think>" not in response
        assert "response1" in response
        assert "response2" in response


# ── StreamingThinkFilter ─────────────────────────────────────


class TestStreamingThinkFilter:
    def test_no_think_tags_passthrough(self):
        """Non-think content passes through immediately."""
        f = StreamingThinkFilter()
        thinking, text = f.feed("Hello world")
        assert thinking == ""
        assert text == "Hello world"

    def test_think_tag_in_single_chunk(self):
        f = StreamingThinkFilter()
        thinking, text = f.feed("<think>reasoning</think>response")
        assert "reasoning" in thinking
        assert "<think>" not in thinking
        assert text == "response"

    def test_think_tag_across_chunks(self):
        f = StreamingThinkFilter()
        t1, r1 = f.feed("<thi")
        assert t1 == "" and r1 == ""

        t2, r2 = f.feed("nk>my thinking</think>response text")
        assert "<think>" not in t2
        assert "my thinking" in t2
        assert r2 == "response text"

    def test_think_tag_stripped_from_output(self):
        f = StreamingThinkFilter()
        thinking, text = f.feed("<think>hello</think>world")
        assert not thinking.startswith("<think>")
        assert thinking == "hello"
        assert text == "world"

    def test_flush_without_closing_tag(self):
        f = StreamingThinkFilter()
        t, r = f.feed("<think>partial thinking")
        assert t == "" and r == ""
        flushed = f.flush()
        assert "partial thinking" in flushed

    def test_flush_empty_when_complete(self):
        f = StreamingThinkFilter()
        f.feed("<think>thought</think>response")
        flushed = f.flush()
        assert flushed == ""

    def test_non_think_content_immediate_passthrough(self):
        """Content not starting with <think> passes through at once."""
        f = StreamingThinkFilter()
        t1, r1 = f.feed("regular content")
        assert t1 == ""
        assert r1 == "regular content"
        t2, r2 = f.feed(" more content")
        assert t2 == "" and r2 == " more content"

    def test_after_done_passthrough(self):
        f = StreamingThinkFilter()
        f.feed("<think>thought</think>first")
        t, r = f.feed("second")
        assert t == "" and r == "second"

    def test_after_done_strips_orphan_close_tags(self):
        """After initial think block, subsequent </think> tags are stripped."""
        f = StreamingThinkFilter()
        f.feed("<think>thought</think>first")
        t, r = f.feed("more thinking</think>second response")
        assert t == ""
        assert "</think>" not in r
        assert "more thinking" in r
        assert "second response" in r

    def test_multiple_close_tags_in_passthrough(self):
        """Early-exit passthrough preserves </think> for safety-net detection.

        When the initial chunk triggers early-exit (no <think> prefix),
        _saw_think_close stays False, so </think> tags in passthrough are NOT
        stripped — they remain for strip_thinking_tags to detect the leak.
        """
        f = StreamingThinkFilter()
        # First chunk triggers early-exit (no <think> prefix); _saw_think_close=False
        t1, r1 = f.feed("step 1 done")
        assert r1 == "step 1 done"
        # </think> preserved (not stripped) — post-stream safety net handles them
        t2, r2 = f.feed("</think>\nresult 1\nthinking again</think>\nresult 2")
        assert t2 == ""
        assert "</think>" in r2  # preserved for safety net
        assert "result 1" in r2
        assert "result 2" in r2

    def test_response_newline_stripped(self):
        f = StreamingThinkFilter()
        t, r = f.feed("<think>thought</think>\n\nresponse")
        assert r == "response"

    def test_whitespace_before_think_tag(self):
        f = StreamingThinkFilter()
        t, r = f.feed("  <think>thought</think>response")
        assert "thought" in t
        assert r == "response"

    def test_buffer_overflow_safety_valve(self):
        f = StreamingThinkFilter()
        huge = "<think>" + "x" * 60_000
        t, r = f.feed(huge)
        assert r == huge
        assert t == ""

    # ── Pattern 2: missing <think> but </think> present ──────

    def test_missing_open_tag_single_chunk(self):
        """vLLM reasoning parser may strip <think> but leave </think>.

        When both thinking and </think> arrive in a single chunk, the
        filter correctly splits them because </think> is checked FIRST.
        """
        f = StreamingThinkFilter()
        t, r = f.feed("reasoning content here</think>\n\nactual response")
        assert "reasoning content here" in t
        assert r == "actual response"

    def test_missing_open_tag_across_chunks_preserves_close_tag(self):
        """Across-chunk missing-<think>: </think> is preserved for safety-net.

        The first chunk triggers early-exit (_saw_think_close stays False).
        The subsequent </think> is NOT stripped so the post-stream
        strip_thinking_tags safety net can retroactively identify the earlier
        chunks as thinking content.
        """
        f = StreamingThinkFilter()
        t1, r1 = f.feed("I need to ")
        assert t1 == "" and r1 == "I need to "

        t2, r2 = f.feed("analyze this carefully")
        assert t2 == "" and r2 == "analyze this carefully"

        t3, r3 = f.feed("</think>\n\nHere is my response")
        assert t3 == ""
        assert "</think>" in r3  # preserved — safety net will detect the thinking leak
        assert "Here is my response" in r3

    def test_missing_open_tag_across_chunks_safety_net_catches_thinking(self):
        """Full-text safety net correctly identifies thinking when </think> is preserved.

        When </think> is not stripped in early-exit passthrough mode, the
        post-stream strip_thinking_tags call can retroactively split
        pre-</think> content as thinking and post-</think> as the response.
        """
        f = StreamingThinkFilter()
        _, r1 = f.feed("I need to ")
        _, r2 = f.feed("think carefully")
        _, r3 = f.feed("</think>\n\nHere is my answer")

        # All chunks emitted as text_delta (filter doesn't know it's thinking yet).
        # full_text still contains </think>, enabling the safety net.
        full_text = r1 + r2 + r3
        leaked_think, clean = strip_thinking_tags(full_text)
        assert leaked_think == "I need to think carefully"
        assert "Here is my answer" in clean
        assert "</think>" not in clean


# ── strip_untagged_thinking ─────────────────────────────────


class TestStripUntaggedThinking:
    """Tests for vLLM tag-stripped thinking detection (no <think>/</ think> tags)."""

    def test_no_thinking_prefix_returns_original(self):
        text = "Hello world, how are you?"
        thinking, response = strip_untagged_thinking(text)
        assert thinking == ""
        assert response == text

    def test_short_text_returns_original(self):
        text = "Thinking Process: ok"
        thinking, response = strip_untagged_thinking(text)
        assert thinking == ""
        assert response == text

    def test_triple_newline_boundary(self):
        thinking_block = "Thinking Process:\n\n1. Analyze\n2. Plan\n3. Execute: say hi"
        actual_response = "こんにちは！"
        text = thinking_block + "\n\n\n" + actual_response
        thinking, response = strip_untagged_thinking(text)
        assert thinking != ""
        assert "Thinking Process:" in thinking
        assert response == actual_response

    def test_double_newline_fallback(self):
        thinking_block = (
            "Thinking Process:\n\n1. The user greeted me.\n\n2. I should greet back.\n\n3. Final: say hello."
        )
        actual_response = "Hi"
        text = thinking_block + "\n\n" + actual_response
        thinking, response = strip_untagged_thinking(text)
        assert thinking != ""
        assert response == actual_response

    def test_japanese_prefix(self):
        thinking_block = "思考プロセス:\n\n1. 分析\n2. 計画\n3. 実行"
        actual_response = "こんにちは！"
        text = thinking_block + "\n\n\n" + actual_response
        thinking, response = strip_untagged_thinking(text)
        assert thinking != ""
        assert response == actual_response

    def test_let_me_think_prefix(self):
        thinking_block = (
            "Let me think about this carefully.\n\nStep 1: Analyze the greeting.\n\nStep 2: Formulate a response."
        )
        actual_response = "Hello!"
        text = thinking_block + "\n\n\n" + actual_response
        thinking, response = strip_untagged_thinking(text)
        assert thinking != ""
        assert response == actual_response

    def test_no_boundary_returns_original(self):
        text = "Thinking Process: " + "x" * 300
        thinking, response = strip_untagged_thinking(text)
        assert thinking == ""
        assert response == text

    def test_response_too_long_ratio_ignored(self):
        """When last segment is > 40% of total, skip double-newline fallback."""
        thinking_block = "Thinking Process:\n\n1. Short thinking."
        long_response = "This is a very long response " * 10
        text = thinking_block + "\n\n" + long_response
        thinking, response = strip_untagged_thinking(text)
        if thinking:
            assert len(response) < len(text) * 0.4
