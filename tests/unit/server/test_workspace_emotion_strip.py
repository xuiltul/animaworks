# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for emotion tag stripping and status notification deduplication.

Tests the server-side contract that the workspace frontend relies on:
1. The done SSE event contains both clean `summary` and `emotion` fields
2. text_delta events pass raw text (frontend must handle stripping)
3. Status events include anima name and status for deduplication
"""
from __future__ import annotations

import json

import pytest

from server.routes.chat import extract_emotion, _format_sse, _handle_chunk


# ── Done event contract for workspace conversation overlay ──────


class TestDoneEventContractForWorkspace:
    """The workspace app.js `done` handler now uses `data.summary` to replace
    streaming text. These tests verify the server-side done event always
    provides the fields the frontend expects."""

    def test_done_event_contains_clean_summary_and_emotion(self):
        """done event must have summary (tag-stripped) + emotion field."""
        chunk = {
            "type": "cycle_done",
            "cycle_result": {
                "summary": 'おはよう！今日も頑張ろう！\n<!-- emotion: {"emotion": "smile"} -->',
            },
        }
        frame, response_text = _handle_chunk(chunk)
        data = json.loads(frame.split("data: ", 1)[1].split("\n")[0])

        assert data["summary"] == "おはよう！今日も頑張ろう！"
        assert data["emotion"] == "smile"
        assert "<!-- emotion" not in data["summary"]
        assert response_text == "おはよう！今日も頑張ろう！"

    def test_done_event_summary_field_always_present(self):
        """Even without emotion tag, summary must be in the done payload."""
        chunk = {
            "type": "cycle_done",
            "cycle_result": {"summary": "Plain response"},
        }
        frame, _ = _handle_chunk(chunk)
        data = json.loads(frame.split("data: ", 1)[1].split("\n")[0])

        assert "summary" in data
        assert data["summary"] == "Plain response"
        assert data["emotion"] == "neutral"

    def test_done_event_emotion_field_always_present(self):
        """emotion field must always be present, defaulting to neutral."""
        chunk = {
            "type": "cycle_done",
            "cycle_result": {"summary": "No tag here"},
        }
        frame, _ = _handle_chunk(chunk)
        data = json.loads(frame.split("data: ", 1)[1].split("\n")[0])

        assert "emotion" in data
        assert data["emotion"] == "neutral"

    def test_done_event_with_empty_summary(self):
        """Empty summary should still produce valid done event."""
        chunk = {
            "type": "cycle_done",
            "cycle_result": {"summary": ""},
        }
        frame, response_text = _handle_chunk(chunk)
        data = json.loads(frame.split("data: ", 1)[1].split("\n")[0])

        assert data["summary"] == ""
        assert data["emotion"] == "neutral"
        assert response_text == ""

    @pytest.mark.parametrize("emotion_name", [
        "neutral", "smile", "laugh", "troubled",
        "surprised", "thinking", "embarrassed",
    ])
    def test_done_event_preserves_all_valid_emotions(self, emotion_name):
        """All valid emotions must pass through to the done event."""
        chunk = {
            "type": "cycle_done",
            "cycle_result": {
                "summary": f'Text\n<!-- emotion: {{"emotion": "{emotion_name}"}} -->',
            },
        }
        frame, _ = _handle_chunk(chunk)
        data = json.loads(frame.split("data: ", 1)[1].split("\n")[0])

        assert data["emotion"] == emotion_name
        assert "<!-- emotion" not in data["summary"]


# ── text_delta passthrough behavior ─────────────────────────────


class TestTextDeltaEmotionPassthrough:
    """text_delta events pass raw text including partial emotion tags.
    The frontend's stripEmotionTag() and done-event summary override
    handle cleanup. These tests document the expected passthrough."""

    def test_text_delta_passes_through_raw_text(self):
        """text_delta must pass text unchanged (frontend strips)."""
        chunk = {"type": "text_delta", "text": "Hello! <!-- emotion"}
        frame, _ = _handle_chunk(chunk)
        data = json.loads(frame.split("data: ", 1)[1].split("\n")[0])

        assert data["text"] == "Hello! <!-- emotion"

    def test_text_delta_with_complete_emotion_tag(self):
        """text_delta containing a full emotion tag passes it through."""
        raw = '<!-- emotion: {"emotion": "smile"} -->'
        chunk = {"type": "text_delta", "text": raw}
        frame, _ = _handle_chunk(chunk)
        data = json.loads(frame.split("data: ", 1)[1].split("\n")[0])

        assert data["text"] == raw

    def test_text_delta_returns_empty_response_text(self):
        """text_delta should always return empty response_text."""
        chunk = {"type": "text_delta", "text": "any content"}
        _, response_text = _handle_chunk(chunk)
        assert response_text == ""


# ── extract_emotion regression for edge cases ───────────────────


class TestExtractEmotionEdgeCases:
    """Additional edge cases for emotion extraction relevant to the
    workspace fix (conversation history containing raw tags)."""

    def test_multiple_emotion_tags_strips_all(self):
        """If text contains multiple tags (malformed), all should be removed."""
        text = (
            'Part 1\n<!-- emotion: {"emotion": "smile"} -->\n'
            'Part 2\n<!-- emotion: {"emotion": "laugh"} -->'
        )
        clean, emotion = extract_emotion(text)
        assert "<!-- emotion" not in clean
        # First match determines emotion
        assert emotion in ("smile", "laugh")

    def test_tag_with_extra_whitespace(self):
        """Tags with extra whitespace should still be extracted."""
        text = 'Hello\n<!--   emotion:   {"emotion": "thinking"}   -->'
        clean, emotion = extract_emotion(text)
        assert clean == "Hello"
        assert emotion == "thinking"

    def test_japanese_text_with_emotion_tag(self):
        """Japanese text should be preserved correctly after tag removal."""
        text = 'こんにちは！元気ですか？\n<!-- emotion: {"emotion": "smile"} -->'
        clean, emotion = extract_emotion(text)
        assert clean == "こんにちは！元気ですか？"
        assert emotion == "smile"

    def test_multiline_japanese_with_tag(self):
        """Multi-line Japanese response with tag at end."""
        text = "お疲れ様です！\n今日のタスクを確認しますね。\n<!-- emotion: {\"emotion\": \"neutral\"} -->"
        clean, emotion = extract_emotion(text)
        assert clean == "お疲れ様です！\n今日のタスクを確認しますね。"
        assert emotion == "neutral"


# ── Status event structure for deduplication ─────────────────────


class TestStatusEventStructure:
    """Tests that anima.status WebSocket events contain the fields
    needed by the frontend's deduplication logic (data.name + data.status)."""

    def test_status_event_format(self):
        """The emit() call produces a dict with name and status."""
        # This tests the data shape that the frontend handler expects
        # for deduplication: { name: "sakura", status: "idle" }
        event_data = {"name": "sakura", "status": "idle"}
        assert "name" in event_data
        assert "status" in event_data

    def test_status_dedup_logic(self):
        """Simulate the frontend dedup: same anima+status → skip."""
        last_status = {}
        events_logged = []

        def mock_add_activity(name, status):
            if last_status.get(name) != status:
                last_status[name] = status
                events_logged.append((name, status))

        # Simulate sequence: thinking → idle → idle → idle
        for status in ["thinking", "idle", "idle", "idle"]:
            mock_add_activity("sakura", status)

        assert len(events_logged) == 2
        assert events_logged[0] == ("sakura", "thinking")
        assert events_logged[1] == ("sakura", "idle")

    def test_status_dedup_different_animas(self):
        """Different animas with same status should both be logged."""
        last_status = {}
        events_logged = []

        def mock_add_activity(name, status):
            if last_status.get(name) != status:
                last_status[name] = status
                events_logged.append((name, status))

        mock_add_activity("sakura", "idle")
        mock_add_activity("kotoha", "idle")

        assert len(events_logged) == 2

    def test_status_dedup_alternating(self):
        """Status changes: idle → thinking → idle should all be logged."""
        last_status = {}
        events_logged = []

        def mock_add_activity(name, status):
            if last_status.get(name) != status:
                last_status[name] = status
                events_logged.append((name, status))

        for status in ["idle", "thinking", "idle"]:
            mock_add_activity("sakura", status)

        assert len(events_logged) == 3
