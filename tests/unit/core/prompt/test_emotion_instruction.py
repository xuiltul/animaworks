"""Unit tests for EMOTION_INSTRUCTION in core/prompt/builder.py."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from core.prompt.builder import EMOTION_INSTRUCTION


class TestEmotionInstruction:
    """Tests for the EMOTION_INSTRUCTION constant."""

    def test_instruction_is_non_empty_string(self):
        assert isinstance(EMOTION_INSTRUCTION, str)
        assert len(EMOTION_INSTRUCTION) > 0

    def test_contains_all_valid_emotions(self):
        for emotion in ["neutral", "smile", "laugh", "troubled",
                        "surprised", "thinking", "embarrassed"]:
            assert emotion in EMOTION_INSTRUCTION

    def test_contains_metadata_format(self):
        assert "<!-- emotion:" in EMOTION_INSTRUCTION
        assert '"emotion"' in EMOTION_INSTRUCTION

    def test_no_invalid_emotions_mentioned(self):
        """Ensure old/invalid emotion names are not in the instruction."""
        # These were removed in v2 design
        lines = EMOTION_INSTRUCTION.lower()
        # 'angry' should not be in the available emotions list
        # (it might appear in example context, but check the emotion list line)
        assert "angry" not in lines
