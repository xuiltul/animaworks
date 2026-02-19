"""Unit tests for expression generation in core/tools/image_gen.py."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.tools.image_gen import (
    _EXPRESSION_PROMPTS,
    _VALID_EXPRESSION_NAMES,
    PipelineResult,
)


class TestExpressionPrompts:
    """Tests for expression prompt definitions."""

    def test_all_expressions_have_prompts(self):
        expected = {"neutral", "smile", "laugh", "troubled",
                    "surprised", "thinking", "embarrassed"}
        assert set(_EXPRESSION_PROMPTS.keys()) == expected

    def test_valid_names_match_prompts(self):
        assert _VALID_EXPRESSION_NAMES == frozenset(_EXPRESSION_PROMPTS.keys())

    def test_prompts_are_non_empty_strings(self):
        for name, prompt in _EXPRESSION_PROMPTS.items():
            assert isinstance(prompt, str), f"{name} prompt is not a string"
            assert len(prompt) > 50, f"{name} prompt is too short"

    def test_prompts_contain_character_reference(self):
        """All prompts should reference 'same character' for consistency."""
        for name, prompt in _EXPRESSION_PROMPTS.items():
            assert "same character" in prompt.lower(), (
                f"{name} prompt missing character reference"
            )

    def test_prompts_specify_anime_style(self):
        for name, prompt in _EXPRESSION_PROMPTS.items():
            assert "anime" in prompt.lower(), (
                f"{name} prompt missing anime style"
            )


class TestPipelineResultExpressions:
    """Tests for bustup_paths in PipelineResult."""

    def test_bustup_paths_default_empty(self):
        result = PipelineResult()
        assert result.bustup_paths == {}

    def test_bustup_paths_in_to_dict(self):
        result = PipelineResult()
        result.bustup_paths = {
            "neutral": Path("/tmp/neutral.png"),
            "smile": Path("/tmp/smile.png"),
        }
        d = result.to_dict()
        assert "bustup_expressions" in d
        assert d["bustup_expressions"]["neutral"] == "/tmp/neutral.png"
        assert d["bustup_expressions"]["smile"] == "/tmp/smile.png"

    def test_empty_bustup_paths_in_to_dict(self):
        result = PipelineResult()
        d = result.to_dict()
        assert d["bustup_expressions"] == {}
