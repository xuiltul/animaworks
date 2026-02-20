from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.


"""Unit tests for expression generation improvements in image_gen.py.

Tests cover:
- _EXPRESSION_PROMPTS dictionary: prompt content, preservation clauses,
  action verbs, coverage of all 7 expressions
- _EXPRESSION_GUIDANCE dictionary: per-expression guidance_scale values
- generate_bustup_expression(): guidance_scale passthrough to FluxKontextClient
- generate_all() 2-stage pipeline: neutral-first strategy, reference chaining,
  fallback to fullbody when neutral fails
- builder.py EMOTION_INSTRUCTION: contains all 7 emotion names
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.schemas import VALID_EMOTIONS
from core.tools.image_gen import (
    ImageGenPipeline,
    _EXPRESSION_GUIDANCE,
    _EXPRESSION_PROMPTS,
)


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def anima_dir(tmp_path: Path) -> Path:
    """Create a temporary anima directory with assets subdirectory."""
    assets = tmp_path / "assets"
    assets.mkdir()
    return tmp_path


@pytest.fixture
def fake_image_bytes() -> bytes:
    """Return minimal fake image bytes for use as reference images."""
    return b"\x89PNG\r\n\x1a\nFAKE_IMAGE_DATA"


@pytest.fixture
def pipeline(anima_dir: Path) -> ImageGenPipeline:
    """Create an ImageGenPipeline pointing at the temp anima directory."""
    return ImageGenPipeline(anima_dir)


# ── 1. _EXPRESSION_PROMPTS Tests ────────────────────────────────────


class TestExpressionPrompts:
    """Tests for the _EXPRESSION_PROMPTS dictionary."""

    def test_all_seven_expressions_defined(self):
        """All 7 valid expressions must have a prompt entry."""
        expected = {"neutral", "smile", "laugh", "troubled",
                    "surprised", "thinking", "embarrassed"}
        assert set(_EXPRESSION_PROMPTS.keys()) == expected

    def test_prompts_match_valid_emotions(self):
        """Expression prompt keys must exactly match VALID_EMOTIONS from schemas."""
        assert set(_EXPRESSION_PROMPTS.keys()) == VALID_EMOTIONS

    def test_no_old_preservation_clause(self):
        """No prompt should contain the old 'Same outfit, same colors, same features' clause."""
        old_clause = "Same outfit, same colors, same features"
        for expression, prompt in _EXPRESSION_PROMPTS.items():
            assert old_clause not in prompt, (
                f"Expression '{expression}' still contains the old preservation clause: "
                f"'{old_clause}'"
            )

    def test_new_preservation_clause_present(self):
        """Every prompt must contain the new preservation clause."""
        new_clause = "Same character identity, outfit, and hairstyle"
        for expression, prompt in _EXPRESSION_PROMPTS.items():
            assert new_clause in prompt, (
                f"Expression '{expression}' is missing the new preservation clause: "
                f"'{new_clause}'"
            )

    def test_non_neutral_prompts_contain_action_verb(self):
        """Non-neutral prompts must contain the 'Change the character's expression' action verb."""
        action_phrase = "Change the character's expression"
        for expression, prompt in _EXPRESSION_PROMPTS.items():
            if expression == "neutral":
                continue
            assert action_phrase in prompt, (
                f"Non-neutral expression '{expression}' is missing the action verb: "
                f"'{action_phrase}'"
            )

    def test_neutral_prompt_does_not_contain_change_action(self):
        """The neutral prompt should NOT contain 'Change the character's expression'."""
        action_phrase = "Change the character's expression"
        assert action_phrase not in _EXPRESSION_PROMPTS["neutral"], (
            "Neutral prompt should not contain the expression change directive"
        )

    def test_all_prompts_are_nonempty_strings(self):
        """Every prompt must be a non-empty string."""
        for expression, prompt in _EXPRESSION_PROMPTS.items():
            assert isinstance(prompt, str), (
                f"Prompt for '{expression}' is not a string: {type(prompt)}"
            )
            assert len(prompt.strip()) > 0, (
                f"Prompt for '{expression}' is empty"
            )

    def test_all_prompts_contain_bustup_portrait(self):
        """Every prompt should mention 'Bust-up portrait' for format consistency."""
        for expression, prompt in _EXPRESSION_PROMPTS.items():
            assert "Bust-up portrait" in prompt, (
                f"Expression '{expression}' is missing 'Bust-up portrait' directive"
            )

    @pytest.mark.parametrize("expression", [
        "smile", "laugh", "troubled", "surprised", "thinking", "embarrassed",
    ])
    def test_non_neutral_prompt_structure(self, expression: str):
        """Each non-neutral prompt should start with 'Change the character's expression'."""
        prompt = _EXPRESSION_PROMPTS[expression]
        assert prompt.startswith("Change the character's expression"), (
            f"Expression '{expression}' prompt does not start with the expected action phrase"
        )


# ── 2. _EXPRESSION_GUIDANCE Tests ───────────────────────────────────


class TestExpressionGuidance:
    """Tests for the _EXPRESSION_GUIDANCE dictionary."""

    def test_all_seven_expressions_have_guidance(self):
        """All 7 expressions must have a guidance_scale mapping."""
        expected = {"neutral", "smile", "laugh", "troubled",
                    "surprised", "thinking", "embarrassed"}
        assert set(_EXPRESSION_GUIDANCE.keys()) == expected

    def test_guidance_matches_valid_emotions(self):
        """Guidance keys must exactly match VALID_EMOTIONS from schemas."""
        assert set(_EXPRESSION_GUIDANCE.keys()) == VALID_EMOTIONS

    def test_neutral_guidance_value(self):
        """Neutral guidance_scale should be 4.0."""
        assert _EXPRESSION_GUIDANCE["neutral"] == 4.0

    def test_smile_guidance_value(self):
        """Smile guidance_scale should be 5.0."""
        assert _EXPRESSION_GUIDANCE["smile"] == 5.0

    def test_laugh_guidance_value(self):
        """Laugh guidance_scale should be 5.0."""
        assert _EXPRESSION_GUIDANCE["laugh"] == 5.0

    def test_troubled_guidance_value(self):
        """Troubled guidance_scale should be 5.5."""
        assert _EXPRESSION_GUIDANCE["troubled"] == 5.5

    def test_surprised_guidance_value(self):
        """Surprised guidance_scale should be 5.0."""
        assert _EXPRESSION_GUIDANCE["surprised"] == 5.0

    def test_thinking_guidance_value(self):
        """Thinking guidance_scale should be 5.0."""
        assert _EXPRESSION_GUIDANCE["thinking"] == 5.0

    def test_embarrassed_guidance_value(self):
        """Embarrassed guidance_scale should be 5.5."""
        assert _EXPRESSION_GUIDANCE["embarrassed"] == 5.5

    def test_all_guidance_values_are_floats(self):
        """All guidance_scale values must be float type."""
        for expression, value in _EXPRESSION_GUIDANCE.items():
            assert isinstance(value, float), (
                f"Guidance for '{expression}' is {type(value)}, expected float"
            )

    def test_all_guidance_values_positive(self):
        """All guidance_scale values must be positive."""
        for expression, value in _EXPRESSION_GUIDANCE.items():
            assert value > 0, (
                f"Guidance for '{expression}' is {value}, expected positive"
            )

    @pytest.mark.parametrize("expression,expected", [
        ("neutral", 4.0),
        ("smile", 5.0),
        ("laugh", 5.0),
        ("troubled", 5.5),
        ("surprised", 5.0),
        ("thinking", 5.0),
        ("embarrassed", 5.5),
    ])
    def test_guidance_values_parametrized(self, expression: str, expected: float):
        """Verify each expression's guidance_scale matches the expected value."""
        assert _EXPRESSION_GUIDANCE[expression] == expected


# ── 3. generate_bustup_expression() Tests ───────────────────────────


class TestGenerateBustupExpression:
    """Tests for ImageGenPipeline.generate_bustup_expression()."""

    @patch("core.tools.image_gen.FluxKontextClient")
    def test_guidance_scale_passed_to_kontext(
        self, mock_kontext_cls: MagicMock, pipeline: ImageGenPipeline,
        fake_image_bytes: bytes,
    ):
        """generate_bustup_expression must pass the expression's guidance_scale
        to FluxKontextClient.generate_from_reference()."""
        mock_client = MagicMock()
        mock_client.generate_from_reference.return_value = b"RESULT_PNG"
        mock_kontext_cls.return_value = mock_client

        pipeline.generate_bustup_expression(
            reference_image=fake_image_bytes,
            expression="smile",
            skip_existing=False,
        )

        mock_client.generate_from_reference.assert_called_once()
        call_kwargs = mock_client.generate_from_reference.call_args
        assert call_kwargs.kwargs.get("guidance_scale") == 5.0 or \
            call_kwargs[1].get("guidance_scale") == 5.0, (
            "guidance_scale for 'smile' should be 5.0"
        )

    @patch("core.tools.image_gen.FluxKontextClient")
    def test_guidance_scale_for_neutral(
        self, mock_kontext_cls: MagicMock, pipeline: ImageGenPipeline,
        fake_image_bytes: bytes,
    ):
        """Neutral expression should use guidance_scale=4.0."""
        mock_client = MagicMock()
        mock_client.generate_from_reference.return_value = b"RESULT_PNG"
        mock_kontext_cls.return_value = mock_client

        pipeline.generate_bustup_expression(
            reference_image=fake_image_bytes,
            expression="neutral",
            skip_existing=False,
        )

        call_kwargs = mock_client.generate_from_reference.call_args
        actual_guidance = (
            call_kwargs.kwargs.get("guidance_scale")
            or call_kwargs[1].get("guidance_scale")
        )
        assert actual_guidance == 4.0

    @patch("core.tools.image_gen.FluxKontextClient")
    def test_guidance_scale_for_laugh(
        self, mock_kontext_cls: MagicMock, pipeline: ImageGenPipeline,
        fake_image_bytes: bytes,
    ):
        """Laugh expression should use guidance_scale=5.0."""
        mock_client = MagicMock()
        mock_client.generate_from_reference.return_value = b"RESULT_PNG"
        mock_kontext_cls.return_value = mock_client

        pipeline.generate_bustup_expression(
            reference_image=fake_image_bytes,
            expression="laugh",
            skip_existing=False,
        )

        call_kwargs = mock_client.generate_from_reference.call_args
        actual_guidance = (
            call_kwargs.kwargs.get("guidance_scale")
            or call_kwargs[1].get("guidance_scale")
        )
        assert actual_guidance == 5.0

    @pytest.mark.parametrize("expression", list(VALID_EMOTIONS))
    @patch("core.tools.image_gen.FluxKontextClient")
    def test_guidance_scale_matches_dict_for_all_expressions(
        self, mock_kontext_cls: MagicMock, expression: str,
        pipeline: ImageGenPipeline, fake_image_bytes: bytes,
    ):
        """Every valid expression must pass its corresponding guidance_scale."""
        mock_client = MagicMock()
        mock_client.generate_from_reference.return_value = b"RESULT_PNG"
        mock_kontext_cls.return_value = mock_client

        pipeline.generate_bustup_expression(
            reference_image=fake_image_bytes,
            expression=expression,
            skip_existing=False,
        )

        call_kwargs = mock_client.generate_from_reference.call_args
        actual_guidance = (
            call_kwargs.kwargs.get("guidance_scale")
            or call_kwargs[1].get("guidance_scale")
        )
        assert actual_guidance == _EXPRESSION_GUIDANCE[expression], (
            f"Expression '{expression}': expected guidance_scale="
            f"{_EXPRESSION_GUIDANCE[expression]}, got {actual_guidance}"
        )

    @patch("core.tools.image_gen.FluxKontextClient")
    def test_output_path_neutral(
        self, mock_kontext_cls: MagicMock, pipeline: ImageGenPipeline,
        fake_image_bytes: bytes,
    ):
        """Neutral expression should write to avatar_bustup.png (no suffix)."""
        mock_client = MagicMock()
        mock_client.generate_from_reference.return_value = b"RESULT_PNG"
        mock_kontext_cls.return_value = mock_client

        result = pipeline.generate_bustup_expression(
            reference_image=fake_image_bytes,
            expression="neutral",
            skip_existing=False,
        )

        assert result is not None
        assert result.name == "avatar_bustup.png"

    @patch("core.tools.image_gen.FluxKontextClient")
    def test_output_path_non_neutral(
        self, mock_kontext_cls: MagicMock, pipeline: ImageGenPipeline,
        fake_image_bytes: bytes,
    ):
        """Non-neutral expressions should write to avatar_bustup_{expression}.png."""
        mock_client = MagicMock()
        mock_client.generate_from_reference.return_value = b"RESULT_PNG"
        mock_kontext_cls.return_value = mock_client

        result = pipeline.generate_bustup_expression(
            reference_image=fake_image_bytes,
            expression="smile",
            skip_existing=False,
        )

        assert result is not None
        assert result.name == "avatar_bustup_smile.png"

    @patch("core.tools.image_gen.FluxKontextClient")
    def test_skip_existing_true(
        self, mock_kontext_cls: MagicMock, pipeline: ImageGenPipeline,
        fake_image_bytes: bytes,
    ):
        """When skip_existing=True and file exists, should skip generation."""
        # Pre-create the output file
        output_path = pipeline._assets_dir / "avatar_bustup_smile.png"
        output_path.write_bytes(b"EXISTING")

        result = pipeline.generate_bustup_expression(
            reference_image=fake_image_bytes,
            expression="smile",
            skip_existing=True,
        )

        assert result == output_path
        # FluxKontextClient should not have been instantiated
        mock_kontext_cls.assert_not_called()

    def test_unknown_expression_returns_none(
        self, pipeline: ImageGenPipeline, fake_image_bytes: bytes,
    ):
        """Unknown expression names should return None without calling the API."""
        result = pipeline.generate_bustup_expression(
            reference_image=fake_image_bytes,
            expression="angry",
            skip_existing=False,
        )
        assert result is None

    @patch("core.tools.image_gen.FluxKontextClient")
    def test_prompt_uses_expression_prompt(
        self, mock_kontext_cls: MagicMock, pipeline: ImageGenPipeline,
        fake_image_bytes: bytes,
    ):
        """The prompt passed to FluxKontextClient should match _EXPRESSION_PROMPTS."""
        mock_client = MagicMock()
        mock_client.generate_from_reference.return_value = b"RESULT_PNG"
        mock_kontext_cls.return_value = mock_client

        pipeline.generate_bustup_expression(
            reference_image=fake_image_bytes,
            expression="surprised",
            skip_existing=False,
        )

        call_kwargs = mock_client.generate_from_reference.call_args
        actual_prompt = (
            call_kwargs.kwargs.get("prompt")
            or call_kwargs[1].get("prompt")
        )
        assert actual_prompt == _EXPRESSION_PROMPTS["surprised"]

    @patch("core.tools.image_gen.FluxKontextClient")
    def test_aspect_ratio_is_3_4(
        self, mock_kontext_cls: MagicMock, pipeline: ImageGenPipeline,
        fake_image_bytes: bytes,
    ):
        """Bustup expression should always use 3:4 aspect ratio."""
        mock_client = MagicMock()
        mock_client.generate_from_reference.return_value = b"RESULT_PNG"
        mock_kontext_cls.return_value = mock_client

        pipeline.generate_bustup_expression(
            reference_image=fake_image_bytes,
            expression="thinking",
            skip_existing=False,
        )

        call_kwargs = mock_client.generate_from_reference.call_args
        actual_aspect = (
            call_kwargs.kwargs.get("aspect_ratio")
            or call_kwargs[1].get("aspect_ratio")
        )
        assert actual_aspect == "3:4"


# ── 4. generate_all() 2-Stage Pipeline Tests ───────────────────────


class TestGenerateAll2StagePipeline:
    """Tests for the 2-stage bust-up pipeline in generate_all()."""

    @patch("core.tools.image_gen.FluxKontextClient")
    @patch("core.tools.image_gen.NovelAIClient")
    def test_default_expression_list_is_all_seven(
        self, mock_novelai_cls: MagicMock, mock_kontext_cls: MagicMock,
        pipeline: ImageGenPipeline,
    ):
        """When expressions=None, all 7 expressions should be generated."""
        mock_novelai = MagicMock()
        mock_novelai.generate_fullbody.return_value = b"FULLBODY_PNG"
        mock_novelai_cls.return_value = mock_novelai

        mock_kontext = MagicMock()
        mock_kontext.generate_from_reference.return_value = b"EXPR_PNG"
        mock_kontext_cls.return_value = mock_kontext

        with patch.dict("os.environ", {"NOVELAI_TOKEN": "test"}):
            result = pipeline.generate_all(
                prompt="test prompt",
                skip_existing=False,
                steps=["fullbody", "bustup"],
            )

        # All 7 expressions should be in bustup_paths
        assert set(result.bustup_paths.keys()) == set(_EXPRESSION_PROMPTS.keys())

    @patch("core.tools.image_gen.FluxKontextClient")
    @patch("core.tools.image_gen.NovelAIClient")
    def test_neutral_generated_first_from_fullbody(
        self, mock_novelai_cls: MagicMock, mock_kontext_cls: MagicMock,
        pipeline: ImageGenPipeline,
    ):
        """Neutral should be generated first, using fullbody as reference."""
        fullbody_bytes = b"FULLBODY_PNG"
        neutral_bytes = b"NEUTRAL_BUSTUP_PNG"

        mock_novelai = MagicMock()
        mock_novelai.generate_fullbody.return_value = fullbody_bytes
        mock_novelai_cls.return_value = mock_novelai

        call_log: list[tuple[str, bytes]] = []

        mock_kontext = MagicMock()

        def track_generate(reference_image, prompt, **kwargs):
            # Determine which expression based on prompt content
            for expr, expr_prompt in _EXPRESSION_PROMPTS.items():
                if prompt.startswith(expr_prompt[:30]):
                    call_log.append((expr, reference_image))
                    break
            return b"RESULT_PNG"

        mock_kontext.generate_from_reference.side_effect = track_generate
        mock_kontext_cls.return_value = mock_kontext

        with patch.dict("os.environ", {"NOVELAI_TOKEN": "test"}):
            pipeline.generate_all(
                prompt="test prompt",
                skip_existing=False,
                steps=["fullbody", "bustup"],
                expressions=["neutral", "smile"],
            )

        # First call should be neutral with fullbody reference
        assert len(call_log) >= 1
        assert call_log[0][0] == "neutral"
        assert call_log[0][1] == fullbody_bytes

    @patch("core.tools.image_gen.FluxKontextClient")
    @patch("core.tools.image_gen.NovelAIClient")
    def test_other_expressions_use_neutral_bustup_as_reference(
        self, mock_novelai_cls: MagicMock, mock_kontext_cls: MagicMock,
        pipeline: ImageGenPipeline,
    ):
        """Non-neutral expressions should use the neutral bustup image as reference,
        not the original fullbody image."""
        fullbody_bytes = b"FULLBODY_PNG_DATA"
        neutral_result = b"NEUTRAL_BUSTUP_RESULT"

        mock_novelai = MagicMock()
        mock_novelai.generate_fullbody.return_value = fullbody_bytes
        mock_novelai_cls.return_value = mock_novelai

        call_references: list[bytes] = []
        call_count = [0]

        mock_kontext = MagicMock()

        def track_generate(reference_image, prompt, **kwargs):
            call_count[0] += 1
            call_references.append(reference_image)
            # Return a distinct result for neutral (first call)
            if call_count[0] == 1:
                return neutral_result
            return b"OTHER_EXPR_RESULT"

        mock_kontext.generate_from_reference.side_effect = track_generate
        mock_kontext_cls.return_value = mock_kontext

        with patch.dict("os.environ", {"NOVELAI_TOKEN": "test"}):
            pipeline.generate_all(
                prompt="test prompt",
                skip_existing=False,
                steps=["fullbody", "bustup"],
                expressions=["neutral", "smile", "laugh"],
            )

        # First call (neutral) should use fullbody as reference
        assert call_references[0] == fullbody_bytes

        # Subsequent calls (smile, laugh) should use neutral bustup as reference.
        # The neutral bustup is written to disk and then read back, so the
        # reference is the written bytes (neutral_result).
        for i in range(1, len(call_references)):
            assert call_references[i] == neutral_result, (
                f"Call {i} should use neutral bustup as reference, "
                f"not fullbody"
            )

    @patch("core.tools.image_gen.FluxKontextClient")
    @patch("core.tools.image_gen.NovelAIClient")
    def test_fallback_to_fullbody_when_neutral_fails(
        self, mock_novelai_cls: MagicMock, mock_kontext_cls: MagicMock,
        pipeline: ImageGenPipeline,
    ):
        """If neutral generation fails, other expressions should fall back
        to using the fullbody image as reference."""
        fullbody_bytes = b"FULLBODY_PNG_DATA"

        mock_novelai = MagicMock()
        mock_novelai.generate_fullbody.return_value = fullbody_bytes
        mock_novelai_cls.return_value = mock_novelai

        call_references: list[bytes] = []
        call_count = [0]

        mock_kontext = MagicMock()

        def track_generate(reference_image, prompt, **kwargs):
            call_count[0] += 1
            call_references.append(reference_image)
            # Neutral generation fails
            if call_count[0] == 1:
                raise RuntimeError("Flux Kontext failed for neutral")
            return b"EXPR_RESULT"

        mock_kontext.generate_from_reference.side_effect = track_generate
        mock_kontext_cls.return_value = mock_kontext

        with patch.dict("os.environ", {"NOVELAI_TOKEN": "test"}):
            result = pipeline.generate_all(
                prompt="test prompt",
                skip_existing=False,
                steps=["fullbody", "bustup"],
                expressions=["neutral", "smile", "laugh"],
            )

        # Neutral should have failed and logged an error
        assert any("neutral" in err for err in result.errors)

        # Remaining calls should use fullbody as reference (fallback)
        for i in range(1, len(call_references)):
            assert call_references[i] == fullbody_bytes, (
                f"Call {i} should fall back to fullbody reference "
                f"when neutral failed"
            )

    @patch("core.tools.image_gen.FluxKontextClient")
    @patch("core.tools.image_gen.NovelAIClient")
    def test_subset_expressions_supported(
        self, mock_novelai_cls: MagicMock, mock_kontext_cls: MagicMock,
        pipeline: ImageGenPipeline,
    ):
        """Specifying a subset of expressions should only generate those."""
        mock_novelai = MagicMock()
        mock_novelai.generate_fullbody.return_value = b"FULLBODY_PNG"
        mock_novelai_cls.return_value = mock_novelai

        mock_kontext = MagicMock()
        mock_kontext.generate_from_reference.return_value = b"EXPR_PNG"
        mock_kontext_cls.return_value = mock_kontext

        with patch.dict("os.environ", {"NOVELAI_TOKEN": "test"}):
            result = pipeline.generate_all(
                prompt="test prompt",
                skip_existing=False,
                steps=["fullbody", "bustup"],
                expressions=["neutral", "smile"],
            )

        assert set(result.bustup_paths.keys()) == {"neutral", "smile"}

    @patch("core.tools.image_gen.FluxKontextClient")
    @patch("core.tools.image_gen.NovelAIClient")
    def test_bustup_path_set_to_neutral(
        self, mock_novelai_cls: MagicMock, mock_kontext_cls: MagicMock,
        pipeline: ImageGenPipeline,
    ):
        """result.bustup_path should point to the neutral expression image."""
        mock_novelai = MagicMock()
        mock_novelai.generate_fullbody.return_value = b"FULLBODY_PNG"
        mock_novelai_cls.return_value = mock_novelai

        mock_kontext = MagicMock()
        mock_kontext.generate_from_reference.return_value = b"EXPR_PNG"
        mock_kontext_cls.return_value = mock_kontext

        with patch.dict("os.environ", {"NOVELAI_TOKEN": "test"}):
            result = pipeline.generate_all(
                prompt="test prompt",
                skip_existing=False,
                steps=["fullbody", "bustup"],
                expressions=["neutral", "smile"],
            )

        assert result.bustup_path is not None
        assert result.bustup_path.name == "avatar_bustup.png"

    @patch("core.tools.image_gen.FluxKontextClient")
    @patch("core.tools.image_gen.NovelAIClient")
    def test_expressions_without_neutral_still_work(
        self, mock_novelai_cls: MagicMock, mock_kontext_cls: MagicMock,
        pipeline: ImageGenPipeline,
    ):
        """When neutral is not in the expression list, should still generate
        requested expressions using fullbody as reference."""
        mock_novelai = MagicMock()
        mock_novelai.generate_fullbody.return_value = b"FULLBODY_PNG"
        mock_novelai_cls.return_value = mock_novelai

        mock_kontext = MagicMock()
        mock_kontext.generate_from_reference.return_value = b"EXPR_PNG"
        mock_kontext_cls.return_value = mock_kontext

        with patch.dict("os.environ", {"NOVELAI_TOKEN": "test"}):
            result = pipeline.generate_all(
                prompt="test prompt",
                skip_existing=False,
                steps=["fullbody", "bustup"],
                expressions=["smile", "laugh"],
            )

        assert "smile" in result.bustup_paths
        assert "laugh" in result.bustup_paths
        assert "neutral" not in result.bustup_paths

    @patch("core.tools.image_gen.FluxKontextClient")
    @patch("core.tools.image_gen.NovelAIClient")
    def test_existing_neutral_used_as_reference_when_not_requested(
        self, mock_novelai_cls: MagicMock, mock_kontext_cls: MagicMock,
        pipeline: ImageGenPipeline,
    ):
        """When neutral is not requested but avatar_bustup.png exists on disk,
        it should be used as reference for other expressions."""
        fullbody_bytes = b"FULLBODY_PNG_DATA"
        existing_neutral = b"EXISTING_NEUTRAL_BUSTUP"

        mock_novelai = MagicMock()
        mock_novelai.generate_fullbody.return_value = fullbody_bytes
        mock_novelai_cls.return_value = mock_novelai

        # Pre-create neutral bustup on disk
        neutral_path = pipeline._assets_dir / "avatar_bustup.png"
        neutral_path.write_bytes(existing_neutral)

        call_references: list[bytes] = []
        mock_kontext = MagicMock()

        def track_generate(reference_image, **kwargs):
            call_references.append(reference_image)
            return b"EXPR_RESULT"

        mock_kontext.generate_from_reference.side_effect = track_generate
        mock_kontext_cls.return_value = mock_kontext

        with patch.dict("os.environ", {"NOVELAI_TOKEN": "test"}):
            pipeline.generate_all(
                prompt="test prompt",
                skip_existing=False,
                steps=["fullbody", "bustup"],
                expressions=["smile"],
            )

        # Should use the existing neutral bustup as reference
        assert len(call_references) == 1
        assert call_references[0] == existing_neutral


# ── 5. EMOTION_INSTRUCTION Tests ───────────────────────────────────


class TestEmotionInstruction:
    """Tests for the EMOTION_INSTRUCTION constant in builder.py."""

    def test_emotion_instruction_importable(self):
        """EMOTION_INSTRUCTION should be importable from core.prompt.builder."""
        from core.prompt.builder import EMOTION_INSTRUCTION
        assert isinstance(EMOTION_INSTRUCTION, str)
        assert len(EMOTION_INSTRUCTION) > 0

    def test_contains_neutral_iga(self):
        """EMOTION_INSTRUCTION should contain 'neutral以外' to encourage
        using non-neutral expressions."""
        from core.prompt.builder import EMOTION_INSTRUCTION
        assert "neutral以外" in EMOTION_INSTRUCTION

    def test_contains_all_seven_emotion_names(self):
        """EMOTION_INSTRUCTION should mention all 7 emotion names."""
        from core.prompt.builder import EMOTION_INSTRUCTION
        for emotion in VALID_EMOTIONS:
            assert emotion in EMOTION_INSTRUCTION, (
                f"EMOTION_INSTRUCTION is missing emotion name: '{emotion}'"
            )

    def test_emotion_instruction_mentions_expression_metadata_format(self):
        """EMOTION_INSTRUCTION should contain the HTML comment format for emotion."""
        from core.prompt.builder import EMOTION_INSTRUCTION
        assert "<!-- emotion:" in EMOTION_INSTRUCTION

    def test_emotion_instruction_contains_emotion_json_key(self):
        """EMOTION_INSTRUCTION should reference the 'emotion' JSON key."""
        from core.prompt.builder import EMOTION_INSTRUCTION
        assert '"emotion"' in EMOTION_INSTRUCTION


# ── 6. Cross-Consistency Tests ──────────────────────────────────────


class TestCrossConsistency:
    """Tests verifying consistency between VALID_EMOTIONS, prompts, guidance,
    and the emotion instruction."""

    def test_prompts_and_guidance_keys_identical(self):
        """_EXPRESSION_PROMPTS and _EXPRESSION_GUIDANCE must have the same keys."""
        assert set(_EXPRESSION_PROMPTS.keys()) == set(_EXPRESSION_GUIDANCE.keys())

    def test_prompts_keys_match_valid_emotions(self):
        """_EXPRESSION_PROMPTS keys must match VALID_EMOTIONS."""
        assert set(_EXPRESSION_PROMPTS.keys()) == VALID_EMOTIONS

    def test_guidance_keys_match_valid_emotions(self):
        """_EXPRESSION_GUIDANCE keys must match VALID_EMOTIONS."""
        assert set(_EXPRESSION_GUIDANCE.keys()) == VALID_EMOTIONS

    def test_emotion_instruction_covers_valid_emotions(self):
        """EMOTION_INSTRUCTION should reference every emotion in VALID_EMOTIONS."""
        from core.prompt.builder import EMOTION_INSTRUCTION
        for emotion in VALID_EMOTIONS:
            assert emotion in EMOTION_INSTRUCTION

    def test_seven_emotions_count(self):
        """There should be exactly 7 valid emotions."""
        assert len(VALID_EMOTIONS) == 7
        assert len(_EXPRESSION_PROMPTS) == 7
        assert len(_EXPRESSION_GUIDANCE) == 7
