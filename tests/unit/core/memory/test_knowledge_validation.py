from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.


"""Tests for NLI + LLM cascade knowledge validation."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.memory.validation import KnowledgeValidator


# ── Fixtures ────────────────────────────────────────────────


@pytest.fixture
def validator() -> KnowledgeValidator:
    """Create a KnowledgeValidator with NLI disabled."""
    v = KnowledgeValidator()
    v._nli_available = False  # Force LLM-only mode for most tests
    return v


@pytest.fixture
def validator_with_nli() -> KnowledgeValidator:
    """Create a KnowledgeValidator with a mock NLI pipeline."""
    v = KnowledgeValidator()
    v._nli_pipeline = MagicMock()
    v._nli_available = True
    return v


# ── NLI Check Tests ──────────────────────────────────────────


class TestNLICheck:
    """Test the NLI classification stage."""

    def test_nli_unavailable_returns_neutral(self, validator: KnowledgeValidator) -> None:
        """When NLI is unavailable, return neutral with score 0."""
        label, score = validator._nli_check("hypothesis", "premise")
        assert label == "neutral"
        assert score == 0.0

    def test_nli_entailment(self, validator_with_nli: KnowledgeValidator) -> None:
        """NLI pipeline returning entailment is surfaced correctly."""
        validator_with_nli._nli_pipeline.return_value = [
            {"label": "entailment", "score": 0.92}
        ]

        label, score = validator_with_nli._nli_check(
            "The server runs on port 8080",
            "We configured the server to use port 8080",
        )

        assert label == "entailment"
        assert score == 0.92

    def test_nli_contradiction(self, validator_with_nli: KnowledgeValidator) -> None:
        """NLI pipeline returning contradiction is surfaced correctly."""
        validator_with_nli._nli_pipeline.return_value = [
            {"label": "contradiction", "score": 0.85}
        ]

        label, score = validator_with_nli._nli_check(
            "The server runs on port 3000",
            "We configured the server to use port 8080",
        )

        assert label == "contradiction"
        assert score == 0.85

    def test_nli_neutral(self, validator_with_nli: KnowledgeValidator) -> None:
        """NLI pipeline returning neutral is surfaced correctly."""
        validator_with_nli._nli_pipeline.return_value = [
            {"label": "neutral", "score": 0.55}
        ]

        label, score = validator_with_nli._nli_check(
            "The weather is sunny",
            "We had a meeting about deployment",
        )

        assert label == "neutral"
        assert score == 0.55

    def test_nli_exception_returns_neutral(
        self, validator_with_nli: KnowledgeValidator,
    ) -> None:
        """NLI pipeline exceptions are caught and return neutral."""
        validator_with_nli._nli_pipeline.side_effect = RuntimeError("model error")

        label, score = validator_with_nli._nli_check("h", "p")
        assert label == "neutral"
        assert score == 0.0


# ── Validate Tests (Full Pipeline) ───────────────────────────


class TestValidate:
    """Test the full validate() method."""

    @pytest.mark.asyncio
    async def test_nli_entailment_accepted(
        self, validator_with_nli: KnowledgeValidator,
    ) -> None:
        """Items with NLI entailment above threshold get confidence=0.9."""
        validator_with_nli._nli_pipeline.return_value = [
            {"label": "entailment", "score": 0.85}
        ]

        items = [{"content": "The API key is stored in config.json"}]
        episodes = "We stored the API key in config.json for security."

        results = await validator_with_nli.validate(items, episodes)

        assert len(results) == 1
        assert results[0]["confidence"] == 0.9

    @pytest.mark.asyncio
    async def test_nli_contradiction_rejected(
        self, validator_with_nli: KnowledgeValidator,
    ) -> None:
        """Items with NLI contradiction above threshold are rejected."""
        validator_with_nli._nli_pipeline.return_value = [
            {"label": "contradiction", "score": 0.80}
        ]

        items = [{"content": "The server uses PostgreSQL"}]
        episodes = "We decided to use MySQL for the database."

        results = await validator_with_nli.validate(items, episodes)

        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_neutral_falls_through_to_llm(
        self, validator_with_nli: KnowledgeValidator,
    ) -> None:
        """Neutral NLI results trigger LLM review."""
        validator_with_nli._nli_pipeline.return_value = [
            {"label": "neutral", "score": 0.45}
        ]

        items = [{"content": "The team uses agile methodology"}]
        episodes = "Sprint planning was held on Monday."

        with patch.object(
            validator_with_nli, "_llm_review", new_callable=AsyncMock,
        ) as mock_review:
            mock_review.return_value = True
            results = await validator_with_nli.validate(items, episodes)

        assert len(results) == 1
        assert results[0]["confidence"] == 0.7
        mock_review.assert_called_once()

    @pytest.mark.asyncio
    async def test_llm_review_rejection(
        self, validator_with_nli: KnowledgeValidator,
    ) -> None:
        """Items rejected by LLM review are excluded."""
        validator_with_nli._nli_pipeline.return_value = [
            {"label": "neutral", "score": 0.40}
        ]

        items = [{"content": "Hallucinated fact"}]
        episodes = "Real episode content."

        with patch.object(
            validator_with_nli, "_llm_review", new_callable=AsyncMock,
        ) as mock_review:
            mock_review.return_value = False
            results = await validator_with_nli.validate(items, episodes)

        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_multiple_items_mixed_results(
        self, validator_with_nli: KnowledgeValidator,
    ) -> None:
        """Multiple items: some accepted, some rejected."""
        # Configure NLI to return different results for each call
        validator_with_nli._nli_pipeline.side_effect = [
            [{"label": "entailment", "score": 0.90}],
            [{"label": "contradiction", "score": 0.80}],
            [{"label": "neutral", "score": 0.50}],
        ]

        items = [
            {"content": "Fact A"},
            {"content": "Wrong B"},
            {"content": "Maybe C"},
        ]
        episodes = "Source episodes text."

        with patch.object(
            validator_with_nli, "_llm_review", new_callable=AsyncMock,
        ) as mock_review:
            mock_review.return_value = True
            results = await validator_with_nli.validate(items, episodes)

        assert len(results) == 2  # A accepted, B rejected, C accepted
        assert results[0]["content"] == "Fact A"
        assert results[0]["confidence"] == 0.9
        assert results[1]["content"] == "Maybe C"
        assert results[1]["confidence"] == 0.7

    @pytest.mark.asyncio
    async def test_empty_content_skipped(
        self, validator_with_nli: KnowledgeValidator,
    ) -> None:
        """Items with empty content are skipped."""
        items = [{"content": ""}, {"content": "  "}]
        episodes = "Some source text."

        # The second item has whitespace but not empty via `.get()`;
        # the first is genuinely empty and should be skipped
        validator_with_nli._nli_pipeline.return_value = [
            {"label": "entailment", "score": 0.90}
        ]

        results = await validator_with_nli.validate(items, episodes)

        # Only the whitespace item gets through (it's truthy after get())
        assert len(results) <= 1

    @pytest.mark.asyncio
    async def test_nli_only_mode(self, validator: KnowledgeValidator) -> None:
        """When NLI is unavailable, all items go to LLM review."""
        items = [{"content": "Knowledge item"}]
        episodes = "Episode text."

        with patch.object(
            validator, "_llm_review", new_callable=AsyncMock,
        ) as mock_review:
            mock_review.return_value = True
            results = await validator.validate(items, episodes)

        assert len(results) == 1
        assert results[0]["confidence"] == 0.7
        mock_review.assert_called_once()


# ── LLM Review Tests ─────────────────────────────────────────


class TestLLMReview:
    """Test the _llm_review fallback method."""

    @pytest.mark.asyncio
    async def test_valid_response(self, validator: KnowledgeValidator) -> None:
        """LLM returning valid=true passes the item."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"valid": true, "reason": "OK"}'

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response
            result = await validator._llm_review(
                "knowledge text", "episode text", "test-model",
            )

        assert result is True

    @pytest.mark.asyncio
    async def test_invalid_response(self, validator: KnowledgeValidator) -> None:
        """LLM returning valid=false rejects the item."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = (
            '{"valid": false, "reason": "Not derivable from episodes"}'
        )

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response
            result = await validator._llm_review(
                "hallucinated text", "episode text", "test-model",
            )

        assert result is False

    @pytest.mark.asyncio
    async def test_json_embedded_in_text(self, validator: KnowledgeValidator) -> None:
        """JSON embedded in natural language text is extracted correctly."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = (
            'Based on my analysis, the result is: {"valid": true, "reason": "Correct"}'
        )

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response
            result = await validator._llm_review(
                "knowledge", "episodes", "test-model",
            )

        assert result is True

    @pytest.mark.asyncio
    async def test_llm_failure_returns_true(
        self, validator: KnowledgeValidator,
    ) -> None:
        """LLM call failure conservatively passes the item."""
        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = RuntimeError("API error")
            result = await validator._llm_review(
                "knowledge", "episodes", "test-model",
            )

        assert result is True  # Conservative: allow on failure

    @pytest.mark.asyncio
    async def test_unparseable_response_returns_true(
        self, validator: KnowledgeValidator,
    ) -> None:
        """Unparseable LLM response conservatively passes the item."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "I can't decide."

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response
            result = await validator._llm_review(
                "knowledge", "episodes", "test-model",
            )

        assert result is True


# ── NLI Model Loading ─────────────────────────────────────────


class TestNLIModelLoading:
    """Test NLI model loading behavior."""

    def test_load_failure_disables_nli(self) -> None:
        """When transformers is unavailable, NLI is disabled gracefully."""
        v = KnowledgeValidator()

        with patch.dict("sys.modules", {"transformers": None}):
            with patch("builtins.__import__", side_effect=ImportError("no transformers")):
                v._load_nli_model()

        assert v._nli_available is False
        assert v._nli_pipeline is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
