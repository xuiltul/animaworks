"""Tests for extract_facts reference_time parameter (Issue #174, Measure C)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.memory.extraction.extractor import FactExtractor


class TestExtractFactsReferenceTime:
    """Verify reference_time flows through extract_facts."""

    @pytest.mark.asyncio
    async def test_reference_time_passed_to_prompt(self) -> None:
        extractor = FactExtractor(model="test-model")
        llm_response = json.dumps({"facts": []})

        captured_prompt: list[str] = []

        async def capture_llm(_sys: str, user: str) -> str:
            captured_prompt.append(user)
            return llm_response

        entity = MagicMock(spec=["name", "model_dump"])
        entity.name = "Alice"
        entity.model_dump = lambda mode="json": {
            "name": "Alice",
            "entity_type": "Person",
            "summary": "",
        }

        with patch.object(extractor, "_call_llm", side_effect=capture_llm):
            await extractor.extract_facts(
                "Alice went to the park",
                [entity],
                reference_time="2023-05-08T10:00:00",
            )

        assert len(captured_prompt) == 1
        assert "2023-05-08T10:00:00" in captured_prompt[0]

    @pytest.mark.asyncio
    async def test_reference_time_defaults_to_now_iso(self) -> None:
        extractor = FactExtractor(model="test-model")
        llm_response = json.dumps({"facts": []})

        captured_prompt: list[str] = []

        async def capture_llm(_sys: str, user: str) -> str:
            captured_prompt.append(user)
            return llm_response

        entity = MagicMock(spec=["name", "model_dump"])
        entity.name = "Alice"
        entity.model_dump = lambda mode="json": {
            "name": "Alice",
            "entity_type": "Person",
            "summary": "",
        }

        with (
            patch.object(extractor, "_call_llm", side_effect=capture_llm),
            patch(
                "core.memory.extraction.extractor.now_iso",
                return_value="2026-01-01T00:00:00+09:00",
            ),
        ):
            await extractor.extract_facts("Alice went to the park", [entity])

        assert len(captured_prompt) == 1
        assert "2026-01-01T00:00:00+09:00" in captured_prompt[0]

    @pytest.mark.asyncio
    async def test_reference_time_none_falls_back(self) -> None:
        extractor = FactExtractor(model="test-model")
        llm_response = json.dumps({"facts": []})

        captured_prompt: list[str] = []

        async def capture_llm(_sys: str, user: str) -> str:
            captured_prompt.append(user)
            return llm_response

        entity = MagicMock(spec=["name", "model_dump"])
        entity.name = "Alice"
        entity.model_dump = lambda mode="json": {
            "name": "Alice",
            "entity_type": "Person",
            "summary": "",
        }

        with (
            patch.object(extractor, "_call_llm", side_effect=capture_llm),
            patch(
                "core.memory.extraction.extractor.now_iso",
                return_value="2026-02-02T12:00:00+09:00",
            ),
        ):
            await extractor.extract_facts(
                "Alice went to the park",
                [entity],
                reference_time=None,
            )

        assert len(captured_prompt) == 1
        assert "2026-02-02T12:00:00+09:00" in captured_prompt[0]
