# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Live observation test for repeated voice-front monologues."""

from __future__ import annotations

import re
from pathlib import Path

import httpx
import pytest

from core.prompt.builder import build_voice_front_prompt
from core.voice.front import READ_MEMORY_TOOL, VoiceFrontLane
from core.voice.session import (
    MONOLOGUE_MAX_TOKENS,
    MONOLOGUE_TEMPERATURE,
    build_proactive_prompt,
    proactive_turn_uses_memory,
    read_memory_snippets,
)

pytestmark = [pytest.mark.e2e, pytest.mark.timeout(900)]

_API_BASE = "http://xserverng2:8000/v1"
_EMOTION_TAG = re.compile(r"\s*<!--\s*emotion:.*?-->\s*", re.DOTALL)


def _character_bigrams(text: str) -> set[str]:
    compact = "".join(text.split())
    if len(compact) < 2:
        return {compact} if compact else set()
    return {compact[index : index + 2] for index in range(len(compact) - 1)}


def _jaccard(left: str, right: str) -> float:
    left_grams = _character_bigrams(left)
    right_grams = _character_bigrams(right)
    union = left_grams | right_grams
    return len(left_grams & right_grams) / len(union) if union else 1.0


@pytest.mark.asyncio
async def test_voice_monologue_with_real_front_model() -> None:
    anima_dir = Path.home() / ".animaworks" / "animas" / "mei"
    if not anima_dir.is_dir():
        pytest.skip("~/.animaworks/animas/mei is unavailable")
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f"{_API_BASE}/models")
            response.raise_for_status()
    except (httpx.HTTPError, OSError) as exc:
        pytest.skip(f"voice front model endpoint is unavailable: {exc}")

    lane = VoiceFrontLane(
        model="openai/qwen3.6-35b-a3b",
        api_base=_API_BASE,
        system_prompt=build_voice_front_prompt(anima_dir, anima_name="mei"),
    )
    memory_queries: list[str] = []
    turn = {"count": 0}

    def _read_memory(args: dict) -> str:
        query = str(args.get("query", ""))
        memory_queries.append(query)
        # Same rotation the session applies: a new page per corner cycle.
        return read_memory_snippets(anima_dir, query, page=turn["count"] // 5)

    outputs: list[str] = []
    recent: list[str] = []
    for count in range(8):
        turn["count"] = count
        queries_before = len(memory_queries)
        output = "".join(
            [
                delta
                async for delta in lane.stream(
                    build_proactive_prompt(count, recent[-5:]),
                    tools=[READ_MEMORY_TOOL],
                    tool_executors={"read_memory": _read_memory},
                    tool_choice="required" if proactive_turn_uses_memory(count) else None,
                    keep_history=False,
                    max_tokens=MONOLOGUE_MAX_TOKENS,
                    temperature=MONOLOGUE_TEMPERATURE,
                )
            ]
        )
        outputs.append(output)
        body = _EMOTION_TAG.sub("", output).strip()
        if body:
            recent.append(body[:60])
        used = memory_queries[queries_before:]
        print(f"\n===== voice monologue turn {count} (read_memory={used}) =====\n{output}")

    bodies = [_EMOTION_TAG.sub("", output).strip() for output in outputs]
    assert all(outputs), "one or more monologue turns were empty"
    assert all(len(body) <= 200 for body in bodies), [len(body) for body in bodies]
    for left_index in range(len(bodies)):
        for right_index in range(left_index + 1, len(bodies)):
            similarity = _jaccard(bodies[left_index], bodies[right_index])
            assert similarity < 0.5, f"turns {left_index} and {right_index} repeated too closely: {similarity:.3f}"
    expected_memory_turns = sum(proactive_turn_uses_memory(count) for count in range(8))
    assert len(memory_queries) >= expected_memory_turns, (memory_queries, expected_memory_turns)
