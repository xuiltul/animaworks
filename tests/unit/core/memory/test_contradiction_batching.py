from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.config.schemas import ConsolidationConfig
from core.memory.contradiction import ContradictionDetector, ContradictionResult


@pytest.fixture
def detector(tmp_path: Path) -> ContradictionDetector:
    anima_dir = tmp_path / "animas" / "test"
    (anima_dir / "knowledge").mkdir(parents=True)
    return ContradictionDetector(anima_dir, "test", nli_prefilter_threshold=None)


def _candidates(count: int) -> list[tuple[Path, str, Path, str]]:
    return [(Path(f"a-{index}.md"), f"a {index}", Path(f"b-{index}.md"), f"b {index}") for index in range(count)]


def test_contradiction_config_defaults_preserve_entailment_filter() -> None:
    config = ConsolidationConfig()
    assert config.contradiction_batch_size == 20
    assert config.contradiction_nli_prefilter_threshold == 0.70


@pytest.mark.asyncio
async def test_batch_prompt_and_json_array_parse(detector: ContradictionDetector) -> None:
    response = (
        '[{"pair_id": 0, "is_contradiction": true, "resolution": "supersede", '
        '"reason": "newer", "merged_content": null}, '
        '{"pair_id": 1, "is_contradiction": false, "resolution": "coexist", '
        '"reason": "consistent", "merged_content": null}]'
    )
    with patch(
        "core.memory._llm_utils.one_shot_completion",
        new_callable=AsyncMock,
        return_value=response,
    ) as completion:
        results = await detector._check_contradiction_llm_batch(
            [("a", "b", "a.md", "b.md"), ("c", "d", "c.md", "d.md")],
            "test-model",
        )

    assert results is not None
    assert [result.is_contradiction for result in results] == [True, False]
    prompt = completion.await_args.args[0]
    assert '"pair_id": 0' in prompt
    assert '"pair_id": 1' in prompt


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid_result",
    [
        {
            "pair_id": 0,
            "is_contradiction": "false",
            "resolution": "coexist",
            "reason": "consistent",
            "merged_content": None,
        },
        {
            "pair_id": 0,
            "is_contradiction": False,
            "resolution": "coexist",
            "merged_content": None,
        },
        {
            "pair_id": 0,
            "is_contradiction": False,
            "resolution": "ignore",
            "reason": "consistent",
            "merged_content": None,
        },
        {
            "pair_id": True,
            "is_contradiction": False,
            "resolution": "coexist",
            "reason": "consistent",
            "merged_content": None,
        },
        {
            "pair_id": 0,
            "is_contradiction": 0,
            "resolution": "coexist",
            "reason": "consistent",
            "merged_content": None,
        },
        {
            "pair_id": 0,
            "is_contradiction": False,
            "resolution": "coexist",
            "reason": None,
            "merged_content": None,
        },
        {
            "pair_id": 0,
            "is_contradiction": False,
            "resolution": "coexist",
            "reason": "consistent",
            "merged_content": 123,
        },
    ],
    ids=[
        "string-false",
        "missing-reason",
        "invalid-resolution",
        "bool-pair-id",
        "integer-boolean",
        "invalid-reason",
        "invalid-merged-content",
    ],
)
async def test_batch_response_rejects_missing_or_invalid_fields(
    detector: ContradictionDetector,
    invalid_result: dict[str, object],
) -> None:
    with patch(
        "core.memory._llm_utils.one_shot_completion",
        new_callable=AsyncMock,
        return_value=json.dumps([invalid_result]),
    ):
        results = await detector._check_contradiction_llm_batch(
            [("a", "b", "a.md", "b.md")],
            "test-model",
        )

    assert results is None


@pytest.mark.asyncio
async def test_bad_batch_output_falls_back_per_pair(detector: ContradictionDetector) -> None:
    candidates = _candidates(2)
    with (
        patch.object(detector, "_find_candidate_pairs", return_value=candidates),
        patch.object(detector, "_check_contradiction_nli", new_callable=AsyncMock, return_value=(False, 0.0, False)),
        patch.object(detector, "_check_contradiction_llm_batch", new_callable=AsyncMock, return_value=None),
        patch.object(
            detector,
            "_check_contradiction_llm",
            new_callable=AsyncMock,
            return_value=ContradictionResult(False, "coexist", "none"),
        ) as single_check,
    ):
        await detector.scan_contradictions(model="test-model")

    assert single_check.await_count == 2
    assert detector.last_scan_stats["llm_checks"] == 2
    assert detector.last_scan_stats["llm_calls"] == 3


@pytest.mark.asyncio
async def test_batch_size_one_uses_legacy_path(tmp_path: Path) -> None:
    detector = ContradictionDetector(tmp_path, "test", batch_size=1, nli_prefilter_threshold=None)
    with (
        patch.object(detector, "_find_candidate_pairs", return_value=_candidates(2)),
        patch.object(detector, "_check_contradiction_nli", new_callable=AsyncMock, return_value=(False, 0.0, False)),
        patch.object(detector, "_check_contradiction_llm_batch", new_callable=AsyncMock) as batch_check,
        patch.object(
            detector,
            "_check_contradiction_llm",
            new_callable=AsyncMock,
            return_value=ContradictionResult(False, "coexist", "none"),
        ) as single_check,
    ):
        await detector.scan_contradictions(model="test-model")

    batch_check.assert_not_awaited()
    assert single_check.await_count == 2
    assert detector.last_scan_stats["llm_calls"] == 2


@pytest.mark.asyncio
async def test_nli_threshold_filters_neutral_and_entailment(tmp_path: Path) -> None:
    detector = ContradictionDetector(tmp_path, "test", nli_prefilter_threshold=0.9)
    nli = MagicMock()
    nli.check.side_effect = [("neutral", 0.95), ("entailment", 0.91)]
    detector._nli_model = nli

    is_contradiction, _, prefiltered = await detector._check_contradiction_nli("a", "b")

    assert is_contradiction is False
    assert prefiltered is True


@pytest.mark.asyncio
async def test_none_disables_nli_prefilter(detector: ContradictionDetector) -> None:
    nli = MagicMock()
    nli.check.return_value = ("entailment", 0.99)
    detector._nli_model = nli

    is_contradiction, _, prefiltered = await detector._check_contradiction_nli("a", "b")

    assert is_contradiction is False
    assert prefiltered is False


@pytest.mark.asyncio
async def test_unavailable_nli_gracefully_reaches_llm(detector: ContradictionDetector) -> None:
    candidates = _candidates(1)
    with (
        patch.object(detector, "_find_candidate_pairs", return_value=candidates),
        patch.object(detector, "_get_nli_model", return_value=None),
        patch.object(
            detector,
            "_check_contradiction_llm",
            new_callable=AsyncMock,
            return_value=ContradictionResult(False, "coexist", "none"),
        ) as single_check,
    ):
        await detector.scan_contradictions(model="test-model")

    single_check.assert_awaited_once()


@pytest.mark.asyncio
async def test_multiple_targets_are_combined_and_deduplicated(detector: ContradictionDetector) -> None:
    shared = (Path("a.md"), "a", Path("b.md"), "b")
    unique = (Path("a.md"), "a", Path("c.md"), "c")
    find_results = [[shared], [shared, unique]]
    no_contradiction = [ContradictionResult(False, "coexist", "none")] * 2
    with (
        patch.object(detector, "_find_candidate_pairs", side_effect=find_results),
        patch.object(detector, "_check_contradiction_nli", new_callable=AsyncMock, return_value=(False, 0.0, False)),
        patch.object(
            detector,
            "_check_contradiction_llm_batch",
            new_callable=AsyncMock,
            return_value=no_contradiction,
        ) as batch_check,
    ):
        await detector.scan_contradictions(
            target_files=[Path("target-a.md"), Path("target-b.md")],
            model="test-model",
        )

    batch_check.assert_awaited_once()
    assert detector.last_scan_stats["candidate_pairs"] == 2
    assert detector.last_scan_stats["llm_calls"] == 1
