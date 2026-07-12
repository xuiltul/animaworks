from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from core.lifecycle.system_consolidation import (
    run_weekly_full_contradiction_scan,
    run_weekly_pattern_distillation,
)


@pytest.mark.asyncio
async def test_weekly_pattern_distillation_calls_distiller(monkeypatch, tmp_path: Path) -> None:
    calls: dict[str, object] = {}

    class FakeDistiller:
        def __init__(self, anima_dir: Path, anima_name: str) -> None:
            calls["init"] = (anima_dir, anima_name)

        async def weekly_pattern_distill(self, *, model: str, days: int) -> dict[str, object]:
            calls["distill"] = {"model": model, "days": days}
            return {"procedures_created": ["procedures/runbook.md"], "patterns_detected": 1}

    monkeypatch.setattr("core.memory.distillation.ProceduralDistiller", FakeDistiller)

    await run_weekly_pattern_distillation(tmp_path / "animas" / "sakura", "sakura", model="test-model")

    assert calls["init"] == (tmp_path / "animas" / "sakura", "sakura")
    assert calls["distill"] == {"model": "test-model", "days": 7}


@pytest.mark.asyncio
async def test_weekly_full_contradiction_scan_uses_full_scan_limit(monkeypatch, tmp_path: Path) -> None:
    calls: dict[str, object] = {}

    class FakeDetector:
        last_scan_stats = {"candidate_pairs": 2, "llm_checks": 2, "limit_reached": False}

        def __init__(self, *args, **kwargs) -> None:
            calls["init_args"] = args
            calls["init_kwargs"] = kwargs

        async def scan_contradictions(self, *, model: str, max_llm_checks: int):
            calls["scan"] = {"model": model, "max_llm_checks": max_llm_checks}
            return [object(), object()]

        async def resolve_contradictions(self, pairs, model: str) -> int:
            calls["resolve"] = {"pairs": len(pairs), "model": model}
            return len(pairs)

    monkeypatch.setattr("core.memory.contradiction.ContradictionDetector", FakeDetector)

    await run_weekly_full_contradiction_scan(
        tmp_path / "animas" / "sakura",
        "sakura",
        SimpleNamespace(
            weekly_full_contradiction_max_pairs=7,
            contradiction_batch_size=5,
            contradiction_nli_prefilter_threshold=0.91,
        ),
        model="test-model",
    )

    assert calls["scan"] == {"model": "test-model", "max_llm_checks": 7}
    assert calls["resolve"] == {"pairs": 2, "model": "test-model"}
    assert calls["init_kwargs"]["batch_size"] == 5
    assert calls["init_kwargs"]["nli_prefilter_threshold"] == 0.91
