from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from core.lifecycle.knowledge_correction import (
    KnowledgeCorrectionLimits,
    run_post_consolidation_knowledge_correction,
)
from core.memory.contradiction import ContradictionPair


@pytest.fixture
def anima_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    data_dir = tmp_path / ".animaworks"
    monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(data_dir))
    for sub in ("shared", "common_knowledge", "common_skills", "company"):
        (data_dir / sub).mkdir(parents=True, exist_ok=True)
    anima = data_dir / "animas" / "test_anima"
    for sub in ("knowledge", "episodes", "procedures", "skills", "state", "activity_log", "archive"):
        (anima / sub).mkdir(parents=True, exist_ok=True)
    return anima


def _write_knowledge(anima_dir: Path, name: str, body: str, meta: dict) -> Path:
    path = anima_dir / "knowledge" / name
    fm = yaml.dump(meta, default_flow_style=False, allow_unicode=True)
    path.write_text(f"---\n{fm}---\n\n{body}", encoding="utf-8")
    return path


@pytest.mark.asyncio
async def test_post_consolidation_correction_supersedes_recent_knowledge(
    anima_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_path = _write_knowledge(
        anima_dir,
        "old-policy.md",
        "Deploy to the legacy host.",
        {"created_at": "2026-01-01T00:00:00", "failure_count": 0, "success_count": 0},
    )
    new_path = _write_knowledge(
        anima_dir,
        "new-policy.md",
        "Deploy to the new host.",
        {"created_at": "2026-06-10T00:00:00", "failure_count": 0, "success_count": 0},
    )
    scan_calls = 0

    async def fake_scan(self, target_file=None, model="", target_files=None, max_llm_checks=None):  # noqa: ANN001, ARG001
        nonlocal scan_calls
        scan_calls += 1
        self.last_scan_stats = {"candidate_pairs": 1, "llm_checks": 1, "limit_reached": False}
        if target_files and new_path in target_files:
            return [
                ContradictionPair(
                    file_a=old_path,
                    file_b=new_path,
                    text_a="Deploy to the legacy host.",
                    text_b="Deploy to the new host.",
                    confidence=0.9,
                    resolution="supersede",
                    reason="new policy replaces old",
                )
            ]
        return []

    monkeypatch.setattr("core.memory.contradiction.ContradictionDetector.scan_contradictions", fake_scan)

    summary = await run_post_consolidation_knowledge_correction(
        anima_dir,
        "test_anima",
        model="test-model",
        limits=KnowledgeCorrectionLimits(
            max_contradiction_pairs=20,
            max_reconsolidation_files=5,
            timeout_seconds=5,
            recent_hours=1,
        ),
    )

    archived = anima_dir / "archive" / "superseded" / "old-policy.md"
    assert archived.exists()
    assert not old_path.exists()

    from core.memory.manager import MemoryManager

    meta = MemoryManager(anima_dir).read_knowledge_metadata(archived)
    assert meta["superseded_by"] == "new-policy.md"
    assert meta["valid_until"]
    assert summary["contradiction"]["detected"] == 1
    assert summary["contradiction"]["resolved"]["superseded"] == 1
    assert scan_calls == 1


@pytest.mark.asyncio
async def test_post_consolidation_reconsolidates_failing_knowledge_with_file_cap(
    anima_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _write_knowledge(
        anima_dir,
        "failing-a.md",
        "Incorrect procedure detail A.",
        {"failure_count": 3, "confidence": 0.2, "success_count": 0, "version": 1},
    )
    second = _write_knowledge(
        anima_dir,
        "failing-b.md",
        "Incorrect procedure detail B.",
        {"failure_count": 4, "confidence": 0.1, "success_count": 0, "version": 1},
    )

    async def fake_revise(self, content, meta, model):  # noqa: ANN001, ARG001
        return f"Revised: {content}"

    monkeypatch.setattr("core.memory.reconsolidation.ReconsolidationEngine._revise_knowledge", fake_revise)

    summary = await run_post_consolidation_knowledge_correction(
        anima_dir,
        "test_anima",
        model="test-model",
        limits=KnowledgeCorrectionLimits(
            max_contradiction_pairs=0,
            max_reconsolidation_files=1,
            timeout_seconds=5,
            recent_hours=1,
        ),
    )

    from core.memory.manager import MemoryManager

    mm = MemoryManager(anima_dir)
    first_meta = mm.read_knowledge_metadata(first)
    second_meta = mm.read_knowledge_metadata(second)
    assert summary["reconsolidation"]["knowledge"]["targets_found"] == 1
    assert summary["reconsolidation"]["knowledge"]["updated"] == 1
    assert first_meta["failure_count"] == 0
    assert first_meta["version"] == 2
    assert second_meta["failure_count"] == 4


@pytest.mark.asyncio
async def test_post_consolidation_timeout_returns_partial_summary(
    anima_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.lifecycle import knowledge_correction

    async def fast_contradiction(*args, **kwargs):  # noqa: ANN002, ANN003, ARG001
        return None

    async def slow_reconsolidation(*args, **kwargs):  # noqa: ANN002, ANN003, ARG001
        await asyncio.sleep(0.2)

    monkeypatch.setattr(knowledge_correction, "_run_contradiction_stage", fast_contradiction)
    monkeypatch.setattr(knowledge_correction, "_run_reconsolidation_stage", slow_reconsolidation)

    summary = await run_post_consolidation_knowledge_correction(
        anima_dir,
        "test_anima",
        model="test-model",
        limits=KnowledgeCorrectionLimits(timeout_seconds=0.01),
    )

    assert summary["timed_out"] is True


@pytest.mark.asyncio
async def test_system_consolidation_helper_uses_configured_limits(tmp_path: Path) -> None:
    from core.lifecycle.system_consolidation import SystemConsolidationMixin

    anima = MagicMock()
    anima.memory.anima_dir = tmp_path
    cfg = MagicMock(
        knowledge_self_correction_enabled=True,
        knowledge_self_correction_max_contradiction_pairs=7,
        knowledge_self_correction_max_reconsolidation_files=3,
        knowledge_self_correction_timeout_seconds=11,
        knowledge_self_correction_recent_hours=6,
        contradiction_batch_size=4,
        contradiction_nli_prefilter_threshold=0.93,
    )

    with patch(
        "core.lifecycle.knowledge_correction.run_post_consolidation_knowledge_correction",
        new_callable=AsyncMock,
        return_value={"ok": True},
    ) as mock_run:
        await SystemConsolidationMixin._run_knowledge_self_correction_if_enabled(
            anima,
            "test_anima",
            cfg,
            model="test-model",
        )

    _, _, kwargs = mock_run.mock_calls[0]
    assert kwargs["model"] == "test-model"
    assert kwargs["limits"].max_contradiction_pairs == 7
    assert kwargs["limits"].max_reconsolidation_files == 3
    assert kwargs["limits"].timeout_seconds == 11
    assert kwargs["limits"].recent_hours == 6
    assert kwargs["limits"].contradiction_batch_size == 4
    assert kwargs["limits"].contradiction_nli_prefilter_threshold == 0.93
