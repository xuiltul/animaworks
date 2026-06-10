from __future__ import annotations

from pathlib import Path

import pytest

from core.memory import fact_extraction
from core.memory.consolidation import ConsolidationEngine
from core.memory.conversation_finalize import _extract_session_facts_nonfatal
from core.memory.fact_extraction import FactExtractionOutcome
from core.memory.fact_observability import reset_warning_rate_limits
from core.memory.facts import FactRecord


def _fact() -> FactRecord:
    return FactRecord(
        text="Alice tracks LoCoMo memory scores.",
        source_entity="Alice",
        target_entity="LoCoMo",
        edge_type="TRACKS",
        recorded_at="2026-06-03T10:00:00+09:00",
    )


@pytest.mark.asyncio
@pytest.mark.unit
async def test_session_fact_extraction_completion_log_includes_counters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def fake_extract(*args, **kwargs):
        return FactExtractionOutcome([_fact()])

    monkeypatch.setattr("core.memory.fact_extraction.extract_and_store_facts_with_outcome", fake_extract)

    with caplog.at_level("INFO", logger="animaworks.conversation_memory"):
        result = await _extract_session_facts_nonfatal(
            tmp_path / "alice",
            "Alice tracks LoCoMo memory scores.",
            source_episode="episodes/2026-06-03.md",
            source_session_id="session-1",
            reference_time="2026-06-03T10:00:00+09:00",
        )

    assert result == (1, 0)
    assert "facts_extracted=1 facts_failed=0" in caplog.text


@pytest.mark.asyncio
@pytest.mark.unit
async def test_consolidation_fact_extraction_completion_log_includes_counters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def fake_extract(*args, **kwargs):
        return FactExtractionOutcome([], True, "extract", "failed")

    monkeypatch.setattr("core.memory.fact_extraction.extract_and_store_facts_with_outcome", fake_extract)
    engine = ConsolidationEngine(tmp_path / "alice", "alice")

    with caplog.at_level("INFO", logger="animaworks.consolidation"):
        outcome = await engine.extract_facts_from_text_outcome(
            "Alice tracks LoCoMo memory scores.",
            source_episode="episodes/2026-06-03.md",
        )

    assert outcome.facts_extracted == 0
    assert outcome.facts_failed == 1
    assert "facts_extracted=0 facts_failed=1" in caplog.text


@pytest.mark.unit
def test_fact_reconciliation_failure_warns_once_per_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    reset_warning_rate_limits()

    def fail_reconcile(*args, **kwargs):
        raise RuntimeError("reconcile failed")

    monkeypatch.setattr(fact_extraction, "_facts_reconcile_enabled", lambda: True)
    monkeypatch.setattr(fact_extraction, "reconcile_new_fact", fail_reconcile)

    with caplog.at_level("WARNING", logger="animaworks.memory.fact_extraction"):
        first = fact_extraction._reconcile_extracted_facts(
            tmp_path / "alice",
            [_fact()],
            as_of_time=None,
        )
        second = fact_extraction._reconcile_extracted_facts(
            tmp_path / "alice",
            [_fact()],
            as_of_time=None,
        )

    assert first[0] == [_fact()]
    assert second[0] == [_fact()]
    records = [
        record for record in caplog.records if record.message == "Fact reconciliation failed; appending extracted fact"
    ]
    assert len(records) == 1
    assert records[0].exc_info
