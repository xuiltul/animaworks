from __future__ import annotations

from pathlib import Path

import pytest

from core.memory.fact_invalidation import (
    FactCandidate,
    ReconcileAction,
    ReconcileConfig,
    reconcile_new_fact,
)
from core.memory.facts import FactRecord, append_fact_records, fact_file_for_record, read_fact_records


def _store(anima_dir: Path, record: FactRecord) -> Path:
    append_fact_records(anima_dir, [record])
    return fact_file_for_record(anima_dir, record)


def _candidate(record: FactRecord, path: Path, score: float = 0.95) -> FactCandidate:
    return FactCandidate(record=record, score=score, path=path, doc_id=f"doc-{record.fact_id}")


@pytest.mark.unit
def test_reconcile_below_similarity_threshold_adds_without_llm(tmp_path: Path) -> None:
    anima_dir = tmp_path / "alice"
    old = FactRecord(text="Alice uses LoCoMo rubric A.", recorded_at="2026-06-03T09:00:00+09:00")
    path = _store(anima_dir, old)
    new = FactRecord(text="Alice uses LoCoMo rubric B.", recorded_at="2026-06-03T10:00:00+09:00")

    def fail_classifier(*args, **kwargs):
        raise AssertionError("LLM classifier should not be called below threshold")

    result = reconcile_new_fact(
        anima_dir,
        new,
        classifier=fail_classifier,
        candidate_search=lambda *_args: [_candidate(old, path, score=0.3)],
        config=ReconcileConfig(enabled=True, threshold=0.82, top_k=5),
    )

    assert result.action == ReconcileAction.ADD
    assert result.should_append is True
    assert result.reason == "below_similarity_threshold"
    assert read_fact_records(path, include_expired=True)[0].valid_until == ""


@pytest.mark.unit
def test_reconcile_search_or_llm_failure_falls_back_to_add(tmp_path: Path) -> None:
    new = FactRecord(text="Alice prefers LoCoMo score deltas.", recorded_at="2026-06-03T10:00:00+09:00")

    search_failed = reconcile_new_fact(
        tmp_path / "alice",
        new,
        candidate_search=lambda *_args: (_ for _ in ()).throw(RuntimeError("chroma down")),
        config=ReconcileConfig(enabled=True),
    )
    assert search_failed.action == ReconcileAction.ADD
    assert search_failed.should_append is True
    assert search_failed.error == "chroma down"

    old = FactRecord(text="Alice prefers LoCoMo score summaries.", recorded_at="2026-06-03T09:00:00+09:00")
    anima_dir = tmp_path / "bob"
    path = _store(anima_dir, old)
    llm_failed = reconcile_new_fact(
        anima_dir,
        new,
        classifier=lambda *_args: (_ for _ in ()).throw(RuntimeError("llm down")),
        candidate_search=lambda *_args: [_candidate(old, path)],
        config=ReconcileConfig(enabled=True),
    )
    assert llm_failed.action == ReconcileAction.ADD
    assert llm_failed.should_append is True
    assert llm_failed.error == "llm down"


@pytest.mark.unit
def test_reconcile_invalid_label_adds(tmp_path: Path) -> None:
    anima_dir = tmp_path / "alice"
    old = FactRecord(text="Alice prefers concise reports.", recorded_at="2026-06-03T09:00:00+09:00")
    path = _store(anima_dir, old)
    new = FactRecord(text="Alice prefers detailed reports.", recorded_at="2026-06-03T10:00:00+09:00")

    result = reconcile_new_fact(
        anima_dir,
        new,
        classifier=lambda *_args: "UNKNOWN",
        candidate_search=lambda *_args: [_candidate(old, path)],
        config=ReconcileConfig(enabled=True),
    )

    assert result.action == ReconcileAction.ADD
    assert result.should_append is True
    assert result.reason == "invalid_label"


@pytest.mark.unit
def test_reconcile_duplicate_skips_new_fact(tmp_path: Path) -> None:
    anima_dir = tmp_path / "alice"
    old = FactRecord(text="Alice tracks LoCoMo memory scores.", recorded_at="2026-06-03T09:00:00+09:00")
    path = _store(anima_dir, old)
    new = FactRecord(text="Alice tracks LoCoMo memory scores.", recorded_at="2026-06-03T10:00:00+09:00")

    result = reconcile_new_fact(
        anima_dir,
        new,
        classifier=lambda *_args: "DUPLICATE",
        candidate_search=lambda *_args: [_candidate(old, path)],
        config=ReconcileConfig(enabled=True),
    )

    assert result.action == ReconcileAction.SKIP
    assert result.should_append is False
    assert result.affected_fact_ids == (old.fact_id,)


@pytest.mark.unit
def test_reconcile_contradiction_invalidates_all_candidates_and_appends_new(tmp_path: Path) -> None:
    anima_dir = tmp_path / "alice"
    old_a = FactRecord(text="Alice's LoCoMo score is 70.", recorded_at="2026-06-03T09:00:00+09:00")
    old_b = FactRecord(
        text="Alice still uses the old LoCoMo score.",
        recorded_at="2026-06-03T09:05:00+09:00",
        valid_until="2026-12-31T00:00:00+09:00",
    )
    path_a = _store(anima_dir, old_a)
    path_b = _store(anima_dir, old_b)
    new = FactRecord(
        text="Alice's LoCoMo score is 85.",
        valid_at="2026-06-03T12:00:00+09:00",
        recorded_at="2026-06-03T12:05:00+09:00",
    )

    result = reconcile_new_fact(
        anima_dir,
        new,
        classifier=lambda *_args: "CONTRADICT",
        candidate_search=lambda *_args: [_candidate(old_a, path_a), _candidate(old_b, path_b, score=0.91)],
        config=ReconcileConfig(enabled=True, threshold=0.82, top_k=5),
    )
    if result.should_append:
        append_fact_records(anima_dir, [new])

    records = read_fact_records(path_a, include_expired=True)
    assert result.action == ReconcileAction.INVALIDATE_OLD
    assert result.should_append is True
    assert {record.fact_id for record in result.updated_records} == {old_a.fact_id, old_b.fact_id}
    assert {record.valid_until for record in result.updated_records} == {"2026-06-03T12:00:00+09:00"}
    assert [record.text for record in records] == [old_a.text, old_b.text, new.text]


@pytest.mark.unit
def test_reconcile_contradiction_uses_recorded_at_without_valid_at(tmp_path: Path) -> None:
    anima_dir = tmp_path / "alice"
    old = FactRecord(text="Alice uses rubric A.", recorded_at="2026-06-03T09:00:00+09:00")
    path = _store(anima_dir, old)
    new = FactRecord(text="Alice uses rubric B.", recorded_at="2026-06-03T10:00:00+09:00")

    result = reconcile_new_fact(
        anima_dir,
        new,
        classifier=lambda *_args: "CONTRADICT",
        candidate_search=lambda *_args: [_candidate(old, path)],
        config=ReconcileConfig(enabled=True),
    )

    assert result.action == ReconcileAction.INVALIDATE_OLD
    assert result.updated_records[0].valid_until == "2026-06-03T10:00:00+09:00"


@pytest.mark.unit
def test_reconcile_complement_updates_best_candidate_without_append(tmp_path: Path) -> None:
    anima_dir = tmp_path / "alice"
    old_best = FactRecord(
        text="Alice prefers LoCoMo reports.",
        entities=["Alice", "LoCoMo"],
        confidence=0.5,
        recorded_at="2026-06-03T09:00:00+09:00",
    )
    old_other = FactRecord(text="Alice mentions LoCoMo.", recorded_at="2026-06-03T09:05:00+09:00")
    path_best = _store(anima_dir, old_best)
    path_other = _store(anima_dir, old_other)
    new = FactRecord(
        text="Alice prefers LoCoMo reports with score deltas.",
        entities=["Score Deltas"],
        confidence=0.9,
        recorded_at="2026-06-03T10:00:00+09:00",
    )

    result = reconcile_new_fact(
        anima_dir,
        new,
        classifier=lambda *_args: "COMPLEMENT",
        candidate_search=lambda *_args: [_candidate(old_best, path_best, 0.95), _candidate(old_other, path_other, 0.9)],
        config=ReconcileConfig(enabled=True),
    )

    updated_records = read_fact_records(path_best, include_expired=True)
    assert result.action == ReconcileAction.UPDATE
    assert result.should_append is False
    assert "score deltas" in updated_records[0].text.lower()
    assert updated_records[0].entities == ["Alice", "LoCoMo", "Score Deltas"]
    assert updated_records[0].confidence == 0.9


@pytest.mark.unit
def test_reconcile_ignores_candidate_expired_at_new_fact_time(tmp_path: Path) -> None:
    anima_dir = tmp_path / "alice"
    old = FactRecord(
        text="Alice used an obsolete LoCoMo rubric.",
        recorded_at="2026-06-03T08:00:00+09:00",
        valid_until="2026-06-03T09:00:00+09:00",
    )
    path = _store(anima_dir, old)
    new = FactRecord(
        text="Alice uses the current LoCoMo rubric.",
        valid_at="2026-06-03T10:00:00+09:00",
        recorded_at="2026-06-03T10:01:00+09:00",
    )

    result = reconcile_new_fact(
        anima_dir,
        new,
        classifier=lambda *_args: (_ for _ in ()).throw(AssertionError("expired candidate should be ignored")),
        candidate_search=lambda *_args: [_candidate(old, path)],
        config=ReconcileConfig(enabled=True),
    )

    assert result.action == ReconcileAction.ADD
    assert result.reason == "no_active_candidates"


@pytest.mark.unit
def test_reconcile_invalidation_write_failure_skips_new_append(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    anima_dir = tmp_path / "alice"
    old = FactRecord(text="Alice uses rubric A.", recorded_at="2026-06-03T09:00:00+09:00")
    path = _store(anima_dir, old)
    new = FactRecord(text="Alice uses rubric B.", recorded_at="2026-06-03T10:00:00+09:00")

    monkeypatch.setattr(
        "core.memory.fact_invalidation.update_fact_records_by_id",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("rewrite failed")),
    )

    result = reconcile_new_fact(
        anima_dir,
        new,
        classifier=lambda *_args: "CONTRADICT",
        candidate_search=lambda *_args: [_candidate(old, path)],
        config=ReconcileConfig(enabled=True),
    )

    assert result.action == ReconcileAction.SKIP
    assert result.should_append is False
    assert result.reason == "invalidate_failed"
    assert result.error == "rewrite failed"
