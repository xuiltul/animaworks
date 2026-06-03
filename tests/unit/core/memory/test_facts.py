from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.memory.facts import (
    FactRecord,
    append_fact_records,
    fact_file_for_record,
    is_valid_until_active,
    read_fact_records,
    rewrite_fact_records,
    update_fact_record_by_id,
)


@pytest.mark.unit
def test_fact_record_generates_stable_id_and_entities() -> None:
    first = FactRecord(
        text="Alice prefers LoCoMo reports.",
        source_entity="Alice",
        target_entity="LoCoMo",
        edge_type="PREFERS",
        valid_at="2026-06-03T09:00:00+09:00",
        entities=["Alice", "alice", "LoCoMo"],
    )
    second = FactRecord(
        text=" Alice prefers LoCoMo reports. ",
        source_entity="alice",
        target_entity="locomo",
        edge_type="PREFERS",
        valid_at="2026-06-03T09:00:00+09:00",
    )

    assert first.fact_id == second.fact_id
    assert first.entities == ["Alice", "LoCoMo"]
    assert first.dedup_key == second.dedup_key


@pytest.mark.unit
def test_append_read_rewrite_and_dedup(tmp_path: Path) -> None:
    anima_dir = tmp_path / "alice"
    record = FactRecord(
        text="Alice prefers compact progress reports.",
        source_entity="Alice",
        target_entity="progress reports",
        edge_type="PREFERS",
        recorded_at="2026-06-03T10:00:00+09:00",
    )
    duplicate = FactRecord(
        text="Alice prefers compact progress reports.",
        source_entity="Alice",
        target_entity="progress reports",
        edge_type="PREFERS",
        recorded_at="2026-06-03T10:05:00+09:00",
    )
    expired = FactRecord(
        text="Alice used the legacy scoring rubric.",
        source_entity="Alice",
        target_entity="legacy scoring rubric",
        edge_type="USED",
        recorded_at="2026-06-03T10:10:00+09:00",
        valid_until="2000-01-01T00:00:00+00:00",
    )

    stored = append_fact_records(anima_dir, [record, duplicate, expired])
    fact_file = fact_file_for_record(anima_dir, record)

    assert [r.text for r in stored] == [record.text, expired.text]
    assert fact_file.exists()
    assert fact_file.with_suffix(".jsonl.lock").exists()
    assert [r.text for r in read_fact_records(fact_file)] == [record.text]
    assert [r.text for r in read_fact_records(fact_file, include_expired=True)] == [
        record.text,
        expired.text,
    ]

    rewrite_fact_records(fact_file, [expired, record, duplicate])
    assert [r.text for r in read_fact_records(fact_file, include_expired=True)] == [
        expired.text,
        record.text,
    ]


@pytest.mark.unit
def test_valid_until_as_of_and_update_by_fact_id(tmp_path: Path) -> None:
    anima_dir = tmp_path / "alice"
    record = FactRecord(
        text="Alice uses a temporary LoCoMo rubric.",
        recorded_at="2026-06-03T10:00:00+09:00",
        valid_until="2026-06-03T12:00:00+09:00",
    )
    append_fact_records(anima_dir, [record])
    fact_file = fact_file_for_record(anima_dir, record)

    assert is_valid_until_active(record.valid_until, as_of_time="2026-06-03T11:00:00+09:00")
    assert not is_valid_until_active(record.valid_until, as_of_time="2026-06-03T12:00:00+09:00")
    assert [r.fact_id for r in read_fact_records(fact_file, as_of_time="2026-06-03T11:00:00+09:00")] == [record.fact_id]
    assert read_fact_records(fact_file, as_of_time="2026-06-03T12:00:00+09:00") == []

    updated = update_fact_record_by_id(
        anima_dir,
        record.fact_id,
        lambda current: FactRecord.from_dict({**current.to_dict(), "valid_until": "2026-06-03T11:30:00+09:00"}),
    )

    assert updated is not None
    assert updated.path == fact_file
    assert read_fact_records(fact_file, include_expired=True)[0].valid_until == "2026-06-03T11:30:00+09:00"


@pytest.mark.unit
def test_read_fact_records_skips_invalid_lines(tmp_path: Path) -> None:
    path = tmp_path / "facts.jsonl"
    record = FactRecord(text="Valid fact", recorded_at="2026-06-03T10:00:00+09:00")
    path.write_text(
        "\n".join(
            [
                "{not json}",
                json.dumps({"text": "Missing id is fine"}, ensure_ascii=False),
                record.to_json_line(),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    records = read_fact_records(path)

    assert [r.text for r in records] == ["Missing id is fine", "Valid fact"]
