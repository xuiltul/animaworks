from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _load_script():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "check_facts_health.py"
    spec = importlib.util.spec_from_file_location("check_facts_health", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
def test_collect_facts_health_reports_counts_and_dormant_status(tmp_path: Path) -> None:
    mod = _load_script()
    data_dir = tmp_path / ".animaworks"
    alice = data_dir / "animas" / "alice"
    bob = data_dir / "animas" / "bob"
    (alice / "facts").mkdir(parents=True)
    (alice / "episodes").mkdir()
    (bob / "facts").mkdir(parents=True)
    alice.joinpath("status.json").write_text(json.dumps({"enabled": True}), encoding="utf-8")
    bob.joinpath("status.json").write_text(json.dumps({"enabled": False}), encoding="utf-8")
    alice.joinpath("facts", "2026-06-10.jsonl").write_text('{"text":"a"}\n{"text":"b"}\n', encoding="utf-8")
    alice.joinpath("episodes", "2026-06-10.md").write_text("episode", encoding="utf-8")

    rows = mod.collect_facts_health(data_dir)

    assert [(row.anima, row.status, row.facts_count, row.last_fact_date) for row in rows] == [
        ("alice", "active", 2, "2026-06-10"),
        ("bob", "dormant", 0, ""),
    ]

    table = mod.render_table(rows)
    assert "active_with_facts=1/1" in table
    assert "dormant=1" in table
