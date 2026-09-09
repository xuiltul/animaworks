from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _module():
    source = Path(__file__).resolve().parents[3] / "scripts/slim_runtime_live.py"
    spec = importlib.util.spec_from_file_location("slim_runtime_live", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reasoning_score_requires_decision_and_provenance():
    score = _module().score_result
    case = {"expected_decision": "paused", "required_evidence": "D-2"}
    assert score('{"decision":"paused","evidence":"D-2"}', case)["passed"]
    assert score('{"decision":"paused","evidence":"D-2"}\n<!-- emotion: {"emotion":"thinking"} -->', case)["passed"]
    assert not score('{"decision":"paused","evidence":"old note"}', case)["passed"]
    assert not score('{"decision":"pilot_only","evidence":"D-2"}', case)["passed"]
    assert not score("Provider error: request unauthorized", case)["passed"]


def test_quality_fixture_has_twelve_unique_anonymous_cases():
    source = Path(__file__).resolve().parents[3] / "tests/fixtures/slim_runtime/live_quality.json"
    cases = json.loads(source.read_text())
    assert len(cases) == len({item["id"] for item in cases}) == 12
    assert all(item["question"] and item["expected_decision"] for item in cases)
