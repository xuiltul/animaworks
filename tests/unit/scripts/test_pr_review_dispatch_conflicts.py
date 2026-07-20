"""check_conflicts() のユニットテスト（scripts/pr-review-dispatch.py）。"""

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "pr-review-dispatch.py"


def _pr(
    number: int,
    mergeable: str,
    sha: str = "a" * 40,
    *,
    draft: bool = False,
    title: str = "t",
) -> dict:
    return {
        "number": number,
        "title": title,
        "headRefName": f"feat/{number}",
        "baseRefName": "main",
        "headRefOid": sha,
        "mergeable": mergeable,
        "isDraft": draft,
    }


@pytest.fixture
def mod(tmp_path, monkeypatch):
    monkeypatch.setenv("ANIMAWORKS_SHARED_DIR", str(tmp_path))
    monkeypatch.setenv("PR_DISPATCH_REPOS", "o/r")
    spec = importlib.util.spec_from_file_location("pr_review_dispatch_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.REPOS = ["o/r"]
    return module


def _run(mod, state: dict, prs: list[dict]) -> list[str]:
    sent: list[str] = []
    mod.gh = lambda args: json.dumps(prs)
    mod.send = lambda to, content: sent.append(content)
    mod.check_conflicts(state)
    return sent


def test_conflicting_pr_notifies_once(mod):
    state = mod.default_state()
    sent = _run(mod, state, [_pr(1, "CONFLICTING")])
    assert len(sent) == 1
    assert "マージコンフリクト検知" in sent[0]
    assert "o/r#1" in sent[0]
    # 同一headのままの再巡回では再通知しない
    assert _run(mod, state, [_pr(1, "CONFLICTING")]) == []


def test_new_head_still_conflicting_renotifies(mod):
    state = mod.default_state()
    _run(mod, state, [_pr(1, "CONFLICTING", sha="a" * 40)])
    sent = _run(mod, state, [_pr(1, "CONFLICTING", sha="b" * 40)])
    assert len(sent) == 1


def test_mergeable_clears_record_and_reconflict_renotifies(mod):
    state = mod.default_state()
    _run(mod, state, [_pr(1, "CONFLICTING")])
    _run(mod, state, [_pr(1, "MERGEABLE")])
    assert state["conflict_notified"] == {}
    # baseが進んで同一headが再びCONFLICTINGになったら再通知する
    sent = _run(mod, state, [_pr(1, "CONFLICTING")])
    assert len(sent) == 1


def test_unknown_and_draft_are_skipped(mod):
    state = mod.default_state()
    assert _run(mod, state, [_pr(1, "UNKNOWN"), _pr(2, "CONFLICTING", draft=True)]) == []
    assert state["conflict_notified"] == {}


def test_closed_pr_record_is_pruned(mod):
    state = mod.default_state()
    _run(mod, state, [_pr(1, "CONFLICTING")])
    assert "o/r#1" in state["conflict_notified"]
    _run(mod, state, [])
    assert state["conflict_notified"] == {}
