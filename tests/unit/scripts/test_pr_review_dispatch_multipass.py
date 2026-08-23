"""Multi-pass FRC review（scripts/pr-review-dispatch.py）のユニットテスト。

PR_DISPATCH_REVIEW_MODELS 設定時のマルチパス発行と、全パス完了後の統合タスク
発行を検証する。dispatch_direct_task / TaskQueueManager は使わず、
dispatch_task / _review_task をモックしてスクリプトのロジックだけを扱う。
"""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "pr-review-dispatch.py"

SHA = "a" * 40


def _pr(number: int, sha: str = SHA) -> dict:
    return {"number": number, "title": f"PR {number}", "headRefOid": sha, "isDraft": False}


def _load_module(tmp_path, monkeypatch, review_models: str = "", synth_model: str = ""):
    monkeypatch.setenv("ANIMAWORKS_SHARED_DIR", str(tmp_path))
    monkeypatch.setenv("PR_DISPATCH_REPOS", "o/r")
    monkeypatch.setenv("PR_DISPATCH_QUIET_SECONDS", "0")
    monkeypatch.setenv("PR_DISPATCH_REVIEW_MODELS", review_models)
    monkeypatch.setenv("PR_DISPATCH_SYNTH_MODEL", synth_model)
    spec = importlib.util.spec_from_file_location("pr_review_dispatch_multipass_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.REPOS = ["o/r"]
    return module


@pytest.fixture
def mod(tmp_path, monkeypatch):
    return _load_module(tmp_path, monkeypatch)


@pytest.fixture
def mod_mp(tmp_path, monkeypatch):
    """Multi-pass enabled: two model passes + a synthesis model."""
    return _load_module(
        tmp_path,
        monkeypatch,
        review_models="x:grok/grok-4.5,c:codex/gpt-5.6-sol",
        synth_model="c:codex/gpt-5.6-sol",
    )


def _run_check_commits(mod, state: dict, prs: list[dict]) -> list[dict]:
    tasks: list[dict] = []
    mod.gh = lambda args: json.dumps(prs)
    mod.dispatch_task = lambda **kwargs: tasks.append(kwargs) or True
    mod.check_commits(state)
    return tasks


# ── _model_slug ───────────────────────────────────────────────────────────
def test_model_slug_normalizes_and_strips_mode_prefix(mod):
    assert mod._model_slug("x:grok/grok-4.5") == "grok-grok-4-5"
    assert mod._model_slug("c:codex/gpt-5.6-sol") == "codex-gpt-5-6-sol"
    assert mod._model_slug("plain-model") == "plain-model"
    assert mod._model_slug("a b&c!d") == "a-b-c-d"


# ── 未設定時は従来通り（送信のみ・modelなし） ────────────────────────────
def test_unset_models_keeps_single_send_without_tasks(mod):
    state = mod.default_state()
    state["prs"]["o/r#1"] = {"sha": SHA, "sha_seen_at": mod.iso(mod.now_utc()), "title": "t"}
    sent: list[tuple[str, str]] = []
    mod.send = lambda to, content: sent.append((to, content))
    mod.dispatch_task = lambda **kwargs: pytest.fail("no task should be dispatched")
    mod.gh = lambda args: json.dumps([_pr(1)])
    mod.check_commits(state)
    assert len(sent) == 1
    assert sent[0][0] == mod.REVIEWER
    assert "静穏を確認済み" in sent[0][1]


# ── 設定時: モデルごとにタスクを発行 ─────────────────────────────────────
def test_multi_pass_dispatches_one_task_per_model(mod_mp):
    mod = mod_mp
    state = mod.default_state()
    state["prs"]["o/r#1"] = {"sha": SHA, "sha_seen_at": mod.iso(mod.now_utc()), "title": "t"}
    tasks = _run_check_commits(mod, state, [_pr(1)])
    assert len(tasks) == 2
    base = mod._ci_task_id("o/r", 1, SHA)
    ids = {t["task_id"] for t in tasks}
    assert ids == {f"{base}-m-grok-grok-4-5", f"{base}-m-codex-gpt-5-6-sol"}
    for t in tasks:
        assert t["target"] == mod.REVIEWER
        assert t["model"] in ("x:grok/grok-4.5", "c:codex/gpt-5.6-sol")
        assert "-m-" in t["task_id"]
        assert "指摘の列挙に徹すること" in t["instruction"]
        assert "最終判定（APPROVE操作等）はこのパスでは行わず" in t["instruction"]
        assert "reviews/pr1-frc-" in t["instruction"] and "-frc-" in t["instruction"]
    info = state["multi_model_passes"][base]
    assert info["synth_dispatched"] is False
    assert set(info["task_ids"]) == {f"{base}-m-grok-grok-4-5", f"{base}-m-codex-gpt-5-6-sol"}


def test_multi_pass_idempotent_within_scan(mod_mp):
    mod = mod_mp
    state = mod.default_state()
    state["prs"]["o/r#1"] = {"sha": SHA, "sha_seen_at": mod.iso(mod.now_utc()), "title": "t"}
    _run_check_commits(mod, state, [_pr(1)])
    # notified_sha が同SHAなので再巡回では ready にならない（重複発行なし）
    assert _run_check_commits(mod, state, [_pr(1)]) == []


# ── 統合タスク ──────────────────────────────────────────────────────────
def _terminal_modules(mod):
    base = mod._ci_task_id("o/r", 1, SHA)
    mods = ["x:grok/grok-4.5", "c:codex/gpt-5.6-sol"]
    task_ids = [f"{base}-m-{mod._model_slug(m)}" for m in mods]
    return base, task_ids


def test_synth_dispatched_once_when_all_done(mod_mp):
    mod = mod_mp
    state = mod.default_state()
    base, task_ids = _terminal_modules(mod)
    state["multi_model_passes"] = {
        base: {
            "repo": "o/r", "number": 1, "sha": SHA, "models": ["x:grok/grok-4.5", "c:codex/gpt-5.6-sol"],
            "task_ids": task_ids, "synth_dispatched": False,
        }
    }
    mod._review_task = lambda tid: SimpleNamespace(task_id=tid, status="done")
    tasks: list[dict] = []
    mod.dispatch_task = lambda **kwargs: tasks.append(kwargs) or True

    mod.check_multipass_synth(state)
    assert len(tasks) == 1
    assert tasks[0]["task_id"] == f"{base}-synth"
    assert tasks[0]["model"] == "c:codex/gpt-5.6-sol"  # PR_DISPATCH_SYNTH_MODEL
    assert "reviews/pr1-frc-" in tasks[0]["instruction"]
    assert "マージ・重複排除" in tasks[0]["instruction"]
    assert "欠落" not in tasks[0]["instruction"]

    # 2回目のscanでは発行しない
    mod._review_task = lambda tid: SimpleNamespace(task_id=tid, status="done")
    mod.check_multipass_synth(state)
    assert len(tasks) == 1


def test_synth_waits_for_active_pass(mod_mp):
    mod = mod_mp
    state = mod.default_state()
    base, task_ids = _terminal_modules(mod)
    state["multi_model_passes"] = {
        base: {
            "repo": "o/r", "number": 1, "sha": SHA, "models": ["x:grok/grok-4.5", "c:codex/gpt-5.6-sol"],
            "task_ids": task_ids, "synth_dispatched": False,
        }
    }
    statuses = {task_ids[0]: "done", task_ids[1]: "in_progress"}
    mod._review_task = lambda tid: SimpleNamespace(task_id=tid, status=statuses.get(tid, "done"))
    tasks: list[dict] = []
    mod.dispatch_task = lambda **kwargs: tasks.append(kwargs) or True
    mod.check_multipass_synth(state)
    assert tasks == []


def test_synth_dispatched_with_failed_note_when_partial(mod_mp):
    mod = mod_mp
    state = mod.default_state()
    base, task_ids = _terminal_modules(mod)
    state["multi_model_passes"] = {
        base: {
            "repo": "o/r", "number": 1, "sha": SHA, "models": ["x:grok/grok-4.5", "c:codex/gpt-5.6-sol"],
            "task_ids": task_ids, "synth_dispatched": False,
        }
    }
    statuses = {task_ids[0]: "done", task_ids[1]: "failed"}
    mod._review_task = lambda tid: SimpleNamespace(task_id=tid, status=statuses.get(tid, "done"))
    tasks: list[dict] = []
    mod.dispatch_task = lambda **kwargs: tasks.append(kwargs) or True
    mod.check_multipass_synth(state)
    assert len(tasks) == 1
    assert "欠落" in tasks[0]["instruction"]


def test_synth_not_dispatched_when_all_pass_failed(mod_mp):
    mod = mod_mp
    state = mod.default_state()
    base, task_ids = _terminal_modules(mod)
    state["multi_model_passes"] = {
        base: {
            "repo": "o/r", "number": 1, "sha": SHA, "models": ["x:grok/grok-4.5", "c:codex/gpt-5.6-sol"],
            "task_ids": task_ids, "synth_dispatched": False,
        }
    }
    mod._review_task = lambda tid: SimpleNamespace(task_id=tid, status="failed")
    tasks: list[dict] = []
    mod.dispatch_task = lambda **kwargs: tasks.append(kwargs) or True
    mod.check_multipass_synth(state)
    assert tasks == []
    assert state["multi_model_passes"][base]["synth_dispatched"] is False
