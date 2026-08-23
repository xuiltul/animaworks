"""scripts/pr-review-dispatch.py のマルチパス統合レイヤのユニットテスト。

ロジック本体は core.review_multipass に移ったため、ここではスクリプト側の
(1) 有効モデル解決（env 上書き > config）、(2) 未設定時は従来通り単発送信、
(3) config 設定時の multipass 分岐、を検証する。
"""

import importlib.util
import json
from pathlib import Path

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


def test_env_models_dispatch_multipass_via_common(tmp_path, monkeypatch):
    mod = _load_module(
        tmp_path, monkeypatch, review_models="c:gpt-5.6-sol,s:gpt-5.6-sol"
    )
    state = mod.default_state()
    state["prs"]["o/r#1"] = {"sha": SHA, "sha_seen_at": mod.iso(mod.now_utc()), "title": "t"}
    tasks: list[dict] = []
    mod.send = lambda to, content: pytest.fail("no dm when multipass configured")
    mod.dispatch_task = lambda **kwargs: tasks.append(kwargs) or True
    mod.gh = lambda args: json.dumps([_pr(1)])
    mod.check_commits(state)
    base = mod._ci_task_id("o/r", 1, SHA)
    ids = {t["task_id"] for t in tasks}
    assert ids == {f"{base}-m-c-gpt-5-6-sol", f"{base}-m-s-gpt-5-6-sol"}
    assert state["multi_model_passes"][base]["attempt"] == 1


def test_effective_models_prefer_env_over_config(mod, monkeypatch):
    # env 空なら config（github_webhook）の値を使う
    mod.PR_DISPATCH_REVIEW_MODELS = []

    class _Cfg:
        review_multipass_models = ["c:codex/gpt-5.6-sol"]
        review_synth_model = "c:codex/gpt-5.6-sol"

    class _Webhook:
        github_webhook = _Cfg()

    def _fake_load():
        return _Webhook()

    monkeypatch.setattr("core.config.models.load_config", _fake_load)
    assert mod._review_models() == ["c:codex/gpt-5.6-sol"]
    assert mod._synth_model() == "c:codex/gpt-5.6-sol"

    # env 設定があれば上書き
    mod.PR_DISPATCH_REVIEW_MODELS = ["x:grok/grok-4.5"]
    mod.PR_DISPATCH_SYNTH_MODEL = "x:grok/grok-4.5"
    assert mod._review_models() == ["x:grok/grok-4.5"]
    assert mod._synth_model() == "x:grok/grok-4.5"
