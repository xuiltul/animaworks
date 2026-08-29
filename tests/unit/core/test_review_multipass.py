"""Multi-pass FRC review（core.review_multipass）のユニットテスト。

計画書 20260823_revB の修正（H-5 エントリ先行永続化・archive-aware照会・全滅時の
-r2再発行、M-1 slugのmode保持、M-6 設定空でも既存エントリ処理）を直接検証する。
dispatch / review はモックし、TaskQueue / ディスク・GitHubを使わない。
"""

import json
from pathlib import Path
from types import SimpleNamespace

from core.review_multipass import (
    check_multipass_synth,
    default_state,
    dispatch_multipass_reviews,
    load_state,
    locked_state,
    model_slug,
    save_state,
)

SHA = "a" * 40
BASE = "gh-ci-o-r#1-aaaaaaaa"
TASK_IDS = [f"{BASE}-m-c-codex-gpt-5-6-sol", f"{BASE}-m-x-grok-grok-4-5"]


class _DispatchRecorder:
    """Captures dispatch kwargs; each task can be marked to raise."""

    def __init__(self, raise_on: set | None = None):
        self.tasks: list[dict] = []
        self._raise_on = raise_on or set()

    def __call__(self, **kwargs):
        if kwargs["task_id"] in self._raise_on:
            raise RuntimeError(f"boom {kwargs['task_id']}")
        self.tasks.append(kwargs)
        return True


def _entry() -> dict:
    return {
        "repo": "o/r",
        "number": 1,
        "sha": SHA,
        "models": ["c:codex/gpt-5.6-sol", "x:grok/grok-4.5"],
        "task_ids": list(TASK_IDS),
        "attempt": 1,
    }


def _state_with_entry(**overrides) -> dict:
    entry = _entry()
    entry.update(overrides)
    state = default_state()
    state["multi_model_passes"] = {BASE: entry}
    return state


# ── M-1: slug規則 ─────────────────────────────────────────────────────────
def test_model_slug_keeps_mode_and_preserves_model_tag():
    # c: と s: で異なる slug（task_id 衝突なし）
    assert model_slug("c:gpt-5.6-sol") == "c-gpt-5-6-sol"
    assert model_slug("s:gpt-5.6-sol") == "s-gpt-5-6-sol"
    assert model_slug("c:codex/gpt-5.6-sol") == "c-codex-gpt-5-6-sol"
    # 2文字目が ':' でないので mode として扱わず、タグのコロンを保持
    assert model_slug("ollama/qwen3:14b") == "ollama-qwen3-14b"
    assert model_slug("plain-model") == "plain-model"
    assert model_slug("a b&c!d") == "a-b-c-d"
    assert model_slug("") == "model"


def test_model_slug_keeps_mode_and_unique_task_ids():
    recorder = _DispatchRecorder()
    state = default_state()
    dispatch_multipass_reviews(
        state,
        [{"repo": "o/r", "number": 1, "sha": SHA, "title": "t"}],
        reviewer="sumire",
        models=["c:gpt-5.6-sol", "s:gpt-5.6-sol"],
        dispatch=recorder,
    )
    ids = {t["task_id"] for t in recorder.tasks}
    assert f"{BASE}-m-c-gpt-5-6-sol" in ids
    assert f"{BASE}-m-s-gpt-5-6-sol" in ids
    assert len(ids) == 2


# ── エントリ先行永続化 (H-5.1) ──────────────────────────────────────────
def test_entry_persisted_even_when_dispatch_raises():
    recorder = _DispatchRecorder(raise_on={TASK_IDS[0]})
    state = default_state()
    dispatch_multipass_reviews(
        state,
        [{"repo": "o/r", "number": 1, "sha": SHA, "title": "t"}],
        reviewer="sumire",
        models=["c:codex/gpt-5.6-sol", "x:grok/grok-4.5"],
        dispatch=recorder,
    )
    # 例外したパスはスキップ、正常パスは発行、エントリは残る
    assert len(recorder.tasks) == 1
    assert recorder.tasks[0]["task_id"] == TASK_IDS[1]
    entry = state["multi_model_passes"][BASE]
    assert entry["attempt"] == 1
    assert set(entry["task_ids"]) == set(TASK_IDS)
    # dispatch_direct_task が False（既存）でも正常に継続できることを確認
    state2 = default_state()
    dispatch_multipass_reviews(
        state2,
        [{"repo": "o/r", "number": 1, "sha": SHA, "title": "t"}],
        reviewer="sumire",
        models=["c:codex/gpt-5.6-sol"],
        dispatch=lambda **_: False,
    )
    assert BASE in state2["multi_model_passes"]


def test_dispatch_is_idempotent_within_scan():
    recorder = _DispatchRecorder()
    state = default_state()
    items = [{"repo": "o/r", "number": 1, "sha": SHA, "title": "t"}]
    dispatch_multipass_reviews(state, items, reviewer="sumire", models=["c:codex/gpt-5.6-sol"], dispatch=recorder)
    dispatch_multipass_reviews(state, items, reviewer="sumire", models=["c:codex/gpt-5.6-sol"], dispatch=recorder)
    assert len(recorder.tasks) == 1


def test_newer_exact_supersedes_queued_reviews_of_older_exact(tmp_path):
    """新pushが来たら、旧exactの未着手レビューは放棄して最新だけを走らせる。"""
    from core.memory.task_queue import TaskQueueManager

    animas_dir = tmp_path / "animas"
    reviewer_dir = animas_dir / "sumire"
    (reviewer_dir / "state" / "pending").mkdir(parents=True)
    manager = TaskQueueManager(reviewer_dir)
    old_sha = "b" * 40
    for task_id, status in (("old-pending", "pending"), ("old-running", "in_progress")):
        manager.add_task(
            source="anima",
            original_instruction="review",
            assignee="sumire",
            summary="旧exactのレビュー",
            task_id=task_id,
            meta={"repo": "o/r", "number": 1, "sha": old_sha, "multipass": True},
        )
        manager.update_status(task_id, status, summary="旧exactのレビュー")

    recorder = _DispatchRecorder()
    dispatch_multipass_reviews(
        default_state(),
        [{"repo": "o/r", "number": 1, "sha": SHA, "title": "t"}],
        reviewer="sumire",
        models=["c:codex/gpt-5.6-sol"],
        dispatch=recorder,
        animas_dir=animas_dir,
    )

    # 未着手の旧exactと実行中の旧exactの両方を放棄する（cancelled）
    assert manager.get_task_by_id("old-pending").status == "cancelled"
    assert "superseded" in manager.get_task_by_id("old-pending").summary
    assert manager.get_task_by_id("old-running").status == "cancelled"
    assert "superseded" in manager.get_task_by_id("old-running").summary
    assert len(recorder.tasks) == 1


def test_new_exact_dispatch_drops_older_multipass_entry(tmp_path):
    """新shaをdispatchすると、同一repo/numberで異なるshaのmulti_model_passes entryが消える。"""
    old_sha = "b" * 40
    old_base = f"gh-ci-o-r#1-{old_sha[:8]}"
    state = default_state()
    state["multi_model_passes"] = {
        old_base: {
            "repo": "o/r",
            "number": 1,
            "sha": old_sha,
            "models": ["c:gpt-5.6-sol"],
            "task_ids": [f"{old_base}-m-c-gpt-5-6-sol"],
            "attempt": 1,
        }
    }
    recorder = _DispatchRecorder()
    dispatch_multipass_reviews(
        state,
        [{"repo": "o/r", "number": 1, "sha": SHA, "title": "t"}],
        reviewer="sumire",
        models=["c:gpt-5.6-sol"],
        dispatch=recorder,
    )
    # 旧shaのentryは消え、新shaのentryだけが残る
    assert old_base not in state["multi_model_passes"]
    assert BASE in state["multi_model_passes"]
    assert len(recorder.tasks) == 1


def test_same_exact_review_is_not_superseded(tmp_path):
    """同じexactの既存レビューは巻き込まない。"""
    from core.memory.task_queue import TaskQueueManager

    animas_dir = tmp_path / "animas"
    reviewer_dir = animas_dir / "sumire"
    (reviewer_dir / "state" / "pending").mkdir(parents=True)
    manager = TaskQueueManager(reviewer_dir)
    manager.add_task(
        source="anima",
        original_instruction="review",
        assignee="sumire",
        summary="同一exactのレビュー",
        task_id="same-exact",
        meta={"repo": "o/r", "number": 1, "sha": SHA, "multipass": True},
    )

    dispatch_multipass_reviews(
        default_state(),
        [{"repo": "o/r", "number": 1, "sha": SHA, "title": "t"}],
        reviewer="sumire",
        models=["c:codex/gpt-5.6-sol"],
        dispatch=_DispatchRecorder(),
        animas_dir=animas_dir,
    )
    assert manager.get_task_by_id("same-exact").status == "pending"


def test_empty_models_noop():
    recorder = _DispatchRecorder()
    state = default_state()
    dispatch_multipass_reviews(
        state, [{"repo": "o/r", "number": 1, "sha": SHA}], reviewer="sumire", models=[], dispatch=recorder
    )
    assert recorder.tasks == []
    assert "multi_model_passes" not in state


# ── check_multipass_synth: 通常の統合発行 ──────────────────────────────
def test_synth_dispatched_once_when_all_done(monkeypatch):
    state = _state_with_entry()
    dispatched: list[dict] = []

    def fake_review_task(reviewer, tid, animas_dir=None):
        return SimpleNamespace(task_id=tid, status="done")

    monkeypatch.setattr("core.review_multipass.review_task", fake_review_task)
    check_multipass_synth(
        state,
        reviewer="sumire",
        synth_model="c:codex/gpt-5.6-sol",
        dispatch=lambda **kw: dispatched.append(kw) or True,
    )
    assert [d["task_id"] for d in dispatched] == [f"{BASE}-synth"]
    assert dispatched[0]["model"] == "c:codex/gpt-5.6-sol"
    assert "reviews/pr1-frc-" in dispatched[0]["instruction"]
    assert "欠落" not in dispatched[0]["instruction"]
    # 親エントリは除去される（再発行なし）
    assert BASE not in state["multi_model_passes"]


def test_synth_dispatched_with_failed_note_when_partial(monkeypatch):
    state = _state_with_entry()

    def fake_review_task(reviewer, tid, animas_dir=None):
        return SimpleNamespace(task_id=tid, status="failed" if tid == TASK_IDS[0] else "done")

    monkeypatch.setattr("core.review_multipass.review_task", fake_review_task)
    dispatched: list[dict] = []
    check_multipass_synth(
        state, reviewer="sumire", synth_model=None, dispatch=lambda **kw: dispatched.append(kw) or True
    )
    assert len(dispatched) == 1
    assert "欠落" in dispatched[0]["instruction"]


def test_synth_waits_for_active_pass(monkeypatch):
    state = _state_with_entry()

    def fake_review_task(reviewer, tid, animas_dir=None):
        return SimpleNamespace(task_id=tid, status="in_progress" if tid == TASK_IDS[0] else "done")

    monkeypatch.setattr("core.review_multipass.review_task", fake_review_task)
    dispatched: list[dict] = []
    check_multipass_synth(
        state, reviewer="sumire", synth_model=None, dispatch=lambda **kw: dispatched.append(kw) or True
    )
    assert dispatched == []
    assert BASE in state["multi_model_passes"]


# ── archive-aware 照会 (H-5.2) ─────────────────────────────────────────
def test_archive_done_counts_as_completed(monkeypatch):
    state = default_state()
    state["multi_model_passes"] = {BASE: _entry()}
    # live queue には無いが archive に done がある → 完了扱いで統合発行
    live = {}

    def fake_review_task(reviewer, tid, animas_dir=None):
        if tid in live:
            return live[tid]
        return SimpleNamespace(task_id=tid, status="done")  # archive hit

    monkeypatch.setattr("core.review_multipass.review_task", fake_review_task)
    dispatched: list[dict] = []
    check_multipass_synth(
        state, reviewer="sumire", synth_model=None, dispatch=lambda **kw: dispatched.append(kw) or True
    )
    assert [d["task_id"] for d in dispatched] == [f"{BASE}-synth"]


def test_missing_task_waits(monkeypatch):
    state = _state_with_entry()

    def fake_review_task(reviewer, tid, animas_dir=None):
        return None  # 未公開 → 待つ

    monkeypatch.setattr("core.review_multipass.review_task", fake_review_task)
    dispatched: list[dict] = []
    check_multipass_synth(
        state, reviewer="sumire", synth_model=None, dispatch=lambda **kw: dispatched.append(kw) or True
    )
    assert dispatched == []


# ── 全滅時: -r2 再発行 → 2回目全滅で放棄 (H-5.3) ─────────────────────
def test_all_cancelled_superseded_entry_dropped_without_r2(monkeypatch):
    """全パスがcancelled（superseded）のentryは -r2 を投入せずに消える。"""
    state = _state_with_entry()

    def fake_review_task(reviewer, tid, animas_dir=None):
        return SimpleNamespace(task_id=tid, status="cancelled")

    monkeypatch.setattr("core.review_multipass.review_task", fake_review_task)
    dispatched: list[dict] = []
    check_multipass_synth(
        state, reviewer="sumire", synth_model=None, dispatch=lambda **kw: dispatched.append(kw) or True
    )
    assert dispatched == []
    assert BASE not in state["multi_model_passes"]


def test_all_failed_redispatch_with_r2_once(monkeypatch):
    state = _state_with_entry()

    def fake_review_task(reviewer, tid, animas_dir=None):
        return SimpleNamespace(task_id=tid, status="failed")

    monkeypatch.setattr("core.review_multipass.review_task", fake_review_task)
    dispatched: list[dict] = []
    check_multipass_synth(
        state, reviewer="sumire", synth_model=None, dispatch=lambda **kw: dispatched.append(kw) or True
    )
    r2_ids = {d["task_id"] for d in dispatched}
    assert r2_ids == {f"{tid}-r2" for tid in TASK_IDS}
    # attempt=2 に更新、エントリは残る（次回sweepで2回目判定）
    assert state["multi_model_passes"][BASE]["attempt"] == 2
    assert set(state["multi_model_passes"][BASE]["task_ids"]) == {f"{tid}-r2" for tid in TASK_IDS}


def test_all_failed_second_attempt_abandons(monkeypatch):
    state = _state_with_entry(attempt=2, task_ids=[f"{tid}-r2" for tid in TASK_IDS])

    def fake_review_task(reviewer, tid, animas_dir=None):
        return SimpleNamespace(task_id=tid, status="failed")

    monkeypatch.setattr("core.review_multipass.review_task", fake_review_task)
    dispatched: list[dict] = []
    check_multipass_synth(
        state, reviewer="sumire", synth_model=None, dispatch=lambda **kw: dispatched.append(kw) or True
    )
    assert dispatched == []
    assert BASE not in state["multi_model_passes"]


def test_all_failed_r2_then_one_done_synths(monkeypatch):
    state = _state_with_entry(attempt=2, task_ids=[f"{tid}-r2" for tid in TASK_IDS])

    def fake_review_task(reviewer, tid, animas_dir=None):
        return SimpleNamespace(task_id=tid, status="done")

    monkeypatch.setattr("core.review_multipass.review_task", fake_review_task)
    dispatched: list[dict] = []
    check_multipass_synth(
        state, reviewer="sumire", synth_model=None, dispatch=lambda **kw: dispatched.append(kw) or True
    )
    assert [d["task_id"] for d in dispatched] == [f"{BASE}-synth"]


# ── M-6: 設定を空にしても既存エントリは処理継続 ────────────────────────
def test_synth_sweep_continues_when_models_unset(monkeypatch):
    # これを呼ぶ関数は models パラメータを持たない ＝ 設定に関わらず処理する
    state = _state_with_entry()

    def fake_review_task(reviewer, tid, animas_dir=None):
        return SimpleNamespace(task_id=tid, status="done")

    monkeypatch.setattr("core.review_multipass.review_task", fake_review_task)
    dispatched: list[dict] = []
    check_multipass_synth(
        state, reviewer="sumire", synth_model=None, dispatch=lambda **kw: dispatched.append(kw) or True
    )
    assert [d["task_id"] for d in dispatched] == [f"{BASE}-synth"]


# ── state ロック・読み書き (H-4 共有化) ────────────────────────────────
def test_locked_state_preserves_unknown_fields(tmp_path: Path):
    state_file = tmp_path / "state.json"
    state_file.write_text(json.dumps({"prs": {}, "future_schema": {"x": 1}}), encoding="utf-8")
    with locked_state(state_file) as state:
        state["seen_comments"]["k"] = "v"
    restored = json.loads(state_file.read_text(encoding="utf-8"))
    assert restored["future_schema"] == {"x": 1}
    assert restored["seen_comments"]["k"] == "v"
    assert (tmp_path / "state.lock").exists()


def test_load_state_recovers_default_on_invalid(tmp_path: Path):
    state_file = tmp_path / "state.json"
    state_file.write_text("not-json", encoding="utf-8")
    state = load_state(state_file)
    assert set(("prs", "seen_comments", "ci_notified")) <= state.keys()

    state_file.write_text("[]", encoding="utf-8")
    assert load_state(state_file)["prs"] == {}

    save_state(tmp_path / "s.json", {"prs": {}})
    assert load_state(tmp_path / "s.json")["prs"] == {}
