from __future__ import annotations

"""Shared multi-pass FRC review dispatch for the PR review loop.

This module is the single source of truth (SSoT) for the configured
multi-model review passes and their final synthesis pass.  It is used by
both the webhook gateway (:mod:`server.github_gateway`) and the cron
fallback poller (``scripts/pr-review-dispatch.py``), which share the same
on-disk dispatcher state file and the same exclusive flock, so neither
process double-dispatches and the synthesis sweep converges once.

Rationale for the behaviour here (see docs/plans/20260823_revB_multipass_fixes.md)
- H-4: the multipass settings live in config (``github_webhook`` section) and are
  honoured by the gateway (the primary path), not only the cron fallback.
- H-5: the ``multi_model_passes`` entry is persisted *before* tasks are
  dispatched; task lookup falls back to the task archive; an all-failed pass is
  re-issued once with an ``-r2`` suffix and then abandoned with an escalation log.
- M-1: ``model_slug`` uses the same mode-prefix rule as ``parse_fallback_entry``
  and keeps the mode in the slug so ``c:gpt-...`` and ``s:gpt-...`` don't collide.
- M-6: the synthesis sweep runs regardless of whether new dispatching is
  currently configured, so existing in-flight entries keep resolving.
"""

import fcntl
import json
import os
import re
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from core.memory.task_queue import TaskQueueManager
from core.paths import get_animas_dir
from core.tasks_dispatch import dispatch_direct_task

# Shared with the cron fallback — never rename without updating both callers.
STATE_FILENAME = "pr-review-dispatch-state.json"

# Terminal statuses copied from the poller so this module stays independent.
TERMINAL_TASK_STATUSES = frozenset({"failed", "cancelled", "done"})


# ── Dispatcher state (shared with the cron fallback) ───────────────────


def default_state() -> dict:
    """A complete, empty dispatcher state document."""
    return {
        "prs": {},
        "last_comment_check": _now_iso(),
        "seen_comments": {},
        "ci_notified": {},
        "ci_failure_signatures": {},
        "conflict_notified": {},
        "failed_task_retries": {},
        "review_tasks": {},
        "stale_watch": {},
        "consecutive_failures": 0,
    }


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_state(state_file: Path) -> dict:
    """Read dispatcher state, recovering to a complete default on any problem.

    Unknown/forward-schema fields are preserved so a newer writer's data
    survives an older reader round-trip.
    """
    try:
        raw = state_file.read_text(encoding="utf-8")
    except FileNotFoundError:
        return default_state()
    try:
        state = json.loads(raw) if raw.strip() else default_state()
    except json.JSONDecodeError:
        return default_state()
    if not isinstance(state, dict):
        return default_state()
    state.setdefault("prs", {})
    state.setdefault("seen_comments", {})
    state.setdefault("ci_notified", {})
    state.setdefault("ci_failure_signatures", {})
    state.setdefault("conflict_notified", {})
    state.setdefault("failed_task_retries", {})
    state.setdefault("review_tasks", {})
    state.setdefault("stale_watch", {})
    return state


def save_state(state_file: Path, state: dict) -> None:
    """Atomically persist the dispatcher state under the caller's lock."""
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=state_file.parent,
            prefix=f".{state_file.name}.",
            delete=False,
        ) as temp_handle:
            json.dump(state, temp_handle, indent=1, ensure_ascii=False)
            temp_handle.write("\n")
            temp_handle.flush()
            os.fsync(temp_handle.fileno())
            temp_path = Path(temp_handle.name)
        temp_path.replace(state_file)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


@contextmanager
def locked_state(state_file: Path) -> Iterator[dict]:
    """Lock the shared dispatcher state across a full read/modify/write cycle.

    A stable sidecar inode is locked while the JSON is atomically replaced.
    Both the gateway and the cron fallback use the same sidecar, so neither
    process can overwrite state loaded by the other.
    """
    state_file.parent.mkdir(parents=True, exist_ok=True)
    with state_file.with_suffix(".lock").open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            state = load_state(state_file)
            yield state
            save_state(state_file, state)
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


# ── Task helpers ───────────────────────────────────────────────────────


def _ci_task_id(repo: str, number: int, sha: str) -> str:
    return f"gh-ci-{repo.replace('/', '-')}#{number}-{sha[:8]}"


def model_slug(entry: str) -> str:
    """Normalize a ``mode:model`` entry into a lowercase ``[a-z0-9-]`` slug.

    Mirrors ``parse_fallback_entry``: only a one-letter prefix followed by
    ``:`` counts as an explicit mode, which preserves model tags such as
    ``ollama/qwen3:14b``.  The mode prefix (when present) is *kept* in the
    slug so ``c:gpt-5.6-sol`` and ``s:gpt-5.6-sol`` produce distinct,
    collision-free task ids and file names.
    """
    value = (entry or "").strip()
    if len(value) >= 2 and value[1] == ":":
        parts = (value[0], value[2:])
    else:
        parts = (value,)
    pieces = [re.sub(r"[^a-zA-Z0-9]+", "-", part).strip("-").lower() for part in parts if part]
    slug = re.sub(r"-+", "-", "-".join(pieces)).strip("-")
    return slug or "model"


def review_task(reviewer: str, task_id: str, animas_dir: Path | None = None):
    """Return the reviewer's task record, consulting the live queue then the archive.

    The anima heartbeat compacts terminal (done/cancelled/failed) tasks into
    ``state/task_queue_archive.jsonl``, so a pass can only be resolved via the
    archive after compaction.  An archive hit with ``done`` therefore counts as a
    completed pass (H-5).
    """
    target_dir = (animas_dir or get_animas_dir()) / reviewer
    if not target_dir.is_dir():
        return None
    tqm = TaskQueueManager(target_dir)
    task = tqm.get_task_by_id(task_id)
    if task is not None:
        return task
    archived_status = TaskQueueManager._search_archive(target_dir, task_id)
    if archived_status is None:
        return None
    return SimpleNamespace(task_id=task_id, status=archived_status)


# ── Review instruction templates ────────────────────────────────────


def review_instruction_base(quiet_seconds: float) -> str:
    """Baseline FRC instructions shown to a reviewer model pass."""
    return (
        f"最終pushから{int(quiet_seconds) // 60}分以上静穏を確認済みです。"
        "上記PRの current HEAD に対する差分レビュー/FRCを直ちに実施してください。"
        "過去HEADへのレビューは新push時点で無効です。"
        "2回目以降のレビューは収束ルール（heartbeat.md記載・2026-07-15 オーナー指示）に従い、"
        "前回blocking findingsの解消確認と新push差分に限定してください。"
        "full PRの再レビューをやり直さないこと。"
        "HOLD回数でレビューを打ち切らないこと（2026-08-28 オーナー指示で通算3回の停止ルールは撤廃）。"
        "レビューOK・CI OKに到達するまで続けるのが正であり、人間の回答待ちを停止理由にしないこと。"
        "同一blocking findingが未解消かつ新pushなしを2回連続で観測したときだけ、"
        "同じレビューの再投稿を止めてrinへ事実と選択肢を報告し、rinの判断で再開してください。"
        "複数件ある場合はbackgroundタスクとして並列に処理して構いません。"
    )


# ── Multipass dispatch ───────────────────────────────────────────────


def dispatch_multipass_reviews(
    state: dict,
    items: list[dict],
    *,
    reviewer: str,
    models: list[str],
    quiet_seconds: float = 180,
    dispatch: Callable[..., Any] = dispatch_direct_task,
    animas_dir: Path | None = None,
    logger: Callable[[str], None] | None = None,
) -> None:
    """Emit one review task per configured model, then arm the synthesis step.

    The ``multi_model_passes`` entry (including the task ids) is written to
    the state dict *before* any task is dispatched, so a crash or a per-task
    exception still leaves the entry behind for the next synthesis sweep
    (H-5).  A ``False`` return from *dispatch* (task already exists) is
    normal; an exception is logged and the loop continues.
    """
    if not models:
        return
    multipass = state.setdefault("multi_model_passes", {})
    log = logger or (lambda _msg: None)
    for it in items:
        base_id = _ci_task_id(it["repo"], it["number"], it["sha"])
        if base_id in multipass:
            continue
        sha = str(it["sha"])
        number = int(it["number"])
        repo = str(it["repo"])
        model_tasks = [f"{base_id}-m-{model_slug(entry)}" for entry in models]
        # Entry-first persistence: record before dispatching any task.
        multipass[base_id] = {
            "repo": repo,
            "number": number,
            "sha": sha,
            "models": list(models),
            "task_ids": model_tasks,
            "attempt": 1,
        }
        for entry in models:
            task_id = f"{base_id}-m-{model_slug(entry)}"
            instruction = (
                f"GitHub の {repo}#{number}（{it.get('title', '')}）の review 依頼。\n\n"
                + review_instruction_base(quiet_seconds)
                + f"\n\nこれは {entry} によるレビューパスである。最終判定（APPROVE操作等）はこのパスでは行わず、"
                "指摘の列挙に徹すること。原文の句読点を変えずに事実を挙げ、各指摘に file:line を添えること。"
                f"FRCファイルは reviews/pr{number}-frc-{reviewer}-{sha}-{model_slug(entry)}.md に書くこと。"
            )
            kwargs: dict[str, Any] = {
                "target": reviewer,
                "task_id": task_id,
                "summary": f"PRマルチパスレビュー {repo}#{number} ({entry})",
                "instruction": instruction,
                "model": entry,
                "meta": {"repo": repo, "number": number, "sha": sha, "model": entry, "multipass": True},
            }
            if animas_dir is not None:
                kwargs["animas_dir"] = animas_dir
            try:
                dispatch(**kwargs)
            except Exception as exc:  # noqa: BLE001 - keep sweeping
                log(f"multi-pass dispatch error {task_id}: {exc}")
        log(
            f"multi-pass review dispatch -> {reviewer}: {repo}#{number} "
            f"models={','.join(models)} tasks={','.join(model_tasks)}"
        )


def check_multipass_synth(
    state: dict,
    *,
    reviewer: str,
    synth_model: str | None,
    quiet_seconds: float = 180,
    dispatch: Callable[..., Any] = dispatch_direct_task,
    animas_dir: Path | None = None,
    logger: Callable[[str], None] | None = None,
) -> None:
    """Dispatch the final synthesis task once every model pass is terminal.

    Runs regardless of whether new model-pass dispatch is currently
    configured (M-6): existing in-flight entries are still swept after the
    config is emptied.  If every pass is terminal but none is done, the pass
    set is re-issued once with an ``-r2`` suffix (attempt=2); if the retry
    also finishes without a single done, the entry is dropped with an
    escalation log.  ``notified_sha`` remains set, so nothing loops forever.
    """
    multipass = state.setdefault("multi_model_passes", {})
    log = logger or (lambda _msg: None)
    for base_id, info in list(multipass.items()):
        records = {tid: review_task(reviewer, tid, animas_dir=animas_dir) for tid in info["task_ids"]}
        if any(rec is None or rec.status not in TERMINAL_TASK_STATUSES for rec in records.values()):
            continue  # still running or unpublished; wait for the next sweep

        done = [tid for tid, rec in records.items() if rec.status == "done"]
        repo = str(info["repo"])
        number = int(info["number"])
        sha = str(info["sha"])
        models = list(info.get("models") or [])

        if done:
            failed = [tid for tid, rec in records.items() if rec.status != "done"]
            synth_id = f"{base_id}-synth"
            file_paths = "\n".join(f"- reviews/pr{number}-frc-{reviewer}-{sha}-{model_slug(m)}.md" for m in models)
            failed_notes = (
                "\n\n注意: 以下のモデルパスは失敗（terminalだがdoneでない）したためファイルが欠落している。"
                f"\n{chr(10).join('- ' + t for t in failed)}"
                if failed
                else ""
            )
            instruction = (
                f"各モデルパスのFRCファイル（下記パス）を読み、指摘をマージ・重複排除し、最終判定を "
                f"reviews/pr{number}-frc-{reviewer}-{sha}.md （従来のファイル名）に書く。"
                "GitHub上のレビュー操作（APPROVE・CHANGES_REQUESTED等）はこの統合パスで行う。"
                f"\n\n対象ファイル:\n{file_paths}\n"
            ) + failed_notes
            kwargs: dict[str, Any] = {
                "target": reviewer,
                "task_id": synth_id,
                "summary": f"PRレビュー統合判定 {repo}#{number}",
                "instruction": instruction,
                "model": synth_model,
                "meta": {"repo": repo, "number": number, "sha": sha, "multipass": "synth"},
            }
            if animas_dir is not None:
                kwargs["animas_dir"] = animas_dir
            try:
                dispatch(**kwargs)
                log(f"multi-pass synth dispatch -> {reviewer}: {synth_id} models={','.join(models)}")
            except Exception as exc:  # noqa: BLE001
                log(f"multi-pass synth dispatch error {synth_id}: {exc}")
            # Drop the entry so state stays bounded to in-flight PRs.
            multipass.pop(base_id, None)
            continue

        # All passes terminal but none done: re-issue once, then abandon.
        attempt = int(info.get("attempt", 1))
        if attempt < 2:
            retry_tasks = [f"{base_id}-m-{model_slug(m)}-r2" for m in models]
            info["task_ids"] = retry_tasks
            info["attempt"] = 2
            for entry in models:
                task_id = f"{base_id}-m-{model_slug(entry)}-r2"
                instruction = (
                    f"前回の全モデルパスが完了せず（失敗）再投入される。\n"
                    f"GitHub の {repo}#{number} の review 依頼。\n\n"
                    + review_instruction_base(quiet_seconds)
                    + f"\n\nこれは {entry} によるレビューパス（再試行）である。最終判定（APPROVE操作等）はこのパスでは行わず、"
                    "指摘の列挙に徹すること。原文の句読点を変えずに事実を挙げ、各指摘に file:line を添えること。"
                    f"FRCファイルは reviews/pr{number}-frc-{reviewer}-{sha}-{model_slug(entry)}-r2.md に書くこと。"
                )
                kwargs = {
                    "target": reviewer,
                    "task_id": task_id,
                    "summary": f"PRマルチパスレビュー再試行 {repo}#{number} ({entry})",
                    "instruction": instruction,
                    "model": entry,
                    "meta": {"repo": repo, "number": number, "sha": sha, "model": entry, "multipass": True},
                }
                if animas_dir is not None:
                    kwargs["animas_dir"] = animas_dir
                try:
                    dispatch(**kwargs)
                except Exception as exc:  # noqa: BLE001
                    log(f"multi-pass retry dispatch error {task_id}: {exc}")
            log(f"multi-pass review all failed -> reissue -r2 for {repo}#{number} tasks={','.join(retry_tasks)}")
        else:
            log(
                f"MULTI-PASS ALL MODEL PASSES FAILED (attempt {attempt}) for {repo}#{number}: "
                f"models={','.join(models)} tasks={','.join(info['task_ids'])}. "
                "ESCALATION: manual/human review required."
            )
            multipass.pop(base_id, None)
