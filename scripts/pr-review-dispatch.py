#!/usr/bin/env python3
"""Fallback PR review dispatcher for a 15-minute cron schedule.

The GitHub webhook gateway is the primary detector.  This poller retains the
existing recovery path and shares its state and flock with the gateway so a
delivery made by either process is not repeated by the other.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

# Deployment-specific values come from the environment so no site-specific
# repos, accounts, or paths are baked into the public source tree.
SHARED_DIR = Path(os.environ.get("ANIMAWORKS_SHARED_DIR", "~/.animaworks/shared")).expanduser()
STATE_FILE = SHARED_DIR / "pr-review-dispatch-state.json"
STATE_LOCK = STATE_FILE.with_suffix(".lock")
LOG_FILE = SHARED_DIR / "pr-review-dispatch.log"
GH_CONFIG_DIR = os.environ.get("GH_CONFIG_DIR", str(SHARED_DIR / "gh-bot"))

# Comma-separated "owner/repo" list; the poller refuses to run without it.
REPOS = [r.strip() for r in os.environ.get("PR_DISPATCH_REPOS", "").split(",") if r.strip()]
QUIET_SECONDS = float(os.environ.get("PR_DISPATCH_QUIET_SECONDS", "180"))
# Bot account whose own comments are ignored; empty disables the exclusion.
BOT_LOGIN = os.environ.get("PR_DISPATCH_BOT_LOGIN", "")
# Dedicated review-bot login (e.g. animaworks-reviewer); treated like BOT_LOGIN.
REVIEWER_LOGIN = os.environ.get("PR_DISPATCH_REVIEWER_LOGIN", "")
REVIEWER = os.environ.get("PR_DISPATCH_REVIEWER", "sumire")
# Multi-pass FRC review: comma-separated "mode:model" list emitted as one review
# pass per entry.  Unset/empty keeps the historic single (model-less) dispatch.
PR_DISPATCH_REVIEW_MODELS = [
    e.strip() for e in os.environ.get("PR_DISPATCH_REVIEW_MODELS", "").split(",") if e.strip()
]
# Model used for the final synthesis pass; ``None`` falls back to the reviewer
# default model.
PR_DISPATCH_SYNTH_MODEL = os.environ.get("PR_DISPATCH_SYNTH_MODEL", "").strip() or None
DISPATCHER = os.environ.get("PR_DISPATCH_DISPATCHER", "rin")
FIXER = os.environ.get("PR_DISPATCH_FIXER", "natsume")
ESCALATION_TARGET = os.environ.get("PR_DISPATCH_ESCALATION", "sakura")
ALERT_EVERY = 5

# Stale unaddressed-review warning thresholds (hours).
# 2026-07-27 taka指示: 15分警告 → 以後15分毎に再警告(30/45分) → 60分でsakuraエスカレーション
STALE_WARN_HOURS = float(os.environ.get("PR_STALE_WARN_HOURS", "0.25"))
STALE_REWARN_HOURS = float(os.environ.get("PR_STALE_REWARN_HOURS", "0.25"))
STALE_ESCALATE_HOURS = float(os.environ.get("PR_STALE_ESCALATE_HOURS", "1"))

# When set (1/true/yes), send() logs instead of delivering DMs.
DRY_RUN = os.environ.get("PR_DISPATCH_DRY_RUN", "").strip().lower() in ("1", "true", "yes")

# Auxiliary body match for mention-gated comments.  A mention without these
# words (e.g. "LGTMです @reviewer") is not a fix request.
FIX_REQUEST_PATTERN = re.compile(
    r"修正|直して|対応して|お願いします|fix|please|change|address|required",
    re.IGNORECASE,
)

sys.path.insert(
    0,
    os.environ.get("ANIMAWORKS_REPO_ROOT", str(Path(__file__).resolve().parents[1])),
)

from core.memory.task_queue import TaskQueueManager
from core.paths import get_animas_dir
from core.tasks_dispatch import FAILING_CI_CONCLUSIONS, dispatch_direct_task

MAX_FAILED_REDISPATCHES = 2
HUMAN_ESCALATION_HOURS = 24
# Reality-check redispatch: terminal task + problem still observed on GitHub = whiff (空振り).
TERMINAL_TASK_STATUSES = frozenset({"failed", "cancelled", "done"})
ACTIVE_TASK_STATUSES = frozenset({"waiting", "in_progress", "pending"})
REDISPATCH_COOLDOWN = timedelta(minutes=30)
# Reminders decay in frequency but never stop (taka指示 2026-08-13: 抑制禁止).
REWARN_CAP_HOURS = 4.0
ACTIVE_TASK_REWARN_FLOOR_HOURS = 1.0
ESCALATE_REPEAT_HOURS = 8.0
# Patrol liveness: alert when the LLM patrol stops leaving audit artifacts.
PATROL_GLOB = os.environ.get("PR_DISPATCH_PATROL_GLOB", "")
PATROL_MAX_AGE_HOURS = float(os.environ.get("PR_DISPATCH_PATROL_MAX_AGE_HOURS", "3"))
PATROL_ALERT_REPEAT_HOURS = 6.0


def is_our_bot(login: str) -> bool:
    """True when *login* is BOT_LOGIN or REVIEWER_LOGIN (our bot accounts)."""
    if not login:
        return False
    return (bool(BOT_LOGIN) and login == BOT_LOGIN) or (bool(REVIEWER_LOGIN) and login == REVIEWER_LOGIN)


def collect_mention_logins(*values: str | None) -> tuple[str, ...]:
    """Deduplicate mention handles.  Leading @ is stripped; blanks are skipped."""
    seen: set[str] = set()
    logins: list[str] = []
    for raw in values:
        login = (raw or "").strip().lstrip("@")
        if not login:
            continue
        key = login.casefold()
        if key in seen:
            continue
        seen.add(key)
        logins.append(login)
    return tuple(logins)


def _github_webhook_mention_logins() -> tuple[str, ...]:
    """bot_login / reviewer_login / implementer_anima from github_webhook config."""
    try:
        from core.config.models import load_config

        cfg = load_config().github_webhook
    except Exception:
        return ()
    return collect_mention_logins(
        getattr(cfg, "bot_login", None),
        getattr(cfg, "reviewer_login", None),
        getattr(cfg, "implementer_anima", None),
    )


def configured_mention_logins() -> tuple[str, ...]:
    """Handles that count as a directed mention (env PR_DISPATCH_* + github_webhook)."""
    return collect_mention_logins(
        BOT_LOGIN,
        REVIEWER_LOGIN,
        FIXER,
        *_github_webhook_mention_logins(),
    )


def body_mentions_any(body: str, logins: Iterable[str]) -> bool:
    """True when *body* contains ``@login`` for any configured handle (case-insensitive)."""
    text = body.casefold()
    return any(f"@{login.lstrip('@')}".casefold() in text for login in logins if login and login.strip())


def is_fix_request(
    *,
    body: str = "",
    review_state: str | None = None,
    mention_logins: Iterable[str] = (),
) -> bool:
    """True when an item should be tracked as a stale-watch fix request.

    OR of:
    1. review state is CHANGES_REQUESTED (structured; body ignored)
    2. body mentions a configured bot/implementer handle AND matches FIX_REQUEST_PATTERN
    """
    if str(review_state or "").upper() == "CHANGES_REQUESTED":
        return True
    if not body_mentions_any(body, mention_logins):
        return False
    return bool(FIX_REQUEST_PATTERN.search(body))


def now_utc() -> datetime:
    return datetime.now(UTC)


def iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    with LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write(f"[{iso(now_utc())}] {msg}\n")


def gh(args: list[str]) -> str:
    env = dict(os.environ, GH_CONFIG_DIR=GH_CONFIG_DIR)
    proc = subprocess.run(
        ["gh", *args],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"gh {' '.join(args[:3])}... failed: {proc.stderr.strip()[:500]}")
    return proc.stdout


def send(to: str, content: str) -> None:
    if DRY_RUN:
        log(f"DRY_RUN send -> {to}: {content[:300]}")
        return
    from core.messenger import Messenger

    Messenger(SHARED_DIR, "pr-review-dispatch").send(
        to=to,
        content=content,
        intent="report",
        skip_logging=True,
        meta={"source": "pr-review-dispatch.py"},
        source="system",
    )


def dispatch_task(**kwargs: Any) -> bool:
    """Dispatch unless this poller is running in dry-run mode."""
    if DRY_RUN:
        log(f"DRY_RUN task -> {kwargs['target']}: {kwargs['task_id']} {kwargs['summary']}")
        return False
    return dispatch_direct_task(**kwargs)


def default_state() -> dict:
    return {
        "prs": {},
        "last_comment_check": iso(now_utc()),
        "seen_comments": {},
        "ci_notified": {},
        "ci_failure_signatures": {},
        "conflict_notified": {},
        "failed_task_retries": {},
        "review_tasks": {},
        "stale_watch": {},
        "consecutive_failures": 0,
    }


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            if isinstance(state, dict):
                state.setdefault("prs", {})
                state.setdefault("seen_comments", {})
                state.setdefault("ci_notified", {})
                state.setdefault("ci_failure_signatures", {})
                state.setdefault("conflict_notified", {})
                state.setdefault("failed_task_retries", {})
                state.setdefault("review_tasks", {})
                state.setdefault("stale_watch", {})
                return state
        except (json.JSONDecodeError, OSError):
            log("state file unreadable; starting fresh")
    return default_state()


def parse_gh_time(value: str | None) -> datetime | None:
    """Parse GitHub ISO timestamps into timezone-aware UTC datetimes."""
    if not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def is_addressed(
    *,
    pr_closed: bool,
    thread_resolved: bool = False,
    item_created_at: datetime | None = None,
    bot_commit_at: datetime | None = None,
    bot_comment_at: datetime | None = None,
    kind: str = "comment",
    review_dismissed: bool = False,
    review_decision: str | None = None,
    ci_still_failing: bool = True,
    head_sha_changed: bool = False,
) -> bool:
    """Return True when the fix request no longer needs tracking.

    kind:
      - review: CHANGES_REQUESTED stays open until PR close, dismiss, or
        reviewDecision leaves CHANGES_REQUESTED. Bot commits/comments do NOT clear it.
      - thread / comment: resolve, subsequent bot commit, or bot reply clears it.
      - ci: cleared when PR closes, CI no longer failing on the item SHA, or
        head SHA moved on (new failing SHA becomes a fresh item).
    """
    if pr_closed:
        return True
    if kind == "review":
        if review_dismissed:
            return True
        decision = (review_decision or "").upper()
        return bool(decision and decision != "CHANGES_REQUESTED")
    if kind == "ci":
        # Old SHA items are retired when head moves; a failing new SHA is a new item_id.
        if head_sha_changed:
            return True
        return not ci_still_failing
    # thread / comment (default)
    if thread_resolved:
        return True
    if item_created_at is None:
        return False
    if bot_commit_at is not None and bot_commit_at > item_created_at:
        return True
    return bot_comment_at is not None and bot_comment_at > item_created_at


def ci_stale_item_id(repo: str, number: int, sha: str) -> str:
    """Stable stale-watch key for a CI failure on a specific PR head SHA."""
    return f"ci:{repo}#{number}:{sha}"


def _ci_task_id(repo: str, number: int, sha: str) -> str:
    return f"gh-ci-{repo.replace('/', '-')}#{number}-{sha[:8]}"


def _conflict_task_id(repo: str, number: int, sha: str) -> str:
    return f"gh-conflict-{repo.replace('/', '-')}#{number}-{sha[:8]}"


def _direct_task(task_id: str):
    if DRY_RUN:
        return None
    target_dir = get_animas_dir() / FIXER
    if not target_dir.is_dir():
        return None
    return TaskQueueManager(target_dir).get_task_by_id(task_id)


def _attempt_task_id(base_id: str, attempt: int) -> str:
    """Attempt 1 keeps the base ID; retries get an -rN suffix so history survives."""
    return base_id if attempt <= 1 else f"{base_id}-r{attempt}"


def _latest_attempt_task(base_id: str, retries: dict):
    """Return the most recent attempt's task record for a base task ID."""
    attempts = int(retries.get(base_id, 0))
    for attempt in range(attempts + 1, 0, -1):
        task = _direct_task(_attempt_task_id(base_id, attempt))
        if task is not None:
            return task
    return None


def _task_updated_at(task) -> datetime | None:
    return parse_gh_time(getattr(task, "updated_at", None) or getattr(task, "ts", None))


def reopen_stalled_dispatches(state: dict) -> None:
    """Release CI/conflict latches whose task ended (failed/cancelled/done alike).

    Popping a latch is safe: check_ci/check_conflicts only redispatch when the
    failure or conflict is *currently observed* on GitHub, so resolved problems
    never respawn. A done task whose problem persists is a whiff (空振り) and
    gets redispatched with its history injected into the new instruction.
    """
    retries = state.setdefault("failed_task_retries", {})
    retry_context = state.setdefault("retry_context", {})
    candidates: list[tuple[str, str, str]] = []
    for key in state.setdefault("ci_notified", {}):
        try:
            repo_number, sha = key.rsplit("_", 1)
            repo, number = repo_number.rsplit("#", 1)
            candidates.append(("ci_notified", key, _ci_task_id(repo, int(number), sha)))
        except (ValueError, TypeError):
            continue
    for key, sha in state.setdefault("conflict_notified", {}).items():
        try:
            repo, number = key.rsplit("#", 1)
            candidates.append(("conflict_notified", key, _conflict_task_id(repo, int(number), str(sha))))
        except (ValueError, TypeError):
            continue

    now = now_utc()
    for latch_name, key, base_id in candidates:
        task = _latest_attempt_task(base_id, retries)
        if task is None or task.status not in TERMINAL_TASK_STATUSES:
            continue
        attempts = int(retries.get(base_id, 0))
        if attempts >= MAX_FAILED_REDISPATCHES:
            continue
        updated = _task_updated_at(task)
        if updated is not None and now - updated < REDISPATCH_COOLDOWN:
            continue
        state[latch_name].pop(key, None)
        retries[base_id] = attempts + 1
        retry_context[base_id] = {
            "last_task_id": task.task_id,
            "last_status": task.status,
            "at": iso(now),
        }


def _whiff_preamble(base_id: str, state: dict) -> str:
    """History-of-failure prefix for a redispatched task instruction."""
    attempts = int(state.get("failed_task_retries", {}).get(base_id, 0))
    if attempts <= 0:
        return ""
    context = state.get("retry_context", {}).get(base_id, {})
    prev_id = context.get("last_task_id", base_id)
    prev_status = context.get("last_status", "?")
    if prev_status == "done":
        detail = (
            f"前回タスク {prev_id} は done と宣言されたが、GitHub上の問題は解消されていない。"
            "宣言と現実が食い違っている＝前回の対応は空振りだった。"
        )
    else:
        detail = f"前回タスク {prev_id} は {prev_status} で終わり、問題は放置されたままだ。"
    return (
        f"【再投入 {attempts + 1}回目 — 前回は空振り】{detail}\n"
        "今回は必ず (a) 問題を実際に解消して現実（CI/mergeable）を変える、"
        "(b) 解消不能な構造的理由をPRコメントに明記してrinへ報告する、"
        "のいずれかを完遂せよ。現実を変えないままのdone宣言は許されない。\n\n"
    )


def failed_check_names(status_check_rollup: list[dict[str, Any]] | None) -> list[str]:
    """Return names of checks whose conclusion counts as failing (see FAILING_CI_CONCLUSIONS)."""
    return [
        str(check.get("name") or "?")
        for check in (status_check_rollup or [])
        if str(check.get("conclusion") or "").upper() in FAILING_CI_CONCLUSIONS
    ]


GREEN_CI_CONCLUSIONS = frozenset({"SUCCESS", "NEUTRAL", "SKIPPED"})


def ci_all_green(status_check_rollup: list[dict[str, Any]] | None) -> bool:
    """True when every check is terminal and none is failing (rollup mixes CheckRun/StatusContext)."""
    rollup = status_check_rollup or []
    if not rollup:
        return False
    for check in rollup:
        if "state" in check:  # StatusContext (commit status)
            if str(check.get("state") or "").upper() != "SUCCESS":
                return False
        else:  # CheckRun
            if str(check.get("status") or "").upper() != "COMPLETED":
                return False
            if str(check.get("conclusion") or "").upper() not in GREEN_CI_CONCLUSIONS:
                return False
    return True


def decayed_interval_hours(base_hours: float, count: int, cap_hours: float = REWARN_CAP_HOURS) -> float:
    """Reminder interval doubles per prior reminder, capped — decays but never stops."""
    return min(base_hours * (2 ** max(0, count - 1)), cap_hours)


def determine_warning_stage(
    *,
    item_created_at: datetime,
    now: datetime,
    last_warned: datetime | None,
    escalated_at: datetime | None,
    warn_hours: float = 0.25,
    rewarn_hours: float = 0.25,
    escalate_hours: float = 1.0,
    warn_count: int = 0,
) -> str:
    """Return warning stage: none | warn | rewarn | escalate.

    Reminders never terminate: rewarns continue at a decaying interval
    (rewarn_hours doubling per reminder, capped at REWARN_CAP_HOURS), and
    escalation repeats every ESCALATE_REPEAT_HOURS while the item persists.
    """
    age = now - item_created_at
    if age < timedelta(hours=warn_hours):
        return "none"

    escalate_due = age >= timedelta(hours=escalate_hours) and (
        escalated_at is None or now - escalated_at >= timedelta(hours=ESCALATE_REPEAT_HOURS)
    )
    if last_warned is None:
        dispatcher_stage = "warn"
    elif (now - last_warned) >= timedelta(hours=decayed_interval_hours(rewarn_hours, warn_count)):
        dispatcher_stage = "rewarn"
    else:
        dispatcher_stage = "none"

    if escalate_due:
        return "escalate"
    return dispatcher_stage


def _elapsed_label(created_at: datetime, now: datetime) -> str:
    minutes = max(0, int((now - created_at).total_seconds() // 60))
    return f"{minutes // 60}h{minutes % 60:02d}m" if minutes >= 60 else f"{minutes}m"


def _format_stale_line(
    *,
    repo: str,
    number: int,
    author: str,
    body: str,
    url: str,
    created_at: datetime,
    now: datetime,
    kind: str = "comment",
    sha: str = "",
    failed_checks: list[str] | None = None,
    note: str = "",
) -> str:
    elapsed = _elapsed_label(created_at, now)
    if kind == "review":
        return f"- PR #{number} が CHANGES_REQUESTED のまま未解除です（@{author}/経過{elapsed}）{note}\n  {url}"
    if kind == "ci":
        checks = ", ".join((failed_checks or [])[:6]) or "?"
        sha8 = (sha or "")[:8] or "?"
        return (
            f"- PR #{number} のCI失敗が未修正のまま放置されています（{sha8}・{checks}・経過{elapsed}）{note}\n  {url}"
        )
    snippet = (body or "").replace("\n", " ").strip()[:140]
    return f"- {repo}#{number} (経過{elapsed}) @{author}: {snippet}{note}\n  {url}"


def _urgency_header(max_age_hours: float) -> str:
    """Tone escalates with age — the point is to make staleness feel like an emergency."""
    if max_age_hours >= 12:
        return (
            f"【異常事態】{int(max_age_hours)}時間以上止まっているPRがあります。"
            "通常フローは機能していません。フローの外に出てでも解決してください"
        )
    if max_age_hours >= 2:
        return "【警告・長期化】放置が長引いています。今のやり方では解決していません"
    return "【警告】"


def _stale_message(lines: list[str], *, kind: str = "comment", max_age_hours: float = 0.0) -> str:
    detail = "\n".join(lines[:20])
    more = f"\n…他{len(lines) - 20}件" if len(lines) > 20 else ""
    header = _urgency_header(max_age_hours)
    if kind == "review":
        return (
            f"{header}\nPR が CHANGES_REQUESTED のまま未解除です\n\n"
            f"{detail}{more}\n\n"
            "修正を積んだ場合は reviewer に再レビューを依頼するか、"
            "対応方針をPRコメントで明示してください"
        )
    if kind == "ci":
        return (
            f"{header}\nPR のCI失敗が未修正のまま放置されています\n\n"
            f"{detail}{more}\n\n"
            "修正commitをpushするか、対応不能ならその理由をPRコメントに残してください。"
            "「タスクはdoneになっている」は解決ではありません。CIが赤い限り問題は残っています"
        )
    return (
        f"{header}\nPR修正依頼が未対応です\n\n"
        f"{detail}{more}\n\n"
        "修正commit・返信・resolveのいずれかで対応するか、"
        "対応しない理由をスレッドへ返信してください"
    )


def _latest_bot_activity(
    *,
    commits: list[dict[str, Any]],
    comments: list[dict[str, Any]],
) -> tuple[datetime | None, datetime | None]:
    """Return (latest bot commit time, latest bot comment time) on a PR."""
    bot_commit_at: datetime | None = None
    bot_comment_at: datetime | None = None
    if BOT_LOGIN or REVIEWER_LOGIN:
        for commit in commits:
            author = (commit.get("author") or {}).get("login") or ""
            committer = (commit.get("committer") or {}).get("login") or ""
            if not is_our_bot(author) and not is_our_bot(committer):
                continue
            committed = parse_gh_time(
                ((commit.get("commit") or {}).get("committer") or {}).get("date")
            ) or parse_gh_time(((commit.get("commit") or {}).get("author") or {}).get("date"))
            if committed is not None and (bot_commit_at is None or committed > bot_commit_at):
                bot_commit_at = committed
    for comment in comments:
        author = (comment.get("user") or {}).get("login") or (comment.get("author") or {}).get("login", "")
        if not is_our_bot(author):
            continue
        created = parse_gh_time(comment.get("created_at") or comment.get("createdAt") or comment.get("submitted_at"))
        if created is not None and (bot_comment_at is None or created > bot_comment_at):
            bot_comment_at = created
    return bot_commit_at, bot_comment_at


def _dispatch_review_task_once(state: dict, item: dict[str, Any]) -> None:
    review_id = str(item["review_id"])
    review_tasks = state.setdefault("review_tasks", {})
    if review_id in review_tasks:
        return
    repo = str(item["repo"])
    number = int(item["number"])
    author = str(item.get("author") or "unknown")
    body = str(item.get("body") or "")[:500]
    bot_note = "bot由来のCHANGES_REQUESTEDです。" if item.get("bot_derived") else "人間レビュアー由来です。"
    dispatch_task(
        target=FIXER,
        task_id=f"gh-review-{repo.replace('/', '-')}#{number}-{review_id}",
        summary=f"レビュー指摘対応 {repo}#{number}",
        instruction=(
            f"PR #{number} ({item.get('url') or f'https://github.com/{repo}/pull/{number}'}) に "
            f"@{author} から CHANGES_REQUESTED が投稿された。{bot_note}\n\n"
            f"レビュー本文:\n{body}\n\n"
            "指摘を確認して必要な修正を行うこと。レビュアーが人間の場合、指摘に技術的に"
            "同意できない時は独断で押し切らず上長(rin)へ報告して判断を仰ぐこと。"
        ),
        meta={
            "repo": repo,
            "number": number,
            "review_id": review_id,
            "reviewer": author,
            "url": item.get("url") or "",
            "bot_derived": bool(item.get("bot_derived")),
        },
    )
    review_tasks[review_id] = iso(now_utc())


def _direct_task_for_stale_item(item: dict[str, Any], retries: dict | None = None):
    if item.get("kind") not in {"ci", "conflict"}:
        return None
    repo = str(item["repo"])
    number = int(item["number"])
    sha = str(item.get("sha") or "")
    task_ids = (
        (_ci_task_id(repo, number, sha), _conflict_task_id(repo, number, sha))
        if item.get("kind") == "ci"
        else (_conflict_task_id(repo, number, sha), _ci_task_id(repo, number, sha))
    )
    for task_id in task_ids:
        task = _latest_attempt_task(task_id, retries or {})
        if task is not None:
            return task
    return None


def _collect_pr_stale_items(
    repo: str,
    number: int,
    *,
    pr_meta: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Collect unaddressed fix-request / CI-failure candidates for one open PR.

    pr_meta may include headRefOid, statusCheckRollup, reviewDecision, url from
    a single `gh pr list --json` call to avoid extra API round-trips.
    """
    owner, _, name = repo.partition("/")
    items: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    meta = pr_meta or {}
    review_decision = str(meta.get("reviewDecision") or "")
    head_sha = str(meta.get("headRefOid") or "")
    pr_url = str(meta.get("url") or f"https://github.com/{repo}/pull/{number}")

    reviews = json.loads(gh(["api", f"repos/{repo}/pulls/{number}/reviews", "--paginate"]))
    issue_comments = json.loads(gh(["api", f"repos/{repo}/issues/{number}/comments?per_page=100"]))
    review_comments = json.loads(gh(["api", f"repos/{repo}/pulls/{number}/comments?per_page=100"]))
    commits = json.loads(gh(["api", f"repos/{repo}/pulls/{number}/commits", "--paginate"]))

    thread_query = (
        "query($owner:String!,$name:String!,$number:Int!){"
        "repository(owner:$owner,name:$name){"
        "pullRequest(number:$number){"
        "reviewThreads(first:100){nodes{id isResolved "
        "comments(last:1){nodes{author{login} body createdAt url}}}}}}}"
    )
    graphql = json.loads(
        gh(
            [
                "api",
                "graphql",
                "-f",
                f"query={thread_query}",
                "-F",
                f"owner={owner}",
                "-F",
                f"name={name}",
                "-F",
                f"number={number}",
            ]
        )
    )
    threads = (
        (((graphql.get("data") or {}).get("repository") or {}).get("pullRequest") or {}).get("reviewThreads") or {}
    ).get("nodes") or []

    mention_logins = configured_mention_logins()
    all_comment_like: list[dict[str, Any]] = list(issue_comments) + list(review_comments)
    for review in reviews:
        if review.get("body"):
            all_comment_like.append(
                {
                    "user": review.get("user"),
                    "created_at": review.get("submitted_at"),
                    "body": review.get("body"),
                }
            )
    bot_commit_at, bot_comment_at = _latest_bot_activity(commits=commits, comments=all_comment_like)

    for review in reviews:
        state_upper = str(review.get("state", "")).upper()
        if state_upper == "DISMISSED":
            continue
        # Reviews qualify only via structured state; body mention gating is for comments.
        if not is_fix_request(review_state=state_upper):
            continue
        author = (review.get("user") or {}).get("login", "")
        created = parse_gh_time(review.get("submitted_at"))
        if created is None:
            continue
        item_id = f"review:{review.get('id')}"
        if item_id in seen_ids:
            continue
        seen_ids.add(item_id)
        items.append(
            {
                "item_id": item_id,
                "kind": "review",
                "review_id": review.get("id"),
                "repo": repo,
                "number": number,
                "author": author,
                "body": review.get("body") or "(CHANGES_REQUESTED)",
                "url": review.get("html_url") or pr_url,
                "created_at": created,
                "thread_resolved": False,
                "bot_commit_at": bot_commit_at,
                "bot_comment_at": bot_comment_at,
                "review_dismissed": False,
                "review_decision": review_decision,
                "bot_derived": is_our_bot(author),
                "skip_stale": is_our_bot(author),
            }
        )

    for thread in threads:
        if thread.get("isResolved"):
            continue
        last_comments = ((thread.get("comments") or {}).get("nodes")) or []
        if not last_comments:
            continue
        last = last_comments[-1]
        author = (last.get("author") or {}).get("login", "")
        if is_our_bot(author):
            continue
        created = parse_gh_time(last.get("createdAt"))
        if created is None:
            continue
        item_id = f"thread:{thread.get('id')}"
        if item_id in seen_ids:
            continue
        seen_ids.add(item_id)
        items.append(
            {
                "item_id": item_id,
                "kind": "thread",
                "repo": repo,
                "number": number,
                "author": author,
                "body": last.get("body") or "",
                "url": last.get("url") or pr_url,
                "created_at": created,
                "thread_resolved": False,
                "bot_commit_at": bot_commit_at,
                "bot_comment_at": bot_comment_at,
            }
        )

    for comment in list(issue_comments) + list(review_comments):
        author = (comment.get("user") or {}).get("login", "")
        if is_our_bot(author):
            continue
        body = comment.get("body") or ""
        if not is_fix_request(body=body, mention_logins=mention_logins):
            continue
        created = parse_gh_time(comment.get("created_at"))
        if created is None:
            continue
        item_id = f"comment:{comment.get('id')}"
        if item_id in seen_ids:
            continue
        seen_ids.add(item_id)
        items.append(
            {
                "item_id": item_id,
                "kind": "comment",
                "repo": repo,
                "number": number,
                "author": author,
                "body": body,
                "url": comment.get("html_url") or pr_url,
                "created_at": created,
                "thread_resolved": False,
                "bot_commit_at": bot_commit_at,
                "bot_comment_at": bot_comment_at,
            }
        )

    # CI failure on current head SHA (persistent rewarn path; check_ci handles first-shot).
    failed = failed_check_names(meta.get("statusCheckRollup"))
    if head_sha and failed:
        item_id = ci_stale_item_id(repo, number, head_sha)
        if item_id not in seen_ids:
            seen_ids.add(item_id)
            items.append(
                {
                    "item_id": item_id,
                    "kind": "ci",
                    "repo": repo,
                    "number": number,
                    "author": "ci",
                    "body": ", ".join(failed[:6]),
                    "url": pr_url,
                    "created_at": None,  # age uses watch first_seen
                    "sha": head_sha,
                    "failed_checks": failed,
                    "ci_still_failing": True,
                    "head_sha_changed": False,
                    "thread_resolved": False,
                    "bot_commit_at": None,
                    "bot_comment_at": None,
                }
            )

    return items


def check_unaddressed(state: dict) -> None:
    """Warn on unaddressed external fix requests / CI failures; escalate long-running ones."""
    watch = state.setdefault("stale_watch", {})
    ci_notified = state.setdefault("ci_notified", {})
    now = now_utc()
    active_ids: set[str] = set()
    # kind -> lines for dispatcher / escalate (separate message templates)
    dispatcher_by_kind: dict[str, list[str]] = {"review": [], "ci": [], "comment": []}
    escalate_by_kind: dict[str, list[str]] = {"review": [], "ci": [], "comment": []}
    human_by_kind: dict[str, list[str]] = {"review": [], "ci": [], "comment": []}
    max_age_by_kind: dict[str, float] = {"review": 0.0, "ci": 0.0, "comment": 0.0}
    open_pr_count = 0

    for repo in REPOS:
        prs = json.loads(
            gh(
                [
                    "pr",
                    "list",
                    "--repo",
                    repo,
                    "--state",
                    "open",
                    "--json",
                    "number,headRefOid,statusCheckRollup,reviewDecision,url",
                    "--limit",
                    "100",
                ]
            )
        )
        open_pr_count += len(prs)
        for pr in prs:
            number = pr["number"]
            for item in _collect_pr_stale_items(repo, number, pr_meta=pr):
                kind = str(item.get("kind") or "comment")
                if is_addressed(
                    pr_closed=False,
                    thread_resolved=bool(item.get("thread_resolved")),
                    item_created_at=item.get("created_at"),
                    bot_commit_at=item.get("bot_commit_at"),
                    bot_comment_at=item.get("bot_comment_at"),
                    kind=kind,
                    review_dismissed=bool(item.get("review_dismissed")),
                    review_decision=item.get("review_decision"),
                    ci_still_failing=bool(item.get("ci_still_failing", True)),
                    head_sha_changed=bool(item.get("head_sha_changed")),
                ):
                    continue
                if kind == "review":
                    _dispatch_review_task_once(state, item)
                    if item.get("skip_stale"):
                        continue
                item_id = item["item_id"]
                active_ids.add(item_id)
                entry = watch.get(item_id)
                if entry is None:
                    # CI: if check_ci already notified this SHA, treat that as first warn
                    # so stale side starts rewarn after REWARN interval (no duplicate immediate alert).
                    already_ci_notified = False
                    if kind == "ci":
                        sha = str(item.get("sha") or "")
                        ci_key = f"{repo}#{number}_{sha[:8]}"
                        already_ci_notified = ci_key in ci_notified
                    entry = {
                        "first_seen": iso(now),
                        "last_warned": iso(now) if already_ci_notified else None,
                        "escalated_at": None,
                        "kind": kind,
                    }
                    watch[item_id] = entry
                if kind == "ci":
                    failure_signature = "\n".join(sorted(item.get("failed_checks") or []))
                    previous_signature = entry.get("failure_signature")
                    if previous_signature and previous_signature != failure_signature:
                        entry = {
                            "first_seen": iso(now),
                            "last_warned": None,
                            "escalated_at": None,
                            "kind": kind,
                        }
                        watch[item_id] = entry
                    entry["failure_signature"] = failure_signature
                last_warned = parse_gh_time(entry.get("last_warned"))
                escalated_at = parse_gh_time(entry.get("escalated_at"))
                # CI age is measured from first_seen (no event timestamp on the check rollup).
                if kind == "ci":
                    item_created_at = parse_gh_time(entry.get("first_seen")) or now
                    warn_hours = 0.0  # first warn immediate (unless already_ci_notified set last_warned)
                else:
                    item_created_at = item["created_at"]
                    warn_hours = STALE_WARN_HOURS
                retries = state.get("failed_task_retries", {})
                task = _direct_task_for_stale_item(item, retries)
                note = ""
                rewarn_base = STALE_REWARN_HOURS
                if task is not None and task.status in ACTIVE_TASK_STATUSES:
                    # A live task damps reminder frequency but never silences it.
                    entry["active_task"] = task.task_id
                    note = f"（対応タスク {task.task_id} 実行中）"
                    rewarn_base = max(STALE_REWARN_HOURS, ACTIVE_TASK_REWARN_FLOOR_HOURS)
                else:
                    entry.pop("active_task", None)
                    if task is not None and task.status in TERMINAL_TASK_STATUSES:
                        # Terminal task + problem still observed = whiff; full urgency.
                        note = f"（対応タスク {task.task_id} は{task.status}で終了したが問題は残存＝空振り）"
                entry.pop("suppressed_by_task", None)
                entry.pop("suppressed_last_sent", None)
                age_hours = max(0.0, (now - item_created_at).total_seconds() / 3600.0)
                msg_kind = kind if kind in ("review", "ci") else "comment"

                first_escalated = parse_gh_time(entry.get("first_escalated_at")) or escalated_at
                human_last = parse_gh_time(entry.get("human_notified_at"))
                if (
                    first_escalated is not None
                    and now - first_escalated >= timedelta(hours=HUMAN_ESCALATION_HOURS)
                    and (human_last is None or now - human_last >= timedelta(hours=HUMAN_ESCALATION_HOURS))
                ):
                    line = _format_stale_line(
                        repo=item["repo"],
                        number=item["number"],
                        author=item.get("author") or "",
                        body=item.get("body") or "",
                        url=item.get("url") or "",
                        created_at=item_created_at,
                        now=now,
                        kind=kind,
                        sha=str(item.get("sha") or ""),
                        failed_checks=item.get("failed_checks"),
                        note=note,
                    )
                    human_by_kind[msg_kind].append(line)
                    entry["human_notified_at"] = iso(now)
                    continue
                stage = determine_warning_stage(
                    item_created_at=item_created_at,
                    now=now,
                    last_warned=last_warned,
                    escalated_at=escalated_at,
                    warn_hours=warn_hours,
                    rewarn_hours=rewarn_base,
                    escalate_hours=STALE_ESCALATE_HOURS,
                    warn_count=int(entry.get("warn_count") or 0),
                )
                if stage == "none":
                    continue
                line = _format_stale_line(
                    repo=item["repo"],
                    number=item["number"],
                    author=item.get("author") or "",
                    body=item.get("body") or "",
                    url=item.get("url") or "",
                    created_at=item_created_at,
                    now=now,
                    kind=kind,
                    sha=str(item.get("sha") or ""),
                    failed_checks=item.get("failed_checks"),
                    note=note,
                )
                dispatcher_due = stage in ("warn", "rewarn") or (
                    stage == "escalate"
                    and (
                        last_warned is None
                        or (now - last_warned)
                        >= timedelta(hours=decayed_interval_hours(rewarn_base, int(entry.get("warn_count") or 0)))
                    )
                )
                if dispatcher_due:
                    dispatcher_by_kind[msg_kind].append(line)
                    max_age_by_kind[msg_kind] = max(max_age_by_kind[msg_kind], age_hours)
                    entry["last_warned"] = iso(now)
                    entry["warn_count"] = int(entry.get("warn_count") or 0) + 1
                if stage == "escalate":
                    escalate_by_kind[msg_kind].append(line)
                    max_age_by_kind[msg_kind] = max(max_age_by_kind[msg_kind], age_hours)
                    entry["escalated_at"] = iso(now)
                    entry.setdefault("first_escalated_at", entry["escalated_at"])

    if open_pr_count == 0 and not watch:
        return

    # Drop resolved / closed-PR items and 14-day leftovers only after a full scan.
    cutoff = iso(now - timedelta(days=14))
    state["stale_watch"] = {
        key: value
        for key, value in watch.items()
        if key in active_ids and (value.get("first_seen") or iso(now)) >= cutoff
    }

    for msg_kind, lines in dispatcher_by_kind.items():
        if lines:
            send(DISPATCHER, _stale_message(lines, kind=msg_kind, max_age_hours=max_age_by_kind[msg_kind]))
            log(f"stale warn ({msg_kind}) -> {DISPATCHER}: {len(lines)} item(s)")
    for msg_kind, lines in escalate_by_kind.items():
        if lines:
            send(ESCALATION_TARGET, _stale_message(lines, kind=msg_kind, max_age_hours=max_age_by_kind[msg_kind]))
            log(f"stale escalate ({msg_kind}) -> {ESCALATION_TARGET}: {len(lines)} item(s)")
    for msg_kind, lines in human_by_kind.items():
        if lines:
            send(
                ESCALATION_TARGET,
                "【人間判断が必要】\n\n"
                + "\n".join(lines)
                + "\n\n状態が24時間以上変化していません。takaへ call_human で報告すること。"
                "解決するまでこの通知は24時間ごとに繰り返されます。",
            )
            log(f"stale human escalation ({msg_kind}) -> {ESCALATION_TARGET}: {len(lines)} item(s)")


def save_state(state: dict) -> None:
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=STATE_FILE.parent,
            prefix=f".{STATE_FILE.name}.",
            delete=False,
        ) as handle:
            json.dump(state, handle, indent=1, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            temp_path = Path(handle.name)
        temp_path.replace(STATE_FILE)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


@contextmanager
def locked_state() -> Iterator[dict]:
    """Lock the shared state across the complete read/modify/write cycle."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with STATE_LOCK.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            state = load_state()
            yield state
            save_state(state)
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


_REVIEW_INSTRUCTION_BASE = (
    f"最終pushから{QUIET_SECONDS // 60}分以上静穏を確認済みです。"
    "上記PRの current HEAD に対する差分レビュー/FRCを直ちに実施してください。"
    "過去HEADへのレビューは新push時点で無効です。"
    "2回目以降のレビューは収束ルール（heartbeat.md記載・2026-07-15 taka指示）に従い、"
    "前回blocking findingsの解消確認と新push差分に限定してください。"
    "full PRの再レビューをやり直さないこと。"
    "同一PRのHOLDが通算3回に達している場合は自動レビューを停止し、rinへエスカレーションしてください。"
    "複数件ある場合はbackgroundタスクとして並列に処理して構いません。"
)


def _model_slug(entry: str) -> str:
    """Normalize a ``mode:model`` entry into a lowercase ``[a-z0-9-]`` slug.

    The mode prefix is stripped so ``x:grok/grok-4.5`` -> ``grok-grok-4-5``,
    keeping the slug short and file-safe.
    """
    model_part = entry.split(":", 1)[1] if ":" in entry else entry
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", model_part).strip("-").lower()
    return slug or "model"


def _review_task(task_id: str):
    """Return the reviewer's task record, or ``None`` when absent."""
    if DRY_RUN:
        return None
    target_dir = get_animas_dir() / REVIEWER
    if not target_dir.is_dir():
        return None
    return TaskQueueManager(target_dir).get_task_by_id(task_id)


def _dispatch_multipass_reviews(state: dict, items: list[dict]) -> None:
    """Emit one review task per configured model, then arm the synthesis step."""
    multipass = state.setdefault("multi_model_passes", {})
    for it in items:
        base_id = _ci_task_id(it["repo"], it["number"], it["sha"])
        if base_id in multipass:
            continue
        sha = it["sha"]
        number = it["number"]
        repo = it["repo"]
        model_tasks: list[str] = []
        for entry in PR_DISPATCH_REVIEW_MODELS:
            slug = _model_slug(entry)
            task_id = f"{base_id}-m-{slug}"
            instruction = (
                f"GitHub の {repo}#{number}（{it.get('title', '')}）の review 依頼。\n\n"
                + _REVIEW_INSTRUCTION_BASE
                + f"\n\nこれは {entry} によるレビューパスである。最終判定（APPROVE操作等）はこのパスでは行わず、"
                "指摘の列挙に徹すること。原文の句読点を変えずに事実を挙げ、各指摘に file:line を添えること。"
                f"FRCファイルは reviews/pr{number}-frc-{REVIEWER}-{sha}-{slug}.md に書くこと。"
            )
            dispatch_task(
                target=REVIEWER,
                task_id=task_id,
                summary=f"PRマルチパスレビュー {repo}#{number} ({entry})",
                instruction=instruction,
                model=entry,
                meta={"repo": repo, "number": number, "sha": sha, "model": entry, "multipass": True},
            )
            model_tasks.append(task_id)
        multipass[base_id] = {
            "repo": repo,
            "number": number,
            "sha": sha,
            "models": list(PR_DISPATCH_REVIEW_MODELS),
            "task_ids": model_tasks,
            "synth_dispatched": False,
        }
        log(
            f"multi-pass review dispatch -> {REVIEWER}: {repo}#{number} "
            f"models={','.join(PR_DISPATCH_REVIEW_MODELS)} tasks={','.join(model_tasks)}"
        )


def check_multipass_synth(state: dict) -> None:
    """Dispatch the final synthesis task once every model pass is terminal."""
    if not PR_DISPATCH_REVIEW_MODELS:
        return
    multipass = state.setdefault("multi_model_passes", {})
    for base_id, info in list(multipass.items()):
        if info.get("synth_dispatched"):
            continue
        records = {tid: _review_task(tid) for tid in info["task_ids"]}
        if any(record is None or record.status not in TERMINAL_TASK_STATUSES for record in records.values()):
            continue  # still running or unpublished; wait for the next cron sweep
        done = [tid for tid, rec in records.items() if rec.status == "done"]
        if not done:
            continue  # all passes failed -> rely on the reality-check redispatch path
        failed = [tid for tid, rec in records.items() if rec.status != "done"]
        repo = info["repo"]
        number = info["number"]
        sha = info["sha"]
        synth_id = f"{base_id}-synth"
        file_paths = "\n".join(
            f"- reviews/pr{number}-frc-{REVIEWER}-{sha}-{_model_slug(m)}.md" for m in info["models"]
        )
        if failed:
            failed_notes = (
                "\n\n注意: 以下のモデルパスは失敗（terminalだがdoneでない）したためファイルが欠落している。"
                f"\n{chr(10).join('- ' + t for t in failed)}"
            )
        else:
            failed_notes = ""
        instruction = (
            f"各モデルパスのFRCファイル（下記パス）を読み、指摘をマージ・重複排除し、最終判定を "
            f"reviews/pr{number}-frc-{REVIEWER}-{sha}.md （従来のファイル名）に書く。"
            "GitHub上のレビュー操作（APPROVE・CHANGES_REQUESTED等）はこの統合パスで行う。"
            f"\n\n対象ファイル:\n{file_paths}\n"
        ) + failed_notes
        dispatch_task(
            target=REVIEWER,
            task_id=synth_id,
            summary=f"PRレビュー統合判定 {repo}#{number}",
            instruction=instruction,
            model=PR_DISPATCH_SYNTH_MODEL,
            meta={"repo": repo, "number": number, "sha": sha, "multipass": "synth"},
        )
        log(f"multi-pass synth dispatch -> {REVIEWER}: {synth_id} models={','.join(info['models'])}")
        # Drop the entry so state stays bounded to in-flight PRs; same-sha
        # re-dispatch is prevented by notified_sha and add_task_if_absent.
        multipass.pop(base_id, None)


def check_commits(state: dict) -> None:
    """Detect a stable PR head and dispatch it once to the reviewer."""
    now = now_utc()
    ready: list[dict] = []
    ready_lines: list[str] = []
    open_keys: set[str] = set()

    for repo in REPOS:
        prs = json.loads(
            gh(
                [
                    "pr",
                    "list",
                    "--repo",
                    repo,
                    "--state",
                    "open",
                    "--json",
                    "number,title,headRefOid,isDraft",
                    "--limit",
                    "100",
                ]
            )
        )
        for pr in prs:
            if pr.get("isDraft"):
                continue
            key = f"{repo}#{pr['number']}"
            open_keys.add(key)
            sha = pr["headRefOid"]
            entry = state["prs"].get(key)
            if entry is None or entry.get("sha") != sha:
                state["prs"][key] = {
                    "sha": sha,
                    "sha_seen_at": iso(now),
                    "notified_sha": (entry or {}).get("notified_sha", ""),
                    "title": pr["title"][:120],
                }
                continue
            if entry.get("notified_sha") == sha:
                continue
            seen_at = datetime.strptime(entry["sha_seen_at"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
            if now - seen_at >= timedelta(seconds=QUIET_SECONDS):
                ready.append(
                    {"repo": repo, "number": pr["number"], "sha": sha, "title": entry.get("title", "")}
                )
                ready_lines.append(f"- {key} {sha[:8]}: {entry.get('title', '')}")
                entry["notified_sha"] = sha

    state["prs"] = {key: value for key, value in state["prs"].items() if key in open_keys}
    if ready:
        if PR_DISPATCH_REVIEW_MODELS:
            _dispatch_multipass_reviews(state, ready)
        else:
            send(
                REVIEWER,
                "【PR新規コミット検出（push静穏確認済み）】\n\n"
                + "\n".join(ready_lines)
                + "\n\n"
                + _REVIEW_INSTRUCTION_BASE,
            )
            log(f"review dispatch -> {REVIEWER}: {len(ready)} PR(s)")


def check_comments(state: dict) -> None:
    """Dispatch new non-bot review and issue comments once."""
    since = state.get("last_comment_check") or iso(now_utc() - timedelta(hours=1))
    now = now_utc()
    seen = state.setdefault("seen_comments", {})
    lines: list[str] = []

    for repo in REPOS:
        for endpoint, kind in (
            (f"repos/{repo}/pulls/comments", "review-comment"),
            (f"repos/{repo}/issues/comments", "issue-comment"),
        ):
            comments = json.loads(gh(["api", f"{endpoint}?since={since}&per_page=100"]))
            for comment in comments:
                author = (comment.get("user") or {}).get("login", "")
                if is_our_bot(author):
                    continue
                dedupe_key = f"{kind}:{comment.get('id')}"
                if dedupe_key in seen:
                    continue
                raw_body = str(comment.get("body") or "")
                if BOT_LOGIN and f"@{BOT_LOGIN}".casefold() in raw_body.casefold():
                    resource_url = str(
                        comment.get("pull_request_url" if kind == "review-comment" else "issue_url") or ""
                    )
                    try:
                        number = int(resource_url.rstrip("/").rsplit("/", 1)[-1])
                    except ValueError:
                        number = 0
                    url = str(comment.get("html_url") or "")
                    instruction = (
                        f"GitHub の {repo}#{number} に次のコメントが投稿された。\n\n"
                        f"{raw_body}\n\nURL: {url}\n\n"
                        "上記コメントの指示に従って対応せよ。コンフリクト解消の場合は "
                        "procedures/pr-conflict-resolution.md の手順（worktreeでorigin/baseをmerge・"
                        "テスト通過確認・通常push・force-push禁止）に従う。"
                    )
                    dispatch_task(
                        target=FIXER,
                        task_id=f"gh-cmd-{comment.get('id')}",
                        summary=f"GitHubコメント対応 {repo}#{number}",
                        instruction=instruction,
                        meta={"repo": repo, "number": number, "url": url, "kind": kind},
                    )
                    seen[dedupe_key] = iso(now)
                    continue
                seen[dedupe_key] = iso(now)
                body = raw_body.replace("\n", " ")[:140]
                lines.append(f"- [{kind}] {author}: {body}\n  {comment.get('html_url', '')}")

    state["last_comment_check"] = iso(now)
    cutoff = iso(now - timedelta(days=14))
    state["seen_comments"] = {key: value for key, value in seen.items() if value >= cutoff}
    if lines:
        detail = "\n".join(lines[:20])
        more = f"\n…他{len(lines) - 20}件" if len(lines) > 20 else ""
        send(
            DISPATCHER,
            "【外部レビューコメント検知】\n\n"
            f"{detail}{more}\n\n"
            "bot以外による新規コメントです。ACTION_REQUIRED判定と"
            f"{FIXER}への修正ディスパッチを procedures/pr-event-detection-patrol.md "
            "に従って実施してください。",
        )
        log(f"comment dispatch -> {DISPATCHER}: {len(lines)} comment(s)")


def check_ci(state: dict) -> None:
    """Dispatch CI failures once per PR and head SHA.

    2026-08-11 taka指示: CI待ちでレビュー保留したPRが「全チェックgreen化」を検知できず
    放置される問題への対処として、review dispatch済みHEADのCI全green化を
    REVIEWERへ一度だけ再通知する（green_notified でSHAごとにラッチ）。
    """
    green_notified = state.setdefault("ci_green_notified", {})
    failure_signatures = state.setdefault("ci_failure_signatures", {})
    green_ready: list[str] = []
    for repo in REPOS:
        prs = json.loads(
            gh(
                [
                    "pr",
                    "list",
                    "--repo",
                    repo,
                    "--state",
                    "open",
                    "--json",
                    "number,headRefOid,statusCheckRollup,reviewDecision",
                    "--limit",
                    "100",
                ]
            )
        )
        for pr in prs:
            failed = failed_check_names(pr.get("statusCheckRollup"))
            if not failed:
                number = pr["number"]
                sha = pr["headRefOid"]
                pr_key = f"{repo}#{number}"
                green_key = f"{pr_key}_{sha[:8]}"
                entry = state["prs"].get(pr_key) or {}
                # review dispatch済みのHEADに限る（未通知HEADはcheck_commitsが通常経路で拾う）。
                # APPROVED済みはレビュー保留ではないので通知しない。
                if (
                    green_key not in green_notified
                    and entry.get("notified_sha") == sha
                    and str(pr.get("reviewDecision") or "").upper() != "APPROVED"
                    and ci_all_green(pr.get("statusCheckRollup"))
                ):
                    green_ready.append(
                        f"- {pr_key} {sha[:8]}: {entry.get('title', '')}\n  https://github.com/{repo}/pull/{number}"
                    )
                    green_notified[green_key] = iso(now_utc())
                continue
            key = f"{repo}#{pr['number']}_{pr['headRefOid'][:8]}"
            signature = "\n".join(sorted(failed))
            if failure_signatures.get(key) not in (None, signature):
                state["ci_notified"].pop(key, None)
            elif key in state["ci_notified"]:
                failure_signatures.setdefault(key, signature)
            if key in state["ci_notified"]:
                continue
            workflow_url = next(
                (
                    str(check.get("detailsUrl") or "")
                    for check in (pr.get("statusCheckRollup") or [])
                    if str(check.get("conclusion") or "").upper() in FAILING_CI_CONCLUSIONS
                ),
                "",
            )
            number = pr["number"]
            sha = pr["headRefOid"]
            pr_url = f"https://github.com/{repo}/pull/{number}"
            workflow_name = ", ".join(failed[:6])
            base_id = _ci_task_id(repo, number, sha)
            attempts = int(state.get("failed_task_retries", {}).get(base_id, 0))
            dispatch_task(
                target=FIXER,
                task_id=_attempt_task_id(base_id, attempts + 1),
                summary=f"CI失敗修正 {repo}#{number}",
                instruction=_whiff_preamble(base_id, state)
                + (
                    f"PR #{number} ({pr_url}) の CI ({workflow_name}) が head {sha} で失敗。"
                    f"原因を調査し修正をpushせよ。\nworkflow URL: {workflow_url}"
                ),
                meta={"repo": repo, "number": number, "sha": sha, "workflow_url": workflow_url},
            )
            state["ci_notified"][key] = iso(now_utc())
            failure_signatures[key] = signature

    if green_ready:
        send(
            REVIEWER,
            "【CI全チェックgreen化検知】\n\n" + "\n".join(green_ready) + "\n\n"
            "上記PRは既にレビュー通知済みのHEADで、全CIチェックがterminal・greenになりました。"
            "CI待ちで保留していた場合は直ちにレビュー/FRCに着手してください。"
            "レビュー投稿済みの場合はこの通知への対応は不要です。",
        )
        log(f"ci-green dispatch -> {REVIEWER}: {len(green_ready)} PR(s)")

    cutoff = iso(now_utc() - timedelta(days=30))
    state["ci_notified"] = {key: value for key, value in state["ci_notified"].items() if value >= cutoff}
    state["ci_failure_signatures"] = {
        key: value for key, value in failure_signatures.items() if key in state["ci_notified"]
    }
    state["ci_green_notified"] = {key: value for key, value in green_notified.items() if value >= cutoff}


def check_conflicts(state: dict) -> None:
    """Dispatch once per head and loudly remind Rin on every conflicting scan."""
    notified = state.setdefault("conflict_notified", {})
    open_keys: set[str] = set()
    dispatched = 0
    conflict_lines: list[str] = []
    for repo in REPOS:
        prs = json.loads(
            gh(
                [
                    "pr",
                    "list",
                    "--repo",
                    repo,
                    "--state",
                    "open",
                    "--json",
                    "number,title,headRefName,baseRefName,headRefOid,mergeable,isDraft,url",
                    "--limit",
                    "100",
                ]
            )
        )
        for pr in prs:
            if pr.get("isDraft"):
                continue
            key = f"{repo}#{pr['number']}"
            open_keys.add(key)
            mergeable = str(pr.get("mergeable", "")).upper()
            if mergeable == "MERGEABLE":
                # 解消済み: 同一headのまま再コンフリクトしても再通知できるよう記録を消す
                notified.pop(key, None)
                continue
            if mergeable != "CONFLICTING":
                # UNKNOWN はGitHub側の算出待ち。次回巡回で確定値を見る。
                continue
            sha = pr["headRefOid"][:8]
            url = str(pr.get("url") or f"https://github.com/{repo}/pull/{pr['number']}")
            conflict_lines.append(f"- {key} {sha}: {url}")
            if notified.get(key) == sha:
                continue
            notified[key] = sha
            base_id = _conflict_task_id(repo, pr["number"], sha)
            attempts = int(state.get("failed_task_retries", {}).get(base_id, 0))
            dispatch_task(
                target=FIXER,
                task_id=_attempt_task_id(base_id, attempts + 1),
                summary=f"コンフリクト解消 {repo}#{pr['number']}",
                instruction=_whiff_preamble(base_id, state)
                + (
                    f"PR #{pr['number']}「{pr.get('title', '')}」"
                    f"（{pr.get('headRefName', '')} -> {pr.get('baseRefName', '')}）\nURL: {url}\n\n"
                    "baseブランチとのコンフリクトを解消せよ。手順は "
                    "procedures/pr-conflict-resolution.md に従う（該当ブランチのworktreeで "
                    "origin/base をmergeして解消・テスト通過確認・通常push・force-push禁止）。"
                    "解消pushの後は既存の静穏検知で差分レビューが自動起動する。"
                ),
                meta={"repo": repo, "number": pr["number"], "sha": pr["headRefOid"], "url": url},
            )
            dispatched += 1

    state["conflict_notified"] = {key: value for key, value in notified.items() if key in open_keys}
    if dispatched:
        log(f"conflict task dispatch -> {FIXER}: {dispatched} PR(s)")
    if conflict_lines:
        send(
            DISPATCHER,
            "【要対応・マージコンフリクト継続検知】\n\n"
            + "\n".join(conflict_lines)
            + "\n\n同じ通知が繰り返されても無視しないこと。"
            f"{FIXER}のcanonical laneが解消pushを完了し、mergeable=MERGEABLEになるまで追跡せよ。",
        )
        log(f"conflict reminder -> {DISPATCHER}: {len(conflict_lines)} PR(s)")


def check_human_waits(state: dict) -> None:
    """Daily digest of PRs blocked only on a human approval (no anima can act).

    Keeps these out of the warn/escalate loops so urgency stays pointed at
    problems animas can actually move; humans get one consolidated list.
    """
    if not REVIEWER_LOGIN:
        return
    now = now_utc()
    today = iso(now)[:10]
    if state.get("human_wait_digest_date") == today:
        return
    lines: list[str] = []
    for repo in REPOS:
        prs = json.loads(
            gh(
                [
                    "pr",
                    "list",
                    "--repo",
                    repo,
                    "--state",
                    "open",
                    "--json",
                    "number,title,url,isDraft,reviewDecision,statusCheckRollup,latestReviews",
                    "--limit",
                    "100",
                ]
            )
        )
        for pr in prs:
            if pr.get("isDraft"):
                continue
            # Only REVIEW_REQUIRED counts as pure human-approval wait;
            # APPROVED needs nothing and CHANGES_REQUESTED is actionable by animas.
            if str(pr.get("reviewDecision") or "REVIEW_REQUIRED").upper() != "REVIEW_REQUIRED":
                continue
            bot_approved = any(
                ((review.get("author") or {}).get("login") == REVIEWER_LOGIN)
                and str(review.get("state") or "").upper() == "APPROVED"
                for review in (pr.get("latestReviews") or [])
            )
            if not bot_approved or not ci_all_green(pr.get("statusCheckRollup")):
                continue
            lines.append(f"- {repo}#{pr['number']}: {str(pr.get('title') or '')[:80]}\n  {pr.get('url') or ''}")
    state["human_wait_digest_date"] = today
    if not lines:
        return
    send(
        ESCALATION_TARGET,
        "【人間レビュー待ち日次ダイジェスト】\n\n"
        + "\n".join(lines)
        + "\n\n上記はbotレビュー承認済み・CI全green・人間レビュアーの承認のみ待ちのPRです。"
        "animaに打つ手はないため、警告ループの対象外にしています。"
        "takaへ call_human でまとめて報告してください。",
    )
    log(f"human-wait digest -> {ESCALATION_TARGET}: {len(lines)} PR(s)")


def check_patrol_liveness(state: dict) -> None:
    """Alert when the LLM patrol stops leaving audit artifacts (hollow patrol)."""
    if not PATROL_GLOB:
        return
    import glob as glob_module

    now = now_utc()
    newest = max(
        (os.path.getmtime(path) for path in glob_module.glob(os.path.expanduser(PATROL_GLOB))),
        default=0.0,
    )
    age_hours = (now.timestamp() - newest) / 3600.0 if newest else float("inf")
    if age_hours < PATROL_MAX_AGE_HOURS:
        state.pop("patrol_alert_last", None)
        return
    last = parse_gh_time(state.get("patrol_alert_last"))
    if last is not None and now - last < timedelta(hours=PATROL_ALERT_REPEAT_HOURS):
        return
    age_label = "不明(証跡ゼロ)" if age_hours == float("inf") else f"{age_hours:.1f}時間"
    message = (
        f"【巡回空洞化検知】PR巡回の監査証跡が {age_label} 出ていません。"
        "巡回cronは動いているのに成果物が出ていない＝巡回が形骸化しています。"
        "直ちに全Open PRの実巡回を行い、証跡を保存してください。"
        f"\n監視対象: {PATROL_GLOB}"
    )
    send(DISPATCHER, message)
    send(ESCALATION_TARGET, message)
    state["patrol_alert_last"] = iso(now)
    log(f"patrol liveness alert -> {DISPATCHER},{ESCALATION_TARGET}: artifact age {age_label}")


def main() -> int:
    if not REPOS:
        sys.stderr.write("PR_DISPATCH_REPOS is not set (comma-separated owner/repo list); refusing to run.\n")
        return 2
    with locked_state() as state:
        try:
            reopen_stalled_dispatches(state)
            check_commits(state)
            check_multipass_synth(state)
            check_comments(state)
            check_ci(state)
            check_conflicts(state)
            check_unaddressed(state)
            check_human_waits(state)
            check_patrol_liveness(state)
        except Exception as exc:
            state["consecutive_failures"] = state.get("consecutive_failures", 0) + 1
            count = state["consecutive_failures"]
            log(f"FAILURE #{count}: {exc}")
            if count % ALERT_EVERY == 0:
                try:
                    send(
                        DISPATCHER,
                        f"【pr-review-dispatch異常】gh連続失敗 {count}回。"
                        f"PRレビュー自動起動が止まっています。エラー: {str(exc)[:300]}\n"
                        f"ログ: {LOG_FILE}",
                    )
                except Exception as alert_exc:
                    log(f"alert send failed: {alert_exc}")
            return 1
        state["consecutive_failures"] = 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
