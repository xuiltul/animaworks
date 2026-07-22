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
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

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
REVIEWER = os.environ.get("PR_DISPATCH_REVIEWER", "sumire")
DISPATCHER = os.environ.get("PR_DISPATCH_DISPATCHER", "rin")
FIXER = os.environ.get("PR_DISPATCH_FIXER", "natsume")
ALERT_EVERY = 5

sys.path.insert(
    0,
    os.environ.get("ANIMAWORKS_REPO_ROOT", str(Path(__file__).resolve().parents[1])),
)


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
    from core.messenger import Messenger

    Messenger(SHARED_DIR, "pr-review-dispatch").send(
        to=to,
        content=content,
        intent="report",
        skip_logging=True,
        meta={"source": "pr-review-dispatch.py"},
        source="system",
    )


def default_state() -> dict:
    return {
        "prs": {},
        "last_comment_check": iso(now_utc()),
        "seen_comments": {},
        "ci_notified": {},
        "conflict_notified": {},
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
                state.setdefault("conflict_notified", {})
                return state
        except (json.JSONDecodeError, OSError):
            log("state file unreadable; starting fresh")
    return default_state()


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


def check_commits(state: dict) -> None:
    """Detect a stable PR head and dispatch it once to the reviewer."""
    now = now_utc()
    ready: list[str] = []
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
                ready.append(f"- {key} {sha[:8]}: {entry.get('title', '')}")
                entry["notified_sha"] = sha

    state["prs"] = {key: value for key, value in state["prs"].items() if key in open_keys}
    if ready:
        send(
            REVIEWER,
            "【PR新規コミット検出（push静穏確認済み）】\n\n"
            + "\n".join(ready)
            + "\n\n"
            + f"最終pushから{QUIET_SECONDS // 60}分以上静穏を確認済みです。"
            "上記PRの current HEAD に対する差分レビュー/FRCを直ちに実施してください。"
            "過去HEADへのレビューは新push時点で無効です。"
            "2回目以降のレビューは収束ルール（heartbeat.md記載・2026-07-15 taka指示）に従い、"
            "前回blocking findingsの解消確認と新push差分に限定してください。"
            "full PRの再レビューをやり直さないこと。"
            "同一PRのHOLDが通算3回に達している場合は自動レビューを停止し、rinへエスカレーションしてください。"
            "複数件ある場合はbackgroundタスクとして並列に処理して構いません。",
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
                if BOT_LOGIN and author == BOT_LOGIN:
                    continue
                dedupe_key = f"{kind}:{comment.get('id')}"
                if dedupe_key in seen:
                    continue
                seen[dedupe_key] = iso(now)
                body = (comment.get("body") or "").replace("\n", " ")[:140]
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
    """Dispatch CI failures once per PR and head SHA."""
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
                    "number,headRefOid,statusCheckRollup",
                    "--limit",
                    "100",
                ]
            )
        )
        for pr in prs:
            failed = [
                check.get("name", "?")
                for check in pr.get("statusCheckRollup") or []
                if str(check.get("conclusion", "")).upper() == "FAILURE"
            ]
            if not failed:
                continue
            key = f"{repo}#{pr['number']}_{pr['headRefOid'][:8]}"
            if key in state["ci_notified"]:
                continue
            state["ci_notified"][key] = iso(now_utc())
            lines.append(f"- {repo}#{pr['number']} {pr['headRefOid'][:8]}: {', '.join(failed[:6])}")

    cutoff = iso(now_utc() - timedelta(days=30))
    state["ci_notified"] = {key: value for key, value in state["ci_notified"].items() if value >= cutoff}
    if lines:
        send(
            DISPATCHER,
            "【CI FAILURE検知】\n\n"
            + "\n".join(lines)
            + f"\n\n修正担当（{FIXER}）へのディスパッチをお願いします。",
        )
        log(f"ci dispatch -> {DISPATCHER}: {len(lines)} PR(s)")


def check_conflicts(state: dict) -> None:
    """Dispatch merge conflicts once per PR head; renotify after re-conflict."""
    notified = state.setdefault("conflict_notified", {})
    open_keys: set[str] = set()
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
                    "number,title,headRefName,baseRefName,headRefOid,mergeable,isDraft",
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
            if notified.get(key) == sha:
                continue
            notified[key] = sha
            lines.append(
                f"- {key} {sha} ({pr.get('headRefName', '')} -> {pr.get('baseRefName', '')}): "
                f"{pr.get('title', '')[:100]}"
            )

    state["conflict_notified"] = {key: value for key, value in notified.items() if key in open_keys}
    if lines:
        send(
            DISPATCHER,
            "【マージコンフリクト検知】\n\n"
            + "\n".join(lines)
            + "\n\n上記PRがbaseブランチとコンフリクトしています（mergeable=CONFLICTING）。"
            f"{FIXER}へコンフリクト解消をディスパッチしてください。"
            f"解消手順は {FIXER} の procedures/pr-conflict-resolution.md に従うこと"
            "（該当ブランチのworktreeで origin/base をmergeして解消・テスト通過確認・"
            "通常push。force-push禁止）。解消pushの後は既存の静穏検知で差分レビューが"
            "自動起動します。",
        )
        log(f"conflict dispatch -> {DISPATCHER}: {len(lines)} PR(s)")


def main() -> int:
    if not REPOS:
        sys.stderr.write("PR_DISPATCH_REPOS is not set (comma-separated owner/repo list); refusing to run.\n")
        return 2
    with locked_state() as state:
        try:
            check_commits(state)
            check_comments(state)
            check_ci(state)
            check_conflicts(state)
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
