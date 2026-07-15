from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""GitHub webhook gateway for PR review and dispatcher notifications."""

import asyncio
import fcntl
import json
import logging
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from core.config.models import load_config
from core.config.schemas import GitHubWebhookConfig
from core.messenger import Messenger
from core.paths import get_shared_dir

logger = logging.getLogger("animaworks.github_gateway")

STATE_FILENAME = "pr-review-dispatch-state.json"


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _default_state() -> dict[str, Any]:
    return {
        "prs": {},
        "last_comment_check": _now_iso(),
        "seen_comments": {},
        "ci_notified": {},
        "consecutive_failures": 0,
    }


@contextmanager
def locked_dispatch_state(state_file: Path) -> Iterator[dict[str, Any]]:
    """Read, mutate, and persist dispatcher state under an exclusive flock.

    A stable sidecar inode is locked while the JSON is atomically replaced.
    The fallback cron uses the same sidecar, so neither process can overwrite
    state loaded by the other.
    """
    state_file.parent.mkdir(parents=True, exist_ok=True)
    lock_file = state_file.with_suffix(".lock")
    with lock_file.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            try:
                raw = state_file.read_text(encoding="utf-8")
            except FileNotFoundError:
                raw = ""
            try:
                state = json.loads(raw) if raw.strip() else _default_state()
            except json.JSONDecodeError:
                logger.warning("GitHub dispatcher state is invalid JSON; starting fresh")
                state = _default_state()
            if not isinstance(state, dict):
                state = _default_state()
            state.setdefault("prs", {})
            state.setdefault("seen_comments", {})
            state.setdefault("ci_notified", {})
            yield state
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
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


class GitHubWebhookManager:
    """Dispatch signed GitHub webhook events to the configured Animas."""

    def __init__(
        self,
        *,
        config: GitHubWebhookConfig | None = None,
        shared_dir: Path | None = None,
        state_file: Path | None = None,
    ) -> None:
        self._config = config or GitHubWebhookConfig()
        self._config_injected = config is not None
        self._shared_dir = shared_dir
        self._state_file = state_file
        self._started = False
        self._debounce_tasks: dict[str, asyncio.Task[None]] = {}
        self._event_tasks: set[asyncio.Task[None]] = set()

    async def start(self) -> None:
        """Load configuration and enable webhook processing when configured."""
        if not self._config_injected:
            self._config = load_config().github_webhook
        if self._shared_dir is None:
            self._shared_dir = get_shared_dir()
        if self._state_file is None:
            self._state_file = self._shared_dir / STATE_FILENAME
        self._started = self._config.enabled
        if self._started:
            logger.info(
                "GitHub webhook gateway started (repos=%d, quiet_seconds=%s)",
                len(self._config.repos),
                self._config.quiet_seconds,
            )
        else:
            logger.info("GitHub webhook gateway is disabled")

    async def stop(self) -> None:
        """Cancel pending event and debounce tasks."""
        tasks = [*self._event_tasks, *self._debounce_tasks.values()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._event_tasks.clear()
        self._debounce_tasks.clear()
        self._started = False
        logger.info("GitHub webhook gateway stopped")

    def reload(self) -> None:
        """Reload runtime config for future webhook events."""
        if not self._config_injected:
            self._config = load_config().github_webhook
        self._started = self._config.enabled

    def dispatch_event(self, event: str, payload: dict[str, Any]) -> None:
        """Schedule event processing and return immediately."""
        if not self._started:
            return
        task = asyncio.create_task(self.handle_event(event, payload))
        self._event_tasks.add(task)
        task.add_done_callback(self._event_done)

    def _event_done(self, task: asyncio.Task[None]) -> None:
        self._event_tasks.discard(task)
        if task.cancelled():
            return
        try:
            task.result()
        except Exception:
            logger.exception("GitHub webhook background processing failed")

    async def handle_event(self, event: str, payload: dict[str, Any]) -> None:
        """Process one verified GitHub webhook payload."""
        if not self._started:
            return
        repo = str((payload.get("repository") or {}).get("full_name") or "")
        if repo not in self._config.repos:
            return

        if event == "pull_request":
            await self._handle_pull_request(repo, payload)
        elif event == "pull_request_review":
            await self._handle_review(repo, payload)
        elif event in {"pull_request_review_comment", "issue_comment"}:
            await self._handle_comment(event, repo, payload)
        elif event == "workflow_run":
            await self._handle_workflow_run(repo, payload)

    async def _handle_pull_request(self, repo: str, payload: dict[str, Any]) -> None:
        action = str(payload.get("action") or "")
        pr = payload.get("pull_request") or {}
        number = int(pr.get("number") or payload.get("number") or 0)
        if not number:
            return
        key = f"{repo}#{number}"

        if action == "closed":
            task = self._debounce_tasks.pop(key, None)
            if task:
                task.cancel()
            await asyncio.to_thread(self._remove_pr_state, key)
            return

        if action not in {"opened", "synchronize", "reopened", "ready_for_review"}:
            return
        if bool(pr.get("draft")) and action != "ready_for_review":
            return

        sha = str((pr.get("head") or {}).get("sha") or "")
        if not sha:
            return
        title = str(pr.get("title") or "")[:120]
        await asyncio.to_thread(self._record_pr_state, key, sha, title)
        self._restart_debounce(key, sha, title)

    def _record_pr_state(self, key: str, sha: str, title: str) -> None:
        with locked_dispatch_state(self._require_state_file()) as state:
            previous = state["prs"].get(key) or {}
            state["prs"][key] = {
                "sha": sha,
                "sha_seen_at": _now_iso(),
                "notified_sha": previous.get("notified_sha", ""),
                "title": title,
            }

    def _remove_pr_state(self, key: str) -> None:
        with locked_dispatch_state(self._require_state_file()) as state:
            state["prs"].pop(key, None)

    def _restart_debounce(self, key: str, sha: str, title: str) -> None:
        previous = self._debounce_tasks.get(key)
        if previous:
            previous.cancel()
        task = asyncio.create_task(self._debounce_then_dispatch(key, sha, title))
        self._debounce_tasks[key] = task

    async def _debounce_then_dispatch(self, key: str, sha: str, title: str) -> None:
        task = asyncio.current_task()
        try:
            await asyncio.sleep(self._config.quiet_seconds)
            await asyncio.to_thread(self._dispatch_review_if_current, key, sha, title)
        except asyncio.CancelledError:
            raise
        finally:
            if self._debounce_tasks.get(key) is task:
                self._debounce_tasks.pop(key, None)

    def _dispatch_review_if_current(self, key: str, sha: str, title: str) -> None:
        with locked_dispatch_state(self._require_state_file()) as state:
            entry = state["prs"].get(key)
            if not entry or entry.get("sha") != sha or entry.get("notified_sha") == sha:
                return
            quiet = self._format_quiet_period()
            content = (
                "【PR新規コミット検出（push静穏確認済み）】\n\n"
                f"- {key} {sha[:8]}: {title}\n\n"
                f"最終pushから{quiet}以上静穏を確認済みです。"
                "上記PRの current HEAD に対する差分レビュー/FRCを直ちに実施してください。"
                "過去HEADへのレビューは新push時点で無効です。"
                "複数件ある場合はbackgroundタスクとして並列に処理して構いません。"
            )
            self._send(self._config.reviewer_anima, content, "review", key)
            entry["notified_sha"] = sha

    def _format_quiet_period(self) -> str:
        seconds = self._config.quiet_seconds
        if seconds >= 60 and seconds % 60 == 0:
            return f"{int(seconds // 60)}分"
        return f"{seconds:g}秒"

    async def _handle_review(self, repo: str, payload: dict[str, Any]) -> None:
        if payload.get("action") != "submitted":
            return
        review = payload.get("review") or {}
        author = str((review.get("user") or {}).get("login") or "")
        review_id = review.get("id")
        if review_id is None:
            return
        pr = payload.get("pull_request") or {}
        number = int(pr.get("number") or 0)
        if self._is_bot(author):
            # bot自身のreviewはFRC判定(HOLD/PASS)の投稿なので、除外せずrinへ転送する
            await asyncio.to_thread(
                self._dispatch_frc_result_once,
                f"review:{review_id}",
                repo,
                number,
                str((pr.get("head") or {}).get("sha") or ""),
                str(review.get("body") or ""),
                str(review.get("html_url") or ""),
            )
            return
        state = str(review.get("state") or "").upper()
        emphasis = "【CHANGES_REQUESTED】 " if state == "CHANGES_REQUESTED" else ""
        await asyncio.to_thread(
            self._dispatch_comment_once,
            f"review:{review_id}",
            repo,
            number,
            author,
            str(review.get("body") or ""),
            str(review.get("html_url") or ""),
            f"{emphasis}review {state}".strip(),
        )

    async def _handle_comment(self, event: str, repo: str, payload: dict[str, Any]) -> None:
        if payload.get("action") != "created":
            return
        comment = payload.get("comment") or {}
        author = str((comment.get("user") or {}).get("login") or "")
        if self._is_bot(author):
            return
        comment_id = comment.get("id")
        if comment_id is None:
            return
        if event == "pull_request_review_comment":
            kind = "review-comment"
            number = int((payload.get("pull_request") or {}).get("number") or 0)
        else:
            kind = "issue-comment"
            number = int((payload.get("issue") or {}).get("number") or 0)
        await asyncio.to_thread(
            self._dispatch_comment_once,
            f"{kind}:{comment_id}",
            repo,
            number,
            author,
            str(comment.get("body") or ""),
            str(comment.get("html_url") or ""),
            kind,
        )

    def _dispatch_comment_once(
        self,
        dedupe_key: str,
        repo: str,
        number: int,
        author: str,
        body: str,
        url: str,
        kind: str,
    ) -> None:
        with locked_dispatch_state(self._require_state_file()) as state:
            seen = state["seen_comments"]
            if dedupe_key in seen:
                return
            summary = body.replace("\n", " ")[:140]
            content = (
                "【外部レビューコメント検知】\n\n"
                f"- [{kind}] {repo}#{number} {author}: {summary}\n  {url}\n\n"
                "bot以外による新規コメントです。ACTION_REQUIRED判定と"
                "natsumeへの修正ディスパッチを procedures/pr-event-detection-patrol.md "
                "に従って実施してください。"
            )
            self._send(self._config.dispatcher_anima, content, "comment", dedupe_key)
            seen[dedupe_key] = _now_iso()

    def _dispatch_frc_result_once(
        self,
        dedupe_key: str,
        repo: str,
        number: int,
        head_sha: str,
        body: str,
        url: str,
    ) -> None:
        with locked_dispatch_state(self._require_state_file()) as state:
            seen = state["seen_comments"]
            if dedupe_key in seen:
                return
            head = body.lstrip().upper()
            if head.startswith("HOLD"):
                verdict = "HOLD"
            elif head.startswith("PASS"):
                verdict = "PASS"
            else:
                verdict = "判定不明"
            summary = body.replace("\n", " ")[:200]
            content = (
                "【FRC結果検知】\n\n"
                f"- 判定: {verdict}\n"
                f"- PR: {repo}#{number}\n"
                f"- HEAD: {head_sha}\n"
                f"- URL: {url}\n"
                f"- 本文冒頭: {summary}\n\n"
                "HOLDの場合は procedures/pr-event-detection-patrol.md に従って"
                "natsumeへの修正ディスパッチを実施してください。"
            )
            self._send(self._config.dispatcher_anima, content, "frc-result", dedupe_key)
            seen[dedupe_key] = _now_iso()

    async def _handle_workflow_run(self, repo: str, payload: dict[str, Any]) -> None:
        workflow = payload.get("workflow_run") or {}
        if payload.get("action") != "completed" or str(workflow.get("conclusion") or "").lower() != "failure":
            return
        sha = str(workflow.get("head_sha") or "")
        if not sha:
            return
        prs = workflow.get("pull_requests") or []
        items = [
            (int(pr.get("number") or 0), sha)
            for pr in prs
            if int(pr.get("number") or 0)
            and (not (pr.get("head") or {}).get("sha") or (pr.get("head") or {}).get("sha") == sha)
        ]
        if not items:
            return
        await asyncio.to_thread(
            self._dispatch_ci_once,
            repo,
            items,
            str(workflow.get("name") or "workflow"),
            str(workflow.get("html_url") or ""),
        )

    def _dispatch_ci_once(
        self,
        repo: str,
        items: list[tuple[int, str]],
        workflow_name: str,
        url: str,
    ) -> None:
        with locked_dispatch_state(self._require_state_file()) as state:
            notified = state["ci_notified"]
            fresh = [
                (number, sha, f"{repo}#{number}_{sha[:8]}")
                for number, sha in items
                if f"{repo}#{number}_{sha[:8]}" not in notified
            ]
            if not fresh:
                return
            lines = "\n".join(f"- {repo}#{number} {sha[:8]}: {workflow_name}" for number, sha, _ in fresh)
            content = f"【CI FAILURE検知】\n\n{lines}\n  {url}\n\n修正担当（natsume）へのディスパッチをお願いします。"
            self._send(self._config.dispatcher_anima, content, "ci", fresh[0][2])
            now = _now_iso()
            for _, _, key in fresh:
                notified[key] = now

    def _is_bot(self, author: str) -> bool:
        """True when the author is the configured bot account itself."""
        bot = self._config.bot_login
        return bool(bot) and author.casefold() == bot.casefold()

    def _send(self, to: str, content: str, kind: str, key: str) -> None:
        Messenger(self._require_shared_dir(), "pr-review-dispatch").send(
            to=to,
            content=content,
            intent="report",
            skip_logging=True,
            meta={"source": "github-webhook", "kind": kind, "key": key},
            source="system",
        )

    def _require_shared_dir(self) -> Path:
        if self._shared_dir is None:
            raise RuntimeError("GitHub webhook gateway has not been started")
        return self._shared_dir

    def _require_state_file(self) -> Path:
        if self._state_file is None:
            raise RuntimeError("GitHub webhook gateway has not been started")
        return self._state_file
