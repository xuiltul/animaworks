from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""GitHub webhook gateway for PR review and dispatcher notifications."""

import asyncio
import fcntl
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from core.config.models import load_config
from core.config.schemas import GitHubWebhookConfig
from core.i18n import t
from core.memory.task_queue import TaskQueueManager
from core.messenger import Messenger
from core.paths import get_animas_dir, get_shared_dir
import core.review_multipass as review_multipass
from core.tasks_dispatch import FAILING_CI_CONCLUSIONS, dispatch_direct_task

logger = logging.getLogger("animaworks.github_gateway")

STATE_FILENAME = "pr-review-dispatch-state.json"
MAX_FAILED_REDISPATCHES = 2


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _default_state() -> dict[str, Any]:
    return review_multipass.default_state()


@contextmanager
def locked_dispatch_state(state_file: Path) -> Iterator[dict[str, Any]]:
    """Read, mutate, and persist dispatcher state under an exclusive flock.

    A stable sidecar inode is locked while the JSON is atomically replaced.
    The fallback cron uses the same sidecar (via ``core.review_multipass``),
    so neither process can overwrite state loaded by the other.
    """
    with review_multipass.locked_state(state_file) as state:
        yield state


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

        if pr.get("mergeable") is False or str(pr.get("mergeable_state") or "").lower() in {
            "conflicting",
            "dirty",
        }:
            await asyncio.to_thread(
                self._send,
                self._config.dispatcher_anima,
                t(
                    "github_gateway.conflict",
                    repo=repo,
                    number=number,
                    sha=str((pr.get("head") or {}).get("sha") or ""),
                    url=str(pr.get("html_url") or f"https://github.com/{repo}/pull/{number}"),
                ),
                "conflict",
                f"conflict:{key}",
            )

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
        models = list(getattr(self._config, "review_multipass_models", None) or [])
        with locked_dispatch_state(self._require_state_file()) as state:
            entry = state["prs"].get(key)
            if not entry or entry.get("sha") != sha or entry.get("notified_sha") == sha:
                return
            if models:
                repo, _, number = key.partition("#")
                review_multipass.dispatch_multipass_reviews(
                    state,
                    [{"repo": repo, "number": int(number or 0), "sha": sha, "title": title}],
                    reviewer=self._config.reviewer_anima,
                    models=models,
                    quiet_seconds=self._config.quiet_seconds,
                    dispatch=dispatch_direct_task,
                )
            else:
                quiet = self._format_quiet_period()
                content = t(
                    "github_gateway.review_dispatch",
                    pr_key=key,
                    sha=sha[:8],
                    title=title,
                    quiet=quiet,
                )
                self._send(self._config.reviewer_anima, content, "review", key)
            entry["notified_sha"] = sha

    def _format_quiet_period(self) -> str:
        seconds = self._config.quiet_seconds
        if seconds >= 60 and seconds % 60 == 0:
            return t("github_gateway.minutes", value=int(seconds // 60))
        return t("github_gateway.seconds", value=f"{seconds:g}")

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
        state = str(review.get("state") or "").upper()
        if self._is_bot(author):
            # bot自身のreviewはFRC判定(HOLD/PASS)の投稿なので、除外せずrinへ転送する
            # reviewer_login 名義の APPROVED / CHANGES_REQUESTED も同様にFRC結果として扱う
            await asyncio.to_thread(
                self._dispatch_frc_result_once,
                f"review:{review_id}",
                repo,
                number,
                str((pr.get("head") or {}).get("sha") or ""),
                str(review.get("body") or ""),
                str(review.get("html_url") or ""),
                str(review.get("state") or ""),
            )
            if state == "CHANGES_REQUESTED":
                await asyncio.to_thread(
                    self._dispatch_review_task_once,
                    repo,
                    number,
                    str(review_id),
                    author,
                    str(review.get("body") or ""),
                    str(review.get("html_url") or ""),
                    True,
                )
            return
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
            if (
                kind in {"issue-comment", "review-comment"}
                and self._config.bot_login
                and f"@{self._config.bot_login}".casefold() in body.casefold()
            ):
                comment_id = dedupe_key.rsplit(":", 1)[-1]
                instruction = t(
                    "github_gateway.command_task",
                    repo=repo,
                    number=number,
                    body=body,
                    url=url,
                )
                dispatch_direct_task(
                    target=self._config.implementer_anima,
                    task_id=f"gh-cmd-{comment_id}",
                    summary=t("github_gateway.command_summary", repo=repo, number=number),
                    instruction=instruction,
                    meta={"repo": repo, "number": number, "url": url, "kind": kind},
                )
                seen[dedupe_key] = _now_iso()
                return
            dispatch_direct_task(
                target=self._config.implementer_anima,
                task_id=f"gh-comment-{dedupe_key.replace(':', '-')}",
                summary=t("github_gateway.command_summary", repo=repo, number=number),
                instruction=t(
                    "github_gateway.command_task",
                    repo=repo,
                    number=number,
                    body=body,
                    url=url,
                ),
                meta={
                    "repo": repo,
                    "number": number,
                    "url": url,
                    "kind": kind,
                    "author": author,
                },
            )
            content = (
                "【外部レビューコメント検知】\n\n"
                f"- [{kind}] {repo}#{number} {author}\n\n{body}\n\nURL: {url}\n\n"
                "全文をnatsumeへ直接投入済みです。Rinも全文を独立判断し、"
                "必要な追跡を procedures/pr-event-detection-patrol.md に従って実施してください。"
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
        review_state: str = "",
    ) -> None:
        with locked_dispatch_state(self._require_state_file()) as state:
            seen = state["seen_comments"]
            if dedupe_key in seen:
                return
            state_upper = (review_state or "").strip().upper()
            if state_upper == "APPROVED":
                verdict = "PASS"
            elif state_upper == "CHANGES_REQUESTED":
                verdict = "HOLD"
            else:
                head = body.lstrip().upper()
                if head.startswith("HOLD"):
                    verdict = "HOLD"
                elif head.startswith("PASS"):
                    verdict = "PASS"
                else:
                    verdict = t("github_gateway.unknown_verdict")
            content = t(
                "github_gateway.frc_result",
                verdict=verdict,
                repo=repo,
                number=number,
                head_sha=head_sha,
                url=url,
                summary=body,
            )
            self._send(self._config.dispatcher_anima, content, "frc-result", dedupe_key)
            seen[dedupe_key] = _now_iso()

    def _dispatch_review_task_once(
        self,
        repo: str,
        number: int,
        review_id: str,
        author: str,
        body: str,
        url: str,
        bot_derived: bool,
    ) -> None:
        with locked_dispatch_state(self._require_state_file()) as state:
            review_tasks = state["review_tasks"]
            if review_id in review_tasks:
                return
            bot_note = t(
                "github_gateway.review_task_bot_note" if bot_derived else "github_gateway.review_task_human_note"
            )
            dispatch_direct_task(
                target=self._config.implementer_anima,
                task_id=f"gh-review-{repo.replace('/', '-')}#{number}-{review_id}",
                summary=t("github_gateway.review_task_summary", repo=repo, number=number),
                instruction=t(
                    "github_gateway.review_task",
                    repo=repo,
                    number=number,
                    url=url or f"https://github.com/{repo}/pull/{number}",
                    author=author or "unknown",
                    bot_note=bot_note,
                    body=body,
                ),
                meta={
                    "repo": repo,
                    "number": number,
                    "review_id": review_id,
                    "reviewer": author,
                    "url": url,
                    "bot_derived": bot_derived,
                },
            )
            review_tasks[review_id] = _now_iso()

    async def _handle_workflow_run(self, repo: str, payload: dict[str, Any]) -> None:
        workflow = payload.get("workflow_run") or {}
        if (
            payload.get("action") != "completed"
            or str(workflow.get("conclusion") or "").upper() not in FAILING_CI_CONCLUSIONS
        ):
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
            failure_signatures = state["ci_failure_signatures"]
            retries = state["failed_task_retries"]
            target_dir = get_animas_dir() / self._config.implementer_anima
            for number, sha in items:
                key = f"{repo}#{number}_{sha[:8]}"
                if failure_signatures.get(key) not in (None, workflow_name):
                    notified.pop(key, None)
                elif key in notified:
                    failure_signatures.setdefault(key, workflow_name)
                if key not in notified or not target_dir.is_dir():
                    continue
                task_id = f"gh-ci-{repo.replace('/', '-')}#{number}-{sha[:8]}"
                task = TaskQueueManager(target_dir).get_task_by_id(task_id)
                if (
                    task is not None
                    and task.status == "failed"
                    and int(retries.get(task_id, 0)) < MAX_FAILED_REDISPATCHES
                ):
                    notified.pop(key, None)
                    retries[task_id] = int(retries.get(task_id, 0)) + 1
            fresh = [
                (number, sha, f"{repo}#{number}_{sha[:8]}")
                for number, sha in items
                if f"{repo}#{number}_{sha[:8]}" not in notified
            ]
            if not fresh:
                return
            now = _now_iso()
            for number, sha, key in fresh:
                pr_url = f"https://github.com/{repo}/pull/{number}"
                instruction = t(
                    "github_gateway.ci_task",
                    number=number,
                    pr_url=pr_url,
                    workflow_name=workflow_name,
                    sha=sha,
                    workflow_url=url,
                )
                dispatch_direct_task(
                    target=self._config.implementer_anima,
                    task_id=f"gh-ci-{repo.replace('/', '-')}#{number}-{sha[:8]}",
                    summary=t("github_gateway.ci_summary", repo=repo, number=number),
                    instruction=instruction,
                    meta={"repo": repo, "number": number, "sha": sha, "workflow_url": url},
                )
                notified[key] = now
                failure_signatures[key] = workflow_name

    def _is_bot(self, author: str) -> bool:
        """True when the author is bot_login or reviewer_login."""
        if not author:
            return False
        author_cf = author.casefold()
        bot = self._config.bot_login
        reviewer = self._config.reviewer_login
        return (bool(bot) and author_cf == bot.casefold()) or (bool(reviewer) and author_cf == reviewer.casefold())

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
