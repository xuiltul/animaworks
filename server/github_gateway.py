from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""GitHub webhook gateway: verified PR events become one notification to the dispatcher Anima.

The gateway no longer creates tasks or posts to GitHub itself (2026-09
teardown). It records PR state, debounces bursts of pushes, excludes our own
bot/reviewer accounts, and sends exactly one deduplicated report per
(PR, head SHA, event category) to ``dispatcher_anima``. The dispatcher Anima
decides who should act and calls ``delegate_task`` itself.
"""

import asyncio
import fcntl
import json
import logging
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from core.config.models import load_config
from core.config.schemas import GitHubWebhookConfig
from core.i18n import t
from core.messenger import Messenger
from core.paths import get_shared_dir

logger = logging.getLogger("animaworks.github_gateway")

STATE_FILENAME = "pr-review-dispatch-state.json"
FAILING_CI_CONCLUSIONS = frozenset({"FAILURE", "CANCELLED", "TIMED_OUT", "STARTUP_FAILURE"})


def _default_state() -> dict[str, Any]:
    return {"prs": {}}


def _load_state(state_file: Path) -> dict[str, Any]:
    try:
        raw = state_file.read_text(encoding="utf-8")
    except FileNotFoundError:
        return _default_state()
    try:
        state = json.loads(raw) if raw.strip() else _default_state()
    except json.JSONDecodeError:
        return _default_state()
    if not isinstance(state, dict):
        return _default_state()
    state.setdefault("prs", {})
    return state


def _save_state(state_file: Path, state: dict[str, Any]) -> None:
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
def locked_dispatch_state(state_file: Path) -> Iterator[dict[str, Any]]:
    """Read, mutate, and persist dispatcher state under an exclusive flock.

    A stable sidecar inode is locked while the JSON is atomically replaced.
    """
    state_file.parent.mkdir(parents=True, exist_ok=True)
    with state_file.with_suffix(".lock").open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            state = _load_state(state_file)
            yield state
            _save_state(state_file, state)
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


class GitHubWebhookManager:
    """Turn signed GitHub webhook events into notifications for the dispatcher Anima."""

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

    # ── pull_request ─────────────────────────────────────────────────

    async def _handle_pull_request(self, repo: str, payload: dict[str, Any]) -> None:
        action = str(payload.get("action") or "")
        pr = payload.get("pull_request") or {}
        number = int(pr.get("number") or payload.get("number") or 0)
        if not number:
            return
        key = f"{repo}#{number}"
        sha = str((pr.get("head") or {}).get("sha") or "")
        title = str(pr.get("title") or "")[:120]
        url = str(pr.get("html_url") or f"https://github.com/{repo}/pull/{number}")
        author = str((pr.get("user") or {}).get("login") or "")
        body = str(pr.get("body") or "")

        if action == "closed":
            task = self._debounce_tasks.pop(key, None)
            if task:
                task.cancel()
            await asyncio.to_thread(
                self._notify_once,
                key=key,
                repo=repo,
                number=number,
                sha=sha,
                category="closed",
                title=title,
                event_label="closed",
                author=author,
                url=url,
                body=body,
            )
            await asyncio.to_thread(self._remove_pr_state, key)
            return

        if action not in {"opened", "synchronize", "reopened", "ready_for_review"}:
            return
        if bool(pr.get("draft")) and action != "ready_for_review":
            return
        if not sha:
            return
        await asyncio.to_thread(self._record_pr_state, key, sha, title)
        self._restart_debounce(key, sha, title, action, author, url, body)

    def _record_pr_state(self, key: str, sha: str, title: str) -> None:
        with locked_dispatch_state(self._require_state_file()) as state:
            entry = state["prs"].setdefault(key, {"notified": {}})
            entry["sha"] = sha
            entry["title"] = title

    def _remove_pr_state(self, key: str) -> None:
        with locked_dispatch_state(self._require_state_file()) as state:
            state["prs"].pop(key, None)

    def _restart_debounce(self, key: str, sha: str, title: str, action: str, author: str, url: str, body: str) -> None:
        previous = self._debounce_tasks.get(key)
        if previous:
            previous.cancel()
        task = asyncio.create_task(self._debounce_then_notify(key, sha, title, action, author, url, body))
        self._debounce_tasks[key] = task

    async def _debounce_then_notify(
        self, key: str, sha: str, title: str, action: str, author: str, url: str, body: str
    ) -> None:
        task = asyncio.current_task()
        try:
            await asyncio.sleep(self._config.quiet_seconds)
            repo, _, number_s = key.rpartition("#")
            await asyncio.to_thread(
                self._notify_once,
                key=key,
                repo=repo,
                number=int(number_s),
                sha=sha,
                category="push",
                title=title,
                event_label=action,
                author=author,
                url=url,
                body=body,
                require_sha=sha,
            )
        except asyncio.CancelledError:
            raise
        finally:
            if self._debounce_tasks.get(key) is task:
                self._debounce_tasks.pop(key, None)

    # ── pull_request_review ─────────────────────────────────────────────

    async def _handle_review(self, repo: str, payload: dict[str, Any]) -> None:
        if payload.get("action") != "submitted":
            return
        review = payload.get("review") or {}
        author = str((review.get("user") or {}).get("login") or "")
        if self._is_bot(author):
            return
        pr = payload.get("pull_request") or {}
        number = int(pr.get("number") or 0)
        if not number:
            return
        key = f"{repo}#{number}"
        sha = str((pr.get("head") or {}).get("sha") or "")
        state_word = str(review.get("state") or "")
        await asyncio.to_thread(
            self._notify_once,
            key=key,
            repo=repo,
            number=number,
            sha=sha,
            category="review",
            title=str(pr.get("title") or ""),
            event_label=f"review:{state_word}".rstrip(":"),
            author=author,
            url=str(review.get("html_url") or ""),
            body=str(review.get("body") or ""),
            item_id=str(review.get("id") or ""),
        )

    # ── issue_comment / pull_request_review_comment ─────────────────────

    async def _handle_comment(self, event: str, repo: str, payload: dict[str, Any]) -> None:
        if payload.get("action") != "created":
            return
        comment = payload.get("comment") or {}
        author = str((comment.get("user") or {}).get("login") or "")
        if self._is_bot(author):
            return
        if event == "pull_request_review_comment":
            number = int((payload.get("pull_request") or {}).get("number") or 0)
            kind = "review-comment"
        else:
            number = int((payload.get("issue") or {}).get("number") or 0)
            kind = "issue-comment"
        if not number:
            return
        key = f"{repo}#{number}"
        await asyncio.to_thread(
            self._notify_once,
            key=key,
            repo=repo,
            number=number,
            sha="",
            category="comment",
            title="",
            event_label=kind,
            author=author,
            url=str(comment.get("html_url") or ""),
            body=str(comment.get("body") or ""),
            item_id=str(comment.get("id") or ""),
        )

    # ── workflow_run ─────────────────────────────────────────────────

    async def _handle_workflow_run(self, repo: str, payload: dict[str, Any]) -> None:
        workflow = payload.get("workflow_run") or {}
        conclusion = str(workflow.get("conclusion") or "").upper()
        if payload.get("action") != "completed" or conclusion not in FAILING_CI_CONCLUSIONS:
            return
        sha = str(workflow.get("head_sha") or "")
        if not sha:
            return
        prs = workflow.get("pull_requests") or []
        numbers = [
            int(pr.get("number") or 0)
            for pr in prs
            if int(pr.get("number") or 0)
            and (not (pr.get("head") or {}).get("sha") or (pr.get("head") or {}).get("sha") == sha)
        ]
        if not numbers:
            return
        name = str(workflow.get("name") or "workflow")
        url = str(workflow.get("html_url") or "")
        for number in numbers:
            key = f"{repo}#{number}"
            await asyncio.to_thread(
                self._notify_once,
                key=key,
                repo=repo,
                number=number,
                sha=sha,
                category="ci",
                title="",
                event_label="workflow_run",
                author="",
                url=url,
                body="",
                ci_name=name,
                ci_conclusion=conclusion,
            )

    # ── shared notify/dedupe core ───────────────────────────────────────

    def _notify_once(
        self,
        *,
        key: str,
        repo: str,
        number: int,
        sha: str,
        category: str,
        title: str,
        event_label: str,
        author: str,
        url: str,
        body: str,
        ci_name: str = "",
        ci_conclusion: str = "",
        require_sha: str | None = None,
        item_id: str = "",
    ) -> None:
        """Send one notification per (PR, head SHA, category[, item_id]), deduped via state.

        *item_id* (comment id / review id) makes each distinct comment or
        review its own dedupe slot, so a second human comment on the same
        head is still delivered; redeliveries of the same item are not.

        *sha* is the authoritative head SHA when known (pull_request,
        pull_request_review, workflow_run); comment events pass ``""`` and
        fall back to the PR's last-recorded head SHA so redeliveries still
        dedupe. *require_sha* guards the debounced push path against a
        superseded head slipping through after cancellation.
        """
        with locked_dispatch_state(self._require_state_file()) as state:
            entry = state["prs"].setdefault(key, {"notified": {}})
            if require_sha is not None and entry.get("sha") != require_sha:
                return
            effective_sha = sha or str(entry.get("sha") or "")
            if sha:
                entry["sha"] = sha
            if title:
                entry["title"] = title
            notified = entry.setdefault("notified", {})
            dedup_sha = effective_sha or "-"
            slot = f"{category}:{item_id}" if item_id else category
            if notified.get(slot) == dedup_sha:
                return
            content = self._build_notification(
                repo=repo,
                number=number,
                title=str(entry.get("title") or title or f"PR #{number}"),
                event_label=event_label,
                author=author,
                url=url,
                body=body,
                ci_name=ci_name,
                ci_conclusion=ci_conclusion,
            )
            self._send(self._config.dispatcher_anima, content, category, f"{key}:{slot}:{dedup_sha[:12]}")
            notified[slot] = dedup_sha

    def _build_notification(
        self,
        *,
        repo: str,
        number: int,
        title: str,
        event_label: str,
        author: str,
        url: str,
        body: str,
        ci_name: str,
        ci_conclusion: str,
    ) -> str:
        excerpt = (body or "").strip()[:500]
        ci_line = t("github_gateway.notify_ci_line", name=ci_name, conclusion=ci_conclusion) if ci_name else ""
        return t(
            "github_gateway.notify",
            repo=repo,
            number=number,
            title=title,
            event_label=event_label,
            author=author or "unknown",
            url=url,
            excerpt=excerpt,
            ci_line=ci_line,
        )

    def _is_bot(self, author: str) -> bool:
        """True when the author is bot_login or reviewer_login."""
        if not author:
            return False
        author_cf = author.casefold()
        bot = self._config.bot_login
        reviewer = self._config.reviewer_login
        return (bool(bot) and author_cf == bot.casefold()) or (bool(reviewer) and author_cf == reviewer.casefold())

    def _send(self, to: str, content: str, kind: str, key: str) -> None:
        Messenger(self._require_shared_dir(), "github-gateway").send(
            to=to,
            content=content,
            intent="report",
            source="system",
            meta={"source": "github-webhook", "kind": kind, "key": key},
        )

    def _require_shared_dir(self) -> Path:
        if self._shared_dir is None:
            raise RuntimeError("GitHub webhook gateway has not been started")
        return self._shared_dir

    def _require_state_file(self) -> Path:
        if self._state_file is None:
            raise RuntimeError("GitHub webhook gateway has not been started")
        return self._state_file
