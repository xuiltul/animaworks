from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""GitHub webhook gateway for PR review and dispatcher notifications."""

import asyncio
import fcntl  # noqa: F401  # Backward-compatible module attribute for lock tests.
import logging
import os
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import core.review_multipass as review_multipass
from core.config.models import load_config
from core.config.schemas import GitHubWebhookConfig
from core.i18n import t
from core.memory.task_queue import TaskQueueManager
from core.messenger import Messenger
from core.paths import get_animas_dir, get_shared_dir
from core.supervisor.dead_command_reaper import reap_dead_commands
from core.tasks_dispatch import FAILING_CI_CONCLUSIONS, dispatch_direct_task

logger = logging.getLogger("animaworks.github_gateway")

STATE_FILENAME = "pr-review-dispatch-state.json"
MAX_FAILED_REDISPATCHES = 2
# Sweep interval for re-dispatching failed gh-ci tasks without a new push.
CI_RETRY_INTERVAL_SEC = 600.0
SYNTH_SWEEP_INTERVAL_SEC = 120.0
_SLOW_SWEEP_EVERY_TICKS = int(CI_RETRY_INTERVAL_SEC / SYNTH_SWEEP_INTERVAL_SEC)
REVIEW_SLO_THRESHOLD = timedelta(minutes=45)
REVIEW_SLO_STATE_KEY = "review_slo_alerted"

def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _iso_to_utc(value: str | None) -> datetime | None:
    """Parse an ISO-8601 UTC timestamp (with or without trailing Z) to a UTC datetime."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+" + "00:00")).astimezone(UTC)
    except (ValueError, TypeError):
        return None


def gh(args: list[str]) -> str:
    """Run the GitHub CLI; returns stdout (empty string on failure)."""
    env = dict(os.environ)
    proc = subprocess.run(
        ["gh", *args],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    if proc.returncode != 0:
        logger.warning("gh %s failed: %s", " ".join(args[:3]), proc.stderr.strip()[:300])
        return ""
    return proc.stdout


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
        self._ci_retry_task: asyncio.Task[None] | None = None

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
            self._ci_retry_task = asyncio.create_task(self._ci_retry_loop())
        else:
            logger.info("GitHub webhook gateway is disabled")

    async def stop(self) -> None:
        """Cancel pending event and debounce tasks."""
        tasks = [*self._event_tasks, *self._debounce_tasks.values()]
        if self._ci_retry_task is not None:
            tasks.append(self._ci_retry_task)
            self._ci_retry_task = None
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

    async def _ci_retry_loop(self) -> None:
        # Synth dispatch runs every tick (2 min) so a finished model-pass pair
        # waits minutes, not up to 10; the heavier sweeps keep the 10-min cadence.
        tick = 0
        while True:
            try:
                await asyncio.to_thread(self.sweep_multipass_synth)
            except Exception:
                logger.exception("GitHub webhook multipass synth sweep failed")
            if tick % _SLOW_SWEEP_EVERY_TICKS == 0:
                try:
                    await asyncio.to_thread(self.retry_failed_ci_tasks)
                except Exception:
                    logger.exception("GitHub webhook CI retry sweep failed")
                try:
                    await asyncio.to_thread(self.check_review_slo)
                except Exception:
                    logger.exception("GitHub webhook review SLO sweep failed")
                try:
                    await asyncio.to_thread(reap_dead_commands)
                except Exception:
                    logger.exception("dead command reaper sweep failed")
            tick += 1
            await asyncio.sleep(SYNTH_SWEEP_INTERVAL_SEC)

    def sweep_multipass_synth(self) -> None:
        """Dispatch pending synthesis passes for completed multipass reviews.

        ``check_multipass_synth`` used to run only from the fallback cron
        (``scripts/pr-review-dispatch.py``); with that cron disabled the
        final GitHub-posting synth task was never issued (2026-08-31).
        """
        if not self._require_state_file().is_file():
            return
        with locked_dispatch_state(self._require_state_file()) as state:
            review_multipass.check_multipass_synth(
                state,
                reviewer=self._config.reviewer_anima,
                synth_model=self._config.review_synth_model,
                quiet_seconds=self._config.quiet_seconds,
                dispatch=dispatch_direct_task,
                logger=logger.info,
            )

    def _slo_candidate_green_key(self, key: str, sha: str) -> str | None:
        return f"{key}_{sha[:8]}"

    def check_review_slo(self) -> list[dict[str, Any]]:
        """Escalate PRs reviewed (CI green) more than REVIEW_SLO_THRESHOLD ago.

        A PR passed the review beacon (``ci_green_notified``) but the reviewer
        has still not posted on the current head within the SLO window.  For
        each such PR we (a) re-dispatch the multipass review if no pass is in
        flight for that head and (b) dispatch an investigation task to the
        dispatcher anima.  GitHub API calls are limited to PRs that are green-
        notified, past the window, and not yet alerted (no per-sweep sweep of
        every open PR).  Alerts are deduplicated per PR+sha via the state key
        ``review_slo_alerted``.
        """
        if not self._require_state_file().is_file():
            return []
        now = datetime.now(UTC)
        alerted: list[dict[str, Any]] = []
        with locked_dispatch_state(self._require_state_file()) as state:
            prs = state.setdefault("prs", {})
            green = state.setdefault("ci_green_notified", {})
            slo_alerts = state.setdefault(REVIEW_SLO_STATE_KEY, {})
            candidates: list[tuple[str, str]] = []
            for key, entry in prs.items():
                sha = str(entry.get("sha") or "")
                notified_sha = str(entry.get("notified_sha") or "")
                if not sha or notified_sha != sha:
                    continue
                green_key = self._slo_candidate_green_key(key, sha)
                green_at = _iso_to_utc(green.get(green_key))
                if green_at is None or (now - green_at) < REVIEW_SLO_THRESHOLD:
                    continue
                if slo_alerts.get(green_key):
                    continue  # already alerted for this head; don't loop
                candidates.append((key, sha))
            # Only now (crime still possible within the window) hit GitHub.
            for key, sha in candidates:
                repo, _, number = key.partition("#")
                if not repo or not number.isdigit():
                    continue
                current_sha = self._gh_current_head(repo, int(number))
                if not current_sha or current_sha != sha:
                    continue  # PR moved on; a fresher beacon will take over
                if self._gh_has_review(repo, int(number), sha):
                    continue  # reviewer already posted on this head
                self._slo_dispatch(state, key, repo, int(number), sha)
                slo_alerts[self._slo_candidate_green_key(key, sha)] = _now_iso()
                logger.info("review SLO exceeded: %s sha=%s", key, sha[:8])
                alerted.append({"key": key, "sha": sha})
        return alerted

    def _gh_current_head(self, repo: str, number: int) -> str:
        """Return the current head SHA of an open PR, or '' (empty) on error."""
        out = gh(
            [
                "pr",
                "view",
                str(number),
                "--repo",
                repo,
                "--json",
                "headRefOid,state",
                "--jq",
                "select(.state == \"OPEN\") | .headRefOid",
            ]
        )
        return (out or "").strip()

    def _gh_has_review(self, repo: str, number: int, sha: str) -> bool:
        """True when animaworks-reviewer has posted a review on this head."""
        out = gh(
            [
                "api",
                f"repos/{repo}/pulls/{number}/reviews",
                "--paginate",
                "--jq",
                (
                    ".[] | select(.user.login == \"{}\" and .commit_id == \"{}\")"
                    " | select(.state != \"PENDING\")"
                ).format(self._config.reviewer_login, sha),
            ]
        )
        return bool(out and out.strip())

    def _slo_dispatch(self, state: dict, key: str, repo: str, number: int, sha: str) -> None:
        """Re-dispatch the multipass review (if absent) and alert the dispatcher."""
        models = list(getattr(self._config, "review_multipass_models", None) or [])
        if models:
            base_id = f"gh-ci-{repo.replace('/', '-')}#{number}-{sha[:8]}"
            multipass = state.setdefault("multi_model_passes", {})
            if base_id not in multipass:
                entry = state["prs"].get(key) or {}
                review_multipass.dispatch_multipass_reviews(
                    state,
                    [
                        {
                            "repo": repo,
                            "number": number,
                            "sha": sha,
                            "title": str(entry.get("title") or f"PR #{number}"),
                        }
                    ],
                    reviewer=self._config.reviewer_anima,
                    models=models,
                    quiet_seconds=self._config.quiet_seconds,
                    dispatch=dispatch_direct_task,
                )
        dispatch_direct_task(
            target=self._config.dispatcher_anima,
            task_id=f"gh-slo-{repo.replace('/', '-')}#{number}-{sha[:8]}",
            summary=t("github_gateway.review_slo_summary", repo=repo, number=number),
            instruction=t(
                "github_gateway.review_slo_task",
                repo=repo,
                number=number,
                sha=sha,
                url=f"https://github.com/{repo}/pull/{number}",
            ),
            meta={"repo": repo, "number": number, "sha": sha, "kind": "review-slo"},
        )

    def retry_failed_ci_tasks(self) -> int:
        """Re-dispatch failed gh-ci tasks for PRs still sitting on the failing head.

        The webhook path only retries when GitHub sends another ``workflow_run``,
        i.e. after a new push.  A task killed mid-way (2026-08-30 hang storm)
        was therefore never retried.  ``_dispatch_ci_once`` owns the retry
        budget (``MAX_FAILED_REDISPATCHES``), so this only feeds it candidates.
        """
        candidates: list[tuple[str, int, str, str, str]] = []
        if not self._require_state_file().is_file():
            return 0  # nothing was ever dispatched; don't materialise state on a sweep
        with locked_dispatch_state(self._require_state_file()) as state:
            urls = state.get("ci_workflow_urls") or {}
            for key in list(state["ci_notified"]):
                repo, _, rest = key.rpartition("#")
                number, _, sha8 = rest.partition("_")
                sha = str((state["prs"].get(f"{repo}#{number}") or {}).get("sha") or "")
                if not number.isdigit() or not sha8 or not sha.startswith(sha8):
                    continue
                workflow_name = str(state["ci_failure_signatures"].get(key) or "CI")
                candidates.append((repo, int(number), sha, workflow_name, str(urls.get(key) or "")))
        dispatched = 0
        for repo, number, sha, workflow_name, url in candidates:
            dispatched += self._dispatch_ci_once(repo, [(number, sha)], workflow_name, url)
        if dispatched:
            logger.info("GitHub webhook CI retry sweep re-dispatched %d task(s)", dispatched)
        return dispatched

    def _dispatch_ci_once(
        self,
        repo: str,
        items: list[tuple[int, str]],
        workflow_name: str,
        url: str,
    ) -> int:
        """Dispatch CI-fix tasks for *items*; returns how many were (re)dispatched."""
        with locked_dispatch_state(self._require_state_file()) as state:
            notified = state["ci_notified"]
            failure_signatures = state["ci_failure_signatures"]
            retries = state["failed_task_retries"]
            urls = state.setdefault("ci_workflow_urls", {})
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
                return 0
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
                urls[key] = url
            return len(fresh)

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
