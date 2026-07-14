"""Unit tests for GitHub webhook PR dispatch and shared state handling."""

from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from core.config.schemas import GitHubWebhookConfig
from server import github_gateway
from server.github_gateway import GitHubWebhookManager, locked_dispatch_state

REPO = "FutureSync/AI-Schreiber"
BOT_LOGIN = "example-bot"
OTHER_REPO = "elsewhere/project"
SHA_1 = "1" * 40
SHA_2 = "2" * 40


def _pr_payload(
    *,
    action: str = "opened",
    repo: str = REPO,
    number: int = 17,
    sha: str = SHA_1,
    draft: bool = False,
    title: str = "Webhook dispatch",
) -> dict[str, Any]:
    return {
        "action": action,
        "number": number,
        "repository": {"full_name": repo},
        "pull_request": {
            "number": number,
            "draft": draft,
            "title": title,
            "head": {"sha": sha},
        },
    }


def _comment_payload(
    *,
    event: str = "issue_comment",
    action: str = "created",
    author: str = "reviewer",
    comment_id: int = 101,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "action": action,
        "repository": {"full_name": REPO},
        "comment": {
            "id": comment_id,
            "user": {"login": author},
            "body": "Please fix this\nedge case.",
            "html_url": f"https://github.test/comments/{comment_id}",
        },
    }
    if event == "pull_request_review_comment":
        payload["pull_request"] = {"number": 17}
    else:
        payload["issue"] = {"number": 17}
    return payload


def _review_payload(
    *,
    action: str = "submitted",
    author: str = "reviewer",
    review_id: int = 202,
    state: str = "changes_requested",
) -> dict[str, Any]:
    return {
        "action": action,
        "repository": {"full_name": REPO},
        "pull_request": {"number": 17},
        "review": {
            "id": review_id,
            "user": {"login": author},
            "state": state,
            "body": "The race remains.",
            "html_url": f"https://github.test/reviews/{review_id}",
        },
    }


def _workflow_payload(
    *,
    action: str = "completed",
    conclusion: str = "failure",
    head_sha: str = SHA_1,
    pr_head_sha: str | None = SHA_1,
    number: int = 17,
) -> dict[str, Any]:
    pr: dict[str, Any] = {"number": number}
    if pr_head_sha is not None:
        pr["head"] = {"sha": pr_head_sha}
    return {
        "action": action,
        "repository": {"full_name": REPO},
        "workflow_run": {
            "name": "quality-gate",
            "conclusion": conclusion,
            "head_sha": head_sha,
            "html_url": "https://github.test/actions/runs/1",
            "pull_requests": [pr],
        },
    }


@pytest.fixture
async def gateway(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Started manager with captured local Messenger sends."""
    shared_dir = tmp_path / "shared"
    state_file = shared_dir / github_gateway.STATE_FILENAME
    config = GitHubWebhookConfig(
        enabled=True,
        repos=[REPO],
        reviewer_anima="sumire",
        dispatcher_anima="rin",
        bot_login=BOT_LOGIN,
        quiet_seconds=3600,
    )
    manager = GitHubWebhookManager(
        config=config,
        shared_dir=shared_dir,
        state_file=state_file,
    )
    sends: list[dict[str, str]] = []

    def capture_send(to: str, content: str, kind: str, key: str) -> None:
        sends.append({"to": to, "content": content, "kind": kind, "key": key})

    monkeypatch.setattr(manager, "_send", capture_send)
    await manager.start()
    try:
        yield manager, sends, state_file
    finally:
        await manager.stop()


class TestConfigurationAndGating:
    def test_config_defaults_are_completely_disabled(self) -> None:
        config = GitHubWebhookConfig()
        assert config.enabled is False
        assert config.repos == []
        assert config.reviewer_anima == "sumire"
        assert config.dispatcher_anima == "rin"
        assert config.quiet_seconds == 180

    async def test_disabled_manager_does_not_create_state(self, tmp_path: Path) -> None:
        state_file = tmp_path / "state.json"
        manager = GitHubWebhookManager(
            config=GitHubWebhookConfig(enabled=False, repos=[REPO]),
            shared_dir=tmp_path,
            state_file=state_file,
        )
        await manager.start()
        await manager.handle_event("pull_request", _pr_payload())
        manager.dispatch_event("pull_request", _pr_payload())
        assert manager._started is False
        assert manager._event_tasks == set()
        assert not state_file.exists()

    async def test_start_loads_runtime_config_when_not_injected(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = GitHubWebhookConfig(enabled=True, repos=[REPO], quiet_seconds=12)
        load = MagicMock(return_value=MagicMock(github_webhook=runtime))
        monkeypatch.setattr(github_gateway, "load_config", load)
        manager = GitHubWebhookManager(shared_dir=tmp_path)
        await manager.start()
        try:
            assert manager._started is True
            assert manager._config is runtime
            load.assert_called_once_with()
        finally:
            await manager.stop()

    async def test_allowlist_rejects_other_repository(self, gateway) -> None:
        manager, sends, state_file = gateway
        await manager.handle_event("pull_request_review", _review_payload() | {"repository": {"full_name": OTHER_REPO}})
        await manager.handle_event("pull_request", _pr_payload(repo=OTHER_REPO))
        assert sends == []
        assert manager._debounce_tasks == {}
        assert not state_file.exists()


class TestPullRequestDebounce:
    async def test_draft_is_ignored_until_ready_for_review(self, gateway) -> None:
        manager, _, state_file = gateway
        await manager.handle_event("pull_request", _pr_payload(draft=True))
        assert not state_file.exists()
        assert manager._debounce_tasks == {}

        await manager.handle_event(
            "pull_request",
            _pr_payload(action="ready_for_review", draft=True),
        )
        state = json.loads(state_file.read_text(encoding="utf-8"))
        assert state["prs"][f"{REPO}#17"]["sha"] == SHA_1
        assert f"{REPO}#17" in manager._debounce_tasks

    async def test_push_resets_debounce_and_only_latest_sha_dispatches(
        self,
        gateway,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        manager, sends, state_file = gateway
        real_sleep = asyncio.sleep
        sleep_gates: list[asyncio.Event] = []

        async def controlled_sleep(_delay: float) -> None:
            gate = asyncio.Event()
            sleep_gates.append(gate)
            await gate.wait()

        monkeypatch.setattr(github_gateway.asyncio, "sleep", controlled_sleep)

        await manager.handle_event("pull_request", _pr_payload(action="opened", sha=SHA_1))
        await real_sleep(0)
        first_task = manager._debounce_tasks[f"{REPO}#17"]
        assert len(sleep_gates) == 1

        await manager.handle_event("pull_request", _pr_payload(action="synchronize", sha=SHA_2))
        await real_sleep(0)
        second_task = manager._debounce_tasks[f"{REPO}#17"]
        assert second_task is not first_task
        assert first_task.cancelled()
        assert len(sleep_gates) == 2

        sleep_gates[1].set()
        await second_task
        assert len(sends) == 1
        assert sends[0]["to"] == "sumire"
        assert sends[0]["kind"] == "review"
        assert SHA_2[:8] in sends[0]["content"]
        assert SHA_1[:8] not in sends[0]["content"]
        state = json.loads(state_file.read_text(encoding="utf-8"))
        assert state["prs"][f"{REPO}#17"]["notified_sha"] == SHA_2

    async def test_closed_cancels_timer_and_removes_only_pr_state(self, gateway) -> None:
        manager, _, state_file = gateway
        await manager.handle_event("pull_request", _pr_payload())
        key = f"{REPO}#17"
        task = manager._debounce_tasks[key]
        with locked_dispatch_state(state_file) as state:
            state["deployment_marker"] = {"keep": True}

        await manager.handle_event("pull_request", _pr_payload(action="closed"))
        await asyncio.sleep(0)
        state = json.loads(state_file.read_text(encoding="utf-8"))
        assert key not in state["prs"]
        assert state["deployment_marker"] == {"keep": True}
        assert key not in manager._debounce_tasks
        assert task.cancelled()

    async def test_stale_or_already_notified_sha_is_not_sent(self, gateway) -> None:
        manager, sends, state_file = gateway
        key = f"{REPO}#17"
        with locked_dispatch_state(state_file) as state:
            state["prs"][key] = {
                "sha": SHA_2,
                "sha_seen_at": "2026-07-14T00:00:00Z",
                "notified_sha": SHA_2,
                "title": "latest",
            }
        manager._dispatch_review_if_current(key, SHA_1, "stale")
        manager._dispatch_review_if_current(key, SHA_2, "latest")
        assert sends == []


class TestReviewAndCommentDispatch:
    @pytest.mark.parametrize("event", ["issue_comment", "pull_request_review_comment"])
    async def test_bot_comments_are_ignored(self, gateway, event: str) -> None:
        manager, sends, state_file = gateway
        await manager.handle_event(
            event,
            _comment_payload(event=event, author=BOT_LOGIN),
        )
        assert sends == []
        assert not state_file.exists()

    async def test_unset_bot_login_does_not_exclude_anyone(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        shared_dir = tmp_path / "shared"
        state_file = shared_dir / github_gateway.STATE_FILENAME
        manager = GitHubWebhookManager(
            config=GitHubWebhookConfig(enabled=True, repos=[REPO]),
            shared_dir=shared_dir,
            state_file=state_file,
        )
        sends: list[dict[str, str]] = []
        monkeypatch.setattr(
            manager,
            "_send",
            lambda to, content, kind, key: sends.append({"to": to, "kind": kind}),
        )
        await manager.start()
        try:
            await manager.handle_event(
                "issue_comment", _comment_payload(event="issue_comment", author=BOT_LOGIN)
            )
        finally:
            await manager.stop()
        assert len(sends) == 1

    async def test_bot_review_is_ignored(self, gateway) -> None:
        manager, sends, state_file = gateway
        await manager.handle_event(
            "pull_request_review",
            _review_payload(author=BOT_LOGIN),
        )
        assert sends == []
        assert not state_file.exists()

    @pytest.mark.parametrize(
        ("event", "dedupe_key"),
        [
            ("issue_comment", "issue-comment:101"),
            ("pull_request_review_comment", "review-comment:101"),
        ],
    )
    async def test_created_comment_is_deduped_by_kind_and_id(
        self,
        gateway,
        event: str,
        dedupe_key: str,
    ) -> None:
        manager, sends, state_file = gateway
        payload = _comment_payload(event=event)
        await manager.handle_event(event, payload)
        await manager.handle_event(event, payload)
        assert len(sends) == 1
        assert sends[0]["to"] == "rin"
        assert sends[0]["key"] == dedupe_key
        assert "Please fix this edge case." in sends[0]["content"]
        state = json.loads(state_file.read_text(encoding="utf-8"))
        assert dedupe_key in state["seen_comments"]

    async def test_updated_comment_is_ignored(self, gateway) -> None:
        manager, sends, state_file = gateway
        await manager.handle_event(
            "issue_comment",
            _comment_payload(action="edited"),
        )
        assert sends == []
        assert not state_file.exists()

    async def test_review_is_deduped_and_changes_requested_is_emphasized(self, gateway) -> None:
        manager, sends, state_file = gateway
        payload = _review_payload()
        await manager.handle_event("pull_request_review", payload)
        await manager.handle_event("pull_request_review", payload)
        assert len(sends) == 1
        assert sends[0]["key"] == "review:202"
        assert "【CHANGES_REQUESTED】" in sends[0]["content"]
        state = json.loads(state_file.read_text(encoding="utf-8"))
        assert "review:202" in state["seen_comments"]

    async def test_non_submitted_review_is_ignored(self, gateway) -> None:
        manager, sends, state_file = gateway
        await manager.handle_event(
            "pull_request_review",
            _review_payload(action="edited"),
        )
        assert sends == []
        assert not state_file.exists()


class TestWorkflowRunDispatch:
    async def test_ci_failure_is_deduped_by_pr_and_sha(self, gateway) -> None:
        manager, sends, state_file = gateway
        payload = _workflow_payload()
        await manager.handle_event("workflow_run", payload)
        await manager.handle_event("workflow_run", payload)
        assert len(sends) == 1
        assert sends[0]["to"] == "rin"
        assert sends[0]["kind"] == "ci"
        key = f"{REPO}#17_{SHA_1[:8]}"
        assert sends[0]["key"] == key
        state = json.loads(state_file.read_text(encoding="utf-8"))
        assert key in state["ci_notified"]

    @pytest.mark.parametrize(
        "payload",
        [
            _workflow_payload(action="requested"),
            _workflow_payload(conclusion="success"),
            _workflow_payload() | {"workflow_run": {"conclusion": "failure", "head_sha": SHA_1, "pull_requests": []}},
        ],
    )
    async def test_non_completed_failure_or_unassociated_run_is_ignored(self, gateway, payload) -> None:
        manager, sends, state_file = gateway
        await manager.handle_event("workflow_run", payload)
        assert sends == []
        assert not state_file.exists()

    async def test_ci_failure_for_non_head_sha_is_ignored(self, gateway) -> None:
        manager, sends, state_file = gateway
        await manager.handle_event(
            "workflow_run",
            _workflow_payload(head_sha=SHA_1, pr_head_sha=SHA_2),
        )
        assert sends == []
        assert not state_file.exists()


class TestSharedStateLocking:
    def test_fallback_poller_respects_webhook_dedupe_state(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        script_path = Path(__file__).parents[3] / "scripts" / "pr-review-dispatch.py"
        spec = importlib.util.spec_from_file_location("pr_review_dispatch", script_path)
        assert spec is not None and spec.loader is not None
        poller = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(poller)

        state = {
            "prs": {
                f"{REPO}#17": {
                    "sha": SHA_1,
                    "sha_seen_at": "2026-07-14T00:00:00Z",
                    "notified_sha": SHA_1,
                    "title": "already sent by webhook",
                }
            },
            "last_comment_check": "2026-07-14T00:00:00Z",
            "seen_comments": {
                "review-comment:55": "2026-07-14T00:00:00Z",
                "issue-comment:66": "2026-07-14T00:00:00Z",
            },
            "ci_notified": {f"{REPO}#17_{SHA_1[:8]}": "2026-07-14T00:00:00Z"},
            "consecutive_failures": 0,
        }

        def fake_gh(args: list[str]) -> str:
            if args[0] == "api":
                comment_id = 55 if "/pulls/comments" in args[1] else 66
                return json.dumps(
                    [
                        {
                            "id": comment_id,
                            "user": {"login": "reviewer"},
                            "body": "already seen",
                            "html_url": "https://github.test/comment",
                        }
                    ]
                )
            fields = args[args.index("--json") + 1]
            if fields == "number,title,headRefOid,isDraft":
                return json.dumps(
                    [{"number": 17, "title": "PR", "headRefOid": SHA_1, "isDraft": False}]
                )
            return json.dumps(
                [
                    {
                        "number": 17,
                        "headRefOid": SHA_1,
                        "statusCheckRollup": [{"name": "test", "conclusion": "FAILURE"}],
                    }
                ]
            )

        send = MagicMock()
        monkeypatch.setattr(poller, "REPOS", [REPO])
        monkeypatch.setattr(poller, "gh", fake_gh)
        monkeypatch.setattr(poller, "send", send)

        poller.check_commits(state)
        poller.check_comments(state)
        poller.check_ci(state)

        send.assert_not_called()
        assert poller.STATE_LOCK.name == "pr-review-dispatch-state.lock"

    def test_flock_wraps_update_and_unknown_fields_survive(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        state_file = tmp_path / "pr-review-dispatch-state.json"
        original = {
            "prs": {},
            "seen_comments": {},
            "ci_notified": {},
            "last_comment_check": "2026-07-14T01:02:03Z",
            "consecutive_failures": 4,
            "future_schema_field": {"nested": [1, 2, 3]},
        }
        state_file.write_text(json.dumps(original), encoding="utf-8")
        calls: list[int] = []
        real_flock = github_gateway.fcntl.flock

        def recording_flock(fd: int, operation: int) -> None:
            calls.append(operation)
            real_flock(fd, operation)

        monkeypatch.setattr(github_gateway.fcntl, "flock", recording_flock)
        with locked_dispatch_state(state_file) as state:
            state["seen_comments"]["issue-comment:9"] = "2026-07-14T02:00:00Z"

        restored = json.loads(state_file.read_text(encoding="utf-8"))
        assert restored["future_schema_field"] == {"nested": [1, 2, 3]}
        assert restored["last_comment_check"] == original["last_comment_check"]
        assert restored["consecutive_failures"] == 4
        assert restored["seen_comments"]["issue-comment:9"] == "2026-07-14T02:00:00Z"
        assert calls == [github_gateway.fcntl.LOCK_EX, github_gateway.fcntl.LOCK_UN]
        assert state_file.with_suffix(".lock").exists()

    @pytest.mark.parametrize("initial", ["", "not-json", "[]"])
    def test_missing_or_invalid_schema_recovers_to_complete_state(
        self,
        tmp_path: Path,
        initial: str,
    ) -> None:
        state_file = tmp_path / "state.json"
        if initial:
            state_file.write_text(initial, encoding="utf-8")
        with locked_dispatch_state(state_file) as state:
            state["seen_comments"]["review:1"] = "now"
        restored = json.loads(state_file.read_text(encoding="utf-8"))
        assert set(("prs", "seen_comments", "ci_notified")) <= restored.keys()
        assert restored["seen_comments"]["review:1"] == "now"
