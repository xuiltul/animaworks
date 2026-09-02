"""Unit tests for the GitHub webhook gateway's single-notification dispatch."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from core.config.schemas import GitHubWebhookConfig
from server import github_gateway
from server.github_gateway import GitHubWebhookManager, locked_dispatch_state

REPO = "example-org/example-repo"
BOT_LOGIN = "example-bot"
REVIEWER_LOGIN = "example-reviewer"
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
    author: str = "human",
    body: str = "Please review.",
) -> dict[str, Any]:
    return {
        "action": action,
        "number": number,
        "repository": {"full_name": repo},
        "pull_request": {
            "number": number,
            "draft": draft,
            "title": title,
            "body": body,
            "user": {"login": author},
            "head": {"sha": sha},
            "html_url": f"https://github.test/pulls/{number}",
        },
    }


def _comment_payload(
    *,
    event: str = "issue_comment",
    action: str = "created",
    author: str = "reviewer",
    comment_id: int = 101,
    body: str = "Please fix this\nedge case.",
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "action": action,
        "repository": {"full_name": REPO},
        "comment": {
            "id": comment_id,
            "user": {"login": author},
            "body": body,
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
    body: str = "The race remains.",
) -> dict[str, Any]:
    return {
        "action": action,
        "repository": {"full_name": REPO},
        "pull_request": {"number": 17, "title": "Webhook dispatch", "head": {"sha": SHA_1}},
        "review": {
            "id": review_id,
            "user": {"login": author},
            "state": state,
            "body": body,
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
    name: str = "quality-gate",
) -> dict[str, Any]:
    pr: dict[str, Any] = {"number": number}
    if pr_head_sha is not None:
        pr["head"] = {"sha": pr_head_sha}
    return {
        "action": action,
        "repository": {"full_name": REPO},
        "workflow_run": {
            "name": name,
            "conclusion": conclusion,
            "head_sha": head_sha,
            "html_url": "https://github.test/actions/runs/1",
            "pull_requests": [pr],
        },
    }


@pytest.fixture
async def gateway(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Started manager with captured local Messenger sends."""
    monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(tmp_path / "data"))
    shared_dir = tmp_path / "shared"
    state_file = shared_dir / github_gateway.STATE_FILENAME
    config = GitHubWebhookConfig(
        enabled=True,
        repos=[REPO],
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
        assert config.dispatcher_anima == "rin"
        assert config.quiet_seconds == 180

    def test_config_ignores_legacy_keys_without_raising(self) -> None:
        """H-4 compat: an old config.json may still carry removed keys."""
        legacy = {
            "enabled": True,
            "repos": ["o/r"],
            "reviewer_anima": "sumire",
            "implementer_anima": "natsume",
            "review_multipass_models": ["c:gpt-5.6-sol"],
            "review_synth_model": "c:gpt-5.6-sol",
        }
        config = GitHubWebhookConfig(**legacy)
        assert config.enabled is True
        assert config.repos == ["o/r"]

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
        from unittest.mock import MagicMock

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

    async def test_push_resets_debounce_and_only_latest_sha_notifies(
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
        # Only the second (latest) debounce actually notifies; the first was
        # cancelled by the second push before its quiet period elapsed.
        assert len(sends) == 1
        assert sends[0]["to"] == "rin"
        assert sends[0]["kind"] == "push"
        state = json.loads(state_file.read_text(encoding="utf-8"))
        assert state["prs"][f"{REPO}#17"]["notified"]["push"] == SHA_2

    async def test_closed_notifies_immediately_and_removes_only_pr_state(self, gateway) -> None:
        manager, sends, state_file = gateway
        await manager.handle_event("pull_request", _pr_payload())
        key = f"{REPO}#17"
        task = manager._debounce_tasks[key]
        with locked_dispatch_state(state_file) as state:
            state["deployment_marker"] = {"keep": True}

        await manager.handle_event("pull_request", _pr_payload(action="closed"))
        await asyncio.sleep(0)

        assert len(sends) == 1
        assert sends[0]["kind"] == "closed"
        state = json.loads(state_file.read_text(encoding="utf-8"))
        assert key not in state["prs"]
        assert state["deployment_marker"] == {"keep": True}
        assert key not in manager._debounce_tasks
        assert task.cancelled()

    async def test_superseded_debounce_does_not_notify_for_stale_head(
        self, gateway, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """require_sha guard: a debounce whose head was superseded must not fire."""
        manager, sends, state_file = gateway
        key = f"{REPO}#17"
        with locked_dispatch_state(state_file) as state:
            state["prs"][key] = {"sha": SHA_2, "title": "latest", "notified": {}}
        await asyncio.to_thread(
            manager._notify_once,
            key=key,
            repo=REPO,
            number=17,
            sha=SHA_1,
            category="push",
            title="stale",
            event_label="synchronize",
            author="human",
            url="https://github.test/pulls/17",
            body="",
            require_sha=SHA_1,
        )
        assert sends == []

    async def test_repeat_notify_once_on_same_category_and_sha_is_deduped(self, gateway) -> None:
        manager, sends, _state_file = gateway
        kwargs = dict(
            key=f"{REPO}#17",
            repo=REPO,
            number=17,
            sha=SHA_1,
            category="push",
            title="t",
            event_label="opened",
            author="human",
            url="https://github.test/pulls/17",
            body="",
        )
        manager._notify_once(**kwargs)
        manager._notify_once(**kwargs)
        assert len(sends) == 1


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

    @pytest.mark.parametrize("event", ["issue_comment", "pull_request_review_comment"])
    async def test_created_comment_notifies_and_dedupes_by_head_sha(self, gateway, event: str) -> None:
        manager, sends, state_file = gateway
        # Seed a known head SHA so the comment (which carries none itself)
        # dedupes against it.
        key = f"{REPO}#17"
        with locked_dispatch_state(state_file) as state:
            state["prs"][key] = {"sha": SHA_1, "title": "t", "notified": {}}

        payload = _comment_payload(event=event)
        await manager.handle_event(event, payload)
        await manager.handle_event(event, payload)

        assert len(sends) == 1
        assert sends[0]["to"] == "rin"
        assert sends[0]["kind"] == "comment"
        assert "Please fix this\nedge case." in sends[0]["content"]
        state = json.loads(state_file.read_text(encoding="utf-8"))
        assert state["prs"][key]["notified"][f"comment:{payload['comment']['id']}"] == SHA_1

    async def test_second_comment_on_same_head_is_also_notified(self, gateway) -> None:
        manager, sends, state_file = gateway
        key = f"{REPO}#17"
        with locked_dispatch_state(state_file) as state:
            state["prs"][key] = {"sha": SHA_1, "title": "t", "notified": {}}
        await manager.handle_event("issue_comment", _comment_payload(comment_id=1001))
        await manager.handle_event("issue_comment", _comment_payload(comment_id=1002))
        assert len(sends) == 2

    async def test_updated_comment_is_ignored(self, gateway) -> None:
        manager, sends, state_file = gateway
        await manager.handle_event(
            "issue_comment",
            _comment_payload(action="edited"),
        )
        assert sends == []
        assert not state_file.exists()

    @pytest.mark.parametrize("review_state", ["commented", "approved", "changes_requested"])
    async def test_every_external_review_notifies_once(self, gateway, review_state: str) -> None:
        manager, sends, state_file = gateway
        payload = _review_payload(state=review_state, body="The race remains.")
        await manager.handle_event("pull_request_review", payload)
        await manager.handle_event("pull_request_review", payload)
        assert len(sends) == 1
        assert sends[0]["kind"] == "review"
        assert "The race remains." in sends[0]["content"]
        state = json.loads(state_file.read_text(encoding="utf-8"))
        assert state["prs"][f"{REPO}#17"]["notified"][f"review:{payload['review']['id']}"] == SHA_1

    async def test_body_excerpt_is_truncated_to_500_chars(self, gateway) -> None:
        manager, sends, _state_file = gateway
        body = "x" * 800 + "TAIL-MARKER"
        await manager.handle_event("pull_request_review", _review_payload(body=body))
        assert len(sends) == 1
        assert "TAIL-MARKER" not in sends[0]["content"]
        assert "x" * 500 in sends[0]["content"]

    async def test_non_submitted_review_is_ignored(self, gateway) -> None:
        manager, sends, state_file = gateway
        await manager.handle_event(
            "pull_request_review",
            _review_payload(action="edited"),
        )
        assert sends == []
        assert not state_file.exists()

    async def test_bot_review_is_excluded_not_forwarded(self, gateway) -> None:
        manager, sends, _state_file = gateway
        await manager.handle_event(
            "pull_request_review",
            _review_payload(author=BOT_LOGIN, state="commented", body="PASS"),
        )
        assert sends == []

    async def test_reviewer_login_reviews_are_excluded(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        shared_dir = tmp_path / "shared"
        state_file = shared_dir / github_gateway.STATE_FILENAME
        manager = GitHubWebhookManager(
            config=GitHubWebhookConfig(
                enabled=True,
                repos=[REPO],
                bot_login=BOT_LOGIN,
                reviewer_login=REVIEWER_LOGIN,
            ),
            shared_dir=shared_dir,
            state_file=state_file,
        )
        sends: list[dict[str, str]] = []
        monkeypatch.setattr(
            manager,
            "_send",
            lambda to, content, kind, key: sends.append({"to": to, "content": content, "kind": kind, "key": key}),
        )
        await manager.start()
        try:
            await manager.handle_event(
                "pull_request_review",
                _review_payload(author=REVIEWER_LOGIN, state="approved"),
            )
        finally:
            await manager.stop()
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
            await manager.handle_event("issue_comment", _comment_payload(event="issue_comment", author=BOT_LOGIN))
        finally:
            await manager.stop()
        assert len(sends) == 1


class TestWorkflowRunDispatch:
    @pytest.mark.parametrize("conclusion", ["failure", "cancelled", "timed_out", "startup_failure"])
    async def test_ci_failure_notifies_and_includes_run_name(self, gateway, conclusion: str) -> None:
        manager, sends, state_file = gateway
        payload = _workflow_payload(conclusion=conclusion)
        await manager.handle_event("workflow_run", payload)
        await manager.handle_event("workflow_run", payload)
        assert len(sends) == 1
        assert sends[0]["to"] == "rin"
        assert sends[0]["kind"] == "ci"
        assert "quality-gate" in sends[0]["content"]
        assert conclusion.upper() in sends[0]["content"]
        key = f"{REPO}#17"
        state = json.loads(state_file.read_text(encoding="utf-8"))
        assert state["prs"][key]["notified"]["ci"] == SHA_1

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

    async def test_new_workflow_failure_on_same_sha_is_deduped_by_category(self, gateway) -> None:
        """The category-level dedupe is deliberately coarse: a second failing
        run on the same head does not re-notify (2026-09 teardown noise cut)."""
        manager, sends, _state_file = gateway
        first = _workflow_payload()
        second = _workflow_payload(name="different-check")

        await manager.handle_event("workflow_run", first)
        await manager.handle_event("workflow_run", second)

        assert len(sends) == 1
        assert "quality-gate" in sends[0]["content"]


class TestSharedStateLocking:
    def test_flock_wraps_update_and_unknown_fields_survive(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        state_file = tmp_path / "pr-review-dispatch-state.json"
        original = {
            "prs": {},
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
            state["prs"]["o/r#1"] = {"sha": "a", "title": "t", "notified": {}}

        restored = json.loads(state_file.read_text(encoding="utf-8"))
        assert restored["future_schema_field"] == {"nested": [1, 2, 3]}
        assert restored["prs"]["o/r#1"]["sha"] == "a"
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
            state["prs"]["o/r#1"] = {"sha": "a", "title": "t", "notified": {}}
        restored = json.loads(state_file.read_text(encoding="utf-8"))
        assert "prs" in restored
        assert restored["prs"]["o/r#1"]["sha"] == "a"
