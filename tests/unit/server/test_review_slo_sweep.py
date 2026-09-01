"""Unit tests for the PR review SLO sweep (server.github_gateway.check_review_slo)."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from core.config.schemas import GitHubWebhookConfig
from server import github_gateway
from server.github_gateway import GitHubWebhookManager, locked_dispatch_state

REPO = "example-org/example-repo"
SHA_1 = "1" * 40
REVIEWER = "animaworks-reviewer"


def _now() -> datetime:
    return datetime.now(UTC)


def _iso_minutes_ago(minutes: int) -> str:
    return (_now() - timedelta(minutes=minutes)).strftime("%Y-%m-%dT%H:%M:%SZ")


class _Harness:
    """A started manager with gh / dispatch / send all mocked."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **config_kw):
        self.shared_dir = tmp_path / "shared"
        self.state_file = self.shared_dir / github_gateway.STATE_FILENAME
        defaults = dict(
            enabled=True,
            repos=[REPO],
            reviewer_anima="sumire",
            dispatcher_anima="rin",
            reviewer_login=REVIEWER,
        )
        defaults.update(config_kw)
        self.manager = GitHubWebhookManager(
            config=GitHubWebhookConfig(**defaults),
            shared_dir=self.shared_dir,
            state_file=self.state_file,
        )
        monkeypatch.setattr(github_gateway, "dispatch_direct_task", MagicMock(return_value=True))
        self.sends: list[dict] = []
        monkeypatch.setattr(
            self.manager,
            "_send",
            lambda to, content, kind, key: self.sends.append({"to": to, "content": content, "kind": kind, "key": key}),
        )
        self.gh_responses: dict[str, str] = {"head": SHA_1, "reviews": ""}

        def fake_gh(args):
            if args and args[0] == "pr":
                return self.gh_responses["head"]
            if args and args[0] == "api":
                return self.gh_responses["reviews"]
            return ""

        monkeypatch.setattr(github_gateway, "gh", fake_gh)

    def seed_state(self, prs: dict) -> None:
        with locked_dispatch_state(self.state_file) as state:
            for key, entry in prs.items():
                state["prs"][key] = entry
            for key, entry in prs.items():
                sha = entry["sha"]
                if entry.get("green_ago") is not None:
                    state.setdefault("ci_green_notified", {})[f"{key}_{sha[:8]}"] = _iso_minutes_ago(entry["green_ago"])

    def read_state(self) -> dict:
        return json.loads(self.state_file.read_text(encoding="utf-8"))


def _pr_entry(sha: str, *, notified: str | None = None, green_ago: int | None = None) -> dict:
    return {
        "sha": sha,
        "sha_seen_at": _now().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "notified_sha": notified if notified is not None else sha,
        "title": "SLOW PR",
        "green_ago": green_ago,
    }


def test_slo_exceeded_redispatches_and_investigates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    h = _Harness(
        tmp_path,
        monkeypatch,
        review_multipass_models=["c:gpt-5.6-sol"],
    )
    h.seed_state({f"{REPO}#17": _pr_entry(SHA_1, green_ago=90)})

    alerted = h.manager.check_review_slo()

    assert len(alerted) == 1
    direct = github_gateway.dispatch_direct_task
    # multipass model pass + investigation task both dispatched
    assert direct.call_count == 2
    slo_calls = [c for c in direct.call_args_list if c.kwargs["task_id"].startswith("gh-slo-")]
    assert len(slo_calls) == 1
    assert slo_calls[0].kwargs["target"] == "rin"
    assert f"{REPO}#17" in slo_calls[0].kwargs["instruction"]
    state = h.read_state()
    slo_key = f"{REPO}#17_{SHA_1[:8]}"
    assert slo_key in state[github_gateway.REVIEW_SLO_STATE_KEY]
    # multipass entry created because none existed
    base = f"gh-ci-example-org-example-repo#17-{SHA_1[:8]}"
    assert base in state.setdefault("multi_model_passes", {})


def test_slo_deduped_per_pr_and_sha(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    h = _Harness(tmp_path, monkeypatch)
    h.seed_state({f"{REPO}#17": _pr_entry(SHA_1, green_ago=90)})

    h.manager.check_review_slo()
    h.manager.check_review_slo()

    direct = github_gateway.dispatch_direct_task
    slo_calls = [c for c in direct.call_args_list if c.kwargs["task_id"].startswith("gh-slo-")]
    assert len(slo_calls) == 1  # investigated only once


def test_within_slo_window_is_not_escalated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    h = _Harness(tmp_path, monkeypatch)
    h.seed_state({f"{REPO}#17": _pr_entry(SHA_1, green_ago=5)})

    alerted = h.manager.check_review_slo()

    assert alerted == []
    github_gateway.dispatch_direct_task.assert_not_called()
    assert not h.state_file.exists() or h.read_state().get(github_gateway.REVIEW_SLO_STATE_KEY) in (None, {})


def test_sha_mismatch_is_skipped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    h = _Harness(tmp_path, monkeypatch)
    # reviewed head still pending, but current GH head moved on
    h.gh_responses["head"] = "f" * 40
    h.seed_state({f"{REPO}#17": _pr_entry(SHA_1, green_ago=90)})

    alerted = h.manager.check_review_slo()

    assert alerted == []
    github_gateway.dispatch_direct_task.assert_not_called()


def test_already_reviewed_head_is_skipped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    h = _Harness(tmp_path, monkeypatch)
    h.gh_responses["reviews"] = json.dumps([{"id": 1, "user": {"login": REVIEWER}, "commit_id": SHA_1, "state": "APPROVED"}])
    h.seed_state({f"{REPO}#17": _pr_entry(SHA_1, green_ago=90)})

    alerted = h.manager.check_review_slo()

    assert alerted == []
    github_gateway.dispatch_direct_task.assert_not_called()


def test_missing_ci_green_is_not_escalated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    h = _Harness(tmp_path, monkeypatch)
    h.seed_state({f"{REPO}#17": _pr_entry(SHA_1, green_ago=None)})

    alerted = h.manager.check_review_slo()

    assert alerted == []
    github_gateway.dispatch_direct_task.assert_not_called()
