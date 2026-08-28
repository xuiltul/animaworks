"""is_addressed / determine_warning_stage のユニットテスト（pr-review-dispatch.py）。"""

from __future__ import annotations

import importlib.util
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "pr-review-dispatch.py"

T0 = datetime(2026, 7, 1, 12, 0, 0, tzinfo=UTC)


@pytest.fixture
def mod(tmp_path, monkeypatch):
    monkeypatch.setenv("ANIMAWORKS_SHARED_DIR", str(tmp_path))
    monkeypatch.setenv("PR_DISPATCH_REPOS", "o/r")
    monkeypatch.setenv("PR_DISPATCH_BOT_LOGIN", "bot-user")
    spec = importlib.util.spec_from_file_location("pr_stale_comment_warning", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.REPOS = ["o/r"]
    module.BOT_LOGIN = "bot-user"
    return module


# ---------------------------------------------------------------------------
# is_addressed
# ---------------------------------------------------------------------------


def test_is_addressed_pr_closed(mod):
    assert mod.is_addressed(
        pr_closed=True,
        thread_resolved=False,
        item_created_at=T0,
        bot_commit_at=None,
        bot_comment_at=None,
    )


def test_is_addressed_thread_resolved(mod):
    assert mod.is_addressed(
        pr_closed=False,
        thread_resolved=True,
        item_created_at=T0,
        bot_commit_at=None,
        bot_comment_at=None,
    )


def test_is_addressed_bot_commit_after(mod):
    assert mod.is_addressed(
        pr_closed=False,
        thread_resolved=False,
        item_created_at=T0,
        bot_commit_at=T0 + timedelta(hours=1),
        bot_comment_at=None,
    )


def test_is_addressed_bot_commit_before_not_addressed(mod):
    assert not mod.is_addressed(
        pr_closed=False,
        thread_resolved=False,
        item_created_at=T0,
        bot_commit_at=T0 - timedelta(minutes=1),
        bot_comment_at=None,
    )


def test_is_addressed_bot_comment_after(mod):
    assert mod.is_addressed(
        pr_closed=False,
        thread_resolved=False,
        item_created_at=T0,
        bot_commit_at=None,
        bot_comment_at=T0 + timedelta(minutes=30),
    )


def test_is_addressed_still_open(mod):
    assert not mod.is_addressed(
        pr_closed=False,
        thread_resolved=False,
        item_created_at=T0,
        bot_commit_at=None,
        bot_comment_at=None,
    )


# ---------------------------------------------------------------------------
# is_addressed — kind=review (CHANGES_REQUESTED)
# ---------------------------------------------------------------------------


def test_review_kind_bot_commit_does_not_address(mod):
    """Subsequent bot commits do NOT clear CHANGES_REQUESTED reviews."""
    assert not mod.is_addressed(
        pr_closed=False,
        thread_resolved=False,
        item_created_at=T0,
        bot_commit_at=T0 + timedelta(hours=3),
        bot_comment_at=None,
        kind="review",
        review_decision="CHANGES_REQUESTED",
    )


def test_review_kind_bot_comment_does_not_address(mod):
    assert not mod.is_addressed(
        pr_closed=False,
        item_created_at=T0,
        bot_commit_at=None,
        bot_comment_at=T0 + timedelta(hours=1),
        kind="review",
        review_decision="CHANGES_REQUESTED",
    )


def test_review_kind_dismissed_addresses(mod):
    assert mod.is_addressed(
        pr_closed=False,
        item_created_at=T0,
        kind="review",
        review_dismissed=True,
        review_decision="CHANGES_REQUESTED",
    )


def test_review_kind_decision_cleared_addresses(mod):
    assert mod.is_addressed(
        pr_closed=False,
        item_created_at=T0,
        kind="review",
        review_dismissed=False,
        review_decision="APPROVED",
    )


def test_review_kind_decision_review_required_addresses(mod):
    assert mod.is_addressed(
        pr_closed=False,
        item_created_at=T0,
        kind="review",
        review_decision="REVIEW_REQUIRED",
    )


def test_review_kind_still_open_when_changes_requested(mod):
    assert not mod.is_addressed(
        pr_closed=False,
        item_created_at=T0,
        kind="review",
        review_dismissed=False,
        review_decision="CHANGES_REQUESTED",
    )


def test_review_kind_pr_closed_addresses(mod):
    assert mod.is_addressed(
        pr_closed=True,
        item_created_at=T0,
        kind="review",
        review_decision="CHANGES_REQUESTED",
    )


# ---------------------------------------------------------------------------
# is_addressed / helpers — kind=ci
# ---------------------------------------------------------------------------


def test_ci_kind_still_failing_not_addressed(mod):
    assert not mod.is_addressed(
        pr_closed=False,
        kind="ci",
        ci_still_failing=True,
        head_sha_changed=False,
    )


def test_ci_kind_resolved_when_no_longer_failing(mod):
    assert mod.is_addressed(
        pr_closed=False,
        kind="ci",
        ci_still_failing=False,
        head_sha_changed=False,
    )


def test_ci_kind_head_sha_changed_retires_old_item(mod):
    """Old SHA item is retired; a failing new SHA becomes a new item_id."""
    assert mod.is_addressed(
        pr_closed=False,
        kind="ci",
        ci_still_failing=True,
        head_sha_changed=True,
    )


def test_ci_kind_pr_closed_addresses(mod):
    assert mod.is_addressed(
        pr_closed=True,
        kind="ci",
        ci_still_failing=True,
    )


def test_ci_stale_item_id_includes_full_sha(mod):
    assert mod.ci_stale_item_id("o/r", 42, "abcdef0123456789") == "ci:o/r#42:abcdef0123456789"


def test_ci_item_id_changes_with_sha(mod):
    """SHA change + new failure is tracked as a distinct item from the start."""
    old_id = mod.ci_stale_item_id("o/r", 1, "aaa111")
    new_id = mod.ci_stale_item_id("o/r", 1, "bbb222")
    assert old_id != new_id
    # old item addressed via head_sha_changed; new item still open
    assert mod.is_addressed(pr_closed=False, kind="ci", ci_still_failing=True, head_sha_changed=True)
    assert not mod.is_addressed(pr_closed=False, kind="ci", ci_still_failing=True, head_sha_changed=False)


def test_failed_check_names_filters_failure_only(mod):
    rollup = [
        {"name": "lint", "conclusion": "SUCCESS"},
        {"name": "test", "conclusion": "FAILURE"},
        {"name": "build", "conclusion": "NEUTRAL"},
        {"name": "optional", "conclusion": "SKIPPED"},
        {"name": "pending", "conclusion": None},
    ]
    assert mod.failed_check_names(rollup) == ["test"]
    assert mod.failed_check_names([]) == []
    assert mod.failed_check_names(None) == []


def test_failed_check_names_includes_cancelled_and_timed_out(mod):
    """CANCELLED (e.g. 60-min job timeout) and TIMED_OUT count as failing CI (2026-07-27)."""
    rollup = [
        {"name": "unit", "conclusion": "SUCCESS"},
        {"name": "feature", "conclusion": "CANCELLED"},
        {"name": "e2e", "conclusion": "TIMED_OUT"},
        {"name": "setup", "conclusion": "STARTUP_FAILURE"},
        {"name": "skipped", "conclusion": "SKIPPED"},
    ]
    assert mod.failed_check_names(rollup) == ["feature", "e2e", "setup"]
    assert {"FAILURE", "CANCELLED", "TIMED_OUT"} <= mod.FAILING_CI_CONCLUSIONS


def test_ci_rewarn_stage_after_interval(mod):
    """Same SHA failing continuously: rewarn after REWARN hours from last_warned."""
    first_seen = T0
    last_warned = T0  # immediate first warn (or check_ci pre-seeded last_warned)
    now = last_warned + timedelta(hours=4)
    assert (
        mod.determine_warning_stage(
            item_created_at=first_seen,
            now=now,
            last_warned=last_warned,
            escalated_at=None,
            warn_hours=0.0,
            rewarn_hours=4.0,
            escalate_hours=8.0,
        )
        == "rewarn"
    )


def test_ci_immediate_warn_with_zero_warn_hours(mod):
    now = T0  # age 0
    assert (
        mod.determine_warning_stage(
            item_created_at=T0,
            now=now,
            last_warned=None,
            escalated_at=None,
            warn_hours=0.0,
        )
        == "warn"
    )


def test_ci_suppressed_when_check_ci_already_notified(mod):
    """If check_ci already warned, last_warned is seeded → no immediate rewarn."""
    now = T0 + timedelta(hours=1)
    assert (
        mod.determine_warning_stage(
            item_created_at=T0,
            now=now,
            last_warned=T0,
            escalated_at=None,
            warn_hours=0.0,
            rewarn_hours=4.0,
            escalate_hours=8.0,
        )
        == "none"
    )


# ---------------------------------------------------------------------------
# format / message templates
# ---------------------------------------------------------------------------


def test_format_review_stale_line(mod):
    line = mod._format_stale_line(
        repo="o/r",
        number=3814,
        author="reviewer1",
        body="please fix",
        url="https://example/pr/3814",
        created_at=T0,
        now=T0 + timedelta(hours=5),
        kind="review",
    )
    assert "CHANGES_REQUESTED" in line
    assert "#3814" in line
    assert "@reviewer1" in line
    assert "経過5h" in line


def test_format_ci_stale_line(mod):
    line = mod._format_stale_line(
        repo="o/r",
        number=3849,
        author="ci",
        body="test",
        url="https://example/pr/3849",
        created_at=T0,
        now=T0 + timedelta(hours=3),
        kind="ci",
        sha="abcdef012345",
        failed_checks=["unit", "lint"],
    )
    assert "CI失敗" in line
    assert "abcdef01" in line
    assert "unit" in line
    assert "経過3h" in line


def test_stale_message_review_template(mod):
    msg = mod._stale_message(["- PR #1 ..."], kind="review")
    assert "CHANGES_REQUESTED" in msg
    assert "再レビュー" in msg


def test_stale_message_ci_template(mod):
    msg = mod._stale_message(["- PR #1 ..."], kind="ci")
    assert "CI失敗" in msg
    assert "修正commit" in msg


# ---------------------------------------------------------------------------
# determine_warning_stage — boundaries
# ---------------------------------------------------------------------------


def test_stage_none_just_before_warn(mod):
    now = T0 + timedelta(hours=2) - timedelta(seconds=1)
    assert (
        mod.determine_warning_stage(
            item_created_at=T0,
            now=now,
            last_warned=None,
            escalated_at=None,
            warn_hours=2.0,
            rewarn_hours=4.0,
            escalate_hours=8.0,
        )
        == "none"
    )


def test_stage_warn_just_after_warn_threshold(mod):
    now = T0 + timedelta(hours=2)
    assert (
        mod.determine_warning_stage(
            item_created_at=T0,
            now=now,
            last_warned=None,
            escalated_at=None,
            warn_hours=2.0,
            rewarn_hours=4.0,
            escalate_hours=8.0,
        )
        == "warn"
    )


def test_stage_rewarn_suppressed_within_interval(mod):
    last = T0 + timedelta(hours=2)
    now = last + timedelta(hours=4) - timedelta(seconds=1)
    assert (
        mod.determine_warning_stage(
            item_created_at=T0,
            now=now,
            last_warned=last,
            escalated_at=None,
            warn_hours=2.0,
            rewarn_hours=4.0,
            escalate_hours=8.0,
        )
        == "none"
    )


def test_stage_rewarn_after_interval(mod):
    last = T0 + timedelta(hours=2)
    now = last + timedelta(hours=4)
    assert (
        mod.determine_warning_stage(
            item_created_at=T0,
            now=now,
            last_warned=last,
            escalated_at=None,
            warn_hours=2.0,
            rewarn_hours=4.0,
            escalate_hours=8.0,
        )
        == "rewarn"
    )


def test_stage_escalate_first_time(mod):
    now = T0 + timedelta(hours=8)
    last = T0 + timedelta(hours=2)
    assert (
        mod.determine_warning_stage(
            item_created_at=T0,
            now=now,
            last_warned=last,
            escalated_at=None,
            warn_hours=2.0,
            rewarn_hours=4.0,
            escalate_hours=8.0,
        )
        == "escalate"
    )


def test_stage_escalate_recently_suppressed_within_repeat_interval(mod):
    """Escalated recently → no re-escalate until ESCALATE_REPEAT_HOURS elapses."""
    now = T0 + timedelta(hours=10)
    last_warned = T0 + timedelta(hours=9, minutes=59)  # within rewarn window
    escalated_at = T0 + timedelta(hours=8)
    assert (
        mod.determine_warning_stage(
            item_created_at=T0,
            now=now,
            last_warned=last_warned,
            escalated_at=escalated_at,
            warn_hours=2.0,
            rewarn_hours=4.0,
            escalate_hours=8.0,
        )
        == "none"
    )


def test_stage_escalates_again_after_repeat_interval(mod):
    """Escalation repeats (decayed, never zero) after ESCALATE_REPEAT_HOURS."""
    escalated_at = T0 + timedelta(hours=8)
    now = escalated_at + timedelta(hours=mod.ESCALATE_REPEAT_HOURS)
    assert (
        mod.determine_warning_stage(
            item_created_at=T0,
            now=now,
            last_warned=now - timedelta(minutes=1),
            escalated_at=escalated_at,
            warn_hours=2.0,
            rewarn_hours=4.0,
            escalate_hours=8.0,
        )
        == "escalate"
    )


def test_decayed_interval_doubles_and_caps(mod):
    assert mod.decayed_interval_hours(0.25, 0) == 0.25
    assert mod.decayed_interval_hours(0.25, 1) == 0.25
    assert mod.decayed_interval_hours(0.25, 2) == 0.5
    assert mod.decayed_interval_hours(0.25, 3) == 1.0
    assert mod.decayed_interval_hours(0.25, 10) == mod.REWARN_CAP_HOURS


def test_stage_custom_thresholds(mod):
    now = T0 + timedelta(hours=1)
    assert (
        mod.determine_warning_stage(
            item_created_at=T0,
            now=now,
            last_warned=None,
            escalated_at=None,
            warn_hours=0.5,
            rewarn_hours=1.0,
            escalate_hours=24.0,
        )
        == "warn"
    )


# ---------------------------------------------------------------------------
# FIX_REQUEST_PATTERN
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body,expect",
    [
        ("ここを修正してください", True),
        ("Please fix this", True),
        ("looks good to me", False),
        ("CHANGE the name", True),
        ("対応してください", True),
        ("お願いします", True),
        ("直してほしい", True),
        ("required change", True),
        ("address the edge case", True),
    ],
)
def test_fix_request_pattern(mod, body, expect):
    matched = mod.FIX_REQUEST_PATTERN.search(body) is not None
    assert matched is expect


REVIEWER_LOGIN = "animaworks-reviewer"


def test_plain_required_field_is_not_fix_request(mod):
    assert not mod.is_fix_request(
        body="住所は required フィールドです",
        mention_logins=[REVIEWER_LOGIN],
    )


def test_mentioned_please_fix_is_fix_request(mod):
    assert mod.is_fix_request(
        body="@animaworks-reviewer ここ直してください",
        mention_logins=[REVIEWER_LOGIN],
    )


def test_changes_requested_is_fix_request_regardless_of_body(mod):
    assert mod.is_fix_request(body="", review_state="CHANGES_REQUESTED", mention_logins=[])
    assert mod.is_fix_request(body="LGTM", review_state="changes_requested", mention_logins=[])
    assert mod.is_fix_request(
        body="住所は required フィールドです",
        review_state="CHANGES_REQUESTED",
        mention_logins=[],
    )


def test_mention_without_pattern_is_not_fix_request(mod):
    assert not mod.is_fix_request(
        body="LGTMです @animaworks-reviewer",
        mention_logins=[REVIEWER_LOGIN],
    )


def test_mention_match_is_case_insensitive(mod):
    assert mod.is_fix_request(
        body="@AnimaWorks-Reviewer please fix this",
        mention_logins=[REVIEWER_LOGIN],
    )


def test_collect_mention_logins_strips_and_dedupes(mod):
    assert mod.collect_mention_logins(
        "@animaworks-reviewer",
        "animaworks-reviewer",
        "",
        None,
        " natsume ",
        "Natsume",
    ) == ("animaworks-reviewer", "natsume")


def test_configured_mention_logins_from_env_and_config(mod, monkeypatch):
    mod.BOT_LOGIN = "animaworks-dev-team"
    mod.REVIEWER_LOGIN = "animaworks-reviewer"
    mod.FIXER = "natsume"
    monkeypatch.setattr(mod, "_github_webhook_mention_logins", lambda: ("cfg-bot", "mio"))
    assert set(mod.configured_mention_logins()) == {
        "animaworks-dev-team",
        "animaworks-reviewer",
        "natsume",
        "cfg-bot",
        "mio",
    }


def test_configured_mention_logins_not_hardcoded(mod, monkeypatch):
    mod.BOT_LOGIN = ""
    mod.REVIEWER_LOGIN = ""
    mod.FIXER = ""
    monkeypatch.setattr(mod, "_github_webhook_mention_logins", lambda: ())
    assert mod.configured_mention_logins() == ()


def test_github_webhook_mention_logins_reads_config_fields(mod, monkeypatch):
    webhook = SimpleNamespace(
        bot_login="cfg-bot",
        reviewer_login="cfg-reviewer",
        implementer_anima="mio",
    )
    monkeypatch.setattr(
        "core.config.models.load_config",
        lambda: SimpleNamespace(github_webhook=webhook),
    )
    assert set(mod._github_webhook_mention_logins()) == {"cfg-bot", "cfg-reviewer", "mio"}


def _collect_stale(mod, monkeypatch, *, reviews=None, issue_comments=None, review_comments=None):
    reviews = reviews or []
    issue_comments = issue_comments or []
    review_comments = review_comments or []

    def fake_gh(args: list[str]) -> str:
        endpoint = args[1] if len(args) > 1 else ""
        if endpoint == "graphql":
            return json.dumps({"data": {"repository": {"pullRequest": {"reviewThreads": {"nodes": []}}}}})
        if endpoint.endswith("/reviews"):
            return json.dumps(reviews)
        if "/issues/" in endpoint and "/comments" in endpoint:
            return json.dumps(issue_comments)
        if "/pulls/" in endpoint and "/comments" in endpoint:
            return json.dumps(review_comments)
        if "/commits" in endpoint:
            return json.dumps([])
        raise AssertionError(f"unexpected gh args: {args}")

    monkeypatch.setattr(mod, "gh", fake_gh)
    monkeypatch.setattr(mod, "_github_webhook_mention_logins", lambda: ())
    mod.BOT_LOGIN = "animaworks-dev-team"
    mod.REVIEWER_LOGIN = "animaworks-reviewer"
    mod.FIXER = "natsume"
    return mod._collect_pr_stale_items("o/r", 1, pr_meta={})


def test_collect_plain_required_comment_is_not_stale_item(mod, monkeypatch):
    items = _collect_stale(
        mod,
        monkeypatch,
        issue_comments=[
            {
                "id": 11,
                "user": {"login": "human"},
                "body": "住所は required フィールドです",
                "created_at": "2026-06-01T00:00:00Z",
                "html_url": "https://gh.test/c/11",
            }
        ],
    )
    assert items == []


def test_collect_mentioned_fix_comment_is_stale_item(mod, monkeypatch):
    items = _collect_stale(
        mod,
        monkeypatch,
        issue_comments=[
            {
                "id": 12,
                "user": {"login": "human"},
                "body": "@animaworks-reviewer ここ直してください",
                "created_at": "2026-06-01T00:00:00Z",
                "html_url": "https://gh.test/c/12",
            }
        ],
    )
    assert [item["item_id"] for item in items] == ["comment:12"]


def test_collect_changes_requested_review_is_stale_item(mod, monkeypatch):
    items = _collect_stale(
        mod,
        monkeypatch,
        reviews=[
            {
                "id": 99,
                "state": "CHANGES_REQUESTED",
                "user": {"login": "human-reviewer"},
                "body": "looks fine overall",
                "submitted_at": "2026-06-01T00:00:00Z",
                "html_url": "https://gh.test/r/99",
            }
        ],
    )
    assert [item["item_id"] for item in items] == ["review:99"]


def test_collect_commented_review_with_mention_is_not_stale_item(mod, monkeypatch):
    items = _collect_stale(
        mod,
        monkeypatch,
        reviews=[
            {
                "id": 88,
                "state": "COMMENTED",
                "user": {"login": "human-reviewer"},
                "body": "@animaworks-reviewer please fix this",
                "submitted_at": "2026-06-01T00:00:00Z",
                "html_url": "https://gh.test/r/88",
            }
        ],
    )
    assert items == []


# ---------------------------------------------------------------------------
# dry-run send
# ---------------------------------------------------------------------------


def test_dry_run_send_does_not_call_messenger(mod, monkeypatch, tmp_path):
    monkeypatch.setattr(mod, "DRY_RUN", True)
    monkeypatch.setattr(mod, "LOG_FILE", tmp_path / "dispatch.log")
    mod.LOG_FILE.write_text("", encoding="utf-8")
    mod.send("rin", "【警告】test body")
    log_text = mod.LOG_FILE.read_text(encoding="utf-8")
    assert "DRY_RUN send -> rin" in log_text


def test_parse_gh_time_zulu(mod):
    dt = mod.parse_gh_time("2026-07-01T12:00:00Z")
    assert dt == T0


# ---------------------------------------------------------------------------
# check_unaddressed DRY_RUN + mock (no exception)
# ---------------------------------------------------------------------------


def test_check_unaddressed_dry_run_with_mocks(mod, monkeypatch, tmp_path):
    """End-to-end check_unaddressed under DRY_RUN with mocked gh responses."""
    monkeypatch.setattr(mod, "DRY_RUN", True)
    monkeypatch.setattr(mod, "LOG_FILE", tmp_path / "dispatch.log")
    mod.LOG_FILE.write_text("", encoding="utf-8")

    pr_list = [
        {
            "number": 10,
            "headRefOid": "deadbeefcafebabe",
            "statusCheckRollup": [
                {"name": "tests", "conclusion": "FAILURE"},
                {"name": "lint", "conclusion": "SUCCESS"},
            ],
            "reviewDecision": "CHANGES_REQUESTED",
            "url": "https://github.com/o/r/pull/10",
        }
    ]
    reviews = [
        {
            "id": 99,
            "state": "CHANGES_REQUESTED",
            "user": {"login": "human-reviewer"},
            "body": "please fix the edge case",
            "submitted_at": "2026-06-01T00:00:00Z",
            "html_url": "https://github.com/o/r/pull/10#pullrequestreview-99",
        }
    ]
    empty = []
    graphql = {"data": {"repository": {"pullRequest": {"reviewThreads": {"nodes": []}}}}}

    def fake_gh(args: list[str]) -> str:
        joined = " ".join(args)
        if args[:2] == ["pr", "list"] or (len(args) >= 2 and args[0] == "pr" and args[1] == "list"):
            return json.dumps(pr_list)
        if "pulls/10/reviews" in joined:
            return json.dumps(reviews)
        if "issues/10/comments" in joined or "pulls/10/comments" in joined:
            return json.dumps(empty)
        if "pulls/10/commits" in joined:
            return json.dumps(empty)
        if "graphql" in args:
            return json.dumps(graphql)
        raise AssertionError(f"unexpected gh args: {args}")

    monkeypatch.setattr(mod, "gh", fake_gh)
    # Force ages past warn threshold for review; CI uses warn_hours=0
    fixed_now = datetime(2026, 7, 1, 12, 0, 0, tzinfo=UTC)
    monkeypatch.setattr(mod, "now_utc", lambda: fixed_now)

    state = mod.default_state()
    # Seed check_ci notification so CI stale path does not double-warn immediately
    state["ci_notified"]["o/r#10_deadbeef"] = "2026-07-01T11:00:00Z"

    mod.check_unaddressed(state)

    # review item tracked; CI item also tracked (with last_warned seeded)
    assert any(k.startswith("review:") for k in state["stale_watch"])
    assert any(k.startswith("ci:") for k in state["stale_watch"])
    ci_entry = next(v for k, v in state["stale_watch"].items() if k.startswith("ci:"))
    assert ci_entry["last_warned"] is not None  # pre-seeded from ci_notified
    log_text = mod.LOG_FILE.read_text(encoding="utf-8")
    # review is old enough for warn; CI suppressed by last_warned seed
    assert "DRY_RUN send" in log_text or "stale warn" in log_text or state["stale_watch"]


# ---------------------------------------------------------------------------
# determine_warning_stage — new default ladder (15m warn / 15m rewarn / 60m escalate)
# ---------------------------------------------------------------------------


def test_default_ladder_15_30_45_60(mod):
    """オーナー指示(2026-07-27): 15分警告→30/45分再警告→60分エスカレーション。"""
    kw = dict(item_created_at=T0, escalated_at=None)
    assert mod.determine_warning_stage(now=T0 + timedelta(minutes=14), last_warned=None, **kw) == "none"
    assert mod.determine_warning_stage(now=T0 + timedelta(minutes=15), last_warned=None, **kw) == "warn"
    assert (
        mod.determine_warning_stage(now=T0 + timedelta(minutes=30), last_warned=T0 + timedelta(minutes=15), **kw)
        == "rewarn"
    )
    assert (
        mod.determine_warning_stage(now=T0 + timedelta(minutes=45), last_warned=T0 + timedelta(minutes=30), **kw)
        == "rewarn"
    )
    assert (
        mod.determine_warning_stage(
            item_created_at=T0,
            now=T0 + timedelta(minutes=60),
            last_warned=T0 + timedelta(minutes=45),
            escalated_at=None,
        )
        == "escalate"
    )


def test_default_env_ladder_constants(mod):
    assert mod.STALE_WARN_HOURS == 0.25
    assert mod.STALE_REWARN_HOURS == 0.25
    assert mod.STALE_ESCALATE_HOURS == 1.0


def _ci_stale_item() -> dict:
    sha = "a" * 40
    return {
        "item_id": f"ci:o/r#1:{sha}",
        "kind": "ci",
        "repo": "o/r",
        "number": 1,
        "author": "ci",
        "body": "tests",
        "url": "https://gh.test/o/r/pull/1",
        "created_at": None,
        "sha": sha,
        "failed_checks": ["tests"],
        "ci_still_failing": True,
        "head_sha_changed": False,
        "thread_resolved": False,
        "bot_commit_at": None,
        "bot_comment_at": None,
    }


def _stub_ci_stale_scan(mod, monkeypatch, task, now_box, sends):
    monkeypatch.setattr(mod, "now_utc", lambda: now_box[0])
    monkeypatch.setattr(mod, "gh", lambda args: json.dumps([{"number": 1}]))
    monkeypatch.setattr(mod, "_collect_pr_stale_items", lambda repo, number, pr_meta: [_ci_stale_item()])
    monkeypatch.setattr(mod, "_direct_task_for_stale_item", lambda item, retries=None: task)
    monkeypatch.setattr(mod, "send", lambda to, content: sends.append((to, content)))
    monkeypatch.setattr(mod, "log", lambda message: None)


def test_done_task_does_not_suppress_and_marks_whiff(mod, monkeypatch):
    """done で終わったのに CI が赤いまま → 抑制せず「空振り」として警告継続。"""
    task = SimpleNamespace(task_id="gh-ci-o-r#1-aaaaaaaa", status="done", summary="infra failure diagnosed")
    now_box = [T0]
    sends: list[tuple[str, str]] = []
    _stub_ci_stale_scan(mod, monkeypatch, task, now_box, sends)
    state = mod.default_state()

    mod.check_unaddressed(state)

    assert len(sends) == 1
    to, body = sends[0]
    assert to == "rin"
    assert "空振り" in body
    entry = state["stale_watch"][_ci_stale_item()["item_id"]]
    assert "suppressed_by_task" not in entry


def test_in_progress_task_damps_but_never_silences(mod, monkeypatch):
    """実行中タスクありは頻度を落とす（1h floor）が、ゼロにはしない。"""
    task = SimpleNamespace(task_id="gh-ci-o-r#1-aaaaaaaa", status="in_progress", summary="fixing")
    now_box = [T0]
    sends: list[tuple[str, str]] = []
    _stub_ci_stale_scan(mod, monkeypatch, task, now_box, sends)
    state = mod.default_state()

    mod.check_unaddressed(state)  # immediate first warn
    now_box[0] = T0 + timedelta(minutes=30)
    mod.check_unaddressed(state)  # within 1h floor → damped
    warns_after_damp = len(sends)
    now_box[0] = T0 + timedelta(hours=20)
    mod.check_unaddressed(state)  # far past any decayed interval → reminded again

    assert warns_after_damp == 1
    assert len(sends) >= 2
    assert all("実行中" in body for _, body in sends)


def test_escalated_item_sends_repeating_human_judgment_dm(mod, monkeypatch):
    now_box = [T0]
    sends: list[tuple[str, str]] = []
    _stub_ci_stale_scan(mod, monkeypatch, None, now_box, sends)
    item_id = _ci_stale_item()["item_id"]
    state = mod.default_state()
    state["stale_watch"][item_id] = {
        "first_seen": mod.iso(T0 - timedelta(hours=30)),
        "last_warned": mod.iso(T0 - timedelta(hours=29)),
        "escalated_at": mod.iso(T0 - timedelta(hours=24)),
        "kind": "ci",
    }

    mod.check_unaddressed(state)
    mod.check_unaddressed(state)  # same tick → no duplicate

    human_sends = [s for s in sends if "上司判断で再開させること" in s[1]]
    assert len(human_sends) == 1
    assert human_sends[0][0] == "sakura"
    assert "まずあなたが解決策" in human_sends[0][1]
    assert "作業を止めないこと" in human_sends[0][1]
    assert state["stale_watch"][item_id]["human_notified_at"]

    # 24h後にもう一度届く（1回きりで終わらない）
    now_box[0] = T0 + timedelta(hours=24)
    mod.check_unaddressed(state)
    human_sends = [s for s in sends if "上司判断で再開させること" in s[1]]
    assert len(human_sends) == 2


def test_new_failure_type_resets_escalated_stale_entry(mod, monkeypatch):
    now_box = [T0]
    sends: list[tuple[str, str]] = []
    _stub_ci_stale_scan(mod, monkeypatch, None, now_box, sends)
    item_id = _ci_stale_item()["item_id"]
    state = mod.default_state()
    state["stale_watch"][item_id] = {
        "first_seen": mod.iso(T0 - timedelta(hours=30)),
        "last_warned": mod.iso(T0 - timedelta(hours=29)),
        "escalated_at": mod.iso(T0 - timedelta(hours=24)),
        "human_notified_at": mod.iso(T0 - timedelta(hours=1)),
        "failure_signature": "old-check",
        "kind": "ci",
    }

    mod.check_unaddressed(state)

    entry = state["stale_watch"][item_id]
    assert entry["failure_signature"] == "tests"
    assert entry["escalated_at"] is None
    assert "human_notified_at" not in entry
    assert len(sends) == 1 and sends[0][0] == "rin"
