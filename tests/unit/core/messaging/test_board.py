from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.messaging.board import (
    Decision,
    Principal,
    anima_principal,
    board_access,
    clear_post_hooks,
    external_principal,
    human_principal,
    post_to_board,
    register_post_hook,
)
from core.messaging.messenger import ChannelMeta, save_channel_meta


@pytest.fixture
def shared_dir(tmp_path: Path) -> Path:
    shared = tmp_path / "shared"
    (shared / "channels").mkdir(parents=True)
    for name in ("alice", "bob", "charlie"):
        _write_status(tmp_path / "animas" / name)
    return shared


@pytest.fixture(autouse=True)
def _clear_hooks():
    clear_post_hooks()
    yield
    clear_post_hooks()


def _write_status(anima_dir: Path, *, company: str | None = None, enabled: object = True) -> None:
    anima_dir.mkdir(parents=True, exist_ok=True)
    status: dict[str, object] = {"enabled": enabled}
    if company is not None:
        status["company"] = company
    (anima_dir / "status.json").write_text(json.dumps(status), encoding="utf-8")


def _write_channel(shared_dir: Path, name: str = "general") -> Path:
    path = shared_dir / "channels" / f"{name}.jsonl"
    path.touch()
    return path


def _mapping_config(monkeypatch: pytest.MonkeyPatch) -> None:
    config = SimpleNamespace(
        external_messaging=SimpleNamespace(
            slack=SimpleNamespace(board_mapping={"C1": "general"}),
            discord=SimpleNamespace(board_mapping={"D1": "ops"}),
        )
    )
    monkeypatch.setattr("core.config.models.load_config", lambda: config)


def test_meta_unreadable_denies_every_principal(shared_dir: Path) -> None:
    path = shared_dir / "channels" / "general.meta.json"
    path.write_text("not json", encoding="utf-8")
    _write_channel(shared_dir)

    principals = [
        Principal("anima", "alice", company=None),
        human_principal(),
        Principal("operator", "operator"),
        external_principal("slack"),
    ]
    for principal in principals:
        for action in ("read", "write", "mention_target", "outbound"):
            assert board_access(shared_dir, "general", principal, action) == Decision(False, "meta_unreadable")


@pytest.mark.parametrize("kind", ["human", "operator"])
def test_human_and_operator_can_access_closed_board_for_read_only(shared_dir: Path, kind: str) -> None:
    save_channel_meta(shared_dir, "general", ChannelMeta(members=[], closed=True))
    principal = Principal(kind, kind)

    assert board_access(shared_dir, "general", principal, "read").allowed
    for action in ("write", "mention_target", "outbound"):
        assert board_access(shared_dir, "general", principal, action) == Decision(False, "closed")


def test_closed_board_denies_anima_and_external(shared_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _mapping_config(monkeypatch)
    save_channel_meta(shared_dir, "general", ChannelMeta(members=[], closed=True))

    assert board_access(shared_dir, "general", Principal("anima", "alice", company=None), "read").reason == "closed"
    assert board_access(shared_dir, "general", external_principal("slack"), "write").reason == "closed"


def test_membership_and_company_boundary_rules(shared_dir: Path) -> None:
    _write_status(shared_dir.parent / "animas" / "alice", company="north")
    _write_status(shared_dir.parent / "animas" / "bob", company="north")
    _write_status(shared_dir.parent / "animas" / "charlie", company="south")
    save_channel_meta(shared_dir, "general", ChannelMeta(members=["alice", "bob"]))
    _write_channel(shared_dir)

    assert board_access(
        shared_dir, "general", anima_principal("alice", animas_dir=shared_dir.parent / "animas"), "write"
    ).allowed
    assert board_access(
        shared_dir, "general", anima_principal("charlie", animas_dir=shared_dir.parent / "animas"), "write"
    ) == Decision(False, "not_member")
    save_channel_meta(shared_dir, "general", ChannelMeta(members=["alice", "charlie"]))
    assert board_access(
        shared_dir, "general", anima_principal("alice", animas_dir=shared_dir.parent / "animas"), "write"
    ) == Decision(False, "cross_company")


def test_company_scoped_board_allows_same_and_unassigned_but_denies_mismatch(shared_dir: Path) -> None:
    save_channel_meta(shared_dir, "general", ChannelMeta(members=[], company="north"))
    _write_channel(shared_dir)
    animas_dir = shared_dir.parent / "animas"
    _write_status(animas_dir / "alice", company="north")
    _write_status(animas_dir / "bob", company="south")

    assert board_access(shared_dir, "general", anima_principal("alice", animas_dir=animas_dir), "write").allowed
    assert board_access(shared_dir, "general", anima_principal("bob", animas_dir=animas_dir), "write") == Decision(
        False, "company_mismatch"
    )
    assert board_access(shared_dir, "general", anima_principal("charlie", animas_dir=animas_dir), "write").allowed
    (animas_dir / "alice" / "status.json").unlink()
    assert board_access(shared_dir, "general", anima_principal("alice", animas_dir=animas_dir), "write") == Decision(
        False, "company_unreadable"
    )


def test_legacy_board_denies_assigned_or_unreadable_anima(shared_dir: Path) -> None:
    _write_channel(shared_dir)
    animas_dir = shared_dir.parent / "animas"
    _write_status(animas_dir / "alice", company="north")

    assert board_access(shared_dir, "general", anima_principal("bob", animas_dir=animas_dir), "write").allowed
    assert board_access(shared_dir, "general", anima_principal("alice", animas_dir=animas_dir), "write") == Decision(
        False, "company_assigned_legacy"
    )
    (animas_dir / "alice" / "status.json").unlink()
    assert board_access(shared_dir, "general", anima_principal("alice", animas_dir=animas_dir), "write") == Decision(
        False, "company_unreadable"
    )


def test_external_write_requires_board_mapping(shared_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_channel(shared_dir)
    _mapping_config(monkeypatch)
    external = external_principal("slack-user", external_kind="bot")

    assert board_access(shared_dir, "general", external, "write").allowed
    assert board_access(shared_dir, "ops", external, "write").allowed
    assert board_access(shared_dir, "unknown", external, "write") == Decision(False, "not_mapped")
    assert board_access(shared_dir, "general", external, "read") == Decision(False, "action_not_allowed")


def test_post_commits_uniform_row_and_records_activity_only_for_anima(shared_dir: Path) -> None:
    animas_dir = shared_dir.parent / "animas"
    alice = anima_principal("alice", animas_dir=animas_dir)
    _write_channel(shared_dir)

    result = post_to_board("general", "hello", alice, origin="internal", shared_dir=shared_dir)
    row = json.loads((shared_dir / "channels" / "general.jsonl").read_text(encoding="utf-8"))

    assert result.committed
    assert set(row) == {"ts", "from", "text", "source", "id", "kind"}
    assert row["from"] == "alice"
    assert row["source"] == "anima"
    assert row["kind"] == "anima"
    assert row["id"] == result.id
    activity = list((animas_dir / "alice" / "activity_log").glob("*.jsonl"))
    assert len(activity) == 1
    assert json.loads(activity[0].read_text(encoding="utf-8").strip())["type"] == "channel_post"

    human_result = post_to_board("general", "human note", human_principal(), origin="http", shared_dir=shared_dir)
    assert human_result.committed
    assert len(activity) == 1


def test_request_id_deduplicates_last_posts_and_skips_hooks(shared_dir: Path) -> None:
    _write_channel(shared_dir)
    principal = human_principal()
    called: list[str] = []
    register_post_hook("capture", lambda post: called.append(post.id) or "sent")

    first = post_to_board("general", "same", principal, request_id="req-1", origin="http", shared_dir=shared_dir)
    second = post_to_board("general", "same", principal, request_id="req-1", origin="http", shared_dir=shared_dir)
    rows = (shared_dir / "channels" / "general.jsonl").read_text(encoding="utf-8").splitlines()

    assert first.committed and not first.duplicate
    assert not second.committed and second.duplicate and second.id == first.id
    assert len(rows) == 1
    assert called == [first.id]


def test_post_hooks_are_best_effort_and_report_status(shared_dir: Path) -> None:
    _write_channel(shared_dir)
    register_post_hook("sent", lambda post: "sent")
    register_post_hook("skipped", lambda post: "skipped")

    def broken(post):
        raise RuntimeError("hook error")

    register_post_hook("broken", broken)
    result = post_to_board("general", "hello", human_principal(), origin="http", shared_dir=shared_dir)

    assert result.committed
    assert result.hooks == {"sent": "sent", "skipped": "skipped", "broken": "failed"}


@pytest.mark.parametrize(
    ("external_kind", "sender_name"),
    [("user", "alice"), ("bot", "ops-bot")],
)
def test_external_posts_notify_even_without_running_socket(
    shared_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    external_kind: str,
    sender_name: str,
) -> None:
    _mapping_config(monkeypatch)
    _write_channel(shared_dir)

    result = post_to_board(
        "general",
        "Please check @alice",
        external_principal(sender_name, external_kind=external_kind),
        request_id="slack:1",
        origin="slack",
        shared_dir=shared_dir,
    )

    inbox = list((shared_dir / "inbox" / "alice").glob("*.json"))
    assert result.notified == ["alice"]
    assert result.not_notified == {}
    assert len(inbox) == 1
    message = json.loads(inbox[0].read_text(encoding="utf-8"))
    assert message["from_person"] == sender_name
    assert message["type"] == "board_mention"
    assert message["meta"] == {"board_channel": "general", "board_post_id": result.id}
    assert f"[board_reply:channel=general,from={sender_name}]" in message["content"]
    row = json.loads((shared_dir / "channels" / "general.jsonl").read_text(encoding="utf-8"))
    assert row["kind"] == f"external_{external_kind}"
    assert row["source"] == "slack"


def test_at_all_records_disabled_and_unreadable_recipients(shared_dir: Path) -> None:
    _write_channel(shared_dir)
    _write_status(shared_dir.parent / "animas" / "bob", enabled=False)
    (shared_dir.parent / "animas" / "charlie" / "status.json").write_text("bad", encoding="utf-8")

    result = post_to_board("general", "Hello @all", human_principal(), origin="http", shared_dir=shared_dir)

    assert result.notified == ["alice"]
    assert result.not_notified == {"bob": "disabled", "charlie": "status_unreadable"}


def test_at_all_skips_recipients_without_explicit_enabled_true(shared_dir: Path) -> None:
    _write_channel(shared_dir)
    _write_status(shared_dir.parent / "animas" / "alice", enabled=True)
    bob_status = shared_dir.parent / "animas" / "bob" / "status.json"
    bob_status.parent.mkdir(parents=True, exist_ok=True)
    bob_status.write_text('{"other": true}', encoding="utf-8")

    result = post_to_board("general", "Hello @all", human_principal(), origin="http", shared_dir=shared_dir)

    assert result.notified == ["alice", "charlie"]
    assert result.not_notified == {"bob": "disabled"}


def test_suppressed_mirror_does_not_send_mentions(shared_dir: Path) -> None:
    _write_channel(shared_dir)
    result = post_to_board(
        "general", "@all", human_principal("guest"), origin="mirror", notify=False, shared_dir=shared_dir
    )

    assert result.committed
    assert result.notified == []
    assert result.not_notified == {}
    assert not (shared_dir / "inbox").exists()


def test_post_rejections_do_not_append_or_log_activity(shared_dir: Path) -> None:
    _write_channel(shared_dir)
    save_channel_meta(shared_dir, "general", ChannelMeta(members=["bob"]))
    principal = anima_principal("alice", animas_dir=shared_dir.parent / "animas")

    denied = post_to_board("general", "no", principal, origin="internal", shared_dir=shared_dir)
    missing = post_to_board("missing", "no", principal, origin="internal", shared_dir=shared_dir)
    invalid = post_to_board("../bad", "no", principal, origin="internal", shared_dir=shared_dir)

    assert denied.error == "denied:not_member"
    assert missing.error == "not_found"
    assert invalid.error == "invalid_name"
    assert (shared_dir / "channels" / "general.jsonl").read_text(encoding="utf-8") == ""
    assert not (shared_dir.parent / "animas" / "alice" / "activity_log").exists()


def test_board_module_has_no_external_or_server_imports() -> None:
    source = Path(__file__).resolve().parents[4] / "core" / "messaging" / "board.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    forbidden = ("server", "slack", "discord")
    imported: list[str] = []
    dynamic_imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name.lower() for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append((node.module or "").lower())
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "import_module" and node.args and isinstance(node.args[0], ast.Constant):
                if isinstance(node.args[0].value, str):
                    dynamic_imports.append(node.args[0].value.lower())
    assert not any(any(term in item for term in forbidden) for item in imported + dynamic_imports)
