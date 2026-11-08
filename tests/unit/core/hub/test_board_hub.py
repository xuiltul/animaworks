from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.hub.members import HubMember, token_digest
from core.messaging.board import (
    Decision,
    board_access,
    clear_post_hooks,
    hub_principal,
    post_to_board_in_shared_dir,
)
from core.messaging.messenger import ChannelMeta, save_channel_meta


@pytest.fixture
def shared_dir(tmp_path: Path) -> Path:
    shared = tmp_path / "shared"
    (shared / "channels").mkdir(parents=True)
    return shared


@pytest.fixture(autouse=True)
def _clear_hooks():
    clear_post_hooks()
    yield
    clear_post_hooks()


@pytest.fixture
def member() -> HubMember:
    return HubMember(
        id="grok",
        display_name="Grok",
        company="a",
        token_sha256=token_digest("token"),
        message_route={"type": "poll"},
    )


def _hub_config(monkeypatch: pytest.MonkeyPatch, *, enabled: bool = True, board: str = "hub") -> None:
    config = SimpleNamespace(
        hub=SimpleNamespace(
            enabled=enabled,
            board=board,
            public_hosts=[],
            max_posts_per_hour=20,
            max_round_trips=4,
        )
    )
    monkeypatch.setattr("core.config.models.load_config", lambda: config)


def _save_member(shared_dir: Path, member: HubMember) -> None:
    from core.hub.members import save_member

    save_member(shared_dir, member)


def _make_board(shared_dir: Path, name: str = "hub", *, company: str | None = "a") -> None:
    (shared_dir / "channels" / f"{name}.jsonl").touch()
    save_channel_meta(shared_dir, name, ChannelMeta(members=[], company=company))


def test_hub_principal_is_external_agent_identity(member: HubMember) -> None:
    principal = hub_principal(member)
    assert principal.kind == "hub"
    assert principal.name == "grok"
    assert principal.company == "a"


def test_hub_can_read_write_and_mention_on_its_board(
    shared_dir: Path, member: HubMember, monkeypatch: pytest.MonkeyPatch
) -> None:
    _hub_config(monkeypatch)
    _save_member(shared_dir, member)
    _make_board(shared_dir)

    principal = hub_principal(member)
    for action in ("read", "write", "mention_target"):
        assert board_access(shared_dir, "hub", principal, action).allowed


def test_hub_denies_outbound_and_other_boards(
    shared_dir: Path, member: HubMember, monkeypatch: pytest.MonkeyPatch
) -> None:
    _hub_config(monkeypatch)
    _save_member(shared_dir, member)
    _make_board(shared_dir)
    _make_board(shared_dir, "other")

    principal = hub_principal(member)
    assert board_access(shared_dir, "hub", principal, "outbound") == Decision(False, "action_not_allowed")
    assert board_access(shared_dir, "other", principal, "read") == Decision(False, "not_hub_board")


def test_hub_requires_enabled_same_company_and_open_board(
    shared_dir: Path, member: HubMember, monkeypatch: pytest.MonkeyPatch
) -> None:
    _hub_config(monkeypatch)
    _save_member(shared_dir, member)
    _make_board(shared_dir)

    assert board_access(shared_dir, "hub", hub_principal(member), "read").allowed

    _save_member(shared_dir, HubMember(**{**member.to_dict(), "enabled": False}))  # type: ignore[arg-type]
    assert board_access(shared_dir, "hub", hub_principal(member), "read") == Decision(False, "not_enabled")

    other = HubMember(**{**member.to_dict(), "company": "b"})  # type: ignore[arg-type]
    _save_member(shared_dir, other)
    assert board_access(shared_dir, "hub", hub_principal(other), "read") == Decision(False, "company_mismatch")

    save_channel_meta(shared_dir, "hub", ChannelMeta(members=[], company="b", closed=True))
    assert board_access(shared_dir, "hub", hub_principal(other), "read") == Decision(False, "closed")


def test_hub_denied_when_disabled_or_board_has_no_company(
    shared_dir: Path, member: HubMember, monkeypatch: pytest.MonkeyPatch
) -> None:
    _save_member(shared_dir, member)
    _make_board(shared_dir)
    _hub_config(monkeypatch, enabled=False)
    assert board_access(shared_dir, "hub", hub_principal(member), "read") == Decision(False, "hub_disabled")

    _hub_config(monkeypatch)
    save_channel_meta(shared_dir, "hub", ChannelMeta(members=[]))
    assert board_access(shared_dir, "hub", hub_principal(member), "read") == Decision(False, "company_mismatch")


def test_hub_post_records_hub_source_and_external_agent_kind(
    shared_dir: Path, member: HubMember, monkeypatch: pytest.MonkeyPatch
) -> None:
    _hub_config(monkeypatch)
    _save_member(shared_dir, member)
    _make_board(shared_dir)

    result = post_to_board_in_shared_dir(
        shared_dir,
        "hub",
        "hello from outside",
        hub_principal(member),
        origin="hub",
        notify=False,
    )
    assert result.committed and not result.error
    assert result.id is not None

    records = [
        json.loads(line)
        for line in (shared_dir / "channels" / "hub.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    record = records[-1]
    assert record["id"] == result.id
    assert record["source"] == "hub"
    assert record["kind"] == "external_agent"
    assert record["from"] == "grok"
