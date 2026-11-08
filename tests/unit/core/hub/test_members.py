from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path

import pytest

from core.hub.members import (
    HubMember,
    find_member_by_token,
    last_seen,
    load_members,
    member_path,
    save_member,
    token_digest,
    touch_last_seen,
    validate_member,
)


@pytest.fixture
def shared_dir(tmp_path: Path) -> Path:
    shared = tmp_path / "shared"
    shared.mkdir()
    return shared


def _member(**overrides: object) -> HubMember:
    data: dict[str, object] = {
        "id": "grok",
        "display_name": "Grok",
        "kind": "vendor_bot",
        "company": "a",
        "enabled": True,
        "token_sha256": token_digest("secret-token"),
        "message_route": {"type": "poll"},
        "expect_every": "24h",
    }
    data.update(overrides)
    return HubMember(**data)  # type: ignore[arg-type]


def test_save_and_load_roundtrip(shared_dir: Path) -> None:
    save_member(shared_dir, _member())

    path = member_path(shared_dir, "grok")
    assert path.is_file()
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["token_sha256"] == token_digest("secret-token")
    assert "secret-token" not in path.read_text(encoding="utf-8")

    loaded = load_members(shared_dir)
    assert loaded["grok"] == _member()


def test_find_member_by_token_only_enabled_members(shared_dir: Path) -> None:
    save_member(shared_dir, _member())

    assert find_member_by_token(shared_dir, "secret-token") == _member()
    assert find_member_by_token(shared_dir, "wrong-token") is None
    assert find_member_by_token(shared_dir, "") is None

    disabled = _member(enabled=False)
    save_member(shared_dir, disabled)
    assert find_member_by_token(shared_dir, "secret-token") is None


def test_load_skips_invalid_reserved_and_anima_members(shared_dir: Path) -> None:
    (shared_dir.parent / "animas" / "alice").mkdir(parents=True)
    save_member(shared_dir, _member(id="grok"))
    save_member(shared_dir, _member(id="alice"))
    save_member(shared_dir, _member(id="human"))
    (shared_dir / "hub" / "members" / "broken.json").write_text("not json", encoding="utf-8")
    (shared_dir / "hub" / "members" / "mismatch.json").write_text(
        json.dumps(_member(id="other").to_dict()),
        encoding="utf-8",
    )

    assert set(load_members(shared_dir)) == {"grok"}


def test_validate_member_rules() -> None:
    validate_member(_member())
    with pytest.raises(ValueError):
        validate_member(_member(id="Bad ID"))
    with pytest.raises(ValueError):
        validate_member(_member(company=""))
    with pytest.raises(ValueError):
        validate_member(_member(expect_every="0m"))
    with pytest.raises(ValueError):
        validate_member(_member(token_sha256="ABC"))
    with pytest.raises(ValueError):
        validate_member(_member(message_route={"type": "carrier-pigeon"}))
    with pytest.raises(ValueError):
        validate_member(_member(message_route={"type": "webhook", "url_ref": "x"}))


def test_webhook_route_roundtrip(shared_dir: Path) -> None:
    member = _member(message_route={"type": "webhook", "url_ref": "hub.url", "key_ref": "hub.key"})
    save_member(shared_dir, member)
    assert load_members(shared_dir)["grok"].message_route == {
        "type": "webhook",
        "url_ref": "hub.url",
        "key_ref": "hub.key",
    }


def test_touch_last_seen_records_timestamp(shared_dir: Path) -> None:
    assert last_seen(shared_dir, "grok") is None
    touch_last_seen(shared_dir, "grok")
    recorded = last_seen(shared_dir, "grok")
    assert recorded is not None
    assert "T" in recorded
