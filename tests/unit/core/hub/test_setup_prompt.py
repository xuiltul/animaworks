from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.hub.members import HubMember, save_member, token_digest
from core.hub.setup_prompt import SetupPromptError, build_setup_prompt
from core.messaging.messenger import ChannelMeta, save_channel_meta


def _config(
    monkeypatch: pytest.MonkeyPatch,
    *,
    public_hosts: list[str] | None = None,
    locale: str = "ja",
) -> None:
    config = SimpleNamespace(
        locale=locale,
        hub=SimpleNamespace(
            enabled=True,
            company="a",
            board="hub",
            public_hosts=public_hosts or [],
            max_posts_per_hour=12,
            max_round_trips=3,
        ),
    )
    monkeypatch.setattr("core.config.models.load_config", lambda: config)
    monkeypatch.setattr("core.hub.setup_prompt.load_config", lambda: config)


def _shared(tmp_path: Path) -> Path:
    shared = tmp_path / "shared"
    (shared / "channels").mkdir(parents=True)
    (shared / "channels" / "hub.jsonl").touch()
    save_channel_meta(shared, "hub", ChannelMeta(members=[], company="a"))
    return shared


def _anima(
    shared: Path,
    name: str,
    *,
    company: str = "a",
    enabled: bool = True,
    supervisor: str | None = None,
    role: str = "staff",
    speciality: str | None = None,
) -> None:
    anima_dir = shared.parent / "animas" / name
    anima_dir.mkdir(parents=True)
    status: dict[str, object] = {"enabled": enabled, "company": company, "role": role}
    if supervisor:
        status["supervisor"] = supervisor
    if speciality:
        status["speciality"] = speciality
    (anima_dir / "status.json").write_text(json.dumps(status), encoding="utf-8")


def _member(member_id: str = "muse", **overrides: object) -> HubMember:
    data: dict[str, object] = {
        "id": member_id,
        "display_name": "Muse",
        "company": "a",
        "token_sha256": token_digest("top-secret-token"),
        "message_route": {"type": "poll"},
        "expect_every": "30m",
    }
    data.update(overrides)
    return HubMember(**data)  # type: ignore[arg-type]


def test_setup_prompt_uses_roster_route_and_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _config(monkeypatch, public_hosts=["hub.example.com"])
    shared = _shared(tmp_path)
    _anima(shared, "alice", supervisor="bob", speciality="研究")
    _anima(shared, "bob")
    _anima(shared, "carol", enabled=False)
    _anima(shared, "dave", company="b")
    save_member(shared, _member())

    text = build_setup_prompt(shared, "muse")

    assert "https://hub.example.com/hub/mcp" in text
    assert "muse" in text
    assert "Muse" in text
    assert "alice" in text and "研究" in text
    assert "bob" in text
    assert "carol" not in text
    assert "dave" not in text
    assert "30分ごと" in text
    assert "定期タスク" in text
    assert "12件" in text
    assert "4000字" in text
    assert "3回" in text
    assert "top-secret-token" not in text
    assert token_digest("top-secret-token") not in text


def test_setup_prompt_prefers_supervisor_less_first_contact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _config(monkeypatch, public_hosts=["hub.example.com"])
    shared = _shared(tmp_path)
    _anima(shared, "alice", supervisor="bob")
    _anima(shared, "bob")
    save_member(shared, _member())

    text = build_setup_prompt(shared, "muse")

    assert "まず @bob" in text


def test_setup_prompt_webhook_route_shows_refs_not_secrets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _config(monkeypatch, public_hosts=["hub.example.com"])
    shared = _shared(tmp_path)
    save_member(
        shared,
        _member(message_route={"type": "webhook", "url_ref": "hub.hook", "key_ref": "hub.key"}),
    )

    text = build_setup_prompt(shared, "muse")

    assert "Webhook" in text
    assert "hub.hook" in text and "hub.key" in text
    assert "30分ごと" not in text
    assert "https://hooks.example.com" not in text


def test_setup_prompt_url_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _config(monkeypatch)
    shared = _shared(tmp_path)
    save_member(shared, _member())

    text = build_setup_prompt(shared, "muse", url="https://custom.example.com/hub/mcp")

    assert "https://custom.example.com/hub/mcp" in text


def test_setup_prompt_requires_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _config(monkeypatch)
    shared = _shared(tmp_path)
    save_member(shared, _member())

    with pytest.raises(SetupPromptError, match="--url"):
        build_setup_prompt(shared, "muse")


def test_setup_prompt_rejects_unknown_and_disabled_members(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _config(monkeypatch, public_hosts=["hub.example.com"])
    shared = _shared(tmp_path)
    save_member(shared, _member())

    with pytest.raises(SetupPromptError, match="unknown"):
        build_setup_prompt(shared, "unknown")

    save_member(shared, _member(enabled=False))
    with pytest.raises(SetupPromptError, match="無効"):
        build_setup_prompt(shared, "muse")


def test_setup_prompt_omits_anima_without_mention_target_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _config(monkeypatch, public_hosts=["hub.example.com"])
    shared = _shared(tmp_path)
    _anima(shared, "alice")
    _anima(shared, "eve", company="b")
    save_channel_meta(shared, "hub", ChannelMeta(members=["alice", "eve"], company="a"))
    save_member(shared, _member())

    text = build_setup_prompt(shared, "muse")
    assert "alice" not in text
    assert "まず @human" in text


def test_setup_prompt_falls_back_to_human_when_roster_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _config(monkeypatch, public_hosts=["hub.example.com"])
    shared = _shared(tmp_path)
    save_member(shared, _member())

    text = build_setup_prompt(shared, "muse")

    assert "まず @human" in text
    assert "該当する anima はいません" in text


def test_setup_prompt_uses_the_configured_locale(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _config(monkeypatch, public_hosts=["hub.example.com"], locale="en")
    shared = tmp_path / "shared"
    save_member(shared, _member())

    text = build_setup_prompt(shared, "muse")
    assert "あなたは AnimaWorks のハブ" not in text
    assert "AnimaWorks" in text
    assert "https://hub.example.com/hub/mcp" in text
    assert "{" not in text.replace("{{", "")
