"""Board read-access filtering for priming Channel B and the prompt channel list."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.memory.priming.channel_b import read_shared_channels
from core.prompt.messaging import _collect_accessible_board_channels


def _make_anima(data_dir: Path, name: str, status: dict[str, object]) -> Path:
    anima_dir = data_dir / "animas" / name
    anima_dir.mkdir(parents=True)
    (anima_dir / "status.json").write_text(json.dumps(status), encoding="utf-8")
    return anima_dir


def _write_meta(shared_dir: Path, channel: str, meta: dict[str, object] | str) -> None:
    path = shared_dir / "channels" / f"{channel}.meta.json"
    content = meta if isinstance(meta, str) else json.dumps(meta)
    path.write_text(content, encoding="utf-8")


def _write_channel(shared_dir: Path, channel: str, text: str) -> None:
    channels_dir = shared_dir / "channels"
    channels_dir.mkdir(parents=True, exist_ok=True)
    (channels_dir / f"{channel}.jsonl").write_text(
        json.dumps(
            {
                "ts": "2026-07-20T00:00:00+09:00",
                "from": "bob",
                "text": text,
                "source": "anima",
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _build_channels(data_dir: Path, shared_dir: Path) -> None:
    _make_anima(data_dir, "bob", {"company": "beta"})
    _write_channel(shared_dir, "same", "same-company")
    _write_meta(shared_dir, "same", {"members": [], "company": "alpha"})
    _write_channel(shared_dir, "other", "other-company")
    _write_meta(shared_dir, "other", {"members": [], "company": "beta"})
    _write_channel(shared_dir, "restricted", "not-a-member")
    _write_meta(shared_dir, "restricted", {"members": ["bob"], "company": "alpha"})
    _write_channel(shared_dir, "closed", "closed-channel")
    _write_meta(shared_dir, "closed", {"members": ["alice"], "company": "alpha", "closed": True})
    _write_channel(shared_dir, "broken", "unreadable-meta")
    _write_meta(shared_dir, "broken", "{not json")
    _write_channel(shared_dir, "legacy", "legacy-open")


def _shared(tmp_path: Path) -> Path:
    shared_dir = tmp_path / "shared"
    shared_dir.mkdir()
    return shared_dir


def test_priming_limits_channels_to_board_read_access(tmp_path: Path) -> None:
    anima_dir = _make_anima(tmp_path, "alice", {"company": "alpha"})
    shared_dir = _shared(tmp_path)
    _build_channels(tmp_path, shared_dir)

    result = read_shared_channels(anima_dir, shared_dir)

    assert {entry.channel for entry in result} == {"same"}


def test_priming_allows_board_scope_for_unassigned_anima(tmp_path: Path) -> None:
    anima_dir = _make_anima(tmp_path, "alice", {})
    shared_dir = _shared(tmp_path)
    _build_channels(tmp_path, shared_dir)

    result = read_shared_channels(anima_dir, shared_dir)

    assert {entry.channel for entry in result} == {"same", "other", "legacy"}


def test_priming_denied_roots_branch_uses_board_access(tmp_path: Path) -> None:
    anima_dir = _make_anima(tmp_path, "alice", {"company": "alpha"})
    shared_dir = _shared(tmp_path)
    _build_channels(tmp_path, shared_dir)
    denied = tmp_path / "private"
    denied.mkdir()

    result = read_shared_channels(anima_dir, shared_dir, denied_roots=(denied,))

    assert {entry.channel for entry in result} == {"same"}


def test_prompt_channel_list_limits_channels_to_board_read_access(tmp_path: Path) -> None:
    anima_dir = _make_anima(tmp_path, "alice", {"company": "alpha"})
    shared_dir = _shared(tmp_path)
    _build_channels(tmp_path, shared_dir)

    restricted, open_channels = _collect_accessible_board_channels(anima_dir)

    assert restricted == []
    assert open_channels == ["same"]


def test_prompt_channel_list_allows_board_scope_for_unassigned_anima(tmp_path: Path) -> None:
    anima_dir = _make_anima(tmp_path, "alice", {})
    shared_dir = _shared(tmp_path)
    _build_channels(tmp_path, shared_dir)

    restricted, open_channels = _collect_accessible_board_channels(anima_dir)

    assert restricted == []
    assert open_channels == ["legacy", "other", "same"]


@pytest.mark.asyncio
async def test_legacy_channel_fallback_uses_board_access(tmp_path: Path) -> None:
    from core.memory.priming.channel_b import read_old_channels

    anima_dir = _make_anima(tmp_path, "alice", {"company": "alpha"})
    shared_dir = _shared(tmp_path)
    _write_channel(shared_dir, "general", "general-content")
    _write_meta(shared_dir, "general", {"members": [], "company": "alpha"})
    _write_channel(shared_dir, "ops", "ops-content")
    _write_meta(shared_dir, "ops", {"members": [], "company": "beta"})

    result = await read_old_channels(anima_dir, shared_dir)

    assert "general-content" in result
    assert "ops-content" not in result
