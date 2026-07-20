"""Unit tests for shared/users write filter via anima roster."""

from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from core.anima_roster import (
    get_anima_roster,
    invalidate_anima_roster,
    is_anima_name,
    load_anima_names,
    refresh_anima_roster,
)
from core._anima_messaging import MessagingMixin


class _Harness(MessagingMixin):
    """Minimal MessagingMixin host for unit tests."""

    def __init__(self, anima_dir: Path) -> None:
        self.anima_dir = anima_dir
        self.name = anima_dir.name


@pytest.fixture(autouse=True)
def _clear_roster_cache():
    invalidate_anima_roster()
    yield
    invalidate_anima_roster()


def _make_layout(tmp_path: Path, *, anima_names: list[str], tombstone: str | None = None) -> Path:
    data_dir = tmp_path / "data"
    animas = data_dir / "animas"
    shared = data_dir / "shared"
    animas.mkdir(parents=True)
    shared.mkdir(parents=True)
    for name in anima_names:
        (animas / name).mkdir()
        (animas / name / "identity.md").write_text(f"# {name}\n", encoding="utf-8")
    if tombstone:
        tdir = animas / tombstone
        tdir.mkdir(exist_ok=True)
        (tdir / "identity.md").write_text(f"# {tombstone}\n", encoding="utf-8")
        (tdir / "status.json").write_text(
            json.dumps({"enabled": False}, ensure_ascii=False),
            encoding="utf-8",
        )
    return data_dir


def test_load_anima_names_includes_active_and_tombstone(tmp_path: Path) -> None:
    data_dir = _make_layout(tmp_path, anima_names=["sakura"], tombstone="old_merge")
    names = load_anima_names(data_dir)
    assert "sakura" in names
    assert "old_merge" in names


def test_is_anima_name_exact_match_only(tmp_path: Path) -> None:
    data_dir = _make_layout(tmp_path, anima_names=["sakura"])
    with patch("core.anima_roster.get_data_dir", return_value=data_dir), patch(
        "core.anima_roster.get_animas_dir", return_value=data_dir / "animas"
    ):
        refresh_anima_roster()
        assert is_anima_name("sakura") is True
        assert is_anima_name("Sakura") is False  # case-sensitive exact match
        assert is_anima_name("human_alice") is False
        assert is_anima_name("") is False


def test_log_human_conversation_skips_anima_name(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    data_dir = _make_layout(tmp_path, anima_names=["sakura", "hinata"])
    harness = _Harness(data_dir / "animas" / "sakura")
    with patch("core.anima_roster.get_data_dir", return_value=data_dir), patch(
        "core.anima_roster.get_animas_dir", return_value=data_dir / "animas"
    ):
        refresh_anima_roster()
        with caplog.at_level("INFO", logger="animaworks.anima"):
            harness._log_human_conversation("hello from peer", "hinata")

    users_dir = data_dir / "shared" / "users"
    assert not (users_dir / "hinata").exists()
    assert any("Skipping shared/users conversation log" in r.message for r in caplog.records)


def test_log_human_conversation_allows_human_name(tmp_path: Path) -> None:
    data_dir = _make_layout(tmp_path, anima_names=["sakura"])
    harness = _Harness(data_dir / "animas" / "sakura")
    with patch("core.anima_roster.get_data_dir", return_value=data_dir), patch(
        "core.anima_roster.get_animas_dir", return_value=data_dir / "animas"
    ):
        refresh_anima_roster()
        harness._log_human_conversation("hello human", "alice")

    log_root = data_dir / "shared" / "users" / "alice" / "conversations"
    assert log_root.is_dir()
    files = list(log_root.glob("*.jsonl"))
    assert len(files) == 1
    line = files[0].read_text(encoding="utf-8").strip()
    record = json.loads(line)
    assert record["content"] == "hello human"
    assert record["anima"] == "sakura"


def test_log_human_conversation_skips_tombstone_name(tmp_path: Path) -> None:
    data_dir = _make_layout(tmp_path, anima_names=["sakura"], tombstone="retired_bot")
    harness = _Harness(data_dir / "animas" / "sakura")
    with patch("core.anima_roster.get_data_dir", return_value=data_dir), patch(
        "core.anima_roster.get_animas_dir", return_value=data_dir / "animas"
    ):
        refresh_anima_roster()
        harness._log_human_conversation("from tombstone", "retired_bot")

    assert not (data_dir / "shared" / "users" / "retired_bot").exists()


def test_log_human_conversation_skips_empty_from_person(tmp_path: Path) -> None:
    data_dir = _make_layout(tmp_path, anima_names=["sakura"])
    harness = _Harness(data_dir / "animas" / "sakura")
    with patch("core.anima_roster.get_data_dir", return_value=data_dir), patch(
        "core.anima_roster.get_animas_dir", return_value=data_dir / "animas"
    ):
        refresh_anima_roster()
        harness._log_human_conversation("empty sender", "")

    users_dir = data_dir / "shared" / "users"
    if users_dir.exists():
        assert list(users_dir.iterdir()) == []


def test_roster_cache_refreshes(tmp_path: Path) -> None:
    data_dir = _make_layout(tmp_path, anima_names=["sakura"])
    with patch("core.anima_roster.get_data_dir", return_value=data_dir), patch(
        "core.anima_roster.get_animas_dir", return_value=data_dir / "animas"
    ):
        roster = refresh_anima_roster()
        assert "sakura" in roster
        assert "newbot" not in roster

        (data_dir / "animas" / "newbot").mkdir()
        # stale cache until refresh
        assert "newbot" not in get_anima_roster()
        refresh_anima_roster()
        assert "newbot" in get_anima_roster()
