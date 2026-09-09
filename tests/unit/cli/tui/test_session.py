# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

from cli.tui.session import (
    MAX_SESSIONS,
    SessionInfo,
    latest_session,
    list_sessions,
    load_session,
    new_session,
    save_session,
    session_dir_path,
    tui_base_dir,
)


def _make(anima="sora", **kw) -> SessionInfo:
    s = new_session(anima=anima, gateway_url="http://localhost:18500")
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def test_new_session_has_unique_ids(monkeypatch):
    a = new_session(anima="sora")
    b = new_session(anima="sora")
    assert a.session_id != b.session_id
    assert a.in_flight is False
    assert a.recent_animas == ["sora"]


def test_save_and_load_roundtrip(tmp_path):
    s = _make()
    path = save_session(s, base_dir=tmp_path)
    assert path.exists()
    loaded = load_session(s.session_id, base_dir=tmp_path)
    assert loaded is not None
    assert loaded.anima == s.anima
    assert loaded.session_id == s.session_id


def test_list_sessions_orders_by_updated_at_desc(tmp_path):
    s1 = _make(anima="a", updated_at="2026-01-01T00:00:00+00:00")
    s2 = _make(anima="b", updated_at="2026-02-01T00:00:00+00:00")
    s3 = _make(anima="c", updated_at="2026-03-01T00:00:00+00:00")
    save_session(s1, base_dir=tmp_path)
    save_session(s2, base_dir=tmp_path)
    save_session(s3, base_dir=tmp_path)
    infos = list_sessions(base_dir=tmp_path)
    assert [i.anima for i in infos] == ["c", "b", "a"]


def test_latest_session(tmp_path):
    s1 = _make(anima="a")
    s2 = _make(anima="b")
    save_session(s1, base_dir=tmp_path)
    t = s2.updated_at
    s2.updated_at = "2099-01-01T00:00:00+00:00"
    save_session(s2, base_dir=tmp_path)
    latest = latest_session(base_dir=tmp_path)
    assert latest is not None and latest.anima == "b"
    s2.updated_at = t


def test_rotation_keeps_max(tmp_path):
    for i in range(MAX_SESSIONS + 10):
        s = _make(anima=f"a{i}")
        save_session(s, base_dir=tmp_path)
    infos = list_sessions(base_dir=tmp_path)
    assert len(infos) == MAX_SESSIONS


def test_broken_json_is_ignored(tmp_path):
    s = _make()
    save_session(s, base_dir=tmp_path)
    bad = tmp_path / "broken.json"
    bad.write_text("{ not json", encoding="utf-8")
    infos = list_sessions(base_dir=tmp_path)
    assert all(i.session_id == s.session_id for i in infos)


def test_missing_session_returns_none(tmp_path):
    assert load_session("nope", base_dir=tmp_path) is None


def test_unknown_fields_tolerated(tmp_path):
    data = {
        "session_id": "x1",
        "created_at": "c",
        "updated_at": "u",
        "anima": "sora",
        "bogus_field": "ignored",
    }
    (tmp_path / "x1.json").write_text(json.dumps(data), encoding="utf-8")
    loaded = load_session("x1", base_dir=tmp_path)
    assert loaded is not None
    assert loaded.anima == "sora"


def test_respects_anima_works_tui_dir(monkeypatch, tmp_path):
    d = tmp_path / "custom" / "tui"
    monkeypatch.setenv("ANIMAWORKS_TUI_DIR", str(d))
    assert tui_base_dir() == d
    sdir = session_dir_path()
    assert sdir == d / "sessions"
    assert sdir.exists()
