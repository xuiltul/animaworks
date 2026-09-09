# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

from cli.tui.app import AnimaChatApp
from cli.tui.commands import _cmd_model, iter_commands
from cli.tui.session import SessionInfo, load_session, new_session, save_session

CATALOG = [
    {
        "id": "c:codex/gpt-5.6-sol",
        "label": "GPT-5.6-Sol",
        "credential": "codex",
        "mode": "c",
        "model": "codex/gpt-5.6-sol",
        "group": "Codex",
        "note": "説明文",
        "source": "codex-cli",
    },
    {
        "id": "g:grok-4.5",
        "label": "Grok-4.5",
        "mode": "g",
        "model": "grok-4.5",
        "group": "Grok",
        "note": "",
    },
]


class FakePalette:
    def __init__(self):
        self.is_open = False
        self.items = []

    def close(self):
        self.is_open = False

    def set_items(self, items):
        self.is_open = True
        self.items = items


class FakeStatusBar:
    def __init__(self):
        self.model = None

    def set_model(self, model):
        self.model = model


class FakeClient:
    base_url = "http://localhost:18500"
    from_person = "human"

    async def list_available_models(self, *, refresh=False):
        return list(CATALOG)


def _make_app() -> AnimaChatApp:
    session = new_session(anima="sora")
    session.session_id = "test-session"
    app = AnimaChatApp(FakeClient(), "sora", session=session)
    app.palette = FakePalette()
    app.status_bar = FakeStatusBar()
    app.show_transient = lambda text: None
    return app


def test_model_registered_with_takes_args():
    cmds = {c.name: c for c in iter_commands()}
    assert "model" in cmds
    assert cmds["model"].takes_args is True


def test_cmd_model_forwards_to_choose_model():
    calls = []

    class StubApp:
        def choose_model(self, args):
            calls.append(args)

    _cmd_model([], StubApp())
    _cmd_model(["grok-4.5"], StubApp())
    assert calls == [[], ["grok-4.5"]]


def test_session_model_roundtrip(tmp_path):
    s = new_session(anima="sora")
    s.model = "c:codex/gpt-5.6-sol"
    save_session(s, base_dir=tmp_path)
    loaded = load_session(s.session_id, base_dir=tmp_path)
    assert loaded is not None
    assert loaded.model == "c:codex/gpt-5.6-sol"


def test_from_dict_without_model_defaults_empty(tmp_path):
    data = {
        "session_id": "x1",
        "created_at": "c",
        "updated_at": "u",
        "anima": "sora",
    }
    info = SessionInfo.from_dict(data)
    assert info.model == ""


def test_choose_model_default_resets_chat_model():
    app = _make_app()
    app.chat_model = "c:codex/gpt-5.6-sol"
    app.choose_model(["default"])
    assert app.chat_model == ""


def test_choose_model_resolves_from_catalog():
    app = _make_app()
    app._models = list(CATALOG)
    app.choose_model(["gpt-5.6-sol"])
    assert app.chat_model == "c:codex/gpt-5.6-sol"


def test_choose_model_unknown_string_set_as_is():
    app = _make_app()
    app._models = list(CATALOG)
    app.choose_model(["custom-model"])
    assert app.chat_model == "custom-model"


def test_set_chat_model_updates_session_and_writes(tmp_path):
    app = _make_app()
    app.session_dir = tmp_path
    app.set_chat_model("c:codex/gpt-5.6-sol")
    assert app.session.model == "c:codex/gpt-5.6-sol"
    saved = json.loads((tmp_path / f"{app.session.session_id}.json").read_text(encoding="utf-8"))
    assert saved["model"] == "c:codex/gpt-5.6-sol"
    assert app.status_bar.model == "c:codex/gpt-5.6-sol"
