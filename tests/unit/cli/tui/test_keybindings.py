# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from cli.tui.keybindings import (
    APP_BINDING_IDS,
    DEFAULT_KEYBINDINGS,
    app_keymap,
    load_keybindings,
    merge_keybindings,
)


def test_defaults_have_all_keys():
    assert set(DEFAULT_KEYBINDINGS) >= {
        "send",
        "newline",
        "interrupt",
        "toggle_sidebar",
        "toggle_thinking",
        "quit",
        "focus_input",
        "scroll_up",
        "scroll_down",
    }


def test_merge_keeps_defaults():
    merged, warns = merge_keybindings(DEFAULT_KEYBINDINGS, {"quit": "ctrl+q"})
    assert merged["quit"] == "ctrl+q"
    assert merged["interrupt"] == "escape"  # default
    assert warns == []


def test_merge_unknown_key_warns():
    merged, warns = merge_keybindings(DEFAULT_KEYBINDINGS, {"mystery": "x"})
    assert "mystery" not in merged
    assert any("mystery" in w for w in warns)


def test_merge_invalid_value_falls_back():
    merged, warns = merge_keybindings(DEFAULT_KEYBINDINGS, {"quit": ""})
    assert merged["quit"] == DEFAULT_KEYBINDINGS["quit"]
    assert any("quit" in w for w in warns)


def test_load_missing_file_returns_defaults():
    keymap, warns = load_keybindings(Path("/nonexistent/keys.json"))
    assert keymap == DEFAULT_KEYBINDINGS
    assert warns == []


def test_load_broken_file_returns_defaults(tmp_path):
    p = tmp_path / "keybindings.json"
    p.write_text("not json", encoding="utf-8")
    keymap, warns = load_keybindings(p)
    assert keymap == DEFAULT_KEYBINDINGS
    assert len(warns) == 1


def test_load_valid_file(tmp_path):
    p = tmp_path / "keybindings.json"
    p.write_text('{"quit": "ctrl+q", "interrupt": "ctrl+\\u5e2f"}', encoding="utf-8")
    keymap, warns = load_keybindings(p)
    assert keymap["quit"] == "ctrl+q"
    assert warns == []


def test_app_keymap_filters_to_binding_ids():
    keymap = dict(DEFAULT_KEYBINDINGS)
    keymap["send"] = "tab"
    out = app_keymap(keymap)
    assert "send" not in out
    assert set(out) <= APP_BINDING_IDS
