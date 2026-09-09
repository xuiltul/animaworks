# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Keybinding configuration for the TUI.

Keybindings live in ``$ANIMAWORKS_TUI_DIR/keybindings.json`` (default
``~/.animaworks/tui/keybindings.json``). Loading is a pure function that
merges the defaults with any custom file, ignoring unknown keys and
falling back to the default value on invalid entries.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cli.tui.session import tui_base_dir

DEFAULT_KEYBINDINGS: dict[str, str] = {
    "send": "enter",
    "newline": "shift+enter",
    # Enter copies while text is selected and submits otherwise, the way
    # tmux copy-mode ends a selection.
    "copy_selection": "enter",
    "interrupt": "escape",
    "toggle_sidebar": "ctrl+b",
    "toggle_thinking": "ctrl+t",
    "quit": "ctrl+d",
    "focus_input": "ctrl+l",
    "scroll_up": "pageup",
    "scroll_down": "pagedown",
}

# Subset of config keys that map onto App-level Textual bindings (each must
# have a matching ``Binding(..., id=...)`` in ``AnimaChatApp.BINDINGS``).
APP_BINDING_IDS = {
    "copy_selection",
    "interrupt",
    "toggle_sidebar",
    "toggle_thinking",
    "quit",
    "focus_input",
    "scroll_up",
    "scroll_down",
}


def merge_keybindings(defaults: dict[str, str], custom: dict[str, Any]) -> tuple[dict[str, str], list[str]]:
    """Merge ``custom`` into ``defaults``.

    Returns ``(merged, warnings)``. Unknown keys and invalid values are
    skipped (keeping the default) and reported via ``warnings``.
    """
    merged = dict(defaults)
    warnings: list[str] = []
    if not isinstance(custom, dict):
        return merged, ["keybindings: expected an object, using defaults"]
    for key, value in custom.items():
        if key not in defaults:
            warnings.append(f"keybindings: unknown key '{key}' ignored")
            continue
        if not isinstance(value, str) or not value.strip():
            warnings.append(f"keybindings: invalid value for '{key}', using default")
            continue
        merged[key] = value.strip()
    return merged, warnings


def load_keybindings(path: Path | None = None) -> tuple[dict[str, str], list[str]]:
    """Load keybindings from ``path`` (default: the user config file).

    Returns ``(keymap, warnings)``. A missing or unreadable file yields
    the defaults with no warnings.
    """
    if path is None:
        path = tui_base_dir() / "keybindings.json"
    if not path.exists():
        return dict(DEFAULT_KEYBINDINGS), []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return dict(DEFAULT_KEYBINDINGS), ["keybindings: unreadable file, using defaults"]
    return merge_keybindings(DEFAULT_KEYBINDINGS, data)


def app_keymap(keymap: dict[str, str]) -> dict[str, str]:
    """Reduce a full keymap to the App-level bindings for Textual."""
    return {k: v for k, v in keymap.items() if k in APP_BINDING_IDS}
