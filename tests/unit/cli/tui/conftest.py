# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for TUI tests.

Points ``ANIMAWORKS_TUI_DIR`` at a per-test temporary directory so the
session / keybinding files written by the app never touch the real home
directory.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _tui_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("ANIMAWORKS_TUI_DIR", str(tmp_path / "tui"))
    yield
