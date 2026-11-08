from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path

AVATAR_JS = Path(__file__).resolve().parents[3] / "server" / "static" / "pages" / "board" / "avatar.js"
I18N_DIR = Path(__file__).resolve().parents[3] / "server" / "static" / "i18n"

EXPECTED = {"en": "External AI", "ja": "社外 AI", "ko": "외부 AI"}


def test_avatar_labels_external_agent_as_external_ai() -> None:
    source = AVATAR_JS.read_text(encoding="utf-8")
    assert 'value === "external_agent"' in source
    assert 't("board.kind_external_agent")' in source


def test_external_agent_i18n_key_exists_in_all_locales() -> None:
    for locale, expected in EXPECTED.items():
        data = json.loads((I18N_DIR / f"{locale}.json").read_text(encoding="utf-8"))
        assert data["board.kind_external_agent"] == expected
