from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DELIVERY_JS = ROOT / "server" / "static" / "pages" / "board" / "delivery.js"
I18N_DIR = ROOT / "server" / "static" / "i18n"

EXPECTED = {
    "ja": "読みに来たら届く",
    "en": "Delivered when they next poll",
    "ko": "다음에 읽으러 오면 전달됩니다",
}


def test_delivery_js_translates_poll_reason() -> None:
    source = DELIVERY_JS.read_text(encoding="utf-8")
    assert 'reason === "polls"' in source
    assert 't("board.delivery_reason_polls")' in source


def test_poll_reason_i18n_key_exists_in_all_locales() -> None:
    for locale, expected in EXPECTED.items():
        data = json.loads((I18N_DIR / f"{locale}.json").read_text(encoding="utf-8"))
        assert data["board.delivery_reason_polls"] == expected
