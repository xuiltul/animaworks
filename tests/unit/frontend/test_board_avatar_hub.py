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


def test_external_avatars_use_images_and_hub_store() -> None:
    avatar = AVATAR_JS.read_text(encoding="utf-8")
    assert 'group === "external"' in avatar
    assert "board-avatar-img" in avatar

    board_dir = AVATAR_JS.parent
    post = (board_dir / "post.js").read_text(encoding="utf-8")
    assert 'kind === "external_agent" ? ctx.avatarUrl(message.from)' in post

    store = (board_dir / "store.js").read_text(encoding="utf-8")
    assert 'api("/api/hub/members")' in store
    assert "hubMembers" in store
    assert "loadHubMembers" in store

    delivery = (board_dir / "delivery.js").read_text(encoding="utf-8")
    assert "store?.hubMember(name)" in delivery
