from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Discord integration i18n strings."""

STRINGS: dict[str, dict[str, str]] = {
    "discord.annotation_mentioned": {
        "ja": "あなたがメンションされています",
        "en": "you are mentioned",
    },
    "discord.annotation_no_mention": {
        "ja": "あなたへの直接メンションはありません",
        "en": "no direct mention for you",
    },
    "discord.auto_reply_instruction": {
        "ja": (
            "あなたの最終回答はDiscordに自動投稿されます。"
            "discord_channel_postを絶対に呼ばないでください。呼ぶと二重投稿になります"
        ),
        "en": (
            "Your final response will be auto-posted to Discord. "
            "Do NOT call discord_channel_post — it will cause duplicate messages"
        ),
    },
}
