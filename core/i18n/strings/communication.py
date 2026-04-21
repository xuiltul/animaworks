# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Domain-specific i18n strings."""

from __future__ import annotations

STRINGS: dict[str, dict[str, str]] = {
    "slack.annotation.channel_mentioned": {
        "ja": "[slack:channel{channel} — あなたがメンションされています]",
        "en": "[slack:channel{channel} — you are mentioned]",
    },
    "slack.annotation.channel_no_mention": {
        "ja": "[slack:channel{channel} — あなたへの直接メンションはありません]",
        "en": "[slack:channel{channel} — no direct mention of you]",
    },
    "slack.annotation.external_addressee": {
        "ja": "[宛先注意: このメッセージは {names} に向けられています。あなたへの指示ではない可能性があります]",
        "en": "[Addressee notice: This message is directed at {names}. It may not be an instruction for you]",
    },
    "slack.observe_only_hint": {
        "ja": "observe: このメッセージはあなたへの直接の指示ではありません。状況把握のみに留め、返信やアクションの実行は不要です",
        "en": "observe: This message is not a direct instruction for you. Use it for situational awareness only — no reply or action required",
    },
    "inbox.reply_placeholder": {
        "ja": "{返信内容}",
        "en": "{reply_content}",
    },
    "messenger.depth_exceeded": {
        "ja": "ConversationDepthExceeded: {to}との会話が10分間に6ターンに達しました。次のハートビートサイクルまでお待ちください",
        "en": (
            "ConversationDepthExceeded: Conversation with {to} reached 6 turns in 10 minutes. Please wait until the next heartbeat cycle."
        ),
    },
    "messenger.more_count": {
        "ja": "(+{count}件)",
        "en": "(+{count} more)",
    },
    "messenger.read_receipt": {
        "ja": "[既読通知] {count}件のメッセージを受信しました: {summary}",
        "en": "[Read receipt] Received {count} messages: {summary}",
    },
}
