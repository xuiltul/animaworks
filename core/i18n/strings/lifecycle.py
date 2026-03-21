# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Domain-specific i18n strings."""

from __future__ import annotations

STRINGS: dict[str, dict[str, str]] = {
    "cascade.activity_read_failed": {
        "ja": "GlobalOutboundLimitExceeded: アクティビティログ読み取り失敗のため送信をブロックしました",
        "en": ("GlobalOutboundLimitExceeded: Sending blocked because the activity log could not be read"),
    },
    "cascade.daily_limit": {
        "ja": (
            "GlobalOutboundLimitExceeded: 24時間あたりの送信上限（{max_per_day}通）に到達しています（現在{daily_count}通/24h）。 このターンではsend_messageを使わず、送信内容をcurrent_state.mdに記録して次のセッションで送信してください。"
        ),
        "en": (
            "GlobalOutboundLimitExceeded: Daily send limit ({max_per_day} messages) reached ({daily_count} msgs/24h). Do not use send_message this turn. Record the message content in current_state.md and send it in the next session."
        ),
    },
    "cascade.hourly_limit": {
        "ja": (
            "GlobalOutboundLimitExceeded: 1時間あたりの送信上限（{max_per_hour}通）に到達しています（現在{hourly_count}通/1h, {daily_count}通/24h）。{reset_at} このターンではsend_messageを使わず、送信内容をcurrent_state.mdに記録して次のセッションで送信してください。"
        ),
        "en": (
            "GlobalOutboundLimitExceeded: Hourly send limit ({max_per_hour} messages) reached ({hourly_count} msgs/1h, {daily_count} msgs/24h).{reset_at} Do not use send_message this turn. Record the message content in current_state.md and send it in the next session."
        ),
    },
    "cascade.hourly_reset_at": {
        "ja": " 次の送信可能時刻（目安）: {reset_time}",
        "en": " Estimated next send time: {reset_time}",
    },
    "heartbeat.current_state_cleanup_required": {
        "ja": (
            "⚠ **current_state.md 圧縮が必要です**（現在 {current_chars} 文字 / 上限 {max_chars} 文字）\n\n**このHBの最初のアクション**として以下を実行してください:\n1. 解決済み・完了済みのタスクをすべて削除する\n2. 進行中・保留・待ち状態のタスクのみ残す\n3. {max_chars} 文字以内に圧縮して書き込む\n4. その後、通常のHBチェックリストを実行する"
        ),
        "en": (
            "⚠ **current_state.md cleanup required** (current: {current_chars} chars / limit: {max_chars} chars)\n\n**As the first action of this heartbeat**, do the following:\n1. Delete all resolved/completed tasks\n2. Keep only in-progress, pending, and waiting tasks\n3. Compress to under {max_chars} chars and save\n4. Then proceed with the normal heartbeat checklist"
        ),
    },
    "heartbeat.history_plan_entry": {
        "ja": "- {ts}: [計画] {plan}",
        "en": "- {ts}: [Plan] {plan}",
    },
    "scheduler.cron_fallback_description": {
        "ja": "cron.mdの「{task_name}」の指示に従って処理してください。",
        "en": "Follow the instructions for '{task_name}' in cron.md.",
    },
}
