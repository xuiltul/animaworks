# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Domain-specific i18n strings (handler part 1)."""

from __future__ import annotations

STRINGS: dict[str, dict[str, str]] = {
    "handler.activity_dm_history": {
        "ja": "DM履歴を確認",
        "en": "Checked DM history",
    },
    "handler.activity_recent_items": {
        "ja": "最新{limit}件を確認",
        "en": "Checked latest {limit} items",
    },
    "handler.all_descendants": {
        "ja": "全配下",
        "en": "All descendants",
    },
    "handler.already_disabled": {
        "ja": "'{target_name}' は既に休止中です",
        "en": "'{target_name}' is already disabled",
    },
    "handler.already_enabled": {
        "ja": "'{target_name}' は既に有効です",
        "en": "'{target_name}' is already enabled",
    },
    "handler.anima_not_found": {
        "ja": "Anima '{target_name}' は存在しません",
        "en": "Anima '{target_name}' does not exist",
    },
    "handler.audit_activity_counts": {
        "ja": (
            "  受信: {msg_recv} | 応答: {resp_sent} | DM送信: {dm_sent} | ツール: {tool_use} | HB: {hb} | Cron: {cron} | エラー: {errors}"
        ),
        "en": (
            "  Received: {msg_recv} | Responses: {resp_sent} | DM sent: {dm_sent} | Tools: {tool_use} | HB: {hb} | Cron: {cron} | Errors: {errors}"
        ),
    },
    "handler.audit_label_cron": {
        "ja": "Cron: {task_name}",
        "en": "Cron: {task_name}",
    },
    "handler.audit_label_dm": {
        "ja": "→ {peer}",
        "en": "→ {peer}",
    },
    "handler.audit_label_error": {
        "ja": "エラー (phase: {phase})",
        "en": "Error (phase: {phase})",
    },
    "handler.audit_label_heartbeat": {
        "ja": "ハートビート完了",
        "en": "Heartbeat completed",
    },
    "handler.audit_label_reflection": {
        "ja": "振り返り",
        "en": "Reflection",
    },
    "handler.audit_label_resolved": {
        "ja": "Issue解決",
        "en": "Issue resolved",
    },
    "handler.audit_label_response": {
        "ja": "応答",
        "en": "Response",
    },
    "handler.audit_label_task_done": {
        "ja": "タスク完了",
        "en": "Task completed",
    },
    "handler.audit_label_tool": {
        "ja": "ツール: {tool}",
        "en": "Tool: {tool}",
    },
    "handler.audit_log_summary": {
        "ja": "{target_name}の監査レポート生成（{hours}h）",
        "en": "Generated audit report for {target_name} ({hours}h)",
    },
    "handler.audit_merged_footer": {
        "ja": (
            "─── 統計: 全{count}名 | 活動{total}件 | ツール{tools} | HB{hb} | 応答{resp_sent} | DM{dm_sent} | エラー{errors} ───"
        ),
        "en": (
            "─── Stats: {count} animas | {total} events | Tools {tools} | HB {hb} | Responses {resp_sent} | DM {dm_sent} | Errors {errors} ───"
        ),
    },
    "handler.audit_merged_title": {
        "ja": "═══ 組織タイムライン (直近{hours}h) — {count}名 ═══",
        "en": "═══ Org Timeline (last {hours}h) — {count} animas ═══",
    },
    "handler.audit_merged_title_since": {
        "ja": "═══ 組織タイムライン ({since}〜) — {count}名 ═══",
        "en": "═══ Org Timeline (since {since}) — {count} animas ═══",
    },
    "handler.audit_merged_tool_header": {
        "ja": "■ ツール使用サマリー",
        "en": "■ Tool Usage Summary",
    },
    "handler.audit_no_activity": {
        "ja": "(この期間の活動ログはありません)",
        "en": "(No activity log for this period)",
    },
    "handler.audit_report_footer": {
        "ja": ("─── 統計: 活動{total}件 | ツール{tools} | HB{hb} | 応答{resp_sent} | DM{dm_sent} | エラー{errors} ───"),
        "en": (
            "─── Stats: {total} events | Tools {tools} | HB {hb} | Responses {resp_sent} | DM {dm_sent} | Errors {errors} ───"
        ),
    },
    "handler.audit_report_title": {
        "ja": "═══ {name} — 行動レポート (直近{hours}h) ═══",
        "en": "═══ {name} — Activity Report (last {hours}h) ═══",
    },
    "handler.audit_report_title_since": {
        "ja": "═══ {name} — 行動レポート ({since}〜) ═══",
        "en": "═══ {name} — Activity Report (since {since}) ═══",
    },
    "handler.audit_report_truncated": {
        "ja": "  ... 他{remaining}件省略",
        "en": "  ... {remaining} more entries omitted",
    },
    "handler.audit_section_actions": {
        "ja": "■ コミュニケーション・タスク",
        "en": "■ Communication & Tasks",
    },
    "handler.audit_section_activity": {
        "ja": "■ アクティビティ",
        "en": "■ Activity",
    },
    "handler.audit_section_comms": {
        "ja": "■ 通信先",
        "en": "■ Communications",
    },
    "handler.audit_section_errors": {
        "ja": "■ エラー詳細",
        "en": "■ Error Details",
    },
    "handler.audit_section_errors_report": {
        "ja": "■ エラー",
        "en": "■ Errors",
    },
    "handler.audit_section_responses": {
        "ja": "■ 対話・応答",
        "en": "■ Dialogue & Responses",
    },
    "handler.audit_section_tasks": {
        "ja": "■ タスク",
        "en": "■ Tasks",
    },
    "handler.audit_section_thinking": {
        "ja": "■ 思考・判断（ハートビート / 振り返り）",
        "en": "■ Thinking & Decisions (Heartbeat / Reflection)",
    },
    "handler.audit_section_tool_summary": {
        "ja": "■ ツール使用サマリー（全{count}回）",
        "en": "■ Tool Usage Summary ({count} total)",
    },
    "handler.audit_status_line": {
        "ja": "状態: {status} | モデル: {model}",
        "en": "Status: {status} | Model: {model}",
    },
    "handler.audit_summary_title": {
        "ja": "═══ {name} — 監査サマリー (直近{hours}h) ═══",
        "en": "═══ {name} — Audit Summary (last {hours}h) ═══",
    },
    "handler.audit_summary_title_since": {
        "ja": "═══ {name} — 監査サマリー ({since}〜) ═══",
        "en": "═══ {name} — Audit Summary (since {since}) ═══",
    },
    "handler.audit_task_counts": {
        "ja": "  保留中: {pending} | 進行中: {in_progress} | 完了: {done} | 滞留(>30min): {stale}",
        "en": ("  Pending: {pending} | In progress: {in_progress} | Done: {done} | Stale(>30min): {stale}"),
    },
    "handler.audit_tool_line": {
        "ja": "{name} (全{total}回): ",
        "en": "{name} ({total} total): ",
    },
    "handler.background_task_started": {
        "ja": "タスクをバックグラウンドで実行開始しました (task_id: {task_id})",
        "en": "Task started in background (task_id: {task_id})",
    },
    "handler.bg_cmd_started": {
        "ja": "コマンドをバックグラウンドで実行開始しました。進捗は出力ファイルをReadで確認できます。",
        "en": "Command started in background. Read the output file to check progress.",
    },
    "handler.bg_invalid_status": {
        "ja": "Error: 無効なステータス: {status}。有効値: running, completed, failed, pending",
        "en": ("Error: Invalid status: {status}. Valid values: running, completed, failed, pending"),
    },
    "handler.bg_model_change_log": {
        "ja": "{target_name}のbackground_modelを{model}に変更",
        "en": "Changing {target_name}'s background_model to {model}",
    },
    "handler.bg_model_changed": {
        "ja": "{target_name}のbackground_modelを'{model}'に変更しました。反映にはrestart_subordinateが必要です。",
        "en": ("Changed {target_name}'s background_model to '{model}'. Call restart_subordinate to apply."),
    },
    "handler.bg_model_cleared": {
        "ja": "{target_name}のbackground_modelをクリアしました（メインモデルを使用）。",
        "en": "Cleared {target_name}'s background_model (will use main model).",
    },
    "handler.bg_not_enabled": {
        "ja": "Error: バックグラウンドタスク機能が無効です",
        "en": "Error: Background task feature is not enabled",
    },
    "handler.bg_task_id_required": {
        "ja": "Error: task_id は必須です",
        "en": "Error: task_id is required",
    },
    "handler.bg_task_not_found": {
        "ja": "Error: タスク {task_id} が見つかりません",
        "en": "Error: Task {task_id} not found",
    },
    "handler.board_mention_content": {
        "ja": (
            '{from_name}さんがボード #{channel} であなたをメンションしました:\n\n{text}\n\n返信するには post_channel(channel="{channel}", text="返信内容") を使ってください。'
        ),
        "en": (
            '{from_name} mentioned you on board #{channel}:\n\n{text}\n\nTo reply, use post_channel(channel="{channel}", text="your reply").'
        ),
    },
    "handler.body_param_required": {
        "ja": "`body` パラメータは必須です。",
        "en": "The `body` parameter is required.",
    },
    "handler.channel_acl_denied": {
        "ja": (
            'Error: #{channel} へのアクセス権がありません。manage_channel(action="info", channel="{channel}") でメンバーを確認してください。'
        ),
        "en": (
            'Error: You do not have access to #{channel}. Use manage_channel(action="info", channel="{channel}") to check members.'
        ),
    },
    "handler.channel_acl_not_member": {
        "ja": "Error: #{channel} のメンバーではないため、メンバー管理操作はできません。",
        "en": "Error: You are not a member of #{channel} and cannot manage its membership.",
    },
    "handler.channel_add_member_open_denied": {
        "ja": (
            'Error: #{channel} はオープンチャネルです。add_memberするには、まず manage_channel(action="create") で制限チャネルとして再作成してください。'
        ),
        "en": (
            'Error: #{channel} is an open channel. To add members, first recreate it as a restricted channel with manage_channel(action="create").'
        ),
    },
    "handler.channel_already_exists": {
        "ja": "Error: チャネル #{channel} は既に存在します",
        "en": "Error: Channel #{channel} already exists",
    },
    "handler.channel_created": {
        "ja": "チャネル #{channel} を作成しました（メンバー: {members}）",
        "en": "Channel #{channel} created (members: {members})",
    },
    "handler.channel_members_added": {
        "ja": "#{channel} にメンバーを追加しました: {members}",
        "en": "Added members to #{channel}: {members}",
    },
    "handler.channel_members_removed": {
        "ja": "#{channel} からメンバーを削除しました: {members}",
        "en": "Removed members from #{channel}: {members}",
    },
    "handler.channel_not_found": {
        "ja": "Error: チャネル #{channel} が見つかりません",
        "en": "Error: Channel #{channel} not found",
    },
    "handler.channel_open": {
        "ja": "#{channel} はオープンチャネルです（全Animaがアクセス可能）",
        "en": "#{channel} is an open channel (all Animas can access)",
    },
    "handler.cmd_denied": {
        "ja": "{cmd} 禁止",
        "en": "{cmd} blocked",
    },
    "handler.config_load_failed": {
        "ja": "設定読み込みに失敗: {e}",
        "en": "Failed to load config: {e}",
    },
    "handler.current_state_none": {
        "ja": "なし",
        "en": "None",
    },
    "handler.current_state_unreadable": {
        "ja": "読取不可",
        "en": "Unreadable",
    },
    "handler.dashboard_last": {
        "ja": "最終: {activity}",
        "en": "Last: {activity}",
    },
    "handler.dashboard_summary": {
        "ja": "配下{count}名のダッシュボード表示",
        "en": "Dashboard for {count} subordinate(s)",
    },
    "handler.dashboard_tasks": {
        "ja": "タスク: {count}件",
        "en": "Tasks: {count}",
    },
    "handler.dashboard_working_on": {
        "ja": "作業中: {task}",
        "en": "Working on: {task}",
    },
    "handler.delegate_log": {
        "ja": "{target_name}にタスク委譲: {summary}",
        "en": "Delegated task to {target_name}: {summary}",
    },
    "handler.delegated_success": {
        "ja": (
            "タスクを {target_name} に委譲しました。\n- 部下側タスクID: {sub_id}\n- 追跡用タスクID: {own_id}\n- {dm_result}"
        ),
        "en": (
            "Task delegated to {target_name}.\n- Subordinate task ID: {sub_id}\n- Tracking task ID: {own_id}\n- {dm_result}"
        ),
    },
    "handler.delegation_dm_content": {
        "ja": ("[タスク委譲]\n{instruction}\n\n期限: {deadline}\nタスクID: {task_id}"),
        "en": ("[Task delegation]\n{instruction}\n\nDeadline: {deadline}\nTask ID: {task_id}"),
    },
    "handler.delegation_intent_deprecated": {
        "ja": (
            "Error: intent='delegation' は廃止されました。タスクを委任するには delegate_task ツールを使用してください。send_message は report / question のみ対応しています。"
        ),
        "en": (
            "Error: intent='delegation' has been deprecated. Use the delegate_task tool to assign tasks to subordinates. send_message only supports 'report' and 'question' intents."
        ),
    },
    "handler.delegation_summary": {
        "ja": "[委譲] {summary}",
        "en": "[Delegated] {summary}",
    },
    "handler.descendant_activity": {
        "ja": "配下のactivity_log",
        "en": "Descendant activity_log",
    },
    "handler.descendant_pending": {
        "ja": "配下のstate/pending/",
        "en": "Descendant state/pending/",
    },
    "handler.descendant_state": {
        "ja": "配下のstatus.json, identity.md, injection.md, state/, task_queue.jsonl",
        "en": "Descendant status.json, identity.md, injection.md, state/, task_queue.jsonl",
    },
    "handler.description_field_required": {
        "ja": "`description` フィールドが必要です。",
        "en": "The `description` field is required.",
        "zh": "需要 `description` 字段。",
        "ko": "`description` 필드가 필요합니다.",
    },
    "handler.description_keyword_warning": {
        "ja": "descriptionに「」キーワードがありません。自動マッチング精度が低下する可能性があります。",
        "en": "No 「」keywords in description. Auto-matching accuracy may be reduced.",
        "zh": "描述中缺少「」关键词。自动匹配精度可能会降低。",
        "ko": "설명에 「」 키워드가 없습니다. 자동 매칭 정확도가 떨어질 수 있습니다.",
    },
    "handler.description_param_required": {
        "ja": "`description` パラメータは必須です。",
        "en": "The `description` parameter is required.",
    },
    "handler.disable_log_summary": {
        "ja": "{target_name} を休止",
        "en": "Disabling {target_name}",
    },
    "handler.disable_reason": {
        "ja": " (理由: {reason})",
        "en": " (reason: {reason})",
    },
    "handler.disabled_success": {
        "ja": "'{target_name}' を休止にしました。Reconciliation が30秒以内にプロセスを停止します。",
        "en": ("'{target_name}' has been disabled. Reconciliation will stop the process within 30 seconds."),
    },
    "handler.dm_already_sent": {
        "ja": "Error: このrunで既に {to} にメッセージを送信済みです。追加の連絡はBoardを使用してください。",
        "en": ("Error: Message already sent to {to} in this run. Use Board for additional communication."),
    },
    "handler.dm_intent_error": {
        "ja": (
            "Error: DMのintentは 'report', 'question' のみ許可されています。acknowledgment・感謝・FYIはBoardを使用してください（post_channel ツール）。"
        ),
        "en": (
            "Error: DM intent must be 'report' or 'question' only. Use Board (post_channel tool) for acknowledgments, thanks, or FYI."
        ),
    },
    "handler.dm_max_recipients": {
        "ja": (
            "Error: 1回のrunでDMを送れるのは最大{limit}人までです。{limit}人以上への伝達はBoardを使用してください（post_channel ツール）。"
        ),
        "en": ("Error: Maximum {limit} DM recipients per run. Use Board (post_channel tool) for {limit}+ recipients."),
    },
    "handler.dm_send_failed": {
        "ja": "DM送信失敗: {e}",
        "en": "DM send failed: {e}",
    },
    "handler.dm_sent": {
        "ja": "DM送信済み",
        "en": "DM sent",
    },
    "handler.meeting_tool_blocked": {
        "ja": "会議中は {tool} を使用できません。会議内で @メンション を使って発言してください。",
        "en": "Tool '{tool}' is not available during meetings. Use @mentions to communicate within the meeting.",
    },
}
