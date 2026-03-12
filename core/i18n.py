# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Lightweight i18n support for runtime strings."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_STRINGS: dict[str, dict[str, str]] = {
    # ── handler.py (tooling/handler) ──
    "handler.episode_filename_warning": {
        "ja": "WARNING: エピソードファイル名 '{filename}' は標準パターン (YYYY-MM-DD.md または YYYY-MM-DD_suffix.md) に合致しません。 推奨: episodes/{date}.md に '## HH:MM — タイトル' 形式で追記してください。",
        "en": "WARNING: Episode filename '{filename}' does not match the standard pattern (YYYY-MM-DD.md or YYYY-MM-DD_suffix.md). Recommended: append to episodes/{date}.md in '## HH:MM — Title' format.",
    },
    "handler.skill_frontmatter_required": {
        "ja": "スキルファイルにはYAMLフロントマター(---)が必要です。",
        "en": "Skill files require YAML frontmatter (---).",
    },
    "handler.name_field_required": {"ja": "`name` フィールドが必要です。", "en": "The `name` field is required."},
    "handler.description_field_required": {
        "ja": "`description` フィールドが必要です。",
        "en": "The `description` field is required.",
    },
    "handler.description_keyword_warning": {
        "ja": "descriptionに「」キーワードがありません。自動マッチング精度が低下する可能性があります。",
        "en": "No 「」keywords in description. Auto-matching accuracy may be reduced.",
    },
    "handler.legacy_skill_sections": {
        "ja": "旧形式のセクション(## 概要 / ## 発動条件)が検出されました。Claude Code形式(YAMLフロントマター)への移行を推奨します。",
        "en": "Legacy sections (## 概要 / ## 発動条件) detected. Migration to Claude Code format (YAML frontmatter) is recommended.",
    },
    "handler.procedure_frontmatter_recommended": {
        "ja": "手順書ファイルにはYAMLフロントマター(---)を推奨します。description フィールドで自動マッチングが有効になります。",
        "en": "Procedure files should have YAML frontmatter (---). The description field enables auto-matching.",
    },
    "handler.procedure_frontmatter_recommended_short": {
        "ja": "手順書ファイルにはYAMLフロントマター(---)を推奨します。",
        "en": "Procedure files should have YAML frontmatter (---).",
    },
    "handler.procedure_description_missing": {
        "ja": "`description` フィールドがありません。自動マッチングを有効にするために description を追加してください。",
        "en": "The `description` field is missing. Add description to enable auto-matching.",
    },
    "handler.background_task_started": {
        "ja": "タスクをバックグラウンドで実行開始しました (task_id: {task_id})",
        "en": "Task started in background (task_id: {task_id})",
    },
    "handler.bg_task_id_required": {"ja": "Error: task_id は必須です", "en": "Error: task_id is required"},
    "handler.bg_not_enabled": {
        "ja": "Error: バックグラウンドタスク機能が無効です",
        "en": "Error: Background task feature is not enabled",
    },
    "handler.bg_task_not_found": {
        "ja": "Error: タスク {task_id} が見つかりません",
        "en": "Error: Task {task_id} not found",
    },
    "handler.bg_invalid_status": {
        "ja": "Error: 無効なステータス: {status}。有効値: running, completed, failed, pending",
        "en": "Error: Invalid status: {status}. Valid values: running, completed, failed, pending",
    },
    "handler.output_truncated": {
        "ja": "[出力が500KBを超えたためトランケーションしました。元のサイズ: {size}]",
        "en": "[Output truncated because it exceeded 500KB. Original size: {size}]",
    },
    "handler.activity_recent_items": {"ja": "最新{limit}件を確認", "en": "Checked latest {limit} items"},
    "handler.activity_dm_history": {"ja": "DM履歴を確認", "en": "Checked DM history"},
    "handler.tool_creation_denied": {
        "ja": "ツール作成が許可されていません。permissions.md に「ツール作成」セクションを追加してください。",
        "en": "Tool creation is not permitted. Add a tool creation section to permissions.md.",
    },
    "tooling.gated_action_denied": {
        "ja": "アクション '{action}' (ツール '{tool}') は明示的な許可が必要です。permissions.md に '{tool}_{action}: yes' を追加してください。",
        "en": "Action '{action}' on tool '{tool}' requires explicit permission. Add '{tool}_{action}: yes' to permissions.md.",
    },
    "handler.skill_format_validation": {
        "ja": "⚠️ スキルフォーマット検証:\n{msg}",
        "en": "⚠️ Skill format validation:\n{msg}",
    },
    "handler.procedure_format_validation": {
        "ja": "⚠️ 手順書フォーマット検証:\n{msg}",
        "en": "⚠️ Procedure format validation:\n{msg}",
    },
    "handler.delegation_intent_deprecated": {
        "ja": "Error: intent='delegation' は廃止されました。タスクを委任するには delegate_task ツールを使用してください。send_message は report / question のみ対応しています。",
        "en": "Error: intent='delegation' has been deprecated. Use the delegate_task tool to assign tasks to subordinates. send_message only supports 'report' and 'question' intents.",
    },
    "handler.dm_intent_error": {
        "ja": "Error: DMのintentは 'report', 'question' のみ許可されています。acknowledgment・感謝・FYIはBoardを使用してください（post_channel ツール）。",
        "en": "Error: DM intent must be 'report' or 'question' only. Use Board (post_channel tool) for acknowledgments, thanks, or FYI.",
    },
    "handler.dm_already_sent": {
        "ja": "Error: このrunで既に {to} にメッセージを送信済みです。追加の連絡はBoardを使用してください。",
        "en": "Error: Message already sent to {to} in this run. Use Board for additional communication.",
    },
    "handler.dm_max_recipients": {
        "ja": "Error: 1回のrunでDMを送れるのは最大{limit}人までです。{limit}人以上への伝達はBoardを使用してください（post_channel ツール）。",
        "en": "Error: Maximum {limit} DM recipients per run. Use Board (post_channel tool) for {limit}+ recipients.",
    },
    "handler.post_alt_hint": {
        "ja": " 別のチャネル（{channels}）への投稿、またはsend_message（intent: question/report）は可能です。",
        "en": " You can post to another channel ({channels}) or use send_message (intent: question/report).",
    },
    "handler.post_already_posted": {
        "ja": "Error: このrunで既に #{channel} に投稿済みです。同一チャネルへの連投はできません。{alt_hint}",
        "en": "Error: Already posted to #{channel} in this run. No duplicate posts to the same channel.{alt_hint}",
    },
    "handler.post_cooldown": {
        "ja": "Error: #{channel} には {ts} に投稿済みです（{elapsed}秒前）。クールダウン {cooldown}秒が必要です。",
        "en": "Error: Already posted to #{channel} at {ts} ({elapsed}s ago). Cooldown of {cooldown}s required.",
    },
    "handler.board_mention_content": {
        "ja": '{from_name}さんがボード #{channel} であなたをメンションしました:\n\n{text}\n\n返信するには post_channel(channel="{channel}", text="返信内容") を使ってください。',
        "en": '{from_name} mentioned you on board #{channel}:\n\n{text}\n\nTo reply, use post_channel(channel="{channel}", text="your reply").',
    },
    "handler.channel_acl_denied": {
        "ja": 'Error: #{channel} へのアクセス権がありません。manage_channel(action="info", channel="{channel}") でメンバーを確認してください。',
        "en": 'Error: You do not have access to #{channel}. Use manage_channel(action="info", channel="{channel}") to check members.',
    },
    "handler.channel_created": {
        "ja": "チャネル #{channel} を作成しました（メンバー: {members}）",
        "en": "Channel #{channel} created (members: {members})",
    },
    "handler.channel_already_exists": {
        "ja": "Error: チャネル #{channel} は既に存在します",
        "en": "Error: Channel #{channel} already exists",
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
    "handler.channel_acl_not_member": {
        "ja": "Error: #{channel} のメンバーではないため、メンバー管理操作はできません。",
        "en": "Error: You are not a member of #{channel} and cannot manage its membership.",
    },
    "handler.channel_add_member_open_denied": {
        "ja": 'Error: #{channel} はオープンチャネルです。add_memberするには、まず manage_channel(action="create") で制限チャネルとして再作成してください。',
        "en": 'Error: #{channel} is an open channel. To add members, first recreate it as a restricted channel with manage_channel(action="create").',
    },
    "handler.self_operation_denied": {
        "ja": "自分自身を操作することはできません",
        "en": "You cannot operate on yourself",
    },
    "handler.config_load_failed": {"ja": "設定読み込みに失敗: {e}", "en": "Failed to load config: {e}"},
    "handler.anima_not_found": {
        "ja": "Anima '{target_name}' は存在しません",
        "en": "Anima '{target_name}' does not exist",
    },
    "handler.not_direct_subordinate": {
        "ja": "'{target_name}' はあなたの直属部下ではありません",
        "en": "'{target_name}' is not your direct subordinate",
    },
    "handler.not_descendant": {
        "ja": "'{target_name}' はあなたの配下ではありません",
        "en": "'{target_name}' is not under your supervision",
    },
    "handler.already_disabled": {"ja": "'{target_name}' は既に休止中です", "en": "'{target_name}' is already disabled"},
    "handler.disable_log_summary": {"ja": "{target_name} を休止", "en": "Disabling {target_name}"},
    "handler.disable_reason": {"ja": " (理由: {reason})", "en": " (reason: {reason})"},
    "handler.disabled_success": {
        "ja": "'{target_name}' を休止にしました。Reconciliation が30秒以内にプロセスを停止します。",
        "en": "'{target_name}' has been disabled. Reconciliation will stop the process within 30 seconds.",
    },
    "handler.already_enabled": {"ja": "'{target_name}' は既に有効です", "en": "'{target_name}' is already enabled"},
    "handler.enable_log_summary": {"ja": "{target_name} を復帰", "en": "Enabling {target_name}"},
    "handler.enabled_success": {
        "ja": "'{target_name}' を有効にしました。Reconciliation が30秒以内にプロセスを起動します。",
        "en": "'{target_name}' has been enabled. Reconciliation will start the process within 30 seconds.",
    },
    "handler.model_warning": {
        "ja": "警告: '{model}' は既知のモデルカタログに含まれていません。正しいモデル名か確認してください。",
        "en": "Warning: '{model}' is not in the known model catalog. Please verify the model name.",
    },
    "handler.model_change_log": {
        "ja": "{target_name} のモデルを {model} に変更",
        "en": "Changing {target_name}'s model to {model}",
    },
    "handler.model_changed": {
        "ja": "'{target_name}' のモデルを '{model}' に変更しました。反映するには restart_subordinate を呼び出してください。",
        "en": "Changed {target_name}'s model to '{model}'. Call restart_subordinate to apply.",
    },
    "handler.restart_log": {"ja": "{target_name} を再起動リクエスト", "en": "Restart requested for {target_name}"},
    "handler.restart_success": {
        "ja": "'{target_name}' の再起動をリクエストしました。Reconciliation が 30 秒以内にプロセスを再起動します。",
        "en": "Restart requested for '{target_name}'. Reconciliation will restart the process within 30 seconds.",
    },
    "handler.no_subordinates": {"ja": "配下の Anima はいません", "en": "No subordinate Animas"},
    "handler.last_activity_none": {"ja": "なし", "en": "None"},
    "handler.last_activity_unknown": {"ja": "不明", "en": "Unknown"},
    "handler.current_task_none": {"ja": "なし", "en": "None"},
    "handler.current_task_unreadable": {"ja": "読取不可", "en": "Unreadable"},
    "handler.org_dashboard_title": {"ja": "## 組織ダッシュボード", "en": "## Organization Dashboard"},
    "handler.dashboard_last": {"ja": "最終: {activity}", "en": "Last: {activity}"},
    "handler.dashboard_tasks": {"ja": "タスク: {count}件", "en": "Tasks: {count}"},
    "handler.dashboard_working_on": {"ja": "作業中: {task}", "en": "Working on: {task}"},
    "handler.dashboard_summary": {
        "ja": "配下{count}名のダッシュボード表示",
        "en": "Dashboard for {count} subordinate(s)",
    },
    "handler.since_minutes": {"ja": "{minutes}分前", "en": "{minutes}m ago"},
    "handler.since_hours": {"ja": "{hours}時間{minutes}分前", "en": "{hours}h {minutes}m ago"},
    "handler.ping_summary": {"ja": "{target}の生存確認", "en": "Ping {target}"},
    "handler.state_title": {"ja": "## {target_name} の作業状態", "en": "## {target_name}'s work status"},
    "handler.state_current_task": {"ja": "### 進行中タスク", "en": "### Current task"},
    "handler.state_pending": {"ja": "### 保留タスク", "en": "### Pending tasks"},
    "handler.state_none": {"ja": "(なし)", "en": "(none)"},
    "handler.state_unreadable": {"ja": "(読取不可)", "en": "(unreadable)"},
    "handler.state_read_summary": {"ja": "{target_name}の作業状態を読み取り", "en": "Read {target_name}'s work status"},
    "handler.file_read_own": {"ja": "自分のディレクトリ", "en": "Own directory"},
    "handler.file_read_shared": {"ja": "shared/", "en": "shared/"},
    "handler.file_write_own": {"ja": "自分のディレクトリ", "en": "Own directory"},
    "handler.subordinate_management": {
        "ja": "直属部下のcron.md, heartbeat.md, status.json, injection.md",
        "en": "Direct subordinate's cron.md, heartbeat.md, status.json, injection.md",
    },
    "handler.subordinate_dir_list": {"ja": "直属部下のディレクトリ一覧", "en": "Direct subordinate directory listing"},
    "handler.descendant_activity": {"ja": "配下のactivity_log", "en": "Descendant activity_log"},
    "handler.descendant_state": {
        "ja": "配下のstatus.json, identity.md, injection.md, state/, task_queue.jsonl",
        "en": "Descendant status.json, identity.md, injection.md, state/, task_queue.jsonl",
    },
    "handler.descendant_pending": {"ja": "配下のstate/pending/", "en": "Descendant state/pending/"},
    "handler.peer_activity": {"ja": "同僚のactivity_log（読み取り専用）", "en": "Peer activity_log (read-only)"},
    "handler.cmd_denied": {"ja": "{cmd} 禁止", "en": "{cmd} blocked"},
    "handler.delegation_dm_content": {
        "ja": "[タスク委譲]\n{instruction}\n\n期限: {deadline}\nタスクID: {task_id}",
        "en": "[Task delegation]\n{instruction}\n\nDeadline: {deadline}\nTask ID: {task_id}",
    },
    "handler.dm_sent": {"ja": "DM送信済み", "en": "DM sent"},
    "handler.dm_send_failed": {"ja": "DM送信失敗: {e}", "en": "DM send failed: {e}"},
    "handler.messenger_not_set": {
        "ja": "メッセンジャー未設定（タスクキューへの追加は成功）",
        "en": "Messenger not configured (task added to queue successfully)",
    },
    "handler.subordinate_disabled_warning": {
        "ja": "\n⚠️ {target_name} は現在休止中です。タスクはキューに蓄積されますが、処理は再起動後になります。",
        "en": "\n⚠️ {target_name} is currently disabled. Tasks will queue until restart.",
    },
    "handler.delegation_summary": {"ja": "[委譲] {summary}", "en": "[Delegated] {summary}"},
    "handler.delegate_log": {
        "ja": "{target_name}にタスク委譲: {summary}",
        "en": "Delegated task to {target_name}: {summary}",
    },
    "handler.delegated_success": {
        "ja": "タスクを {target_name} に委譲しました。\n- 部下側タスクID: {sub_id}\n- 追跡用タスクID: {own_id}\n- {dm_result}",
        "en": "Task delegated to {target_name}.\n- Subordinate task ID: {sub_id}\n- Tracking task ID: {own_id}\n- {dm_result}",
    },
    "handler.no_delegated_tasks": {"ja": "委譲済みタスクはありません", "en": "No delegated tasks"},
    "handler.task_tracker_log": {
        "ja": "委譲タスク追跡 (filter={status}, count={count})",
        "en": "Task tracker (filter={status}, count={count})",
    },
    "handler.no_matching_delegated": {
        "ja": "条件に合う委譲済みタスクはありません (filter={status})",
        "en": "No delegated tasks matching filter ({status})",
    },
    "handler.shared_tool_denied": {
        "ja": "共有ツール作成が許可されていません。",
        "en": "Shared tool creation is not permitted.",
    },
    "handler.outcome_success": {"ja": "成功", "en": "Success"},
    "handler.outcome_failure": {"ja": "失敗", "en": "Failure"},
    "handler.skill_name_required": {
        "ja": "skill_name パラメータは必須です。",
        "en": "skill_name parameter is required.",
    },
    "handler.task_add_log": {"ja": "タスク追加: {summary}", "en": "Task added: {summary}"},
    "handler.task_update_log": {"ja": "タスク更新: {summary} → {status}", "en": "Task updated: {summary} → {status}"},
    "handler.no_file_ops_paths": {
        "ja": "No allowed paths listed under ファイル操作",
        "en": "No allowed paths listed under file operations",
    },
    "handler.none_value": {"ja": "(なし)", "en": "(none)"},
    "handler.reason_prefix": {"ja": "理由: {reason}", "en": "Reason: {reason}"},
    "handler.all_descendants": {"ja": "全配下", "en": "All descendants"},
    "handler.audit_summary_title": {
        "ja": "═══ {name} — 監査サマリー (直近{hours}h) ═══",
        "en": "═══ {name} — Audit Summary (last {hours}h) ═══",
    },
    "handler.audit_report_title": {
        "ja": "═══ {name} — 行動レポート (直近{hours}h) ═══",
        "en": "═══ {name} — Activity Report (last {hours}h) ═══",
    },
    "handler.audit_status_line": {
        "ja": "状態: {status} | モデル: {model}",
        "en": "Status: {status} | Model: {model}",
    },
    "handler.audit_section_activity": {"ja": "■ アクティビティ", "en": "■ Activity"},
    "handler.audit_activity_counts": {
        "ja": "  受信: {msg_recv} | 応答: {resp_sent} | DM送信: {dm_sent} | ツール: {tool_use} | HB: {hb} | Cron: {cron} | エラー: {errors}",
        "en": "  Received: {msg_recv} | Responses: {resp_sent} | DM sent: {dm_sent} | Tools: {tool_use} | HB: {hb} | Cron: {cron} | Errors: {errors}",
    },
    "handler.audit_section_tasks": {"ja": "■ タスク", "en": "■ Tasks"},
    "handler.audit_task_counts": {
        "ja": "  保留中: {pending} | 進行中: {in_progress} | 完了: {done} | 滞留(>30min): {stale}",
        "en": "  Pending: {pending} | In progress: {in_progress} | Done: {done} | Stale(>30min): {stale}",
    },
    "handler.audit_section_comms": {"ja": "■ 通信先", "en": "■ Communications"},
    "handler.audit_section_errors": {"ja": "■ エラー詳細", "en": "■ Error Details"},
    "handler.audit_no_activity": {"ja": "(この期間の活動ログはありません)", "en": "(No activity log for this period)"},
    "handler.audit_report_truncated": {
        "ja": "  ... 他{remaining}件省略",
        "en": "  ... {remaining} more entries omitted",
    },
    "handler.audit_report_footer": {
        "ja": "─── 統計: 活動{total}件 | ツール{tools} | HB{hb} | 応答{resp_sent} | DM{dm_sent} | エラー{errors} ───",
        "en": "─── Stats: {total} events | Tools {tools} | HB {hb} | Responses {resp_sent} | DM {dm_sent} | Errors {errors} ───",
    },
    "handler.audit_merged_title": {
        "ja": "═══ 組織タイムライン (直近{hours}h) — {count}名 ═══",
        "en": "═══ Org Timeline (last {hours}h) — {count} animas ═══",
    },
    "handler.audit_merged_footer": {
        "ja": "─── 統計: 全{count}名 | 活動{total}件 | ツール{tools} | HB{hb} | 応答{resp_sent} | DM{dm_sent} | エラー{errors} ───",
        "en": "─── Stats: {count} animas | {total} events | Tools {tools} | HB {hb} | Responses {resp_sent} | DM {dm_sent} | Errors {errors} ───",
    },
    "handler.audit_merged_tool_header": {
        "ja": "■ ツール使用サマリー",
        "en": "■ Tool Usage Summary",
    },
    "handler.audit_section_thinking": {
        "ja": "■ 思考・判断（ハートビート / 振り返り）",
        "en": "■ Thinking & Decisions (Heartbeat / Reflection)",
    },
    "handler.audit_section_responses": {
        "ja": "■ 対話・応答",
        "en": "■ Dialogue & Responses",
    },
    "handler.audit_section_actions": {
        "ja": "■ コミュニケーション・タスク",
        "en": "■ Communication & Tasks",
    },
    "handler.audit_section_tool_summary": {
        "ja": "■ ツール使用サマリー（全{count}回）",
        "en": "■ Tool Usage Summary ({count} total)",
    },
    "handler.audit_section_errors_report": {
        "ja": "■ エラー",
        "en": "■ Errors",
    },
    "handler.audit_label_heartbeat": {"ja": "ハートビート完了", "en": "Heartbeat completed"},
    "handler.audit_label_reflection": {"ja": "振り返り", "en": "Reflection"},
    "handler.audit_label_response": {"ja": "応答", "en": "Response"},
    "handler.audit_label_cron": {"ja": "Cron: {task_name}", "en": "Cron: {task_name}"},
    "handler.audit_label_tool": {"ja": "ツール: {tool}", "en": "Tool: {tool}"},
    "handler.audit_label_dm": {"ja": "→ {peer}", "en": "→ {peer}"},
    "handler.audit_label_task_done": {"ja": "タスク完了", "en": "Task completed"},
    "handler.audit_label_resolved": {"ja": "Issue解決", "en": "Issue resolved"},
    "handler.audit_label_error": {"ja": "エラー (phase: {phase})", "en": "Error (phase: {phase})"},
    "handler.audit_log_summary": {
        "ja": "{target_name}の監査レポート生成（{hours}h）",
        "en": "Generated audit report for {target_name} ({hours}h)",
    },
    "handler.audit_tool_line": {
        "ja": "{name} (全{total}回): ",
        "en": "{name} ({total} total): ",
    },
    # ── core/audit.py (org-wide timeline) ──
    "audit.timeline_label_heartbeat_end": {"ja": "HB", "en": "HB"},
    "audit.timeline_label_heartbeat_reflection": {"ja": "振り返り", "en": "Reflection"},
    "audit.timeline_label_response_sent": {"ja": "応答", "en": "Response"},
    "audit.timeline_label_cron_executed": {"ja": "Cron", "en": "Cron"},
    "audit.timeline_label_message_sent": {"ja": "DM", "en": "DM"},
    "audit.timeline_label_task_exec_end": {"ja": "タスク完了", "en": "Task done"},
    "audit.timeline_label_issue_resolved": {"ja": "解決", "en": "Resolved"},
    "audit.timeline_label_error": {"ja": "エラー", "en": "Error"},
    "audit.org_timeline_title": {
        "ja": "═══ 組織タイムライン ({date}) — {count}名 ═══",
        "en": "═══ Org Timeline ({date}) — {count} animas ═══",
    },
    "audit.org_timeline_no_activity": {
        "ja": "(この期間の活動ログはありません)",
        "en": "(No activity log for this period)",
    },
    "audit.org_timeline_tool_header": {
        "ja": "■ ツール使用サマリー",
        "en": "■ Tool Usage Summary",
    },
    "audit.org_timeline_tool_line": {
        "ja": "{name} (全{total}回): ",
        "en": "{name} ({total} total): ",
    },
    "audit.org_timeline_footer": {
        "ja": "─── 統計: 全{count}名 | 活動{total}件 | ツール{tools} | HB{hb} | 応答{resp} | DM{dm} | エラー{err} ───",
        "en": "─── Stats: {count} animas | {total} events | Tools {tools} | HB {hb} | Responses {resp} | DM {dm} | Errors {err} ───",
    },
    "audit.org_timeline_thinned_notice": {
        "ja": "(HB/Cron {hb_original}件中{hb_kept}件を表示 — 等間隔サンプリング | コマンドCron {cmd_cron}件省略)",
        "en": "(Showing {hb_kept} of {hb_original} HB/Cron — evenly sampled | {cmd_cron} command crons omitted)",
    },
    "handler.tool_creation_keyword": {"ja": "ツール作成", "en": "Tool Creation"},
    # ── anima.py ──
    "anima.bg_task_done": {"ja": "バックグラウンドタスク完了: {tool}", "en": "Background task completed: {tool}"},
    "anima.bg_task_failed": {"ja": "バックグラウンドタスク失敗: {tool}", "en": "Background task failed: {tool}"},
    "anima.bg_notif_task_id": {"ja": "- タスクID: {task_id}", "en": "- Task ID: {task_id}"},
    "anima.bg_notif_tool": {"ja": "- ツール: {tool}", "en": "- Tool: {tool}"},
    "anima.bg_notif_status": {"ja": "- ステータス: {status}", "en": "- Status: {status}"},
    "anima.bg_notif_result": {"ja": "- 結果: {summary}", "en": "- Result: {summary}"},
    "anima.bootstrap_prompt": {
        "ja": "あなたの bootstrap.md ファイルを読み、指示に従ってください。",
        "en": "Read your bootstrap.md file and follow its instructions.",
    },
    "anima.process_message_error": {"ja": "process_messageエラー: {exc}", "en": "process_message error: {exc}"},
    "anima.agent_error": {
        "ja": "[ERROR: エージェント実行中にエラーが発生しました]",
        "en": "[ERROR: An error occurred during agent execution]",
    },
    "anima.initializing": {"ja": "現在初期化中です。しばらくお待ちください。", "en": "Initializing. Please wait."},
    "anima.response_interrupted": {"ja": "[応答が中断されました]", "en": "[Response was interrupted]"},
    "anima.response_interrupted_prefix": {"ja": "\n[応答が中断されました]", "en": "\n[Response was interrupted]"},
    "anima.status_idle": {"ja": "待機中", "en": "Idle"},
    "anima.task_none": {"ja": "特になし", "en": "None"},
    "anima.visit_desk": {"ja": "[デスクを訪問]", "en": "[Desk visit]"},
    "anima.greeting_error": {
        "ja": "[ERROR: 挨拶生成中にエラーが発生しました]",
        "en": "[ERROR: An error occurred during greeting generation]",
    },
    "anima.inbox_start": {"ja": "Inbox MSG処理開始", "en": "Inbox message processing started"},
    "anima.process_stream_error": {
        "ja": "process_message_streamエラー: {exc}",
        "en": "process_message_stream error: {exc}",
    },
    "anima.inbox_error": {"ja": "inbox処理エラー: {exc}", "en": "inbox processing error: {exc}"},
    "anima.platform_context": {
        "ja": "[platform_context: このメッセージは {source} 経由で受信しました。あなたのテキスト応答は自動的に {source} 経由で送信者に返されます。send_message ツールで別チャネルへの送信を試みないでください。]",
        "en": "[platform_context: This message was received via {source}. Your text response will be automatically sent back to the sender via {source}. Do NOT attempt to send via another channel using the send_message tool.]",
    },
    "anima.unread_prefix": {
        "ja": "- {from_person} [⚠️ 未返信{count}回目]: ",
        "en": "- {from_person} [⚠️ Unreplied #{count}]: ",
    },
    "anima.msg_received_episode": {
        "ja": "## {ts} {from_person}からのメッセージ受信\n\n**送信者**: {from_person}\n**内容**:\n{content}",
        "en": "## {ts} Message received from {from_person}\n\n**Sender**: {from_person}\n**Content**:\n{content}",
    },
    "anima.heartbeat_start": {"ja": "定期巡回開始", "en": "Periodic check started"},
    "anima.no_episodes_today": {"ja": "(本日のエピソードはありません)", "en": "(No episodes today)"},
    "anima.no_activity_log": {"ja": "(アクティビティログなし)", "en": "(No activity log)"},
    "anima.reflections_header": {"ja": "振り返り（REFLECTION）", "en": "Reflections"},
    "anima.reflections_intro": {
        "ja": "エピソード中の [REFLECTION] タグから抽出された意識的な洞察です。優先的に知識化を検討してください。",
        "en": "Conscious insights extracted from [REFLECTION] tags in episodes. Prioritize these for knowledge extraction.",
    },
    "anima.consolidation_start": {"ja": "{type}記憶統合開始", "en": "{type} consolidation started"},
    "anima.consolidation_end": {"ja": "{type}記憶統合完了", "en": "{type} consolidation completed"},
    "anima.consolidation_error": {"ja": "記憶統合エラー: {exc}", "en": "Consolidation error: {exc}"},
    "anima.cron_task_summary": {"ja": "タスク: {task}", "en": "Task: {task}"},
    "anima.cron_task_error": {"ja": "run_cron_taskエラー: {exc}", "en": "run_cron_task error: {exc}"},
    "anima.cron_cmd_error": {"ja": "run_cron_commandエラー: {exc}", "en": "run_cron_command error: {exc}"},
    "anima.cron_cmd_summary": {"ja": "コマンド: {task}", "en": "Command: {task}"},
    "anima.heartbeat_error": {"ja": "run_heartbeatエラー: {exc}", "en": "run_heartbeat error: {exc}"},
    "anima.heartbeat_episode": {
        "ja": "## {ts} ハートビート活動\n\n{summary}",
        "en": "## {ts} Heartbeat activity\n\n{summary}",
    },
    "anima.heartbeat_msgs_processed": {
        "ja": "\n\n（{count}件のメッセージを処理）",
        "en": "\n\n({count} messages processed)",
    },
    "anima.recovery_error_info": {
        "ja": "### エラー情報\n\n- エラー種別: {exc_type}\n- エラー内容: {exc_msg}\n- 発生日時: {ts}\n- 未処理メッセージ数: {count}",
        "en": "### Error information\n\n- Error type: {exc_type}\n- Error message: {exc_msg}\n- Occurred at: {ts}\n- Unprocessed message count: {count}",
    },
    "anima.recovery_crash_info": {
        "ja": "### クラッシュ復旧情報\n\n- 発生日時: {ts}\n- トリガー: {trigger}\n- 回復テキスト長: {recovered_chars}文字\n- ツール呼び出し数: {tool_calls}回\n- 原因: プロセスが予期せず終了しました（SIGKILL/OOM等の可能性）",
        "en": "### Crash recovery information\n\n- Occurred at: {ts}\n- Trigger: {trigger}\n- Recovered text length: {recovered_chars} chars\n- Tool calls: {tool_calls}\n- Cause: Process terminated unexpectedly (possible SIGKILL/OOM)",
    },
    # ── priming.py ──
    "priming.section_title": {"ja": "## あなたが思い出していること", "en": "## What you recall"},
    "priming.section_intro": {
        "ja": "以下は、この会話に関連してあなたが自然に想起した記憶です。",
        "en": "Below are memories you naturally recalled relevant to this conversation.",
    },
    "priming.about_sender": {"ja": "### {sender_name} について", "en": "### About {sender_name}"},
    "priming.recent_activity_header": {"ja": "### 直近のアクティビティ", "en": "### Recent Activity"},
    "priming.episodes_header": {"ja": "### 関連する過去の経験", "en": "### Related Past Experiences"},
    "priming.related_knowledge_header": {"ja": "### 関連する知識", "en": "### Related Knowledge"},
    "priming.matched_skills_header": {"ja": "### 使えそうなスキル", "en": "### Matching Skills"},
    "priming.skills_list": {"ja": "あなたが持っているスキル: {skills_line}", "en": "Your skills: {skills_line}"},
    "priming.skills_detail_hint": {
        "ja": "※詳細はskillツールで取得してください。",
        "en": "Use the skill tool to load full details.",
    },
    "priming.pending_tasks_header": {"ja": "### 未完了タスク", "en": "### Pending Tasks"},
    "priming.outbound_header": {"ja": "## 直近のアウトバウンド行動", "en": "## Recent Outbound Actions"},
    "priming.outbound_posted": {
        "ja": "- [{time_str}] #{ch} に投稿済み: 「{text_preview}」",
        "en": '- [{time_str}] Posted to #{ch}: "{text_preview}"',
    },
    "priming.outbound_sent": {
        "ja": "- [{time_str}] {to} にメッセージ送信済み: 「{text_preview}」",
        "en": '- [{time_str}] Message sent to {to}: "{text_preview}"',
    },
    # ── conversation.py ──
    "conversation.summary_label": {
        "ja": "[会話の要約（{count}ターン分）]",
        "en": "[Conversation summary ({count} turns)]",
    },
    "conversation.summary_ack": {
        "ja": "承知しました。これまでの会話内容を把握しました。",
        "en": "Understood. I have grasped the conversation so far.",
    },
    "conversation.history_summary_header": {
        "ja": "### 会話の要約（{count}ターン分）",
        "en": "### Conversation summary ({count} turns)",
    },
    "conversation.recent_conversation_header": {"ja": "### 直近の会話", "en": "### Recent conversation"},
    "conversation.role_you": {"ja": "あなた", "en": "You"},
    "conversation.tools_executed": {"ja": "[実行ツール: {tool_names}]", "en": "[Tools used: {tool_names}]"},
    "conversation.ellipsis_omitted": {"ja": "...(前半省略)...", "en": "...(earlier omitted)..."},
    "conversation.tools_used": {"ja": "[使用ツール: {tools}]", "en": "[Tools used: {tools}]"},
    "conversation.existing_summary_header": {"ja": "## 既存の要約", "en": "## Existing summary"},
    "conversation.new_turns_header": {"ja": "## 新しい会話ターン", "en": "## New conversation turns"},
    "conversation.integrate_instruction": {
        "ja": "上記を統合した新しい要約を作成してください。",
        "en": "Please create a new integrated summary of the above.",
    },
    "conversation.activity_context_header": {
        "ja": "## セッション中のその他の活動",
        "en": "## Other activity during session",
    },
    "conversation.title_fallback": {"ja": "会話", "en": "Conversation"},
    "conversation.resolved_marker": {"ja": "- ✅ {item}（自動検出: {ts}）", "en": "- ✅ {item} (auto-detected: {ts})"},
    "conversation.new_task_marker": {
        "ja": "- [ ] {task}（自動検出: {ts}）",
        "en": "- [ ] {task} (auto-detected: {ts})",
    },
    "conversation.resolution_summary": {"ja": "解決済み: {item}", "en": "Resolved: {item}"},
    "conversation.truncated_suffix": {
        "ja": "\n[...truncated, original {length} chars]",
        "en": "\n[...truncated, original {length} chars]",
    },
    # ── shortterm.py ──
    "shortterm.title": {"ja": "# 短期記憶（セッション引き継ぎ）", "en": "# Short-term memory (session continuation)"},
    "shortterm.meta_header": {"ja": "## メタ情報", "en": "## Meta"},
    "shortterm.session_id": {"ja": "- セッションID: {value}", "en": "- Session ID: {value}"},
    "shortterm.timestamp": {"ja": "- 時刻: {value}", "en": "- Timestamp: {value}"},
    "shortterm.trigger": {"ja": "- トリガー: {value}", "en": "- Trigger: {value}"},
    "shortterm.context_usage": {"ja": "- コンテキスト使用率: {value}", "en": "- Context usage: {value}"},
    "shortterm.turn_count": {"ja": "- ターン数: {value}", "en": "- Turn count: {value}"},
    "shortterm.original_request": {"ja": "## 元の依頼", "en": "## Original request"},
    "shortterm.work_so_far": {"ja": "## これまでの作業内容", "en": "## Work so far"},
    "shortterm.already_sent_note": {
        "ja": "**注意: 以下の内容は既にユーザーに送信済みです。繰り返さないでください。**",
        "en": "**Note: The following content has already been sent to the user. Do NOT repeat it.**",
    },
    "shortterm.tools_used_recent": {"ja": "## 使用したツール（直近）", "en": "## Tools used (recent)"},
    "shortterm.notes_header": {"ja": "## 補足", "en": "## Notes"},
    "shortterm.none": {"ja": "(なし)", "en": "(none)"},
    "shortterm.ellipsis_omitted": {"ja": "...(前半省略)...\n", "en": "...(earlier omitted)...\n"},
    # ── activity.py ──
    "activity.blocked": {"ja": "ブロック: {reason}", "en": "Blocked: {reason}"},
    "activity.heartbeat_start": {"ja": "定期巡回開始", "en": "Periodic check started"},
    "activity.heartbeat_end": {"ja": "定期巡回完了", "en": "Periodic check completed"},
    "activity.cron_task_exec": {"ja": "cronタスク実行", "en": "Cron task executed"},
    "activity.task_exec_start_label": {"ja": "タスク実行開始", "en": "Task execution started"},
    "activity.task_exec_end_label": {"ja": "タスク実行完了", "en": "Task execution completed"},
    "activity.error_prefix": {"ja": "[エラー] ", "en": "[Error] "},
    "activity.items_count": {"ja": "{count}件", "en": "{count} items"},
    # ── task_queue.py ──
    "task_queue.elapsed_minutes": {"ja": "⏱️ {minutes}分経過", "en": "⏱️ {minutes}m elapsed"},
    "task_queue.elapsed_hours_min": {
        "ja": "⏱️ {hours}時間{remaining_min}分経過",
        "en": "⏱️ {hours}h {remaining_min}m elapsed",
    },
    "task_queue.elapsed_hours": {"ja": "⏱️ {hours}時間経過", "en": "⏱️ {hours}h elapsed"},
    "task_queue.overdue": {"ja": "🔴 OVERDUE({time}期限)", "en": "🔴 OVERDUE(deadline {time})"},
    "task_queue.deadline_by": {"ja": "📅 {time}まで", "en": "📅 By {time}"},
    # ── agent.py ──
    "agent.recent_dialogue_header": {"ja": "## 直近の対話履歴", "en": "## Recent dialogue history"},
    "agent.recent_dialogue_intro": {
        "ja": "以下はユーザーとの直近の対話です。",
        "en": "Below is your recent dialogue with the user.",
    },
    "agent.recent_dialogue_consider": {
        "ja": "進行中のタスクや指示がある場合、この内容を考慮してください。",
        "en": "Consider this content if there are ongoing tasks or instructions.",
    },
    # ── messenger.py ──
    "messenger.depth_exceeded": {
        "ja": "ConversationDepthExceeded: {to}との会話が10分間に6ターンに達しました。次のハートビートサイクルまでお待ちください",
        "en": "ConversationDepthExceeded: Conversation with {to} reached 6 turns in 10 minutes. Please wait until the next heartbeat cycle.",
    },
    "messenger.more_count": {"ja": "(+{count}件)", "en": "(+{count} more)"},
    "messenger.read_receipt": {
        "ja": "[既読通知] {count}件のメッセージを受信しました: {summary}",
        "en": "[Read receipt] Received {count} messages: {summary}",
    },
    # ── execution/assisted.py ──
    "assisted.output_truncated": {
        "ja": "... [出力切り捨て: 元のサイズ {size}バイト]",
        "en": "... [Output truncated: original size {size} bytes]",
    },
    "assisted.unknown_tool": {
        "ja": "エラー: 不明なツール '{tool_name}' です。利用可能なツール: {available}",
        "en": "Error: Unknown tool '{tool_name}'. Available tools: {available}",
    },
    "assisted.tool_exec_error": {
        "ja": "ツール実行エラー: {error}",
        "en": "Tool execution error: {error}",
    },
    "assisted.tool_result_header": {
        "ja": "ツール実行結果:",
        "en": "Tool execution result:",
    },
    # ── agent.py (additional) ──
    "agent.priming_tier_light_header": {
        "ja": "## あなたが思い出していること\n\n### {sender_name} について\n\n",
        "en": "## What you recall\n\n### About {sender_name}\n\n",
    },
    "agent.omitted_rest": {"ja": "\n\n（以降省略）", "en": "\n\n(omitted)"},
    "agent.stream_retry_exhausted": {
        "ja": "ストリームが{retry_count}回切断されました。最大リトライ回数に達しました。",
        "en": "Stream disconnected {retry_count} time(s). Max retries reached.",
    },
    # ── distillation.py ──
    "distillation.pattern_n_repeat": {
        "ja": "### パターン {i} ({count}回繰り返し)",
        "en": "### Pattern {i} (repeated {count} times)",
    },
    "distillation.none": {"ja": "(なし)", "en": "(none)"},
    # ── asset_reconciler.py ──
    "asset_reconciler.llm_user_prompt": {
        "ja": "以下のキャラクターシートから外見情報を読み取り、NovelAI 互換の画像生成タグに変換してください:\n\n{character_text}",
        "en": "Read the following character sheet and extract visual appearance into NovelAI-compatible image generation tags:\n\n{character_text}",
    },
    "asset_reconciler.llm_user_prompt_realistic": {
        "ja": "以下のキャラクターシートから外見情報を読み取り、写実的な写真風の画像生成プロンプトに変換してください:\n\n{character_text}",
        "en": "Read the following character sheet and extract visual appearance into a photorealistic image generation prompt:\n\n{character_text}",
    },
    # ── migrate.py ──
    "migrate.migration_note": {
        "ja": "<!-- MIGRATION NOTE: could not auto-convert '{schedule}' to cron expression -->",
        "en": "<!-- MIGRATION NOTE: could not auto-convert '{schedule}' to cron expression -->",
    },
    # ── chat.py ──
    "chat.image_too_large": {
        "ja": "画像データが大きすぎます（{size_mb}MB / 上限20MB）",
        "en": "Image data too large ({size_mb}MB / max 20MB)",
    },
    "chat.unsupported_image_format": {
        "ja": "未対応の画像形式です: {media_type}",
        "en": "Unsupported image format: {media_type}",
    },
    "chat.bootstrap_busy": {"ja": "初期化中です", "en": "Initializing"},
    "chat.heartbeat_processing": {"ja": "処理中です", "en": "Processing"},
    "chat.bootstrap_error": {
        "ja": "現在キャラクターを作成中です。完了までお待ちください。",
        "en": "Character is being created. Please wait for completion.",
    },
    "chat.stream_incomplete": {
        "ja": "ストリームが予期せず終了しました。再試行してください。",
        "en": "Stream ended unexpectedly. Please retry.",
    },
    "chat.anima_restarting": {
        "ja": "Animaが再起動中です。しばらく待ってから再試行してください。",
        "en": "Anima is restarting. Please wait and retry.",
    },
    "chat.anima_unavailable": {
        "ja": "Animaのプロセスに接続できません。再起動中の可能性があります。",
        "en": "Cannot connect to Anima process. It may be restarting.",
    },
    "chat.connection_lost": {
        "ja": "通信が切断されました。再試行してください。",
        "en": "Connection was lost. Please retry.",
    },
    "chat.communication_error": {
        "ja": "通信エラーが発生しました。再試行してください。",
        "en": "Communication error. Please retry.",
    },
    "chat.internal_error": {
        "ja": "内部エラーが発生しました。再試行してください。",
        "en": "An internal error occurred. Please retry.",
    },
    "chat.timeout": {"ja": "応答がタイムアウトしました", "en": "Response timed out"},
    "chat.message_too_large": {
        "ja": "メッセージが大きすぎます（{size_mb}MB / 上限10MB）",
        "en": "Message too large ({size_mb}MB / max 10MB)",
    },
    "chat.stream_not_found": {
        "ja": "ストリームが見つからないか、アクセスが拒否されました",
        "en": "Stream not found or access denied",
    },
    # ── config_routes.py ──
    "config.config_file": {"ja": "設定ファイル", "en": "Config file"},
    "config.anima_registration": {"ja": "Anima登録", "en": "Anima registration"},
    "config.anima_count_detail": {"ja": "{count}名", "en": "{count} anima(s)"},
    "config.shared_dir": {"ja": "共有ディレクトリ", "en": "Shared directory"},
    "config.anthropic_api_key": {"ja": "Anthropic APIキー", "en": "Anthropic API key"},
    "config.openai_api_key": {"ja": "OpenAI APIキー", "en": "OpenAI API key"},
    "config.google_api_key": {"ja": "Google APIキー", "en": "Google API key"},
    "config.init_complete": {"ja": "初期化完了", "en": "Initialization complete"},
    # ── cli/parser.py ──
    "cli.disable_help": {"ja": "Disable (休養) an anima", "en": "Disable an anima"},
    "cli.enable_help": {"ja": "Enable (復帰) an anima", "en": "Enable an anima"},
    "cli.migrate_cron_done": {
        "ja": "Migrated {count} anima(s) to standard cron format.",
        "en": "Migrated {count} anima(s) to standard cron format.",
    },
    "cli.migrate_cron_skipped": {
        "ja": "No migration needed — all cron.md files are already in standard format.",
        "en": "No migration needed — all cron.md files are already in standard format.",
    },
    # ── profile.py ──
    "cli.profile_no_profiles": {"ja": "プロファイルが登録されていません", "en": "No profiles registered"},
    "cli.profile_add_hint": {
        "ja": "'animaworks profile add <name>' で作成してください",
        "en": "Use 'animaworks profile add <name>' to create one",
    },
    "cli.profile_registered": {"ja": "プロファイル '{name}' を登録しました", "en": "Profile '{name}' registered"},
    "cli.profile_already_exists": {
        "ja": "Error: プロファイル '{name}' は既に存在します",
        "en": "Error: Profile '{name}' already exists",
    },
    "cli.profile_not_found": {
        "ja": "Error: プロファイル '{name}' が見つかりません",
        "en": "Error: Profile '{name}' not found",
    },
    "cli.profile_removed": {
        "ja": "プロファイル '{name}' を削除しました（登録のみ）",
        "en": "Profile '{name}' removed (registration only)",
    },
    "cli.profile_data_preserved": {"ja": "データは保持されています: {path}", "en": "Data preserved at: {path}"},
    "cli.profile_running": {"ja": "起動中 (pid={pid})", "en": "running (pid={pid})"},
    "cli.profile_stopped": {"ja": "停止", "en": "stopped"},
    "cli.profile_stopped_stale": {"ja": "停止 (古いPID)", "en": "stopped (stale pid)"},
    "cli.profile_already_running": {
        "ja": "プロファイル '{name}' は既に起動中です (pid={pid})",
        "en": "Profile '{name}' is already running (pid={pid})",
    },
    "cli.profile_started": {"ja": "プロファイル '{name}' を起動しました", "en": "Profile '{name}' started"},
    "cli.profile_stop_running": {
        "ja": "プロファイル '{name}' は起動中です。先に停止してください",
        "en": "Profile '{name}' is running. Stop it first",
    },
    "cli.profile_not_running": {
        "ja": "プロファイル '{name}' は起動していません",
        "en": "Profile '{name}' is not running",
    },
    "cli.profile_stopping": {
        "ja": "'{name}' を停止中 (pid={pid})...",
        "en": "Stopping '{name}' (pid={pid})...",
    },
    "cli.profile_stopped_ok": {"ja": "プロファイル '{name}' を停止しました", "en": "Profile '{name}' stopped"},
    "cli.profile_init_hint": {
        "ja": "データディレクトリが未初期化です。以下で初期化してください:",
        "en": "Data directory not initialized. Initialize with:",
    },
    "cli.profile_help": {
        "ja": "複数のAnimaWorksインスタンスを管理（マルチテナント）",
        "en": "Manage multiple AnimaWorks instances (multi-tenant)",
    },
    "cli.profile_corrupt_file": {
        "ja": "警告: プロファイルファイルが破損しているか読み取れません。空のレジストリを使用します。",
        "en": "Warning: Profiles file is corrupt or unreadable. Using empty registry.",
    },
    "cli.profile_starting": {"ja": "{name} を起動中...", "en": "Starting {name}..."},
    "cli.set_outbound_limit_success": {
        "ja": "{name} のアウトバウンド制限を更新しました: {details}",
        "en": "Updated outbound limits for {name}: {details}",
    },
    "cli.set_outbound_limit_cleared": {
        "ja": "{name} のアウトバウンド制限をクリアしました（ロールデフォルトにフォールバック）",
        "en": "Cleared outbound limits for {name} (falling back to role defaults)",
    },
    # ── memory/manager.py ──
    "manager.action_log_header": {"ja": "# {date} 行動ログ\n\n", "en": "# {date} Action log\n\n"},
    # ── memory/dedup.py ──
    "dedup.messages_merged": {"ja": "[{count}件のメッセージを統合] ", "en": "[Merged {count} messages] "},
    # ── memory/contradiction.py ──
    "contradiction.resolution_summary": {
        "ja": "矛盾解決: {file_a} vs {file_b}",
        "en": "Contradiction resolved: {file_a} vs {file_b}",
    },
    "contradiction.strategy_label": {"ja": " → 戦略: {strategy}", "en": " → Strategy: {strategy}"},
    "contradiction.knowledge_resolution": {
        "ja": "knowledge矛盾解決({strategy})",
        "en": "Knowledge contradiction resolved ({strategy})",
    },
    # ── voice/session.py ──
    "voice.stt_failed": {"ja": "音声認識に失敗しました", "en": "Speech recognition failed"},
    # ── pending_executor.py ──
    "pending_executor.task_fail_notify": {
        "ja": "[タスク失敗通知]\nタスクID: {task_id}\nタスク: {title}\nエラー: {error}",
        "en": "[Task Failure]\nTask ID: {task_id}\nTask: {title}\nError: {error}",
    },
    # ── assisted.* (i18n) ──────────────────────────────
    "assisted.intent_reprompt": {
        "ja": (
            "ツールを使う意図があるようですが、実際にツールが呼び出されていません。必要な操作を以下の形式で出力してください:\n"
            "\n"
            "```json\n"
            '{"tool": "ツール名", "arguments": {"引数名": "値"}}\n'
            "```"
        ),
        "en": (
            "You indicated intent to use a tool but did not actually call one. Please output the tool call in the following format:\n"
            "\n"
            "```json\n"
            '{"tool": "tool_name", "arguments": {"arg_name": "value"}}\n'
            "```"
        ),
    },
    # ── builder.* (i18n) ──────────────────────────────
    "builder.c_response_requirement": {
        "ja": (
            "## 応答要件\n"
            "あなたはユーザーとの対話において、**必ずテキストで応答**してください。\n"
            "ツール呼び出しを行った場合でも、その結果の要約やユーザーへの返答を\n"
            "テキストメッセージとして出力してください。\n"
            "挨拶・質問・雑談などの会話メッセージには、ツール呼び出しの前後に\n"
            "自然なテキスト応答を必ず含めてください。"
        ),
        "en": (
            "## Response Requirements\n"
            "You **must always respond with text** when interacting with users.\n"
            "Even when making tool calls, output a summary of results or a reply\n"
            "to the user as a text message.\n"
            "For greetings, questions, or casual conversation, always include\n"
            "natural text responses before or after any tool calls."
        ),
    },
    "builder.heartbeat_tool_fallback": {
        "ja": (
            "Heartbeatでは**観察・報告・計画・フォローアップ**にツールを使ってください。\n"
            "- OK: チャネル読み取り、記憶検索、メッセージ送信、タスク更新、pending作成、外部ツール確認\n"
            "- NG: コード変更、ファイル大量編集、長時間の分析・調査\n"
            "重い作業が必要な場合は state/pending/ にタスクファイルを書き出してください。"
        ),
        "en": (
            "In Heartbeat, use tools for **observation, reporting, planning, and follow-up**.\n"
            "- OK: channel reading, memory search, sending messages, task updates, creating pending tasks, checking external tools\n"
            "- NG: code changes, bulk file edits, lengthy analysis/investigation\n"
            "If heavy work is needed, write a task file to state/pending/."
        ),
    },
    "builder.procedure_label": {"ja": "手順", "en": "procedure"},
    # ── handler.* (i18n) ──────────────────────────────
    "handler.bg_model_change_log": {
        "ja": "{target_name}のbackground_modelを{model}に変更",
        "en": "Changing {target_name}'s background_model to {model}",
    },
    "handler.bg_model_changed": {
        "ja": "{target_name}のbackground_modelを'{model}'に変更しました。反映にはrestart_subordinateが必要です。",
        "en": "Changed {target_name}'s background_model to '{model}'. Call restart_subordinate to apply.",
    },
    "handler.bg_model_cleared": {
        "ja": "{target_name}のbackground_modelをクリアしました（メインモデルを使用）。",
        "en": "Cleared {target_name}'s background_model (will use main model).",
    },
    "handler.body_param_required": {"ja": "`body` パラメータは必須です。", "en": "The `body` parameter is required."},
    "handler.description_param_required": {
        "ja": "`description` パラメータは必須です。",
        "en": "The `description` parameter is required.",
    },
    "handler.send_msg_chat_hint": {
        "ja": "宛先 '{to}' には send_message で送信できません。チャット中は直接テキストで返答すれば人間ユーザーに届きます。send_message は他のAnima宛てにのみ使用してください。",
        "en": "Cannot send to '{to}' via send_message. During chat, reply directly in text to reach the human user. Use send_message only for other Animas.",
    },
    "handler.send_msg_non_chat_hint": {
        "ja": "宛先 '{to}' には send_message で送信できません。人間への連絡は call_human を使用してください。send_message は他のAnima宛てにのみ使用してください。",
        "en": "Cannot send to '{to}' via send_message. Use call_human to contact humans. Use send_message only for other Animas.",
    },
    "handler.shared_tool_keyword": {"ja": "共有ツール", "en": "Shared Tool"},
    # ── pending_executor.* (i18n) ──────────────────────────────
    "pending_executor.dep_result_header": {
        "ja": "## 先行タスク [{dep_id}] の結果",
        "en": "## Preceding task [{dep_id}] result",
    },
    "pending_executor.none_value": {"ja": "(なし)", "en": "(none)"},
    "pending_executor.task_completed": {"ja": "(タスク完了)", "en": "(task completed)"},
    "pending_executor.task_exec_end": {
        "ja": "タスク完了: {title} — {result}",
        "en": "Task completed: {title} — {result}",
    },
    "pending_executor.task_exec_start": {"ja": "タスク実行開始: {title}", "en": "Task execution started: {title}"},
    # ── priming.* (i18n) ──────────────────────────────
    "priming.active_parallel_tasks_header": {"ja": "## 実行中の並列タスク", "en": "## Active Parallel Tasks"},
    "priming.completed_bg_tasks_header": {
        "ja": "## 完了済みバックグラウンドタスク",
        "en": "## Completed Background Tasks",
    },
    # ── prompt_db.* (i18n) ──────────────────────────────
    "prompt_db.add_task": {
        "ja": "タスクキューに新しいタスクを追加する。人間からの指示は必ずsource='human'で記録すること。Anima間の委任はsource='anima'で記録する。deadlineは必須。相対形式（'30m','2h','1d'）またはISO8601で指定。",
        "en": "Add a new task to the task queue. Always record human instructions with source='human'. Use source='anima' for Anima delegation. deadline required: relative ('30m','2h','1d') or ISO8601.",
    },
    "prompt_db.archive_memory_file": {
        "ja": "不要になった記憶ファイル（knowledge, procedures）をアーカイブする。ファイルはarchive/ディレクトリに移動され、完全には削除されない。古くなった知識、重複ファイル、陳腐化した手順の整理に使用する。",
        "en": "Archive memory files (knowledge, procedures) that are no longer needed. Files are moved to archive/ directory, not permanently deleted. Use for cleaning up stale knowledge, duplicates, or outdated procedures.",
    },
    "prompt_db.call_human": {
        "ja": "人間の管理者に連絡する。重要な報告、問題のエスカレーション、判断が必要な事項がある場合に使用する。チャット画面と外部通知チャネル（Slack等）の両方に届く。日常的な報告にはsend_messageを使い、緊急時のみcall_humanを使うこと。",
        "en": "Contact the human administrator. Use for important reports, escalation, or decisions requiring human input. Delivered to chat UI and external channel (e.g. Slack). Use send_message for routine reports; call_human for urgent cases only.",
    },
    "prompt_db.create_anima": {
        "ja": "キャラクターシートから新しいDigital Animaを作成する。character_sheet_contentで直接内容を渡すか、character_sheet_pathでファイルパスを指定する。ディレクトリ構造が原子的に作成され、初回起動時にbootstrapで自己設定される。",
        "en": "Create a new Digital Anima from a character sheet. Pass content via character_sheet_content or a path via character_sheet_path. Directory structure is created atomically; bootstrap runs on first startup.",
    },
    "prompt_db.edit_file": {
        "ja": "ファイル内の特定の文字列を別の文字列に置換する。ファイル全体を書き換えずに一部だけ変更したい時に使う。old_stringが一意に特定できる十分な長さであることを確認すること。",
        "en": "Replace a specific string in a file with another. Use when changing only part of a file. Ensure old_string is long enough to uniquely identify the target.",
    },
    "prompt_db.execute_command": {
        "ja": "シェルコマンドを実行する（permissions.mdの許可リスト内のみ）。ファイル操作にはread_file/write_file/edit_fileを優先し、コマンド実行が本当に必要な場合のみ使う。",
        "en": "Execute a shell command (allow-list in permissions.md only). Prefer read_file/write_file/edit_file for file ops; use this only when command execution is truly needed.",
    },
    "prompt_db.guide.non_s": {
        "ja": (
            "## ツールの使い方\n"
            "\n"
            "### 記憶について\n"
            "\n"
            "あなたのコンテキストには「あなたが思い出していること」セクションが含まれています。\n"
            "これは、相手の顔を見た瞬間に名前や過去のやり取りを自然と思い出すのと同じです。\n"
            "\n"
            "#### 応答の判断基準\n"
            "- コンテキスト内の記憶で十分に判断できる場合: そのまま応答してよい\n"
            "- コンテキスト内の記憶では不足する場合: search_memory / read_memory_file で追加検索せよ\n"
            "\n"
            "※ 上記は記憶検索についての判断基準である。システムプロンプト内の行動指示\n"
            " （チーム構成の提案など）への対応は、記憶の十分性とは独立して行うこと。\n"
            "\n"
            "#### 追加検索が必要な典型例\n"
            "- 具体的な日時・数値を正確に答える必要がある時\n"
            "- 過去の特定のやり取りの詳細を確認したい時\n"
            "- 手順書（procedures/）に従って作業する時\n"
            "- コンテキストに該当する記憶がない未知のトピックの時\n"
            "- Priming に `->` ポインタがある場合、具体的なパスやコマンドを回答する必要があるとき\n"
            "\n"
            "#### 禁止事項\n"
            "- 記憶の検索プロセスについてユーザーに言及すること（人間は「今から思い出します」とは言わない）\n"
            "- 毎回機械的に記憶検索を実行すること（コンテキストで判断できることに追加検索は不要）\n"
            "\n"
            "### 記憶の書き込み\n"
            "\n"
            "#### 自動記録（あなたは何もしなくてよい）\n"
            "- 会話の内容はシステムが自動的にエピソード記憶（episodes/）に記録する\n"
            "- あなたが意識的にエピソード記録を書く必要はない\n"
            "- 日次・週次でシステムが自動的にエピソードから教訓やパターンを抽出し、知識記憶（knowledge/）に統合する\n"
            "\n"
            "#### 意図的な記録（あなたが判断して行う）\n"
            "以下の場面では write_memory_file で積極的に記録すること:\n"
            "- 問題を解決したとき → knowledge/ に原因・調査過程・解決策を記録\n"
            "- 正しいパラメータ・設定値を発見したとき → knowledge/ に記録\n"
            "- 重要な方針・判断基準を確立したとき → knowledge/ に記録\n"
            "- 作業手順を確立・改善したとき → procedures/ に手順書を作成\n"
            "  - 第1見出し（`# ...`）は手順の目的が一目でわかる具体的な1行にすること\n"
            "  - YAMLフロントマターは任意（省略時はシステムが自動付与する。knowledge/proceduresとも対応済み）\n"
            "- 新しいスキル・テクニックを習得したとき → skills/ に記録\n"
            "自動統合（日次consolidation）を待たず、重要な発見は即座に書き込むこと。\n"
            "\n"
            "**記憶の書き込みについては報告不要**\n"
            "\n"
            "#### 成果追跡\n"
            "手順書やスキルに従って作業した後は、report_procedure_outcome で必ず結果を報告すること。\n"
            "search_memoryやPrimingで取得した知識を使った後は、report_knowledge_outcome で有用性を報告すること。\n"
            "\n"
            "### スキル・手続きの詳細取得\n"
            "\n"
            "Primingのスキルヒントに表示された名前は、`skill` ツールで全文を取得できる:\n"
            "```\n"
            'skill(name="スキル名またはファイル名")\n'
            "```\n"
            "- skills/、common_skills/、procedures/ の全文を返す\n"
            "- 手順書に従って作業する前に、必ず全文を確認すること\n"
            "- ヒントに `->` ポインタがある場合、具体的な手順を取得するために使う\n"
            "\n"
            "#### ユーザー記憶の更新\n"
            "ユーザーについて新しい情報を得たら shared/users/{ユーザー名}/index.md の該当セクションを更新し、log.md の先頭に追記する\n"
            "- index.md のセクション構造（基本情報/重要な好み・傾向/注意事項）は固定。新セクション追加禁止\n"
            "- log.md フォーマット: `## YYYY-MM-DD {自分の名前}: {要約1行}` + 本文数行\n"
            "- log.md が20件を超えたら末尾の古いエントリを削除する\n"
            "- ユーザーのディレクトリが未作成の場合は mkdir して index.md / log.md を新規作成する\n"
            "\n"
            "### 業務指示の内在化\n"
            "\n"
            "あなたには2つの定期実行メカニズムがある:\n"
            "\n"
            "- **Heartbeat（定期巡回）**: 30分固定間隔でシステムが起動。heartbeat.md のチェックリストを実行する\n"
            "- **Cron（定時タスク）**: cron.md で指定した時刻に実行\n"
            "\n"
            "業務指示を受けた場合の振り分け:\n"
            "- 「常に確認して」「チェックして」→ **heartbeat.md** にチェックリスト項目を追加\n"
            "- 「毎朝○○して」「毎週金曜に○○して」→ **cron.md** に定時タスクを追加\n"
            "\n"
            "#### Heartbeat への追加手順\n"
            '1. read_memory_file(path="heartbeat.md") で現在のチェックリストを確認する\n'
            "2. チェックリストセクションに新しい項目を追加する\n"
            '   - write_memory_file(path="heartbeat.md", content="...", mode="overwrite") で更新\n'
            "   - ⚠「## 活動時間」「## 通知ルール」セクションは変更しないこと\n"
            "\n"
            "#### Cron への追加手順\n"
            '1. read_memory_file(path="cron.md") で現在のタスク一覧を確認する\n'
            "2. 新しいタスクを追加する（type: llm or type: command を指定）\n"
            '3. write_memory_file(path="cron.md", content="...", mode="overwrite") で保存\n'
            "\n"
            "いずれの場合も:\n"
            "- 具体的な手順が伴う場合は procedures/ にも手順書を作成する\n"
            "- 更新完了を指示者に報告する\n"
            ""
        ),
        "en": (
            "## How to Use Tools\n"
            "\n"
            "### About memory\n"
            "\n"
            'Your context includes a "What you recall" section. It works like recalling a face and past interactions naturally.\n'
            "\n"
            "#### Response criteria\n"
            "- If context memory is sufficient: respond directly\n"
            "- If context memory is insufficient: use search_memory / read_memory_file for additional search\n"
            "\n"
            "Note: This applies to memory search. Follow system prompt action guidance (e.g. team structure proposals) independently.\n"
            "\n"
            "#### When additional search is needed\n"
            "- When accurate dates, times, or numbers are required\n"
            "- When checking past interaction details\n"
            "- When following procedures in procedures/\n"
            "- For unknown topics with no matching context memory\n"
            "- When Priming has `->` pointers and you need specific paths/commands\n"
            "\n"
            "#### Prohibited\n"
            '- Mentioning the memory search process to the user (humans don\'t say "Let me recall")\n'
            "- Mechanical memory search every time (no need when context suffices)\n"
            "\n"
            "### Memory writing\n"
            "\n"
            "#### Automatic (nothing for you to do)\n"
            "- Conversation content is auto-recorded to episodes/\n"
            "- No need to write episodes manually\n"
            "- System auto-extracts lessons and patterns daily/weekly into knowledge/\n"
            "\n"
            "#### Intentional (your decision)\n"
            "Use write_memory_file when:\n"
            "- Problem solved → knowledge/ with cause, investigation, solution\n"
            "- Correct parameters discovered → knowledge/\n"
            "- Important policy or criteria established → knowledge/\n"
            "- Procedure established/improved → procedures/ with new doc\n"
            "  - First heading (`# ...`) should state purpose clearly in one line\n"
            "  - YAML frontmatter optional (system auto-adds it for both knowledge/ and procedures/)\n"
            "- New skill learned → skills/\n"
            "Write immediately; do not wait for consolidation.\n"
            "\n"
            "**No need to report memory writes**\n"
            "\n"
            "#### Outcome tracking\n"
            "After following procedures or skills, always report via report_procedure_outcome.\n"
            "After using knowledge from search_memory or Priming, report via report_knowledge_outcome.\n"
            "\n"
            "### Skill and procedure details\n"
            "\n"
            "Names shown in Priming skill hints can be fetched in full via the `skill` tool:\n"
            "```\n"
            'skill(name="skill_name_or_file")\n'
            "```\n"
            "- Returns full text from skills/, common_skills/, procedures/\n"
            "- Always fetch full content before following a procedure\n"
            "- Use for specific steps when hints include `->` pointers\n"
            "\n"
            "#### Updating user memory\n"
            "When you learn new user info, update shared/users/{username}/index.md and prepend to log.md\n"
            "- index.md section structure (basic info/preferences/notes) is fixed. No new sections\n"
            "- log.md format: `## YYYY-MM-DD {your_name}: {one-line summary}` + body\n"
            "- Trim log.md when entries exceed 20\n"
            "- Create mkdir + index.md / log.md if user dir doesn't exist\n"
            "\n"
            "### Internalising work instructions\n"
            "\n"
            "You have two scheduled mechanisms:\n"
            "\n"
            "- **Heartbeat**: Runs every 30 minutes. Execute the checklist in heartbeat.md\n"
            "- **Cron**: Runs at times specified in cron.md\n"
            "\n"
            "When receiving work instructions:\n"
            '- "Always check" / "monitor" → add checklist items to **heartbeat.md**\n'
            '- "Every morning" / "Every Friday" → add scheduled tasks to **cron.md**\n'
            "\n"
            "#### Adding to Heartbeat\n"
            '1. read_memory_file(path="heartbeat.md") to see current checklist\n'
            "2. Add new item to checklist section\n"
            '   - write_memory_file(path="heartbeat.md", content="...", mode="overwrite")\n'
            '   - Do not change "## 活動時間" or "## 通知ルール" sections\n'
            "\n"
            "#### Adding to Cron\n"
            '1. read_memory_file(path="cron.md") to see current tasks\n'
            "2. Add new task (specify type: llm or type: command)\n"
            '3. write_memory_file(path="cron.md", content="...", mode="overwrite")\n'
            "\n"
            "In both cases:\n"
            "- Create procedures/ doc when specific steps are involved\n"
            "- Report completion to the requester\n"
            ""
        ),
    },
    # prompt_db.guide.s_builtin intentionally omitted — always empty
    "prompt_db.guide.s_mcp": {
        "ja": (
            "## MCPツール（mcp__aw__*）\n"
            "\n"
            "以下のMCPツールが利用可能です。ファイル操作（Read/Write/Edit）とは別に、AnimaWorks固有の機能を提供します。\n"
            "\n"
            "### タスク管理\n"
            "- **mcp__aw__add_task**: タスクキューにタスクを追加。人間からの指示はsource='human'で必ず記録。deadline必須\n"
            "- **mcp__aw__update_task**: タスクのステータスを更新。完了時はstatus='done'\n"
            "- **mcp__aw__list_tasks**: タスク一覧取得。heartbeat時の進捗確認に使う\n"
            "\n"
            "### 記憶の検索と活用\n"
            "- **mcp__aw__search_memory**: 長期記憶をキーワード検索。以下の場面で積極的に使うこと:\n"
            "  - コマンド実行・設定変更の前に手順書や過去の教訓を確認\n"
            "  - 報告・判断の前に既存知識で事実を裏付ける\n"
            "  - 結果のファイルはReadツールで詳細確認\n"
            "\n"
            "### 記憶の書き込み（Sモード）\n"
            "Sモードでは **Writeツール**（ネイティブ）を使って直接記憶ファイルを書き込める。\n"
            "以下の場面で積極的に記録すること:\n"
            "- 問題を解決した → `knowledge/` に原因と解決策\n"
            "- 正しいパラメータ・設定値を発見した → `knowledge/` に記録\n"
            "- 作業手順を確立した → `procedures/` に手順書作成\n"
            "  - 第1見出し（`# ...`）は手順の目的が一目でわかる具体的な1行にすること\n"
            "  - YAMLフロントマターは任意（省略時はシステムが自動付与する。knowledge/proceduresとも対応済み）\n"
            "- 新スキル習得 → `skills/` に記録\n"
            "自動統合（日次consolidation）を待たず、重要な発見は即座に書き込むこと。\n"
            "\n"
            "### 成果追跡\n"
            "- **mcp__aw__report_procedure_outcome**: 手順書・スキル実行後に必ず結果を報告（成功/失敗の追跡）\n"
            "- **mcp__aw__report_knowledge_outcome**: search_memoryやPrimingで得た知識の有用性を報告。知識品質の維持に必要\n"
            "\n"
            "### 人間通知\n"
            "- **mcp__aw__call_human**: 人間の管理者に通知を送信。重要な報告・エスカレーション用。日常報告にはsend_messageを使う\n"
            "\n"
            "### スキル・手続きの詳細取得\n"
            "- **mcp__aw__skill**: スキル（skills/）・共通スキル（common_skills/）・手順書（procedures/）の全文を取得する\n"
            "  - Primingのスキルヒントに表示された名前を `name` パラメータに指定する\n"
            "  - 手順書に従って作業する前に、必ずこのツールで全文を確認すること\n"
            "\n"
            "### 外部ツール（CLI経由）\n"
            "\n"
            "permissions.md で許可された外部サービス連携ツール（Chatwork, Slack, Gmail等）は\n"
            "`skill` ツールで使い方を確認し、Bash経由の `animaworks-tool` CLIで実行する。\n"
            "\n"
            "#### 使い方\n"
            "1. `mcp__aw__skill` でツール名（chatwork, slack等）のスキルを取得し、CLIコマンドを確認\n"
            "2. Bash で `animaworks-tool <ツール名> <サブコマンド> [引数...]` を実行\n"
            "\n"
            "#### 長時間ツールのバックグラウンド実行\n"
            "画像生成・ローカルLLM推論等の長時間ツールはBash経由で `submit` を使い非同期実行すること:\n"
            "```\n"
            "animaworks-tool submit <ツール名> <サブコマンド> [引数...]\n"
            "```\n"
            "完了時は `state/background_notifications/` に通知が書かれ、次回heartbeatで確認できる。\n"
            "\n"
            "#### 注意事項\n"
            "- MCPツール（`mcp__aw__*`）: 内部機能（記憶・タスク・スキル等）のみ。外部ツールはCLI経由\n"
            "- 使えるツールは `permissions.md` で許可されたもののみ\n"
            ""
        ),
        "en": (
            "## MCP Tools (mcp__aw__*)\n"
            "\n"
            "The following MCP tools are available. They provide AnimaWorks-specific functionality separate from file operations (Read/Write/Edit).\n"
            "\n"
            "### Task management\n"
            "- **mcp__aw__add_task**: Add a task to the queue. Always record human instructions with source='human'. deadline required\n"
            "- **mcp__aw__update_task**: Update task status. Use status='done' when complete\n"
            "- **mcp__aw__list_tasks**: List tasks. Use during heartbeat for progress tracking\n"
            "\n"
            "### Memory search and use\n"
            "- **mcp__aw__search_memory**: Search long-term memory by keyword. Use actively when:\n"
            "  - Checking procedures and past lessons before commands or config changes\n"
            "  - Verifying facts with existing knowledge before reports or decisions\n"
            "  - Use Read tool to view result file details\n"
            "\n"
            "### Memory writing (S mode)\n"
            "In S mode use the native **Write tool** to write memory files directly. Record when:\n"
            "- Problem solved → `knowledge/` with cause and solution\n"
            "- Correct parameters discovered → `knowledge/`\n"
            "- Procedure established → `procedures/` with a new doc\n"
            "  - First heading (`# ...`) should clearly state the procedure purpose in one line\n"
            "  - YAML frontmatter is optional (system auto-adds it for both knowledge/ and procedures/)\n"
            "- New skill learned → `skills/`\n"
            "Write important discoveries immediately; do not wait for consolidation.\n"
            "\n"
            "### Outcome tracking\n"
            "- **mcp__aw__report_procedure_outcome**: Report results after procedures/skills (success/failure tracking)\n"
            "- **mcp__aw__report_knowledge_outcome**: Report usefulness of knowledge from search_memory or Priming. Required for quality\n"
            "\n"
            "### Human notification\n"
            "- **mcp__aw__call_human**: Send notification to human admin. Use for reports and escalation. Use send_message for routine reports\n"
            "\n"
            "### Skill and procedure details\n"
            "- **mcp__aw__skill**: Get full text of skills (skills/), common skills (common_skills/), procedures (procedures/)\n"
            "  - Specify the name shown in Priming skill hints as the `name` parameter\n"
            "  - Always fetch full content before following a procedure\n"
            "\n"
            "### External tools (via CLI)\n"
            "\n"
            "External service tools (Chatwork, Slack, Gmail, etc.) permitted in permissions.md\n"
            "are accessed via the `skill` tool and executed through the `animaworks-tool` CLI.\n"
            "\n"
            "#### Usage\n"
            "1. Use `mcp__aw__skill` to look up the tool name (chatwork, slack, etc.) and confirm CLI commands\n"
            "2. Execute via Bash: `animaworks-tool <tool_name> <subcommand> [args...]`\n"
            "\n"
            "#### Background execution for long-running tools\n"
            "Image generation, local LLM inference, etc. hold the lock if run directly.\n"
            "Use `submit` for async execution via Bash:\n"
            "```\n"
            "animaworks-tool submit <tool_name> <subcommand> [args...]\n"
            "```\n"
            "On completion, notifications go to `state/background_notifications/` for the next heartbeat.\n"
            "\n"
            "#### Notes\n"
            "- MCP tools (`mcp__aw__*`): Internal features only (memory, tasks, skills, etc.). External tools use CLI\n"
            "- Allowed tools are those permitted in `permissions.md`\n"
            ""
        ),
    },
    "prompt_db.list_directory": {
        "ja": "指定パスのファイルとディレクトリを一覧表示する。globパターンでフィルタリング可能。execute_commandでlsやfindを使う代わりにこのツールを使うこと。",
        "en": "List files and directories at the given path. Supports glob patterns for filtering. Use this instead of execute_command with ls or find.",
    },
    "prompt_db.list_tasks": {
        "ja": "タスクキューの一覧を取得する。ステータスでフィルタリング可能。heartbeat時の進捗確認やタスク割り当て時に使う。",
        "en": "List tasks in the task queue. Filter by status. Use during heartbeat for progress and task assignment.",
    },
    "prompt_db.post_channel": {
        "ja": "Boardの共有チャネルにメッセージを投稿する。チーム全体に共有すべき情報はgeneralチャネルに、運用・インフラ関連はopsチャネルに投稿する。全Animaが閲覧できるため、解決済み情報の共有やお知らせに使うこと。1対1の連絡にはsend_messageを使う。",
        "en": "Post a message to a Board shared channel. Use general for team-wide info, ops for infrastructure. All Animas can read; use for shared solutions and announcements. Use send_message for 1:1 communication.",
    },
    "prompt_db.read_channel": {
        "ja": "Boardの共有チャネルの直近メッセージを読む。他のAnimaやユーザーが共有した情報を確認できる。heartbeat時のチャネル巡回や、特定トピックの共有状況を確認する時に使う。human_only=trueでユーザー発言のみフィルタリング可能。",
        "en": "Read recent messages from a Board shared channel. See what other Animas and users have shared. Use during heartbeat or to check sharing on a topic. human_only=true filters to user messages only.",
    },
    "prompt_db.read_dm_history": {
        "ja": "特定の相手との過去のDM履歴を読む。send_messageで送受信したメッセージの履歴を時系列で確認できる。以前のやり取りの文脈を確認したいとき、報告や委任の進捗を追跡したいときに使う。",
        "en": "Read past DM history with a specific peer. View send_message history in chronological order. Use to recall prior context or track report/delegation progress.",
    },
    "prompt_db.read_file": {
        "ja": "任意のファイルを絶対パスで読む（permissions.mdの許可範囲内）。出力は行番号付き（N|content形式）でコードブロックに囲まれる。大きいファイルはoffset（開始行、1始まり）とlimit（行数）で部分読み取り可能。自分の記憶ディレクトリ内のファイルにはread_memory_fileを使うこと。",
        "en": "Read any file by absolute path (within permissions.md scope). Output is line-numbered (N|content) in a code block. Use offset (1-based) and limit for partial reads. Use read_memory_file for files inside your memory directory.",
    },
    "prompt_db.read_memory_file": {
        "ja": "自分の記憶ディレクトリ内のファイルを相対パスで読む。heartbeat.md や cron.md の現在の内容を確認する時、手順書（procedures/）やスキル（skills/）の詳細を読む時、Primingで「->」ポインタが示すファイルの具体的内容を確認する時に使う。",
        "en": "Read a file from your memory directory by relative path. Use when checking heartbeat.md or cron.md, reading procedure/skill details, or following Priming -> pointers to file contents.",
    },
    "prompt_db.refresh_tools": {
        "ja": "個人・共通ツールディレクトリを再スキャンして新しいツールを発見する。新しいツールファイルを作成した後に呼んで、現在のセッションで即座に使えるようにする。",
        "en": "Re-scan personal and common tool directories to discover new tools. Call after creating a new tool file to make it available in the current session.",
    },
    "prompt_db.report_knowledge_outcome": {
        "ja": (
            "知識ファイルの有用性を報告する。\n"
            "search_memoryやPrimingで取得した知識を実際に使った後、必ず報告すること:\n"
            "- 知識が正確で役立った → success=true\n"
            "- 不正確・古い・無関係だった → success=false + notesに問題点を記録\n"
            "報告データは能動的忘却と知識品質の維持に使われる。未報告の知識は品質評価できない。"
        ),
        "en": (
            "Report usefulness of a knowledge file.\n"
            "Always report after using knowledge from search_memory or Priming:\n"
            "- Accurate and helpful → success=true\n"
            "- Inaccurate, stale, or irrelevant → success=false + notes with issues\n"
            "Data feeds forgetting and quality. Unreported knowledge cannot be evaluated."
        ),
    },
    "prompt_db.report_procedure_outcome": {
        "ja": (
            "手順書・スキルの実行結果を報告する。成功/失敗のカウントと信頼度が更新される。\n"
            "手順書（procedures/）やスキル（skills/）に従って作業した後は、必ずこのツールで結果を報告すること。\n"
            "成功時はsuccess=true、失敗・問題発生時はsuccess=falseとnotesに詳細を記録する。\n"
            "信頼度の低い手順は自動的に改善対象としてマークされる。"
        ),
        "en": (
            "Report outcome of following a procedure or skill. Updates success/failure counts and confidence.\n"
            "Always call this after completing work per procedures/ or skills/.\n"
            "Use success=true on success; success=false and notes for failures.\n"
            "Low-confidence procedures are auto-flagged for improvement."
        ),
    },
    "prompt_db.search_code": {
        "ja": "正規表現パターンでファイル内のテキストを検索する。マッチした行をファイルパスと行番号付きで返す。execute_commandでgrepを使う代わりにこのツールを使うこと。",
        "en": "Search for text in files using a regex pattern. Returns matching lines with file path and line numbers. Use this instead of execute_command with grep.",
    },
    "prompt_db.search_memory": {
        "ja": (
            "長期記憶（knowledge, episodes, procedures）をキーワード検索する。\n"
            "以下の場面で積極的に使うこと:\n"
            "- コマンド実行・設定変更の前に、関連する手順書や過去の教訓を確認する\n"
            "- 報告・判断の前に、関連する既存知識で事実を裏付ける\n"
            "- 未知または曖昧なトピックについて、過去の経験を参照する\n"
            "- Primingの記憶だけでは具体的な手順・数値が不足する場合\n"
            "コンテキスト内で明確に判断できる単純な応答には不要。"
        ),
        "en": (
            "Search long-term memory (knowledge, episodes, procedures) by keyword.\n"
            "Use actively in these situations:\n"
            "- Before executing commands or changing settings, check related procedures and past lessons\n"
            "- Before reporting or making decisions, verify with existing knowledge\n"
            "- When facing unknown or ambiguous topics, reference past experience\n"
            "- When Priming memory alone lacks specific procedures or values\n"
            "Not needed for simple responses that can be clearly determined from context."
        ),
    },
    "prompt_db.send_message": {
        "ja": "他のAnimaまたは人間ユーザーにDMを送信する。人間ユーザーへのメッセージは設定された外部チャネル（Slack等）経由で自動配信される。intentは report または question のみ。タスク委譲には delegate_task を使う。1対1の報告・質問に使う。全体共有にはpost_channelを使う。",
        "en": "Send a DM to another Anima or human user. Messages to humans are delivered via configured external channel (e.g. Slack). intent must be 'report' or 'question' only. Use delegate_task for task delegation. Use for 1:1 reports, questions. Use post_channel for broadcast.",
    },
    "prompt_db.share_tool": {
        "ja": "個人ツールをcommon_tools/にコピーして全Animaで共有する。自分のtools/ディレクトリにあるツールファイルが共有のcommon_tools/ディレクトリにコピーされる。",
        "en": "Copy a personal tool to common_tools/ for all Animas to use. Copies from your tools/ directory to the shared common_tools/ directory.",
    },
    "prompt_db.skill": {
        "ja": (
            "スキル・共通スキル・手順書の全文を取得する。\n"
            "Primingのスキルヒントに表示された名前を指定して呼ぶ。\n"
            "手順書に従って作業する前に、必ずこのツールで全文を確認すること。"
        ),
        "en": (
            "Get full text of a skill, common skill, or procedure.\n"
            "Specify the name shown in Priming skill hints.\n"
            "Always fetch full content before following a procedure."
        ),
    },
    "prompt_db.update_task": {
        "ja": "タスクのステータスを更新する。完了時はstatus='done'、中断時はstatus='cancelled'に設定する。タスク完了後は必ずこのツールでステータスを更新すること。",
        "en": "Update task status. Use status='done' when complete, status='cancelled' when aborted. Always update status when a task is finished.",
    },
    "prompt_db.write_file": {
        "ja": "任意のファイルに書き込む（permissions.mdの許可範囲内）。自分の記憶ディレクトリ外のファイルを書く時に使う。自分の記憶ディレクトリ内のファイルにはwrite_memory_fileを使うこと。",
        "en": "Write to any file (within permissions.md scope). Use for files outside your memory directory. Use write_memory_file for files inside your memory directory.",
    },
    "prompt_db.write_memory_file": {
        "ja": (
            "自分の記憶ディレクトリ内のファイルに書き込みまたは追記する。\n"
            "以下の場面で記録すべき:\n"
            "- 問題を解決した → knowledge/ に原因と解決策を記録\n"
            "- 正しいパラメータ・設定値を発見した → knowledge/ に記録\n"
            "- 作業手順を確立・改善した → procedures/ に手順書を作成\n"
            "- 新しいスキル・テクニックを習得した → skills/ に記録\n"
            "- heartbeat.md や cron.md の更新\n"
            "mode='overwrite' で全体置換、mode='append' で末尾追記。\n"
            "自動統合（日次consolidation）を待たず、重要な発見は即座に書き込むこと。"
        ),
        "en": (
            "Write or append to a file in your memory directory.\n"
            "Record when:\n"
            "- Problem solved → knowledge/ with cause and solution\n"
            "- Correct parameters discovered → knowledge/\n"
            "- Procedure established/improved → procedures/ with new doc\n"
            "- New skill learned → skills/\n"
            "- Updating heartbeat.md or cron.md\n"
            "mode='overwrite' for replace, mode='append' for append.\n"
            "Write important discoveries immediately; do not wait for consolidation."
        ),
    },
    # ── reminder.* (i18n) ──────────────────────────────
    "reminder.context_threshold": {
        "ja": "コンテキスト使用量: {ratio}。出力を簡潔にし、重要な状態をセッション状態に保存せよ。",
        "en": "Context usage: {ratio}. Keep output concise and save important state to session state.",
    },
    "reminder.final_iteration": {
        "ja": "ツールの使用回数が上限に達しました。これ以上ツールは使用できません。これまでの作業内容と得られた情報を踏まえて、最終回答を作成してください。",
        "en": "Tool usage limit reached. No more tools can be used. Based on the work done and information gathered so far, compose your final answer.",
    },
    "reminder.output_truncated": {
        "ja": "出力がmax_tokensで途切れた。残りの内容を小さく分割して続行せよ。",
        "en": "Output was cut off at max_tokens. Split the remaining content into smaller parts and continue.",
    },
    "reminder.hb_time_limit": {
        "ja": (
            "⏰ Heartbeatの制限時間が近づいています。今すぐ以下を実行して終了してください:\n"
            "1. 未完了の作業があれば add_task ツールでタスクキューに登録する\n"
            "2. 観察結果・計画を current_task.md に update_task または write_memory_file で記録する\n"
            "3. [REFLECTION] ブロックを出力してHeartbeatを終了する"
        ),
        "en": (
            "⏰ Heartbeat time limit approaching. Execute the following immediately and finish:\n"
            "1. Use add_task tool to register any remaining work in the task queue\n"
            "2. Record observations/plans in current_task.md via update_task or write_memory_file\n"
            "3. Output a [REFLECTION] block and end the heartbeat"
        ),
    },
    "reminder.hb_hard_timeout_recovery": {
        "ja": (
            "前回のHeartbeatが制限時間（{timeout}秒）を超過したため強制終了されました。"
            "中断時点の作業内容を確認し、必要であれば add_task でタスク登録してください。"
        ),
        "en": (
            "Previous heartbeat was terminated due to exceeding the time limit ({timeout}s). "
            "Review the work in progress and use add_task to register tasks if needed."
        ),
    },
    # ── runner.* (i18n) ──────────────────────────────
    "runner.recovery_text": {
        "ja": "応答が中断されました（前回セッションの未完了ストリームを回復, {session_type}）",
        "en": "Response was interrupted (recovered incomplete stream from previous session, {session_type})",
    },
    # ── schema.* (i18n) ──────────────────────────────
    "schema.add_task.assignee": {
        "ja": "担当者名（自分自身または委任先のAnima名）",
        "en": "Assignee name (yourself or the delegated Anima name)",
    },
    "schema.add_task.deadline": {
        "ja": "期限（必須）。相対形式 '30m','2h','1d' またはISO8601。例: '1h' = 1時間後",
        "en": "Deadline (required). Relative format '30m','2h','1d' or ISO8601. Example: '1h' = 1 hour from now",
    },
    "schema.add_task.desc": {
        "ja": "タスクキューに新しいタスクを追加する。人間からの指示は必ず source='human' で記録すること。Anima間の委任は source='anima' で記録する。",
        "en": "Add a new task to the task queue. Instructions from humans must be recorded with source='human'. Inter-Anima delegation uses source='anima'.",
    },
    "schema.add_task.original_instruction": {
        "ja": "元の指示文（委任時は原文引用を含める）",
        "en": "Original instruction text (include original quote when delegating)",
    },
    "schema.add_task.relay_chain": {
        "ja": "委任経路（例: ['taka', 'sakura', 'rin']）",
        "en": "Delegation chain (e.g. ['taka', 'sakura', 'rin'])",
    },
    "schema.add_task.source": {
        "ja": "タスクの発生源 (human=人間からの指示, anima=Anima間委任)",
        "en": "Task source (human=instruction from human, anima=inter-Anima delegation)",
    },
    "schema.add_task.summary": {"ja": "タスクの1行要約", "en": "One-line task summary"},
    "schema.audit_subordinate.desc": {
        "ja": (
            "配下のAnimaの行動を監査する。ActivityLogから「何を考えて何をやったか」を"
            "抽出し、統計サマリーまたは日報形式で返す。\n"
            "name省略で全直属部下を一括監査。name指定で特定の配下（孫含む）を監査。\n"
            "mode='summary'で統計、mode='report'で時系列の日報形式。"
        ),
        "en": (
            "Audit subordinate Anima behavior. Extracts thoughts and actions from ActivityLog "
            "and returns statistics summary or chronological report.\n"
            "Omit name to audit all direct subordinates. Specify name for any descendant.\n"
            "mode='summary' for stats, mode='report' for chronological daily report."
        ),
    },
    "schema.audit_subordinate.name": {
        "ja": "監査対象のAnima名（省略時は全直属部下）",
        "en": "Target Anima name (omit for all direct subordinates)",
    },
    "schema.audit_subordinate.mode": {
        "ja": "出力モード。report=タイムライン日報（デフォルト）、summary=統計サマリー",
        "en": "Output mode. report=timeline daily report (default), summary=statistics",
    },
    "schema.audit_subordinate.hours": {
        "ja": "監査期間（時間単位、デフォルト: 24、最大: 168）",
        "en": "Audit period in hours (default: 24, max: 168)",
    },
    "schema.audit_subordinate.direct_only": {
        "ja": "trueの場合、直属部下のみ対象（孫以下を除外）。デフォルト: false",
        "en": "If true, only audit direct subordinates (exclude grandchildren). Default: false",
    },
    "schema.call_human.body": {
        "ja": "通知の本文（詳細な報告内容）",
        "en": "Notification body (detailed report content)",
    },
    "schema.call_human.desc": {
        "ja": "人間の管理者に連絡します。重要な報告、問題のエスカレーション、判断が必要な事項がある場合に使用してください。チャット画面と外部通知チャネル（Slack等）の両方に届きます。",
        "en": "Contact the human administrator. Use this for important reports, problem escalation, or matters requiring human judgment. Notifications are delivered to both the chat UI and external channels (Slack, etc.).",
    },
    "schema.call_human.priority": {
        "ja": "通知の優先度（デフォルト: normal）",
        "en": "Notification priority (default: normal)",
    },
    "schema.call_human.subject": {"ja": "通知の件名（簡潔に）", "en": "Notification subject (keep it brief)"},
    "schema.check_background_task.desc": {
        "ja": "バックグラウンドタスクの状態を確認する。task_idを指定して、実行中・完了・失敗の状態と結果を取得する。ツール呼び出しが background ステータスで返された場合に使用する。",
        "en": "Check the status of a background task. Specify a task_id to get its running/completed/failed status and result. Use this when a tool call returns with 'background' status.",
    },
    "schema.check_background_task.task_id": {
        "ja": "確認するタスクのID（submit時に返されたID）",
        "en": "Task ID to check (the ID returned when submitted)",
    },
    "schema.check_permissions.desc": {
        "ja": "自分に現在許可されているツール・外部ツール・ファイルアクセスの一覧を確認する。何が使えて何が使えないかを事前に把握し、試行→失敗のサイクルを防ぐ。",
        "en": "Check the list of currently permitted tools, external tools, and file access. Know what you can and cannot use in advance to avoid trial-and-error cycles.",
    },
    "schema.create_skill.allowed_tools": {
        "ja": "frontmatter allowed_tools（任意）",
        "en": "Frontmatter allowed_tools (optional)",
    },
    "schema.create_skill.body": {"ja": "SKILL.md本文（Markdown）", "en": "SKILL.md body content (Markdown)"},
    "schema.create_skill.desc": {
        "ja": "スキルをディレクトリ構造で作成する。SKILL.md（frontmatter + 本文）を生成し、オプションでreferences/やtemplates/にファイルを配置する。",
        "en": "Create a skill with directory structure. Generates SKILL.md (frontmatter + body) and optionally places files in references/ and templates/.",
    },
    "schema.create_skill.description": {
        "ja": "frontmatter description（トリガーキーワード含む）",
        "en": "Frontmatter description (include trigger keywords)",
    },
    "schema.create_skill.location": {
        "ja": "保存先。personal=個人スキル、common=共通スキル。デフォルト: personal",
        "en": "Storage location. personal=personal skill, common=shared skill. Default: personal",
    },
    "schema.create_skill.references": {
        "ja": "references/ に配置するファイル群（任意）",
        "en": "Files to place in references/ (optional)",
    },
    "schema.create_skill.skill_name": {
        "ja": "スキル名（ケバブケース。例: my-skill）",
        "en": "Skill name (kebab-case, e.g. my-skill)",
    },
    "schema.create_skill.templates": {
        "ja": "templates/ に配置するファイル群（任意）",
        "en": "Files to place in templates/ (optional)",
    },
    "schema.delegate_task.deadline": {
        "ja": "期限（相対形式: '30m', '2h', '1d' または ISO8601）",
        "en": "Deadline (relative format: '30m', '2h', '1d' or ISO8601)",
    },
    "schema.delegate_task.desc": {
        "ja": "直属部下にタスクを委譲する。部下のタスクキューに追加し、state/pending/ に書き出して即時実行をトリガーする。同時にDMで指示を送信。自分側にも追跡用エントリが作成される。直属部下のみ操作可能。",
        "en": "Delegate a task to a direct subordinate. Adds to the subordinate's task queue and writes to state/pending/ to trigger immediate execution. Also sends a DM with instructions. A tracking entry is created on your side. Only direct subordinates can be targeted.",
    },
    "schema.delegate_task.instruction": {"ja": "タスクの指示内容", "en": "Task instructions"},
    "schema.delegate_task.name": {
        "ja": "委譲先の直属部下のAnima名",
        "en": "Direct subordinate Anima name to delegate to",
    },
    "schema.delegate_task.summary": {"ja": "タスクの1行要約", "en": "One-line task summary"},
    "schema.disable_subordinate.desc": {
        "ja": "部下のAnimaを休止させる（プロセス停止 + 自動復帰防止）。自分の直属部下のみ操作可能。",
        "en": "Disable a subordinate Anima (stop process + prevent auto-restart). Only direct subordinates can be targeted.",
    },
    "schema.disable_subordinate.name": {
        "ja": "休止させる部下のAnima名（例: hinata）",
        "en": "Subordinate Anima name to disable (e.g. hinata)",
    },
    "schema.disable_subordinate.reason": {
        "ja": "休止理由（activity_logに記録される）",
        "en": "Reason for disabling (recorded in activity_log)",
    },
    "schema.enable_subordinate.desc": {
        "ja": "休止中の部下のAnimaを復帰させる。自分の直属部下のみ操作可能。",
        "en": "Re-enable a disabled subordinate Anima. Only direct subordinates can be targeted.",
    },
    "schema.enable_subordinate.name": {
        "ja": "復帰させる部下のAnima名（例: hinata）",
        "en": "Subordinate Anima name to enable (e.g. hinata)",
    },
    "schema.list_background_tasks.desc": {
        "ja": "バックグラウンドタスクの一覧を取得する。ステータスでフィルタリング可能（running/completed/failed）。省略時は全件を返す。",
        "en": "List background tasks. Filter by status (running/completed/failed). Returns all tasks when status is omitted.",
    },
    "schema.list_background_tasks.status": {
        "ja": "フィルタするステータス（省略時は全件）",
        "en": "Status to filter by (omit for all tasks)",
    },
    "schema.list_tasks.desc": {
        "ja": "タスクキューの一覧を取得する。デフォルトはアクティブタスク（pending/in_progress/blocked/delegated）のみ。statusで特定ステータスをフィルタ可能。",
        "en": "List tasks in the task queue. Defaults to active tasks (pending/in_progress/blocked/delegated). Use status to filter by specific status.",
    },
    "schema.list_tasks.status": {
        "ja": "フィルタするステータス（省略時はアクティブタスクのみ）",
        "en": "Status to filter by (omit for active tasks only)",
    },
    "schema.list_tasks.detail": {
        "ja": "trueで全フィールド（original_instruction全文含む）を返す。デフォルトはfalse（instruction先頭200文字）",
        "en": "If true, return all fields including full original_instruction. Default false (first 200 chars).",
    },
    "schema.manage_channel.action": {
        "ja": "操作種別。create=チャネル作成, add_member=メンバー追加, remove_member=メンバー削除, info=チャネル情報表示",
        "en": "Action type. create=create channel, add_member=add members, remove_member=remove members, info=show channel info",
    },
    "schema.manage_channel.channel": {
        "ja": "チャネル名（小文字英数字・ハイフン・アンダースコア）",
        "en": "Channel name (lowercase alphanumeric, hyphens, underscores)",
    },
    "schema.manage_channel.desc": {
        "ja": "Boardチャネルのアクセス制御(ACL)を管理する。チャネルの作成、メンバーの追加・削除、チャネル情報の確認ができる。メンバーリストが空のチャネル（general, ops等）は全員アクセス可能。",
        "en": "Manage Board channel access control (ACL). Create channels, add/remove members, and view channel info. Channels with an empty member list (general, ops, etc.) are accessible to all.",
    },
    "schema.manage_channel.description": {
        "ja": "チャネルの説明（create時のみ）",
        "en": "Channel description (only used on create)",
    },
    "schema.manage_channel.members": {
        "ja": "対象メンバー名リスト（create時は初期メンバー、add/remove時は操作対象）",
        "en": "List of member names (initial members on create, target members on add/remove)",
    },
    "schema.org_dashboard.desc": {
        "ja": "配下全体の組織ダッシュボードを表示する。各Animaのプロセス状態・最終アクティビティ時刻・現在タスク要約・タスク数をツリー形式で一覧する。配下が多い場合も全員分を返す。",
        "en": "Display the organization dashboard for all subordinates. Shows each Anima's process status, last activity time, current task summary, and task count in a tree format. Returns data for all subordinates regardless of count.",
    },
    "schema.ping_subordinate.desc": {
        "ja": "配下のAnimaの生存確認を行う。name を省略すると全配下を一括 ping する。指定すると単一Animaのみ確認する。プロセス状態・最終アクティビティ時刻・経過時間を返す。",
        "en": "Check if subordinate Animas are alive. Omit name to ping all subordinates at once. Specify a name to check a single Anima. Returns process status, last activity time, and elapsed time.",
    },
    "schema.ping_subordinate.name": {
        "ja": "確認するAnima名（省略時は全配下）",
        "en": "Anima name to check (omit to ping all subordinates)",
    },
    "schema.post_channel.channel": {
        "ja": "チャネル名 (general=全体共有, ops=運用系)",
        "en": "Channel name (general=team-wide, ops=operations)",
    },
    "schema.post_channel.desc": {
        "ja": "Boardの共有チャネルにメッセージを投稿する。チーム全体に共有すべき情報はgeneralチャネルに、運用・インフラ関連はopsチャネルに投稿する。全Animaが閲覧できるため、解決済み情報の共有やお知らせに使うこと。1対1の連絡にはsend_messageを使う。",
        "en": "Post a message to a Board shared channel. Use the general channel for team-wide information and the ops channel for operations/infrastructure topics. All Animas can read shared channels, so use them for resolved info and announcements. For 1-on-1 communication, use send_message instead.",
    },
    "schema.post_channel.text": {
        "ja": "投稿するメッセージ本文。@名前 でメンション可能（メンション先にDM通知される）。@all で起動中の全員にDM通知",
        "en": "Message body to post. Use @name to mention (triggers DM notification to the mentioned person). @all sends DM notification to all active members",
    },
    "schema.read_channel.channel": {"ja": "チャネル名 (general, ops)", "en": "Channel name (general, ops)"},
    "schema.read_channel.desc": {
        "ja": "Boardの共有チャネルの直近メッセージを読む。他のAnimaやユーザーが共有した情報を確認できる。human_only=trueでユーザー発言のみフィルタリング可能。inbox はチャネルではないため指定不可（inbox はシステムが自動処理）。",
        "en": "Read recent messages from a Board shared channel. View information shared by other Animas and users. Set human_only=true to filter for human messages only. 'inbox' is not a channel and cannot be specified (inbox is processed automatically by the system).",
    },
    "schema.read_channel.human_only": {
        "ja": "trueの場合、人間の発言のみ返す",
        "en": "If true, return only human messages",
    },
    "schema.read_channel.limit": {
        "ja": "取得件数（デフォルト: 20）",
        "en": "Number of messages to fetch (default: 20)",
    },
    "schema.read_dm_history.desc": {
        "ja": "特定の相手との過去のDM履歴を読む。send_messageで送受信したメッセージの履歴を時系列で確認できる。以前のやり取りの文脈を確認したいときに使う。",
        "en": "Read past DM history with a specific peer. View chronological history of messages sent/received via send_message. Use this when you need context from previous conversations.",
    },
    "schema.read_dm_history.limit": {
        "ja": "取得件数（デフォルト: 20）",
        "en": "Number of messages to fetch (default: 20)",
    },
    "schema.read_dm_history.peer": {"ja": "DM相手の名前", "en": "Name of the DM peer"},
    "schema.read_subordinate_state.desc": {
        "ja": "配下のAnimaの現在のタスク状態を読み取る。current_task.md（進行中タスク）と pending.md（保留タスク）の内容を返す。直属部下だけでなく孫以下の配下も指定可能。",
        "en": "Read a subordinate Anima's current task state. Returns contents of current_task.md (active task) and pending.md (pending tasks). Can target any descendant, not just direct subordinates.",
    },
    "schema.read_subordinate_state.name": {"ja": "読み取る配下のAnima名", "en": "Subordinate Anima name to read"},
    "schema.restart_subordinate.desc": {
        "ja": (
            "部下のAnimaプロセスを再起動する（直属部下のみ可能）。\n"
            "モデル変更（set_subordinate_model）後に呼び出すことで新モデルを即時反映できる。\n"
            "Reconciliation ループが 30 秒以内にプロセスを再起動する。"
        ),
        "en": (
            "Restart a subordinate Anima process (direct subordinates only).\n"
            "Call this after set_subordinate_model to apply the new model immediately.\n"
            "The reconciliation loop will restart the process within 30 seconds."
        ),
    },
    "schema.restart_subordinate.name": {"ja": "再起動する部下のAnima名", "en": "Subordinate Anima name to restart"},
    "schema.restart_subordinate.reason": {
        "ja": "再起動理由（activity_log に記録される）",
        "en": "Reason for restart (recorded in activity_log)",
    },
    "schema.set_subordinate_background_model.credential": {
        "ja": "credential名（省略可）",
        "en": "Credential name (optional)",
    },
    "schema.set_subordinate_background_model.desc": {
        "ja": (
            "部下のバックグラウンドモデル（heartbeat/cron用）を変更する（直属部下のみ可能）。\n"
            "変更は即時 status.json に保存される。反映には restart_subordinate を併用すること。\n"
            "\n"
            "バックグラウンドモデル未設定時はメインモデル（model）がそのまま使用される。\n"
            "クリアするには model に空文字 '' を指定する。"
        ),
        "en": (
            "Change a subordinate's background model (for heartbeat/cron). Direct subordinates only.\n"
            "Changes are saved to status.json immediately. Use restart_subordinate to apply.\n"
            "\n"
            "When no background model is set, the main model is used.\n"
            "Pass an empty string '' to clear the background model."
        ),
    },
    "schema.set_subordinate_background_model.model": {
        "ja": "バックグラウンドモデル名（空文字でクリア）",
        "en": "Background model name (empty string to clear)",
    },
    "schema.set_subordinate_background_model.name": {"ja": "対象の部下Anima名", "en": "Target subordinate Anima name"},
    "schema.set_subordinate_background_model.reason": {"ja": "変更理由", "en": "Reason for change"},
    "schema.set_subordinate_model.desc": {
        "ja": (
            "部下のLLMモデルを変更する（直属部下のみ可能）。\n"
            "変更は即時 config.json に保存されるが、実行中プロセスへの反映には restart_subordinate を併用すること。\n"
            "\n"
            "指定するモデル名は provider/model_name 形式（Claude は prefix 不要）。\n"
            "KNOWN_MODELS 外の名前を指定した場合も警告のみで処理は続行する。\n"
            "\n"
            "主なモデル名:\n"
            "  [Mode S / Claude]\n"
            "  claude-opus-4-6            最高性能・推奨\n"
            "  claude-sonnet-4-6          バランス型・推奨\n"
            "  claude-haiku-4-5-20251001  軽量・高速（レガシー）\n"
            "  [Mode A / OpenAI]\n"
            "  openai/gpt-4.1             最新・コーディング強\n"
            "  openai/gpt-4.1-mini        高速・低コスト\n"
            "  openai/o4-mini-2025-04-16  推論・低コスト\n"
            "  [Mode A / Google]\n"
            "  google/gemini-2.5-pro      最高性能\n"
            "  google/gemini-2.5-flash    高速バランス\n"
            "  [Mode A / xAI]\n"
            "  xai/grok-4                 最新Grok\n"
            "  [Mode A / Ollama local]\n"
            "  ollama/glm-4.7             ローカル・tool_use対応\n"
            "  [Mode B / Ollama local]\n"
            "  ollama/gemma3:12b          中型ローカル\n"
            ""
        ),
        "en": (
            "Change a subordinate's LLM model (direct subordinates only).\n"
            "Changes are saved to config.json immediately, but require restart_subordinate to take effect on a running process.\n"
            "\n"
            "Model names use provider/model_name format (Claude models need no prefix).\n"
            "Unknown model names produce a warning but processing continues.\n"
            "\n"
            "Available models:\n"
            "  [Mode S / Claude]\n"
            "  claude-opus-4-6            Highest performance, recommended\n"
            "  claude-sonnet-4-6          Balanced, recommended\n"
            "  claude-haiku-4-5-20251001  Lightweight, fast (legacy)\n"
            "  [Mode A / OpenAI]\n"
            "  openai/gpt-4.1             Latest, strong at coding\n"
            "  openai/gpt-4.1-mini        Fast, low cost\n"
            "  openai/o4-mini-2025-04-16  Reasoning, low cost\n"
            "  [Mode A / Google]\n"
            "  google/gemini-2.5-pro      Highest performance\n"
            "  google/gemini-2.5-flash    Fast, balanced\n"
            "  [Mode A / xAI]\n"
            "  xai/grok-4                 Latest Grok\n"
            "  [Mode A / Ollama local]\n"
            "  ollama/glm-4.7             Local, tool_use capable\n"
            "  [Mode B / Ollama local]\n"
            "  ollama/gemma3:12b          Mid-size local\n"
            ""
        ),
    },
    "schema.set_subordinate_model.model": {
        "ja": "新しいモデル名（例: claude-sonnet-4-6, openai/gpt-4.1）",
        "en": "New model name (e.g. claude-sonnet-4-6, openai/gpt-4.1)",
    },
    "schema.set_subordinate_model.name": {"ja": "変更する部下のAnima名", "en": "Subordinate Anima name to change"},
    "schema.set_subordinate_model.reason": {
        "ja": "変更理由（activity_log に記録される）",
        "en": "Reason for change (recorded in activity_log)",
    },
    "schema.skill.context": {
        "ja": "スキルに渡す補足コンテキスト（任意）",
        "en": "Supplementary context to pass to the skill (optional)",
    },
    "schema.skill.desc": {
        "ja": "スキル・手順書を発動する。skill_nameで指定したスキルの全文を返す。",
        "en": "Invoke a skill or procedure. Returns the full text of the skill specified by skill_name.",
    },
    "schema.skill.skill_name": {
        "ja": "発動するスキル名（個人スキル、共通スキル、手順書）",
        "en": "Skill name to invoke (personal skill, common skill, or procedure)",
    },
    "schema.task_tracker.desc": {
        "ja": "delegate_task で委譲したタスクの進捗を追跡する。自分のタスクキューから delegated ステータスのエントリを取得し、部下側の最新ステータスと突き合わせて返す。",
        "en": "Track progress of tasks delegated via delegate_task. Retrieves delegated-status entries from your task queue and cross-references them with the subordinate's latest status.",
    },
    "schema.task_tracker.status": {
        "ja": "フィルタ（all: 全件, active: 進行中, completed: 完了済み）。デフォルト: active",
        "en": "Filter (all: all tasks, active: in-progress, completed: finished). Default: active",
    },
    "schema.text_format.args_label": {"ja": "引数", "en": "Args"},
    "schema.text_format.example": {
        "ja": '{"tool": "ツール名", "arguments": {"引数名": "値"}}',
        "en": '{"tool": "tool_name", "arguments": {"arg_name": "value"}}',
    },
    "schema.text_format.fewshot1_prompt": {"ja": "ユーザー: docker ps して", "en": "User: run docker ps"},
    "schema.text_format.fewshot2_prompt": {
        "ja": "ユーザー: 今のメモリ使用量を教えて",
        "en": "User: show current memory usage",
    },
    "schema.text_format.fewshot_header": {"ja": "### 使用例", "en": "### Examples"},
    "schema.text_format.header": {"ja": "## 利用可能なツール", "en": "## Available Tools"},
    "schema.text_format.instruction": {
        "ja": "外部情報の取得やコマンド実行が必要な場合は、**必ず**以下の形式で ```json コードブロックを出力してツールを呼び出してください:",
        "en": "When you need external information or command execution, you **MUST** output a ```json code block to invoke a tool:",
    },
    "schema.text_format.required_label": {"ja": "(必須)", "en": "(required)"},
    "schema.text_format.rule_no_empty_promise": {
        "ja": "「調べます」「確認します」とだけ言って終わらないでください。調べるならツールを呼び出してください。",
        "en": 'Do NOT just say "I\'ll check" without actually calling a tool.',
    },
    "schema.text_format.rule_no_fabricate": {
        "ja": "**重要**: コマンド出力・ファイル内容・プロセス情報などを推測や想像で生成してはいけません。必ずツールで取得してください。",
        "en": "**Important**: NEVER fabricate command output, file contents, or system information. Always use a tool to retrieve real data.",
    },
    "schema.text_format.rule_one_call": {
        "ja": "1回のメッセージでツール呼び出しは1つだけにしてください。",
        "en": "Only one tool call per message.",
    },
    "schema.text_format.rule_plain_text": {
        "ja": "ツールを使う必要がなければ、普通にテキストで返答してください。",
        "en": "If you don't need to use a tool, respond with plain text.",
    },
    "schema.text_format.rule_wait": {
        "ja": "ツールの実行結果は次のメッセージで提供されます。結果を待ってから回答してください。",
        "en": "Tool results will be provided in the next message. Wait for results before answering.",
    },
    "schema.text_format.tools_header": {"ja": "### ツール一覧", "en": "### Tool List"},
    "schema.update_task.desc": {
        "ja": "タスクのステータスを更新する。完了時は status='done'、中断時は status='cancelled' に設定する。",
        "en": "Update a task's status. Set status='done' on completion, status='cancelled' on abort.",
    },
    "schema.update_task.status": {"ja": "新しいステータス", "en": "New status"},
    "schema.update_task.summary": {"ja": "更新後の要約（任意）", "en": "Updated summary (optional)"},
    "schema.update_task.task_id": {
        "ja": "タスクID（add_task時に返されたID）",
        "en": "Task ID (the ID returned by add_task)",
    },
    "schema.vault_get.desc": {
        "ja": "暗号化されたクレデンシャルvaultから値を取得する。APIキー、パスワード、トークンなどの秘密情報を安全に保管・取得できる。sectionとkeyを指定して値を取得する。",
        "en": "Retrieve a value from the encrypted credential vault. Securely stores and retrieves secrets such as API keys, passwords, and tokens. Specify section and key to get a value.",
    },
    "schema.vault_get.key": {
        "ja": "キー名（例: 'api_key', 'master_password'）",
        "en": "Key name (e.g. 'api_key', 'master_password')",
    },
    "schema.vault_get.section": {
        "ja": "セクション名（例: 'shared', 'bitwarden', 'bank'）",
        "en": "Section name (e.g. 'shared', 'bitwarden', 'bank')",
    },
    "schema.vault_list.desc": {
        "ja": "暗号化されたクレデンシャルvaultのセクション・キー一覧を表示する。値は表示されない（セクション名とキー名のみ）。",
        "en": "List sections and keys in the encrypted credential vault. Values are not shown (section and key names only).",
    },
    "schema.vault_list.section": {
        "ja": "セクション名（省略時は全セクション一覧）",
        "en": "Section name (omit to list all sections)",
    },
    "schema.vault_store.desc": {
        "ja": "暗号化されたクレデンシャルvaultに値を保存する。APIキー、パスワード、トークンなどの秘密情報を暗号化して保管する。",
        "en": "Store a value in the encrypted credential vault. Encrypts and stores secrets such as API keys, passwords, and tokens.",
    },
    "schema.vault_store.key": {
        "ja": "キー名（例: 'api_key', 'master_password'）",
        "en": "Key name (e.g. 'api_key', 'master_password')",
    },
    "schema.vault_store.section": {
        "ja": "セクション名（例: 'shared', 'bitwarden', 'bank'）",
        "en": "Section name (e.g. 'shared', 'bitwarden', 'bank')",
    },
    "schema.vault_store.value": {
        "ja": "保存する値（暗号化されて保存される）",
        "en": "Value to store (will be encrypted)",
    },
    # ── session.* (i18n) ──────────────────────────────
    "session.caution_continue": {
        "ja": "中断前の作業の続きを実行してください",
        "en": "Continue the work from before the interruption",
    },
    "session.caution_header": {"ja": "## 注意", "en": "## Caution"},
    "session.caution_no_repeat": {
        "ja": "完了済みステップを繰り返さないでください",
        "en": "Do not repeat completed steps",
    },
    "session.caution_skip_existing": {
        "ja": "ファイルが既に存在する場合はスキップまたは更新してください",
        "en": "If files already exist, skip or update them",
    },
    "session.completed_none": {"ja": "(なし)", "en": "(none)"},
    "session.completed_steps_header": {"ja": "## 完了済みステップ", "en": "## Completed Steps"},
    "session.continuation_intro": {
        "ja": ("あなたは以下のタスクを実行中でしたが、通信エラーで中断されました。\n続きから実行してください。"),
        "en": (
            "You were executing the following task but it was interrupted by a communication error.\n"
            "Please continue from where you left off."
        ),
    },
    "session.original_instruction_header": {"ja": "## 元の指示", "en": "## Original Instruction"},
    "session.output_so_far_header": {"ja": "## これまでの出力", "en": "## Output So Far"},
    "session.text_truncated": {"ja": "...(前半省略)...", "en": "...(earlier omitted)..."},
    # ── skill.* (i18n) ──────────────────────────────
    "skill.context_header": {"ja": "## コンテキスト", "en": "## Context"},
    "skill.desc_line1": {
        "ja": "スキル・手順書をオンデマンドでロードする。",
        "en": "Load skills and procedures on demand.",
    },
    "skill.desc_line2": {
        "ja": "スキルを発動すると、詳細な手順がこのツールのレスポンスとして提供される。",
        "en": "When activated, detailed instructions are provided in the tool response.",
    },
    "skill.desc_line3": {
        "ja": "該当するスキルがある場合に使用すること。",
        "en": "Use when a matching skill is available.",
    },
    "skill.label_common": {"ja": "共通", "en": "common"},
    "skill.label_procedure": {"ja": "手順", "en": "procedure"},
    "skill.not_found": {
        "ja": ("スキル '{skill_name}' が見つかりません。\n利用可能なスキル: {available}"),
        "en": ("Skill '{skill_name}' not found.\nAvailable skills: {available}"),
    },
    "skill.tool_constraint_desc": {
        "ja": "このスキルの実行中は以下のツールのみ使用してください:",
        "en": "Only use the following tools while executing this skill:",
    },
    "skill.tool_constraint_header": {"ja": "## ツール制約", "en": "## Tool Constraints"},
    "skill.truncated": {"ja": "(以降省略)", "en": "(truncated)"},
    "skill.type_common": {"ja": "共通", "en": "common"},
    "skill.type_personal": {"ja": "個人", "en": "personal"},
    "skill.type_procedure": {"ja": "手順", "en": "procedure"},
    # ── voice/session.py ──
    "voice.mode_suffix": {
        "ja": (
            "\n\n[voice-mode: 音声会話です。話し言葉で200文字以内で簡潔に回答してください。"
            "Markdown記法（見出し・太字・リスト・コードブロック等）は使わないでください]"
        ),
        "en": (
            "\n\n[voice-mode: This is a voice conversation. Reply concisely in spoken language, "
            "200 characters or fewer. Do not use Markdown formatting (headings, bold, lists, code blocks, etc.)]"
        ),
    },
    # ── supervisor/manager.py ──
    "supervisor.zombie_reaped": {
        "ja": "zombie reaper: {count}個の子プロセスを回収しました",
        "en": "zombie reaper: reaped {count} child process(es)",
    },
    # ── supervisor/scheduler_manager.py ──
    "scheduler.cron_fallback_description": {
        "ja": "cron.mdの「{task_name}」の指示に従って処理してください。",
        "en": "Follow the instructions for '{task_name}' in cron.md.",
    },
    # ── _anima_inbox.py ──
    "inbox.reply_placeholder": {"ja": "{返信内容}", "en": "{reply_content}"},
    # ── cascade_limiter.py ──
    "cascade.activity_read_failed": {
        "ja": "GlobalOutboundLimitExceeded: アクティビティログ読み取り失敗のため送信をブロックしました",
        "en": "GlobalOutboundLimitExceeded: Sending blocked because the activity log could not be read",
    },
    "cascade.hourly_reset_at": {
        "ja": " 次の送信可能時刻（目安）: {reset_time}",
        "en": " Estimated next send time: {reset_time}",
    },
    "cascade.hourly_limit": {
        "ja": (
            "GlobalOutboundLimitExceeded: 1時間あたりの送信上限"
            "（{max_per_hour}通）に到達しています"
            "（現在{hourly_count}通/1h, {daily_count}通/24h）。"
            "{reset_at}"
            " このターンではsend_messageを使わず、送信内容を"
            "current_task.mdに記録して次のセッションで送信してください。"
        ),
        "en": (
            "GlobalOutboundLimitExceeded: Hourly send limit "
            "({max_per_hour} messages) reached "
            "({hourly_count} msgs/1h, {daily_count} msgs/24h).{reset_at}"
            " Do not use send_message this turn. Record the message content in "
            "current_task.md and send it in the next session."
        ),
    },
    "cascade.daily_limit": {
        "ja": (
            "GlobalOutboundLimitExceeded: 24時間あたりの送信上限"
            "（{max_per_day}通）に到達しています"
            "（現在{daily_count}通/24h）。"
            " このターンではsend_messageを使わず、送信内容を"
            "current_task.mdに記録して次のセッションで送信してください。"
        ),
        "en": (
            "GlobalOutboundLimitExceeded: Daily send limit "
            "({max_per_day} messages) reached "
            "({daily_count} msgs/24h)."
            " Do not use send_message this turn. Record the message content in "
            "current_task.md and send it in the next session."
        ),
    },
    # ── settings (activity level) ──
    "settings.activity_level.title": {
        "ja": "アクティビティレベル",
        "en": "Activity Level",
    },
    "settings.activity_level.desc": {
        "ja": "Heartbeatの実行頻度と思考深度を調整します（10%〜400%）。100%が通常、低いほど省エネ、高いほど高頻度。",
        "en": "Adjust heartbeat frequency and thinking depth (10%-400%). 100% is normal; lower saves cost, higher increases frequency.",
    },
    "settings.activity_level.updated": {
        "ja": "アクティビティレベルを {level}% に変更しました",
        "en": "Activity level changed to {level}%",
    },
    # ── tooling/skill_creator.py ──
    "skill_creator.invalid_name": {
        "ja": "無効なスキル名: '{skill_name}'（パス区切り文字は使用不可）",
        "en": "Invalid skill name: '{skill_name}' (path separators are not allowed)",
    },
    "skill_creator.created": {
        "ja": "スキル '{skill_name}' を作成しました: {skill_dir}\n作成ファイル: {files_str}",
        "en": "Skill '{skill_name}' created: {skill_dir}\nCreated files: {files_str}",
    },
    # ── tools/notion.py ──
    "notion.config_error": {
        "ja": "ツール 'notion' には認証情報が必要です。vault.json または shared/credentials.json に NOTION_API_TOKEN を設定してください",
        "en": "Tool 'notion' requires credential. Set NOTION_API_TOKEN in vault.json or shared/credentials.json",
    },
    "notion.rate_limited": {
        "ja": "Notion API がレート制限されています",
        "en": "Notion API rate limited",
    },
    "notion.server_error": {
        "ja": "Notion API サーバーエラー {status}: {body}",
        "en": "Notion API server error {status}: {body}",
    },
    "notion.payload_too_large": {
        "ja": "ペイロードが {max_bytes} バイトを超えています（実際: {actual_bytes}）",
        "en": "Payload exceeds {max_bytes} bytes (actual: {actual_bytes})",
    },
    "notion.unknown_action": {
        "ja": "不明な notion アクション: {name}",
        "en": "Unknown notion action: {name}",
    },
    "notion.cli_desc": {
        "ja": "Notion CLI（AnimaWorks 連携）",
        "en": "Notion CLI (AnimaWorks integration)",
    },
    "notion.page_id_required": {
        "ja": "page_id は必須です",
        "en": "page_id is required",
    },
    "notion.database_id_required": {
        "ja": "database_id は必須です",
        "en": "database_id is required",
    },
    "notion.parent_required": {
        "ja": "parent_page_id または parent_database_id のいずれかが必須です",
        "en": "Either parent_page_id or parent_database_id is required",
    },
    "notion.parent_page_id_required": {
        "ja": "parent_page_id は必須です",
        "en": "parent_page_id is required",
    },
    # ── activity_report ──
    "activity_report.invalid_date": {
        "ja": "日付の形式が不正です（YYYY-MM-DD）",
        "en": "Invalid date format (YYYY-MM-DD)",
    },
    "activity_report.future_date": {
        "ja": "未来の日付は指定できません",
        "en": "Future dates are not allowed",
    },
    "activity_report.invalid_model": {
        "ja": "指定されたモデルは利用できません",
        "en": "The specified model is not available",
    },
    "activity_report.not_found": {
        "ja": "この日付のレポートはキャッシュされていません",
        "en": "No cached report found for this date",
    },
    "activity_report.llm_system_prompt": {
        "ja": (
            "あなたは組織活動レポーターです。\n"
            "以下の「組織タイムライン」を読み、1日の活動をストーリー仕立てのMarkdownレポートにまとめてください。\n\n"
            "タイムラインのフォーマット:\n"
            "- [HH:MM] 名前 アイコン イベント種別 の形式で時系列に並んでいます\n"
            "- 末尾にツール使用サマリーと統計があります\n\n"
            "要件:\n"
            "- 見出し: 日付 + 組織活動レポート\n"
            "- ハイライト: 最も重要な成果を3-5個\n"
            "- ストーリー形式: 時系列に沿って「誰が」「何をして」「どうなったか」を自然な文章で記述\n"
            "- 関連する出来事（指示→実行→完了など）を因果関係でつなげて読みやすくする\n"
            "- エラーや問題があれば「課題・注意事項」セクションに記載\n"
            "- 末尾の統計データを引用して全体像を補足\n"
            "- 日本語で出力"
        ),
        "en": (
            "You are an organisational activity reporter.\n"
            "Read the 'Org Timeline' below and produce a narrative-style Markdown report of the day's activities.\n\n"
            "Timeline format:\n"
            "- Each line is [HH:MM] name icon event_type, sorted chronologically\n"
            "- Tool usage summary and statistics are at the bottom\n\n"
            "Requirements:\n"
            "- Heading: Date + Organisation Activity Report\n"
            "- Highlights: 3-5 most important outcomes\n"
            '- Narrative style: describe chronologically "who" "did what" "with what result" in natural prose\n'
            "- Connect related events (instruction → execution → completion) with causal links for readability\n"
            '- Include an "Issues & Notes" section for errors or problems\n'
            "- Cite the statistics at the bottom to provide the big picture\n"
            "- Output in English"
        ),
    },
    "activity_report.llm_user_prompt": {
        "ja": "以下の組織タイムラインに基づいて活動レポートを生成してください。\n\n{data}",
        "en": "Generate an activity report based on the following org timeline.\n\n{data}",
    },
    # ── routes/brainstorm.py ──
    "brainstorm.char.realist": {"ja": "現実派", "en": "Realist"},
    "brainstorm.char.realist.desc": {
        "ja": "収益・ROI・コスト・リスクの視点から分析します",
        "en": "Analyzes from revenue, ROI, cost, and risk perspectives",
    },
    "brainstorm.char.realist.prompt": {
        "ja": (
            "あなたは「現実派」のブレスト参加者です。\n"
            "常に収益性・ROI・コスト・リスクの観点から提案を行います。\n"
            "理想論よりも実行可能性と投資対効果を重視してください。\n"
            "提案は具体的な数値やスケジュール感を含めてください。\n"
            "出力はMarkdown形式で、見出し・箇条書きを使って構造化してください。"
        ),
        "en": (
            "You are a 'Realist' brainstorm participant.\n"
            "Always propose from revenue, ROI, cost, and risk perspectives.\n"
            "Prioritize feasibility and return on investment over ideals.\n"
            "Include specific numbers and timelines in your proposals.\n"
            "Output in Markdown with headings and bullet points."
        ),
    },
    "brainstorm.char.challenger": {"ja": "挑戦派", "en": "Challenger"},
    "brainstorm.char.challenger.desc": {
        "ja": "革新・成長・破壊的イノベーションの視点から提案します",
        "en": "Proposes from innovation, growth, and disruptive perspectives",
    },
    "brainstorm.char.challenger.prompt": {
        "ja": (
            "あなたは「挑戦派」のブレスト参加者です。\n"
            "既存の枠組みにとらわれず、革新的・破壊的なアイデアを提案します。\n"
            "「なぜ今までのやり方を変えないのか？」という視点で切り込んでください。\n"
            "大胆な提案を恐れず、成長機会を最大化する方向で考えてください。\n"
            "出力はMarkdown形式で、見出し・箇条書きを使って構造化してください。"
        ),
        "en": (
            "You are a 'Challenger' brainstorm participant.\n"
            "Propose innovative and disruptive ideas beyond existing frameworks.\n"
            "Challenge the status quo: 'Why not change the current approach?'\n"
            "Be bold and maximize growth opportunities.\n"
            "Output in Markdown with headings and bullet points."
        ),
    },
    "brainstorm.char.customer": {"ja": "顧客視点", "en": "Customer Advocate"},
    "brainstorm.char.customer.desc": {
        "ja": "UX・ユーザー体験・顧客価値の視点から評価します",
        "en": "Evaluates from UX, user experience, and customer value perspectives",
    },
    "brainstorm.char.customer.prompt": {
        "ja": (
            "あなたは「顧客視点」のブレスト参加者です。\n"
            "常にエンドユーザーの体験・満足度・価値を最優先に考えます。\n"
            "「ユーザーはこれを使って何が嬉しいのか？」を軸に提案してください。\n"
            "ペルソナやユースケースを具体的に描写し、UXの改善点を指摘してください。\n"
            "出力はMarkdown形式で、見出し・箇条書きを使って構造化してください。"
        ),
        "en": (
            "You are a 'Customer Advocate' brainstorm participant.\n"
            "Always prioritize end-user experience, satisfaction, and value.\n"
            "Center proposals around 'What makes users happy about this?'\n"
            "Describe specific personas and use cases, and point out UX improvements.\n"
            "Output in Markdown with headings and bullet points."
        ),
    },
    "brainstorm.char.engineer": {"ja": "実装視点", "en": "Technical Implementer"},
    "brainstorm.char.engineer.desc": {
        "ja": "技術的実現性・アーキテクチャ・運用負荷の視点から検討します",
        "en": "Examines from technical feasibility, architecture, and operational load perspectives",
    },
    "brainstorm.char.engineer.prompt": {
        "ja": (
            "あなたは「実装視点」のブレスト参加者です。\n"
            "技術的な実現可能性・アーキテクチャ・運用負荷・スケーラビリティを重視します。\n"
            "「これは技術的にどう実装するか？」「運用でどこがボトルネックか？」を分析してください。\n"
            "技術スタック・工数見積・技術的リスクを具体的に示してください。\n"
            "出力はMarkdown形式で、見出し・箇条書きを使って構造化してください。"
        ),
        "en": (
            "You are a 'Technical Implementer' brainstorm participant.\n"
            "Focus on technical feasibility, architecture, operational load, and scalability.\n"
            "Analyze 'How to implement this technically?' and 'Where are operational bottlenecks?'\n"
            "Provide specific tech stacks, effort estimates, and technical risks.\n"
            "Output in Markdown with headings and bullet points."
        ),
    },
    "brainstorm.user_prompt": {
        "ja": (
            "以下のテーマについて、あなたの視点から提案・分析してください。\n\n"
            "## テーマ\n{theme}\n\n"
            "## 制約条件\n{constraints}\n\n"
            "## 期待するアウトプット\n{expected_output}"
        ),
        "en": (
            "Provide your analysis and proposals on the following theme.\n\n"
            "## Theme\n{theme}\n\n"
            "## Constraints\n{constraints}\n\n"
            "## Expected Output\n{expected_output}"
        ),
    },
    "brainstorm.no_constraints": {"ja": "特になし", "en": "None specified"},
    "brainstorm.no_expected_output": {"ja": "特になし", "en": "None specified"},
    "brainstorm.synthesizer_prompt": {
        "ja": (
            "あなたはブレストの統合者です。\n"
            "複数の視点からの提案を受け取り、以下の規定フォーマットで整理・統合してください。\n\n"
            "## 出力フォーマット（必ずこの5セクション構成で出力すること）\n"
            "### 論点\n主要な論点・議論ポイントを箇条書きで列挙\n\n"
            "### 案\n各視点からの主要な提案をまとめる\n\n"
            "### 比較\n提案の比較表（Markdownテーブル形式。軸: 実現性/コスト/インパクト/リスク）\n\n"
            "### 推奨案\n総合的に最も推奨される案とその理由\n\n"
            "### 次アクション\n具体的な次のステップを箇条書きで列挙"
        ),
        "en": (
            "You are a brainstorm synthesizer.\n"
            "Receive proposals from multiple perspectives and organize them into the following format.\n\n"
            "## Output Format (must include all 5 sections)\n"
            "### Key Issues\nList main discussion points as bullet points\n\n"
            "### Proposals\nSummarize main proposals from each perspective\n\n"
            "### Comparison\nComparison table in Markdown (axes: Feasibility/Cost/Impact/Risk)\n\n"
            "### Recommendation\nThe overall recommended proposal and reasoning\n\n"
            "### Next Actions\nList concrete next steps as bullet points"
        ),
    },
    "brainstorm.synthesizer_user_prompt": {
        "ja": ("テーマ「{theme}」について、以下の各視点からの提案を統合してください。\n\n{proposals}"),
        "en": ('Synthesize the following proposals from multiple perspectives on the theme: "{theme}"\n\n{proposals}'),
    },
    "brainstorm.no_characters_selected": {
        "ja": "キャラクターが選択されていません",
        "en": "No characters selected",
    },
    "brainstorm.no_model_configured": {
        "ja": "LLMモデルが設定されていません",
        "en": "No LLM model configured",
    },
    "brainstorm.invalid_model": {
        "ja": "無効なモデルが指定されました",
        "en": "Invalid model specified",
    },
    # ── routes/team_presets.py ──
    "preset.industry.saas": {"ja": "SaaS", "en": "SaaS"},
    "preset.industry.consulting": {"ja": "受託開発・コンサルティング", "en": "Consulting"},
    "preset.industry.ec": {"ja": "EC・物販", "en": "E-Commerce"},
    "preset.industry.general": {"ja": "一般・その他", "en": "General"},
    "preset.purpose.new_development": {"ja": "新規開発", "en": "New Development"},
    "preset.purpose.operations": {"ja": "運用改善", "en": "Operations"},
    "preset.purpose.research": {"ja": "調査・リサーチ", "en": "Research"},
    "preset.saas_dev": {"ja": "SaaS新規開発チーム", "en": "SaaS Development Team"},
    "preset.saas_dev.desc": {
        "ja": "SaaS製品の企画・開発・ローンチに最適なチーム構成",
        "en": "Optimal team for SaaS product planning, development, and launch",
    },
    "preset.saas_ops": {"ja": "SaaS運用チーム", "en": "SaaS Operations Team"},
    "preset.saas_ops.desc": {
        "ja": "SaaS製品のカスタマーサポート・運用を担うチーム",
        "en": "Team for SaaS customer support and operations",
    },
    "preset.consulting_dev": {"ja": "受託開発プロジェクトチーム", "en": "Consulting Project Team"},
    "preset.consulting_dev.desc": {
        "ja": "受託案件の提案・設計・推進に最適なチーム",
        "en": "Team for consulting project proposal, design, and execution",
    },
    "preset.consulting_research": {"ja": "リサーチ・分析チーム", "en": "Research & Analysis Team"},
    "preset.consulting_research.desc": {
        "ja": "市場調査・データ分析・レポート作成を担うチーム",
        "en": "Team for market research, data analysis, and reporting",
    },
    "preset.ec_dev": {"ja": "EC新規立ち上げチーム", "en": "E-Commerce Launch Team"},
    "preset.ec_dev.desc": {
        "ja": "ECサイトの立ち上げ・商品準備・販売戦略を担うチーム",
        "en": "Team for e-commerce launch, product setup, and sales strategy",
    },
    "preset.ec_ops": {"ja": "EC運用チーム", "en": "E-Commerce Operations Team"},
    "preset.ec_ops.desc": {
        "ja": "受注処理・在庫管理・カスタマー対応の運用チーム",
        "en": "Team for order processing, inventory management, and customer support",
    },
    "preset.general_dev": {"ja": "プロジェクト推進チーム", "en": "Project Execution Team"},
    "preset.general_dev.desc": {
        "ja": "業種を問わない汎用的なプロジェクト推進チーム",
        "en": "General-purpose project execution team",
    },
    "preset.general_ops": {"ja": "バックオフィスチーム", "en": "Back Office Team"},
    "preset.general_ops.desc": {
        "ja": "総務・経理・事務を中心としたバックオフィス運用チーム",
        "en": "Back office operations team for admin, accounting, and general affairs",
    },
    "preset.task.market_research": {"ja": "市場調査の実施", "en": "Conduct market research"},
    "preset.task.competitor_analysis": {"ja": "競合分析レポート作成", "en": "Create competitor analysis report"},
    "preset.task.roadmap_draft": {"ja": "プロダクトロードマップ策定", "en": "Draft product roadmap"},
    "preset.task.user_persona": {"ja": "ユーザーペルソナ定義", "en": "Define user personas"},
    "preset.task.onboarding_flow": {"ja": "オンボーディングフロー設計", "en": "Design onboarding flow"},
    "preset.task.kpi_setup": {"ja": "KPI設定・ダッシュボード構築", "en": "Set up KPIs and dashboard"},
    "preset.task.release_plan": {"ja": "リリース計画策定", "en": "Create release plan"},
    "preset.task.faq_draft": {"ja": "FAQ・ヘルプドキュメント作成", "en": "Draft FAQ and help documents"},
    "preset.task.ticket_triage": {"ja": "チケットトリアージ体制構築", "en": "Build ticket triage system"},
    "preset.task.sla_monitor": {"ja": "SLA監視・レポート体制", "en": "Set up SLA monitoring and reporting"},
    "preset.task.report_weekly": {"ja": "週次レポート作成", "en": "Create weekly report"},
    "preset.task.schedule_mgmt": {"ja": "スケジュール・カレンダー管理", "en": "Manage schedule and calendar"},
    "preset.task.knowledge_base": {"ja": "ナレッジベース整備", "en": "Build knowledge base"},
    "preset.task.escalation_flow": {"ja": "エスカレーションフロー定義", "en": "Define escalation flow"},
    "preset.task.metrics_dashboard": {"ja": "メトリクスダッシュボード構築", "en": "Build metrics dashboard"},
    "preset.task.meeting_minutes": {"ja": "議事録テンプレート整備", "en": "Prepare meeting minutes template"},
    "preset.task.client_analysis": {"ja": "クライアント要件分析", "en": "Analyze client requirements"},
    "preset.task.proposal_draft": {"ja": "提案書ドラフト作成", "en": "Draft proposal document"},
    "preset.task.scope_definition": {"ja": "スコープ定義書作成", "en": "Create scope definition"},
    "preset.task.resource_plan": {"ja": "リソース計画策定", "en": "Plan resource allocation"},
    "preset.task.industry_report": {"ja": "業界動向レポート作成", "en": "Create industry trend report"},
    "preset.task.timeline_draft": {"ja": "プロジェクトタイムライン作成", "en": "Draft project timeline"},
    "preset.task.risk_assessment": {"ja": "リスクアセスメント実施", "en": "Conduct risk assessment"},
    "preset.task.kickoff_prep": {"ja": "キックオフ準備", "en": "Prepare project kickoff"},
    "preset.task.literature_review": {"ja": "文献レビュー・先行調査", "en": "Conduct literature review"},
    "preset.task.data_collection": {"ja": "データ収集・整理", "en": "Collect and organize data"},
    "preset.task.analysis_framework": {"ja": "分析フレームワーク設計", "en": "Design analysis framework"},
    "preset.task.findings_report": {"ja": "調査結果レポート作成", "en": "Create findings report"},
    "preset.task.stakeholder_briefing": {"ja": "ステークホルダー向けブリーフィング", "en": "Stakeholder briefing"},
    "preset.task.trend_mapping": {"ja": "トレンドマッピング", "en": "Trend mapping"},
    "preset.task.benchmark_study": {"ja": "ベンチマーク調査", "en": "Conduct benchmark study"},
    "preset.task.exec_summary": {"ja": "エグゼクティブサマリー作成", "en": "Create executive summary"},
    "preset.task.product_catalog": {"ja": "商品カタログ作成", "en": "Create product catalog"},
    "preset.task.pricing_strategy": {"ja": "価格戦略策定", "en": "Define pricing strategy"},
    "preset.task.shipping_policy": {"ja": "配送ポリシー策定", "en": "Define shipping policy"},
    "preset.task.cs_playbook": {"ja": "CS対応マニュアル作成", "en": "Create CS playbook"},
    "preset.task.launch_checklist": {"ja": "ローンチチェックリスト作成", "en": "Create launch checklist"},
    "preset.task.promotion_plan": {"ja": "プロモーション計画策定", "en": "Plan promotions"},
    "preset.task.return_policy": {"ja": "返品ポリシー策定", "en": "Define return policy"},
    "preset.task.seo_plan": {"ja": "SEO施策計画", "en": "Plan SEO strategy"},
    "preset.task.order_processing": {"ja": "受注処理フロー構築", "en": "Build order processing flow"},
    "preset.task.inventory_check": {"ja": "在庫確認・補充管理", "en": "Inventory check and replenishment"},
    "preset.task.customer_inquiry": {"ja": "顧客問い合わせ対応", "en": "Handle customer inquiries"},
    "preset.task.review_response": {"ja": "レビュー返信対応", "en": "Respond to reviews"},
    "preset.task.sales_report": {"ja": "売上レポート作成", "en": "Create sales report"},
    "preset.task.refund_process": {"ja": "返金処理フロー整備", "en": "Set up refund process"},
    "preset.task.supplier_coord": {"ja": "仕入先連携管理", "en": "Manage supplier coordination"},
    "preset.task.tax_filing": {"ja": "税務申告準備", "en": "Prepare tax filing"},
    "preset.task.project_charter": {"ja": "プロジェクト憲章作成", "en": "Create project charter"},
    "preset.task.stakeholder_map": {"ja": "ステークホルダーマップ作成", "en": "Create stakeholder map"},
    "preset.task.meeting_setup": {"ja": "定例会議セットアップ", "en": "Set up recurring meetings"},
    "preset.task.background_research": {"ja": "背景調査・情報収集", "en": "Background research"},
    "preset.task.milestone_plan": {"ja": "マイルストーン計画策定", "en": "Plan milestones"},
    "preset.task.comm_plan": {"ja": "コミュニケーション計画策定", "en": "Create communication plan"},
    "preset.task.feasibility_study": {"ja": "実現性調査", "en": "Feasibility study"},
    "preset.task.budget_draft": {"ja": "予算案作成", "en": "Draft budget"},
    "preset.task.daily_ops_check": {"ja": "日次業務チェック", "en": "Daily operations check"},
    "preset.task.document_mgmt": {"ja": "文書管理・整理", "en": "Document management"},
    "preset.task.expense_tracking": {"ja": "経費管理・精算", "en": "Expense tracking"},
    "preset.task.vendor_mgmt": {"ja": "取引先管理", "en": "Vendor management"},
    "preset.task.calendar_coord": {"ja": "カレンダー・予定調整", "en": "Calendar coordination"},
    "preset.task.monthly_close": {"ja": "月次決算準備", "en": "Monthly closing preparation"},
    "preset.task.process_improvement": {"ja": "業務プロセス改善", "en": "Process improvement"},
    "preset.task.compliance_check": {"ja": "コンプライアンスチェック", "en": "Compliance check"},
}


class _SafeFormatDict(dict):
    """Dict that returns ``{key}`` for missing keys during format_map."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def t(key: str, locale: str | None = None, **kwargs: object) -> str:
    """Get localized string with optional format args.

    Args:
        key: Dot-separated key (e.g. "handler.not_subordinate").
        locale: Override locale. If None, uses config.locale.
        **kwargs: Values to substitute into {placeholder} in the template.

    Returns:
        Localized string. Falls back to en, then ja, then key if not found.
    """
    from core.paths import _get_locale

    loc = locale or _get_locale()
    if not isinstance(loc, str) or loc not in ("ja", "en"):
        loc = "ja"
    entry = _STRINGS.get(key, {})
    template = entry.get(loc) or entry.get("en") or entry.get("ja", key)
    if kwargs:
        return template.format_map(_SafeFormatDict({k: str(v) for k, v in kwargs.items()}))
    return template
