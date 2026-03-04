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
    "handler.skill_frontmatter_required": {"ja": "スキルファイルにはYAMLフロントマター(---)が必要です。", "en": "Skill files require YAML frontmatter (---)."},
    "handler.name_field_required": {"ja": "`name` フィールドが必要です。", "en": "The `name` field is required."},
    "handler.description_field_required": {"ja": "`description` フィールドが必要です。", "en": "The `description` field is required."},
    "handler.description_keyword_warning": {"ja": "descriptionに「」キーワードがありません。自動マッチング精度が低下する可能性があります。", "en": "No 「」keywords in description. Auto-matching accuracy may be reduced."},
    "handler.legacy_skill_sections": {"ja": "旧形式のセクション(## 概要 / ## 発動条件)が検出されました。Claude Code形式(YAMLフロントマター)への移行を推奨します。", "en": "Legacy sections (## 概要 / ## 発動条件) detected. Migration to Claude Code format (YAML frontmatter) is recommended."},
    "handler.procedure_frontmatter_recommended": {"ja": "手順書ファイルにはYAMLフロントマター(---)を推奨します。description フィールドで自動マッチングが有効になります。", "en": "Procedure files should have YAML frontmatter (---). The description field enables auto-matching."},
    "handler.procedure_frontmatter_recommended_short": {"ja": "手順書ファイルにはYAMLフロントマター(---)を推奨します。", "en": "Procedure files should have YAML frontmatter (---)."},
    "handler.procedure_description_missing": {"ja": "`description` フィールドがありません。自動マッチングを有効にするために description を追加してください。", "en": "The `description` field is missing. Add description to enable auto-matching."},
    "handler.background_task_started": {"ja": "タスクをバックグラウンドで実行開始しました (task_id: {task_id})", "en": "Task started in background (task_id: {task_id})"},
    "handler.output_truncated": {"ja": "[出力が500KBを超えたためトランケーションしました。元のサイズ: {size}]", "en": "[Output truncated because it exceeded 500KB. Original size: {size}]"},
    "handler.activity_recent_items": {"ja": "最新{limit}件を確認", "en": "Checked latest {limit} items"},
    "handler.activity_dm_history": {"ja": "DM履歴を確認", "en": "Checked DM history"},
    "handler.tool_creation_denied": {"ja": "ツール作成が許可されていません。permissions.md に「ツール作成」セクションを追加してください。", "en": "Tool creation is not permitted. Add a tool creation section to permissions.md."},
    "handler.skill_format_validation": {"ja": "⚠️ スキルフォーマット検証:\n{msg}", "en": "⚠️ Skill format validation:\n{msg}"},
    "handler.procedure_format_validation": {"ja": "⚠️ 手順書フォーマット検証:\n{msg}", "en": "⚠️ Procedure format validation:\n{msg}"},
    "handler.dm_intent_error": {"ja": "Error: DMのintentは 'report', 'delegation', 'question' のみ許可されています。acknowledgment・感謝・FYIはBoardを使用してください（post_channel ツール）。", "en": "Error: DM intent must be 'report', 'delegation', or 'question' only. Use Board (post_channel tool) for acknowledgments, thanks, or FYI."},
    "handler.dm_already_sent": {"ja": "Error: このrunで既に {to} にメッセージを送信済みです。追加の連絡はBoardを使用してください。", "en": "Error: Message already sent to {to} in this run. Use Board for additional communication."},
    "handler.dm_max_recipients": {"ja": "Error: 1回のrunでDMを送れるのは最大2人までです。3人以上への伝達はBoardを使用してください（post_channel ツール）。", "en": "Error: Maximum 2 DM recipients per run. Use Board (post_channel tool) for 3+ recipients."},
    "handler.post_alt_hint": {"ja": " 別のチャネル（{channels}）への投稿、またはsend_message（intent: question/report）は可能です。", "en": " You can post to another channel ({channels}) or use send_message (intent: question/report)."},
    "handler.post_already_posted": {"ja": "Error: このrunで既に #{channel} に投稿済みです。同一チャネルへの連投はできません。{alt_hint}", "en": "Error: Already posted to #{channel} in this run. No duplicate posts to the same channel.{alt_hint}"},
    "handler.post_cooldown": {"ja": "Error: #{channel} には {ts} に投稿済みです（{elapsed}秒前）。クールダウン {cooldown}秒が必要です。", "en": "Error: Already posted to #{channel} at {ts} ({elapsed}s ago). Cooldown of {cooldown}s required."},
    "handler.board_mention_content": {"ja": "{from_name}さんがボード #{channel} であなたをメンションしました:\n\n{text}\n\n返信するには post_channel(channel=\"{channel}\", text=\"返信内容\") を使ってください。", "en": "{from_name} mentioned you on board #{channel}:\n\n{text}\n\nTo reply, use post_channel(channel=\"{channel}\", text=\"your reply\")."},
    "handler.channel_acl_denied": {"ja": "Error: #{channel} へのアクセス権がありません。manage_channel(action=\"info\", channel=\"{channel}\") でメンバーを確認してください。", "en": "Error: You do not have access to #{channel}. Use manage_channel(action=\"info\", channel=\"{channel}\") to check members."},
    "handler.channel_created": {"ja": "チャネル #{channel} を作成しました（メンバー: {members}）", "en": "Channel #{channel} created (members: {members})"},
    "handler.channel_already_exists": {"ja": "Error: チャネル #{channel} は既に存在します", "en": "Error: Channel #{channel} already exists"},
    "handler.channel_members_added": {"ja": "#{channel} にメンバーを追加しました: {members}", "en": "Added members to #{channel}: {members}"},
    "handler.channel_members_removed": {"ja": "#{channel} からメンバーを削除しました: {members}", "en": "Removed members from #{channel}: {members}"},
    "handler.channel_not_found": {"ja": "Error: チャネル #{channel} が見つかりません", "en": "Error: Channel #{channel} not found"},
    "handler.channel_open": {"ja": "#{channel} はオープンチャネルです（全Animaがアクセス可能）", "en": "#{channel} is an open channel (all Animas can access)"},
    "handler.channel_acl_not_member": {"ja": "Error: #{channel} のメンバーではないため、メンバー管理操作はできません。", "en": "Error: You are not a member of #{channel} and cannot manage its membership."},
    "handler.channel_add_member_open_denied": {"ja": "Error: #{channel} はオープンチャネルです。add_memberするには、まず manage_channel(action=\"create\") で制限チャネルとして再作成してください。", "en": "Error: #{channel} is an open channel. To add members, first recreate it as a restricted channel with manage_channel(action=\"create\")."},
    "handler.self_operation_denied": {"ja": "自分自身を操作することはできません", "en": "You cannot operate on yourself"},
    "handler.config_load_failed": {"ja": "設定読み込みに失敗: {e}", "en": "Failed to load config: {e}"},
    "handler.anima_not_found": {"ja": "Anima '{target_name}' は存在しません", "en": "Anima '{target_name}' does not exist"},
    "handler.not_direct_subordinate": {"ja": "'{target_name}' はあなたの直属部下ではありません", "en": "'{target_name}' is not your direct subordinate"},
    "handler.not_descendant": {"ja": "'{target_name}' はあなたの配下ではありません", "en": "'{target_name}' is not under your supervision"},
    "handler.already_disabled": {"ja": "'{target_name}' は既に休止中です", "en": "'{target_name}' is already disabled"},
    "handler.disable_log_summary": {"ja": "{target_name} を休止", "en": "Disabling {target_name}"},
    "handler.disable_reason": {"ja": " (理由: {reason})", "en": " (reason: {reason})"},
    "handler.disabled_success": {"ja": "'{target_name}' を休止にしました。Reconciliation が30秒以内にプロセスを停止します。", "en": "'{target_name}' has been disabled. Reconciliation will stop the process within 30 seconds."},
    "handler.already_enabled": {"ja": "'{target_name}' は既に有効です", "en": "'{target_name}' is already enabled"},
    "handler.enable_log_summary": {"ja": "{target_name} を復帰", "en": "Enabling {target_name}"},
    "handler.enabled_success": {"ja": "'{target_name}' を有効にしました。Reconciliation が30秒以内にプロセスを起動します。", "en": "'{target_name}' has been enabled. Reconciliation will start the process within 30 seconds."},
    "handler.model_warning": {"ja": "警告: '{model}' は既知のモデルカタログに含まれていません。正しいモデル名か確認してください。", "en": "Warning: '{model}' is not in the known model catalog. Please verify the model name."},
    "handler.model_change_log": {"ja": "{target_name} のモデルを {model} に変更", "en": "Changing {target_name}'s model to {model}"},
    "handler.model_changed": {"ja": "'{target_name}' のモデルを '{model}' に変更しました。反映するには restart_subordinate を呼び出してください。", "en": "Changed {target_name}'s model to '{model}'. Call restart_subordinate to apply."},
    "handler.restart_log": {"ja": "{target_name} を再起動リクエスト", "en": "Restart requested for {target_name}"},
    "handler.restart_success": {"ja": "'{target_name}' の再起動をリクエストしました。Reconciliation が 30 秒以内にプロセスを再起動します。", "en": "Restart requested for '{target_name}'. Reconciliation will restart the process within 30 seconds."},
    "handler.no_subordinates": {"ja": "配下の Anima はいません", "en": "No subordinate Animas"},
    "handler.last_activity_none": {"ja": "なし", "en": "None"},
    "handler.last_activity_unknown": {"ja": "不明", "en": "Unknown"},
    "handler.current_task_none": {"ja": "なし", "en": "None"},
    "handler.current_task_unreadable": {"ja": "読取不可", "en": "Unreadable"},
    "handler.org_dashboard_title": {"ja": "## 組織ダッシュボード", "en": "## Organization Dashboard"},
    "handler.dashboard_last": {"ja": "最終: {activity}", "en": "Last: {activity}"},
    "handler.dashboard_tasks": {"ja": "タスク: {count}件", "en": "Tasks: {count}"},
    "handler.dashboard_working_on": {"ja": "作業中: {task}", "en": "Working on: {task}"},
    "handler.dashboard_summary": {"ja": "配下{count}名のダッシュボード表示", "en": "Dashboard for {count} subordinate(s)"},
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
    "handler.subordinate_management": {"ja": "直属部下のcron.md, heartbeat.md, status.json, injection.md", "en": "Direct subordinate's cron.md, heartbeat.md, status.json, injection.md"},
    "handler.subordinate_dir_list": {"ja": "直属部下のディレクトリ一覧", "en": "Direct subordinate directory listing"},
    "handler.descendant_activity": {"ja": "配下のactivity_log", "en": "Descendant activity_log"},
    "handler.descendant_state": {"ja": "配下のstatus.json, identity.md, injection.md, state/, task_queue.jsonl", "en": "Descendant status.json, identity.md, injection.md, state/, task_queue.jsonl"},
    "handler.descendant_pending": {"ja": "配下のstate/pending/", "en": "Descendant state/pending/"},
    "handler.peer_activity": {"ja": "同僚のactivity_log（読み取り専用）", "en": "Peer activity_log (read-only)"},
    "handler.cmd_denied": {"ja": "{cmd} 禁止", "en": "{cmd} blocked"},
    "handler.delegation_dm_content": {"ja": "[タスク委譲]\n{instruction}\n\n期限: {deadline}\nタスクID: {task_id}", "en": "[Task delegation]\n{instruction}\n\nDeadline: {deadline}\nTask ID: {task_id}"},
    "handler.dm_sent": {"ja": "DM送信済み", "en": "DM sent"},
    "handler.dm_send_failed": {"ja": "DM送信失敗: {e}", "en": "DM send failed: {e}"},
    "handler.messenger_not_set": {"ja": "メッセンジャー未設定（タスクキューへの追加は成功）", "en": "Messenger not configured (task added to queue successfully)"},
    "handler.subordinate_disabled_warning": {"ja": "\n⚠️ {target_name} は現在休止中です。タスクはキューに蓄積されますが、処理は再起動後になります。", "en": "\n⚠️ {target_name} is currently disabled. Tasks will queue until restart."},
    "handler.delegation_summary": {"ja": "[委譲] {summary}", "en": "[Delegated] {summary}"},
    "handler.delegate_log": {"ja": "{target_name}にタスク委譲: {summary}", "en": "Delegated task to {target_name}: {summary}"},
    "handler.delegated_success": {"ja": "タスクを {target_name} に委譲しました。\n- 部下側タスクID: {sub_id}\n- 追跡用タスクID: {own_id}\n- {dm_result}", "en": "Task delegated to {target_name}.\n- Subordinate task ID: {sub_id}\n- Tracking task ID: {own_id}\n- {dm_result}"},
    "handler.no_delegated_tasks": {"ja": "委譲済みタスクはありません", "en": "No delegated tasks"},
    "handler.task_tracker_log": {"ja": "委譲タスク追跡 (filter={status}, count={count})", "en": "Task tracker (filter={status}, count={count})"},
    "handler.no_matching_delegated": {"ja": "条件に合う委譲済みタスクはありません (filter={status})", "en": "No delegated tasks matching filter ({status})"},
    "handler.shared_tool_denied": {"ja": "共有ツール作成が許可されていません。", "en": "Shared tool creation is not permitted."},
    "handler.outcome_success": {"ja": "成功", "en": "Success"},
    "handler.outcome_failure": {"ja": "失敗", "en": "Failure"},
    "handler.skill_name_required": {"ja": "skill_name パラメータは必須です。", "en": "skill_name parameter is required."},
    "handler.task_add_log": {"ja": "タスク追加: {summary}", "en": "Task added: {summary}"},
    "handler.task_update_log": {"ja": "タスク更新: {summary} → {status}", "en": "Task updated: {summary} → {status}"},
    "handler.no_file_ops_paths": {"ja": "No allowed paths listed under ファイル操作", "en": "No allowed paths listed under file operations"},
    "handler.none_value": {"ja": "(なし)", "en": "(none)"},
    "handler.reason_prefix": {"ja": "理由: {reason}", "en": "Reason: {reason}"},
    "handler.all_descendants": {"ja": "全配下", "en": "All descendants"},
    "handler.audit_title": {"ja": "## {target_name} 監査レポート", "en": "## {target_name} Audit Report"},
    "handler.audit_period": {"ja": "期間: 直近{days}日間", "en": "Period: Last {days} day(s)"},
    "handler.audit_process_status": {"ja": "プロセス状態: {status}", "en": "Process status: {status}"},
    "handler.audit_model": {"ja": "モデル: {model}", "en": "Model: {model}"},
    "handler.audit_activity_header": {"ja": "### 活動サマリー", "en": "### Activity Summary"},
    "handler.audit_total_events": {"ja": "イベント総数: {count}件", "en": "Total events: {count}"},
    "handler.audit_event_type": {"ja": "- {event_type}: {count}件", "en": "- {event_type}: {count}"},
    "handler.audit_no_activity": {"ja": "(この期間の活動ログはありません)", "en": "(No activity log for this period)"},
    "handler.audit_tasks_header": {"ja": "### タスク状況", "en": "### Task Status"},
    "handler.audit_current_task": {"ja": "進行中タスク: {task}", "en": "Current task: {task}"},
    "handler.audit_active_tasks": {"ja": "アクティブタスク: {count}件", "en": "Active tasks: {count}"},
    "handler.audit_completed_tasks": {"ja": "完了タスク（累計）: {count}件", "en": "Completed tasks (total): {count}"},
    "handler.audit_errors_header": {"ja": "### エラーサマリー", "en": "### Error Summary"},
    "handler.audit_error_count": {"ja": "エラー件数: {count}件", "en": "Error count: {count}"},
    "handler.audit_error_entry": {"ja": "- [{ts}] {summary}", "en": "- [{ts}] {summary}"},
    "handler.audit_no_errors": {"ja": "(エラーなし)", "en": "(No errors)"},
    "handler.audit_tools_header": {"ja": "### ツール使用統計", "en": "### Tool Usage Statistics"},
    "handler.audit_tool_entry": {"ja": "- {tool}: {count}回", "en": "- {tool}: {count} time(s)"},
    "handler.audit_no_tool_use": {"ja": "(ツール使用なし)", "en": "(No tool usage)"},
    "handler.audit_comms_header": {"ja": "### 通信パターン", "en": "### Communication Patterns"},
    "handler.audit_msgs_sent": {"ja": "送信メッセージ: {count}件", "en": "Messages sent: {count}"},
    "handler.audit_msgs_received": {"ja": "受信メッセージ: {count}件", "en": "Messages received: {count}"},
    "handler.audit_comms_peer": {"ja": "- {peer}: 送信{sent}件 / 受信{received}件", "en": "- {peer}: sent {sent} / received {received}"},
    "handler.audit_no_comms": {"ja": "(通信なし)", "en": "(No communications)"},
    "handler.audit_log_summary": {"ja": "{target_name}の監査レポート生成（{days}日間）", "en": "Generated audit report for {target_name} ({days} day(s))"},
    "handler.tool_creation_keyword": {"ja": "ツール作成", "en": "Tool Creation"},
    # ── anima.py ──
    "anima.bg_task_done": {"ja": "バックグラウンドタスク完了: {tool}", "en": "Background task completed: {tool}"},
    "anima.bg_task_failed": {"ja": "バックグラウンドタスク失敗: {tool}", "en": "Background task failed: {tool}"},
    "anima.bg_notif_task_id": {"ja": "- タスクID: {task_id}", "en": "- Task ID: {task_id}"},
    "anima.bg_notif_tool": {"ja": "- ツール: {tool}", "en": "- Tool: {tool}"},
    "anima.bg_notif_status": {"ja": "- ステータス: {status}", "en": "- Status: {status}"},
    "anima.bg_notif_result": {"ja": "- 結果: {summary}", "en": "- Result: {summary}"},
    "anima.bootstrap_prompt": {"ja": "あなたの bootstrap.md ファイルを読み、指示に従ってください。", "en": "Read your bootstrap.md file and follow its instructions."},
    "anima.process_message_error": {"ja": "process_messageエラー: {exc}", "en": "process_message error: {exc}"},
    "anima.agent_error": {"ja": "[ERROR: エージェント実行中にエラーが発生しました]", "en": "[ERROR: An error occurred during agent execution]"},
    "anima.initializing": {"ja": "現在初期化中です。しばらくお待ちください。", "en": "Initializing. Please wait."},
    "anima.response_interrupted": {"ja": "[応答が中断されました]", "en": "[Response was interrupted]"},
    "anima.response_interrupted_prefix": {"ja": "\n[応答が中断されました]", "en": "\n[Response was interrupted]"},
    "anima.status_idle": {"ja": "待機中", "en": "Idle"},
    "anima.task_none": {"ja": "特になし", "en": "None"},
    "anima.visit_desk": {"ja": "[デスクを訪問]", "en": "[Desk visit]"},
    "anima.greeting_error": {"ja": "[ERROR: 挨拶生成中にエラーが発生しました]", "en": "[ERROR: An error occurred during greeting generation]"},
    "anima.inbox_start": {"ja": "Inbox MSG処理開始", "en": "Inbox message processing started"},
    "anima.process_stream_error": {"ja": "process_message_streamエラー: {exc}", "en": "process_message_stream error: {exc}"},
    "anima.inbox_error": {"ja": "inbox処理エラー: {exc}", "en": "inbox processing error: {exc}"},
    "anima.unread_prefix": {"ja": "- {from_person} [⚠️ 未返信{count}回目]: ", "en": "- {from_person} [⚠️ Unreplied #{count}]: "},
    "anima.msg_received_episode": {"ja": "## {ts} {from_person}からのメッセージ受信\n\n**送信者**: {from_person}\n**内容**:\n{content}", "en": "## {ts} Message received from {from_person}\n\n**Sender**: {from_person}\n**Content**:\n{content}"},
    "anima.heartbeat_start": {"ja": "定期巡回開始", "en": "Periodic check started"},
    "anima.no_episodes_today": {"ja": "(本日のエピソードはありません)", "en": "(No episodes today)"},
    "anima.no_activity_log": {"ja": "(アクティビティログなし)", "en": "(No activity log)"},
    "anima.consolidation_start": {"ja": "{type}記憶統合開始", "en": "{type} consolidation started"},
    "anima.consolidation_end": {"ja": "{type}記憶統合完了", "en": "{type} consolidation completed"},
    "anima.consolidation_error": {"ja": "記憶統合エラー: {exc}", "en": "Consolidation error: {exc}"},
    "anima.cron_task_summary": {"ja": "タスク: {task}", "en": "Task: {task}"},
    "anima.cron_task_error": {"ja": "run_cron_taskエラー: {exc}", "en": "run_cron_task error: {exc}"},
    "anima.cron_cmd_error": {"ja": "run_cron_commandエラー: {exc}", "en": "run_cron_command error: {exc}"},
    "anima.cron_cmd_summary": {"ja": "コマンド: {task}", "en": "Command: {task}"},
    "anima.heartbeat_error": {"ja": "run_heartbeatエラー: {exc}", "en": "run_heartbeat error: {exc}"},
    "anima.heartbeat_episode": {"ja": "## {ts} ハートビート活動\n\n{summary}", "en": "## {ts} Heartbeat activity\n\n{summary}"},
    "anima.heartbeat_msgs_processed": {"ja": "\n\n（{count}件のメッセージを処理）", "en": "\n\n({count} messages processed)"},
    "anima.recovery_error_info": {"ja": "### エラー情報\n\n- エラー種別: {exc_type}\n- エラー内容: {exc_msg}\n- 発生日時: {ts}\n- 未処理メッセージ数: {count}", "en": "### Error information\n\n- Error type: {exc_type}\n- Error message: {exc_msg}\n- Occurred at: {ts}\n- Unprocessed message count: {count}"},
    # ── priming.py ──
    "priming.section_title": {"ja": "## あなたが思い出していること", "en": "## What you recall"},
    "priming.section_intro": {"ja": "以下は、この会話に関連してあなたが自然に想起した記憶です。", "en": "Below are memories you naturally recalled relevant to this conversation."},
    "priming.about_sender": {"ja": "### {sender_name} について", "en": "### About {sender_name}"},
    "priming.recent_activity_header": {"ja": "### 直近のアクティビティ", "en": "### Recent Activity"},
    "priming.related_knowledge_header": {"ja": "### 関連する知識", "en": "### Related Knowledge"},
    "priming.matched_skills_header": {"ja": "### 使えそうなスキル", "en": "### Matching Skills"},
    "priming.skills_list": {"ja": "あなたが持っているスキル: {skills_line}", "en": "Your skills: {skills_line}"},
    "priming.skills_detail_hint": {"ja": "※詳細はskillツールで取得してください。", "en": "Use the skill tool to load full details."},
    "priming.pending_tasks_header": {"ja": "### 未完了タスク", "en": "### Pending Tasks"},
    "priming.outbound_header": {"ja": "## 直近のアウトバウンド行動", "en": "## Recent Outbound Actions"},
    "priming.outbound_posted": {"ja": "- [{time_str}] #{ch} に投稿済み: 「{text_preview}」", "en": "- [{time_str}] Posted to #{ch}: \"{text_preview}\""},
    "priming.outbound_sent": {"ja": "- [{time_str}] {to} にメッセージ送信済み: 「{text_preview}」", "en": "- [{time_str}] Message sent to {to}: \"{text_preview}\""},
    # ── conversation.py ──
    "conversation.summary_label": {"ja": "[会話の要約（{count}ターン分）]", "en": "[Conversation summary ({count} turns)]"},
    "conversation.summary_ack": {"ja": "承知しました。これまでの会話内容を把握しました。", "en": "Understood. I have grasped the conversation so far."},
    "conversation.history_summary_header": {"ja": "### 会話の要約（{count}ターン分）", "en": "### Conversation summary ({count} turns)"},
    "conversation.recent_conversation_header": {"ja": "### 直近の会話", "en": "### Recent conversation"},
    "conversation.role_you": {"ja": "あなた", "en": "You"},
    "conversation.tools_executed": {"ja": "[実行ツール: {tool_names}]", "en": "[Tools used: {tool_names}]"},
    "conversation.ellipsis_omitted": {"ja": "...(前半省略)...", "en": "...(earlier omitted)..."},
    "conversation.tools_used": {"ja": "[使用ツール: {tools}]", "en": "[Tools used: {tools}]"},
    "conversation.existing_summary_header": {"ja": "## 既存の要約", "en": "## Existing summary"},
    "conversation.new_turns_header": {"ja": "## 新しい会話ターン", "en": "## New conversation turns"},
    "conversation.integrate_instruction": {"ja": "上記を統合した新しい要約を作成してください。", "en": "Please create a new integrated summary of the above."},
    "conversation.activity_context_header": {"ja": "## セッション中のその他の活動", "en": "## Other activity during session"},
    "conversation.title_fallback": {"ja": "会話", "en": "Conversation"},
    "conversation.resolved_marker": {"ja": "- ✅ {item}（自動検出: {ts}）", "en": "- ✅ {item} (auto-detected: {ts})"},
    "conversation.new_task_marker": {"ja": "- [ ] {task}（自動検出: {ts}）", "en": "- [ ] {task} (auto-detected: {ts})"},
    "conversation.resolution_summary": {"ja": "解決済み: {item}", "en": "Resolved: {item}"},
    "conversation.truncated_suffix": {"ja": "\n[...truncated, original {length} chars]", "en": "\n[...truncated, original {length} chars]"},
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
    "shortterm.already_sent_note": {"ja": "**注意: 以下の内容は既にユーザーに送信済みです。繰り返さないでください。**", "en": "**Note: The following content has already been sent to the user. Do NOT repeat it.**"},
    "shortterm.tools_used_recent": {"ja": "## 使用したツール（直近）", "en": "## Tools used (recent)"},
    "shortterm.notes_header": {"ja": "## 補足", "en": "## Notes"},
    "shortterm.none": {"ja": "(なし)", "en": "(none)"},
    "shortterm.ellipsis_omitted": {"ja": "...(前半省略)...\n", "en": "...(earlier omitted)...\n"},
    # ── activity.py ──
    "activity.blocked": {"ja": "ブロック: {reason}", "en": "Blocked: {reason}"},
    "activity.heartbeat_start": {"ja": "定期巡回開始", "en": "Periodic check started"},
    "activity.heartbeat_end": {"ja": "定期巡回完了", "en": "Periodic check completed"},
    "activity.cron_task_exec": {"ja": "cronタスク実行", "en": "Cron task executed"},
    "activity.error_prefix": {"ja": "[エラー] ", "en": "[Error] "},
    "activity.items_count": {"ja": "{count}件", "en": "{count} items"},
    # ── task_queue.py ──
    "task_queue.elapsed_minutes": {"ja": "⏱️ {minutes}分経過", "en": "⏱️ {minutes}m elapsed"},
    "task_queue.elapsed_hours_min": {"ja": "⏱️ {hours}時間{remaining_min}分経過", "en": "⏱️ {hours}h {remaining_min}m elapsed"},
    "task_queue.elapsed_hours": {"ja": "⏱️ {hours}時間経過", "en": "⏱️ {hours}h elapsed"},
    "task_queue.overdue": {"ja": "🔴 OVERDUE({time}期限)", "en": "🔴 OVERDUE(deadline {time})"},
    "task_queue.deadline_by": {"ja": "📅 {time}まで", "en": "📅 By {time}"},
    # ── agent.py ──
    "agent.recent_dialogue_header": {"ja": "## 直近の対話履歴", "en": "## Recent dialogue history"},
    "agent.recent_dialogue_intro": {"ja": "以下はユーザーとの直近の対話です。", "en": "Below is your recent dialogue with the user."},
    "agent.recent_dialogue_consider": {"ja": "進行中のタスクや指示がある場合、この内容を考慮してください。", "en": "Consider this content if there are ongoing tasks or instructions."},
    # ── messenger.py ──
    "messenger.depth_exceeded": {"ja": "ConversationDepthExceeded: {to}との会話が10分間に6ターンに達しました。次のハートビートサイクルまでお待ちください", "en": "ConversationDepthExceeded: Conversation with {to} reached 6 turns in 10 minutes. Please wait until the next heartbeat cycle."},
    "messenger.more_count": {"ja": "(+{count}件)", "en": "(+{count} more)"},
    "messenger.read_receipt": {"ja": "[既読通知] {count}件のメッセージを受信しました: {summary}", "en": "[Read receipt] Received {count} messages: {summary}"},
    # ── execution/assisted.py ──
    "assisted.output_truncated": {"ja": "... [出力切り捨て: 元のサイズ {size}バイト]", "en": "... [Output truncated: original size {size} bytes]"},
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
    "agent.priming_tier_light_header": {"ja": "## あなたが思い出していること\n\n### {sender_name} について\n\n", "en": "## What you recall\n\n### About {sender_name}\n\n"},
    "agent.omitted_rest": {"ja": "\n\n（以降省略）", "en": "\n\n(omitted)"},
    "agent.stream_retry_exhausted": {"ja": "ストリームが{retry_count}回切断されました。最大リトライ回数に達しました。", "en": "Stream disconnected {retry_count} time(s). Max retries reached."},
    # ── distillation.py ──
    "distillation.pattern_n_repeat": {"ja": "### パターン {i} ({count}回繰り返し)", "en": "### Pattern {i} (repeated {count} times)"},
    "distillation.none": {"ja": "(なし)", "en": "(none)"},
    # ── asset_reconciler.py ──
    "asset_reconciler.llm_user_prompt": {"ja": "以下のキャラクターシートから外見情報を読み取り、NovelAI 互換の画像生成タグに変換してください:\n\n{character_text}", "en": "Read the following character sheet and extract visual appearance into NovelAI-compatible image generation tags:\n\n{character_text}"},
    "asset_reconciler.llm_user_prompt_realistic": {"ja": "以下のキャラクターシートから外見情報を読み取り、写実的な写真風の画像生成プロンプトに変換してください:\n\n{character_text}", "en": "Read the following character sheet and extract visual appearance into a photorealistic image generation prompt:\n\n{character_text}"},
    # ── migrate.py ──
    "migrate.migration_note": {"ja": "<!-- MIGRATION NOTE: could not auto-convert '{schedule}' to cron expression -->", "en": "<!-- MIGRATION NOTE: could not auto-convert '{schedule}' to cron expression -->"},
    # ── chat.py ──
    "chat.image_too_large": {"ja": "画像データが大きすぎます（{size_mb}MB / 上限20MB）", "en": "Image data too large ({size_mb}MB / max 20MB)"},
    "chat.unsupported_image_format": {"ja": "未対応の画像形式です: {media_type}", "en": "Unsupported image format: {media_type}"},
    "chat.bootstrap_busy": {"ja": "初期化中です", "en": "Initializing"},
    "chat.heartbeat_processing": {"ja": "処理中です", "en": "Processing"},
    "chat.bootstrap_error": {"ja": "現在キャラクターを作成中です。完了までお待ちください。", "en": "Character is being created. Please wait for completion."},
    "chat.stream_incomplete": {"ja": "ストリームが予期せず終了しました。再試行してください。", "en": "Stream ended unexpectedly. Please retry."},
    "chat.anima_restarting": {"ja": "Animaが再起動中です。しばらく待ってから再試行してください。", "en": "Anima is restarting. Please wait and retry."},
    "chat.anima_unavailable": {"ja": "Animaのプロセスに接続できません。再起動中の可能性があります。", "en": "Cannot connect to Anima process. It may be restarting."},
    "chat.connection_lost": {"ja": "通信が切断されました。再試行してください。", "en": "Connection was lost. Please retry."},
    "chat.communication_error": {"ja": "通信エラーが発生しました。再試行してください。", "en": "Communication error. Please retry."},
    "chat.internal_error": {"ja": "内部エラーが発生しました。再試行してください。", "en": "An internal error occurred. Please retry."},
    "chat.timeout": {"ja": "応答がタイムアウトしました", "en": "Response timed out"},
    "chat.message_too_large": {"ja": "メッセージが大きすぎます（{size_mb}MB / 上限10MB）", "en": "Message too large ({size_mb}MB / max 10MB)"},
    "chat.stream_not_found": {"ja": "ストリームが見つからないか、アクセスが拒否されました", "en": "Stream not found or access denied"},
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
    "cli.migrate_cron_done": {"ja": "Migrated {count} anima(s) to standard cron format.", "en": "Migrated {count} anima(s) to standard cron format."},
    "cli.migrate_cron_skipped": {"ja": "No migration needed — all cron.md files are already in standard format.", "en": "No migration needed — all cron.md files are already in standard format."},
    # ── memory/manager.py ──
    "manager.action_log_header": {"ja": "# {date} 行動ログ\n\n", "en": "# {date} Action log\n\n"},
    # ── memory/dedup.py ──
    "dedup.messages_merged": {"ja": "[{count}件のメッセージを統合] ", "en": "[Merged {count} messages] "},
    # ── memory/contradiction.py ──
    "contradiction.resolution_summary": {"ja": "矛盾解決: {file_a} vs {file_b}", "en": "Contradiction resolved: {file_a} vs {file_b}"},
    "contradiction.strategy_label": {"ja": " → 戦略: {strategy}", "en": " → Strategy: {strategy}"},
    "contradiction.knowledge_resolution": {"ja": "knowledge矛盾解決({strategy})", "en": "Knowledge contradiction resolved ({strategy})"},
    # ── voice/session.py ──
    "voice.stt_failed": {"ja": "音声認識に失敗しました", "en": "Speech recognition failed"},
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
