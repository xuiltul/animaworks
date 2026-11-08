# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Domain-specific i18n strings (handler part 1)."""

from __future__ import annotations

STRINGS: dict[str, dict[str, str]] = {
    "handler.memory_file_renamed": {
        "ja": "{old} は {new} として扱いました",
        "en": "Treated {old} as {new}",
        "ko": "{old}을(를) {new}로 처리했습니다",
    },
    "handler.identity_write_rejected": {
        "ja": "{old} は直接書き換えられません。人格は SOUL.md、プロフィールは IDENTITY.md に書き分けてください",
        "en": "{old} cannot be written directly. Put personality in SOUL.md and profile in IDENTITY.md",
        "ko": "{old}은(는) 직접 수정할 수 없습니다. 인격은 SOUL.md, 프로필은 IDENTITY.md에 작성하세요",
    },
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
    "handler.audit_section_tool_summary": {
        "ja": "■ ツール使用サマリー（全{count}回）",
        "en": "■ Tool Usage Summary ({count} total)",
    },
    "handler.audit_status_line": {
        "ja": "状態: {status} | モデル: {model}",
        "en": "Status: {status} | Model: {model}",
    },
    "handler.audit_tool_line": {
        "ja": "{name} (全{total}回): ",
        "en": "{name} ({total} total): ",
    },
    "handler.background_task_started": {
        "ja": "タスクをバックグラウンドで実行開始しました (task_id: {task_id})",
        "en": "Task started in background (task_id: {task_id})",
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
        "ja": "{target_name}のbackground_modelを'{model}'に変更し、稼働中プロセスへ root から再読み込みを依頼しました。停止中の場合は次回起動時に反映されます。",
        "en": "Changed {target_name}'s background_model to '{model}' and asked root to reload the running process. Stopped Animas will use it on next start.",
    },
    "handler.bg_model_cleared": {
        "ja": "{target_name}のbackground_modelをクリアしました（メインモデルを使用）。稼働中プロセスへ root から再読み込みを依頼しました。",
        "en": "Cleared {target_name}'s background_model (will use main model) and asked root to reload the running process.",
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
            "{from_name}さんがボード #{channel} であなたをメンションしました:\n\n{text}\n\n返信するには `animaworks-tool board post {channel} -` を実行し、本文を標準入力で渡してください。"
        ),
        "en": (
            "{from_name} mentioned you on board #{channel}:\n\n{text}\n\nTo reply, run `animaworks-tool board post {channel} -` and pass the body through stdin."
        ),
    },
    "handler.body_param_required": {
        "ja": "`body` パラメータは必須です。",
        "en": "The `body` parameter is required.",
    },
    "handler.channel_acl_denied": {
        "ja": (
            "Error: #{channel} へのアクセス権がありません。`animaworks-tool internal manage-channel info {channel}` でメンバーを確認してください。"
        ),
        "en": (
            "Error: You do not have access to #{channel}. Use `animaworks-tool internal manage-channel info {channel}` to check members."
        ),
    },
    "handler.channel_acl_not_member": {
        "ja": "Error: #{channel} のメンバーではないため、メンバー管理操作はできません。",
        "en": "Error: You are not a member of #{channel} and cannot manage its membership.",
    },
    "handler.channel_add_member_open_denied": {
        "ja": (
            "Error: #{channel} はオープンチャネルです。制限チャネルを新規作成するには `animaworks-tool internal manage-channel create <channel> --member <name>` を使ってください。"
        ),
        "en": (
            "Error: #{channel} is an open channel. Create a restricted channel with `animaworks-tool internal manage-channel create <channel> --member <name>`."
        ),
    },
    "handler.channel_already_exists": {
        "ja": "Error: チャネル #{channel} は既に存在します",
        "en": "Error: Channel #{channel} already exists",
    },
    "handler.channel_archived_name_reserved": {
        "ja": "Error: #{channel} はアーカイブ済みの名前です。別の名前を使ってください。",
        "en": "Error: The archived name #{channel} is reserved. Use a different name.",
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
        "ja": (
            "Error: チャネル #{channel} が見つかりません。"
            "`animaworks-tool internal manage-channel create {channel}` で明示的に作成してください。"
        ),
        "en": (
            "Error: Channel #{channel} not found. "
            "Create it explicitly with `animaworks-tool internal manage-channel create {channel}`."
        ),
    },
    "handler.channel_empty": {
        "ja": "#{channel} にはまだ投稿がありません。",
        "en": "There are no posts in #{channel} yet.",
    },
    "handler.channel_read_first": {
        "ja": "#{channel} の未読 {unread} 件のうち、最新 {shown} 件を出します。最新だけを見るときは `--latest`。",
        "en": "Showing the latest {shown} of {unread} unread posts in #{channel}. Use `--latest` to read only the newest posts.",
        "ko": "#{channel}의 미독 {unread}건 중 최신 {shown}건을 표시합니다. 최신 글만 보려면 `--latest`를 사용하세요.",
        "zh": "#{channel} 有 {unread} 条未读，显示其中最新 {shown} 条。只想看最新内容时使用 `--latest`。",
    },
    "handler.channel_read_continuation": {
        "ja": "#{channel} の未読 {unread} 件のうち、古い順に {shown} 件を出します。残り {remaining} 件。最新だけを見るときは `--latest`。",
        "en": "Showing {shown} of {unread} unread posts in #{channel}, oldest first. {remaining} remain. Use `--latest` to read only the newest posts.",
        "ko": "#{channel}의 미독 {unread}건 중 오래된 순서로 {shown}건을 표시합니다. 남은 {remaining}건. 최신 글만 보려면 `--latest`를 사용하세요.",
        "zh": "#{channel} 有 {unread} 条未读，按从旧到新显示 {shown} 条，还剩 {remaining} 条。只想看最新内容时使用 `--latest`。",
    },
    "handler.channel_read_none": {
        "ja": "#{channel} に新しい投稿はありません。",
        "en": "There are no new posts in #{channel}.",
        "ko": "#{channel}에 새 글이 없습니다.",
        "zh": "#{channel} 没有新帖子。",
    },
    "handler.channel_inbox_cli": {
        "ja": "'{channel}' はチャネルではありません。Inbox は自動処理されます。Inbox を読むには `animaworks-tool board read` を使わないでください。",
        "en": "'{channel}' is not a channel. Inbox messages are processed automatically; do not use `animaworks-tool board read` for inbox.",
    },
    "handler.memory_read_outside_cli": {
        "ja": "パス '{path}' は自分のディレクトリ外です。'knowledge/foo.md' のような相対パスを使ってください。共有チャネルは `animaworks-tool board read` または `animaworks-tool board post` を使ってください。",
        "en": "Path '{path}' resolves outside your directory. Use relative paths like 'knowledge/foo.md'. For shared channels, use `animaworks-tool board read` or `animaworks-tool board post`.",
    },
    "handler.dm_history_empty": {
        "ja": "{peer} とのDM履歴はありません。",
        "en": "There is no DM history with {peer}.",
    },
    "handler.memory_channel_read_cli": {
        "ja": "共有チャネルはこの方法では読めません。`animaworks-tool board read {channel}` を使ってください。",
        "en": "Shared channels cannot be read this way. Use `animaworks-tool board read {channel}`.",
    },
    "handler.memory_channel_write_cli": {
        "ja": "共有チャネルへこの方法では書き込めません。`animaworks-tool board post {channel} -` を実行し、本文を標準入力で渡してください。",
        "en": "Shared channels cannot be written this way. Run `animaworks-tool board post {channel} -` and pass the body through stdin.",
    },
    "handler.channel_post_invalid_poster": {
        "ja": "Error: 投稿者名が既知の Anima ではありません: {poster}",
        "en": "Error: The channel poster is not a known Anima: {poster}",
    },
    "handler.channel_post_success": {
        "ja": "#{channel} に投稿しました",
        "en": "Posted to #{channel}",
    },
    "handler.channel_post_duplicate": {
        "ja": "#{channel} への投稿は既に登録済みです",
        "en": "The post to #{channel} was already recorded",
    },
    "handler.channel_post_notified": {
        "ja": "通知しました: {names}",
        "en": "Notified: {names}",
    },
    "handler.channel_post_not_notified": {
        "ja": "通知しませんでした: {details}",
        "en": "Not notified: {details}",
    },
    "handler.channel_post_failed": {
        "ja": "Error: ボードへの投稿に失敗しました: {detail}",
        "en": "Error: Board post failed: {detail}",
    },
    "cli.board_post_success": {
        "ja": "#{channel} に投稿しました",
        "en": "Posted to #{channel}",
    },
    "cli.board_post_empty": {
        "ja": "Error: 投稿本文が空です。本文を指定するか、標準入力から渡してください。",
        "en": "Error: Post text is empty. Provide text or pass it through stdin.",
    },
    "cli.board_impersonation_refused": {
        "ja": "Error: anima '{anima}' は '{requested}' として操作できません",
        "en": "Error: anima '{anima}' cannot act as '{requested}'",
    },
    "cli.board_post_invalid_args": {
        "ja": "Error: 投稿の引数が多すぎます。board post <channel> [<text> | -] を使ってください。",
        "en": "Error: Too many post arguments. Use board post <channel> [<text> | -].",
    },
    "cli.board_read_invalid_channel": {
        "ja": "Error: チャネル名が不正です: {detail}",
        "en": "Error: Invalid channel name: {detail}",
    },
    "cli.board_post_parser_help": {
        "ja": "チャネルに投稿",
        "en": "Post to a channel",
    },
    "cli.board_post_parser_description": {
        "ja": "チャネルに投稿します。形式: board post <channel> [<text> | -]。本文を省略するか '-' を指定すると標準入力から読みます。旧形式: board post <from> <channel> <text>。",
        "en": "Post to a channel. Use: board post <channel> [<text> | -]. Omit text or use '-' to read it from stdin. Legacy form: board post <from> <channel> <text>.",
    },
    "cli.board_post_parser_args_help": {
        "ja": "<channel> [<text> | -]、または旧形式 <from> <channel> <text>",
        "en": "<channel> [<text> | -], or legacy <from> <channel> <text>",
    },
    "cli.board_post_as_help": {
        "ja": "投稿者名（ホストからの投稿）",
        "en": "Operator poster name",
    },
    "cli.board_post_failed": {
        "ja": "Error: ボードへの投稿に失敗しました: {detail}",
        "en": "Error: Board post failed: {detail}",
    },
    "cli.timeline_add_parser_help": {
        "ja": "会社のタイムラインに「終わった」「決まった」を追記",
        "en": "Append a done/decided item to the company timeline",
    },
    "cli.timeline_add_kind_help": {
        "ja": "項目の種類（done または decided）",
        "en": "Item kind (done or decided)",
    },
    "cli.timeline_add_text_help": {
        "ja": "タイムラインに載せる本文",
        "en": "Text to record on the timeline",
    },
    "cli.timeline_add_about_help": {
        "ja": "更新する元の項目 ID（tl_...）",
        "en": "Original item id to update (tl_...)",
    },
    "cli.timeline_add_ref_help": {
        "ja": "参照 URL など",
        "en": "Reference URL or similar",
    },
    "cli.timeline_add_from_help": {
        "ja": "記録者名（オペレーター実行時のみ）",
        "en": "Recorder name (operator runs only)",
    },
    "cli.timeline_read_parser_help": {
        "ja": "会社のタイムラインを読む（直近24時間）",
        "en": "Read the company timeline (last 24h)",
    },
    "cli.timeline_read_limit_help": {
        "ja": "取得する最大件数",
        "en": "Maximum number of items",
    },
    "cli.timeline_add_success": {
        "ja": "タイムラインに {kind} を追記しました",
        "en": "Added {kind} to the timeline",
    },
    "cli.timeline_add_failed": {
        "ja": "Error: タイムラインへの追記に失敗しました: {detail}",
        "en": "Error: Timeline append failed: {detail}",
    },
    "cli.timeline_read_failed": {
        "ja": "Error: タイムラインの取得に失敗しました: {detail}",
        "en": "Error: Timeline read failed: {detail}",
    },
    "server.timeline_disabled": {
        "ja": "タイムラインは無効です",
        "en": "Timeline is disabled",
    },
    "server.timeline_item_not_found": {
        "ja": "タイムライン項目が見つかりません: {item_id}",
        "en": "Timeline item not found: {item_id}",
    },
    "server.timeline_company_mismatch": {
        "ja": "この Anima の会社はタイムラインの対象会社と一致しません",
        "en": "This anima's company does not match the timeline company",
    },
    "server.timeline_invalid_request": {
        "ja": "タイムラインの要求が不正です: {detail}",
        "en": "Invalid timeline request: {detail}",
    },
    "handler.channel_post_validation_failed": {
        "ja": "Error: 投稿者名を検証できないため投稿を中止しました",
        "en": "Error: Could not validate the channel poster, so the post was refused",
    },
    "handler.channel_open": {
        "ja": "#{channel} はオープンチャネルです（全Animaがアクセス可能）",
        "en": "#{channel} is an open channel (all Animas can access)",
    },
    "handler.channel_company_unset": {
        "ja": (
            "Error: チャネル #{channel} は会社帰属が未設定です。"
            "管理者に会社帰属の設定を依頼してください。チャネル情報は `animaworks-tool internal manage-channel info {channel}` で確認できます。"
        ),
        "en": (
            "Error: Channel #{channel} has no company attribution set. "
            "Ask an administrator to set it, or use a company-scoped channel. Inspect it with `animaworks-tool internal manage-channel info {channel}`."
        ),
        "zh": (
            "错误：频道 #{channel} 尚未设置公司归属。请联系管理员设置，或使用本公司频道。可通过 `animaworks-tool internal manage-channel info {channel}` 查看频道信息。"
        ),
        "ko": (
            "오류: 채널 #{channel} 에 회사 귀속이 설정되어 있지 않습니다. 관리자에게 설정을 요청하거나 자사 채널을 이용하세요. "
            "`animaworks-tool internal manage-channel info {channel}` 로 채널 정보를 확인할 수 있습니다."
        ),
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
    "handler.delegate_dedup_log": {
        "ja": "{target_name}への委譲は既存の依頼 {task_id} にまとめました: {summary}",
        "en": "Consolidated the delegation to {target_name} into existing task {task_id}: {summary}",
    },
    "handler.delegated_success": {
        "ja": (
            "タスクを {target_name} に委譲しました。\n- 部下側タスクID: {sub_id}\n- 追跡用タスクID: {own_id}\n- {dm_result}"
        ),
        "en": (
            "Task delegated to {target_name}.\n- Subordinate task ID: {sub_id}\n- Tracking task ID: {own_id}\n- {dm_result}"
        ),
    },
    "handler.delegation_deduplicated": {
        "ja": "同じ依頼が未完了で登録済みのため、新規作成せず既存のタスクID {task_id} にまとめました",
        "en": "The same request is already open, so it was consolidated into existing task {task_id}",
    },
    "handler.delegation_deduplicated_note": {
        "ja": (
            "同じ依頼が未完了で登録済みのため、{target_name} への新規タスクは作らず"
            "既存のタスクID {task_id} にまとめました。DM は送っていません。"
        ),
        "en": (
            "The same request is already open, so no new task was created for {target_name}; "
            "it was consolidated into existing task {task_id}. No DM was sent."
        ),
    },
    "handler.delegation_dm_content": {
        "ja": ("[タスク委譲]\n{instruction}\n\nタスクID: {task_id}"),
        "en": ("[Task delegation]\n{instruction}\n\nTask ID: {task_id}"),
    },
    "handler.delegation_dm_skipped": {
        "ja": "DM は送っていません（タスクは相手の実行キューに登録済みです）",
        "en": "DM skipped (the task is already in the assignee's execution queue)",
        "ko": "DM을 보내지 않았습니다 (담당자의 실행 큐에 작업이 등록되어 있습니다)",
    },
    "handler.delegation_intent_deprecated": {
        "ja": (
            "Error: intent='delegation' は廃止されました。タスクを委任するには delegate_task ツールを使用してください。send_message は report / question のみ対応しています。"
        ),
        "en": (
            "Error: intent='delegation' has been deprecated. Use the delegate_task tool to assign tasks to subordinates. send_message only supports 'report' and 'question' intents."
        ),
    },
    "handler.descendant_activity": {
        "ja": "配下のactivity_log",
        "en": "Descendant activity_log",
    },
    "handler.descendant_state": {
        "ja": "配下のstatus.json, IDENTITY.md, AGENTS.md, state/current_state.md, state/task_queue.jsonl, state/plans/",
        "en": "Descendant status.json, IDENTITY.md, AGENTS.md, state/current_state.md, state/task_queue.jsonl, state/plans/",
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
    "handler.status_json_invalid": {
        "ja": "'{target_name}' の status.json が壊れているか読み取れないため、変更しませんでした。",
        "en": "'{target_name}' status.json is invalid or unreadable; no changes were made.",
    },
    "handler.dm_already_sent": {
        "ja": "Error: このrunで既に {to} にメッセージを送信済みです。追加の連絡はBoardを使用してください。",
        "en": ("Error: Message already sent to {to} in this run. Use Board for additional communication."),
    },
    "handler.send_feedback_header": {
        "ja": "[24h] {to}宛: {count}通",
        "en": "[24h] To {to}: {count} messages",
    },
    "handler.send_recipient_inbox_busy": {
        "ja": (
            "⏳ {to} は {since} から別のメッセージの処理中です（経過 {elapsed}）。"
            "このメッセージはその処理が終わるまで読まれません（未読 {count} 件）。"
            "急ぎなら別の手段を検討してください。"
        ),
        "en": (
            "⏳ {to} has been processing other messages since {since} ({elapsed} elapsed). "
            "This message will not be read until that run ends ({count} unread). "
            "If it is urgent, consider another route."
        ),
    },
    "handler.send_recipient_inbox_oldest": {
        "ja": "最も古い未読は {oldest} 着。",
        "en": " Oldest unread arrived at {oldest}.",
    },
    "handler.dm_intent_error": {
        "ja": (
            "Error: DMのintentは 'report', 'question' のみ許可されています。acknowledgment・感謝・FYIは `animaworks-tool board post <channel> <text>` でBoardに投稿してください。"
        ),
        "en": (
            "Error: DM intent must be 'report' or 'question' only. Post acknowledgments, thanks, or FYI with `animaworks-tool board post <channel> <text>`."
        ),
    },
    "handler.dm_self_addressed_error": {
        "ja": (
            "Error: 自分自身へのDM送信はできません。自分用のメモは write_memory_file、作業管理は update_task またはタスクキューを使用してください。"
        ),
        "en": (
            "Error: You cannot send a DM to yourself. Use write_memory_file for notes to self, or update_task/the task queue for work tracking."
        ),
    },
    "handler.dm_send_failed": {
        "ja": "DM送信失敗: {e}",
        "en": "DM send failed: {e}",
    },
    "handler.dm_sent": {
        "ja": "DM送信済み",
        "en": "DM sent",
    },
    "handler.code_search_requires_project": {
        "ja": "code検索にはprojectが必要です",
        "en": "code search requires a project",
        "ko": "code 검색에는 project가 필요합니다",
    },
    "hub.tool.timeline_disabled": {
        "ja": "タイムラインは無効です",
        "en": "The company timeline is disabled.",
    },
    "hub.tool.timeline_empty": {
        "ja": "直近24時間のタイムラインに項目はありません",
        "en": "No timeline items in the last 24 hours.",
    },
    "hub.tool.no_new_posts": {
        "ja": "新着なし ({now})",
        "en": "No new posts ({now})",
    },
    "hub.tool.help_bilingual_ja": {
        "ja": "日本語: 会社のタイムラインは timeline（読み取り専用）、#hub の新着は read、投稿は post。anima との連絡は必ず #hub で行い、人間への直接通知の前に timeline を確認してください。緊急でない用件は #hub に投稿してください。",
        "en": "日本語: 会社のタイムラインは timeline（読み取り専用）、#hub の新着は read、投稿は post。anima との連絡は必ず #hub で行い、人間への直接通知の前に timeline を確認してください。緊急でない用件は #hub に投稿してください。",
    },
    "cli.hub_parser_help": {
        "ja": "ハブの名簿と外部 AI のメンバーを管理",
        "en": "Manage the hub roster and external AI members",
    },
    "cli.hub_member_parser_help": {
        "ja": "ハブのメンバーを管理",
        "en": "Manage hub members",
    },
    "cli.hub_member_add_help": {
        "ja": "メンバーを名簿に追加",
        "en": "Add a member to the roster",
    },
    "cli.hub_member_add_id_help": {
        "ja": "メンバー ID（英小文字で始まる 2〜32 文字）",
        "en": "Member ID (2-32 characters, starts with a lowercase letter)",
    },
    "cli.hub_member_add_company_help": {
        "ja": "メンバーの会社 ID",
        "en": "Member company ID",
    },
    "cli.hub_member_add_route_help": {
        "ja": "連絡方法（webhook または poll）",
        "en": "Message route (webhook or poll)",
    },
    "cli.hub_member_add_url_ref_help": {
        "ja": "webhook URL の vault キー名（shared セクション）",
        "en": "Vault key name for the webhook URL (shared section)",
    },
    "cli.hub_member_add_key_ref_help": {
        "ja": "webhook 認証キーの vault キー名（shared セクション）",
        "en": "Vault key name for the webhook key (shared section)",
    },
    "cli.hub_member_add_expect_every_help": {
        "ja": "期待する連絡間隔（30m / 3h / 1d など）",
        "en": "Expected contact interval (for example 30m, 3h, 1d)",
    },
    "cli.hub_member_add_display_name_help": {
        "ja": "表示名（既定は ID）",
        "en": "Display name (defaults to the ID)",
    },
    "cli.hub_member_added": {
        "ja": "メンバーを追加しました: {id}",
        "en": "Member added: {id}",
    },
    "cli.hub_member_exists": {
        "ja": "メンバーは既に登録されています: {id}",
        "en": "Member already exists: {id}",
    },
    "cli.hub_member_not_found": {
        "ja": "メンバーが見つかりません: {id}",
        "en": "Member not found: {id}",
    },
    "cli.hub_member_enabled": {
        "ja": "メンバーを有効にしました: {id}",
        "en": "Member enabled: {id}",
    },
    "cli.hub_member_disabled": {
        "ja": "メンバーを無効にしました: {id}",
        "en": "Member disabled: {id}",
    },
    "cli.hub_member_token_set": {
        "ja": "トークンを更新しました: {id}",
        "en": "Token updated: {id}",
    },
    "cli.hub_member_token_env_missing": {
        "ja": "環境変数 {var} が未設定または空です",
        "en": "Environment variable {var} is unset or empty",
    },
    "cli.hub_member_webhook_refs": {
        "ja": "webhook には --url-ref と --key-ref が必要です",
        "en": "webhook requires --url-ref and --key-ref",
    },
    "cli.hub_member_invalid": {
        "ja": "メンバーの設定が不正です: {detail}",
        "en": "Invalid member settings: {detail}",
    },
    "cli.hub_member_list_empty": {
        "ja": "ハブのメンバーは登録されていません",
        "en": "No hub members are registered",
    },
    "cli.hub_member_add_kind_help": {
        "ja": "メンバーの種類（既定: vendor_bot）",
        "en": "Member kind (default: vendor_bot)",
    },
    "cli.hub_member_list_help": {
        "ja": "ハブのメンバー一覧を表示（トークンは表示しない）",
        "en": "List hub members (tokens are never shown)",
    },
    "cli.hub_member_set_token_help": {
        "ja": "環境変数からベアラートークンの SHA-256 を登録",
        "en": "Store the SHA-256 of a bearer token read from an environment variable",
    },
    "cli.hub_member_set_token_from_env_help": {
        "ja": "トークンを格納した環境変数名",
        "en": "Environment variable holding the token",
    },
    "cli.hub_member_disable_help": {
        "ja": "メンバーを無効化",
        "en": "Disable a member",
    },
    "cli.hub_member_enable_help": {
        "ja": "メンバーを有効化",
        "en": "Enable a member",
    },
    "cli.hub_setup_prompt_help": {
        "ja": "社外の AI 向けのセットアップ文面を表示",
        "en": "Print setup instructions for an external AI",
    },
    "cli.hub_setup_prompt_url_help": {
        "ja": "接続先 URL（既定: public_hosts の先頭）",
        "en": "Connection URL (defaults to the first public host)",
    },
    "cli.hub_setup_url_required": {
        "ja": "接続先 URL がありません。--url を指定してください",
        "en": "No connection URL is configured; pass --url",
    },
    "cli.hub_setup_member_disabled": {
        "ja": "メンバーは無効です: {id}",
        "en": "Member is disabled: {id}",
    },
    "cli.hub_setup_config_error": {
        "ja": "ハブの設定を読み取れませんでした",
        "en": "Could not read the hub configuration",
    },
    "cli.hub_member_set_avatar_help": {
        "ja": "メンバーのアバター画像を登録（256x256 の PNG に変換）",
        "en": "Register a member avatar (normalized to a 256x256 PNG)",
    },
    "cli.hub_member_set_avatar_file_help": {
        "ja": "画像ファイル（PNG / JPEG / WebP / GIF、5MB まで）",
        "en": "Image file (PNG, JPEG, WebP, or GIF, up to 5MB)",
    },
    "cli.hub_member_set_avatar_clear_help": {
        "ja": "登録済みのアバターを削除",
        "en": "Remove the stored avatar",
    },
    "cli.hub_member_avatar_set": {
        "ja": "アバターを登録しました: {id}",
        "en": "Avatar updated: {id}",
    },
    "cli.hub_member_avatar_cleared": {
        "ja": "アバターを削除しました: {id}",
        "en": "Avatar cleared: {id}",
    },
    "cli.hub_member_avatar_file_required": {
        "ja": "画像ファイルを指定するか --clear を使ってください",
        "en": "Pass an image file or use --clear",
    },
    "cli.hub_member_avatar_read_failed": {
        "ja": "画像ファイルを読み取れませんでした: {path}",
        "en": "Could not read the image file: {path}",
    },
    "cli.hub_member_avatar_invalid": {
        "ja": "アバター画像が不正です: {detail}",
        "en": "Invalid avatar image: {detail}",
    },
    "hub.setup_interval_minutes": {
        "ja": "{n}分ごと",
        "en": "every {n} minutes",
    },
    "hub.setup_interval_hours": {
        "ja": "{n}時間ごと",
        "en": "every {n} hours",
    },
    "hub.setup_interval_days": {
        "ja": "{n}日ごと",
        "en": "every {n} days",
    },
    "hub.setup_interval_default": {
        "ja": "1時間ごと",
        "en": "every hour",
    },
    "hub.setup_roster_empty": {
        "ja": "- （該当する anima はいません）",
        "en": "- (no matching animas)",
    },
    "hub.wake_task": {
        "ja": (
            "AnimaWorks のハブで @{member_id} 宛ての投稿があります。"
            "AnimaWorks Hub コネクタの `hub read` で読み、必要なら "
            "`hub post --reply-to {post_id}` で返事してください。"
        ),
        "en": (
            "There is a post for @{member_id} on the AnimaWorks hub. "
            "Read it with the AnimaWorks Hub connector's `hub read`, and reply with "
            "`hub post --reply-to {post_id}` if needed."
        ),
    },
    "hub.loop_limit_subject": {
        "ja": "ハブの往復が上限に達しました",
        "en": "Hub round trips reached the limit",
    },
    "hub.loop_limit_body": {
        "ja": (
            "ボード #{board} の投稿 {post_id} で、anima と社外の往復が上限に達しました。"
            "このスレッドでは新しい投稿を配りません。"
        ),
        "en": (
            "Anima and external members reached the round-trip limit on board #{board} "
            "at post {post_id}. New posts in this thread are not delivered."
        ),
    },
    "hub.silence_subject": {
        "ja": "{name} がハブに来ていません",
        "en": "{name} has not visited the hub",
    },
    "hub.silence_body": {
        "ja": "{name} は最後に {last_seen} にハブへ来ました。expect_every を過ぎています。",
        "en": "{name} last visited the hub at {last_seen}, which is past expect_every.",
    },
}
