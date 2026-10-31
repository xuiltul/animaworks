# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Domain-specific i18n strings (handler part 2)."""

from __future__ import annotations

STRINGS: dict[str, dict[str, str]] = {
    "handler.company_boundary_unverifiable": {
        "ja": "エラー: 会社境界を確認できないため、この操作は安全のためブロックされました。しばらくしてから再試行するか、人間のオーナーへエスカレーションしてください。",
        "en": (
            "Error: The company boundary could not be verified, so this operation was blocked for safety. "
            "Try again later or escalate this matter to the human owner."
        ),
        "zh": "错误：无法验证公司边界，因此为确保安全，此操作已被阻止。请稍后重试或将此事项升级给人类所有者处理。",
        "ko": "오류: 회사 경계를 확인할 수 없어 안전을 위해 이 작업이 차단되었습니다. 나중에 다시 시도하거나 인간 소유자에게 에스컬레이션하세요.",
    },
    "handler.cross_company_delegation_blocked": {
        "ja": (
            "エラー: 委任先は別会社「{display_name}」に所属しています。会社間の直接のタスク委任は禁止されています。"
            "この用件は人間のオーナーへエスカレーションしてください。"
        ),
        "en": (
            "Error: The delegate belongs to another company, {display_name}. Direct task delegation between companies "
            "is prohibited. Escalate this matter to the human owner."
        ),
        "zh": (
            "错误：委派对象属于另一家公司“{display_name}”。禁止公司之间直接委派任务。请将此事项升级给人类所有者处理。"
        ),
        "ko": (
            "오류: 위임 대상은 다른 회사인 ‘{display_name}’에 소속되어 있습니다. 회사 간 직접 업무 위임은 "
            "금지되어 있습니다. 이 사안을 인간 소유자에게 에스컬레이션하세요."
        ),
    },
    "handler.cross_company_message_blocked": {
        "ja": (
            "エラー: 宛先は別会社「{display_name}」に所属しています。会社間の直接連絡は禁止されています。"
            "この用件は人間のオーナーへエスカレーションしてください。"
        ),
        "en": (
            "Error: The recipient belongs to another company, {display_name}. Direct communication between companies "
            "is prohibited. Escalate this matter to the human owner."
        ),
        "zh": ("错误：收件人属于另一家公司“{display_name}”。禁止公司之间直接联系。请将此事项升级给人类所有者处理。"),
        "ko": (
            "오류: 수신자는 다른 회사인 ‘{display_name}’에 소속되어 있습니다. 회사 간 직접 연락은 금지되어 "
            "있습니다. 이 사안을 인간 소유자에게 에스컬레이션하세요."
        ),
    },
    "handler.enable_log_summary": {
        "ja": "{target_name} を復帰",
        "en": "Enabling {target_name}",
    },
    "handler.enabled_success": {
        "ja": "'{target_name}' を有効にしました。Reconciliation が30秒以内にプロセスを起動します。",
        "en": ("'{target_name}' has been enabled. Reconciliation will start the process within 30 seconds."),
    },
    "handler.episode_filename_warning": {
        "ja": (
            "WARNING: エピソードファイル名 '{filename}' は標準パターン (YYYY-MM-DD.md または YYYY-MM-DD_suffix.md) に合致しません。 推奨: episodes/{date}.md に '## HH:MM — タイトル' 形式で追記してください。"
        ),
        "en": (
            "WARNING: Episode filename '{filename}' does not match the standard pattern (YYYY-MM-DD.md or YYYY-MM-DD_suffix.md). Recommended: append to episodes/{date}.md in '## HH:MM — Title' format."
        ),
        "zh": (
            "警告：剧集文件名 '{filename}' 不符合标准模式 (YYYY-MM-DD.md 或 YYYY-MM-DD_suffix.md)。建议：以 '## HH:MM — 标题' 格式追加到 episodes/{date}.md。"
        ),
        "ko": (
            "경고: 에피소드 파일 이름 '{filename}'이 표준 패턴(YYYY-MM-DD.md 또는 YYYY-MM-DD_suffix.md)과 일치하지 않습니다. 권장: episodes/{date}.md에 '## HH:MM — 제목' 형식으로 추가하십시오."
        ),
    },
    "handler.file_read_own": {
        "ja": "自分のディレクトリ",
        "en": "Own directory",
    },
    "handler.file_read_shared": {
        "ja": "shared/",
        "en": "shared/",
    },
    "handler.file_write_own": {
        "ja": "自分のディレクトリ",
        "en": "Own directory",
    },
    "handler.last_activity_none": {
        "ja": "なし",
        "en": "None",
    },
    "handler.last_activity_unknown": {
        "ja": "不明",
        "en": "Unknown",
    },
    "handler.legacy_skill_sections": {
        "ja": "旧形式のセクション(## 概要 / ## 発動条件)が検出されました。Claude Code形式(YAMLフロントマター)への移行を推奨します。",
        "en": (
            "Legacy sections (## 概要 / ## 発動条件) detected. Migration to Claude Code format (YAML frontmatter) is recommended."
        ),
        "zh": "检测到旧格式的部分 (## 概要 / ## 发动条件)。建议迁移到 Claude Code 格式 (YAML 前言)。",
        "ko": (
            "이전 형식의 섹션(## 概要 / ## 発動条件)이 감지되었습니다. Claude Code 형식(YAML 프론트매터)으로의 마이그레이션을 권장합니다."
        ),
    },
    "handler.messenger_not_set": {
        "ja": "メッセンジャー未設定（タスクキューへの追加は成功）",
        "en": "Messenger not configured (task added to queue successfully)",
    },
    "handler.model_change_log": {
        "ja": "{target_name} のモデルを {model} に変更",
        "en": "Changing {target_name}'s model to {model}",
    },
    "handler.model_changed": {
        "ja": "'{target_name}' のモデルを '{model}' に変更し、稼働中プロセスへ root から再読み込みを依頼しました。停止中の場合は次回起動時に反映されます。",
        "en": "Changed {target_name}'s model and asked root to reload the running process. Stopped Animas will use it on next start.",
    },
    "handler.model_warning": {
        "ja": "警告: '{model}' は既知のモデルカタログに含まれていません。正しいモデル名か確認してください。",
        "en": ("Warning: '{model}' is not in the known model catalog. Please verify the model name."),
    },
    "handler.name_field_required": {
        "ja": "`name` フィールドが必要です。",
        "en": "The `name` field is required.",
        "zh": "需要 `name` 字段。",
        "ko": "`name` 필드가 필요합니다.",
    },
    "handler.no_delegated_tasks": {
        "ja": "委譲済みタスクはありません",
        "en": "No delegated tasks",
    },
    "handler.no_matching_delegated": {
        "ja": "条件に合う委譲済みタスクはありません (filter={status})",
        "en": "No delegated tasks matching filter ({status})",
    },
    "handler.no_subordinates": {
        "ja": "配下の Anima はいません",
        "en": "No subordinate Animas",
    },
    "handler.none_value": {
        "ja": "(なし)",
        "en": "(none)",
    },
    "handler.not_descendant": {
        "ja": "'{target_name}' はあなたの配下ではありません",
        "en": "'{target_name}' is not under your supervision",
    },
    "handler.not_direct_subordinate": {
        "ja": "'{target_name}' はあなたの直属部下ではありません",
        "en": "'{target_name}' is not your direct subordinate",
    },
    "handler.org_dashboard_title": {
        "ja": "## 組織ダッシュボード",
        "en": "## Organization Dashboard",
    },
    "handler.outcome_failure": {
        "ja": "失敗",
        "en": "Failure",
    },
    "handler.outcome_success": {
        "ja": "成功",
        "en": "Success",
    },
    "handler.output_truncated": {
        "ja": "[出力が50KBを超えたためトランケーションしました。元のサイズ: {size}]",
        "en": "[Output truncated because it exceeded 50KB. Original size: {size}]",
    },
    "handler.peer_activity": {
        "ja": "同僚のactivity_log（読み取り専用）",
        "en": "Peer activity_log (read-only)",
    },
    "handler.ping_summary": {
        "ja": "{target}の生存確認",
        "en": "Ping {target}",
    },
    "handler.post_already_posted": {
        "ja": "Error: このrunで既に #{channel} に投稿済みです。同一チャネルへの連投はできません。{alt_hint}",
        "en": ("Error: Already posted to #{channel} in this run. No duplicate posts to the same channel.{alt_hint}"),
    },
    "handler.post_alt_hint": {
        "ja": " 別のチャネル（{channels}）への投稿、またはsend_message（intent: question/report）は可能です。",
        "en": (" You can post to another channel ({channels}) or use send_message (intent: question/report)."),
    },
    "handler.procedure_description_missing": {
        "ja": "`description` フィールドがありません。自動マッチングを有効にするために description を追加してください。",
        "en": "The `description` field is missing. Add description to enable auto-matching.",
    },
    "handler.procedure_format_validation": {
        "ja": ("⚠️ 手順書フォーマット検証:\n{msg}"),
        "en": ("⚠️ Procedure format validation:\n{msg}"),
    },
    "handler.procedure_frontmatter_recommended": {
        "ja": "手順書ファイルにはYAMLフロントマター(---)を推奨します。description フィールドで自動マッチングが有効になります。",
        "en": ("Procedure files should have YAML frontmatter (---). The description field enables auto-matching."),
        "zh": "建议程序文件使用 YAML 前言 (---)。description 字段将启用自动匹配。",
        "ko": "절차서 파일에는 YAML 프론트매터(---)를 권장합니다. description 필드에서 자동 매칭이 활성화됩니다.",
    },
    "handler.procedure_frontmatter_recommended_short": {
        "ja": "手順書ファイルにはYAMLフロントマター(---)を推奨します。",
        "en": "Procedure files should have YAML frontmatter (---).",
    },
    "handler.read_before_write": {
        "ja": (
            "Error: 既存ファイル {path} を上書きする前に read_memory_file で読み取ってください。\n\n既存内容のプレビュー:\n{existing}"
        ),
        "en": ("Error: Read {path} with read_memory_file before overwriting.\n\nExisting content preview:\n{existing}"),
    },
    "handler.skill_read_before_write": {
        "ja": (
            "Error: 既存スキル {path} は、同じセッションで read_memory_file を使って先に読み直してから書き込んでください。\n\n既存内容のプレビュー:\n{existing}"
        ),
        "en": (
            "Error: Re-read existing skill {path} with read_memory_file in this session before writing it.\n\n"
            "Existing content preview:\n{existing}"
        ),
        "ko": (
            "오류: 기존 스킬 {path}을(를) 쓰기 전에 같은 세션에서 read_memory_file로 먼저 다시 읽어 주세요.\n\n"
            "기존 내용 미리보기:\n{existing}"
        ),
    },
    "handler.reason_prefix": {
        "ja": "理由: {reason}",
        "en": "Reason: {reason}",
    },
    "handler.restart_log": {
        "ja": "{target_name} を再起動リクエスト",
        "en": "Restart requested for {target_name}",
    },
    "handler.restart_success": {
        "ja": "'{target_name}' の再起動をリクエストしました。Reconciliation が 30 秒以内にプロセスを再起動します。",
        "en": ("Restart requested for '{target_name}'. Reconciliation will restart the process within 30 seconds."),
    },
    "handler.self_operation_denied": {
        "ja": "自分自身を操作することはできません",
        "en": "You cannot operate on yourself",
    },
    "handler.send_msg_chat_hint": {
        "ja": (
            "宛先 '{to}' には send_message で送信できません。人間宛ての返答はチャット本文ではなく外部DMで送る運用です。"
            " external_messaging.user_aliases の設定を確認し、必要なら call_human を使用してください。"
        ),
        "en": (
            "Cannot send to '{to}' via send_message. Human-directed replies should go via external DM rather than chat text. "
            "Check external_messaging.user_aliases and use call_human if needed."
        ),
    },
    "handler.call_human_confirm_required": {
        "ja": (
            "まだ送信していません。送る前に、この通知の内容について、全てチャットワーク、Slackなど外部で解決済みでないか必ず調査してください。"
            '調査して、それでも人間への通知が必要な内容だけを残し、call_human を sha="{sha}" 付きで再実行してください。'
        ),
        "en": (
            "Not sent yet. Before sending, you must check whether everything in this notification has already been resolved "
            'externally (Chatwork, Slack, etc.). Keep only what still needs a human, then call call_human again with sha="{sha}".'
        ),
    },
    "handler.call_human_confirm_unavailable": {
        "ja": "AnimaWorksサーバーに接続できないか応答を検証できないため、確認キーを確認できませんでした。接続を回復してから再実行してください。通知は送信されていません。",
        "en": "The call_human confirmation key could not be verified with the AnimaWorks server. Restore the connection and retry; the notification was not sent.",
    },
    "handler.send_msg_non_chat_hint": {
        "ja": (
            "宛先 '{to}' には send_message で送信できません。人間への連絡は call_human を使用してください。send_message は他のAnima宛てにのみ使用してください。"
        ),
        "en": (
            "Cannot send to '{to}' via send_message. Use call_human to contact humans. Use send_message only for other Animas."
        ),
    },
    "handler.shared_tool_denied": {
        "ja": "共有ツール作成が許可されていません。",
        "en": "Shared tool creation is not permitted.",
    },
    "handler.shared_tool_keyword": {
        "ja": "共有ツール",
        "en": "Shared Tool",
    },
    "handler.similar_knowledge_hint": {
        "ja": ("類似する既存の知識ファイル（トークン重複）:\n{files}"),
        "en": ("Similar existing knowledge files (token overlap):\n{files}"),
    },
    "handler.case_record_episodes_hint": {
        "ja": "案件記録は episodes/ 向きです。記録は knowledge/ に書き込み済みです。",
        "en": "Case-specific records are better suited to episodes/. The write to knowledge/ was completed.",
        "ko": "사건별 기록은 episodes/에 더 적합합니다. knowledge/ 쓰기는 완료되었습니다.",
    },
    "handler.since_hours": {
        "ja": "{hours}時間{minutes}分前",
        "en": "{hours}h {minutes}m ago",
    },
    "handler.since_minutes": {
        "ja": "{minutes}分前",
        "en": "{minutes}m ago",
    },
    "handler.skill_format_validation": {
        "ja": ("⚠️ スキルフォーマット検証:\n{msg}"),
        "en": ("⚠️ Skill format validation:\n{msg}"),
    },
    "handler.skill_frontmatter_required": {
        "ja": "スキルファイルにはYAMLフロントマター(---)が必要です。",
        "en": "Skill files require YAML frontmatter (---).",
        "zh": "技能文件需要 YAML 前言 (---)。",
        "ko": "스킬 파일에는 YAML 프론트매터(---)가 필요합니다.",
    },
    "handler.skill_name_required": {
        "ja": "skill_name パラメータは必須です。",
        "en": "skill_name parameter is required.",
    },
    "handler.state_current_state": {
        "ja": "### 現在の状態",
        "en": "### Current state",
    },
    "handler.state_none": {
        "ja": "(なし)",
        "en": "(none)",
    },
    "handler.state_read_summary": {
        "ja": "{target_name}の作業状態を読み取り",
        "en": "Read {target_name}'s work status",
    },
    "handler.state_title": {
        "ja": "## {target_name} の作業状態",
        "en": "## {target_name}'s work status",
    },
    "handler.state_unreadable": {
        "ja": "(読取不可)",
        "en": "(unreadable)",
    },
    "handler.subordinate_dir_list": {
        "ja": "配下のディレクトリ一覧",
        "en": "Descendant directory listing",
    },
    "handler.subordinate_disabled_warning": {
        "ja": ("\n⚠️ {target_name} は現在休止中です。タスクはキューに蓄積されますが、処理は再起動後になります。"),
        "en": ("\n⚠️ {target_name} is currently disabled. Tasks will queue until restart."),
    },
    "handler.subordinate_management": {
        "ja": "配下のcron.md, HEARTBEAT.md, status.json, AGENTS.md",
        "en": "Descendant's cron.md, HEARTBEAT.md, status.json, AGENTS.md",
    },
    "handler.task_add_log": {
        "ja": "タスク追加: {summary}",
        "en": "Task added: {summary}",
    },
    "handler.task_tracker_log": {
        "ja": "委譲タスク追跡 (filter={status}, count={count})",
        "en": "Task tracker (filter={status}, count={count})",
    },
    "handler.task_update_log": {
        "ja": "タスク更新: {summary} → {status}",
        "en": "Task updated: {summary} → {status}",
    },
    "handler.task_owned_by_other": {
        "ja": "このタスクは {owner} が持っています。再開や完了は {owner} が行います。必要なら send_message で {owner} に伝えてください。",
        "en": "This task is owned by {owner}. {owner} must resume or complete it. If needed, ask {owner} with send_message.",
        "ko": "이 작업은 {owner}이(가) 맡고 있습니다. 재개나 완료는 {owner}이(가) 직접 해야 합니다. 필요하면 send_message로 {owner}에게 알려 주세요.",
    },
    "task_update.scope_note_notify": {
        "ja": "{owner} がタスク「{title}」の範囲を書き留めました: {note}",
        "en": '{owner} noted the scope of task "{title}": {note}',
        "ko": '{owner}이(가) 태스크 "{title}"의 범위를 기록했습니다: {note}',
    },
    "task_update.cancel_reason_missing": {
        "ja": "理由は指定されていません",
        "en": "No reason was provided",
        "ko": "사유가 지정되지 않았습니다",
    },
    "task_update.cancelled_by_requester": {
        "ja": "依頼元の {requester} が依頼を取り下げました。理由: {reason}",
        "en": "Requester {requester} withdrew the request. Reason: {reason}",
        "ko": "요청자 {requester}이(가) 요청을 철회했습니다. 사유: {reason}",
    },
    "task_update.cancelled_by_owner": {
        "ja": "{owner} がタスクを取り下げました。理由: {reason}",
        "en": "{owner} cancelled the task. Reason: {reason}",
        "ko": "{owner}이(가) 작업을 취소했습니다. 사유: {reason}",
    },
    "handler.skill_scan_safe": {
        "ja": "🔒 セキュリティスキャン: 安全 (脅威なし)",
        "en": "🔒 Security scan: safe (no threats detected)",
    },
    "handler.skill_scan_dangerous": {
        "ja": "🚨 セキュリティスキャン: 危険 ({count}件の脅威検出, カテゴリ: {categories})。このスキルはブロックされます。",
        "en": "🚨 Security scan: dangerous ({count} threat(s) detected, categories: {categories}). This skill is blocked.",
    },
    "handler.skill_scan_warning": {
        "ja": "⚠️ セキュリティスキャン: {verdict} ({count}件の検出あり)",
        "en": "⚠️ Security scan: {verdict} ({count} finding(s) detected)",
    },
    "handler.tool_creation_denied": {
        "ja": "ツール作成が許可されていません。permissions.json の許可設定を確認してください。",
        "en": "Tool creation is not permitted. Check the permissions in permissions.json.",
    },
}
