# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Domain-specific i18n strings."""

from __future__ import annotations

STRINGS: dict[str, dict[str, str]] = {
    "anima.agent_error": {
        "ja": "[ERROR: エージェント実行中にエラーが発生しました]",
        "en": "[ERROR: An error occurred during agent execution]",
    },
    "anima.bg_notif_result": {
        "ja": "- 結果: {summary}",
        "en": "- Result: {summary}",
    },
    "anima.bg_notif_status": {
        "ja": "- ステータス: {status}",
        "en": "- Status: {status}",
    },
    "anima.bg_notif_task_id": {
        "ja": "- タスクID: {task_id}",
        "en": "- Task ID: {task_id}",
    },
    "anima.bg_notif_tool": {
        "ja": "- ツール: {tool}",
        "en": "- Tool: {tool}",
    },
    "anima.bg_task_done": {
        "ja": "バックグラウンドタスク完了: {tool}",
        "en": "Background task completed: {tool}",
    },
    "anima.bg_task_failed": {
        "ja": "バックグラウンドタスク失敗: {tool}",
        "en": "Background task failed: {tool}",
    },
    "anima.bootstrap_prompt": {
        "ja": "あなたの bootstrap.md ファイルを読み、指示に従ってください。",
        "en": "Read your bootstrap.md file and follow its instructions.",
    },
    "anima.chat_reply_delivery_failed": {
        "ja": "{to} への{via}送信に失敗しました。内容はチャットには表示しません。",
        "en": "Failed to send this reply to {to} via {via}. The content is not shown in chat.",
    },
    "anima.chat_reply_sent_via_external": {
        "ja": "この返答は {to} へ{via}で送信しました。",
        "en": "This reply was sent to {to} via {via}.",
    },
    "anima.consolidation_end": {
        "ja": "{type}記憶統合完了",
        "en": "{type} consolidation completed",
    },
    "anima.consolidation_error": {
        "ja": "記憶統合エラー: {exc}",
        "en": "Consolidation error: {exc}",
    },
    "anima.consolidation_start": {
        "ja": "{type}記憶統合開始",
        "en": "{type} consolidation started",
    },
    "anima.cron_cmd_error": {
        "ja": "run_cron_commandエラー: {exc}",
        "en": "run_cron_command error: {exc}",
    },
    "anima.cron_cmd_summary": {
        "ja": "コマンド: {task}",
        "en": "Command: {task}",
    },
    "anima.cron_task_error": {
        "ja": "run_cron_taskエラー: {exc}",
        "en": "run_cron_task error: {exc}",
    },
    "anima.cron_task_summary": {
        "ja": "タスク: {task}",
        "en": "Task: {task}",
    },
    "anima.delivery_via_chatwork": {
        "ja": "Chatwork",
        "en": "Chatwork",
    },
    "anima.delivery_via_external": {
        "ja": "外部DM",
        "en": "external DM",
    },
    "anima.delivery_via_slack": {
        "ja": "Slack DM",
        "en": "Slack DM",
    },
    "scheduler.cron_health_title": {
        "ja": "⚠️ Cronヘルスチェック警告",
        "en": "⚠️ Cron Health Check Warning",
    },
    "scheduler.cron_health_no_valid_schedule": {
        "ja": (
            "{task_count}件のタスク定義がありますが、有効なスケジュールが0件です。"
            "cron.mdのフォーマットを確認してください。"
        ),
        "en": ("{task_count} task(s) defined but 0 valid schedules. Please check cron.md format."),
    },
    "scheduler.cron_health_indented_schedule": {
        "ja": ("cron.mdにインデント付きの schedule: 行が検出されました。パーサーは行頭の schedule: のみ認識します。"),
        "en": (
            "Indented 'schedule:' lines detected in cron.md. "
            "The parser only recognizes 'schedule:' at the start of a line."
        ),
    },
    "scheduler.cron_health_unrecognized_schedule": {
        "ja": (
            "cron.mdに schedule: を含む行がありますが、パーサーに認識されていません。フォーマットを確認してください。"
        ),
        "en": ("Lines containing 'schedule:' found in cron.md but not recognized by the parser. Please check format."),
    },
    "scheduler.cron_health_no_execution": {
        "ja": (
            "{job_count}件のcronジョブが登録されていますが、"
            "直近{hours}時間でcron実行が0件です。"
            "cron.mdとスケジュール設定を確認してください。"
        ),
        "en": (
            "{job_count} cron job(s) registered but 0 executions in the last "
            "{hours} hours. Please check cron.md and schedule settings."
        ),
    },
    "anima.greeting_error": {
        "ja": "[ERROR: 挨拶生成中にエラーが発生しました]",
        "en": "[ERROR: An error occurred during greeting generation]",
    },
    "anima.heartbeat_episode": {
        "ja": ("## {ts} ハートビート活動\n\n{summary}"),
        "en": ("## {ts} Heartbeat activity\n\n{summary}"),
    },
    "anima.heartbeat_error": {
        "ja": "run_heartbeatエラー: {exc}",
        "en": "run_heartbeat error: {exc}",
    },
    "anima.heartbeat_msgs_processed": {
        "ja": ("\n\n（{count}件のメッセージを処理）"),
        "en": ("\n\n({count} messages processed)"),
    },
    "anima.heartbeat_start": {
        "ja": "定期巡回開始",
        "en": "Periodic check started",
    },
    "anima.inbox_error": {
        "ja": "inbox処理エラー: {exc}",
        "en": "inbox processing error: {exc}",
    },
    "anima.inbox_start": {
        "ja": "Inbox MSG処理開始",
        "en": "Inbox message processing started",
    },
    "anima.initializing": {
        "ja": "現在初期化中です。しばらくお待ちください。",
        "en": "Initializing. Please wait.",
    },
    "anima.msg_received_episode": {
        "ja": ("## {ts} {from_person}からのメッセージ受信\n\n**送信者**: {from_person}\n**内容**:\n{content}"),
        "en": ("## {ts} Message received from {from_person}\n\n**Sender**: {from_person}\n**Content**:\n{content}"),
    },
    "anima.no_activity_log": {
        "ja": "(アクティビティログなし)",
        "en": "(No activity log)",
    },
    "anima.no_episodes_today": {
        "ja": "(本日のエピソードはありません)",
        "en": "(No episodes today)",
    },
    "anima.platform_context": {
        "ja": (
            "[platform_context: このメッセージは {source} 経由で受信しました。あなたのテキスト応答は自動的に {source} 経由で送信者に返されます。send_message ツールで別チャネルへの送信を試みないでください。]"
        ),
        "en": (
            "[platform_context: This message was received via {source}. Your text response will be automatically sent back to the sender via {source}. Do NOT attempt to send via another channel using the send_message tool.]"
        ),
    },
    "anima.process_message_error": {
        "ja": "process_messageエラー: {exc}",
        "en": "process_message error: {exc}",
    },
    "anima.process_stream_error": {
        "ja": "process_message_streamエラー: {exc}",
        "en": "process_message_stream error: {exc}",
    },
    "anima.recovery_crash_info": {
        "ja": (
            "### クラッシュ復旧情報\n\n- 発生日時: {ts}\n- トリガー: {trigger}\n- 回復テキスト長: {recovered_chars}文字\n- ツール呼び出し数: {tool_calls}回\n- 原因: プロセスが予期せず終了しました（SIGKILL/OOM等の可能性）"
        ),
        "en": (
            "### Crash recovery information\n\n- Occurred at: {ts}\n- Trigger: {trigger}\n- Recovered text length: {recovered_chars} chars\n- Tool calls: {tool_calls}\n- Cause: Process terminated unexpectedly (possible SIGKILL/OOM)"
        ),
    },
    "anima.recovery_error_info": {
        "ja": (
            "### エラー情報\n\n- エラー種別: {exc_type}\n- エラー内容: {exc_msg}\n- 発生日時: {ts}\n- 未処理メッセージ数: {count}"
        ),
        "en": (
            "### Error information\n\n- Error type: {exc_type}\n- Error message: {exc_msg}\n- Occurred at: {ts}\n- Unprocessed message count: {count}"
        ),
    },
    "anima.reflections_header": {
        "ja": "振り返り（REFLECTION）",
        "en": "Reflections",
    },
    "anima.reflections_intro": {
        "ja": "エピソード中の [REFLECTION] タグから抽出された意識的な洞察です。優先的に知識化を検討してください。",
        "en": (
            "Conscious insights extracted from [REFLECTION] tags in episodes. Prioritize these for knowledge extraction."
        ),
    },
    "anima.response_interrupted": {
        "ja": "[応答が中断されました]",
        "en": "[Response was interrupted]",
    },
    "anima.response_interrupted_prefix": {
        "ja": ("\n[応答が中断されました]"),
        "en": ("\n[Response was interrupted]"),
    },
    "anima.status_idle": {
        "ja": "待機中",
        "en": "Idle",
    },
    "anima.task_none": {
        "ja": "特になし",
        "en": "None",
    },
    "anima.unread_prefix": {
        "ja": "- {from_person} [⚠️ 未返信{count}回目]: ",
        "en": "- {from_person} [⚠️ Unreplied #{count}]: ",
    },
    "anima.visit_desk": {
        "ja": "[デスクを訪問]",
        "en": "[Desk visit]",
    },
    "model_config.credential_auto_switch": {
        "ja": "モデルファミリー変更を検出: credential を '{old}' → '{new}' に自動切替しました",
        "en": "Model family change detected: auto-switched credential from '{old}' to '{new}'",
    },
    "model_config.credential_fallback_defaults": {
        "ja": "ファミリー '{family}' の credential が見つかりません。デフォルト '{default}' を使用します",
        "en": "No credential found for family '{family}'. Using default '{default}'",
    },
    "model_config.credential_keep_current": {
        "ja": "適切な credential を自動解決できません。現在の credential '{current}' を維持します",
        "en": "Could not auto-resolve credential. Keeping current credential '{current}'",
    },
    "config.anima_count_detail": {
        "ja": "{count}名",
        "en": "{count} anima(s)",
    },
    "config.anima_registration": {
        "ja": "Anima登録",
        "en": "Anima registration",
    },
    "config.anthropic_api_key": {
        "ja": "Anthropic APIキー",
        "en": "Anthropic API key",
    },
    "config.anthropic_auth": {
        "ja": "Anthropic APIキー / サブスクリプション認証",
        "en": "Anthropic API key / Subscription auth",
    },
    "config.config_file": {
        "ja": "設定ファイル",
        "en": "Config file",
    },
    "config.google_api_key": {
        "ja": "Google APIキー",
        "en": "Google API key",
    },
    "config.init_complete": {
        "ja": "初期化完了",
        "en": "Initialization complete",
    },
    "config.openai_api_key": {
        "ja": "OpenAI APIキー",
        "en": "OpenAI API key",
    },
    "config.openai_api_key_required": {
        "ja": "auth_mode=api_key の場合、api_key は必須です",
        "en": "api_key is required for auth_mode=api_key",
    },
    "config.openai_auth": {
        "ja": "OpenAI APIキー / Codex Login",
        "en": "OpenAI API key / Codex login",
    },
    "config.openai_auth_invalid_mode": {
        "ja": "auth_mode は 'api_key' または 'codex_login' である必要があります",
        "en": "auth_mode must be 'api_key' or 'codex_login'",
    },
    "config.codex_cli_not_installed": {
        "ja": "Codex CLI がインストールされていません",
        "en": "Codex CLI is not installed",
    },
    "config.codex_login_not_available": {
        "ja": "Codex ログインが利用できません",
        "en": "Codex login is not available",
    },
    "config.shared_dir": {
        "ja": "共有ディレクトリ",
        "en": "Shared directory",
    },
}
