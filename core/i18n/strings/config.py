# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Domain-specific i18n strings."""

from __future__ import annotations

STRINGS: dict[str, dict[str, str]] = {
    "config.status_heartbeat_override": {
        "ja": "status.json の {anima}.{field} に保存し、root のハートビートスケジュールに反映します。",
        "en": "Saves {anima}.{field} to root-owned status.json and updates the root heartbeat schedule.",
        "ko": "root 소유 status.json의 {anima}.{field}에 저장하고 root 하트비트 스케줄에 반영합니다.",
    },
    "config.status_org_setting": {
        "ja": "root 所有の {anima}.{field} 組織設定を status.json と config.json に反映します。",
        "en": "Updates root-owned organization field {anima}.{field} in status.json and config.json.",
        "ko": "root 소유 {anima}.{field} 조직 설정을 status.json과 config.json에 반영합니다.",
    },
    "config.status_field_deprecated": {
        "ja": "Warning: '{path}' は非推奨です。\n  status.json に書き込みます。今後は 'animaworks anima set-model' を使用してください。",
        "en": "Warning: '{path}' is deprecated.\n  Writes to status.json. Use 'animaworks anima set-model' instead.",
        "ko": "경고: '{path}' 경로는 더 이상 사용하지 않습니다.\n  status.json에 저장됩니다. 'animaworks anima set-model'을 사용하세요.",
    },
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
        "ja": "⚠️ cron.md解析警告",
        "en": "⚠️ cron.md Parse Warning",
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
    "scheduler.cron_health_high_frequency_llm": {
        "ja": (
            "- 「{task_name}」は高頻度で LLM を起動します。新着の有無などシェルで判定できるなら、"
            "`type: command` ＋ `skip_pattern` にしてください。"
        ),
        "en": (
            "- '{task_name}' starts an LLM very frequently. If a shell command can detect whether "
            "there is anything new, use `type: command` + `skip_pattern`."
        ),
    },
    "scheduler.cron_health_command_without_skip": {
        "ja": (
            "- 「{task_name}」は command 型で `skip_pattern` が無いため、出力のたびに LLM が起動します。"
            "通知をスクリプト自身が行うなら `trigger_heartbeat: false` にしてください。"
        ),
        "en": (
            "- '{task_name}' is a command task without `skip_pattern`, so every output starts an LLM. "
            "If the script notifies by itself, set `trigger_heartbeat: false`."
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
    "anima.first_meeting_marker": {
        "ja": "ユーザーがセットアップを終え、初めてあなたのチャットを開きました",
        "en": "The user finished setup and opened your chat for the first time",
        "ko": "사용자가 설정을 마치고 처음으로 당신의 채팅을 열었습니다",
    },
    "anima.first_meeting_default_user": {
        "ja": "ユーザー",
        "en": "the user",
        "ko": "사용자",
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
    "config.file_roots_inside_data_dir": {
        "ja": (
            "file_roots のパスは data_dir 配下に置けません: {path}。"
            "作業ディレクトリは data_dir 外に置くか companies/<社>/shared を使ってください。"
        ),
        "en": (
            "file_roots must not point inside data_dir: {path}. "
            "Put the working directory outside data_dir, or use companies/<company>/shared."
        ),
        "ko": (
            "file_roots 경로는 data_dir 아래에 둘 수 없습니다: {path}. "
            "작업 디렉터리를 data_dir 밖에 두거나 companies/<회사>/shared를 사용하세요."
        ),
    },
    "enclave.guard.name_required": {
        "ja": "enclave に name が設定されていません",
        "en": "enclave has no name set",
    },
    "enclave.guard.socket_path_required": {
        "ja": "enclave に socket_path が設定されていません",
        "en": "enclave has no socket_path set",
    },
    "enclave.guard.entry_anima_required": {
        "ja": "enclave に entry_anima が設定されていません",
        "en": "enclave has no entry_anima set",
    },
    "enclave.guard.egress_invalid": {
        "ja": "enclave の egress 設定が不正です（stages を1つ以上、既知の type で設定する必要があります）",
        "en": "enclave egress config is invalid (at least one stage of a known type is required)",
    },
    "enclave.guard.peer_uids_required": {
        "ja": "enclave の allowed_peer_uids が空です（少なくとも1つ設定する必要があります）",
        "en": "enclave allowed_peer_uids is empty (at least one must be set)",
    },
    "enclave.guard.egress_probe_reachable": {
        "ja": "enclave の egress_probe の宛先 {target} に接続できました（外向き通信の制限が効いていません）",
        "en": "enclave egress_probe target {target} is reachable (outbound traffic is not restricted)",
    },
    "enclave.guard.socket_group_missing": {
        "ja": "enclave の socket_group '{group}' が存在しません",
        "en": "enclave socket_group '{group}' does not exist",
    },
    "enclave.gateway.answer_instruction": {
        "ja": (
            "結論を日本語でそのまま書いて返す（形式は自由）。"
            "根拠には record: などの ID を添える。個人を特定できる値は書かない。"
            "回答は伏字処理を通ってから依頼者に届く"
        ),
        "en": (
            "Write the conclusion as plain text in any format. "
            "Cite evidence IDs such as record:. Do not write values that identify a specific person. "
            "The answer is masked before it reaches the requester."
        ),
    },
    "enclave.gateway.progress_instruction": {
        "ja": (
            "この仕事の識別子（job_id）は {job_id} です。"
            "分析が長引く場合は、対象の特定・データ取得・原因の候補など節目ごとに、"
            "途中経過を1〜2文だけ書きなさい。書き方: "
            '`animaworks-tool enclave_progress enclave_progress --job-id {job_id} --text "..."`。'
            "途中経過は伏字処理を通って依頼者に見えるので、個人を特定できる値は書かないこと。"
        ),
        "en": (
            "The job identifier for this work is {job_id}. "
            "If this analysis will take a while, write a short interim update (one or two sentences) "
            "at milestones such as target identification, data retrieval, or candidate causes: "
            '`animaworks-tool enclave_progress enclave_progress --job-id {job_id} --text "..."`. '
            "Updates are masked and shown to the requester, so never include identifying values."
        ),
    },
    "enclave.progress.job_required": {
        "ja": "job_id を指定してください。",
        "en": "A job_id is required.",
    },
    "enclave.progress.text_required": {
        "ja": "text に途中経過を1〜2文で指定してください。",
        "en": "Provide a one-to-two sentence progress text.",
    },
    "enclave.progress.not_found": {
        "ja": "job '{job_id}' が見つかりません。",
        "en": "job '{job_id}' was not found.",
    },
    "enclave.progress.not_running": {
        "ja": "このジョブは実行中ではないため、途中経過を書けません。",
        "en": "This job is not running, so progress cannot be written.",
    },
    "enclave.progress.unavailable": {
        "ja": "途中経過を書き込めませんでした。設定と job の状態を確認してください。",
        "en": "Could not write progress. Check the configuration and job state.",
    },
    "enclave.progress.schema_description": {
        "ja": "実行中の enclave ask ジョブに途中経過を1〜2文だけ書き込みます。",
        "en": "Write a short interim progress note to a running enclave ask job.",
    },
    "enclave.progress.schema_job_id": {
        "ja": "ゲートウェイから渡された job_id",
        "en": "The job_id given by the gateway",
    },
    "enclave.progress.schema_text": {
        "ja": "途中経過の短文（1〜2文）",
        "en": "A short progress note (one or two sentences)",
    },
    "enclave.tool.unknown_enclave": {
        "ja": "エラー: enclave '{enclave}' が設定に存在しません",
        "en": "error: enclave '{enclave}' is not configured",
    },
    "enclave.tool.anima_not_allowed": {
        "ja": "エラー: この anima は enclave へのアクセスを許可されていません",
        "en": "error: this anima is not allowed to access the enclave",
    },
    "enclave.tool.unreachable": {
        "ja": "エラー: enclave に接続できません",
        "en": "error: could not reach the enclave",
    },
    "enclave.tool.failed": {
        "ja": "エラー: enclave が応答を返しませんでした (code={code})",
        "en": "error: enclave returned an error (code={code})",
    },
    "enclave.tool.schema_description": {
        "ja": "隔離された enclave に質問を送り、job_id をすぐ受け取る。状態を確認でき、完了結果は inbox に届く",
        "en": "Submit a question to an isolated enclave and get a job ID immediately. Check its status; completed results arrive in the inbox.",
    },
    "enclave.tool.job_submitted": {
        "ja": "enclave のジョブを受け付けました。job_id: {job_id} / 状態: {status}\n状態確認: {status_command}\n完了時に結果が inbox に届きます。",
        "en": "Enclave job accepted. job_id: {job_id} / status: {status}\nCheck status: {status_command}\nThe result will be delivered to your inbox when complete.",
    },
    "enclave.tool.job_done_no_facts": {
        "ja": "ジョブ {job_id} は完了しましたが、事実は返されませんでした。",
        "en": "Job {job_id} completed without returning facts.",
    },
    "enclave.tool.job_status_header": {
        "ja": "ジョブ状態: {status} (job_id: {job_id}, case_id: {case_id})",
        "en": "Job status: {status} (job_id: {job_id}, case_id: {case_id})",
    },
    "enclave.tool.elapsed": {
        "ja": "経過時間: {elapsed}",
        "en": "Elapsed: {elapsed}",
    },
    "enclave.tool.queued_ahead": {
        "ja": "先行待ちジョブ数: {queued_ahead}",
        "en": "Jobs queued ahead: {queued_ahead}",
    },
    "enclave.tool.last_activity": {
        "ja": "最終アクティビティ: {last_activity_at}",
        "en": "Last activity: {last_activity_at}",
    },
    "enclave.tool.progress_line": {
        "ja": "- [{at}] {text}",
        "en": "- [{at}] {text}",
    },
    "enclave.tool.job_error": {
        "ja": "エラー: {error}",
        "en": "Error: {error}",
    },
    "enclave.tool.job_cancelled": {
        "ja": "キャンセル要求を処理しました。job_id: {job_id} / 状態: {status}",
        "en": "Cancellation request processed. job_id: {job_id} / status: {status}",
    },
    "enclave.tool.job_id_required": {
        "ja": "エラー: job_id を指定してください",
        "en": "error: job_id is required",
    },
    "enclave.tool.audit_id": {
        "ja": "audit_id: {audit_id}",
        "en": "audit_id: {audit_id}",
    },
    "enclave.tool.schema_status_description": {
        "ja": "enclave ジョブの状態・経過時間・途中経過・完了時の事実を確認する",
        "en": "Check an enclave job's status, elapsed time, progress, and facts when complete",
    },
    "enclave.tool.schema_cancel_description": {
        "ja": "実行中または待機中の enclave ジョブをキャンセルする",
        "en": "Cancel a queued or running enclave job",
    },
    "enclave.tool.schema_job_id": {
        "ja": "enclave ask が返したジョブID",
        "en": "Job ID returned by enclave ask",
    },
    "enclave.tool.schema_wait_s": {
        "ja": "結果確認のために待つ最大秒数（既定値: 0。0 の場合はすぐ戻る）",
        "en": "Maximum seconds to wait for a result (default: 0, which returns immediately)",
    },
    "enclave.delivery.case_id": {
        "ja": "case_id: {case_id}",
        "en": "case_id: {case_id}",
    },
    "enclave.delivery.job_id": {
        "ja": "job_id: {job_id}",
        "en": "job_id: {job_id}",
    },
    "enclave.delivery.status": {
        "ja": "状態: {status}",
        "en": "status: {status}",
    },
    "enclave.delivery.audit_id": {
        "ja": "audit_id: {audit_id}",
        "en": "audit_id: {audit_id}",
    },
    "enclave.delivery.error": {
        "ja": "エラー: {error}",
        "en": "error: {error}",
    },
    "enclave.delivery.no_facts": {
        "ja": "事実は返されませんでした。",
        "en": "No facts were returned.",
    },
    "enclave.delivery.invalid_sender": {
        "ja": "enclave {enclave} からの未配達ジョブ {job_id} は、許可されていない呼び手または存在しない anima のため破棄しました。",
        "en": "Discarded undelivered enclave job {job_id} from {enclave}: the caller is not allowed or does not exist on this host.",
    },
    "enclave.delivery.gateway_warning": {
        "ja": "enclave {enclave} の結果取り込みに失敗しました: {reason}",
        "en": "Could not collect enclave results from {enclave}: {reason}",
    },
    "enclave.delivery.job_warning": {
        "ja": "enclave {enclave} のジョブ {job_id} の配達に失敗しました: {reason}",
        "en": "Could not deliver enclave job {job_id} from {enclave}: {reason}",
    },
    "enclave.delivery.ack_warning": {
        "ja": "enclave {enclave} のジョブ {job_id} を配達済みにできませんでした: HTTP {code}",
        "en": "Could not acknowledge enclave job {job_id} from {enclave}: HTTP {code}",
    },
    "enclave.tool.schema_enclave": {
        "ja": "接続する enclave の設定名",
        "en": "name of the enclave config to connect to",
    },
    "enclave.tool.schema_question": {
        "ja": "隔離インスタンスに送る質問",
        "en": "the question to send to the isolated instance",
    },
    "enclave.tool.schema_model": {
        "ja": "省略可なモデル指定（enclave 側の model_profiles の名前。例: luna-max / opus-high。省略時は enclave の既定）",
        "en": "optional model profile name from the enclave's model_profiles (e.g. luna-max / opus-high; defaults to the enclave's default)",
    },
    "enclave.tool.unknown_model_profile": {
        "ja": "エラー: 未知のモデル指定です。使えるもの: {available}",
        "en": "error: unknown model profile. Available: {available}",
    },
    "enclave.tool.model_profile": {
        "ja": "モデル: {model_profile}",
        "en": "Model: {model_profile}",
    },
    "enclave.tool.schema_case_id": {
        "ja": "省略可なケース識別子（省略時は anima 名と日付から生成）",
        "en": "optional case identifier (defaults to anima name plus date)",
    },
    "enclave.tool.fact_line": {
        "ja": "- {text}（根拠: {evidence}）",
        "en": "- {text} (evidence: {evidence})",
    },
    "enclave.records.only": {
        "ja": "このツールは enclave 内でだけ使えます。",
        "en": "This tool can only be used inside an enclave.",
    },
    "enclave.records.dataset_required": {
        "ja": "データセット名を指定してください。",
        "en": "A dataset name is required.",
    },
    "enclave.records.query_required": {
        "ja": "検索語を指定してください。",
        "en": "A search query is required.",
    },
    "enclave.records.limit_integer": {
        "ja": "limit は整数で指定してください。",
        "en": "limit must be an integer.",
    },
    "enclave.records.id_required": {
        "ja": "record_id を指定してください。",
        "en": "record_id is required.",
    },
    "enclave.records.dataset_not_configured": {
        "ja": "データセット '{dataset}' は enclave に設定されていません。",
        "en": "Dataset '{dataset}' is not configured in the enclave.",
    },
    "enclave.records.not_found": {
        "ja": "ID '{record_id}' のレコードは見つかりません。",
        "en": "No record was found for ID '{record_id}'.",
    },
    "enclave.records.unavailable": {
        "ja": "データセットを安全に参照できません。設定とファイルを確認してください。",
        "en": "The dataset could not be read safely. Check its configuration and file.",
    },
    "enclave.records.schema_search": {
        "ja": "隔離 enclave 内の設定済み JSONL データを、検索可能な項目から検索します。",
        "en": "Search configured JSONL datasets using their searchable fields inside the enclave.",
    },
    "enclave.records.schema_get": {
        "ja": "隔離 enclave 内の設定済み JSONL データから、ID を指定して 1 件取得します。",
        "en": "Fetch one configured JSONL record by ID from inside the enclave.",
    },
    "enclave.records.schema_dataset": {
        "ja": "設定済みデータセット名",
        "en": "Configured dataset name",
    },
    "enclave.records.schema_query": {
        "ja": "検索する文字列",
        "en": "Text to search for",
    },
    "enclave.records.schema_limit": {
        "ja": "返すレコードの最大件数（1〜100）",
        "en": "Maximum records to return (1-100)",
    },
    "enclave.records.schema_record_id": {
        "ja": "データセットの ID フィールド値",
        "en": "Value of the dataset's ID field",
    },
    "enclave.sql.only": {
        "ja": "このツールは enclave 内でだけ使えます。",
        "en": "This tool can only be used inside an enclave.",
    },
    "enclave.sql.source_required": {
        "ja": "データソース名を指定してください。",
        "en": "A data source name is required.",
    },
    "enclave.sql.source_not_configured": {
        "ja": "データソース '{source}' は enclave に設定されていません。",
        "en": "Data source '{source}' is not configured in the enclave.",
    },
    "enclave.sql.sql_required": {
        "ja": "SQL を指定してください。",
        "en": "An SQL statement is required.",
    },
    "enclave.sql.multiple_statements": {
        "ja": "SQL は 1 文で指定してください（セミコロン区切りの複数文は使えません）。",
        "en": "Provide a single SQL statement (semicolon-separated statements are not allowed).",
    },
    "enclave.sql.sql_not_allowed": {
        "ja": "この SQL は読み取り専用の安全な形式ではありません。",
        "en": "This SQL is not in a safe read-only form.",
    },
    "enclave.sql.unavailable": {
        "ja": "データソースを安全に参照できません。設定と秘密を確認してください。",
        "en": "The data source could not be read safely. Check its configuration and secrets.",
    },
    "enclave.sql.schema_query": {
        "ja": "隔離 enclave 内の設定済み読取専用 DB に、検証済みの SELECT を実行します。",
        "en": "Run a validated read-only query against a configured enclave database.",
    },
    "enclave.sql.schema_schema": {
        "ja": "隔離 enclave 内の設定済み読取専用 DB のスキーマ（テーブル一覧か列定義）を返します。",
        "en": "Return the schema (table list or column definitions) of a configured enclave database.",
    },
    "enclave.sql.schema_source": {
        "ja": "設定済みデータソース名",
        "en": "Configured data source name",
    },
    "enclave.sql.schema_sql": {
        "ja": "実行する検証済み SELECT 文",
        "en": "Validated SELECT statement to run",
    },
    "enclave.sql.schema_table": {
        "ja": "省略可なテーブル名（指定時は列定義を返す）",
        "en": "Optional table name (returns column definitions when set)",
    },
    "enclave.sql.raw_note": {
        "ja": "全文は raw_path のファイルを Read で読んでください。",
        "en": "Read the file at raw_path to access the full query result.",
    },
    "enclave.aws.only": {
        "ja": "このツールは enclave 内でだけ使えます。",
        "en": "This tool can only be used inside an enclave.",
    },
    "enclave.aws.source_required": {
        "ja": "AWS データソース名を指定してください。",
        "en": "An AWS data source name is required.",
    },
    "enclave.aws.source_not_configured": {
        "ja": "AWS データソースが enclave に設定されていません。",
        "en": "The AWS data source is not configured in the enclave.",
    },
    "enclave.aws.target_not_allowed": {
        "ja": "対象の AWS リソースは設定または許可されていません。",
        "en": "The requested AWS resource is not configured or allowed.",
    },
    "enclave.aws.invalid_input": {
        "ja": "AWS 読み取りツールの引数が無効です。",
        "en": "An AWS read-tool argument is invalid.",
    },
    "enclave.aws.query_timeout": {
        "ja": "Logs Insights クエリが制限時間内に完了しませんでした。",
        "en": "The Logs Insights query did not complete within the time limit.",
    },
    "enclave.aws.invalid_secret": {
        "ja": "AWS 認証情報の秘密が有効な形式ではありません。",
        "en": "The AWS credential secret is not in a valid format.",
    },
    "enclave.aws.dependencies_missing": {
        "ja": "AWS 読み取りに必要な依存パッケージがありません。",
        "en": "A dependency required for AWS reads is unavailable.",
    },
    "enclave.aws.unavailable": {
        "ja": "AWS データを安全に読み取れません。設定と秘密を確認してください。",
        "en": "AWS data could not be read safely. Check the configuration and secrets.",
    },
    "enclave.aws.schema_source": {
        "ja": "enclave.aws_sources に設定された AWS データソース名",
        "en": "AWS data source configured under enclave.aws_sources",
    },
    "enclave.aws.schema_time": {
        "ja": "now、ISO8601 時刻、または -1h / -24h 形式の相対時刻",
        "en": "now, an ISO8601 time, or a relative duration such as -1h or -24h",
    },
    "enclave.aws.schema_logs_query": {
        "ja": "許可された CloudWatch Logs グループで Logs Insights クエリを実行します。",
        "en": "Run a Logs Insights query against an allowed CloudWatch Logs group.",
    },
    "enclave.aws.schema_logs_filter": {
        "ja": "許可された CloudWatch Logs グループからイベントを取得します。",
        "en": "Fetch events from an allowed CloudWatch Logs group.",
    },
    "enclave.aws.schema_log_group": {
        "ja": "enclave.aws_sources で許可されたロググループ名",
        "en": "Log group allowed by enclave.aws_sources",
    },
    "enclave.aws.schema_query": {
        "ja": "実行する Logs Insights クエリ",
        "en": "Logs Insights query to run",
    },
    "enclave.aws.schema_filter_pattern": {
        "ja": "CloudWatch Logs のフィルターパターン（空文字は全イベント）",
        "en": "CloudWatch Logs filter pattern (empty means all events)",
    },
    "enclave.aws.schema_pi_top_sql": {
        "ja": "RDS Performance Insights から負荷上位の SQL ディメンションを取得します。",
        "en": "Return top SQL dimensions from RDS Performance Insights.",
    },
    "enclave.aws.schema_pi_sql_detail": {
        "ja": "Performance Insights の SQL ディメンション詳細（SQL 全文）を取得します。",
        "en": "Fetch Performance Insights SQL dimension details, including the full SQL text.",
    },
    "enclave.aws.schema_group_identifier": {
        "ja": "enclave_pi_top_sql が返した SQL グループ識別子",
        "en": "SQL group identifier returned by enclave_pi_top_sql",
    },
    "enclave.aws.schema_rds_logs": {
        "ja": "設定済み RDS インスタンスのログファイル一覧または一部を取得します。",
        "en": "List log files or fetch a portion of a configured RDS instance log.",
    },
    "enclave.aws.schema_log_file_name": {
        "ja": "取得する RDS ログファイル名（省略時は一覧）",
        "en": "RDS log file name to fetch (omit to list files)",
    },
    "enclave.aws.schema_marker": {
        "ja": "前回応答の継続マーカー",
        "en": "Continuation marker from the previous response",
    },
    "enclave.aws.schema_s3_get": {
        "ja": "許可された S3 バケットからオブジェクトを読み取ります。",
        "en": "Read an object from an allowed S3 bucket.",
    },
    "enclave.aws.schema_s3_list": {
        "ja": "許可された S3 バケットのオブジェクトキーを一覧します。",
        "en": "List object keys in an allowed S3 bucket.",
    },
    "enclave.aws.schema_bucket": {
        "ja": "enclave.aws_sources で許可されたバケット名",
        "en": "Bucket allowed by enclave.aws_sources",
    },
    "enclave.aws.schema_key": {
        "ja": "S3 オブジェクトキー",
        "en": "S3 object key",
    },
    "enclave.cli.doctor_json": {
        "ja": "機械可読な JSON 形式で出力する",
        "en": "Output in machine-readable JSON format",
    },
    "enclave.guard.entry_anima_missing": {
        "ja": "entry_anima '{anima}' のディレクトリが存在しません",
        "en": "entry_anima '{anima}' directory does not exist",
    },
    "enclave.guard.event_export_url": {
        "ja": "event_export.url が設定されています（enclave では外部送信を無効にする必要があります）",
        "en": "event_export.url is set (external export must be disabled in an enclave)",
    },
    "enclave.guard.external_enabled": {
        "ja": "外部連携 {channel} が有効になっています（enclave では無効にする必要があります）",
        "en": "external integration {channel} is enabled (must be disabled in an enclave)",
    },
    "enclave.guard.dir_owner": {
        "ja": "ディレクトリ {path} の所有者が現在のユーザーではありません",
        "en": "directory {path} is not owned by the current user",
    },
    "enclave.guard.dir_mode": {
        "ja": "ディレクトリ {path} に group/other のパーミッションビットがあります",
        "en": "directory {path} has group/other permission bits set",
    },
    "enclave.guard.llm_credentials_required": {
        "ja": "allowed_llm_credentials が空です（少なくとも1つ設定する必要があります）",
        "en": "allowed_llm_credentials is empty (at least one must be set)",
    },
    "enclave.guard.llm_credential_not_allowed": {
        "ja": "LLM クレデンシャル '{credential}' が許可リストに含まれていません",
        "en": "LLM credential '{credential}' is not in the allow-list",
    },
    "enclave.guard.status_unreadable": {
        "ja": "anima {anima} の status.json を読み込めません",
        "en": "could not read status.json for anima {anima}",
    },
    "enclave.guard.file_root_unreadable": {
        "ja": "anima {anima} の permissions を読み込めません",
        "en": "could not read permissions for anima {anima}",
    },
    "enclave.guard.file_root_full_access": {
        "ja": 'anima {anima} の file_roots に "/" があります（danger-full-access）',
        "en": 'anima {anima} has "/" in file_roots (danger-full-access)',
    },
    "enclave.guard.auth_unreadable": {
        "ja": "認証設定 (auth) を読み込めません",
        "en": "could not read authentication config (auth)",
    },
    "enclave.guard.auth_mode": {
        "ja": "auth_mode が '{mode}' です（password または multi_user が必要です）",
        "en": "auth_mode is '{mode}' (password or multi_user required)",
    },
    "enclave.guard.auth_trust_localhost": {
        "ja": "trust_localhost が有効です（enclave では無効にする必要があります）",
        "en": "trust_localhost is enabled (must be disabled in an enclave)",
    },
    "enclave.guard.host": {
        "ja": "bind host '{host}' が loopback アドレスではありません",
        "en": "bind host '{host}' is not a loopback address",
    },
}
