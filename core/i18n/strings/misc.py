# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Domain-specific i18n strings."""

from __future__ import annotations

STRINGS: dict[str, dict[str, str]] = {
    "activity.blocked": {
        "ja": "ブロック: {reason}",
        "en": "Blocked: {reason}",
    },
    "activity.cron_task_exec": {
        "ja": "cronタスク実行",
        "en": "Cron task executed",
    },
    "activity.error_prefix": {
        "ja": "[エラー] ",
        "en": "[Error] ",
    },
    "activity.heartbeat_end": {
        "ja": "定期巡回完了",
        "en": "Periodic check completed",
    },
    "activity.heartbeat_reflection": {
        "ja": "HB振り返り",
        "en": "HB reflection",
    },
    "activity.heartbeat_start": {
        "ja": "定期巡回開始",
        "en": "Periodic check started",
    },
    "activity.items_count": {
        "ja": "{count}件",
        "en": "{count} items",
    },
    "activity.task_exec_end_label": {
        "ja": "タスク実行完了",
        "en": "Task execution completed",
    },
    "activity.task_exec_start_label": {
        "ja": "タスク実行開始",
        "en": "Task execution started",
    },
    "activity_report.future_date": {
        "ja": "未来の日付は指定できません",
        "en": "Future dates are not allowed",
    },
    "activity_report.invalid_date": {
        "ja": "日付の形式が不正です（YYYY-MM-DD）",
        "en": "Invalid date format (YYYY-MM-DD)",
    },
    "activity_report.invalid_model": {
        "ja": "指定されたモデルは利用できません",
        "en": "The specified model is not available",
    },
    "activity_report.llm_system_prompt": {
        "ja": (
            "あなたは組織活動レポーターです。\n以下の「組織タイムライン」を読み、1日の活動をストーリー仕立てのMarkdownレポートにまとめてください。\n\nタイムラインのフォーマット:\n- [HH:MM] 名前 アイコン イベント種別 の形式で時系列に並んでいます\n- 末尾にツール使用サマリーと統計があります\n\n要件:\n- 見出し: 日付 + 組織活動レポート\n- ハイライト: 最も重要な成果を3-5個\n- ストーリー形式: 時系列に沿って「誰が」「何をして」「どうなったか」を自然な文章で記述\n- 関連する出来事（指示→実行→完了など）を因果関係でつなげて読みやすくする\n- エラーや問題があれば「課題・注意事項」セクションに記載\n- 末尾の統計データを引用して全体像を補足\n- 日本語で出力"
        ),
        "en": (
            'You are an organisational activity reporter.\nRead the \'Org Timeline\' below and produce a narrative-style Markdown report of the day\'s activities.\n\nTimeline format:\n- Each line is [HH:MM] name icon event_type, sorted chronologically\n- Tool usage summary and statistics are at the bottom\n\nRequirements:\n- Heading: Date + Organisation Activity Report\n- Highlights: 3-5 most important outcomes\n- Narrative style: describe chronologically "who" "did what" "with what result" in natural prose\n- Connect related events (instruction → execution → completion) with causal links for readability\n- Include an "Issues & Notes" section for errors or problems\n- Cite the statistics at the bottom to provide the big picture\n- Output in English'
        ),
    },
    "activity_report.llm_user_prompt": {
        "ja": ("以下の組織タイムラインに基づいて活動レポートを生成してください。\n\n{data}"),
        "en": ("Generate an activity report based on the following org timeline.\n\n{data}"),
    },
    "activity_report.not_found": {
        "ja": "この日付のレポートはキャッシュされていません",
        "en": "No cached report found for this date",
    },
    "asset_reconciler.llm_user_prompt": {
        "ja": (
            "以下のキャラクターシートから外見情報を読み取り、NovelAI 互換の画像生成タグに変換してください:\n\n{character_text}"
        ),
        "en": (
            "Read the following character sheet and extract visual appearance into NovelAI-compatible image generation tags:\n\n{character_text}"
        ),
    },
    "asset_reconciler.llm_user_prompt_realistic": {
        "ja": (
            "以下のキャラクターシートから外見情報を読み取り、写実的な写真風の画像生成プロンプトに変換してください:\n\n{character_text}"
        ),
        "en": (
            "Read the following character sheet and extract visual appearance into a photorealistic image generation prompt:\n\n{character_text}"
        ),
    },
    "audit.org_timeline_footer": {
        "ja": (
            "─── 統計: 全{count}名 | 活動{total}件 | ツール{tools} | HB{hb} | 応答{resp} | DM{dm} | エラー{err} ───"
        ),
        "en": (
            "─── Stats: {count} animas | {total} events | Tools {tools} | HB {hb} | Responses {resp} | DM {dm} | Errors {err} ───"
        ),
    },
    "audit.org_timeline_no_activity": {
        "ja": "(この期間の活動ログはありません)",
        "en": "(No activity log for this period)",
    },
    "audit.org_timeline_thinned_notice": {
        "ja": "(HB/Cron {hb_original}件中{hb_kept}件を表示 — 等間隔サンプリング | コマンドCron {cmd_cron}件省略)",
        "en": ("(Showing {hb_kept} of {hb_original} HB/Cron — evenly sampled | {cmd_cron} command crons omitted)"),
    },
    "audit.org_timeline_title": {
        "ja": "═══ 組織タイムライン ({date}) — {count}名 ═══",
        "en": "═══ Org Timeline ({date}) — {count} animas ═══",
    },
    "audit.org_timeline_tool_header": {
        "ja": "■ ツール使用サマリー",
        "en": "■ Tool Usage Summary",
    },
    "audit.org_timeline_tool_line": {
        "ja": "{name} (全{total}回): ",
        "en": "{name} ({total} total): ",
    },
    "audit.timeline_label_cron_executed": {
        "ja": "Cron",
        "en": "Cron",
    },
    "audit.timeline_label_error": {
        "ja": "エラー",
        "en": "Error",
    },
    "audit.timeline_label_heartbeat_end": {
        "ja": "HB",
        "en": "HB",
    },
    "audit.timeline_label_heartbeat_reflection": {
        "ja": "振り返り",
        "en": "Reflection",
    },
    "audit.timeline_label_issue_resolved": {
        "ja": "解決",
        "en": "Resolved",
    },
    "audit.timeline_label_message_sent": {
        "ja": "DM",
        "en": "DM",
    },
    "audit.timeline_label_response_sent": {
        "ja": "応答",
        "en": "Response",
    },
    "audit.timeline_label_task_exec_end": {
        "ja": "タスク完了",
        "en": "Task done",
    },
    "builder.c_response_requirement": {
        "ja": (
            "## 応答要件\nあなたはユーザーとの対話において、**必ずテキストで応答**してください。\nツール呼び出しを行った場合でも、その結果の要約やユーザーへの返答を\nテキストメッセージとして出力してください。\n挨拶・質問・雑談などの会話メッセージには、ツール呼び出しの前後に\n自然なテキスト応答を必ず含めてください。"
        ),
        "en": (
            "## Response Requirements\nYou **must always respond with text** when interacting with users.\nEven when making tool calls, output a summary of results or a reply\nto the user as a text message.\nFor greetings, questions, or casual conversation, always include\nnatural text responses before or after any tool calls."
        ),
    },
    "builder.default_workspace": {
        "ja": "あなたのデフォルトワークスペース: {path} ({alias})",
        "en": "Your default workspace: {path} ({alias})",
    },
    "builder.default_workspace_unresolved": {
        "ja": "あなたのデフォルトワークスペース: (未解決: {alias})",
        "en": "Your default workspace: (unresolved: {alias})",
    },
    "builder.heartbeat_tool_fallback": {
        "ja": (
            "Heartbeatでは**観察・報告・計画・フォローアップ**にツールを使ってください。\n- OK: read_channel, search_memory, read_memory_file, send_message, post_channel, update_task, delegate_task, submit_tasks, Write（pending作成用）\n- NG: コード変更、ファイル大量編集、長時間の分析・調査\n重い作業が必要な場合は state/pending/ にタスクファイルを書き出してください。"
        ),
        "en": (
            "In Heartbeat, use tools for **observation, reporting, planning, and follow-up**.\n- OK: read_channel, search_memory, read_memory_file, send_message, post_channel, update_task, delegate_task, submit_tasks, Write (for pending creation)\n- NG: code changes, bulk file edits, lengthy analysis/investigation\nIf heavy work is needed, write a task file to state/pending/."
        ),
    },
    "builder.injection_size_warning": {
        "ja": (
            '⚠️ あなたの injection.md が {size} 文字に肥大化しています（推奨上限: {threshold} 文字）。\n都度指示や学習した知識を knowledge/ に移してください:\n1. read_memory_file(path="injection.md") で内容を確認\n2. 「役割定義」「絶対遵守ルール」以外の記述を knowledge/ に移動\n   - 重要なルールは [IMPORTANT] タグを付けて knowledge/ に書く（常時想起されます）\n   - 手順的な内容は procedures/ に移動\n3. 移動完了後、injection.md を上書きして整理する'
        ),
        "en": (
            '⚠️ Your injection.md has grown to {size} characters (recommended limit: {threshold}).\nMove ad-hoc directives and learned knowledge to knowledge/:\n1. read_memory_file(path="injection.md") to review contents\n2. Move non-core content (not role definition or absolute rules) to knowledge/\n   - Tag important rules with [IMPORTANT] in knowledge/ (they will be always-primed)\n   - Move procedural content to procedures/\n3. Overwrite injection.md with the cleaned-up version'
        ),
    },
    "builder.machine_hint": {
        "ja": (
            '\n\n**machine ツール**: コード変更・調査・分析など重い作業は `animaworks-tool machine run` で外部エージェントに委託できます。詳細は read_memory_file(path="common_skills/machine-tool/SKILL.md") で確認。'
        ),
        "en": (
            '\n\n**machine tool**: For heavy tasks like code changes, investigation, or analysis, delegate to an external agent via `animaworks-tool machine run`. Use read_memory_file(path="common_skills/machine-tool/SKILL.md") for details.'
        ),
    },
    "builder.skill_catalog_header": {
        "ja": "## Available Skills",
        "en": "## Available Skills",
    },
    "builder.skill_catalog_instruction": {
        "ja": "該当するスキルがあれば、表示されているパスで `read_memory_file` を使って全文を読むこと。",
        "en": "When a matching skill exists, use `read_memory_file` with the path shown to load full instructions.",
    },
    "builder.procedure_label": {
        "ja": "手順",
        "en": "procedure",
    },
    "cli.disable_help": {
        "ja": "Disable (休養) an anima",
        "en": "Disable an anima",
    },
    "cli.enable_help": {
        "ja": "Enable (復帰) an anima",
        "en": "Enable an anima",
    },
    "cli.permissions_help": {
        "ja": "Animaの権限設定を表示",
        "en": "Show anima permissions configuration",
    },
    "cli.permissions_not_found": {
        "ja": "Error: Anima '{name}' が見つかりません",
        "en": "Error: Anima '{name}' not found",
    },
    "cli.permissions_file_path": {
        "ja": "ファイル: {path}",
        "en": "File: {path}",
    },
    "cli.migrate_cron_done": {
        "ja": "Migrated {count} anima(s) to standard cron format.",
        "en": "Migrated {count} anima(s) to standard cron format.",
    },
    "cli.migrate_cron_skipped": {
        "ja": "No migration needed — all cron.md files are already in standard format.",
        "en": "No migration needed — all cron.md files are already in standard format.",
    },
    "cli.profile_add_hint": {
        "ja": "'animaworks profile add <name>' で作成してください",
        "en": "Use 'animaworks profile add <name>' to create one",
    },
    "cli.profile_already_exists": {
        "ja": "Error: プロファイル '{name}' は既に存在します",
        "en": "Error: Profile '{name}' already exists",
    },
    "cli.profile_already_running": {
        "ja": "プロファイル '{name}' は既に起動中です (pid={pid})",
        "en": "Profile '{name}' is already running (pid={pid})",
    },
    "cli.profile_corrupt_file": {
        "ja": "警告: プロファイルファイルが破損しているか読み取れません。空のレジストリを使用します。",
        "en": "Warning: Profiles file is corrupt or unreadable. Using empty registry.",
    },
    "cli.profile_data_preserved": {
        "ja": "データは保持されています: {path}",
        "en": "Data preserved at: {path}",
    },
    "cli.profile_help": {
        "ja": "複数のAnimaWorksインスタンスを管理（マルチテナント）",
        "en": "Manage multiple AnimaWorks instances (multi-tenant)",
    },
    "cli.profile_init_hint": {
        "ja": "データディレクトリが未初期化です。以下で初期化してください:",
        "en": "Data directory not initialized. Initialize with:",
    },
    "cli.profile_no_profiles": {
        "ja": "プロファイルが登録されていません",
        "en": "No profiles registered",
    },
    "cli.profile_not_found": {
        "ja": "Error: プロファイル '{name}' が見つかりません",
        "en": "Error: Profile '{name}' not found",
    },
    "cli.profile_not_running": {
        "ja": "プロファイル '{name}' は起動していません",
        "en": "Profile '{name}' is not running",
    },
    "cli.profile_registered": {
        "ja": "プロファイル '{name}' を登録しました",
        "en": "Profile '{name}' registered",
    },
    "cli.profile_removed": {
        "ja": "プロファイル '{name}' を削除しました（登録のみ）",
        "en": "Profile '{name}' removed (registration only)",
    },
    "cli.profile_running": {
        "ja": "起動中 (pid={pid})",
        "en": "running (pid={pid})",
    },
    "cli.profile_started": {
        "ja": "プロファイル '{name}' を起動しました",
        "en": "Profile '{name}' started",
    },
    "cli.profile_starting": {
        "ja": "{name} を起動中...",
        "en": "Starting {name}...",
    },
    "cli.profile_stop_running": {
        "ja": "プロファイル '{name}' は起動中です。先に停止してください",
        "en": "Profile '{name}' is running. Stop it first",
    },
    "cli.profile_stopped": {
        "ja": "停止",
        "en": "stopped",
    },
    "cli.profile_stopped_ok": {
        "ja": "プロファイル '{name}' を停止しました",
        "en": "Profile '{name}' stopped",
    },
    "cli.profile_stopped_stale": {
        "ja": "停止 (古いPID)",
        "en": "stopped (stale pid)",
    },
    "cli.profile_stopping": {
        "ja": "'{name}' を停止中 (pid={pid})...",
        "en": "Stopping '{name}' (pid={pid})...",
    },
    "cli.set_outbound_limit_cleared": {
        "ja": "{name} のアウトバウンド制限をクリアしました（ロールデフォルトにフォールバック）",
        "en": "Cleared outbound limits for {name} (falling back to role defaults)",
    },
    "cli.set_outbound_limit_success": {
        "ja": "{name} のアウトバウンド制限を更新しました: {details}",
        "en": "Updated outbound limits for {name}: {details}",
    },
    "migrate.migration_note": {
        "ja": "<!-- MIGRATION NOTE: could not auto-convert '{schedule}' to cron expression -->",
        "en": "<!-- MIGRATION NOTE: could not auto-convert '{schedule}' to cron expression -->",
    },
    "reminder.context_threshold": {
        "ja": "コンテキスト使用量: {ratio}。出力を簡潔にし、重要な状態をセッション状態に保存せよ。",
        "en": ("Context usage: {ratio}. Keep output concise and save important state to session state."),
    },
    "reminder.final_iteration": {
        "ja": "ツールの使用回数が上限に達しました。これ以上ツールは使用できません。これまでの作業内容と得られた情報を踏まえて、最終回答を作成してください。",
        "en": (
            "Tool usage limit reached. No more tools can be used. Based on the work done and information gathered so far, compose your final answer."
        ),
    },
    "reminder.hb_hard_timeout_recovery": {
        "ja": (
            "前回のHeartbeatが制限時間（{timeout}秒）を超過したため強制終了されました。中断時点の作業内容を確認し、必要であれば backlog_task でタスク登録してください。"
        ),
        "en": (
            "Previous heartbeat was terminated due to exceeding the time limit ({timeout}s). Review the work in progress and use backlog_task to register tasks if needed."
        ),
    },
    "reminder.hb_time_limit": {
        "ja": (
            "⏰ Heartbeatの制限時間が近づいています。今すぐ以下を実行して終了してください:\n1. 未完了の作業があれば backlog_task ツールでタスクキューに登録する\n2. 観察結果・計画を current_state.md に update_task または write_memory_file で記録する\n3. [REFLECTION] ブロックを出力してHeartbeatを終了する"
        ),
        "en": (
            "⏰ Heartbeat time limit approaching. Execute the following immediately and finish:\n1. Use backlog_task tool to register any remaining work in the task queue\n2. Record observations/plans in current_state.md via update_task or write_memory_file\n3. Output a [REFLECTION] block and end the heartbeat"
        ),
    },
    "reminder.output_truncated": {
        "ja": "出力がmax_tokensで途切れた。残りの内容を小さく分割して続行せよ。",
        "en": ("Output was cut off at max_tokens. Split the remaining content into smaller parts and continue."),
    },
    "runner.recovery_text": {
        "ja": "応答が中断されました（前回セッションの未完了ストリームを回復, {session_type}）",
        "en": ("Response was interrupted (recovered incomplete stream from previous session, {session_type})"),
    },
    "session.caution_continue": {
        "ja": "中断前の作業の続きを実行してください",
        "en": "Continue the work from before the interruption",
    },
    "session.caution_header": {
        "ja": "## 注意",
        "en": "## Caution",
    },
    "session.caution_no_repeat": {
        "ja": "完了済みステップを繰り返さないでください",
        "en": "Do not repeat completed steps",
    },
    "session.caution_skip_existing": {
        "ja": "ファイルが既に存在する場合はスキップまたは更新してください",
        "en": "If files already exist, skip or update them",
    },
    "session.completed_none": {
        "ja": "(なし)",
        "en": "(none)",
    },
    "session.completed_steps_header": {
        "ja": "## 完了済みステップ",
        "en": "## Completed Steps",
    },
    "session.continuation_intro": {
        "ja": ("あなたは以下のタスクを実行中でしたが、通信エラーで中断されました。\n続きから実行してください。"),
        "en": (
            "You were executing the following task but it was interrupted by a communication error.\nPlease continue from where you left off."
        ),
    },
    "session.original_instruction_header": {
        "ja": "## 元の指示",
        "en": "## Original Instruction",
    },
    "session.output_so_far_header": {
        "ja": "## これまでの出力",
        "en": "## Output So Far",
    },
    "session.text_truncated": {
        "ja": "...(前半省略)...",
        "en": "...(earlier omitted)...",
    },
    "settings.activity_level.desc": {
        "ja": "Heartbeatの実行頻度と思考深度を調整します（10%〜400%）。100%が通常、低いほど省エネ、高いほど高頻度。",
        "en": (
            "Adjust heartbeat frequency and thinking depth (10%-400%). 100% is normal; lower saves cost, higher increases frequency."
        ),
    },
    "settings.activity_level.title": {
        "ja": "アクティビティレベル",
        "en": "Activity Level",
    },
    "settings.activity_level.updated": {
        "ja": "アクティビティレベルを {level}% に変更しました",
        "en": "Activity level changed to {level}%",
    },
    "skill.desc_line3": {
        "ja": "該当するスキルがある場合に使用すること。",
        "en": "Use when a matching skill is available.",
    },
    "skill.label_common": {
        "ja": "共通",
        "en": "common",
    },
    "skill.label_procedure": {
        "ja": "手順",
        "en": "procedure",
    },
    "skill.truncated": {
        "ja": "(以降省略)",
        "en": "(truncated)",
    },
    "skill_creator.created": {
        "ja": ("スキル '{skill_name}' を作成しました: {skill_dir}\n作成ファイル: {files_str}"),
        "en": ("Skill '{skill_name}' created: {skill_dir}\nCreated files: {files_str}"),
    },
    "skill_creator.invalid_name": {
        "ja": "無効なスキル名: '{skill_name}'（パス区切り文字は使用不可）",
        "en": "Invalid skill name: '{skill_name}' (path separators are not allowed)",
    },
    "task_queue.auto_taskexec": {
        "ja": "(auto: TaskExec)",
        "en": "(auto: TaskExec)",
    },
    "task_queue.deadline_by": {
        "ja": "📅 {time}まで",
        "en": "📅 By {time}",
    },
    "task_queue.elapsed_hours": {
        "ja": "⏱️ {hours}時間経過",
        "en": "⏱️ {hours}h elapsed",
    },
    "task_queue.elapsed_hours_min": {
        "ja": "⏱️ {hours}時間{remaining_min}分経過",
        "en": "⏱️ {hours}h {remaining_min}m elapsed",
    },
    "task_queue.elapsed_minutes": {
        "ja": "⏱️ {minutes}分経過",
        "en": "⏱️ {minutes}m elapsed",
    },
    "task_queue.failed_line": {
        "ja": "- [{task_id}] {summary}",
        "en": "- [{task_id}] {summary}",
    },
    "task_queue.failed_section_header": {
        "ja": ("\n❌ Failed (要対処):"),
        "en": ("\n❌ Failed (action required):"),
    },
    "task_queue.overdue": {
        "ja": "🔴 OVERDUE({time}期限)",
        "en": "🔴 OVERDUE(deadline {time})",
    },
    "task_queue.overdue_aggregate": {
        "ja": '🔴 OVERDUE集約（{count}件）: {summaries}\n  → list_tasks(status="pending") で詳細確認',
        "en": '🔴 OVERDUE aggregate ({count}): {summaries}\n  → list_tasks(status="pending") for details',
    },
    "task_queue.sync_done": {
        "ja": "{orig} (→{target}: 完了)",
        "en": "{orig} (→{target}: done)",
    },
    "task_queue.sync_failed": {
        "ja": "{orig} (→{target}: 失敗 — 再委任を検討)",
        "en": "{orig} (→{target}: failed — consider re-delegation)",
    },
    "task_queue.delegated_unknown": {
        "ja": "不明",
        "en": "unknown",
    },
    "task_queue.delegated_archived": {
        "ja": "アーカイブ済",
        "en": "archived",
    },
    "voice.mode_suffix": {
        "ja": (
            "\n\n[voice-mode: 音声会話です。話し言葉で200文字以内で簡潔に回答してください。Markdown記法（見出し・太字・リスト・コードブロック等）は使わないでください]"
        ),
        "en": (
            "\n\n[voice-mode: This is a voice conversation. Reply concisely in spoken language, 200 characters or fewer. Do not use Markdown formatting (headings, bold, lists, code blocks, etc.)]"
        ),
    },
    "voice.stt_failed": {
        "ja": "音声認識に失敗しました",
        "en": "Speech recognition failed",
    },
}
