# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Domain-specific i18n strings."""

from __future__ import annotations

STRINGS: dict[str, dict[str, str]] = {
    "cost.unknown_pricing": {
        "ja": "不明（{count}件の料金未登録）",
        "en": "Unknown (pricing unavailable for {count} sessions)",
    },
    "cost.partial_pricing": {
        "ja": "{cost} + 不明分（{count}件の料金未登録）",
        "en": "{cost} + unknown (pricing unavailable for {count} sessions)",
    },
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
    "builder.external_tools": {
        "ja": "外部ツール（`animaworks-tool <name>` 経由）: {categories}",
        "en": "External tools (via `animaworks-tool <name>`): {categories}",
        "ko": "외부 도구 (`animaworks-tool <name>` 경유): {categories}",
    },
    "builder.external_tools.personal_count": {
        "ja": "個人ツール: {count}件（一覧は `animaworks-tool --help`）",
        "en": "Personal tools: {count} (list with `animaworks-tool --help`)",
        "ko": "개인 도구: {count}개 (목록은 `animaworks-tool --help`)",
    },
    "builder.external_tools.direct": {
        "ja": "ツール一覧に専用の外部ツールがあれば、そのツール名で直接呼び出す。",
        "en": "If a dedicated external tool is visible in your tool list, call it directly by name.",
        "ko": "도구 목록에 전용 외부 도구가 있으면 해당 도구 이름으로 직접 호출하세요.",
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
            "⚠️ あなたの AGENTS.md が {size} 文字に肥大化しています（目標上限: {threshold} 文字）。\n"
            "AGENTS.md は「行動憲法 + 参照索引」として保つこと:\n"
            "1. 常駐させる: 役割定義、絶対遵守ルール、安全・承認・機密・二重実行防止\n"
            "2. 外へ逃がす: 手順の詳細は procedures/、学習知識や事例は knowledge/ に移す\n"
            '3. AGENTS.md には read_memory_file(path="...") のポインタを残す\n'
            '4. consolidation では read_memory_file(path="AGENTS.md") で確認し、'
            "目標上限を超える場合は自分で外部参照化して上書き整理する"
        ),
        "en": (
            "⚠️ Your AGENTS.md has grown to {size} characters "
            "(target limit: {threshold}).\n"
            "Keep AGENTS.md as a constitution plus pointer index:\n"
            "1. Keep resident: role definition, non-negotiable rules, safety, "
            "approval, confidentiality, and duplicate-action prevention\n"
            "2. Move out: procedural details to procedures/, learned knowledge "
            "and examples to knowledge/\n"
            '3. Leave read_memory_file(path="...") pointers in AGENTS.md\n'
            '4. During consolidation, read_memory_file(path="AGENTS.md") and, '
            "when it exceeds the target limit, rewrite it yourself with external "
            "references"
        ),
    },
    "builder.skill_catalog_header": {
        "ja": "## Available Skills",
        "en": "## Available Skills",
    },
    "builder.superseded_note": {
        "ja": "（旧版。{latest} に統合）",
        "en": "(superseded; folded into {latest})",
        "ko": "(이전 버전. {latest} 에 통합)",
    },
    "builder.skill_catalog_instruction": {
        "ja": "使う前に表示パスを `read_memory_file` で読み、手順・制約を確認する。\n新規作成時は `common_skills/skill-creator/SKILL.md` を読む。`external/...` はホスト側スキル。",
        "en": "Before use, read the shown path with `read_memory_file` and verify its instructions and constraints.\nBefore creating a skill, read `common_skills/skill-creator/SKILL.md`; `external/...` are host-side skills.",
        "ko": "사용 전에 표시된 경로를 `read_memory_file`로 읽고 절차와 제약을 확인하세요.\n새 스킬 작성 전 `common_skills/skill-creator/SKILL.md`를 읽으세요. `external/...`은 호스트측 스킬입니다.",
    },
    "builder.board_channels_none": {
        "ja": "（なし）",
        "en": "(none)",
        "ko": "(없음)",
    },
    "builder.board_channels_more": {
        "ja": "、ほか{count}件",
        "en": ", and {count} more",
        "ko": " 외 {count}개",
    },
    "builder.board_guidance_with_team": {
        "ja": "- 所属チームの限定チャネル（報告先）: {team_channels}（通常の作業報告・完了報告を優先）\n- 現在アクセスできるBoardチャネル: {visible_channels}\n- `general` は全体共有のみ、`ops` は部門横断の運用・インフラ共有のみ。",
        "en": "- Restricted team channel(s) (reporting destination): {team_channels} (prioritize for routine work/completion reports)\n- Accessible Board channels: {visible_channels}\n- Use `general` only for org-wide sharing and `ops` only for cross-team operations/infrastructure.",
        "ko": "- 소속 팀 제한 채널(보고처): {team_channels} (일반 업무/완료 보고 우선)\n- 현재 접근 가능한 Board 채널: {visible_channels}\n- `general`은 전체 공유용으로만, `ops`는 팀 간 운영/인프라 공유용으로만 사용합니다.",
    },
    "builder.board_guidance_without_team": {
        "ja": "- 所属チームの限定チャネル（報告先）: なし\n- 現在アクセスできるBoardチャネル: {visible_channels}\n- 最も関連性の高い共有チャネルを使う（`general` は全体共有のみ、`ops` は部門横断の運用・インフラ共有のみ）。",
        "en": "- Restricted team channel (reporting destination): none\n- Accessible Board channels: {visible_channels}\n- Use the most relevant shared channel (`general` only for org-wide sharing; `ops` only for cross-team operations/infrastructure).",
        "ko": "- 소속 팀 제한 채널(보고처): 없음\n- 현재 접근 가능한 Board 채널: {visible_channels}\n- 가장 관련성 높은 공유 채널을 사용합니다 (`general`은 전체 공유용으로만, `ops`는 팀 간 운영/인프라 공유용으로만 사용).",
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
    "cli.task_autonomy_help": {
        "ja": "委譲まわりの自律性指標を集計する（読み取り専用）",
        "en": "Summarize delegation autonomy metrics (read-only)",
    },
    "cli.task_autonomy_days_help": {
        "ja": "集計期間（日数、既定: 7）",
        "en": "Report period in days (default: 7)",
    },
    "cli.task_autonomy_json_help": {
        "ja": "JSON で出力する",
        "en": "Emit JSON",
    },
    "cli.task_autonomy_anima_help": {
        "ja": "この Anima のみを表示する",
        "en": "Show only this anima",
    },
    "cli.task_autonomy_days_invalid": {
        "ja": "Error: --days には 1 以上を指定してください",
        "en": "Error: --days must be 1 or greater",
    },
    "cli.task_autonomy_title": {
        "ja": "## 自律性レポート（過去 {days} 日間）",
        "en": "## Autonomy report (last {days} days)",
    },
    "cli.task_autonomy_period": {
        "ja": "期間: {since} 〜 {until}",
        "en": "Period: {since} to {until}",
    },
    "cli.task_autonomy_no_animas": {
        "ja": "対象の Anima がありません。",
        "en": "No animas in scope.",
    },
    "cli.task_autonomy_table_header": {
        "ja": (
            "| Anima | 自発 | 委譲 | 人間 | 不明 | 自発比率 | "
            "起動 (HB/cron/inbox/chat/task_exec) | 委譲キャンセル | scope_notes | waiting_for | "
            "費用比率 | 起動比率 | DM中央値 | DM最大 |"
        ),
        "en": (
            "| Anima | Self | Delegated | Human | Unknown | Self ratio | "
            "Activations (HB/cron/inbox/chat/task_exec) | Delegated cancelled | scope_notes | waiting_for | "
            "Cost share | Launch share | DM median | DM max |"
        ),
    },
    "cli.task_autonomy_total": {
        "ja": "合計",
        "en": "Total",
    },
    "cli.task_autonomy_dm_title": {
        "ja": "## DM往復（task_id 別 上位5件）",
        "en": "## DM round trips (top 5 by task_id)",
    },
    "cli.task_autonomy_dm_header": {
        "ja": "| Task ID | Anima | DMイベント数 |",
        "en": "| Task ID | Anima | DM events |",
    },
    "cli.task_autonomy_dm_empty": {
        "ja": "task_id 付きの DM イベントはありません。",
        "en": "No DM events with a task_id.",
    },
    "migrate.migration_note": {
        "ja": "<!-- MIGRATION NOTE: could not auto-convert '{schedule}' to cron expression -->",
        "en": "<!-- MIGRATION NOTE: could not auto-convert '{schedule}' to cron expression -->",
    },
    "reminder.context_threshold": {
        "ja": "コンテキスト使用量: {ratio}。出力を簡潔にし、重要な状態をセッション状態に保存せよ。",
        "en": ("Context usage: {ratio}. Keep output concise and save important state to session state."),
    },
    "reminder.empty_response": {
        "ja": "直前の応答が空でした。ツールを呼び出すか、これまでの作業内容を踏まえたテキストの回答を出力してください。",
        "en": (
            "The previous response was empty. Either call a tool or produce a text answer based on the work done so far."
        ),
    },
    "reminder.final_iteration": {
        "ja": "ツールの使用回数が上限に達しました。これ以上ツールは使用できません。これまでの作業内容と得られた情報を踏まえて、最終回答を作成してください。",
        "en": (
            "Tool usage limit reached. No more tools can be used. Based on the work done and information gathered so far, compose your final answer."
        ),
    },
    "reminder.tool_loop_halt": {
        "ja": "同一のツール呼び出しの繰り返しが上限（{count}回）に達したため、ツールを無効化しました。これまでに得られた結果を踏まえて最終回答を作成してください。",
        "en": (
            "The identical tool call was repeated {count} times, so tools have been disabled. Compose your final answer based on the results gathered so far."
        ),
    },
    "reminder.tool_loop_warning": {
        "ja": "同一のツール呼び出し（{tool_names}）を{count}回連続で繰り返しています。結果が変わらない場合は方針を変えるか、最終回答をまとめてください。",
        "en": (
            "You have repeated the identical tool call ({tool_names}) {count} times in a row. If the result is not changing, change approach or compose your final answer."
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
    "skill.label_common": {
        "ja": "共通",
        "en": "common",
    },
    "skill.label_procedure": {
        "ja": "手順",
        "en": "procedure",
    },
    "skill_creator.created": {
        "ja": ("スキル '{skill_name}' を作成しました: {skill_dir}\n作成ファイル: {files_str}"),
        "en": ("Skill '{skill_name}' created: {skill_dir}\nCreated files: {files_str}"),
    },
    "skill_creator.invalid_name": {
        "ja": "無効なスキル名: '{skill_name}'（パス区切り文字は使用不可）",
        "en": "Invalid skill name: '{skill_name}' (path separators are not allowed)",
    },
    "skill_auto.created": {
        "ja": "スキル化しました: {skill_name} ({path})",
        "en": "スキル化しました: {skill_name} ({path})",
    },
    "task_queue.auto_taskexec": {
        "ja": "(auto: TaskExec)",
        "en": "(auto: TaskExec)",
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
    "task_queue.delegated_unknown": {
        "ja": "不明",
        "en": "unknown",
    },
    "task_queue.delegated_archived": {
        "ja": "アーカイブ済",
        "en": "archived",
    },
    "voice.stt_failed": {
        "ja": "音声認識に失敗しました",
        "en": "Speech recognition failed",
    },
    "voice.ask_anima_tool_description": {
        "ja": (
            "時間のかかる作業・ツール実行・記憶の検索保存・タスク化・調査・実装などを、"
            "自分自身の本体（フルエージェント）に依頼する。会話で即答できないことはこれで依頼する。"
            "挨拶・お礼・雑談・意見や感想・自分で即答できる質問には使わない。"
            "実行中の依頼がある間は新しい依頼をせず、完了を待つ。"
        ),
        "en": (
            "Delegate time-consuming work, tool execution, memory search or storage, task creation, research, "
            "or implementation to your full agent. Use this when you cannot answer immediately from conversation. "
            "Do not use it for greetings, thanks, small talk, opinions, or questions you can answer yourself. "
            "Do not submit a new request while another request is running; wait for it to finish."
        ),
        "ko": (
            "시간이 걸리는 작업, 도구 실행, 기억 검색이나 저장, 작업 생성, 조사, 구현 등을 전체 에이전트인 자신에게 위임한다. "
            "대화만으로 바로 답할 수 없는 경우에 사용한다. 인사, 감사, 잡담, 의견, 스스로 바로 답할 수 있는 질문에는 사용하지 않는다. "
            "실행 중인 요청이 있으면 새 요청을 보내지 말고 완료될 때까지 기다린다."
        ),
    },
    "voice.ask_anima_in_progress": {
        "ja": (
            "job {job}（依頼: {request}）が実行中です。同じ内容なら完了を待ってください。"
            "別の依頼なら、完了後に改めて依頼してください。"
        ),
        "en": (
            "Job {job} (request: {request}) is already running. If this is the same request, wait for it to finish. "
            "If it is a different request, submit it after the current job completes."
        ),
        "ko": (
            "job {job}(요청: {request})이(가) 실행 중입니다. 같은 요청이면 완료될 때까지 기다려 주세요. "
            "다른 요청이면 현재 작업이 끝난 후 다시 요청해 주세요."
        ),
    },
    "voice.delegation_report_subject": {
        "ja": "音声で頼まれた件の結果（{anima}）",
        "en": "Results of the voice request ({anima})",
    },
    "voice.delegation_still_running": {
        "ja": "[ask_anima job {job}: 30分以内に完了を確認できませんでした。依頼: {request}]",
        "en": "[ask_anima job {job}: still running after 30 minutes. Request: {request}]",
    },
    "cli.export_help": {
        "ja": "Anima のワークスペースを OpenClaw / Hermes Agent 形式で書き出す",
        "en": "Export an Anima workspace in OpenClaw or Hermes Agent format",
        "ko": "Anima 워크스페이스를 OpenClaw / Hermes Agent 형식으로 내보내기",
    },
    "cli.export_complete": {
        "ja": "{anima} を {format} 形式で書き出しました: {out} ({count} ファイル)",
        "en": "Exported {anima} in {format} format: {out} ({count} files)",
        "ko": "{anima}를 {format} 형식으로 내보냈습니다: {out} ({count}개 파일)",
    },
    "cli.export_failed": {
        "ja": "エクスポートに失敗しました: {error}",
        "en": "Export failed: {error}",
        "ko": "내보내기 실패: {error}",
    },
}
