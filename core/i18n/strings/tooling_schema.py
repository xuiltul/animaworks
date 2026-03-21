# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Domain-specific i18n strings (schema.*)."""

from __future__ import annotations

STRINGS: dict[str, dict[str, str]] = {
    "schema.audit_subordinate.desc": {
        "ja": (
            "配下のAnimaの行動を監査する。ActivityLogから「何を考えて何をやったか」を抽出し、統計サマリーまたは日報形式で返す。\nname省略で全配下を一括監査。name指定で特定の配下（孫含む）を監査。\nmode='summary'で統計、mode='report'で時系列の日報形式。"
        ),
        "en": (
            "Audit subordinate Anima behavior. Extracts thoughts and actions from ActivityLog and returns statistics summary or chronological report.\nOmit name to audit all descendants. Specify name for any descendant.\nmode='summary' for stats, mode='report' for chronological daily report."
        ),
    },
    "schema.audit_subordinate.direct_only": {
        "ja": "trueの場合、直属部下のみ対象（孫以下を除外）。デフォルト: false",
        "en": "If true, only audit direct subordinates (exclude grandchildren). Default: false",
    },
    "schema.audit_subordinate.hours": {
        "ja": "監査期間（時間単位、デフォルト: 24、最大: 168）",
        "en": "Audit period in hours (default: 24, max: 168)",
    },
    "schema.audit_subordinate.mode": {
        "ja": "出力モード。report=タイムライン日報（デフォルト）、summary=統計サマリー",
        "en": "Output mode. report=timeline daily report (default), summary=statistics",
    },
    "schema.audit_subordinate.name": {
        "ja": "監査対象のAnima名（省略時は全配下）",
        "en": "Target Anima name (omit for all descendants)",
    },
    "schema.audit_subordinate.since": {
        "ja": "開始時刻（HH:MM形式、当日のJST）。指定時はhoursより優先される",
        "en": ("Start time (HH:MM format, today in JST). Takes precedence over hours when specified"),
    },
    "schema.backlog_task.assignee": {
        "ja": "担当者名（自分自身または委任先のAnima名）",
        "en": "Assignee name (yourself or the delegated Anima name)",
    },
    "schema.backlog_task.deadline": {
        "ja": "期限（必須）。相対形式 '30m','2h','1d' またはISO8601。例: '1h' = 1時間後",
        "en": ("Deadline (required). Relative format '30m','2h','1d' or ISO8601. Example: '1h' = 1 hour from now"),
    },
    "schema.backlog_task.desc": {
        "ja": (
            "タスクキューに新しいタスクを追加する。人間からの指示は必ず source='human' で記録すること。Anima間の委任は source='anima' で記録する。"
        ),
        "en": (
            "Add a new task to the task queue. Instructions from humans must be recorded with source='human'. Inter-Anima delegation uses source='anima'."
        ),
    },
    "schema.backlog_task.original_instruction": {
        "ja": "元の指示文（委任時は原文引用を含める）",
        "en": "Original instruction text (include original quote when delegating)",
    },
    "schema.backlog_task.relay_chain": {
        "ja": "委任経路（例: ['taka', 'sakura', 'rin']）",
        "en": "Delegation chain (e.g. ['taka', 'sakura', 'rin'])",
    },
    "schema.backlog_task.source": {
        "ja": "タスクの発生源 (human=人間からの指示, anima=Anima間委任)",
        "en": "Task source (human=instruction from human, anima=inter-Anima delegation)",
    },
    "schema.backlog_task.summary": {
        "ja": "タスクの1行要約",
        "en": "One-line task summary",
    },
    "schema.call_human.body": {
        "ja": "通知の本文（詳細な報告内容）",
        "en": "Notification body (detailed report content)",
    },
    "schema.call_human.desc": {
        "ja": (
            "人間の管理者に連絡します。重要な報告、問題のエスカレーション、判断が必要な事項がある場合に使用してください。チャット画面と外部通知チャネル（Slack等）の両方に届きます。"
        ),
        "en": (
            "Contact the human administrator. Use this for important reports, problem escalation, or matters requiring human judgment. Notifications are delivered to both the chat UI and external channels (Slack, etc.)."
        ),
    },
    "schema.call_human.priority": {
        "ja": "通知の優先度（デフォルト: normal）",
        "en": "Notification priority (default: normal)",
    },
    "schema.call_human.subject": {
        "ja": "通知の件名（簡潔に）",
        "en": "Notification subject (keep it brief)",
    },
    "schema.check_background_task.desc": {
        "ja": (
            "バックグラウンドタスクの状態を確認する。task_idを指定して、実行中・完了・失敗の状態と結果を取得する。ツール呼び出しが background ステータスで返された場合に使用する。"
        ),
        "en": (
            "Check the status of a background task. Specify a task_id to get its running/completed/failed status and result. Use this when a tool call returns with 'background' status."
        ),
    },
    "schema.check_background_task.task_id": {
        "ja": "確認するタスクのID（submit時に返されたID）",
        "en": "Task ID to check (the ID returned when submitted)",
    },
    "schema.check_permissions.desc": {
        "ja": "自分に現在許可されているツール・外部ツール・ファイルアクセスの一覧を確認する。何が使えて何が使えないかを事前に把握し、試行→失敗のサイクルを防ぐ。",
        "en": (
            "Check the list of currently permitted tools, external tools, and file access. Know what you can and cannot use in advance to avoid trial-and-error cycles."
        ),
    },
    "schema.create_skill.allowed_tools": {
        "ja": "frontmatter allowed_tools（任意）",
        "en": "Frontmatter allowed_tools (optional)",
    },
    "schema.create_skill.body": {
        "ja": "SKILL.md本文（Markdown）",
        "en": "SKILL.md body content (Markdown)",
    },
    "schema.create_skill.desc": {
        "ja": (
            "スキルをディレクトリ構造で作成する。SKILL.md（frontmatter + 本文）を生成し、オプションでreferences/やtemplates/にファイルを配置する。"
        ),
        "en": (
            "Create a skill with directory structure. Generates SKILL.md (frontmatter + body) and optionally places files in references/ and templates/."
        ),
    },
    "schema.create_skill.description": {
        "ja": "frontmatter description（トリガーキーワード含む）",
        "en": "Frontmatter description (include trigger keywords)",
    },
    "schema.create_skill.location": {
        "ja": "保存先。personal=個人スキル、common=共通スキル。デフォルト: personal",
        "en": ("Storage location. personal=personal skill, common=shared skill. Default: personal"),
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
        "ja": (
            "【重要】直属部下のAnimaにタスクを委譲する（部下のTaskExecが実行する。あなた自身は実行しない）。"
            "部下のタスクキューに追加し、state/pending/ に書き出して即時実行をトリガーする。"
            "同時にDMで指示を送信。自分側にも追跡用エントリが作成される。直属部下のみ操作可能。"
            "指示内容の書き方は common_knowledge/operations/task-delegation-guide.md を参照（MUST）。"
        ),
        "en": (
            "IMPORTANT: Delegate a task to a direct subordinate Anima — the SUBORDINATE executes it via their own TaskExec (not you). "
            "Adds to the subordinate's task queue and writes to state/pending/ to trigger immediate execution. "
            "Also sends a DM with instructions. A tracking entry is created on your side. Only direct subordinates can be targeted."
            " For instruction writing guidelines, read common_knowledge/operations/task-delegation-guide.md (MUST)."
        ),
    },
    "schema.submit_tasks.desc": {
        "ja": (
            "【重要】このツールで投入したタスクはあなた自身のTaskExecが実行します（部下には送られません）。"
            "部下にタスクを委任する場合は delegate_task を使ってください。"
            "複数タスクをDAGとして並列/直列実行する。parallel=trueのタスクは同時実行。depends_on指定タスクは依存完了後に実行。"
            "TaskExecはあなたの会話履歴を持たない。descriptionの書き方は common_knowledge/operations/task-delegation-guide.md を参照（MUST）。"
        ),
        "en": (
            "IMPORTANT: Tasks submitted here are executed by YOUR OWN TaskExec — they are NOT sent to subordinates. "
            "To delegate work to a subordinate, use delegate_task instead. "
            "Submit multiple tasks as a DAG for parallel/serial execution. "
            "Independent tasks with parallel=true run concurrently. "
            "Tasks with depends_on wait for dependencies to complete."
            " TaskExec has NO access to your conversation history. For description writing guidelines, read common_knowledge/operations/task-delegation-guide.md (MUST)."
        ),
    },
    "schema.delegate_task.instruction": {
        "ja": "タスクの指示内容",
        "en": "Task instructions",
    },
    "schema.delegate_task.name": {
        "ja": "委譲先の直属部下Anima名",
        "en": "Direct subordinate Anima name to delegate to",
    },
    "schema.delegate_task.summary": {
        "ja": "タスクの1行要約",
        "en": "One-line task summary",
    },
    "schema.delegate_task.workspace": {
        "ja": "ワークスペースエイリアスまたはalias#hash。部下がこのディレクトリで作業する",
        "en": "Workspace alias or alias#hash. The subordinate will work in this directory",
    },
    "schema.disable_subordinate.desc": {
        "ja": "配下のAnimaを休止させる（プロセス停止 + 自動復帰防止）。自分の配下であれば操作可能。",
        "en": ("Disable a descendant Anima (stop process + prevent auto-restart). Any descendant can be targeted."),
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
        "ja": "休止中の配下のAnimaを復帰させる。自分の配下であれば操作可能。",
        "en": "Re-enable a disabled descendant Anima. Any descendant can be targeted.",
    },
    "schema.enable_subordinate.name": {
        "ja": "復帰させる部下のAnima名（例: hinata）",
        "en": "Subordinate Anima name to enable (e.g. hinata)",
    },
    "schema.list_background_tasks.desc": {
        "ja": "バックグラウンドタスクの一覧を取得する。ステータスでフィルタリング可能（running/completed/failed）。省略時は全件を返す。",
        "en": (
            "List background tasks. Filter by status (running/completed/failed). Returns all tasks when status is omitted."
        ),
    },
    "schema.list_background_tasks.status": {
        "ja": "フィルタするステータス（省略時は全件）",
        "en": "Status to filter by (omit for all tasks)",
    },
    "schema.list_tasks.desc": {
        "ja": (
            "タスクキューの一覧を取得する。デフォルトはアクティブタスク（pending/in_progress/blocked/delegated）のみ。statusで特定ステータスをフィルタ可能。"
        ),
        "en": (
            "List tasks in the task queue. Defaults to active tasks (pending/in_progress/blocked/delegated). Use status to filter by specific status."
        ),
    },
    "schema.list_tasks.detail": {
        "ja": "trueで全フィールド（original_instruction全文含む）を返す。デフォルトはfalse（instruction先頭200文字）",
        "en": ("If true, return all fields including full original_instruction. Default false (first 200 chars)."),
    },
    "schema.list_tasks.status": {
        "ja": "フィルタするステータス（省略時はアクティブタスクのみ）",
        "en": "Status to filter by (omit for active tasks only)",
    },
    "schema.manage_channel.action": {
        "ja": "操作種別。create=チャネル作成, add_member=メンバー追加, remove_member=メンバー削除, info=チャネル情報表示",
        "en": (
            "Action type. create=create channel, add_member=add members, remove_member=remove members, info=show channel info"
        ),
    },
    "schema.manage_channel.channel": {
        "ja": "チャネル名（小文字英数字・ハイフン・アンダースコア）",
        "en": "Channel name (lowercase alphanumeric, hyphens, underscores)",
    },
    "schema.manage_channel.desc": {
        "ja": (
            "Boardチャネルのアクセス制御(ACL)を管理する。チャネルの作成、メンバーの追加・削除、チャネル情報の確認ができる。メンバーリストが空のチャネル（general, ops等）は全員アクセス可能。"
        ),
        "en": (
            "Manage Board channel access control (ACL). Create channels, add/remove members, and view channel info. Channels with an empty member list (general, ops, etc.) are accessible to all."
        ),
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
        "ja": (
            "配下全体の組織ダッシュボードを表示する。各Animaのプロセス状態・最終アクティビティ時刻・現在タスク要約・タスク数をツリー形式で一覧する。配下が多い場合も全員分を返す。"
        ),
        "en": (
            "Display the organization dashboard for all subordinates. Shows each Anima's process status, last activity time, current task summary, and task count in a tree format. Returns data for all subordinates regardless of count."
        ),
    },
    "schema.ping_subordinate.desc": {
        "ja": (
            "配下のAnimaの生存確認を行う。name を省略すると全配下を一括 ping する。指定すると単一Animaのみ確認する。プロセス状態・最終アクティビティ時刻・経過時間を返す。"
        ),
        "en": (
            "Check if subordinate Animas are alive. Omit name to ping all subordinates at once. Specify a name to check a single Anima. Returns process status, last activity time, and elapsed time."
        ),
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
        "ja": (
            "Boardの共有チャネルにメッセージを投稿する。チーム全体に共有すべき情報はgeneralチャネルに、運用・インフラ関連はopsチャネルに投稿する。全Animaが閲覧できるため、解決済み情報の共有やお知らせに使うこと。1対1の連絡にはsend_messageを使う。"
        ),
        "en": (
            "Post a message to a Board shared channel. Use the general channel for team-wide information and the ops channel for operations/infrastructure topics. All Animas can read shared channels, so use them for resolved info and announcements. For 1-on-1 communication, use send_message instead."
        ),
    },
    "schema.post_channel.text": {
        "ja": "投稿するメッセージ本文。@名前 でメンション可能（メンション先にDM通知される）。@all で起動中の全員にDM通知",
        "en": (
            "Message body to post. Use @name to mention (triggers DM notification to the mentioned person). @all sends DM notification to all active members"
        ),
    },
    "schema.read_channel.channel": {
        "ja": "チャネル名 (general, ops)",
        "en": "Channel name (general, ops)",
    },
    "schema.read_channel.desc": {
        "ja": (
            "Boardの共有チャネルの直近メッセージを読む。他のAnimaやユーザーが共有した情報を確認できる。human_only=trueでユーザー発言のみフィルタリング可能。inbox はチャネルではないため指定不可（inbox はシステムが自動処理）。"
        ),
        "en": (
            "Read recent messages from a Board shared channel. View information shared by other Animas and users. Set human_only=true to filter for human messages only. 'inbox' is not a channel and cannot be specified (inbox is processed automatically by the system)."
        ),
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
        "en": (
            "Read past DM history with a specific peer. View chronological history of messages sent/received via send_message. Use this when you need context from previous conversations."
        ),
    },
    "schema.read_dm_history.limit": {
        "ja": "取得件数（デフォルト: 20）",
        "en": "Number of messages to fetch (default: 20)",
    },
    "schema.read_dm_history.peer": {
        "ja": "DM相手の名前",
        "en": "Name of the DM peer",
    },
    "schema.read_subordinate_state.desc": {
        "ja": (
            "配下のAnimaの現在のタスク状態を読み取る。current_state.md（進行中タスク）と pending.md（保留タスク）の内容を返す。直属部下だけでなく孫以下の配下も指定可能。"
        ),
        "en": (
            "Read a subordinate Anima's current task state. Returns contents of current_state.md (active task) and pending.md (pending tasks). Can target any descendant, not just direct subordinates."
        ),
    },
    "schema.read_subordinate_state.name": {
        "ja": "読み取る配下のAnima名",
        "en": "Subordinate Anima name to read",
    },
    "schema.restart_subordinate.desc": {
        "ja": (
            "配下のAnimaプロセスを再起動する（配下であれば操作可能）。\nモデル変更（set_subordinate_model）後に呼び出すことで新モデルを即時反映できる。\nReconciliation ループが 30 秒以内にプロセスを再起動する。"
        ),
        "en": (
            "Restart a descendant Anima process (any descendant can be targeted).\nCall this after set_subordinate_model to apply the new model immediately.\nThe reconciliation loop will restart the process within 30 seconds."
        ),
    },
    "schema.restart_subordinate.name": {
        "ja": "再起動する部下のAnima名",
        "en": "Subordinate Anima name to restart",
    },
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
            "配下のAnimaのバックグラウンドモデル（heartbeat/cron用）を変更する（配下であれば操作可能）。\n変更は即時 status.json に保存される。反映には restart_subordinate を併用すること。\n\nバックグラウンドモデル未設定時はメインモデル（model）がそのまま使用される。\nクリアするには model に空文字 '' を指定する。"
        ),
        "en": (
            "Change a descendant's background model (for heartbeat/cron). Any descendant can be targeted.\nChanges are saved to status.json immediately. Use restart_subordinate to apply.\n\nWhen no background model is set, the main model is used.\nPass an empty string '' to clear the background model."
        ),
    },
    "schema.set_subordinate_background_model.model": {
        "ja": "バックグラウンドモデル名（空文字でクリア）",
        "en": "Background model name (empty string to clear)",
    },
    "schema.set_subordinate_background_model.name": {
        "ja": "対象の部下Anima名",
        "en": "Target subordinate Anima name",
    },
    "schema.set_subordinate_background_model.reason": {
        "ja": "変更理由",
        "en": "Reason for change",
    },
    "schema.set_subordinate_model.desc": {
        "ja": (
            "配下のAnimaのLLMモデルを変更する（配下であれば操作可能）。\n変更は即時 config.json に保存されるが、実行中プロセスへの反映には restart_subordinate を併用すること。\n\n指定するモデル名は provider/model_name 形式（Claude は prefix 不要）。\nKNOWN_MODELS 外の名前を指定した場合も警告のみで処理は続行する。\n\n主なモデル名:\n  [Mode S / Claude]\n  claude-opus-4-6            最高性能・推奨\n  claude-sonnet-4-6          バランス型・推奨\n  claude-haiku-4-5-20251001  軽量・高速（レガシー）\n  [Mode A / OpenAI]\n  openai/gpt-4.1             最新・コーディング強\n  openai/gpt-4.1-mini        高速・低コスト\n  openai/o4-mini-2025-04-16  推論・低コスト\n  [Mode A / Google]\n  google/gemini-2.5-pro      最高性能\n  google/gemini-2.5-flash    高速バランス\n  [Mode A / xAI]\n  xai/grok-4                 最新Grok\n  [Mode A / Ollama local]\n  ollama/glm-4.7             ローカル・tool_use対応\n  [Mode B / Ollama local]\n  ollama/gemma3:12b          中型ローカル\n"
        ),
        "en": (
            "Change a descendant's LLM model (any descendant can be targeted).\nChanges are saved to config.json immediately, but require restart_subordinate to take effect on a running process.\n\nModel names use provider/model_name format (Claude models need no prefix).\nUnknown model names produce a warning but processing continues.\n\nAvailable models:\n  [Mode S / Claude]\n  claude-opus-4-6            Highest performance, recommended\n  claude-sonnet-4-6          Balanced, recommended\n  claude-haiku-4-5-20251001  Lightweight, fast (legacy)\n  [Mode A / OpenAI]\n  openai/gpt-4.1             Latest, strong at coding\n  openai/gpt-4.1-mini        Fast, low cost\n  openai/o4-mini-2025-04-16  Reasoning, low cost\n  [Mode A / Google]\n  google/gemini-2.5-pro      Highest performance\n  google/gemini-2.5-flash    Fast, balanced\n  [Mode A / xAI]\n  xai/grok-4                 Latest Grok\n  [Mode A / Ollama local]\n  ollama/glm-4.7             Local, tool_use capable\n  [Mode B / Ollama local]\n  ollama/gemma3:12b          Mid-size local\n"
        ),
    },
    "schema.set_subordinate_model.model": {
        "ja": "新しいモデル名（例: claude-sonnet-4-6, openai/gpt-4.1）",
        "en": "New model name (e.g. claude-sonnet-4-6, openai/gpt-4.1)",
    },
    "schema.set_subordinate_model.name": {
        "ja": "変更する部下のAnima名",
        "en": "Subordinate Anima name to change",
    },
    "schema.set_subordinate_model.reason": {
        "ja": "変更理由（activity_log に記録される）",
        "en": "Reason for change (recorded in activity_log)",
    },
}
