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
        "ja": "通知の本文。パッと見て人間が判断できるよう、話し言葉でシンプルに簡潔に書くこと。詳細（ログ・集計・経緯）は本文に貼らず、ファイルパスやURLで参照させる",
        "en": "Notification body. Write simply and concisely in plain conversational language so a human can decide at a glance. Do not paste details (logs, tallies, background) into the body; reference them by file path or URL",
    },
    "schema.call_human.desc": {
        "ja": (
            "人間の管理者に連絡します。重要な報告、問題のエスカレーション、判断が必要な事項がある場合に使用してください。チャット画面と外部通知チャネル（Slack等）の両方に届きます。本文はパッと見て判断できるよう話し言葉で簡潔に。"
        ),
        "en": (
            "Contact the human administrator. Use this for important reports, problem escalation, or matters requiring human judgment. Notifications are delivered to both the chat UI and external channels (Slack, etc.). Keep the body short and conversational so a human can decide at a glance."
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
    "tools.call_human.callback_id": {
        "ja": "callback_id: {callback_id}",
        "en": "callback_id: {callback_id}",
    },
    "schema.call_human.interactive_desc": {
        "ja": "trueの場合、承認ボタン付きメッセージを送信",
        "en": "When true, send message with approval buttons",
    },
    "schema.call_human.options_desc": {
        "ja": "選択肢のリスト",
        "en": "List of response options",
    },
    "schema.call_human.category_desc": {
        "ja": "通知カテゴリ（inbox分岐用）",
        "en": "Notification category for inbox routing",
    },
    "schema.call_human.allowed_users_desc": {
        "ja": "プラットフォーム別の承認者ID",
        "en": "Per-platform approver user IDs",
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
            "【instruction必須ルール】TaskExecはMinimalコンテキスト（identity数行+タスク記述のみ）で動作し、"
            "あなたの会話履歴・DM履歴・短期記憶・Priming結果には一切アクセスできない。"
            "instructionは完全に自己完結させること。"
            "禁止: ❌「先ほどの報告内容をベースに」❌「前回の続き」❌「さっきのDMの内容」等の曖昧な参照（実行者はその情報を持たない）。"
            "必須: ✅具体的データ・本文を直接instructionに埋め込む ✅ファイルパス・完了条件を明記。"
            "詳細は common_knowledge/operations/task-delegation-guide.md を参照。"
        ),
        "en": (
            "IMPORTANT: Delegate a task to a direct subordinate Anima — the SUBORDINATE executes it via their own TaskExec (not you). "
            "Adds to the subordinate's task queue and writes to state/pending/ to trigger immediate execution. "
            "Also sends a DM with instructions. A tracking entry is created on your side. Only direct subordinates can be targeted. "
            "INSTRUCTION RULES: TaskExec runs with Minimal context (identity few lines + task description only) "
            "and has NO access to your conversation history, DM history, short-term memory, or Priming results. "
            "The instruction MUST be completely self-contained. "
            "FORBIDDEN: ❌ 'based on the report you sent earlier' ❌ 'continue from last time' ❌ 'the content from the DM' "
            "— the executor has NONE of that context. "
            "REQUIRED: ✅ Embed concrete data/content directly in the instruction ✅ Specify file paths and completion criteria. "
            "Details: common_knowledge/operations/task-delegation-guide.md."
        ),
        "ko": (
            "【중요】직속 부하 Anima에게 태스크를 위임한다(부하의 TaskExec가 실행. 본인은 실행하지 않음). "
            "부하의 태스크 큐에 추가하고 state/pending/에 기록하여 즉시 실행을 트리거한다. "
            "동시에 DM으로 지시를 전송. 본인 측에도 추적용 엔트리가 생성된다. 직속 부하만 대상 가능. "
            "【instruction 필수 규칙】TaskExec는 Minimal 컨텍스트(identity 수 줄 + 태스크 기술만)로 동작하며, "
            "대화 이력・DM 이력・단기 기억・Priming 결과에 일절 접근할 수 없다. "
            "instruction은 완전히 자기 완결적이어야 한다. "
            "금지: ❌「앞서 보고한 내용을 기반으로」❌「지난번 계속」❌「아까 DM으로 보낸 내용」등 모호한 참조(실행자는 그 정보가 없음). "
            "필수: ✅구체적 데이터・본문을 직접 instruction에 포함 ✅파일 경로・완료 조건을 명기. "
            "상세: common_knowledge/operations/task-delegation-guide.md 참조."
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
        "ja": (
            "タスクの指示内容。完全に自己完結させること（TaskExecは会話履歴・DM履歴を持たない。"
            "「先ほどの内容」等の参照は禁止→データを直接埋め込む）。"
            "成果物パスは委譲先が書き込み可能な場所のみ。自分のknowledge/は指定不可、共有にはcommon_knowledge/を使う"
        ),
        "en": (
            "Task instructions. MUST be fully self-contained (TaskExec has no conversation/DM history — "
            "references like 'the earlier report' are forbidden; embed data directly). "
            "Output paths must be writable by the subordinate — never your own knowledge/; use common_knowledge/ for shared output"
        ),
        "ko": (
            "태스크 지시 내용. 완전히 자기 완결적이어야 함(TaskExec는 대화/DM 이력이 없음. "
            "「앞서 보고한 내용」등의 참조 금지 → 데이터를 직접 포함). "
            "산출물 경로는 위임 대상이 기록 가능한 곳만 지정. 자신의 knowledge/는 지정 불가, 공유에는 common_knowledge/를 사용"
        ),
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
        "ja": "登録aliasを使う（例: aischreiber）。ディレクトリ名表記(AI-Schreiber)も正規化で受理される。ワークスペースエイリアスまたはalias#hash。部下がこのディレクトリで作業する",
        "en": "Use a registered alias (e.g. aischreiber). Directory-style names (AI-Schreiber) are also accepted via normalization. Workspace alias or alias#hash. The subordinate will work in this directory",
    },
    "schema.delegate_task.exclusive_key": {
        "ja": "同時実行を避けたいタスク群で共有する排他キー（例: pr-3999）。同一キーのタスクは直列実行される。省略時は排他なし",
        "en": (
            "An exclusion key shared by tasks that must not run concurrently (e.g. pr-3999). "
            "Tasks with the same key run serially. Omit for no exclusion"
        ),
        "ko": (
            "동시 실행을 피해야 하는 태스크 그룹이 공유하는 배타 키(예: pr-3999). "
            "동일한 키의 태스크는 직렬 실행된다. 생략 시 배타 없음"
        ),
    },
    "schema.delegate_task.acceptance_criteria": {
        "ja": "検証可能な受入条件。部下の実行プロンプトに埋め込まれる",
        "en": "Verifiable acceptance criteria. Embedded in the subordinate's execution prompt",
        "ko": "검증 가능한 수락 조건. 부하의 실행 프롬프트에 포함된다",
    },
    "schema.delegate_task.model": {
        "ja": "このタスクを実行するLLMモデルの上書き指定（例: 'claude-sonnet-4-6' や 'c:codex/gpt-5.6-sol'）。上長がタスクの重さに応じて指定できる。通常は未指定でよい（未指定ならanimaのデフォルトモデルを使う）",
        "en": "Optional LLM model override for this task (e.g. 'claude-sonnet-4-6' or 'c:codex/gpt-5.6-sol'). The manager can specify based on task weight. Usually leave unset (uses the Anima default model)",
        "ko": "이 태스크를 실행할 LLM 모델 오버라이드 (예: 'claude-sonnet-4-6' 또는 'c:codex/gpt-5.6-sol'). 관리자가 태스크 무게에 따라 지정할 수 있다. 보통은 미지정(기본 모델 사용)",
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
            "部下Animaが稼働しているか確認する唯一の正しいツール。"
            "部下の存在確認・生死確認・稼働確認・パス確認はすべてこのツールを使う。"
            "dir / find / search_memory / ReadMemoryFile など他ツールで部下を探してはいけない。"
            "nameを省略すると全部下を一括ping。指定すると単一Animaのみ確認。"
            "プロセス状態・最終アクティビティ時刻・経過時間を返す。"
        ),
        "en": (
            "The ONLY correct tool to verify whether a subordinate Anima is running. "
            "Use this for all subordinate existence checks, liveness checks, and status checks. "
            "Do NOT use dir, find, search_memory, or ReadMemoryFile to look for subordinates. "
            "Omit name to ping all subordinates; specify a name to check one Anima. "
            "Returns process status, last activity time, and elapsed time."
        ),
    },
    "schema.ping_subordinate.name": {
        "ja": "確認するAnima名（省略時は全配下）",
        "en": "Anima name to check (omit to ping all subordinates)",
    },
    "schema.post_channel.channel": {
        "ja": "チャネル名（所属している部門/チームのBoardチャネル名。general=全体共有, ops=運用系）",
        "en": "Channel name (your team/department Board channel, or general for org-wide, ops for operations)",
    },
    "schema.post_channel.desc": {
        "ja": (
            "Boardチャネルにメッセージを投稿する。通常の作業報告・完了報告は、まず所属している部門/チームのチャネルに投稿する。generalは全体共有、opsは部門横断の運用・インフラ共有に使う。1対1の連絡にはsend_messageを使う。"
        ),
        "en": (
            "Post a message to a Board channel. Routine work reports and completion updates should go to your team/department channel first. Use general for org-wide sharing and ops for cross-team operations or infrastructure topics. Use send_message for 1-to-1 communication."
        ),
    },
    "schema.post_channel.text": {
        "ja": "投稿するメッセージ本文。@名前 でメンション可能（メンション先にDM通知される）。@all で起動中の全員にDM通知",
        "en": (
            "Message body to post. Use @name to mention (triggers DM notification to the mentioned person). @all sends DM notification to all active members"
        ),
    },
    "schema.read_channel.channel": {
        "ja": "チャネル名（所属しているBoardチャネル名）",
        "en": "Channel name (any Board channel you belong to)",
    },
    "schema.read_channel.desc": {
        "ja": (
            "Boardチャネルの直近メッセージを読む。他のAnimaやユーザーが共有した情報を確認できる。所属している部門/チームチャネルや general / ops を必要に応じて参照する。human_only=trueでユーザー発言のみフィルタリング可能。inbox はチャネルではないため指定不可（inbox はシステムが自動処理）。"
        ),
        "en": (
            "Read recent messages from a Board channel. Review information shared by other Animas and users in your team/department channels or in general/ops as needed. Set human_only=true to filter for human messages only. 'inbox' is not a channel and cannot be specified (it is processed automatically by the system)."
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
    "schema.read_dm_history.direction": {
        "ja": "フィルタ方向: sent（送信のみ）, received（受信のみ）, both（両方、デフォルト）",
        "en": "Filter direction: sent (outbound only), received (inbound only), both (default)",
    },
    "schema.read_dm_history.hours": {
        "ja": "直近N時間以内に限定（省略時は全期間）",
        "en": "Limit to last N hours (omit for all time)",
    },
    "schema.read_dm_history.keyword": {
        "ja": "メッセージ内容に含まれるキーワードで絞り込み",
        "en": "Filter messages containing this keyword in content",
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
    "tooling_schema.sheets_write_caution": {
        "ja": "上書きに注意。既存データ確認にはread_valuesを先に使う。",
        "en": "Overwrites existing content — call read_values first to inspect the current data.",
        "ko": "덮어쓰기에 주의하세요. 기존 데이터를 먼저 read_values로 확인하세요.",
    },
}
