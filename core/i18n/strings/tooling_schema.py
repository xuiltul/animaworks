# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Domain-specific i18n strings (schema.*)."""

from __future__ import annotations

STRINGS: dict[str, dict[str, str]] = {
    "schema.attach_file.desc": {
        "ja": (
            "Webチャットで人間に返信するときに、手元のファイルを返信へ添付する。"
            "テキスト・Markdown・音声・動画・PDF・Excel・Word・PowerPoint・画像などに対応。"
            "ファイルの中身を本文に長々と貼る代わりに使う。1回で1ファイル、複数なら複数回呼ぶ。"
            "添付は返信の送信時にまとめて表示され、相手が画面で開ける。"
            "パスはAnimaの作業ディレクトリからの相対パスまたは絶対パスで指定する。"
        ),
        "en": (
            "Attach a local file to the reply when responding to a human in a web chat. "
            "Supports text, Markdown, audio, video, PDF, Excel, Word, PowerPoint, images, and more. "
            "Use this instead of pasting a long file body into the message. "
            "Attach one file per call; call it again for multiple files. "
            "Attachments appear together when the reply is sent, and the recipient can open them on screen. "
            "Use a path relative to the Anima working directory or an absolute path."
        ),
        "ko": (
            "웹 채팅에서 사람에게 답장할 때 로컬 파일을 답장에 첨부합니다. "
            "텍스트, Markdown, 오디오, 동영상, PDF, Excel, Word, PowerPoint, 이미지 등에 대응합니다. "
            "파일 내용을 본문에 길게 붙여넣는 대신 사용하세요. 한 번에 파일 하나를 첨부하고, 여러 개면 여러 번 호출합니다. "
            "첨부 파일은 답장 전송 시 함께 표시되며 상대방이 화면에서 열 수 있습니다. "
            "경로는 Anima 작업 디렉터리 기준 상대 경로 또는 절대 경로로 지정합니다."
        ),
    },
    "schema.attach_file.path": {
        "ja": "添付するファイルのパス（Animaの作業ディレクトリからの相対パスまたは絶対パス）",
        "en": "Path to the file to attach (relative to the Anima working directory or absolute)",
        "ko": "첨부할 파일 경로(Anima 작업 디렉터리 기준 상대 경로 또는 절대 경로)",
    },
    "schema.attach_file.name": {
        "ja": "表示名（省略時は元のファイル名）",
        "en": "Display name (defaults to the source filename)",
        "ko": "표시 이름(생략 시 원본 파일명)",
    },
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
        "ja": "委任経路（例: ['owner', 'sakura', 'rin']）",
        "en": "Delegation chain (e.g. ['owner', 'sakura', 'rin'])",
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
    "schema.call_human.sha": {
        "ja": "確認キー（8桁）。初回の呼び出しで返されるので、そのセッションでは以降これを付ける",
        "en": "Confirmation key (8 hex chars). Returned by the first call; include it on later calls in the same session",
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
    "schema.call_human.callback_id": {
        "ja": "インタラクティブ通知で使う安定したcallback ID（省略時は自動生成）",
        "en": "Stable callback ID for an interactive notification (auto-generated if omitted)",
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
    "schema.check_permissions.tool_name": {
        "ja": "特定ツールの許可状態を確認する場合に指定するツール名（任意）",
        "en": "Optional tool name for a focused permission check.",
    },
    "schema.check_permissions.action": {
        "ja": "特定ツールのアクションを確認する場合に指定するアクション名（任意）",
        "en": "Optional action name for a focused permission check.",
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
    "schema.delegate_task.desc": {
        "ja": (
            "直属部下のAnimaにタスクを委譲する（部下のTaskExecが実行する）。"
            "subject（と任意の version）を指定すると、同じ相手・同じ対象で未完了の依頼がある場合は"
            "新しいタスクを作らず既存の依頼のタスクIDを返す。"
            "詳細は operations/task-delegation-guide.md を参照。"
        ),
        "en": (
            "Delegate a task to a direct subordinate Anima (executed by the subordinate's TaskExec). "
            "When subject (and optionally version) is supplied and an open request for the same "
            "recipient and subject exists, it returns the existing task ID instead of creating a new task. "
            "A result with deduplicated=true means the request was consolidated into an existing task. "
            "See operations/task-delegation-guide.md for details."
        ),
        "ko": (
            "직속 부하 Anima에게 태스크를 위임한다(부하의 TaskExec가 실행). "
            "subject(선택적으로 version)를 지정하면 같은 상대·같은 대상의 미완료 요청이 있을 때 "
            "새 태스크를 만들지 않고 기존 요청의 태스크 ID를 반환한다. "
            "deduplicated=true는 기존 요청에 통합되었다는 뜻이다. "
            "상세는 operations/task-delegation-guide.md 참조."
        ),
    },
    "schema.submit_tasks.desc": {
        "ja": (
            "あなた自身のTaskExecでタスクを投入・実行する（部下には送られない）。部下への委任は delegate_task を使う。"
        ),
        "en": (
            "Submit tasks to be executed by your own TaskExec (not sent to subordinates). "
            "Use delegate_task to delegate work to a subordinate."
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
    "schema.delegate_task.acceptance_criteria": {
        "ja": "検証可能な受入条件。部下の実行プロンプトに埋め込まれる",
        "en": "Verifiable acceptance criteria. Embedded in the subordinate's execution prompt",
        "ko": "검증 가능한 수락 조건. 부하의 실행 프롬프트에 포함된다",
    },
    "schema.delegate_task.model": {
        "ja": "このタスクを実行するLLMモデルの上書き指定（例: 'claude-sonnet-5-5' や 'c:codex/gpt-5.6-sol'）。上長がタスクの重さに応じて指定できる。通常は未指定でよい（未指定ならanimaのデフォルトモデルを使う）",
        "en": "Optional LLM model override for this task (e.g. 'claude-sonnet-5-5' or 'c:codex/gpt-5.6-sol'). The manager can specify based on task weight. Usually leave unset (uses the Anima default model)",
        "ko": "이 태스크를 실행할 LLM 모델 오버라이드 (예: 'claude-sonnet-5-5' 또는 'c:codex/gpt-5.6-sol'). 관리자가 태스크 무게에 따라 지정할 수 있다. 보통은 미지정(기본 모델 사용)",
    },
    "schema.delegate_task.subject": {
        "ja": "任意の対象キー（例: github:owner/repo#123、issue:owner/repo#45、または自由文字列）。同じ委譲元・委譲先・subject・version の未完了依頼があれば、新規作成せず既存の依頼のタスクIDを返す。未指定なら従来どおり毎回新規作成する",
        "en": "Optional subject key (e.g. github:owner/repo#123, issue:owner/repo#45, or any free-form string). If an open request with the same delegator, recipient, subject and version exists, the existing task ID is returned instead of creating a new task. When unset, every call creates a new task as before",
        "ko": "임의의 대상 키(예: github:owner/repo#123, issue:owner/repo#45 또는 자유 문자열). 같은 위임자·대상·subject·version의 미완료 요청이 있으면 새로 만들지 않고 기존 요청의 태스크 ID를 반환한다. 미지정이면 기존과 같이 매번 새로 만든다",
    },
    "schema.delegate_task.version": {
        "ja": "任意の版キー（例: 対象の head SHA）。subject と組み合わせて依頼キーにする。版が変われば同じ対象でも別タスクになる",
        "en": "Optional version key (e.g. the target head SHA). Combined with subject to form the request key; a different version creates a separate task even for the same subject",
        "ko": "임의의 버전 키(예: 대상 head SHA). subject와 결합해 요청 키를 만든다. 버전이 다르면 같은 대상이라도 별도 태스크가 된다",
    },
    "schema.submit_tasks.resume": {
        "ja": "終了したpendingタスクを保存済みの実行入力で再開する。trueの場合はtask_idとresumeのみ指定。新規タスクにはtitleとdescriptionが必須。",
        "en": "Resume an ended pending task using its saved execution input. When true, provide only task_id and resume. New tasks require title and description.",
        "ko": "저장된 실행 입력으로 종료된 pending 작업을 재개합니다. true이면 task_id와 resume만 지정합니다. 새 작업에는 title과 description이 필요합니다.",
    },
    "schema.submit_tasks.task_model": {
        "ja": "このタスクを実行するLLMモデルの上書き指定（例: 'claude-sonnet-5-5' や 'c:codex/gpt-5.6-sol'）。通常は未指定でよい（未指定ならanimaのデフォルトモデルを使う）",
        "en": "Optional LLM model override for this task (e.g. 'claude-sonnet-5-5' or 'c:codex/gpt-5.6-sol'). Leave empty to use the Anima default model",
        "ko": "이 태스크를 실행할 LLM 모델 오버라이드 (예: 'claude-sonnet-5-5' 또는 'c:codex/gpt-5.6-sol'). 보통은 미지정(기본 모델 사용)",
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
            "タスクキューの一覧を取得する。デフォルトはアクティブタスク（pending/in_progress/delegated）のみ。statusで特定ステータスをフィルタ可能。"
        ),
        "en": (
            "List tasks in the task queue. Defaults to active tasks (pending/in_progress/delegated). Use status to filter by specific status."
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
        "ja": "操作種別。create=チャネル作成, archive=チャネルアーカイブ, add_member=メンバー追加, remove_member=メンバー削除, info=チャネル情報表示",
        "en": (
            "Action type. create=create channel, archive=archive channel, add_member=add members, remove_member=remove members, info=show channel info"
        ),
    },
    "schema.manage_channel.channel": {
        "ja": "チャネル名（小文字英数字・ハイフン・アンダースコア）",
        "en": "Channel name (lowercase alphanumeric, hyphens, underscores)",
    },
    "schema.manage_channel.desc": {
        "ja": (
            "Boardチャネルのアクセス制御(ACL)を管理する。チャネルの作成・アーカイブ、メンバーの追加・削除、チャネル情報の確認ができる。メンバーリストが空のチャネル（general, ops等）は全員アクセス可能。"
        ),
        "en": (
            "Manage Board channel access control (ACL). Create/archive channels, add/remove members, and view channel info. Channels with an empty member list (general, ops, etc.) are accessible to all."
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
            "Boardチャネルへの投稿は Bash から `animaworks-tool board post <channel> <text>` を実行する。長文は `animaworks-tool board post <channel> - <<'EOF'` で標準入力から渡せる。通常の作業報告・完了報告は所属チームのチャネルへ投稿し、1対1の連絡には send_message を使う。"
        ),
        "en": (
            "Post to a Board channel by running `animaworks-tool board post <channel> <text>` in Bash. For long text, use `animaworks-tool board post <channel> - <<'EOF'` and pass the body through stdin. Send routine work reports to your team channel first; use send_message for 1-to-1 communication."
        ),
    },
    "schema.post_channel.text": {
        "ja": "投稿するメッセージ本文。`animaworks-tool board post <channel> <text>` の本文として渡す。@名前 でメンション可能。",
        "en": ("Message body passed to `animaworks-tool board post <channel> <text>`. Use @name to mention an Anima."),
    },
    "schema.read_channel.channel": {
        "ja": "チャネル名（所属しているBoardチャネル名）",
        "en": "Channel name (any Board channel you belong to)",
    },
    "schema.read_channel.desc": {
        "ja": (
            "Boardチャネルの直近メッセージを読むには Bash から `animaworks-tool board read <channel> --limit N` を実行する。人間の投稿だけ読む場合は `--human-only` を付ける。inbox はチャネルではないため指定不可（inbox はシステムが自動処理）。"
        ),
        "en": (
            "Read recent Board messages by running `animaworks-tool board read <channel> --limit N` in Bash. Add `--human-only` to filter for human posts. 'inbox' is not a channel and cannot be specified (it is processed automatically by the system)."
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
            "配下のAnimaプロセスを再起動する（配下であれば操作可能）。\nモデル変更は root が稼働中プロセスへ reload_config を送るため通常は再起動不要。\nReconciliation ループが 30 秒以内にプロセスを再起動する。"
        ),
        "en": (
            "Restart a descendant Anima process (any descendant can be targeted).\nRoot sends reload_config after model changes, so a restart is normally unnecessary.\nThe reconciliation loop will restart the process within 30 seconds."
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
            "配下のAnimaのバックグラウンドモデル（heartbeat/cron用）を変更する（配下であれば操作可能）。\n変更は即時 status.json に保存され、root が稼働中プロセスへ再読み込みを依頼する。停止中の場合は次回起動時に反映される。\n\nバックグラウンドモデル未設定時はメインモデル（model）がそのまま使用される。\nクリアするには model に空文字 '' を指定する。"
        ),
        "en": (
            "Change a descendant's background model (for heartbeat/cron). Any descendant can be targeted.\nChanges are saved to status.json immediately and root asks running processes to reload it. Stopped Animas use it on next start.\n\nWhen no background model is set, the main model is used.\nPass an empty string '' to clear the background model."
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
            "配下のAnimaのLLMモデルを変更する（配下であれば操作可能）。\n変更は即時 status.json に保存され、root が稼働中プロセスへ再読み込みを依頼する。停止中の場合は次回起動時に反映される。\n\n指定するモデル名は provider/model_name 形式（Claude は prefix 不要）。\nKNOWN_MODELS 外の名前を指定した場合も警告のみで処理は続行する。\n\n主なモデル名:\n  [Mode S / Claude]\n  claude-opus-5-5            最高性能・推奨\n  claude-sonnet-5-5          バランス型・推奨\n  claude-haiku-4-5-20251001  軽量・高速（レガシー）\n  [Mode A / OpenAI]\n  openai/gpt-4.1             最新・コーディング強\n  openai/gpt-4.1-mini        高速・低コスト\n  openai/o4-mini-2025-04-16  推論・低コスト\n  [Mode A / Google]\n  google/gemini-2.5-pro      最高性能\n  google/gemini-2.5-flash    高速バランス\n  [Mode A / xAI]\n  xai/grok-4                 最新Grok\n  [Mode A / Ollama local]\n  ollama/glm-4.7             ローカル・tool_use対応\n  [Mode A / Ollama local]\n  ollama/gemma3:12b          中型ローカル\n"
        ),
        "en": (
            "Change a descendant's LLM model (any descendant can be targeted).\nChanges are saved to status.json immediately and root asks running processes to reload them. Stopped Animas use the new model on next start.\n\nModel names use provider/model_name format (Claude models need no prefix).\nUnknown model names produce a warning but processing continues.\n\nAvailable models:\n  [Mode S / Claude]\n  claude-opus-5-5            Highest performance, recommended\n  claude-sonnet-5-5          Balanced, recommended\n  claude-haiku-4-5-20251001  Lightweight, fast (legacy)\n  [Mode A / OpenAI]\n  openai/gpt-4.1             Latest, strong at coding\n  openai/gpt-4.1-mini        Fast, low cost\n  openai/o4-mini-2025-04-16  Reasoning, low cost\n  [Mode A / Google]\n  google/gemini-2.5-pro      Highest performance\n  google/gemini-2.5-flash    Fast, balanced\n  [Mode A / xAI]\n  xai/grok-4                 Latest Grok\n  [Mode A / Ollama local]\n  ollama/glm-4.7             Local, tool_use capable\n  [Mode A / Ollama local]\n  ollama/gemma3:12b          Mid-size local\n"
        ),
    },
    "schema.set_subordinate_model.model": {
        "ja": "新しいモデル名（例: claude-sonnet-5-5, openai/gpt-4.1）",
        "en": "New model name (e.g. claude-sonnet-5-5, openai/gpt-4.1)",
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
