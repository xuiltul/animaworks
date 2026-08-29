# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Domain-specific i18n strings."""

from __future__ import annotations

STRINGS: dict[str, dict[str, str]] = {
    "startup.page_title": {
        "ja": "AnimaWorks を起動中",
        "en": "Starting AnimaWorks",
        "ko": "AnimaWorks 시작 중",
    },
    "startup.in_progress": {
        "ja": "初期化が完了するまでこのページは自動更新されます。",
        "en": "This page will refresh automatically until initialization completes.",
        "ko": "초기화가 완료될 때까지 이 페이지가 자동으로 새로고침됩니다.",
    },
    "startup.failed": {
        "ja": "初期化に失敗しました。サーバーは起動していますが、詳細を確認してください。",
        "en": "Initialization failed. The server is running, but needs attention.",
        "ko": "초기화에 실패했습니다. 서버는 실행 중이지만 확인이 필요합니다.",
    },
    "startup.label_phase": {
        "ja": "フェーズ",
        "en": "Phase",
        "ko": "단계",
    },
    "startup.label_detail": {
        "ja": "処理対象",
        "en": "Current item",
        "ko": "현재 항목",
    },
    "startup.label_progress": {
        "ja": "進捗",
        "en": "Progress",
        "ko": "진행률",
    },
    "startup.label_elapsed": {
        "ja": "経過時間",
        "en": "Elapsed",
        "ko": "경과 시간",
    },
    "startup.detail_pending": {
        "ja": "準備中",
        "en": "Preparing",
        "ko": "준비 중",
    },
    "startup.detail_starting": {
        "ja": "起動準備中",
        "en": "Preparing startup",
        "ko": "시작 준비 중",
    },
    "startup.detail_vector_worker": {
        "ja": "ベクターワーカーを起動中",
        "en": "Starting vector worker",
        "ko": "벡터 워커 시작 중",
    },
    "startup.detail_preflight": {
        "ja": "RAG preflight を実行中",
        "en": "Running RAG preflight",
        "ko": "RAG preflight 실행 중",
    },
    "startup.detail_spawning": {
        "ja": "Anima プロセスを起動中",
        "en": "Starting Anima processes",
        "ko": "Anima 프로세스 시작 중",
    },
    "startup.detail_ready": {
        "ja": "起動完了",
        "en": "Startup complete",
        "ko": "시작 완료",
    },
    "startup.detail_failed": {
        "ja": "起動初期化に失敗",
        "en": "Startup initialization failed",
        "ko": "시작 초기화 실패",
    },
    "startup.detail_setup_mode": {
        "ja": "セットアップモード",
        "en": "Setup mode",
        "ko": "설정 모드",
    },
    "startup.phase.starting": {
        "ja": "起動準備",
        "en": "Starting",
        "ko": "시작 준비",
    },
    "startup.phase.preflight": {
        "ja": "RAG preflight",
        "en": "RAG preflight",
        "ko": "RAG preflight",
    },
    "startup.phase.repairing": {
        "ja": "RAG repair",
        "en": "RAG repair",
        "ko": "RAG repair",
    },
    "startup.phase.indexing": {
        "ja": "インデックス作成",
        "en": "Indexing",
        "ko": "인덱싱",
    },
    "startup.phase.spawning_animas": {
        "ja": "Anima 起動",
        "en": "Starting Animas",
        "ko": "Anima 시작",
    },
    "startup.phase.ready": {
        "ja": "準備完了",
        "en": "Ready",
        "ko": "준비 완료",
    },
    "startup.phase.failed": {
        "ja": "失敗",
        "en": "Failed",
        "ko": "실패",
    },
    "chat.anima_restarting": {
        "ja": "Animaが再起動中です。しばらく待ってから再試行してください。",
        "en": "Anima is restarting. Please wait and retry.",
    },
    "chat.anima_unavailable": {
        "ja": "Animaのプロセスに接続できません。再起動中の可能性があります。",
        "en": "Cannot connect to Anima process. It may be restarting.",
    },
    "chat.bootstrap_busy": {
        "ja": "初期化中です",
        "en": "Initializing",
    },
    "chat.bootstrap_error": {
        "ja": "現在キャラクターを作成中です。完了までお待ちください。",
        "en": "Character is being created. Please wait for completion.",
    },
    "chat.communication_error": {
        "ja": "通信エラーが発生しました。再試行してください。",
        "en": "Communication error. Please retry.",
    },
    "chat.connection_lost": {
        "ja": "通信が切断されました。再試行してください。",
        "en": "Connection was lost. Please retry.",
    },
    "chat.heartbeat_processing": {
        "ja": "処理中です",
        "en": "Processing",
    },
    "chat.image_too_large": {
        "ja": "画像データが大きすぎます（{size_mb}MB / 上限20MB）",
        "en": "Image data too large ({size_mb}MB / max 20MB)",
    },
    "chat.internal_error": {
        "ja": "内部エラーが発生しました。再試行してください。",
        "en": "An internal error occurred. Please retry.",
    },
    "chat.message_too_large": {
        "ja": "メッセージが大きすぎます（{size_mb}MB / 上限10MB）",
        "en": "Message too large ({size_mb}MB / max 10MB)",
    },
    "chat.stream_incomplete": {
        "ja": "ストリームが予期せず終了しました。再試行してください。",
        "en": "Stream ended unexpectedly. Please retry.",
    },
    "chat.stream_not_found": {
        "ja": "ストリームが見つからないか、アクセスが拒否されました",
        "en": "Stream not found or access denied",
    },
    "chat.timeout": {
        "ja": "応答がタイムアウトしました",
        "en": "Response timed out",
    },
    "chat.unsupported_image_format": {
        "ja": "未対応の画像形式です: {media_type}",
        "en": "Unsupported image format: {media_type}",
    },
    "workspace.dir_not_found": {
        "ja": "ワークスペースディレクトリ '{path}' が存在しません。",
        "en": "Workspace directory '{path}' does not exist.",
    },
    "workspace.not_found": {
        "ja": "ワークスペース '{alias}' が見つかりません。エイリアス、ハッシュ、または絶対パスを確認してください。",
        "en": "Workspace '{alias}' not found. Check the alias, hash, or absolute path.",
    },
    "workspace.not_found_with_suggestions": {
        "ja": "ワークスペース '{alias}' が見つかりません。もしかして: {suggestions}。登録済みワークスペース: {available}",
        "en": "Workspace '{alias}' not found. Did you mean: {suggestions}. Available workspaces: {available}",
    },
    "workspace.not_found_with_list": {
        "ja": "ワークスペース '{alias}' が見つかりません。登録済みワークスペース: {available}",
        "en": "Workspace '{alias}' not found. Available workspaces: {available}",
    },
    "workspace.registered": {
        "ja": "ワークスペースを登録しました: {qualified} → {path}",
        "en": "Workspace registered: {qualified} → {path}",
    },
    "workspace.removed": {
        "ja": "ワークスペース '{alias}' を削除しました。",
        "en": "Workspace '{alias}' removed.",
    },
    "workspace.resolve_error": {
        "ja": "ワークスペースの解決に失敗しました: {error}",
        "en": "Failed to resolve workspace: {error}",
    },
    "setup.cli_tools_auth": {
        "ja": "CLIツール認証状態",
        "en": "CLI Tools Auth Status",
    },
    "setup.cli_tools_claude_code": {
        "ja": "Claude Code CLI",
        "en": "Claude Code CLI",
    },
    "setup.cli_tools_codex_cli": {
        "ja": "Codex CLI",
        "en": "Codex CLI",
    },
    "setup.cli_tools_codex_login": {
        "ja": "Codex Login",
        "en": "Codex Login",
    },
    "setup.cli_tools_cursor_agent": {
        "ja": "Cursor Agent CLI",
        "en": "Cursor Agent CLI",
    },
    "setup.cli_tools_cursor_auth": {
        "ja": "Cursor Agent 認証",
        "en": "Cursor Agent Auth",
    },
    "setup.cli_tools_gemini_cli": {
        "ja": "Gemini CLI",
        "en": "Gemini CLI",
    },
    "setup.cli_tools_gemini_auth": {
        "ja": "Gemini CLI 認証",
        "en": "Gemini CLI Auth",
    },
    "github_gateway.queue_ack": {
        "ja": "受領しました（task `{task_id}`）。同一PRで実行中の先行タスク `{holder_task_id}` が終わり次第、自動で着手します。この間に新しいpushがあった場合は旧headの処理を打ち切り、最新headを優先します。",
        "en": "Received (task `{task_id}`). It will start automatically once the in-flight task `{holder_task_id}` on this PR finishes. If a new push lands in the meantime, work on the old head is dropped and the latest head takes priority.",
    },
    "github_gateway.review_dispatch": {
        "ja": (
            "【PR新規コミット検出（push静穏確認済み）】\n\n"
            "- {pr_key} {sha}: {title}\n\n"
            "最終pushから{quiet}以上静穏を確認済みです。"
            "上記PRの current HEAD に対する差分レビュー/FRCを直ちに実施してください。"
            "過去HEADへのレビューは新push時点で無効です。"
            "複数件ある場合はbackgroundタスクとして並列に処理して構いません。"
        ),
        "en": (
            "[New PR commit detected (push quiet period confirmed)]\n\n"
            "- {pr_key} {sha}: {title}\n\n"
            "No pushes have occurred for at least {quiet}. Review/FRC the current HEAD immediately. "
            "Reviews of previous HEADs became invalid when the new push arrived. "
            "Multiple PRs may be processed in parallel as background tasks."
        ),
    },
    "github_gateway.minutes": {"ja": "{value}分", "en": "{value} minutes"},
    "github_gateway.seconds": {"ja": "{value}秒", "en": "{value} seconds"},
    "github_gateway.conflict": {
        "ja": (
            "【要対応・マージコンフリクト継続検知】\n\n"
            "- PR: {repo}#{number}\n- HEAD: {sha}\n- URL: {url}\n\n"
            "baseブランチとのコンフリクトが残っています。重複通知でも無視せず、"
            "natsumeのcanonical laneが実際に解消pushを完了するまで追跡してください。"
        ),
        "en": (
            "[ACTION REQUIRED: merge conflict still present]\n\n"
            "- PR: {repo}#{number}\n- HEAD: {sha}\n- URL: {url}\n\n"
            "The base-branch conflict remains. Even if this notification repeats, track the canonical "
            "natsume lane until the conflict-resolution push is actually complete."
        ),
    },
    "github_gateway.unknown_verdict": {"ja": "判定不明", "en": "Unknown verdict"},
    "github_gateway.frc_result": {
        "ja": (
            "【FRC結果検知】\n\n"
            "- 判定: {verdict}\n- PR: {repo}#{number}\n- HEAD: {head_sha}\n"
            "- URL: {url}\n- 本文全文:\n{summary}\n\n"
            "HOLDの場合は procedures/pr-event-detection-patrol.md に従って"
            "natsumeへの修正ディスパッチを実施してください。"
        ),
        "en": (
            "[FRC result detected]\n\n"
            "- Verdict: {verdict}\n- PR: {repo}#{number}\n- HEAD: {head_sha}\n"
            "- URL: {url}\n- Full body:\n{summary}\n\n"
            "For HOLD, follow procedures/pr-event-detection-patrol.md and dispatch the fix to natsume."
        ),
    },
    "github_gateway.ci_failure": {
        "ja": "【CI FAILURE検知】\n\n{lines}\n  {url}\n\n修正担当（natsume）へのディスパッチをお願いします。",
        "en": "[CI FAILURE detected]\n\n{lines}\n  {url}\n\nPlease dispatch this to the fix owner (natsume).",
    },
    "github_gateway.command_task": {
        "ja": (
            "GitHub の {repo}#{number} に次のコメントが投稿された。\n\n{body}\n\nURL: {url}\n\n"
            "上記コメントの指示に従って対応せよ。コンフリクト解消の場合は "
            "procedures/pr-conflict-resolution.md の手順（worktreeでorigin/baseをmerge・"
            "テスト通過確認・通常push・force-push禁止）に従う。"
        ),
        "en": (
            "The following comment was posted on GitHub at {repo}#{number}.\n\n{body}\n\nURL: {url}\n\n"
            "Follow the instructions in the comment. For conflict resolution, follow "
            "procedures/pr-conflict-resolution.md: merge origin/base in the branch worktree, "
            "confirm tests pass, use a normal push, and never force-push."
        ),
    },
    "github_gateway.command_summary": {
        "ja": "GitHubコメント対応 {repo}#{number}",
        "en": "Handle GitHub comment {repo}#{number}",
    },
    "github_gateway.ci_task": {
        "ja": (
            "PR #{number} ({pr_url}) の CI ({workflow_name}) が head {sha} で失敗。"
            "原因を調査し修正をpushせよ。\nworkflow URL: {workflow_url}"
        ),
        "en": (
            "CI ({workflow_name}) failed at head {sha} for PR #{number} ({pr_url}). "
            "Investigate the cause, fix it, and push the fix.\nworkflow URL: {workflow_url}"
        ),
    },
    "github_gateway.ci_summary": {
        "ja": "CI失敗修正 {repo}#{number}",
        "en": "Fix CI failure {repo}#{number}",
    },
    "github_gateway.review_task_bot_note": {
        "ja": "bot由来のCHANGES_REQUESTEDです。",
        "en": "This CHANGES_REQUESTED review came from a bot.",
    },
    "github_gateway.review_task_human_note": {
        "ja": "人間レビュアー由来です。",
        "en": "This review came from a human reviewer.",
    },
    "github_gateway.review_task_summary": {
        "ja": "レビュー指摘対応 {repo}#{number}",
        "en": "Address review feedback {repo}#{number}",
    },
    "github_gateway.review_task": {
        "ja": (
            "PR #{number} ({url}) に @{author} から CHANGES_REQUESTED が投稿された。{bot_note}\n\n"
            "レビュー本文:\n{body}\n\n"
            "指摘を確認して必要な修正を行うこと。レビュアーが人間の場合、指摘に技術的に"
            "同意できない時は独断で押し切らず上長(rin)へ報告して判断を仰ぐこと。"
        ),
        "en": (
            "@{author} submitted CHANGES_REQUESTED on PR #{number} ({url}). {bot_note}\n\n"
            "Review body:\n{body}\n\n"
            "Review the feedback and make the necessary changes. If the reviewer is human and you "
            "technically disagree, do not override the feedback unilaterally; report it to your manager "
            "(rin) and ask for a decision."
        ),
    },
}
