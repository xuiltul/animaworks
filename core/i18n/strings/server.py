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
        "ja": "ワークスペース '{alias}' が見つかりません。もしかして: {suggestions}",
        "en": "Workspace '{alias}' not found. Did you mean: {suggestions}",
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
}
