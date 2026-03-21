# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Domain-specific i18n strings."""

from __future__ import annotations

STRINGS: dict[str, dict[str, str]] = {
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
