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
    "startup.detail_preflight": {
        "ja": "実行エンジンの事前確認を実行中",
        "en": "Checking execution engine prerequisites",
        "ko": "실행 엔진 사전 점검 중",
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
    "cli.server_waiting_for_work": {
        "ja": "実行中の作業を待っています（最大 {seconds} 秒）",
        "en": "Waiting for active work to finish (up to {seconds} seconds)",
        "ko": "실행 중인 작업이 끝날 때까지 기다립니다 (최대 {seconds}초)",
    },
    "activity.cron_rerun": {
        "ja": "中断された cron を再実行します: {task_name}",
        "en": "Rerunning interrupted cron: {task_name}",
        "ko": "중단된 cron을 다시 실행합니다: {task_name}",
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
    "chat.attach.too_many_files": {
        "ja": "添付ファイルは{count}個までです",
        "en": "You can attach up to {count} files",
        "ko": "파일은 최대 {count}개까지 첨부할 수 있습니다",
    },
    "chat.attach.invalid_base64": {
        "ja": "ファイル「{name}」のデータを読み取れませんでした",
        "en": "Could not read file data for “{name}”",
        "ko": "파일 “{name}” 데이터를 읽을 수 없습니다",
    },
    "chat.attach.empty_file": {
        "ja": "空のファイル「{name}」は添付できません",
        "en": "The empty file “{name}” cannot be attached",
        "ko": "빈 파일 “{name}”은 첨부할 수 없습니다",
    },
    "chat.attach.file_too_large": {
        "ja": "ファイル「{name}」が大きすぎます（上限{size_mb}MB）",
        "en": "File “{name}” is too large (max {size_mb} MB)",
        "ko": "파일 “{name}”이 너무 큽니다 (최대 {size_mb}MB)",
    },
    "chat.attach.total_too_large": {
        "ja": "添付ファイルの合計が大きすぎます（上限{size_mb}MB）",
        "en": "The combined files are too large (max {size_mb} MB)",
        "ko": "첨부 파일 총 용량이 너무 큽니다 (최대 {size_mb}MB)",
    },
    "chat.attach.outbound_missing_request_id": {
        "ja": "返信のリクエストIDがないため添付できません",
        "en": "Cannot attach the file because the reply request id is missing",
        "ko": "답장 요청 ID가 없어 파일을 첨부할 수 없습니다",
    },
    "chat.attach.outbound_not_file": {
        "ja": "ファイルが見つかりません: {path}",
        "en": "File not found: {path}",
        "ko": "파일을 찾을 수 없습니다: {path}",
    },
    "chat.attach.outbound_empty_file": {
        "ja": "空のファイル「{name}」は添付できません",
        "en": "The empty file “{name}” cannot be attached",
        "ko": "빈 파일 “{name}”은 첨부할 수 없습니다",
    },
    "chat.attach.outbound_file_too_large": {
        "ja": "ファイル「{name}」が大きすぎます（上限{size_mb}MB）",
        "en": "File “{name}” is too large (max {size_mb} MB)",
        "ko": "파일 “{name}”이 너무 큽니다 (최대 {size_mb}MB)",
    },
    "chat.attach.outbound_too_many_files": {
        "ja": "1回の返信に添付できるファイルは{count}個までです",
        "en": "A reply can include up to {count} attached files",
        "ko": "답장 하나에 최대 {count}개의 파일을 첨부할 수 있습니다",
    },
    "chat.attach.preview_unsupported": {
        "ja": "このファイル形式はプレビューできません",
        "en": "This file type cannot be previewed",
        "ko": "이 파일 형식은 미리 볼 수 없습니다",
    },
    "chat.attach.preview_too_large": {
        "ja": "ファイルが大きすぎるためプレビューできません（上限50MB）",
        "en": "The file is too large to preview (max 50 MB)",
        "ko": "파일이 너무 커서 미리 볼 수 없습니다 (최대 50MB)",
    },
    "chat.attach.preview_unreadable": {
        "ja": "ファイルを読み取れないか、内容が破損しています",
        "en": "The file could not be read or is corrupted",
        "ko": "파일을 읽을 수 없거나 내용이 손상되었습니다",
    },
    "chat.attach.prompt_header": {
        "ja": "[添付ファイル]",
        "en": "[Attached files]",
        "ko": "[첨부 파일]",
    },
    "chat.attach.prompt_item": {
        "ja": "- {name}（{media_type}, {size}）: {path}",
        "en": "- {name} ({media_type}, {size}): {path}",
        "ko": "- {name}（{media_type}, {size}）: {path}",
    },
    "chat.attach.prompt_size_bytes": {
        "ja": "{size} B",
        "en": "{size} B",
        "ko": "{size} B",
    },
    "chat.attach.prompt_size_kb": {
        "ja": "{size} KB",
        "en": "{size} KB",
        "ko": "{size} KB",
    },
    "chat.attach.prompt_size_mb": {
        "ja": "{size} MB",
        "en": "{size} MB",
        "ko": "{size} MB",
    },
    "chat.internal_error": {
        "ja": "内部エラーが発生しました。再試行してください。",
        "en": "An internal error occurred. Please retry.",
    },
    "server.internal_auth_required": {
        "ja": "内部 API の認証に失敗しました",
        "en": "Internal API authentication required",
    },
    "server.internal_identity_mismatch": {
        "ja": "呼び出し元 '{caller}' は '{claimed}' として操作できません",
        "en": "Caller '{caller}' cannot act as '{claimed}'",
    },
    "server.internal_not_subordinate": {
        "ja": "呼び出し元 '{caller}' は '{claimed}' を操作する権限がありません",
        "en": "Caller '{caller}' is not authorized to access '{claimed}'",
    },
    "server.internal_newstaff_required": {
        "ja": "Anima の作成には newstaff スキルが必要です",
        "en": "The newstaff skill is required to create an Anima",
    },
    "server.channel_reply_to_not_found": {
        "ja": "返信先の投稿 '{post_id}' がこのチャネルにありません",
        "en": "Reply target '{post_id}' does not exist in this channel",
    },
    "server.channel_settings_forbidden": {
        "ja": "ボードの設定はオーナーまたは管理者だけが変更できます",
        "en": "Only an owner or admin can change board settings",
        "zh": "只有所有者或管理员可以更改看板设置",
        "ko": "보드 설정은 소유자 또는 관리자만 변경할 수 있습니다",
    },
    "server.channel_settings_description_too_long": {
        "ja": "説明は500文字までです",
        "en": "Description must be 500 characters or fewer",
        "zh": "说明最多 500 个字符",
        "ko": "설명은 500자까지입니다",
    },
    "server.channel_settings_member_not_found": {
        "ja": "Anima '{name}' が存在しません",
        "en": "Anima '{name}' does not exist",
        "zh": "Anima '{name}' 不存在",
        "ko": "Anima '{name}'이(가) 존재하지 않습니다",
    },
    "server.channel_settings_members_required": {
        "ja": "メンバー限定にするには、少なくとも1人のメンバーが必要です",
        "en": "At least one member is required for member-only boards",
        "zh": "仅限成员的看板至少需要一名成员",
        "ko": "멤버 전용 보드에는 최소 한 명의 멤버가 필요합니다",
    },
    "server.channel_settings_members_added": {
        "ja": "メンバーに {names} を足しました",
        "en": "Added {names} to the members",
        "zh": "已将 {names} 添加为成员",
        "ko": "멤버에 {names}을(를) 추가했습니다",
    },
    "server.channel_settings_members_removed": {
        "ja": "メンバーから {names} を外しました",
        "en": "Removed {names} from the members",
        "zh": "已将 {names} 从成员中移除",
        "ko": "멤버에서 {names}을(를) 제거했습니다",
    },
    "server.channel_settings_access_open": {
        "ja": "公開範囲を『社内に公開』にしました",
        "en": "Changed visibility to 'Open'",
        "zh": "已将公开范围设置为“公司内公开”",
        "ko": "공개 범위를 '회사 내 공개'로 변경했습니다",
    },
    "server.channel_settings_access_members": {
        "ja": "公開範囲を『メンバー限定』にしました",
        "en": "Changed visibility to 'Members only'",
        "zh": "已将公开范围设置为“仅限成员”",
        "ko": "공개 범위를 '멤버 한정'으로 변경했습니다",
    },
    "server.channel_settings_access_closed": {
        "ja": "公開範囲を『閉じる』にしました",
        "en": "Changed visibility to 'Closed'",
        "zh": "已将公开范围设置为“关闭”",
        "ko": "공개 범위를 '닫기'로 변경했습니다",
    },
    "server.channel_settings_description_changed": {
        "ja": "説明を書き換えました",
        "en": "Updated the description",
        "zh": "已更新说明",
        "ko": "설명을 변경했습니다",
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
    "setup.codex_models_unavailable": {
        "ja": "Codexの利用可能なモデルを確認できませんでした。ログイン状態と接続を確認して、もう一度お試しください。",
        "en": "Could not load available Codex models. Check your login and connection, then try again.",
    },
    "github_gateway.notify": {
        "ja": (
            "【GitHub通知】\n\n"
            "- PR: {repo}#{number} {title}\n"
            "- イベント: {event_label}\n"
            "- 起点: {author}\n"
            "- URL: {url}{ci_line}\n\n"
            "本文（先頭500文字）:\n{excerpt}"
        ),
        "en": (
            "[GitHub notification]\n\n"
            "- PR: {repo}#{number} {title}\n"
            "- Event: {event_label}\n"
            "- Origin: {author}\n"
            "- URL: {url}{ci_line}\n\n"
            "Body (first 500 chars):\n{excerpt}"
        ),
        "ko": (
            "【GitHub 알림】\n\n"
            "- PR: {repo}#{number} {title}\n"
            "- 이벤트: {event_label}\n"
            "- 시작자: {author}\n"
            "- URL: {url}{ci_line}\n\n"
            "본문(첫 500자):\n{excerpt}"
        ),
    },
    "github_gateway.notify_ci_line": {
        "ja": "\n- CI: {name} ({conclusion})",
        "en": "\n- CI: {name} ({conclusion})",
        "ko": "\n- CI: {name} ({conclusion})",
    },
}
