# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Domain-specific i18n strings (machine.*, notion.*)."""

from __future__ import annotations

STRINGS: dict[str, dict[str, str]] = {
    "machine.empty_instruction": {
        "ja": "instruction が空です。工作機械への詳細な作業指示を記述してください。",
        "en": "Instruction is empty. Provide detailed task instructions for the machine.",
    },
    "machine.engine_desc_claude": {
        "ja": "Claude CLI — 高品質・豊富なツール。コード実装・リファクタ・ドキュメント生成に最適",
        "en": ("Claude CLI — High quality, rich tools. Best for implementation, refactoring, documentation"),
    },
    "machine.engine_desc_codex": {
        "ja": "Codex CLI — ネイティブサンドボックス。安全性重視のファイル操作に最適",
        "en": "Codex CLI — Native sandbox. Best for safety-critical file operations",
    },
    "machine.engine_desc_cursor_agent": {
        "ja": "Cursor Agent — IDE統合・高速。コード実装・ワークスペース操作に最適",
        "en": ("Cursor Agent — IDE-integrated, fast. Best for code implementation and workspace operations"),
    },
    "machine.engine_desc_gemini": {
        "ja": "Gemini CLI — Google AI。探索・分析に最適",
        "en": "Gemini CLI — Google AI. Best for exploration and analysis",
    },
    "machine.engine_failed": {
        "ja": "工作機械 '{engine}' がエラーコード {code} で終了しました。",
        "en": "Machine engine '{engine}' exited with code {code}.",
    },
    "machine.engine_disabled": {
        "ja": "エンジン '{engine}' は現在の運用方針で無効です。使用可能: {available}",
        "en": "Engine '{engine}' is disabled by the current policy. Available: {available}",
    },
    "machine.engine_not_found": {
        "ja": "工作機械 '{engine}' が見つかりません。インストールされているか確認してください。",
        "en": "Machine engine '{engine}' not found. Please verify it is installed.",
    },
    "machine.forbidden_directory": {
        "ja": "作業ディレクトリ '{path}' はAnima記憶領域と重複するため使用できません。",
        "en": ("Working directory '{path}' overlaps with Anima memory directories and cannot be used."),
    },
    "machine.fs_sandboxed": {
        "ja": (
            "machine: サンドボックス化されたシェル内から呼ばれています"
            "（~/.config への書き込み不可）。エンジン '{engine}' はEROFSで"
            "失敗するため実行できません。engine=codex はサンドボックス内でも"
            "実行可能です。他のエンジンが必要な場合は、シェルコマンドではなく"
            "ネイティブの machine ツール（ツール呼び出し）を使ってください。"
        ),
        "en": (
            "machine: called from a sandboxed shell (~/.config is not writable). "
            "Engine '{engine}' would fail with EROFS. Use engine=codex, which works "
            "inside the sandbox, or the native machine tool call for other engines."
        ),
    },
    "machine.invalid_engine": {
        "ja": "無効なエンジン '{engine}' です。有効なエンジン: {valid}",
        "en": "Invalid engine '{engine}'. Valid engines: {valid}",
    },
    "machine.list_hint": {
        "ja": "使用するエンジン名をengineパラメータに指定してmachine_runを再度呼び出してください。",
        "en": "Call machine_run again with the desired engine name in the engine parameter.",
    },
    "machine.missing_working_directory": {
        "ja": "working_directory が指定されていません。",
        "en": "working_directory is required.",
    },
    "machine.rate_limit_exceeded": {
        "ja": "工作機械の呼び出し上限に達しました（{limit}回/{period}）。",
        "en": "Machine tool call limit reached ({limit} calls per {period}).",
    },
    "machine.schema.background": {
        "ja": "true: 非同期実行（結果は次回heartbeatで取得）。false: 同期実行（結果を直接返す）",
        "en": ("true: async execution (result at next heartbeat). false: sync execution (result returned directly)"),
    },
    "machine.schema.description_multi": {
        "ja": (
            '外部エージェントCLI（工作機械）にタスクを委託する。推奨エンジン: {top}（他{others}エンジン利用可能。engine="__list__"で一覧取得）\n工作機械は指示されたタスクのみを実行するステートレスな道具であり、Animaの記憶・通信・組織情報にはアクセスできない。\n\n【重要】instruction には以下を必ず含めること:\n- 達成すべきゴールの具体的な記述\n- 対象ファイル・モジュールの明示\n- 制約条件（コーディング規約、既存APIとの整合性等）\n- 期待する出力形式\n曖昧な指示は低品質な結果につながる。職人が工作機械に渡す設計図のように、正確かつ詳細に記述すること。'
        ),
        "en": (
            'Delegate a task to an external agent CLI (machine tool). Recommended engine: {top} ({others} more available; engine="__list__" to list all)\nMachine tools are stateless and execute only the given instruction. They have NO access to Anima memory, messaging, or org info.\n\n[IMPORTANT] instruction MUST include:\n- Clear goal description\n- Target files / modules\n- Constraints (coding conventions, API compatibility, etc.)\n- Expected output format\nVague instructions lead to poor results. Write precise blueprints.'
        ),
    },
    "machine.schema.description_single": {
        "ja": (
            "外部エージェントCLI（工作機械）にタスクを委託する。利用可能エンジン: {top}\n工作機械は指示されたタスクのみを実行するステートレスな道具であり、Animaの記憶・通信・組織情報にはアクセスできない。\n\n【重要】instruction には以下を必ず含めること:\n- 達成すべきゴールの具体的な記述\n- 対象ファイル・モジュールの明示\n- 制約条件（コーディング規約、既存APIとの整合性等）\n- 期待する出力形式\n曖昧な指示は低品質な結果につながる。職人が工作機械に渡す設計図のように、正確かつ詳細に記述すること。"
        ),
        "en": (
            "Delegate a task to an external agent CLI (machine tool). Available engine: {top}\nMachine tools are stateless and execute only the given instruction. They have NO access to Anima memory, messaging, or org info.\n\n[IMPORTANT] instruction MUST include:\n- Clear goal description\n- Target files / modules\n- Constraints (coding conventions, API compatibility, etc.)\n- Expected output format\nVague instructions lead to poor results. Write precise blueprints."
        ),
    },
    "machine.schema.engine_multi": {
        "ja": "使用する工作機械。推奨: {top}（「__list__」で全エンジン一覧取得）",
        "en": 'Machine tool engine. Recommended: {top} (use "__list__" to list all)',
    },
    "machine.schema.engine_single": {
        "ja": "使用する工作機械: {top}",
        "en": "Machine tool engine: {top}",
    },
    "machine.schema.instruction": {
        "ja": "工作機械への詳細な作業指示。ゴール・対象・制約・期待出力を明記する",
        "en": (
            "Detailed task instruction for the machine tool. Specify goal, target, constraints, and expected output"
        ),
    },
    "machine.schema.model": {
        "ja": "使用モデル（省略時はengineのデフォルト）",
        "en": "Model to use (defaults to engine's default if omitted)",
    },
    "machine.schema.timeout": {
        "ja": "タイムアウト秒数。同期時デフォルト600、非同期時デフォルト1800",
        "en": "Timeout in seconds. Default 600 for sync, 1800 for async",
    },
    "machine.schema.working_directory": {
        "ja": "作業ディレクトリの絶対パス。工作機械はこのディレクトリ内でのみ書き込み可能",
        "en": ("Absolute path to the working directory. The machine tool can only write within this directory"),
    },
    "machine.schema.working_directory_with_alias": {
        "ja": "作業ディレクトリ。絶対パスまたはワークスペースエイリアス（例: myproject, myproject#3af4be6e）を指定可能",
        "en": ("Working directory. Accepts absolute path or workspace alias (e.g. myproject, myproject#3af4be6e)"),
    },
    "machine.timeout": {
        "ja": "工作機械 '{engine}' が {seconds} 秒でタイムアウトしました。",
        "en": "Machine engine '{engine}' timed out after {seconds} seconds.",
    },
    "machine.unexpected_error": {
        "ja": "工作機械 '{engine}' の実行中に予期しないエラー: {error}",
        "en": "Unexpected error running machine engine '{engine}': {error}",
    },
    "machine.working_directory_not_found": {
        "ja": "作業ディレクトリ '{path}' が存在しません。",
        "en": "Working directory '{path}' does not exist.",
    },
    "notion.cli_desc": {
        "ja": "Notion CLI（AnimaWorks 連携）",
        "en": "Notion CLI (AnimaWorks integration)",
    },
    "notion.config_error": {
        "ja": (
            "ツール 'notion' には認証情報が必要です。vault.json または shared/credentials.json に NOTION_API_TOKEN を設定してください"
        ),
        "en": ("Tool 'notion' requires credential. Set NOTION_API_TOKEN in vault.json or shared/credentials.json"),
    },
    "notion.database_id_required": {
        "ja": "database_id は必須です",
        "en": "database_id is required",
    },
    "notion.page_id_required": {
        "ja": "page_id は必須です",
        "en": "page_id is required",
    },
    "notion.parent_page_id_required": {
        "ja": "parent_page_id は必須です",
        "en": "parent_page_id is required",
    },
    "notion.parent_required": {
        "ja": "parent_page_id または parent_database_id のいずれかが必須です",
        "en": "Either parent_page_id or parent_database_id is required",
    },
    "notion.payload_too_large": {
        "ja": "ペイロードが {max_bytes} バイトを超えています（実際: {actual_bytes}）",
        "en": "Payload exceeds {max_bytes} bytes (actual: {actual_bytes})",
    },
    "notion.rate_limited": {
        "ja": "Notion API がレート制限されています",
        "en": "Notion API rate limited",
    },
    "notion.server_error": {
        "ja": "Notion API サーバーエラー {status}: {body}",
        "en": "Notion API server error {status}: {body}",
    },
    "notion.unknown_action": {
        "ja": "不明な notion アクション: {name}",
        "en": "Unknown notion action: {name}",
    },
}
