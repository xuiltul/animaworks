from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.


"""SQLite-backed storage for tool descriptions and usage guides.

Tool descriptions (short, for API schemas) and tool guides (long, for
system prompt injection) are stored in a SQLite database so they can be
edited via WebUI without redeploying code.

Database: ``~/.animaworks/tool_prompts.sqlite3`` (WAL mode).
"""

import logging
import sqlite3
from pathlib import Path
from typing import Any

from core.time_utils import now_jst

logger = logging.getLogger(__name__)

# ── Schema SQL ──────────────────────────────────────────────

_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS tool_descriptions (
    name        TEXT PRIMARY KEY,
    description TEXT NOT NULL CHECK(length(description) > 0),
    updated_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tool_guides (
    key         TEXT PRIMARY KEY,
    content     TEXT NOT NULL DEFAULT '',
    updated_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS system_sections (
    key         TEXT PRIMARY KEY,
    content     TEXT NOT NULL CHECK(length(content) > 0),
    condition   TEXT,
    updated_at  TEXT NOT NULL
);
"""

# ── Default descriptions ────────────────────────────────────
#
# Enriched descriptions with "when to use" context.  These are seeded
# on first init and can be edited later via WebUI.
# Each entry is a locale dict (ja/en) for i18n.

DEFAULT_DESCRIPTIONS: dict[str, dict[str, str]] = {
    # -- Memory tools --
    "search_memory": {
        "ja": (
            "長期記憶（knowledge, episodes, procedures）をキーワード検索する。\n"
            "以下の場面で積極的に使うこと:\n"
            "- コマンド実行・設定変更の前に、関連する手順書や過去の教訓を確認する\n"
            "- 報告・判断の前に、関連する既存知識で事実を裏付ける\n"
            "- 未知または曖昧なトピックについて、過去の経験を参照する\n"
            "- Primingの記憶だけでは具体的な手順・数値が不足する場合\n"
            "コンテキスト内で明確に判断できる単純な応答には不要。"
        ),
        "en": (
            "Search long-term memory (knowledge, episodes, procedures) by keyword.\n"
            "Use actively in these situations:\n"
            "- Before executing commands or changing settings, check related procedures and past lessons\n"
            "- Before reporting or making decisions, verify with existing knowledge\n"
            "- When facing unknown or ambiguous topics, reference past experience\n"
            "- When Priming memory alone lacks specific procedures or values\n"
            "Not needed for simple responses that can be clearly determined from context."
        ),
    },
    "read_memory_file": {
        "ja": (
            "自分の記憶ディレクトリ内のファイルを相対パスで読む。"
            "heartbeat.md や cron.md の現在の内容を確認する時、"
            "手順書（procedures/）やスキル（skills/）の詳細を読む時、"
            "Primingで「->」ポインタが示すファイルの具体的内容を確認する時に使う。"
        ),
        "en": (
            "Read a file from your memory directory by relative path. "
            "Use when checking heartbeat.md or cron.md, reading procedure/skill details, "
            "or following Priming -> pointers to file contents."
        ),
    },
    "write_memory_file": {
        "ja": (
            "自分の記憶ディレクトリ内のファイルに書き込みまたは追記する。\n"
            "以下の場面で記録すべき:\n"
            "- 問題を解決した → knowledge/ に原因と解決策を記録\n"
            "- 正しいパラメータ・設定値を発見した → knowledge/ に記録\n"
            "- 作業手順を確立・改善した → procedures/ に手順書を作成\n"
            "- 新しいスキル・テクニックを習得した → skills/ に記録\n"
            "- heartbeat.md や cron.md の更新\n"
            "mode='overwrite' で全体置換、mode='append' で末尾追記。\n"
            "自動統合（日次consolidation）を待たず、重要な発見は即座に書き込むこと。"
        ),
        "en": (
            "Write or append to a file in your memory directory.\n"
            "Record when:\n"
            "- Problem solved → knowledge/ with cause and solution\n"
            "- Correct parameters discovered → knowledge/\n"
            "- Procedure established/improved → procedures/ with new doc\n"
            "- New skill learned → skills/\n"
            "- Updating heartbeat.md or cron.md\n"
            "mode='overwrite' for replace, mode='append' for append.\n"
            "Write important discoveries immediately; do not wait for consolidation."
        ),
    },
    "archive_memory_file": {
        "ja": (
            "不要になった記憶ファイル（knowledge, procedures）をアーカイブする。"
            "ファイルはarchive/ディレクトリに移動され、完全には削除されない。"
            "古くなった知識、重複ファイル、陳腐化した手順の整理に使用する。"
        ),
        "en": (
            "Archive memory files (knowledge, procedures) that are no longer needed. "
            "Files are moved to archive/ directory, not permanently deleted. "
            "Use for cleaning up stale knowledge, duplicates, or outdated procedures."
        ),
    },
    "send_message": {
        "ja": (
            "他のAnimaまたは人間ユーザーにDMを送信する。"
            "人間ユーザーへのメッセージは設定された外部チャネル（Slack等）経由で自動配信される。"
            "intentパラメータで即時処理（delegation/report/question）か"
            "次回heartbeat処理（未指定）かが決まる。"
            "1対1の指示・報告・質問に使う。全体共有にはpost_channelを使う。"
        ),
        "en": (
            "Send a DM to another Anima or human user. "
            "Messages to humans are delivered via configured external channel (e.g. Slack). "
            "intent parameter controls immediate handling (delegation/report/question) vs next heartbeat. "
            "Use for 1:1 instructions, reports, questions. Use post_channel for broadcast."
        ),
    },
    # -- Channel tools --
    "post_channel": {
        "ja": (
            "Boardの共有チャネルにメッセージを投稿する。"
            "チーム全体に共有すべき情報はgeneralチャネルに、"
            "運用・インフラ関連はopsチャネルに投稿する。"
            "全Animaが閲覧できるため、解決済み情報の共有や"
            "お知らせに使うこと。1対1の連絡にはsend_messageを使う。"
        ),
        "en": (
            "Post a message to a Board shared channel. "
            "Use general for team-wide info, ops for infrastructure. "
            "All Animas can read; use for shared solutions and announcements. "
            "Use send_message for 1:1 communication."
        ),
    },
    "read_channel": {
        "ja": (
            "Boardの共有チャネルの直近メッセージを読む。"
            "他のAnimaやユーザーが共有した情報を確認できる。"
            "heartbeat時のチャネル巡回や、特定トピックの共有状況を確認する時に使う。"
            "human_only=trueでユーザー発言のみフィルタリング可能。"
        ),
        "en": (
            "Read recent messages from a Board shared channel. "
            "See what other Animas and users have shared. "
            "Use during heartbeat or to check sharing on a topic. "
            "human_only=true filters to user messages only."
        ),
    },
    "read_dm_history": {
        "ja": (
            "特定の相手との過去のDM履歴を読む。"
            "send_messageで送受信したメッセージの履歴を時系列で確認できる。"
            "以前のやり取りの文脈を確認したいとき、"
            "報告や委任の進捗を追跡したいときに使う。"
        ),
        "en": (
            "Read past DM history with a specific peer. "
            "View send_message history in chronological order. "
            "Use to recall prior context or track report/delegation progress."
        ),
    },
    # -- File tools (Mode A/B) --
    "read_file": {
        "ja": (
            "任意のファイルを絶対パスで読む（permissions.mdの許可範囲内）。"
            "出力は行番号付き（N|content形式）でコードブロックに囲まれる。"
            "大きいファイルはoffset（開始行、1始まり）とlimit（行数）で部分読み取り可能。"
            "自分の記憶ディレクトリ内のファイルにはread_memory_fileを使うこと。"
        ),
        "en": (
            "Read any file by absolute path (within permissions.md scope). "
            "Output is line-numbered (N|content) in a code block. "
            "Use offset (1-based) and limit for partial reads. "
            "Use read_memory_file for files inside your memory directory."
        ),
    },
    "write_file": {
        "ja": (
            "任意のファイルに書き込む（permissions.mdの許可範囲内）。"
            "自分の記憶ディレクトリ外のファイルを書く時に使う。"
            "自分の記憶ディレクトリ内のファイルにはwrite_memory_fileを使うこと。"
        ),
        "en": (
            "Write to any file (within permissions.md scope). "
            "Use for files outside your memory directory. "
            "Use write_memory_file for files inside your memory directory."
        ),
    },
    "edit_file": {
        "ja": (
            "ファイル内の特定の文字列を別の文字列に置換する。"
            "ファイル全体を書き換えずに一部だけ変更したい時に使う。"
            "old_stringが一意に特定できる十分な長さであることを確認すること。"
        ),
        "en": (
            "Replace a specific string in a file with another. "
            "Use when changing only part of a file. "
            "Ensure old_string is long enough to uniquely identify the target."
        ),
    },
    "execute_command": {
        "ja": (
            "シェルコマンドを実行する（permissions.mdの許可リスト内のみ）。"
            "ファイル操作にはread_file/write_file/edit_fileを優先し、"
            "コマンド実行が本当に必要な場合のみ使う。"
        ),
        "en": (
            "Execute a shell command (allow-list in permissions.md only). "
            "Prefer read_file/write_file/edit_file for file ops; "
            "use this only when command execution is truly needed."
        ),
    },
    # -- Search tools (Mode A/B) --
    "search_code": {
        "ja": (
            "正規表現パターンでファイル内のテキストを検索する。"
            "マッチした行をファイルパスと行番号付きで返す。"
            "execute_commandでgrepを使う代わりにこのツールを使うこと。"
        ),
        "en": (
            "Search for text in files using a regex pattern. "
            "Returns matching lines with file path and line numbers. "
            "Use this instead of execute_command with grep."
        ),
    },
    "list_directory": {
        "ja": (
            "指定パスのファイルとディレクトリを一覧表示する。"
            "globパターンでフィルタリング可能。"
            "execute_commandでlsやfindを使う代わりにこのツールを使うこと。"
        ),
        "en": (
            "List files and directories at the given path. "
            "Supports glob patterns for filtering. "
            "Use this instead of execute_command with ls or find."
        ),
    },
    # -- Notification --
    "call_human": {
        "ja": (
            "人間の管理者に連絡する。"
            "重要な報告、問題のエスカレーション、判断が必要な事項がある場合に使用する。"
            "チャット画面と外部通知チャネル（Slack等）の両方に届く。"
            "日常的な報告にはsend_messageを使い、緊急時のみcall_humanを使うこと。"
        ),
        "en": (
            "Contact the human administrator. "
            "Use for important reports, escalation, or decisions requiring human input. "
            "Delivered to chat UI and external channel (e.g. Slack). "
            "Use send_message for routine reports; call_human for urgent cases only."
        ),
    },
    # -- Discovery --
    "discover_tools": {
        "ja": (
            "利用可能な外部ツールカテゴリを確認する。"
            "引数なしで呼ぶとカテゴリ一覧を返す。"
            "カテゴリ名を指定して呼ぶとそのツール群が使えるようになる。"
            "外部サービス（Slack, Chatwork, Gmail等）を使いたい時にまず呼ぶこと。"
        ),
        "en": (
            "Discover available external tool categories. "
            "Call without args for category list; with category name to activate that group. "
            "Call this first when you need external services (Slack, Chatwork, Gmail, etc.)."
        ),
    },
    # -- Tool management --
    "refresh_tools": {
        "ja": (
            "個人・共通ツールディレクトリを再スキャンして新しいツールを発見する。"
            "新しいツールファイルを作成した後に呼んで、"
            "現在のセッションで即座に使えるようにする。"
        ),
        "en": (
            "Re-scan personal and common tool directories to discover new tools. "
            "Call after creating a new tool file to make it available in the current session."
        ),
    },
    "share_tool": {
        "ja": (
            "個人ツールをcommon_tools/にコピーして全Animaで共有する。"
            "自分のtools/ディレクトリにあるツールファイルが"
            "共有のcommon_tools/ディレクトリにコピーされる。"
        ),
        "en": (
            "Copy a personal tool to common_tools/ for all Animas to use. "
            "Copies from your tools/ directory to the shared common_tools/ directory."
        ),
    },
    # -- Admin --
    "create_anima": {
        "ja": (
            "キャラクターシートから新しいDigital Animaを作成する。"
            "character_sheet_contentで直接内容を渡すか、"
            "character_sheet_pathでファイルパスを指定する。"
            "ディレクトリ構造が原子的に作成され、初回起動時にbootstrapで自己設定される。"
        ),
        "en": (
            "Create a new Digital Anima from a character sheet. "
            "Pass content via character_sheet_content or a path via character_sheet_path. "
            "Directory structure is created atomically; bootstrap runs on first startup."
        ),
    },
    # -- Procedure/Knowledge outcome --
    "report_procedure_outcome": {
        "ja": (
            "手順書・スキルの実行結果を報告する。成功/失敗のカウントと信頼度が更新される。\n"
            "手順書（procedures/）やスキル（skills/）に従って作業した後は、必ずこのツールで結果を報告すること。\n"
            "成功時はsuccess=true、失敗・問題発生時はsuccess=falseとnotesに詳細を記録する。\n"
            "信頼度の低い手順は自動的に改善対象としてマークされる。"
        ),
        "en": (
            "Report outcome of following a procedure or skill. Updates success/failure counts and confidence.\n"
            "Always call this after completing work per procedures/ or skills/.\n"
            "Use success=true on success; success=false and notes for failures.\n"
            "Low-confidence procedures are auto-flagged for improvement."
        ),
    },
    "report_knowledge_outcome": {
        "ja": (
            "知識ファイルの有用性を報告する。\n"
            "search_memoryやPrimingで取得した知識を実際に使った後、必ず報告すること:\n"
            "- 知識が正確で役立った → success=true\n"
            "- 不正確・古い・無関係だった → success=false + notesに問題点を記録\n"
            "報告データは能動的忘却と知識品質の維持に使われる。未報告の知識は品質評価できない。"
        ),
        "en": (
            "Report usefulness of a knowledge file.\n"
            "Always report after using knowledge from search_memory or Priming:\n"
            "- Accurate and helpful → success=true\n"
            "- Inaccurate, stale, or irrelevant → success=false + notes with issues\n"
            "Data feeds forgetting and quality. Unreported knowledge cannot be evaluated."
        ),
    },
    # -- Skill tools --
    "skill": {
        "ja": (
            "スキル・共通スキル・手順書の全文を取得する。\n"
            "Primingのスキルヒントに表示された名前を指定して呼ぶ。\n"
            "手順書に従って作業する前に、必ずこのツールで全文を確認すること。"
        ),
        "en": (
            "Get full text of a skill, common skill, or procedure.\n"
            "Specify the name shown in Priming skill hints.\n"
            "Always fetch full content before following a procedure."
        ),
    },
    # -- Task tools --
    "add_task": {
        "ja": (
            "タスクキューに新しいタスクを追加する。"
            "人間からの指示は必ずsource='human'で記録すること。"
            "Anima間の委任はsource='anima'で記録する。"
            "deadlineは必須。相対形式（'30m','2h','1d'）またはISO8601で指定。"
        ),
        "en": (
            "Add a new task to the task queue. "
            "Always record human instructions with source='human'. "
            "Use source='anima' for Anima delegation. "
            "deadline required: relative ('30m','2h','1d') or ISO8601."
        ),
    },
    "update_task": {
        "ja": (
            "タスクのステータスを更新する。"
            "完了時はstatus='done'、中断時はstatus='cancelled'に設定する。"
            "タスク完了後は必ずこのツールでステータスを更新すること。"
        ),
        "en": (
            "Update task status. Use status='done' when complete, status='cancelled' when aborted. "
            "Always update status when a task is finished."
        ),
    },
    "list_tasks": {
        "ja": (
            "タスクキューの一覧を取得する。"
            "ステータスでフィルタリング可能。"
            "heartbeat時の進捗確認やタスク割り当て時に使う。"
        ),
        "en": (
            "List tasks in the task queue. Filter by status. "
            "Use during heartbeat for progress and task assignment."
        ),
    },
}


def _get_locale() -> str:
    """Get locale from config lazily.  Delegates to core.paths."""
    from core.paths import _get_locale as _paths_get_locale

    return _paths_get_locale()


def get_default_description(tool_name: str, locale: str | None = None) -> str:
    """Get default tool description for given locale with fallback."""
    loc = locale or _get_locale()
    entry = DEFAULT_DESCRIPTIONS.get(tool_name, {})
    return entry.get(loc) or entry.get("en") or entry.get("ja", "")


def get_default_guide(key: str, locale: str | None = None) -> str:
    """Get default tool guide for given locale with fallback."""
    loc = locale or _get_locale()
    entry = DEFAULT_GUIDES.get(key, {})
    return entry.get(loc) or entry.get("en") or entry.get("ja", "")

# ── Default guides ──────────────────────────────────────────
#
# s_builtin: S-mode (Claude Code subprocess) guide — empty by default because
# Claude Code includes Read/Write/Edit/Grep/Glob/Bash/git etc. in its system prompt.
# s_mcp / non_s: MCP tools and general tool usage guides with locale support.

DEFAULT_GUIDES: dict[str, dict[str, str]] = {
    "s_builtin": {"ja": "", "en": ""},
    "s_mcp": {
        "ja": """\
## MCPツール（mcp__aw__*）

以下のMCPツールが利用可能です。ファイル操作（Read/Write/Edit）とは別に、AnimaWorks固有の機能を提供します。

### タスク管理
- **mcp__aw__add_task**: タスクキューにタスクを追加。人間からの指示はsource='human'で必ず記録。deadline必須
- **mcp__aw__update_task**: タスクのステータスを更新。完了時はstatus='done'
- **mcp__aw__list_tasks**: タスク一覧取得。heartbeat時の進捗確認に使う

### 記憶の検索と活用
- **mcp__aw__search_memory**: 長期記憶をキーワード検索。以下の場面で積極的に使うこと:
  - コマンド実行・設定変更の前に手順書や過去の教訓を確認
  - 報告・判断の前に既存知識で事実を裏付ける
  - 結果のファイルはReadツールで詳細確認

### 記憶の書き込み（Sモード）
Sモードでは **Writeツール**（ネイティブ）を使って直接記憶ファイルを書き込める。
以下の場面で積極的に記録すること:
- 問題を解決した → `knowledge/` に原因と解決策
- 正しいパラメータ・設定値を発見した → `knowledge/` に記録
- 作業手順を確立した → `procedures/` に手順書作成
  - 第1見出し（`# ...`）は手順の目的が一目でわかる具体的な1行にすること
  - YAMLフロントマターは任意（省略時はシステムが自動付与する。knowledge/proceduresとも対応済み）
- 新スキル習得 → `skills/` に記録
自動統合（日次consolidation）を待たず、重要な発見は即座に書き込むこと。

### 成果追跡
- **mcp__aw__report_procedure_outcome**: 手順書・スキル実行後に必ず結果を報告（成功/失敗の追跡）
- **mcp__aw__report_knowledge_outcome**: search_memoryやPrimingで得た知識の有用性を報告。知識品質の維持に必要

### 人間通知
- **mcp__aw__call_human**: 人間の管理者に通知を送信。重要な報告・エスカレーション用。日常報告にはsend_messageを使う

### スキル・手続きの詳細取得
- **mcp__aw__skill**: スキル（skills/）・共通スキル（common_skills/）・手順書（procedures/）の全文を取得する
  - Primingのスキルヒントに表示された名前を `name` パラメータに指定する
  - 手順書に従って作業する前に、必ずこのツールで全文を確認すること

### 外部ツール（MCP経由）

permissions.md で許可された外部サービス連携ツール（Chatwork, Slack, Gmail等）は
MCPツール（`mcp__aw__*`）として直接利用可能。Bash経由の `animaworks-tool` は不要。

#### 使い方
`mcp__aw__chatwork_send`, `mcp__aw__slack_post` 等、許可されたツールが
MCPツールリストに自動的に含まれる。通常のMCPツールと同じように直接呼び出すこと。

#### 長時間ツールのバックグラウンド実行
画像生成・ローカルLLM推論等の長時間ツールはBash経由で `submit` を使い非同期実行すること:
```
animaworks-tool submit <ツール名> <サブコマンド> [引数...]
```
完了時は `state/background_notifications/` に通知が書かれ、次回heartbeatで確認できる。

#### 注意事項
- MCPツール（`mcp__aw__*`）: 内部機能も外部サービスも全てMCP経由で直接呼び出す
- 使えるツールは `permissions.md` で許可されたもののみ
""",
        "en": """\
## MCP Tools (mcp__aw__*)

The following MCP tools are available. They provide AnimaWorks-specific functionality separate from file operations (Read/Write/Edit).

### Task management
- **mcp__aw__add_task**: Add a task to the queue. Always record human instructions with source='human'. deadline required
- **mcp__aw__update_task**: Update task status. Use status='done' when complete
- **mcp__aw__list_tasks**: List tasks. Use during heartbeat for progress tracking

### Memory search and use
- **mcp__aw__search_memory**: Search long-term memory by keyword. Use actively when:
  - Checking procedures and past lessons before commands or config changes
  - Verifying facts with existing knowledge before reports or decisions
  - Use Read tool to view result file details

### Memory writing (S mode)
In S mode use the native **Write tool** to write memory files directly. Record when:
- Problem solved → `knowledge/` with cause and solution
- Correct parameters discovered → `knowledge/`
- Procedure established → `procedures/` with a new doc
  - First heading (`# ...`) should clearly state the procedure purpose in one line
  - YAML frontmatter is optional (system auto-adds it for both knowledge/ and procedures/)
- New skill learned → `skills/`
Write important discoveries immediately; do not wait for consolidation.

### Outcome tracking
- **mcp__aw__report_procedure_outcome**: Report results after procedures/skills (success/failure tracking)
- **mcp__aw__report_knowledge_outcome**: Report usefulness of knowledge from search_memory or Priming. Required for quality

### Human notification
- **mcp__aw__call_human**: Send notification to human admin. Use for reports and escalation. Use send_message for routine reports

### Skill and procedure details
- **mcp__aw__skill**: Get full text of skills (skills/), common skills (common_skills/), procedures (procedures/)
  - Specify the name shown in Priming skill hints as the `name` parameter
  - Always fetch full content before following a procedure

### External tools (via MCP)

External service tools (Chatwork, Slack, Gmail, etc.) permitted in permissions.md
are available as MCP tools (`mcp__aw__*`) directly. No need for Bash `animaworks-tool`.

#### Usage
Tools like `mcp__aw__chatwork_send`, `mcp__aw__slack_post` are automatically
included in the MCP tool list. Call them directly like any other MCP tool.

#### Background execution for long-running tools
Image generation, local LLM inference, etc. hold the lock if run directly.
Use `submit` for async execution via Bash:
```
animaworks-tool submit <tool_name> <subcommand> [args...]
```
On completion, notifications go to `state/background_notifications/` for the next heartbeat.

#### Notes
- MCP tools (`mcp__aw__*`): Both internal and external tools are called directly via MCP
- Allowed tools are those permitted in `permissions.md`
""",
    },
    "non_s": {
        "ja": """\
## ツールの使い方

### 記憶について

あなたのコンテキストには「あなたが思い出していること」セクションが含まれています。
これは、相手の顔を見た瞬間に名前や過去のやり取りを自然と思い出すのと同じです。

#### 応答の判断基準
- コンテキスト内の記憶で十分に判断できる場合: そのまま応答してよい
- コンテキスト内の記憶では不足する場合: search_memory / read_memory_file で追加検索せよ

※ 上記は記憶検索についての判断基準である。システムプロンプト内の行動指示
 （チーム構成の提案など）への対応は、記憶の十分性とは独立して行うこと。

#### 追加検索が必要な典型例
- 具体的な日時・数値を正確に答える必要がある時
- 過去の特定のやり取りの詳細を確認したい時
- 手順書（procedures/）に従って作業する時
- コンテキストに該当する記憶がない未知のトピックの時
- Priming に `->` ポインタがある場合、具体的なパスやコマンドを回答する必要があるとき

#### 禁止事項
- 記憶の検索プロセスについてユーザーに言及すること（人間は「今から思い出します」とは言わない）
- 毎回機械的に記憶検索を実行すること（コンテキストで判断できることに追加検索は不要）

### 記憶の書き込み

#### 自動記録（あなたは何もしなくてよい）
- 会話の内容はシステムが自動的にエピソード記憶（episodes/）に記録する
- あなたが意識的にエピソード記録を書く必要はない
- 日次・週次でシステムが自動的にエピソードから教訓やパターンを抽出し、知識記憶（knowledge/）に統合する

#### 意図的な記録（あなたが判断して行う）
以下の場面では write_memory_file で積極的に記録すること:
- 問題を解決したとき → knowledge/ に原因・調査過程・解決策を記録
- 正しいパラメータ・設定値を発見したとき → knowledge/ に記録
- 重要な方針・判断基準を確立したとき → knowledge/ に記録
- 作業手順を確立・改善したとき → procedures/ に手順書を作成
  - 第1見出し（`# ...`）は手順の目的が一目でわかる具体的な1行にすること
  - YAMLフロントマターは任意（省略時はシステムが自動付与する。knowledge/proceduresとも対応済み）
- 新しいスキル・テクニックを習得したとき → skills/ に記録
自動統合（日次consolidation）を待たず、重要な発見は即座に書き込むこと。

**記憶の書き込みについては報告不要**

#### 成果追跡
手順書やスキルに従って作業した後は、report_procedure_outcome で必ず結果を報告すること。
search_memoryやPrimingで取得した知識を使った後は、report_knowledge_outcome で有用性を報告すること。

### スキル・手続きの詳細取得

Primingのスキルヒントに表示された名前は、`skill` ツールで全文を取得できる:
```
skill(name="スキル名またはファイル名")
```
- skills/、common_skills/、procedures/ の全文を返す
- 手順書に従って作業する前に、必ず全文を確認すること
- ヒントに `->` ポインタがある場合、具体的な手順を取得するために使う

#### ユーザー記憶の更新
ユーザーについて新しい情報を得たら shared/users/{ユーザー名}/index.md の該当セクションを更新し、log.md の先頭に追記する
- index.md のセクション構造（基本情報/重要な好み・傾向/注意事項）は固定。新セクション追加禁止
- log.md フォーマット: `## YYYY-MM-DD {自分の名前}: {要約1行}` + 本文数行
- log.md が20件を超えたら末尾の古いエントリを削除する
- ユーザーのディレクトリが未作成の場合は mkdir して index.md / log.md を新規作成する

### 業務指示の内在化

あなたには2つの定期実行メカニズムがある:

- **Heartbeat（定期巡回）**: 30分固定間隔でシステムが起動。heartbeat.md のチェックリストを実行する
- **Cron（定時タスク）**: cron.md で指定した時刻に実行

業務指示を受けた場合の振り分け:
- 「常に確認して」「チェックして」→ **heartbeat.md** にチェックリスト項目を追加
- 「毎朝○○して」「毎週金曜に○○して」→ **cron.md** に定時タスクを追加

#### Heartbeat への追加手順
1. read_memory_file(path="heartbeat.md") で現在のチェックリストを確認する
2. チェックリストセクションに新しい項目を追加する
   - write_memory_file(path="heartbeat.md", content="...", mode="overwrite") で更新
   - ⚠「## 活動時間」「## 通知ルール」セクションは変更しないこと

#### Cron への追加手順
1. read_memory_file(path="cron.md") で現在のタスク一覧を確認する
2. 新しいタスクを追加する（type: llm or type: command を指定）
3. write_memory_file(path="cron.md", content="...", mode="overwrite") で保存

いずれの場合も:
- 具体的な手順が伴う場合は procedures/ にも手順書を作成する
- 更新完了を指示者に報告する
""",
        "en": """\
## How to Use Tools

### About memory

Your context includes a "What you recall" section. It works like recalling a face and past interactions naturally.

#### Response criteria
- If context memory is sufficient: respond directly
- If context memory is insufficient: use search_memory / read_memory_file for additional search

Note: This applies to memory search. Follow system prompt action guidance (e.g. team structure proposals) independently.

#### When additional search is needed
- When accurate dates, times, or numbers are required
- When checking past interaction details
- When following procedures in procedures/
- For unknown topics with no matching context memory
- When Priming has `->` pointers and you need specific paths/commands

#### Prohibited
- Mentioning the memory search process to the user (humans don't say "Let me recall")
- Mechanical memory search every time (no need when context suffices)

### Memory writing

#### Automatic (nothing for you to do)
- Conversation content is auto-recorded to episodes/
- No need to write episodes manually
- System auto-extracts lessons and patterns daily/weekly into knowledge/

#### Intentional (your decision)
Use write_memory_file when:
- Problem solved → knowledge/ with cause, investigation, solution
- Correct parameters discovered → knowledge/
- Important policy or criteria established → knowledge/
- Procedure established/improved → procedures/ with new doc
  - First heading (`# ...`) should state purpose clearly in one line
  - YAML frontmatter optional (system auto-adds it for both knowledge/ and procedures/)
- New skill learned → skills/
Write immediately; do not wait for consolidation.

**No need to report memory writes**

#### Outcome tracking
After following procedures or skills, always report via report_procedure_outcome.
After using knowledge from search_memory or Priming, report via report_knowledge_outcome.

### Skill and procedure details

Names shown in Priming skill hints can be fetched in full via the `skill` tool:
```
skill(name="skill_name_or_file")
```
- Returns full text from skills/, common_skills/, procedures/
- Always fetch full content before following a procedure
- Use for specific steps when hints include `->` pointers

#### Updating user memory
When you learn new user info, update shared/users/{username}/index.md and prepend to log.md
- index.md section structure (basic info/preferences/notes) is fixed. No new sections
- log.md format: `## YYYY-MM-DD {your_name}: {one-line summary}` + body
- Trim log.md when entries exceed 20
- Create mkdir + index.md / log.md if user dir doesn't exist

### Internalising work instructions

You have two scheduled mechanisms:

- **Heartbeat**: Runs every 30 minutes. Execute the checklist in heartbeat.md
- **Cron**: Runs at times specified in cron.md

When receiving work instructions:
- "Always check" / "monitor" → add checklist items to **heartbeat.md**
- "Every morning" / "Every Friday" → add scheduled tasks to **cron.md**

#### Adding to Heartbeat
1. read_memory_file(path="heartbeat.md") to see current checklist
2. Add new item to checklist section
   - write_memory_file(path="heartbeat.md", content="...", mode="overwrite")
   - Do not change "## 活動時間" or "## 通知ルール" sections

#### Adding to Cron
1. read_memory_file(path="cron.md") to see current tasks
2. Add new task (specify type: llm or type: command)
3. write_memory_file(path="cron.md", content="...", mode="overwrite")

In both cases:
- Create procedures/ doc when specific steps are involved
- Report completion to the requester
""",
    },
}

# ── Section conditions ─────────────────────────────────────
#
# Metadata for system_sections: key -> condition string (or None).
# Actual content is loaded from runtime prompts at seed time.

SECTION_CONDITIONS: dict[str, str | None] = {
    "behavior_rules": None,
    "environment": None,
    "messaging_s": "mode:s",
    "messaging": "mode:non_s",
    "communication_rules_s": "mode:s",
    "communication_rules": "mode:non_s",
    "emotion_instruction": None,
    "a_reflection": "mode:a",
    "hiring_context": "solo_top_level",
}

# ── ToolPromptStore ─────────────────────────────────────────


class ToolPromptStore:
    """SQLite-backed storage for tool descriptions and guides.

    Follows the same WAL pattern as ``core.tools._cache.BaseMessageCache``.
    Each read opens a fresh connection to ensure WebUI edits are picked up
    immediately.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        """Open a new connection with WAL mode and dict row factory."""
        conn = sqlite3.connect(str(self._db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        """Create tables if they don't exist."""
        with self._connect() as conn:
            conn.executescript(_SCHEMA_SQL)

    # ── Descriptions CRUD ───────────────────────────────────

    def get_description(self, name: str) -> str | None:
        """Return the description for *name*, or ``None`` if not found."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT description FROM tool_descriptions WHERE name = ?",
                (name,),
            ).fetchone()
        return row["description"] if row else None

    def set_description(self, name: str, description: str) -> dict[str, Any]:
        """Insert or update a tool description.  Returns the saved record."""
        ts = now_jst().isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO tool_descriptions (name, description, updated_at) "
                "VALUES (?, ?, ?) "
                "ON CONFLICT(name) DO UPDATE SET description=excluded.description, "
                "updated_at=excluded.updated_at",
                (name, description, ts),
            )
        return {"name": name, "description": description, "updated_at": ts}

    def list_descriptions(self) -> list[dict[str, Any]]:
        """Return all tool descriptions as dicts."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT name, description, updated_at FROM tool_descriptions "
                "ORDER BY name",
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Guides CRUD ─────────────────────────────────────────

    def get_guide(self, key: str) -> str | None:
        """Return the guide content for *key*, or ``None`` if not found or empty.

        Empty string stored in DB acts as "disabled" — returns None so callers
        fall through to DEFAULT_GUIDES or skip injection entirely.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT content FROM tool_guides WHERE key = ?",
                (key,),
            ).fetchone()
        if not row:
            return None
        return row["content"] or None

    def set_guide(self, key: str, content: str) -> dict[str, Any]:
        """Insert or update a tool guide.  Returns the saved record."""
        ts = now_jst().isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO tool_guides (key, content, updated_at) "
                "VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET content=excluded.content, "
                "updated_at=excluded.updated_at",
                (key, content, ts),
            )
        return {"key": key, "content": content, "updated_at": ts}

    def list_guides(self) -> list[dict[str, Any]]:
        """Return all tool guides as dicts."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT key, content, updated_at FROM tool_guides "
                "ORDER BY key",
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Sections CRUD ────────────────────────────────────────

    def get_section(self, key: str) -> str | None:
        """Return the section content for *key*, or ``None`` if not found."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT content FROM system_sections WHERE key = ?",
                (key,),
            ).fetchone()
        return row["content"] if row else None

    def get_section_with_condition(
        self, key: str
    ) -> tuple[str, str | None] | None:
        """Return ``(content, condition)`` for *key*, or ``None``."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT content, condition FROM system_sections WHERE key = ?",
                (key,),
            ).fetchone()
        return (row["content"], row["condition"]) if row else None

    def set_section(
        self,
        key: str,
        content: str,
        condition: str | None = None,
    ) -> dict[str, Any]:
        """Insert or update a system section.  Returns the saved record."""
        ts = now_jst().isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO system_sections (key, content, condition, updated_at) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET content=excluded.content, "
                "condition=excluded.condition, updated_at=excluded.updated_at",
                (key, content, condition, ts),
            )
        return {
            "key": key,
            "content": content,
            "condition": condition,
            "updated_at": ts,
        }

    def list_sections(self) -> list[dict[str, Any]]:
        """Return all system sections as dicts."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT key, content, condition, updated_at "
                "FROM system_sections ORDER BY key",
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Seeding ─────────────────────────────────────────────

    def seed_defaults(
        self,
        descriptions: dict[str, str | dict[str, str]] | None = None,
        guides: dict[str, str | dict[str, str]] | None = None,
        sections: dict[str, tuple[str, str | None]] | None = None,
    ) -> None:
        """Seed default descriptions, guides, and sections.

        Descriptions and guides may be locale dicts (ja/en) or plain strings.
        Uses INSERT OR IGNORE so existing user edits are preserved.
        """
        loc = _get_locale()
        ts = now_jst().isoformat()
        with self._connect() as conn:
            if descriptions:
                flat = {}
                for k, v in descriptions.items():
                    if isinstance(v, dict):
                        flat[k] = v.get(loc) or v.get("en") or v.get("ja", "")
                    else:
                        flat[k] = v
                conn.executemany(
                    "INSERT OR IGNORE INTO tool_descriptions "
                    "(name, description, updated_at) VALUES (?, ?, ?)",
                    [(k, v, ts) for k, v in flat.items()],
                )
            if guides:
                flat_g = {}
                for k, v in guides.items():
                    if isinstance(v, dict):
                        flat_g[k] = v.get(loc) or v.get("en") or v.get("ja", "")
                    else:
                        flat_g[k] = v
                conn.executemany(
                    "INSERT OR IGNORE INTO tool_guides "
                    "(key, content, updated_at) VALUES (?, ?, ?)",
                    [(k, v, ts) for k, v in flat_g.items()],
                )
            if sections:
                conn.executemany(
                    "INSERT OR IGNORE INTO system_sections "
                    "(key, content, condition, updated_at) VALUES (?, ?, ?, ?)",
                    [
                        (k, content, cond, ts)
                        for k, (content, cond) in sections.items()
                    ],
                )


# ── Singleton accessor ──────────────────────────────────────

_store: ToolPromptStore | None = None
_store_initialised: bool = False


def get_prompt_store() -> ToolPromptStore | None:
    """Return the singleton ToolPromptStore, or ``None`` if DB unavailable.

    The DB path is ``{data_dir}/tool_prompts.sqlite3``.  If the file
    does not exist a warning is logged on first call.
    """
    global _store, _store_initialised

    if _store_initialised:
        return _store

    _store_initialised = True

    try:
        from core.paths import get_data_dir

        db_path = get_data_dir() / "tool_prompts.sqlite3"
        if not db_path.parent.exists():
            logger.warning(
                "Data directory does not exist: %s — "
                "tool prompt DB unavailable. Run 'animaworks init'.",
                db_path.parent,
            )
            return None
        _store = ToolPromptStore(db_path)
        return _store
    except Exception:
        logger.warning("Failed to initialise ToolPromptStore", exc_info=True)
        return None


def reset_prompt_store() -> None:
    """Reset the singleton (for testing)."""
    global _store, _store_initialised
    _store = None
    _store_initialised = False
