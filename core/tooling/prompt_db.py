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
    content     TEXT NOT NULL CHECK(length(content) > 0),
    updated_at  TEXT NOT NULL
);
"""

# ── Default descriptions ────────────────────────────────────
#
# Enriched descriptions with "when to use" context.  These are seeded
# on first init and can be edited later via WebUI.

DEFAULT_DESCRIPTIONS: dict[str, str] = {
    # -- Memory tools --
    "search_memory": (
        "長期記憶（knowledge, episodes, procedures）をキーワード検索する。"
        "コンテキスト内の記憶では不足する場合、過去の特定のやり取りの詳細を確認したい時、"
        "手順書に従って作業する時、未知のトピックについて調べる時に使う。"
        "コンテキストで判断できることに対しては不要。"
    ),
    "read_memory_file": (
        "自分の記憶ディレクトリ内のファイルを相対パスで読む。"
        "heartbeat.md や cron.md の現在の内容を確認する時、"
        "手順書（procedures/）やスキル（skills/）の詳細を読む時、"
        "Primingで「->」ポインタが示すファイルの具体的内容を確認する時に使う。"
    ),
    "write_memory_file": (
        "自分の記憶ディレクトリ内のファイルに書き込みまたは追記する。"
        "重要な方針・教訓を即座に記録したい時は knowledge/ に、"
        "作業手順をまとめたい時は procedures/ に、"
        "新しいスキルを習得した時は skills/ に書き込む。"
        "heartbeat.md や cron.md の更新にも使う。"
        "mode='overwrite' で全体置換、mode='append' で末尾追記。"
    ),
    "send_message": (
        "他のAnimaまたは人間ユーザーにDMを送信する。"
        "人間ユーザーへのメッセージは設定された外部チャネル（Slack等）経由で自動配信される。"
        "intentパラメータで即時処理（delegation/report/question）か"
        "次回heartbeat処理（未指定）かが決まる。"
        "1対1の指示・報告・質問に使う。全体共有にはpost_channelを使う。"
    ),
    # -- Channel tools --
    "post_channel": (
        "Boardの共有チャネルにメッセージを投稿する。"
        "チーム全体に共有すべき情報はgeneralチャネルに、"
        "運用・インフラ関連はopsチャネルに投稿する。"
        "全Animaが閲覧できるため、解決済み情報の共有や"
        "お知らせに使うこと。1対1の連絡にはsend_messageを使う。"
    ),
    "read_channel": (
        "Boardの共有チャネルの直近メッセージを読む。"
        "他のAnimaやユーザーが共有した情報を確認できる。"
        "heartbeat時のチャネル巡回や、特定トピックの共有状況を確認する時に使う。"
        "human_only=trueでユーザー発言のみフィルタリング可能。"
    ),
    "read_dm_history": (
        "特定の相手との過去のDM履歴を読む。"
        "send_messageで送受信したメッセージの履歴を時系列で確認できる。"
        "以前のやり取りの文脈を確認したいとき、"
        "報告や委任の進捗を追跡したいときに使う。"
    ),
    # -- File tools (A2/B modes) --
    "read_file": (
        "任意のファイルを絶対パスで読む（permissions.mdの許可範囲内）。"
        "自分の記憶ディレクトリ外のファイルを読む時に使う。"
        "自分の記憶ディレクトリ内のファイルにはread_memory_fileを使うこと。"
    ),
    "write_file": (
        "任意のファイルに書き込む（permissions.mdの許可範囲内）。"
        "自分の記憶ディレクトリ外のファイルを書く時に使う。"
        "自分の記憶ディレクトリ内のファイルにはwrite_memory_fileを使うこと。"
    ),
    "edit_file": (
        "ファイル内の特定の文字列を別の文字列に置換する。"
        "ファイル全体を書き換えずに一部だけ変更したい時に使う。"
        "old_stringが一意に特定できる十分な長さであることを確認すること。"
    ),
    "execute_command": (
        "シェルコマンドを実行する（permissions.mdの許可リスト内のみ）。"
        "ファイル操作にはread_file/write_file/edit_fileを優先し、"
        "コマンド実行が本当に必要な場合のみ使う。"
    ),
    # -- Search tools (A2/B modes) --
    "search_code": (
        "正規表現パターンでファイル内のテキストを検索する。"
        "マッチした行をファイルパスと行番号付きで返す。"
        "execute_commandでgrepを使う代わりにこのツールを使うこと。"
    ),
    "list_directory": (
        "指定パスのファイルとディレクトリを一覧表示する。"
        "globパターンでフィルタリング可能。"
        "execute_commandでlsやfindを使う代わりにこのツールを使うこと。"
    ),
    # -- Notification --
    "call_human": (
        "人間の管理者に連絡する。"
        "重要な報告、問題のエスカレーション、判断が必要な事項がある場合に使用する。"
        "チャット画面と外部通知チャネル（Slack等）の両方に届く。"
        "日常的な報告にはsend_messageを使い、緊急時のみcall_humanを使うこと。"
    ),
    # -- Discovery --
    "discover_tools": (
        "利用可能な外部ツールカテゴリを確認する。"
        "引数なしで呼ぶとカテゴリ一覧を返す。"
        "カテゴリ名を指定して呼ぶとそのツール群が使えるようになる。"
        "外部サービス（Slack, Chatwork, Gmail等）を使いたい時にまず呼ぶこと。"
    ),
    # -- Tool management --
    "refresh_tools": (
        "個人・共通ツールディレクトリを再スキャンして新しいツールを発見する。"
        "新しいツールファイルを作成した後に呼んで、"
        "現在のセッションで即座に使えるようにする。"
    ),
    "share_tool": (
        "個人ツールをcommon_tools/にコピーして全Animaで共有する。"
        "自分のtools/ディレクトリにあるツールファイルが"
        "共有のcommon_tools/ディレクトリにコピーされる。"
    ),
    # -- Admin --
    "create_anima": (
        "キャラクターシートから新しいDigital Animaを作成する。"
        "character_sheet_contentで直接内容を渡すか、"
        "character_sheet_pathでファイルパスを指定する。"
        "ディレクトリ構造が原子的に作成され、初回起動時にbootstrapで自己設定される。"
    ),
    # -- Procedure/Knowledge outcome --
    "report_procedure_outcome": (
        "手順書・スキルの実行結果を報告する。"
        "成功/失敗のカウントと信頼度が更新される。"
        "手順書に従って作業を完了した後に呼んで、その手順の信頼性を追跡する。"
    ),
    "report_knowledge_outcome": (
        "知識ファイルの有用性を報告する。"
        "知識が役に立った場合はsuccess=true、"
        "不正確・無関係だった場合はsuccess=falseで呼ぶ。"
        "知識の品質を追跡し、能動的忘却の判断材料になる。"
    ),
    # -- Task tools --
    "add_task": (
        "タスクキューに新しいタスクを追加する。"
        "人間からの指示は必ずsource='human'で記録すること。"
        "Anima間の委任はsource='anima'で記録する。"
        "deadlineは必須。相対形式（'30m','2h','1d'）またはISO8601で指定。"
    ),
    "update_task": (
        "タスクのステータスを更新する。"
        "完了時はstatus='done'、中断時はstatus='cancelled'に設定する。"
        "タスク完了後は必ずこのツールでステータスを更新すること。"
    ),
    "list_tasks": (
        "タスクキューの一覧を取得する。"
        "ステータスでフィルタリング可能。"
        "heartbeat時の進捗確認やタスク割り当て時に使う。"
    ),
}

# ── Default guides ──────────────────────────────────────────

DEFAULT_GUIDES: dict[str, str] = {
    "a1_builtin": """\
## ビルトインツールの使い方（A1モード）

あなたは以下のビルトインツールを使用できます。これらはファイル操作、検索、コマンド実行のための基本ツールです。

### Read — ファイル読み取り

ファイルの内容を読み取ります。

**いつ使うか:**
- 自分の記憶ディレクトリ内のファイルを確認する時（heartbeat.md, cron.md, knowledge/, procedures/ 等）
- Primingで「->」ポインタが示すファイルの具体的内容を確認する時
- shared/users/{ユーザー名}/index.md でユーザー情報を確認する時
- 共通スキルや共通知識の詳細を読む時

**使い方:**
- パスは絶対パスで指定する（例: 自分の記憶ディレクトリ内の `knowledge/deploy-notes.md` 等）
- 自分のディレクトリ、shared/、common_skills/、common_knowledge/ が読み取り可能

### Write — ファイル書き込み

ファイルの内容を作成または上書きします。

**いつ使うか:**
- 重要な方針・教訓を即座に記録したい時 → knowledge/ に書き込み
- 作業手順をまとめたい時 → procedures/ に書き込み
- 新しいスキルを習得した時 → skills/ に書き込み
- heartbeat.md や cron.md を更新する時
- shared/users/{ユーザー名}/index.md や log.md を更新する時

**注意:**
- 既存ファイルを更新する場合は、まずReadで現在の内容を確認してから書き込むこと
- エピソード記憶（episodes/）は自動記録されるため、手動での書き込みは不要

### Edit — ファイル部分編集

ファイル内の特定の文字列を別の文字列に置換します。

**いつ使うか:**
- heartbeat.md のチェックリストに項目を追加・変更する時
- cron.md に新しい定時タスクを追加する時
- knowledge/ の既存ファイルの一部を更新する時
- ファイル全体を書き換えずに一部だけ変更したい時

**使い方:**
- old_string: 置換対象の文字列（ファイル内で一意に特定できる長さにすること）
- new_string: 置換後の文字列

### Bash — コマンド実行

シェルコマンドを実行します。

**いつ使うか:**
- permissions.md で許可されたコマンドを実行する時
- 外部ツール（animaworks-tool）を実行する時

**注意:**
- ファイル操作にはRead/Write/Editを優先し、cat/head/tail/sed/awk は使わない
- ファイル検索にはGlob/Grepを使い、find/grep コマンドは使わない

### Grep — テキスト検索

ファイル内のテキストを正規表現パターンで検索します。

**いつ使うか:**
- 記憶ディレクトリ内で特定のキーワードを含むファイルを探す時
- 共通知識や手順書内の関連情報を検索する時

### Glob — ファイル検索

ファイルパターンに一致するファイルを検索します。

**いつ使うか:**
- ディレクトリ内のファイル一覧を確認する時
- 特定の拡張子やパターンのファイルを探す時

### 記憶操作の基本パターン

#### 記憶の検索
コンテキスト内の記憶で不足する場合のみ、追加検索を行うこと:
1. `mcp__aw__search_memory` でキーワード検索
2. 結果のファイルパスを `Read` で詳細確認

#### 記憶の書き込み
- knowledge/: 重要な方針・教訓 → `Write` で作成
- procedures/: 作業手順 → `Write` で作成（第1見出しは目的が一目でわかる具体的な1行）
- skills/: 新しいスキル → `Write` で作成

#### Heartbeat/Cron の更新
1. `Read` で現在の heartbeat.md / cron.md を確認
2. `Edit` で必要な項目を追加・変更
   - heartbeat.md: 「## 活動時間」「## 通知ルール」セクションは変更しない
   - cron.md: type: llm / type: command を正しく指定

#### ユーザー記憶の更新
1. `Read` で shared/users/{ユーザー名}/index.md を確認
2. `Edit` または `Write` で該当セクションを更新
3. shared/users/{ユーザー名}/log.md の先頭に追記（`## YYYY-MM-DD {自分の名前}: {要約1行}`）
""",
    "a1_mcp": """\
## MCPツール（mcp__aw__*）

以下のMCPツールが利用可能です。ファイル操作（Read/Write/Edit）とは別に、AnimaWorks固有の機能を提供します。

### タスク管理
- **mcp__aw__add_task**: タスクキューにタスクを追加。人間からの指示はsource='human'で必ず記録。deadline必須
- **mcp__aw__update_task**: タスクのステータスを更新。完了時はstatus='done'
- **mcp__aw__list_tasks**: タスク一覧取得。heartbeat時の進捗確認に使う

### 記憶検索
- **mcp__aw__search_memory**: 長期記憶をキーワード検索。コンテキスト内で不足する場合のみ使う。結果のファイルはReadツールで詳細確認

### 人間通知
- **mcp__aw__call_human**: 人間の管理者に通知を送信。重要な報告・エスカレーション用。日常報告にはsend_messageを使う

### 成果追跡
- **mcp__aw__report_procedure_outcome**: 手順書の実行結果を記録（成功/失敗と信頼度の追跡）
- **mcp__aw__report_knowledge_outcome**: 知識ファイルの有用性を記録（品質追跡と能動的忘却の判断材料）

### ツール発見
- **mcp__aw__discover_tools**: 利用可能な外部ツールカテゴリを確認。外部サービスを使いたい時にまず呼ぶ
""",
    "non_a1": """\
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
以下の場合のみ、write_memory_file で直接書き込むこと:
- 重要な方針・教訓を即座に記録したい時 → knowledge/ に書き込み
- 作業手順をまとめたい時 → procedures/ に書き込み
  - 第1見出し（`# ...`）は手順の目的が一目でわかる具体的な1行にすること
  - YAMLフロントマターは不要（システムが自動付与する）
- 新しいスキルを習得した時 → skills/ に書き込み
- これは「メモを取る」行為であり、記録義務ではない

**記憶の書き込みについては報告不要**

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
        """Return the guide content for *key*, or ``None`` if not found."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT content FROM tool_guides WHERE key = ?",
                (key,),
            ).fetchone()
        return row["content"] if row else None

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

    # ── Seeding ─────────────────────────────────────────────

    def seed_defaults(
        self,
        descriptions: dict[str, str] | None = None,
        guides: dict[str, str] | None = None,
    ) -> None:
        """Seed default descriptions and guides.

        Uses INSERT OR IGNORE so existing user edits are preserved.
        """
        ts = now_jst().isoformat()
        with self._connect() as conn:
            if descriptions:
                conn.executemany(
                    "INSERT OR IGNORE INTO tool_descriptions "
                    "(name, description, updated_at) VALUES (?, ?, ?)",
                    [(k, v, ts) for k, v in descriptions.items()],
                )
            if guides:
                conn.executemany(
                    "INSERT OR IGNORE INTO tool_guides "
                    "(key, content, updated_at) VALUES (?, ?, ?)",
                    [(k, v, ts) for k, v in guides.items()],
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
