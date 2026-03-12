# ツール使用の概要

AnimaWorks のツール体系と、実行モード別の使い方のリファレンス。
自分が利用可能なツールの全体像を把握し、適切な方法で呼び出すためのガイド。

## 実行モードとツールの関係

Anima の実行モードによって、ツールの呼び出し方と利用可能範囲が異なる。
自分のモードは `status.json` のモデル名から自動で決定される。

| モード | 対象モデル | ツール呼び出し方式 |
|--------|-----------|------------------|
| S (SDK) | `claude-*` | MCP ツール (`mcp__aw__*`) + Claude Code 組込み + Bash 経由外部ツール |
| C (Codex) | `codex/*` | Codex CLI 経由。S-mode と同等のツール体系 |
| A (Autonomous) | `openai/*`, `google/*`, `vertex_ai/*` 等 | LiteLLM function calling（ツール名そのまま） |
| B (Basic) | `ollama/*` 等の小型モデル | テキスト JSON 形式（`{"tool": "名前", "arguments": {...}}`） |

### S-mode（Claude Agent SDK）

3系統のツールが利用可能:

1. **Claude Code 組込みツール**: Read, Write, Edit, Grep, Glob, Bash, git 等。ファイル操作・コマンド実行に使う
2. **MCP ツール (`mcp__aw__*`)**: AnimaWorks 固有の内部機能
   - 記憶・通信: `send_message`, `post_channel`, `read_channel`, `manage_channel`, `read_dm_history`
   - タスク: `backlog_task`, `update_task`, `list_tasks`, `submit_tasks`
   - 通知・権限: `call_human`, `search_memory`, `check_permissions`
   - 成果追跡: `report_procedure_outcome`, `report_knowledge_outcome`
   - スキル: `skill`, `create_skill`
   - スーパーバイザー: `disable_subordinate`, `enable_subordinate`, `set_subordinate_model`, `set_subordinate_background_model`, `restart_subordinate`, `org_dashboard`, `ping_subordinate`, `read_subordinate_state`, `delegate_task`, `task_tracker`, `audit_subordinate`
   - バックグラウンド: `check_background_task`, `list_background_tasks`
   - 認証情報: `vault_get`, `vault_store`, `vault_list`
3. **Bash + animaworks-tool**: 外部ツール（chatwork, slack, gmail, web_search 等）は `skill` でCLI使用法を確認し、`animaworks-tool <ツール名> <サブコマンド>` で実行する。長時間ツールは `animaworks-tool submit` で非同期実行

### C-mode（Codex CLI）

S-mode と同等のツール体系。Codex CLI 経由で実行。未インストール時は LiteLLM（Mode A）にフォールバック。

### A-mode（LiteLLM）

2系統のツールが利用可能:

1. **内部ツール**: `send_message`, `search_memory`, `read_file`, `execute_command`, `backlog_task`, `submit_tasks` 等。ツール名をそのまま function calling で呼ぶ。`refresh_tools`, `share_tool` で個人・共通ツールを再スキャン可能
2. **外部ツール**: `skill` ツールでCLI使用法を確認し、`execute_command` で `animaworks-tool <ツール> <サブコマンド>` を実行

### B-mode（Basic）

テキスト JSON 形式でツールを呼び出す。利用可能なツール:

- **記憶系**: search_memory, read_memory_file, write_memory_file, archive_memory_file
- **通信系**: send_message, post_channel, read_channel, read_dm_history
- **ファイル・検索系**: read_file, write_file, edit_file, execute_command, web_fetch, search_code, list_directory
- **スキル系**: skill, create_skill
- **成果追跡**: report_procedure_outcome, report_knowledge_outcome
- **タスク系**: backlog_task, update_task, list_tasks, submit_tasks
- **バックグラウンド**: check_background_task, list_background_tasks（BackgroundTaskManager 設定時）
- **認証情報**: vault_get, vault_store, vault_list
- **通知**: call_human（通知チャネル設定時）
- **外部ツール**: permissions.md で許可されたカテゴリは `use_tool` で構造化呼び出し可能

refresh_tools, share_tool, create_anima は利用不可（A/S-mode のみ）。

---

## ツールカテゴリ

### 内部ツール（常時利用可能）

全モードで利用可能な AnimaWorks 内部機能（モードにより一部省略あり）。

| カテゴリ | ツール | 概要 |
|---------|--------|------|
| 記憶 | `search_memory` | 長期記憶のキーワード検索 |
| 記憶 | `read_memory_file` | 記憶ファイルの読み取り |
| 記憶 | `write_memory_file` | 記憶ファイルへの書き込み |
| 記憶 | `archive_memory_file` | 不要な記憶ファイルを archive/ へ退避 |
| 通信 | `send_message` | DM 送信（intent 必須: report / question。タスク委譲は `delegate_task`） |
| 通信 | `post_channel` | Board チャネルへの投稿（channel, text） |
| 通信 | `read_channel` | Board チャネルの読み取り |
| 通信 | `manage_channel` | チャネル作成・メンバー追加/削除・情報取得 |
| 通信 | `read_dm_history` | DM 履歴の参照 |
| スキル | `skill` | スキル・手順書の全文取得 |
| スキル | `create_skill` | スキルをディレクトリ構造で作成（A/B-mode） |
| 成果追跡 | `report_procedure_outcome` | 手順書実行結果の報告 |
| 成果追跡 | `report_knowledge_outcome` | 知識の有用性報告 |
| 通知 | `call_human` | 人間の管理者への通知 |
| 権限確認 | `check_permissions` | 自分の権限一覧を確認（A/B-mode） |
| 認証情報 | `vault_get`, `vault_store`, `vault_list` | 認証情報の取得・保存・一覧（A/B-mode） |

S-mode では上記のうち MCP 経由で公開されているものに `mcp__aw__` 接頭辞が付く。

### ファイル・検索ツール（A/B-mode）

| ツール | 概要 |
|--------|------|
| `read_file` | ファイルを行番号付きで読み取り |
| `write_file` | ファイルへ書き込み |
| `edit_file` | ファイル内の文字列を置換 |
| `execute_command` | シェルコマンド実行（permissions.md の許可リストに従う） |
| `web_fetch` | URL からコンテンツを取得 |
| `search_code` | 正規表現でファイル内検索 |
| `list_directory` | ディレクトリ一覧・glob フィルタ |

S-mode では Claude Code の Read / Write / Edit / Grep / Glob / Bash で同等の操作を行う。

### タスク管理ツール（A/S-mode）

| ツール | 概要 |
|--------|------|
| `backlog_task` | タスクキューにタスクを追加 |
| `update_task` | タスクのステータスを更新 |
| `list_tasks` | タスク一覧取得 |
| `submit_tasks` | 複数タスクをDAG投入し並列/直列実行 |

B-mode でも backlog_task, update_task, list_tasks, submit_tasks は利用可能。

### バックグラウンドタスクツール（A/B/S-mode、BackgroundTaskManager 設定時）

| ツール | 概要 |
|--------|------|
| `check_background_task` | 指定 task_id の実行状態を確認 |
| `list_background_tasks` | 実行中・完了・失敗・待機中のタスク一覧 |

`animaworks-tool submit` で投入した長時間ツールの状態確認に使用。

### ツール管理ツール（A-mode のみ）

| ツール | 概要 |
|--------|------|
| `refresh_tools` | 個人・共通ツールを再スキャン |
| `share_tool` | 個人ツールを common_tools/ へ共有 |

### スーパーバイザーツール（部下を持つ Anima のみ）

部下を持つ Anima に自動で有効化される組織管理ツール。詳細は `organization/hierarchy-rules.md` を参照。

| ツール | 概要 | S-mode MCP |
|--------|------|------------|
| `disable_subordinate` | 部下の休止 | ○ |
| `enable_subordinate` | 部下の再開 | ○ |
| `set_subordinate_model` | 部下のモデル変更 | ○ |
| `set_subordinate_background_model` | 部下のバックグラウンドモデル変更 | ○ |
| `restart_subordinate` | 部下プロセスの再起動 | ○ |
| `delegate_task` | 部下へのタスク委譲 | ○ |
| `org_dashboard` | 配下全体のプロセス状態をツリー表示 | ○ |
| `ping_subordinate` | 配下の生存確認 | ○ |
| `read_subordinate_state` | 配下の現在タスク読み取り | ○ |
| `task_tracker` | 委譲タスクの進捗追跡 | ○ |
| `audit_subordinate` | 配下の活動タイムラインまたは統計サマリーを生成。`name` 省略で全配下一括監査 | ○ |

全スーパーバイザーツールが S-mode MCP 経由でも利用可能。CLI からも `animaworks anima audit {名前} [--all] [--days N] [--mode report|summary]` で監査を実行できる。

### 管理ツール（条件付き）

| ツール | 概要 | 有効条件 |
|--------|------|----------|
| `create_anima` | キャラクターシートから新規 Anima 作成 | skills/newstaff.md 保持時（A-mode） |

### 外部ツール（権限・有効化が必要）

外部サービスと連携するツール。`permissions.md` の「外部ツール」セクションで許可されたカテゴリのみ利用可能。

主なカテゴリ: `slack`, `chatwork`, `gmail`, `github`, `aws_collector`, `web_search`, `x_search`, `image_gen`, `local_llm`, `transcribe`, `google_calendar`, `notion`

---

## 外部ツールの使い方

### S-mode の場合

1. **スキル確認**: `skill` ツールでCLI使用法を確認（例: `skill("chatwork-tool")`）
2. **実行**: `animaworks-tool <ツール名> <サブコマンド> [引数...]` でBash実行（例: `animaworks-tool chatwork send 123 "メッセージ"`）
3. **長時間ツール**: `animaworks-tool submit <ツール名> <サブコマンド> [引数...]` で非同期実行。タスクは `state/background_tasks/pending/` に投入され、完了時は `state/background_notifications/` に通知が書かれ次回heartbeatで確認できる
4. **ヘルプ**: `animaworks-tool <ツール名> --help` でサブコマンド一覧と引数を確認

長時間ツール（画像生成、ローカルLLM、音声文字起こし等）は必ず `submit` で実行すること。詳細は `operations/background-tasks.md` を参照。

### A-mode の場合

1. **スキル確認**: `skill("slack-tool")` でCLI使用法を取得
2. **実行**: `execute_command("animaworks-tool slack post ...")` でCLI実行
3. **長時間ツール**: 長時間ツールは `animaworks-tool submit` で投入し、`check_background_task` / `list_background_tasks` で状態確認

### B-mode の場合

1. **権限確認**: `check_permissions` で許可されたカテゴリを確認
2. **実行**: 許可された外部ツールは `use_tool(tool_name="...", action="...", args={...})` で構造化呼び出し
3. **長時間ツール**: 上司（A/S-mode）に依頼するか、権限があれば同様に `use_tool` で呼び出し

---

## よくある疑問

### common_knowledge のツール呼び出し例が自分のモードと違う

common_knowledge のドキュメントでは `send_message(to="...", content="...", intent="...")` のように A/B-mode 形式で記載されている。
S-mode の場合は `mcp__aw__` 接頭辞を付けて読み替えること（例: `mcp__aw__send_message`）。

### 利用可能なツールを確認したい

- **A/B-mode**: `check_permissions` ツールで自分の権限を確認できる
- **S-mode**: MCP で公開されているツールは `mcp__aw__` 接頭辞付き。権限の詳細は `permissions.md` を読む
- **A-mode**: `skill` ツールで外部ツールのCLI使用法を確認
- **全モード**: `read_memory_file(path="permissions.md")` で許可内容を確認可能（S-mode は Claude Code の Read で直接読む）

### ツールがエラーになる

→ `troubleshooting/common-issues.md` の「ツールが使えない」セクションを参照
