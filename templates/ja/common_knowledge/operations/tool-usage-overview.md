# ツール使用の概要

AnimaWorks のツール体系と、実行モード別の使い方のリファレンス。
自分が利用可能なツールの全体像を把握し、適切な方法で呼び出すためのガイド。

## 実行モードとツールの関係

Anima の実行モードによって、ツールの呼び出し方と利用可能範囲が異なる。
自分のモードは `status.json` のモデル名から自動で決定される。

| モード | 対象モデル | ツール呼び出し方式 |
|--------|-----------|------------------|
| S (SDK) | `claude-*` | MCP ツール (`mcp__aw__*`) + Claude Code 組込み + Bash 経由外部ツール |
| A (Autonomous) | `openai/*`, `google/*`, `vertex_ai/*` 等 | LiteLLM function calling（ツール名そのまま） |
| B (Basic) | `ollama/*` 等の小型モデル | テキスト JSON 形式 |

### S-mode（Claude Agent SDK）

3系統のツールが利用可能:

1. **Claude Code 組込みツール**: Read, Write, Edit, Grep, Glob, Bash, git 等。ファイル操作・コマンド実行に使う
2. **MCP ツール (`mcp__aw__*`)**: AnimaWorks 固有の内部機能。`mcp__aw__send_message` のように `mcp__aw__` 接頭辞付きで呼ぶ
3. **外部ツール (`animaworks-tool`)**: Slack, Gmail, GitHub 等の外部サービス連携。Bash から `animaworks-tool` コマンドで実行する

### A-mode（LiteLLM）

2系統のツールが利用可能:

1. **内部ツール**: `send_message`, `search_memory`, `add_task` 等。ツール名をそのまま function calling で呼ぶ
2. **外部ツール**: `discover_tools(category="...")` を呼ぶとカテゴリのツールが動的に追加される

### B-mode（Basic）

限定されたツールセットのみ利用可能:

- 記憶系（search_memory, read_memory_file, write_memory_file）
- 通信系（send_message, post_channel, read_channel）
- スキル系（skill）
- 外部ツール・スーパーバイザーツール・タスク管理ツールは利用不可

---

## ツールカテゴリ

### 内部ツール（常時利用可能）

全モードで利用可能な AnimaWorks 内部機能。

| カテゴリ | ツール | 概要 |
|---------|--------|------|
| 記憶 | `search_memory` | 長期記憶のキーワード検索 |
| 記憶 | `read_memory_file` | 記憶ファイルの読み取り |
| 記憶 | `write_memory_file` | 記憶ファイルへの書き込み |
| 通信 | `send_message` | DM 送信 |
| 通信 | `post_channel` | Board チャネルへの投稿 |
| 通信 | `read_channel` | Board チャネルの読み取り |
| 通信 | `read_dm_history` | DM 履歴の参照 |
| スキル | `skill` | スキル・手順書の全文取得 |
| 成果追跡 | `report_procedure_outcome` | 手順書実行結果の報告 |
| 成果追跡 | `report_knowledge_outcome` | 知識の有用性報告 |
| 通知 | `call_human` | 人間の管理者への通知 |

S-mode では上記ツール名に `mcp__aw__` 接頭辞が付く（例: `mcp__aw__send_message`）。

### タスク管理ツール（A/S-mode のみ）

| ツール | 概要 |
|--------|------|
| `add_task` | タスクキューにタスクを追加 |
| `update_task` | タスクのステータスを更新 |
| `list_tasks` | タスク一覧取得 |

### スーパーバイザーツール（部下を持つ Anima のみ、A-mode）

部下を持つ Anima に自動で有効化される組織管理ツール。詳細は `organization/hierarchy-rules.md` を参照。

| ツール | 概要 |
|--------|------|
| `org_dashboard` | 配下全体のプロセス状態をツリー表示 |
| `ping_subordinate` | 配下の生存確認 |
| `read_subordinate_state` | 配下の現在タスク読み取り |
| `delegate_task` | 部下へのタスク委譲 |
| `task_tracker` | 委譲タスクの進捗追跡 |
| `disable_subordinate` | 部下の休止 |
| `enable_subordinate` | 部下の再開 |
| `set_subordinate_model` | 部下のモデル変更 |
| `restart_subordinate` | 部下プロセスの再起動 |

S-mode では `disable_subordinate` と `enable_subordinate` のみ MCP 経由で利用可能。

### 外部ツール（権限・有効化が必要）

外部サービスと連携するツール。`permissions.md` の `tool_categories` で許可されたカテゴリのみ利用可能。

主なカテゴリ: `slack`, `chatwork`, `gmail`, `github`, `aws_collector`, `web_search`, `x_search`, `image_gen`, `local_llm`, `transcribe`

---

## 外部ツールの使い方

### S-mode の場合

1. **カテゴリ確認**: `mcp__aw__discover_tools` を引数なしで呼ぶ
2. **詳細確認**: カテゴリ名を指定して呼ぶ（例: `mcp__aw__discover_tools(category="slack")`）
3. **実行**: Bash で `animaworks-tool <ツール名> <サブコマンド> [引数...]`
4. **ヘルプ**: `animaworks-tool <ツール名> --help` でサブコマンド一覧と引数を確認

長時間ツール（画像生成等）は `submit` で非同期実行する:
```bash
animaworks-tool submit <ツール名> <サブコマンド> [引数...]
```

詳細は `operations/background-tasks.md` を参照。

### A-mode の場合

1. **カテゴリ確認**: `discover_tools()` を引数なしで呼ぶ
2. **有効化**: `discover_tools(category="slack")` でカテゴリを有効化
3. **実行**: 有効化されたツールは通常の function calling で呼べるようになる

### B-mode の場合

外部ツールは通常利用不可。必要な場合は上司に依頼する。

---

## よくある疑問

### common_knowledge のツール呼び出し例が自分のモードと違う

common_knowledge のドキュメントでは `send_message(to="...", content="...")` のように A/B-mode 形式で記載されている。
S-mode の場合は `mcp__aw__` 接頭辞を付けて読み替えること（例: `mcp__aw__send_message`）。

### 利用可能なツールを確認したい

- **全 Anima 共通**: `check_permissions` ツールで自分の権限を確認できる
- **S-mode**: `mcp__aw__discover_tools` で外部ツールのカテゴリを確認
- **A-mode**: `discover_tools()` で外部ツールのカテゴリを確認
- **権限の詳細**: `read_memory_file(path="permissions.md")` で確認

### ツールがエラーになる

→ `troubleshooting/common-issues.md` の「ツールが使えない」セクションを参照
