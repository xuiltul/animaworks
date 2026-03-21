---
name: subagent-cli
description: >-
  外部AIエージェントCLI(codex exec, cursor-agent -p)をBash経由でサブエージェントとして
  非対話モードで実行するスキル。複雑なコーディングタスク・コードレビュー・
  マルチファイル変更を委譲する際の実行手順・オプション・出力処理を提供する。
  Mode S/C/D/G/A/BでBash権限がある場合に適用。Mode C/D/G（codex/* / cursor/* / gemini/*）の場合はフレームワークが
  各エンジンを直接実行するため、対応するCLIの手動呼び出しは原則不要。
  「サブエージェント」「codex」「cursor-agent」「コード書いて」「実装して」
  「コードレビュー」「リファクタリング」
---

# subagent-cli

外部AIエージェントCLIをBash経由でサブプロセスとして実行し、複雑なコーディングタスクを委譲する。
自分のアイデンティティ・判断・記憶は維持したまま、実行能力を拡張するための「パワーツール」として使う。

## フレームワーク実行モードとの関係

このスキルは **Bash ツールが利用可能な場合** にのみ適用される。

| モード | 実装 | Bash | このスキルの適用 |
|--------|------|------|------------------|
| **Mode S** | `agent_sdk.py` (Claude Agent SDK) | デフォルトで利用可能 | 適用される。Claude Code サブプロセス内で Read/Write/Edit/Bash/Grep/Glob/WebFetch/WebSearch + MCP(send_message 等) + Task/Agent が利用可能。Bash 実行時の cwd は anima_dir |
| **Mode C** | `codex_sdk.py` (Codex SDK) | Codex CLI のツールセットに依存 | **codex exec は不要** — フレームワークが Codex を直接実行。cursor-agent / claude -p は Bash 経由で呼べる（Bash が利用可能な場合） |
| **Mode D** | Cursor Agent（cursor-agent サブプロセス） | Cursor CLI のツールセットに依存 | **cursor-agent -p は不要** — フレームワークが cursor-agent を直接実行。MCP 統合。Mode S に近いツールアクセスだが実体は cursor-agent バイナリ。codex exec / claude -p は Bash 経由で呼べる（Bash が利用可能な場合） |
| **Mode G** | Gemini CLI（gemini サブプロセス） | Gemini CLI のツールセットに依存 | **Gemini CLI の手動起動は不要** — フレームワークが直接実行。MCP 統合、stream-json 出力。他 CLI は Bash 経由で呼べる（Bash が利用可能な場合） |
| **Mode A/B** | LiteLLM + tool_use / 1ショット | permissions.json で許可時のみ | Bash 許可があれば適用 |

**重要**: Mode C（`codex/*`）、Mode D（`cursor/*`）、Mode G（`gemini/*`）の Anima は、フレームワークが各エンジンを直接実行する。この場合、自分で `codex exec`（Mode C）や `cursor-agent -p`（Mode D）、Gemini CLI（Mode G）を Bash から呼ぶ必要はない。別の CLI（cursor-agent / claude -p / codex exec 等）を明示的に使いたい場合のみ、このスキルの該当セクションを参照する。

## ツール選択の優先順位

**コスト効率順に選択すること。**

| 優先度 | ツール | コスト | 得意領域 |
|--------|--------|--------|----------|
| 1 | `codex exec` | 最安（Codex） | コード生成・編集・レビュー |
| 2 | `cursor-agent -p` | 安い（Cursor） | コード生成・編集・マルチファイル |
| 3 | `claude -p` | 高い（Claude API） | 最終手段。上2つで解決しない場合のみ |

**原則: codex exec を最初に試す。失敗時や不得意なタスクのみ cursor-agent → claude の順にフォールバック。**

## 使うべきタイミング

- マルチファイルにまたがるコード変更
- テスト作成・テスト修正
- コードレビュー
- リファクタリング
- バグ修正の調査と実装
- 新機能の実装

## 使うべきでないタイミング

- 1ファイルの小さな編集（自分で直接やる）
- 記憶の読み書き（自分のツールを使う）
- 外部API呼び出し（専用ツールを使う）
- 情報の検索・調査のみ（web_search や Read で十分）

---

## 1. codex exec（推奨）

**適用条件**: Mode S または Mode A/B（Bash 許可）のとき。Mode C の場合はフレームワークが Codex を実行するため、このセクションは不要。Mode D/G の場合もフレームワークが各エンジンを実行するため、代替として codex を使う必要がなければ参照不要。

### 基本構文

```bash
codex exec --full-auto -C /path/to/workspace "プロンプト"
```

作業ディレクトリ `-C` には対象プロジェクトの絶対パスを指定する。Mode S の Bash 実行時には `ANIMAWORKS_ANIMA_DIR`（Anima のデータディレクトリ）と `ANIMAWORKS_PROJECT_DIR`（AnimaWorks フレームワークのルート）が環境変数として設定される。AnimaWorks 自体の開発が対象の場合は `-C "$ANIMAWORKS_PROJECT_DIR"` が使える。

### 重要オプション

| オプション | 説明 |
|-----------|------|
| `--full-auto` | 自動承認＋サンドボックス（workspace-write） |
| `-C /path` | 作業ディレクトリ指定（**必須**） |
| `-m model` | モデル指定（例: `o4-mini`, `o3`） |
| `--sandbox workspace-write` | ワークスペース書き込み許可（full-autoに含まれる） |
| `--json` | JSONL形式で出力 |
| `-o file` | 最終メッセージをファイルに書き出し |
| `--ephemeral` | セッションファイルを保存しない |

### 実行例

#### コード生成

```bash
codex exec --full-auto --ephemeral -C /home/main/dev/myproject \
  "src/utils/parser.py にMarkdownパーサーを実装して。既存のテストを壊さないこと。"
```

#### コードレビュー

```bash
codex exec --full-auto --ephemeral -C /home/main/dev/myproject \
  review
```

#### テスト作成

```bash
codex exec --full-auto --ephemeral -C /home/main/dev/myproject \
  "src/utils/parser.py のユニットテストを tests/test_parser.py に作成して。"
```

#### 結果をファイルに保存

```bash
codex exec --full-auto --ephemeral -C /home/main/dev/myproject \
  -o /tmp/codex_result.txt \
  "このプロジェクトのアーキテクチャを分析して改善案を出して。"
```

---

## 2. cursor-agent -p（代替）

**適用条件**: Mode S または Mode A/B（Bash 許可）。Mode C/G でも Bash が利用可能な場合は適用可能。Mode D ではフレームワークが cursor-agent を実行するため、このセクションの手動 `cursor-agent -p` は原則不要。

### 基本構文

```bash
cursor-agent -p --trust --force --workspace /path/to/workspace "プロンプト"
```

### 重要オプション

| オプション | 説明 |
|-----------|------|
| `-p` / `--print` | 非対話モード（**必須**） |
| `--trust` | ワークスペースを自動信頼 |
| `--force` | コマンド自動承認 |
| `--workspace /path` | 作業ディレクトリ指定（**必須**） |
| `--model model` | モデル指定（例: `sonnet-4`, `gpt-5`） |
| `--output-format text\|json` | 出力形式 |
| `--mode plan\|ask` | 読み取り専用モード（調査向け） |

### 実行例

#### コード生成

```bash
cursor-agent -p --trust --force \
  --workspace /home/main/dev/myproject \
  "src/api/routes.py にPOST /users エンドポイントを追加して。バリデーション付き。"
```

#### 読み取り専用調査

```bash
cursor-agent -p --trust --mode ask \
  --workspace /home/main/dev/myproject \
  "この認証フローにセキュリティ上の問題はある？"
```

#### 結果をファイルに保存

```bash
cursor-agent -p --trust --force \
  --workspace /home/main/dev/myproject \
  --output-format text \
  "テストカバレッジが低いモジュールを特定して改善して" > /tmp/cursor_result.txt
```

---

## 3. claude -p（フォールバック）

**適用条件**: Mode S または Mode A/B（Bash 許可）。Mode C/D/G でも Bash が利用可能な場合は適用可能。

codex/cursor-agentで対応できないときのみ使用。APIコストが高い。

### 基本構文

```bash
claude -p --dangerously-skip-permissions --output-format text "プロンプト"
```

### 重要オプション

| オプション | 説明 |
|-----------|------|
| `-p` / `--print` | 非対話モード（**必須**） |
| `--dangerously-skip-permissions` | 権限チェック省略 |
| `--model model` | モデル指定（例: `sonnet`, `haiku`） |
| `--allowedTools "tools"` | 許可ツール制限（例: `"Read Edit Bash(git:*)"`) |
| `--output-format text\|json` | 出力形式 |
| `--max-budget-usd N` | コスト上限（ドル）|
| `--no-session-persistence` | セッション保存しない |

### 実行例

```bash
claude -p --dangerously-skip-permissions --no-session-persistence \
  --model haiku --max-budget-usd 0.5 \
  --output-format text \
  "src/core/parser.py のエラーハンドリングを改善して"
```

---

## プロンプトの書き方

サブエージェントにはAnimaWorksの文脈がない。明確で自己完結したプロンプトを書くこと。

### 良いプロンプト

```
以下の要件でPythonモジュールを実装して:

ファイル: src/utils/validator.py

要件:
- Pydantic v2のBaseModelを使ったバリデータ
- email, username, passwordフィールド
- パスワードは8文字以上、英数字混合
- バリデーションエラー時にカスタム例外を投げる

制約:
- from __future__ import annotations を先頭に
- Google-style docstring
- 既存のテストを壊さないこと
```

### 悪いプロンプト

```
いい感じにバリデーションを直して
```

→ コンテキストがなく、「いい感じ」が不明確。

---

## 出力の処理

### 標準出力をキャプチャ

```bash
RESULT=$(codex exec --full-auto --ephemeral -C /path "プロンプト" 2>/dev/null)
echo "$RESULT"
```

### ファイル経由（codex推奨）

```bash
codex exec --full-auto --ephemeral -C /path \
  -o /tmp/result.txt "プロンプト"
# 結果を読む
cat /tmp/result.txt
```

### 終了コードで成否判定

```bash
codex exec --full-auto --ephemeral -C /path "プロンプト"
if [ $? -eq 0 ]; then
  echo "成功"
else
  echo "失敗 — cursor-agentにフォールバック"
  cursor-agent -p --trust --force --workspace /path "同じプロンプト"
fi
```

---

## バックグラウンド実行（重要）

サブエージェントの実行は **5分〜20分以上** かかることがある。
フォアグラウンドで待つとセッションがブロックされるため、**必ずバックグラウンドで実行する**こと。

### 基本パターン: nohup + 結果ファイル

```bash
nohup codex exec --full-auto --ephemeral -C /path/to/workspace \
  -o /tmp/codex_result.txt \
  "プロンプト" > /tmp/codex_stdout.log 2>&1 &
echo "PID: $!"
```

cursor-agent の場合:

```bash
nohup cursor-agent -p --trust --force \
  --workspace /path/to/workspace \
  "プロンプト" > /tmp/cursor_result.txt 2>&1 &
echo "PID: $!"
```

### 完了確認

```bash
# プロセスがまだ動いているか確認
ps -p <PID> > /dev/null 2>&1 && echo "実行中" || echo "完了"

# 結果を読む（完了後）
cat /tmp/codex_result.txt
# または
cat /tmp/cursor_result.txt
```

### タイムアウト付き実行

暴走を防ぐために `timeout` を併用する:

```bash
nohup timeout 30m codex exec --full-auto --ephemeral -C /path \
  -o /tmp/codex_result.txt \
  "プロンプト" > /tmp/codex_stdout.log 2>&1 &
```

- 推奨タイムアウト: **30分**（`30m`）
- 小さなタスク: **10分**（`10m`）
- 大きなリファクタリング: **60分**（`60m`）

### 実行中に他の作業を継続

バックグラウンド実行後、完了を待たずに他のタスクを進めてよい。
定期的にプロセスの生存を確認し、完了したら結果を読み取ってepisodes/に記録する。

---

## 安全ガイドライン

1. **作業ディレクトリを必ず指定する** — 未指定だとカレントディレクトリで実行される
2. **機密情報をプロンプトに含めない** — APIキー、パスワード等
3. **codexは `--full-auto` でサンドボックス内実行** — ワークスペース外への書き込みは制限される
4. **実行後にgit diffで変更を確認する** — 意図しない変更がないかチェック
5. **--ephemeral を付ける** — セッションファイルが不要に蓄積されるのを防ぐ

---

## フォールバック戦略

```
1. codex exec で試行
   ↓ 失敗 or 品質不足
2. cursor-agent -p で再試行
   ↓ 失敗 or 品質不足
3. claude -p（--max-budget-usd でコスト制限）で最終試行
   ↓ それでも失敗
4. 自分で実行を試みるか、上司に報告する
```

## 注意事項

- サブエージェントはAnimaWorksの記憶・ツールにアクセスできない。あくまで「コーディングの手」
- 実行結果は自分のepisodes/に記録し、学んだパターンはknowledge/に蓄積すること
- 実行には5分〜20分以上かかる。必ずバックグラウンドで実行し、timeout を設定すること
- git管理されたリポジトリで作業すること（変更の追跡・取り消しが容易）
- Mode S では Bash 実行時に `ANIMAWORKS_ANIMA_DIR`（Anima のデータディレクトリ）と `ANIMAWORKS_PROJECT_DIR`（AnimaWorks フレームワークのルート）が環境変数として設定される（`agent_sdk.py` の `_build_env()` で注入）
