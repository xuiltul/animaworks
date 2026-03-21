# モデル選択・設定ガイド

AnimaWorks のモデル設定に関する包括ガイド。
実行モード、対応モデル、設定方法、コンテキストウィンドウの仕組みを解説する。

---

## 実行モード

AnimaWorks はモデル名から実行モードを自動判定する。6種類の実行モードがある:

| モード | 名称 | 概要 | 対象モデル例 |
|--------|------|------|-------------|
| **S** | SDK | Claude Agent SDK経由。最も高機能 | `claude-opus-4-6`, `claude-sonnet-4-6` |
| **C** | Codex | Codex CLI経由 | `codex/o4-mini`, `codex/gpt-4.1` |
| **D** | Cursor Agent | Cursor Agent CLI（`cursor-agent`）経由。MCP 統合 | `cursor/*` |
| **G** | Gemini CLI | Gemini CLI 経由。MCP 統合 | `gemini/*` |
| **A** | Autonomous | LiteLLM + tool_useループ | `openai/gpt-4.1`, `google/gemini-2.5-pro`, `ollama/qwen3:14b` |
| **B** | Basic | 1ショット実行。フレームワークが記憶I/Oを代行 | `ollama/gemma3:4b`, `ollama/deepseek-r1*` |

### モード判定の優先順位

1. Per-anima `status.json` の `execution_mode` 明示指定
2. `~/.animaworks/models.json`（ユーザー編集可）
3. `config.json` `model_modes`（非推奨）
4. コードデフォルトのパターンマッチ
5. 不明 → Mode B（安全側）

---

## 対応モデル一覧

`animaworks models list` で最新の一覧を表示できる。主要なモデル:

### Claude / Anthropic（Mode S）

| モデル | 説明 |
|--------|------|
| `claude-opus-4-6` | 最高性能・推奨 |
| `claude-sonnet-4-6` | バランス型・推奨 |
| `claude-haiku-4-5-20251001` | 軽量・高速 |

### OpenAI（Mode A）

| モデル | 説明 |
|--------|------|
| `openai/gpt-4.1` | 最新・コーディング強 |
| `openai/gpt-4.1-mini` | 高速・低コスト |
| `openai/o3-2025-04-16` | 推論特化 |

### Google Gemini（Mode A）

| モデル | 説明 |
|--------|------|
| `google/gemini-2.5-pro` | 最高性能 |
| `google/gemini-2.5-flash` | 高速バランス |

### Azure OpenAI（Mode A）

| モデル | 説明 |
|--------|------|
| `azure/gpt-4.1-mini` | Azure OpenAI |
| `azure/gpt-4.1` | Azure OpenAI |

### Vertex AI（Mode A）

| モデル | 説明 |
|--------|------|
| `vertex_ai/gemini-2.5-flash` | Vertex AI Flash |
| `vertex_ai/gemini-2.5-pro` | Vertex AI Pro |

### ローカルモデル / vLLM / Ollama

| モデル | モード | 説明 |
|--------|--------|------|
| `openai/qwen3.5-35b-a3b` | A | **推奨** — Sonnet同等性能（ベンチマーク検証済み） |
| `ollama/qwen3:14b` | A | 中型・tool_use対応 |
| `ollama/glm-4.7` | A | tool_use対応 |
| `ollama/gemma3:4b` | B | 軽量 |

### AWS Bedrock

| モデル | モード | 説明 |
|--------|--------|------|
| `openai/zai.glm-4.7` | A | Bedrock Mantle経由。単発タスク向き |
| `bedrock/qwen.qwen3-next-80b-a3b` | A | ツールコール能力が不十分（非推奨） |

---

## 推奨OSSモデル（ベンチマーク検証済み）

### Qwen3.5-35B — ローカルGPU推奨モデル

`openai/qwen3.5-35b-a3b`（vLLM経由）は、AnimaWorks Mode Aエージェントとしてベンチマーク検証済みの**推奨ローカルモデル**。
Claude Sonnet 4.6 と同等の総合スコアを記録し、**background_model として最適**。

#### ベンチマークデータ（2026-03-11 実施）

測定条件: Mode A（LiteLLM tool_useループ）統一、15タスク×3ラン/モデル

| モデル | T1 基本操作 | T2 マルチステップ | T3 判断・エラー | 総合 | 平均時間 | コスト |
|--------|:----------:|:----------------:|:--------------:|:----:|:-------:|:-----:|
| **Qwen3.5-35B (local)** | **100%** | **100%** | 60% | **88%** | 9.6s | **$0** |
| Claude Sonnet 4.6 | 100% | 100% | 60% | 88% | 8.5s | ~$0.015/task |
| GLM-4.7 (Bedrock) | 87% | 33% | 53% | 55% | 5.9s | ~$0.003/task |
| Qwen3-Next 80B (Bedrock) | 40% | 27% | 40% | 35% | 5.2s | ~$0.005/task |

#### 特筆事項

- **T1（基本操作: ファイルI/O、ツールコール）**: Sonnetと完全一致で100%
- **T2（マルチステップ: CSV集計、JSON解析→書き込み等）**: Sonnetと完全一致で100%
- **計算精度（T3-3）**: Qwen3.5が3/3、Sonnetが1/3でQwen3.5が上回る
- **プロンプトインジェクション耐性（T3-4）**: 全モデル0/3（フレームワークレベル対策が必要）
- パラメータ数は性能に直結しない（80BのQwen3-Nextより35BのQwen3.5が大幅に上）

#### 推奨設定

```bash
# vLLM credential設定
# config.json > credentials に追加:
# "vllm-local": { "api_key": "dummy", "base_url": "http://<vllm-host>:8000/v1" }

# models.json に追加
# "openai/qwen3.5*": { "mode": "A", "context_window": 64000 }

# background_model として設定（Chat=Sonnet, HB/Inbox/Cron=Qwen3.5）
animaworks anima set-background-model {名前} openai/qwen3.5-35b-a3b --credential vllm-local
```

#### vLLM起動例

```bash
vllm serve Qwen/Qwen3.5-35B-A3B \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.95 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

#### 用途別モデル選択ガイド

| 用途 | 推奨モデル | 理由 |
|------|-----------|------|
| background_model（HB/Inbox/Cron） | **Qwen3.5-35B** | コスト$0でSonnet同等の安定性 |
| foreground（人間Chat） | Sonnet 4.6 | エラー処理の安定性と日本語品質 |
| TaskExec（委譲タスク実行） | Qwen3.5-35B | コスト$0でツール連鎖が安定 |
| 軽量単純応答（分類・要約） | GLM-4.7 | 最速だがマルチステップ不可 |

---

## models.json

`~/.animaworks/models.json` でモデルごとの実行モードとコンテキストウィンドウを定義する。
fnmatch ワイルドカードパターンが使用可能。

### スキーマ

```json
{
  "パターン": {
    "mode": "S" | "C" | "D" | "G" | "A" | "B",
    "context_window": トークン数
  }
}
```

### 例

```json
{
  "claude-opus-4-6":    { "mode": "S", "context_window": 1000000 },
  "claude-sonnet-4-6":  { "mode": "S", "context_window": 1000000 },
  "claude-*":           { "mode": "S", "context_window": 200000 },
  "openai/gpt-4.1*":   { "mode": "A", "context_window": 1000000 },
  "openai/*":           { "mode": "A", "context_window": 128000 },
  "ollama/gemma3*":     { "mode": "B", "context_window": 8192 }
}
```

具体的なパターンが優先。`claude-opus-4-6` は `claude-*` より先にマッチする。

### 確認コマンド

```bash
animaworks models show       # models.json の内容表示
animaworks models info {モデル名}  # 解決結果の確認
```

---

## モデル変更手順

### 特定Animaのモデルを変更

```bash
# 1. モデル設定（status.json を更新）
animaworks anima set-model {名前} {モデル名}

# 2. credential が必要な場合
animaworks anima set-model {名前} {モデル名} --credential {credential名}

# 3. サーバー起動中なら再起動
animaworks anima restart {名前}
```

### 全Animaを一括変更

```bash
animaworks anima set-model --all {モデル名}
```

### 現在の設定を確認

```bash
animaworks anima info {名前}    # モデル・実行モード・credential等を表示
animaworks anima list --local   # 全Animaのモデル一覧
```

---

## コンテキストウィンドウ

### 解決順序

1. `models.json` の `context_window`
2. `config.json` の `model_context_windows`（ワイルドカードパターン）
3. コードのハードコードデフォルト（`MODEL_CONTEXT_WINDOWS`）
4. 最終フォールバック: 128,000 トークン

### 閾値の自動スケール

コンテキストウィンドウサイズに応じてコンパクション閾値が自動調整される:

- **200K以上**: 設定値そのまま（デフォルト 0.50）
- **200K未満**: 0.98 に向けて線形スケール

小モデルではシステムプロンプトだけでコンテキストの大半を占めるため、閾値を高くして誤発動を防ぐ。

---

## プロバイダ別 credential 設定

### Anthropic（デフォルト）

```json
{
  "credentials": {
    "anthropic": {
      "api_key": "sk-ant-..."
    }
  }
}
```

### Azure OpenAI

```json
{
  "credentials": {
    "azure": {
      "api_key": "",
      "base_url": "https://YOUR_RESOURCE.openai.azure.com",
      "keys": { "api_version": "2024-12-01-preview" }
    }
  }
}
```

### Vertex AI

```json
{
  "credentials": {
    "vertex": {
      "keys": {
        "vertex_project": "my-gcp-project",
        "vertex_location": "us-central1",
        "vertex_credentials": "/path/to/service-account.json"
      }
    }
  }
}
```

### vLLM（ローカルGPU推論）

```json
{
  "credentials": {
    "vllm-local": {
      "api_key": "dummy",
      "base_url": "http://192.168.1.100:8000/v1"
    }
  }
}
```

credential を設定後、Anima に紐付け:

```bash
animaworks anima set-model {名前} {モデル名} --credential {credential名}
```

---

## バックグラウンドモデル（コスト最適化）

Heartbeat / Inbox / Cron はメインモデルとは別の軽量モデルで実行できる。
`background_model` を設定すると、これらのバックグラウンド処理のコストを大幅に削減可能。

### foreground / background の区分

| 区分 | 使用モデル | 対象トリガー |
|------|-----------|-------------|
| **foreground** | メインモデル（`model`） | `chat`（人間との対話）、`task:*`（TaskExec実作業） |
| **background** | `background_model`（未設定時はメインモデル） | `heartbeat`、`inbox:*`（Anima間DM）、`cron:*` |

Heartbeat / Inbox / Cron は「判断・トリアージ」が主目的で、実行は TaskExec（メインモデル）が担う。

### 解決順序

1. Per-anima `status.json` の `background_model`
2. `config.json` の `heartbeat.default_model`（グローバルデフォルト）
3. メインモデル（`model`）にフォールバック

### 設定方法

```bash
# 特定Animaにbackground_model を設定
animaworks anima set-background-model {名前} claude-sonnet-4-6

# credential が異なるプロバイダの場合
animaworks anima set-background-model {名前} azure/gpt-4.1-mini --credential azure

# 全Animaに一括設定
animaworks anima set-background-model --all claude-sonnet-4-6

# background_model を削除（メインモデルにフォールバック）
animaworks anima set-background-model {名前} --clear

# サーバー起動中なら再起動
animaworks anima restart {名前}
```

### status.json での確認

```json
{
  "model": "claude-opus-4-6",
  "background_model": "claude-sonnet-4-6",
  "background_credential": null
}
```

`background_model` が未設定またはメインモデルと同一の場合、切替はスキップされる。

---

## ロールテンプレートとデフォルトモデル

`animaworks anima set-role` でロールを変更すると、デフォルトモデルも変更される:

| ロール | デフォルトモデル | background_model | max_turns | max_chains |
|--------|---------------|-----------------|-----------|------------|
| engineer | claude-opus-4-6 | claude-sonnet-4-6 | 200 | 10 |
| manager | claude-opus-4-6 | claude-sonnet-4-6 | 50 | 3 |
| writer | claude-sonnet-4-6 | — | 80 | 5 |
| researcher | claude-sonnet-4-6 | — | 30 | 2 |
| ops | openai/glm-4.7-flash | — | 30 | 2 |
| general | claude-sonnet-4-6 | — | 20 | 2 |

Opus 系ロール（engineer, manager）は `background_model` として Sonnet が自動設定される。
Sonnet 以下のロールは既にコスト効率が良いため、`background_model` は未設定。

---

## よくある質問

### モデルを変更したのに反映されない

`set-model` は `status.json` を更新するだけ。サーバー起動中は `anima restart {名前}` または `anima reload {名前}` が必要。

### models.json を編集したのに反映されない

models.json はファイルの mtime で自動リロードされる。`anima reload` でも反映可能。

### コンテキストウィンドウを増やしたい

`models.json` の `context_window` を変更するか、`config.json` の `model_context_windows` でオーバーライド。

### どのモデルを選べばよいかわからない

- **高品質・自律実行が必要** → `claude-opus-4-6`（Mode S）
- **バランス・コスト重視** → `claude-sonnet-4-6`（Mode S）
- **低コスト・大量処理** → `openai/gpt-4.1-mini`（Mode A）
- **ローカルGPU・コスト$0** → `openai/qwen3.5-35b-a3b`（Mode A, vLLM）**推奨**
- **ローカル・軽量** → `ollama/qwen3:14b`（Mode A）

### Heartbeat / Cron のコストを下げたい

`background_model` を設定する。詳細は上記「バックグラウンドモデル（コスト最適化）」セクションを参照。
Opus をメインに使っている場合、`background_model` に Sonnet を設定するだけで Heartbeat + Inbox コストを約73%削減できる。

vLLMで `openai/qwen3.5-35b-a3b` を `background_model` に設定すれば、**バックグラウンド処理コストを完全に$0にできる**。Sonnet同等の88%総合スコアが確認済み。
