# モデル選択・設定ガイド

AnimaWorks のモデル設定に関する包括ガイド。
実行モード、対応モデル、設定方法、コンテキストウィンドウの仕組みを解説する。

---

## 実行モード

AnimaWorks はモデル名から実行モードを自動判定する。4種類の実行モードがある:

| モード | 名称 | 概要 | 対象モデル例 |
|--------|------|------|-------------|
| **S** | SDK | Claude Agent SDK経由。最も高機能 | `claude-opus-4-6`, `claude-sonnet-4-6` |
| **C** | Codex | Codex CLI経由 | `codex/o4-mini`, `codex/gpt-4.1` |
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

### ローカルモデル / Ollama

| モデル | モード | 説明 |
|--------|--------|------|
| `ollama/qwen3:14b` | A | 中型・tool_use対応 |
| `ollama/glm-4.7` | A | tool_use対応 |
| `ollama/gemma3:4b` | B | 軽量 |

---

## models.json

`~/.animaworks/models.json` でモデルごとの実行モードとコンテキストウィンドウを定義する。
fnmatch ワイルドカードパターンが使用可能。

### スキーマ

```json
{
  "パターン": {
    "mode": "S" | "A" | "B" | "C",
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
    "vllm-gpu41": {
      "api_key": "dummy",
      "base_url": "http://192.168.12.41:8000/v1"
    }
  }
}
```

credential を設定後、Anima に紐付け:

```bash
animaworks anima set-model {名前} {モデル名} --credential {credential名}
```

---

## ロールテンプレートとデフォルトモデル

`animaworks anima set-role` でロールを変更すると、デフォルトモデルも変更される:

| ロール | デフォルトモデル | max_turns | max_chains |
|--------|---------------|-----------|------------|
| engineer | claude-opus-4-6 | 200 | 10 |
| manager | claude-opus-4-6 | 50 | 3 |
| writer | claude-sonnet-4-6 | 80 | 5 |
| researcher | claude-sonnet-4-6 | 30 | 2 |
| ops | openai/glm-4.7-flash | 30 | 2 |
| general | claude-sonnet-4-6 | 20 | 2 |

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
- **ローカル・プライベート** → `ollama/qwen3:14b`（Mode A）
