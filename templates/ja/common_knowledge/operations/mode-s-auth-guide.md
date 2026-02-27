# Mode S（Agent SDK）認証モード設定ガイド

Mode S（Claude Agent SDK）で使用する認証方式を Anima ごとに切り替える方法。
credential の設定内容から認証モードが自動判定される。

## 認証モード一覧

| モード | 条件 | 接続先 | 用途 |
|--------|------|--------|------|
| **API 直接** | credential に `api_key` がある | Anthropic API | 最速ストリーミング。API クレジット消費 |
| **Bedrock** | credential の `keys` に `aws_access_key_id` がある | AWS Bedrock | AWS 統合・VPC 内利用 |
| **Vertex AI** | credential の `keys` に `vertex_project` がある | Google Vertex AI | GCP 統合 |
| **Max plan** | 上記いずれにも該当しない（デフォルト） | Anthropic Max plan | サブスクリプション認証。API クレジット不要 |

判定は上から順に行われる。`api_key` がある場合は常に API 直接モードになる。

## 設定方法

### 1. API 直接モード

Anthropic API に直接接続する。ストリーミングが最もスムーズ。

**config.json の credential 設定:**

```json
{
  "credentials": {
    "anthropic": {
      "api_key": "sk-ant-api03-xxxxx"
    }
  }
}
```

**status.json（Anima 個別）:**

```json
{
  "model": "claude-sonnet-4-6",
  "credential": "anthropic"
}
```

### 2. Bedrock モード

AWS Bedrock 経由で接続する。

**config.json の credential 設定:**

```json
{
  "credentials": {
    "bedrock": {
      "api_key": "",
      "keys": {
        "aws_access_key_id": "AKIA...",
        "aws_secret_access_key": "...",
        "aws_region_name": "us-east-1"
      }
    }
  }
}
```

**status.json（Anima 個別）:**

```json
{
  "model": "claude-sonnet-4-6",
  "credential": "bedrock"
}
```

### 3. Vertex AI モード

Google Vertex AI 経由で接続する。

**config.json の credential 設定:**

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

**status.json（Anima 個別）:**

```json
{
  "model": "claude-sonnet-4-6",
  "credential": "vertex"
}
```

### 4. Max plan モード（デフォルト）

`api_key` なし・プロバイダ keys なしの credential を指定する。
Claude Code のサブスクリプション認証（Max plan 等）を使用する。

**config.json の credential 設定:**

```json
{
  "credentials": {
    "max": {
      "api_key": ""
    }
  }
}
```

**status.json（Anima 個別）:**

```json
{
  "model": "claude-sonnet-4-6",
  "credential": "max"
}
```

## Anima ごとの使い分け例

同じ組織内で認証モードを混在させる場合:

```json
{
  "credentials": {
    "anthropic": { "api_key": "sk-ant-api03-xxxxx" },
    "max": { "api_key": "" },
    "bedrock": { "api_key": "", "keys": { "aws_access_key_id": "AKIA...", "aws_secret_access_key": "...", "aws_region_name": "us-east-1" } }
  }
}
```

| Anima | credential | 認証モード | 理由 |
|-------|-----------|-----------|------|
| sakura | `"max"` | Max plan | マネージャー。API コスト不要 |
| kotoha | `"anthropic"` | API 直接 | 高速ストリーミングが必要 |
| rin | `"bedrock"` | Bedrock | AWS VPC 内からのみアクセス |

## 注意事項

- 認証モードは `_build_env()` で Claude Code 子プロセスの環境変数として渡される
- credential の `api_key` とプロバイダ keys の両方がある場合、`api_key` が優先される（API 直接モード）
- Bedrock を使いたい場合は `api_key` を空にした別 credential を作ること
- 設定変更後はサーバー再起動が必要
- Mode A/B では従来通り LiteLLM が credential を使用する（この設定は Mode S 専用）
