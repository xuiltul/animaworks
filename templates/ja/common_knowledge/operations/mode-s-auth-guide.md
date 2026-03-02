# Mode S（Agent SDK）認証モード設定ガイド

Mode S（Claude Agent SDK）で使用する認証方式を Anima ごとに切り替える方法。
認証モードは **`mode_s_auth`** という明示的な設定で指定する（credential の自動判定ではない）。

## 認証モード一覧

| モード | mode_s_auth 値 | 接続先 | 用途 |
|--------|----------------|--------|------|
| **API 直接** | `"api"` | Anthropic API | 最速ストリーミング。API クレジット消費 |
| **Bedrock** | `"bedrock"` | AWS Bedrock | AWS 統合・VPC 内利用 |
| **Vertex AI** | `"vertex"` | Google Vertex AI | GCP 統合 |
| **Max plan** | `"max"` または未設定 | Anthropic Max plan | サブスクリプション認証。API クレジット不要 |

`mode_s_auth` が未設定（`null` または省略）の場合は Max plan になる。

## 設定の優先順位

`mode_s_auth` は次の順で解決される:

1. **status.json**（Anima 個別）— 最優先
2. **config.json anima_defaults** — グローバルデフォルト

credential の内容からは自動判定されない。明示的に `mode_s_auth` を指定すること。

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
  "credential": "anthropic",
  "mode_s_auth": "api"
}
```

`mode_s_auth` が `"api"` で credential に `api_key` がない場合、Max plan にフォールバックする。

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
  "credential": "bedrock",
  "execution_mode": "S",
  "mode_s_auth": "bedrock"
}
```

Bedrock を Mode S で使う場合は `execution_mode: "S"` と `mode_s_auth: "bedrock"` の両方が必要。

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
  "credential": "vertex",
  "execution_mode": "S",
  "mode_s_auth": "vertex"
}
```

### 4. Max plan モード（デフォルト）

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

`mode_s_auth` を省略するか `"max"` にすると Max plan になる。

## Anima ごとの使い分け例

同じ組織内で認証モードを混在させる場合、各 Anima の status.json で `mode_s_auth` と `credential` を指定する:

```json
{
  "credentials": {
    "anthropic": { "api_key": "sk-ant-api03-xxxxx" },
    "max": { "api_key": "" },
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

| 例（役割） | credential | mode_s_auth | 認証モード | 理由 |
|-----------|-----------|-------------|-----------|------|
| Max plan 利用の Anima | `"max"` | 省略 | Max plan | API コスト不要 |
| API 直接利用の Anima | `"anthropic"` | `"api"` | API 直接 | 高速ストリーミングが必要 |
| Bedrock 利用の Anima | `"bedrock"` | `"bedrock"` | Bedrock | AWS VPC 内からのみアクセス |

**現在の構成を確認する方法:** 各 Anima の `status.json` で `credential` と `mode_s_auth` を確認する:

```bash
# 特定 Anima の mode_s_auth 確認
cat ~/.animaworks/animas/{name}/status.json | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'model={d.get(\"model\")}, credential={d.get(\"credential\")}, mode_s_auth={d.get(\"mode_s_auth\")}')"

# 全 Anima の一覧
for d in ~/.animaworks/animas/*/; do name=$(basename "$d"); python3 -c "import json; d=json.load(open('$d/status.json')); print(f'$name: credential={d.get(\"credential\")}, mode_s_auth={d.get(\"mode_s_auth\")}')" 2>/dev/null; done
```

## グローバルデフォルト（anima_defaults）

全 Anima で Bedrock をデフォルトにしたい場合、config.json の `anima_defaults` に設定する:

```json
{
  "anima_defaults": {
    "mode_s_auth": "bedrock"
  },
  "credentials": {
    "bedrock": { "api_key": "", "keys": { "aws_access_key_id": "...", ... } }
  }
}
```

個別 Anima の status.json で `mode_s_auth` を上書きできる。

## 注意事項

- 認証モードは `_build_env()` で Claude Code 子プロセスの環境変数として渡される
- `mode_s_auth` は credential の内容から自動判定されない。明示指定が必須
- `mode_s_auth=api` で credential に `api_key` がない場合、Max plan にフォールバックする
- Bedrock / Vertex を使う場合は credential の `keys` にプロバイダ固有のキーを設定し、`mode_s_auth` でモードを指定する
- 設定変更後はサーバー再起動が必要
- Mode A/B では従来通り LiteLLM が credential を使用する（この設定は Mode S 専用）
