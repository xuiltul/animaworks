# Gmail Tool 認証設定ガイド

## 概要

Gmail toolを利用するには、`permissions.md` での許可に加え、ランタイム上にOAuthトークンファイル（`token.json`）を配置する必要がある。

## 前提条件

1. `permissions.md` に `gmail: yes` が設定されていること
2. `~/.animaworks/credentials/gmail/token.json` が存在すること

**重要**: permissions.mdで許可されているだけでは動作しない。token.jsonが必要。

## 認証フロー（GmailClient._get_credentials）

GmailClientは以下の順序で認証情報を探索する:

1. **MCP token** — `~/.mcp-cache/workspace-mcp/token.json`（MCP-GSuite連携用）
2. **保存済みtoken** — `~/.animaworks/credentials/gmail/token.json`
3. **新規OAuthフロー** — credentials.json または環境変数 `GMAIL_CLIENT_ID` / `GMAIL_CLIENT_SECRET` を使用（ブラウザ認証が必要）

通常は手順2の `token.json` で運用する。

## token.json のフォーマット

`google.oauth2.credentials.Credentials.to_json()` が出力するJSON形式:

```json
{
  "token": "ya29.xxx...",
  "refresh_token": "1//xxx...",
  "token_uri": "https://oauth2.googleapis.com/token",
  "client_id": "xxxxx.apps.googleusercontent.com",
  "client_secret": "GOCSPX-xxx...",
  "scopes": [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/gmail.modify"
  ]
}
```

**注意**: `client_id` と `client_secret` がJSON内に含まれる。token更新時にこの値が使われるため、`credentials.json` や環境変数の値と一致している必要はない（JSON内の値が優先される）。

## よくある問題

### 症状: Gmail toolがエラーになる

```
ValueError: No OAuth credentials found. Place credentials.json or set GMAIL_CLIENT_ID / GMAIL_CLIENT_SECRET.
```

### 原因

`~/.animaworks/credentials/gmail/token.json` が存在しない。

### 対処手順

1. 管理者にtoken.jsonの生成を依頼する
2. token.jsonの生成には既存のOAuthトークン（pickle形式等）からの変換、またはブラウザ認証が必要
3. 自分でOAuthフローを実行することはできない（ブラウザ操作が必要なため）

### token.pickleからの変換手順（管理者向け）

既存のpickle形式トークンがある場合:

```python
import pickle
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request

# 1. pickle読み込み
with open("path/to/token.pickle", "rb") as f:
    creds = pickle.load(f)

# 2. client_secretが含まれていない場合は設定
if not creds.client_secret:
    creds._client_secret = "対応するclient_secret"

# 3. リフレッシュ
creds.refresh(Request())

# 4. JSON形式で保存
import os
target = os.path.expanduser("~/.animaworks/credentials/gmail/token.json")
os.makedirs(os.path.dirname(target), exist_ok=True)
with open(target, "w") as f:
    f.write(creds.to_json())
```

### client_id不一致の問題

token.jsonに含まれる `client_id` は、そのトークンを最初に生成したOAuthクライアントのIDと一致している必要がある。異なるクライアントIDで生成されたトークンを使うと、リフレッシュ時に認証エラーになる。

## 関連ファイル

| パス | 内容 |
|------|------|
| `~/.animaworks/credentials/gmail/token.json` | OAuth認証トークン（必須） |
| `~/.animaworks/credentials/gmail/credentials.json` | OAuthクライアント情報（新規フロー時のみ） |
| `~/.animaworks/shared/credentials.json` | 環境変数設定（`GMAIL_CLIENT_ID`, `GMAIL_CLIENT_SECRET`） |
| `core/tools/gmail.py` | Gmail toolの実装 |
