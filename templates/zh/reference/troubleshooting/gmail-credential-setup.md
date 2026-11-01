# Gmail Tool 认证配置指南

## 概述

要使用 Gmail tool，除了在 `permissions.json` 中获得许可外，还需要在运行时上放置 OAuth 令牌文件（`token.json`）。

## 前提条件

1. `permissions.json` 中已允许 `gmail`（包含在 external_tools.allow_all 或 allow 中）
2. `~/.animaworks/credentials/gmail/token.json` 存在

**重要**：仅在 permissions.json 中允许并不能正常工作。需要 token.json。

## 认证流程（GmailClient._get_credentials）

GmailClient 按以下顺序查找认证信息：

1. **MCP token** — `~/.mcp-cache/workspace-mcp/token.json`（用于 MCP-GSuite 集成）
2. **已保存的 token** — `~/.animaworks/credentials/gmail/token.json`
3. **新 OAuth 流程** — 使用 credentials.json 或环境变量 `GMAIL_CLIENT_ID` / `GMAIL_CLIENT_SECRET`（需要浏览器认证）

通常使用步骤 2 中的 `token.json` 进行运行。

## token.json 的格式

`google.oauth2.credentials.Credentials.to_json()` 输出的 JSON 格式：

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

**注意**：JSON 中包含 `client_id` 和 `client_secret`。由于 token 更新时会使用该值，因此不必与 `credentials.json` 或环境变量的值一致（以 JSON 内的值为准）。

## 常见问题

### 症状：Gmail tool 报错

```
ValueError: No OAuth credentials found. Place credentials.json or set GMAIL_CLIENT_ID / GMAIL_CLIENT_SECRET.
```

### 原因

`~/.animaworks/credentials/gmail/token.json` 不存在。

### 处理步骤

1. 请管理员生成 token.json
2. 生成 token.json 需要从现有 OAuth 令牌（pickle 格式等）转换，或进行浏览器认证
3. 无法自行执行 OAuth 流程（因为需要浏览器操作）

### 从 token.pickle 转换的步骤（面向管理员）

如果已有 pickle 格式的令牌：

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

### client_id 不一致的问题

token.json 中包含的 `client_id` 必须与该令牌最初生成时的 OAuth 客户端 ID 一致。如果使用由不同客户端 ID 生成的令牌，刷新时会出现认证错误。

## 相关文件

| 路径 | 内容 |
|------|------|
| `~/.animaworks/credentials/gmail/token.json` | OAuth 认证令牌（必需） |
| `~/.animaworks/credentials/gmail/credentials.json` | OAuth 客户端信息（仅新流程时） |
| `~/.animaworks/shared/credentials.json` | 环境变量配置（`GMAIL_CLIENT_ID`, `GMAIL_CLIENT_SECRET`） |
| `core/integrations/gmail.py` | Gmail tool 的实现 |
