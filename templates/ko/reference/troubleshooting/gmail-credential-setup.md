# Gmail Tool 인증 설정 가이드

## 개요

Gmail tool을 사용하려면 `permissions.json`에서의 허가에 더해 런타임 경로에 OAuth 토큰 파일(`token.json`)을 배치해야 합니다.

## 전제 조건

1. `permissions.json`에 `gmail: yes`가 설정되어 있을 것
2. `~/.animaworks/credentials/gmail/token.json`이 존재할 것

**중요**: permissions.json에서 허가되어 있는 것만으로는 동작하지 않습니다. token.json이 필요합니다.

## 인증 플로우 (GmailClient._get_credentials)

GmailClient는 다음 순서로 인증 정보를 탐색합니다:

1. **MCP token** — `~/.mcp-cache/workspace-mcp/token.json` (MCP-GSuite 연동용)
2. **저장된 token** — `~/.animaworks/credentials/gmail/token.json`
3. **새 OAuth 플로우** — credentials.json 또는 환경 변수 `GMAIL_CLIENT_ID` / `GMAIL_CLIENT_SECRET`을 사용 (브라우저 인증 필요)

일반적으로 2번의 `token.json`으로 운용합니다.

## token.json 형식

`google.oauth2.credentials.Credentials.to_json()`이 출력하는 JSON 형식입니다:

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

**참고**: `client_id`와 `client_secret`이 JSON 안에 포함되어 있습니다. 토큰 갱신 시 이 값이 사용되므로, `credentials.json`이나 환경 변수의 값과 일치할 필요는 없습니다 (JSON 내의 값이 우선됨).

## 자주 발생하는 문제

### 증상: Gmail tool에서 오류 발생

```
ValueError: No OAuth credentials found. Place credentials.json or set GMAIL_CLIENT_ID / GMAIL_CLIENT_SECRET.
```

### 원인

`~/.animaworks/credentials/gmail/token.json`이 존재하지 않습니다.

### 대처 절차

1. 관리자에게 token.json 생성을 요청합니다
2. token.json 생성에는 기존 OAuth 토큰(pickle 형식 등)의 변환 또는 브라우저 인증이 필요합니다
3. 직접 OAuth 플로우를 실행할 수 없습니다 (브라우저 조작이 필요하므로)

### token.pickle에서의 변환 절차 (관리자용)

기존의 pickle 형식 토큰이 있는 경우:

```python
import pickle
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request

# 1. pickle 읽기
with open("path/to/token.pickle", "rb") as f:
    creds = pickle.load(f)

# 2. client_secret이 포함되어 있지 않은 경우 설정
if not creds.client_secret:
    creds._client_secret = "해당하는_client_secret"

# 3. 갱신
creds.refresh(Request())

# 4. JSON 형식으로 저장
import os
target = os.path.expanduser("~/.animaworks/credentials/gmail/token.json")
os.makedirs(os.path.dirname(target), exist_ok=True)
with open(target, "w") as f:
    f.write(creds.to_json())
```

### client_id 불일치 문제

token.json에 포함된 `client_id`는 해당 토큰을 최초로 생성한 OAuth 클라이언트의 ID와 일치해야 합니다. 다른 클라이언트 ID로 생성된 토큰을 사용하면 갱신 시 인증 오류가 발생합니다.

## 관련 파일

| 경로 | 내용 |
|------|------|
| `~/.animaworks/credentials/gmail/token.json` | OAuth 인증 토큰 (필수) |
| `~/.animaworks/credentials/gmail/credentials.json` | OAuth 클라이언트 정보 (새 플로우 시에만) |
| `~/.animaworks/shared/credentials.json` | 환경 변수 설정 (`GMAIL_CLIENT_ID`, `GMAIL_CLIENT_SECRET`) |
| `core/tools/gmail.py` | Gmail tool 구현 |
