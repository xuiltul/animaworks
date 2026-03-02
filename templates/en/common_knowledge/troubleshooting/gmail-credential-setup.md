# Gmail Tool Credential Setup Guide

## Overview

To use the Gmail tool, you need both `gmail: yes` in `permissions.md` AND an OAuth token file (`token.json`) placed at the correct runtime path.

## Prerequisites

1. `permissions.md` must include `gmail: yes`
2. `~/.animaworks/credentials/gmail/token.json` must exist

**Important**: Permission alone is not enough. The token.json file is required.

## Authentication Flow (GmailClient._get_credentials)

GmailClient searches for credentials in this order:

1. **MCP token** — `~/.mcp-cache/workspace-mcp/token.json` (for MCP-GSuite integration)
2. **Saved token** — `~/.animaworks/credentials/gmail/token.json`
3. **New OAuth flow** — Uses credentials.json or env vars `GMAIL_CLIENT_ID` / `GMAIL_CLIENT_SECRET` (requires browser authentication)

Normal operation uses step 2 with `token.json`.

## token.json Format

JSON format output by `google.oauth2.credentials.Credentials.to_json()`:

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

**Note**: `client_id` and `client_secret` are embedded in the JSON. These values are used for token refresh, so they don't need to match `credentials.json` or environment variables (the JSON values take priority).

## Common Issues

### Symptom: Gmail tool throws an error

```
ValueError: No OAuth credentials found. Place credentials.json or set GMAIL_CLIENT_ID / GMAIL_CLIENT_SECRET.
```

### Cause

`~/.animaworks/credentials/gmail/token.json` does not exist.

### Resolution

1. Ask an administrator to generate token.json
2. Token generation requires converting an existing OAuth token (e.g., pickle format) or running a browser-based OAuth flow
3. You cannot run the OAuth flow yourself (browser interaction required)

### Converting from token.pickle (Admin Only)

If an existing pickle-format token is available:

```python
import pickle
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request

# 1. Load pickle
with open("path/to/token.pickle", "rb") as f:
    creds = pickle.load(f)

# 2. Set client_secret if not embedded
if not creds.client_secret:
    creds._client_secret = "matching_client_secret"

# 3. Refresh
creds.refresh(Request())

# 4. Save as JSON
import os
target = os.path.expanduser("~/.animaworks/credentials/gmail/token.json")
os.makedirs(os.path.dirname(target), exist_ok=True)
with open(target, "w") as f:
    f.write(creds.to_json())
```

### client_id Mismatch

The `client_id` in token.json must match the OAuth client that originally generated the token. Using a token with a different client_id will cause authentication errors during refresh.

## Related Files

| Path | Description |
|------|-------------|
| `~/.animaworks/credentials/gmail/token.json` | OAuth token (required) |
| `~/.animaworks/credentials/gmail/credentials.json` | OAuth client info (new flow only) |
| `~/.animaworks/shared/credentials.json` | Env var settings (`GMAIL_CLIENT_ID`, `GMAIL_CLIENT_SECRET`) |
| `core/tools/gmail.py` | Gmail tool implementation |
