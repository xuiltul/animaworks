# Slack Bot Token Configuration Guide

How Slack bot tokens work in AnimaWorks and the rules for configuring per-Anima tokens.

## Two Types of Bot Tokens

AnimaWorks Slack integration uses **shared bots** and **per-Anima bots**.

| Type | Key Name | Purpose |
|------|----------|---------|
| Shared bot | `SLACK_BOT_TOKEN` / `SLACK_APP_TOKEN` | System-wide fallback. Used by Animas without per-Anima tokens |
| Per-Anima bot | `SLACK_BOT_TOKEN__<name>` / `SLACK_APP_TOKEN__<name>` | Dedicated Slack App for a specific Anima |

**Per-Anima bot tokens take priority when configured.** The shared bot is only a fallback.

## Per-Anima Token Naming Convention

Append `__` (double underscore) + Anima name (lowercase) as a suffix.

```
SLACK_BOT_TOKEN__sumire    ← sumire's dedicated Bot User OAuth Token
SLACK_APP_TOKEN__sumire    ← sumire's dedicated App-Level Token
```

## Storage Location

Tokens are stored in `shared/credentials.json`.

```json
{
  "SLACK_BOT_TOKEN": "xoxb-...(shared bot)",
  "SLACK_APP_TOKEN": "xapp-...(shared app)",
  "SLACK_BOT_TOKEN__sumire": "xoxb-...(sumire's bot)",
  "SLACK_APP_TOKEN__sumire": "xapp-...(sumire's app)"
}
```

## Critical Rules

### NEVER Overwrite Shared Tokens

**MUST**: When adding per-Anima tokens, **add new keys**. Never replace the existing `SLACK_BOT_TOKEN` or `SLACK_APP_TOKEN` with your own token.

Shared tokens are used by other Animas and system-wide features. Overwriting them will:

- Break Slack communication for other Animas
- Cause a mismatch between Socket Mode connection and Bot Token App
- Trigger unexpected errors like `not_in_channel`

### How to Add Tokens via File Edit

```bash
# Read credentials.json, add new keys, write back
python3 -c "
import json
from pathlib import Path
p = Path.home() / '.animaworks/shared/credentials.json'
d = json.loads(p.read_text())
d['SLACK_BOT_TOKEN__<your_name>'] = 'xoxb-...'
d['SLACK_APP_TOKEN__<your_name>'] = 'xapp-...'
p.write_text(json.dumps(d, indent=2))
"
```

**Note**: Use JSON key addition, not string replacement of existing lines.

## Server Discovery and Restart

Per-Anima tokens are detected at **server startup**. After adding tokens to `shared/credentials.json`, a **server restart is required**.

At startup, the server:

1. Discovers `SLACK_BOT_TOKEN__*` and `SLACK_APP_TOKEN__*` pairs
2. Registers a per-Anima Socket Mode handler for each pair
3. Runs `auth.test` to obtain Bot User ID for channel routing

After restart, the following log entry confirms success:

```
Per-anima Slack bot registered: <name> (bot_uid=U...)
```

## Troubleshooting

### not_in_channel Error

**Symptom**: Getting `not_in_channel` error when trying to reply in a Slack channel

**Cause**: Per-Anima tokens are not configured, so the shared bot is used. The shared bot is not a member of that channel.

**Fix**:
1. Add `SLACK_BOT_TOKEN__<name>` and `SLACK_APP_TOKEN__<name>` to `shared/credentials.json`
2. Restart the server
3. Verify the per-Anima bot is invited to the target channel

### Falling Back to Shared Bot

**Symptom**: Your messages appear under the shared bot name instead of your dedicated bot

**Cause**: `SLACK_BOT_TOKEN__<name>` / `SLACK_APP_TOKEN__<name>` do not exist in `credentials.json`

**How to check**:
```bash
cat ~/.animaworks/shared/credentials.json | python3 -c "
import sys, json
d = json.load(sys.stdin)
for k in sorted(d):
    if 'SLACK' in k:
        print(f'{k}: {d[k][:20]}...')
"
```

### How to Obtain Tokens

Ask the administrator (human) to provide the Slack App's Bot User OAuth Token (`xoxb-`) and App-Level Token (`xapp-`).

- Bot Token: Slack App settings → OAuth & Permissions → Bot User OAuth Token
- App-Level Token: Slack App settings → Basic Information → App-Level Tokens (scope: `connections:write`)
