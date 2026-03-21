---
name: slack-tool
description: >-
  Slack integration tool. Send/receive messages, search, check unreplied, list channels, emoji reactions.
  "slack" "channel" "thread" "reaction"
tags: [communication, slack, external]
---

# Slack Tool

External tool for Slack messaging, search, reactions, and channel management.

## Invocation via Bash

Use **Bash** with `animaworks-tool slack <subcommand> [args]`:

```bash
animaworks-tool slack send CHANNEL MESSAGE [--thread TS]
animaworks-tool slack messages CHANNEL [-n 20]
animaworks-tool slack search KEYWORD [-c CHANNEL] [-n 50]
animaworks-tool slack unreplied [--json]
animaworks-tool slack channels
```

## Actions

### send — Send message
```json
{"tool_name": "slack", "action": "send", "args": {"channel": "#channel-name", "message": "text", "thread": "thread ts (optional)"}}
```

### messages — Get messages
```json
{"tool_name": "slack", "action": "messages", "args": {"channel": "#channel-name", "limit": 20}}
```

### search — Search messages
```json
{"tool_name": "slack", "action": "search", "args": {"keyword": "search term", "channel": "#channel (optional)", "limit": 50}}
```

### unreplied — Check unreplied messages
```json
{"tool_name": "slack", "action": "unreplied", "args": {}}
```

### channels — List channels
```json
{"tool_name": "slack", "action": "channels", "args": {}}
```

### react — Add emoji reaction
```json
{"tool_name": "slack", "action": "react", "args": {"channel": "#channel-name", "emoji": "thumbsup", "message_ts": "message timestamp"}}
```
- `emoji`: Slack emoji name without colons (e.g. `thumbsup`, `eyes`, `white_check_mark`)
- `message_ts`: Timestamp of the target message (available from `messages` action results)

## CLI Usage (S/C/D/G-mode)

```bash
animaworks-tool slack send CHANNEL MESSAGE [--thread TS]
animaworks-tool slack messages CHANNEL [-n 20]
animaworks-tool slack search KEYWORD [-c CHANNEL] [-n 50]
animaworks-tool slack unreplied [--json]
animaworks-tool slack channels
```

> The `react` action is not available via CLI. Use MCP instead.

## Notes

- Slack Bot Token must be configured in credentials
- Channel can be specified with # prefix or by channel ID
- Reactions require the `reactions:write` scope
