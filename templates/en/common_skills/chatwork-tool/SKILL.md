---
name: chatwork-tool
description: >-
  Chatwork integration tool. Send/receive messages, search, check unreplied, list rooms.
  "chatwork" "CW" "unreplied" "room" "mention"
tags: [communication, chatwork, external]
---

# Chatwork Tool

External tool for Chatwork messaging, search, and room management.

## Invocation via Bash

Use **Bash** with `animaworks-tool chatwork <subcommand> [args]`. See Actions below for syntax.

## Actions

### send — Send message
```json
{"tool_name": "chatwork", "action": "send", "args": {"room": "room name or ID", "message": "text"}}
```

### messages — Get messages
```json
{"tool_name": "chatwork", "action": "messages", "args": {"room": "room name or ID", "limit": 20}}
```

### search — Search messages
```json
{"tool_name": "chatwork", "action": "search", "args": {"keyword": "search term", "room": "room (optional)", "limit": 50}}
```

### unreplied — Check unreplied messages
```json
{"tool_name": "chatwork", "action": "unreplied", "args": {"include_toall": false}}
```
- `include_toall` (optional, default: false): Include messages addressed to all

### rooms — List rooms
```json
{"tool_name": "chatwork", "action": "rooms", "args": {}}
```

### mentions — Get mentions
```json
{"tool_name": "chatwork", "action": "mentions", "args": {"include_toall": false}}
```
- `include_toall` (optional, default: false): Include messages addressed to all

### delete — Delete message (own messages only)
```json
{"tool_name": "chatwork", "action": "delete", "args": {"room": "room name or ID", "message_id": "message ID"}}
```

### sync — Sync messages (cache update)
```json
{"tool_name": "chatwork", "action": "sync", "args": {"room": "room name or ID"}}
```

## CLI Usage (S/C/D/G-mode)

```bash
animaworks-tool chatwork send ROOM MESSAGE
animaworks-tool chatwork messages ROOM [-n 20]
animaworks-tool chatwork search KEYWORD [-r ROOM] [-n 50]
animaworks-tool chatwork unreplied [--json]
animaworks-tool chatwork rooms
animaworks-tool chatwork mentions [--json]
animaworks-tool chatwork delete ROOM MESSAGE_ID
animaworks-tool chatwork sync [ROOM]
```

## Notes

- Chatwork API Token must be configured in credentials
- Room can be specified by name or room ID
- Write token may be required for sending messages
