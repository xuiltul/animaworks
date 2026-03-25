---
name: discord-tool
description: >-
  Discord integration tool. Send/receive messages, search, list servers (guilds) and channels, add reactions.
  "Discord" "server" "guild" "channel" "reaction"
tags: [communication, discord, external]
---

# Discord Tool

External tool for Discord messaging, search, listing guilds and channels, and reactions.

## Invocation via Bash

Use **Bash** with `animaworks-tool discord <subcommand> [args]`:

```bash
animaworks-tool discord guilds
animaworks-tool discord channels GUILD_ID
animaworks-tool discord send CHANNEL_ID MESSAGE [--reply-to MESSAGE_ID]
animaworks-tool discord messages CHANNEL_ID [-n 20]
animaworks-tool discord search KEYWORD [-c CHANNEL_ID] [-n 50]
```

## Actions

### guilds — List servers (guilds)
```bash
animaworks-tool discord guilds
```

### channels — List channels
```bash
animaworks-tool discord channels GUILD_ID
```
- `GUILD_ID`: Snowflake ID of the target guild (server), **required**

### send — Send message
```bash
animaworks-tool discord send CHANNEL_ID MESSAGE [--reply-to MESSAGE_ID]
```
- `CHANNEL_ID`: Snowflake ID of the target text channel
- `--reply-to`: Optional. Message ID to reply to

### messages — Fetch messages
```bash
animaworks-tool discord messages CHANNEL_ID [-n 20]
```

### search — Search messages
```bash
animaworks-tool discord search KEYWORD [-c CHANNEL_ID] [-n 50]
```

### react — Add reaction (MCP only; not available via CLI)
- Add an emoji reaction to a message. **MCP only**; the CLI does not support this action.

## CLI usage (consolidated)

```bash
animaworks-tool discord guilds
animaworks-tool discord channels GUILD_ID
animaworks-tool discord send CHANNEL_ID MESSAGE [--reply-to MESSAGE_ID]
animaworks-tool discord messages CHANNEL_ID [-n 20]
animaworks-tool discord search KEYWORD [-c CHANNEL_ID] [-n 50]
```

## Notes

- Discord Bot Token must be configured in credentials beforehand
- Copy channel IDs with Developer Mode: right-click the channel → Copy ID
- Messages are limited to 2000 characters
- Guild, channel, and message IDs are numeric strings (Snowflake IDs)
- The bot needs appropriate permissions: Send Messages, Read Message History, Add Reactions, View Channels
