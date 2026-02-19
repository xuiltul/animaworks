# Slack Socket Mode Setup Guide

Setup instructions for AnimaWorks to receive Slack messages in real time.

## Overview

Socket Mode is a method for receiving events pushed from Slack via WebSocket.
It requires no public URL and works on servers behind NAT.

```
[Slack] ←WebSocket→ [SlackSocketModeManager] → Messenger.receive_external() → [Anima inbox]
                              ↑
                     Auto-started in server/app.py lifespan
                     Controlled by config.json external_messaging.slack
```

## Prerequisites

- AnimaWorks server is running
- `slack-bolt` and `aiohttp` are installed (included in `pyproject.toml`)

## 1. Slack App Configuration (Slack Admin Console)

Configure your app at https://api.slack.com/apps.

### Enabling Socket Mode

1. Select "Socket Mode" from the left menu
2. Turn on "Enable Socket Mode"
3. Generate an App-Level Token (scope: `connections:write`)
4. Save the generated `xapp-...` token

### Event Subscriptions

1. Select "Event Subscriptions" from the left menu
2. Turn on "Enable Events"
3. No Request URL is needed (because Socket Mode is used)
4. Add the following under "Subscribe to bot events":

| Event | Description |
|-------|-------------|
| `message.channels` | Messages in public channels |
| `message.groups` | Messages in private channels |
| `message.im` | Direct messages |
| `message.mpim` | Group DMs |
| `app_mention` | @mentions |

### OAuth Scopes (Bot Token Scopes)

Add the following under "OAuth & Permissions" in the left menu:

| Scope | Purpose |
|-------|---------|
| `channels:history` | Read public channel history |
| `channels:read` | List channels |
| `chat:write` | Send messages |
| `groups:history` | Read private channel history |
| `groups:read` | List private channels |
| `im:history` | Read DM history |
| `im:read` | List DMs |
| `im:write` | Open DMs |
| `mpim:history` | Read group DM history |
| `mpim:read` | List group DMs |
| `users:read` | Retrieve user information |
| `app_mentions:read` | Read @mentions |

### App Home

1. Select "App Home" from the left menu
2. Enable the "Messages Tab"
3. Check "Allow users to send Slash commands and messages from the messages tab"

### Installing to the Workspace

1. Select "Install App" from the left menu
2. Click "Install to Workspace"
3. Save the `xoxb-...` token displayed after authorization

### Inviting the Bot to a Channel

Run `/invite @BotName` in each channel where you want to receive messages.

## 2. Credential Configuration (AnimaWorks Side)

Set the following keys in `~/.animaworks/shared/credentials.json`:

```json
{
  "SLACK_BOT_TOKEN": "xoxb-...",
  "SLACK_APP_TOKEN": "xapp-..."
}
```

| Key | Prefix | Purpose |
|-----|--------|---------|
| `SLACK_BOT_TOKEN` | `xoxb-` | Slack API calls (sending messages, retrieving info) |
| `SLACK_APP_TOKEN` | `xapp-` | Establishing the Socket Mode WebSocket connection |

You can also use the environment variables `SLACK_BOT_TOKEN` and `SLACK_APP_TOKEN` (credentials.json takes precedence).

## 3. config.json Configuration

Add the `external_messaging` section to `~/.animaworks/config.json`:

```json
{
  "external_messaging": {
    "slack": {
      "enabled": true,
      "mode": "socket",
      "anima_mapping": {
        "C0ACT663B5L": "sakura"
      }
    }
  }
}
```

### Configuration Options

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | bool | `false` | Enable/disable Slack message reception |
| `mode` | string | `"socket"` | `"socket"` (recommended) or `"webhook"` |
| `anima_mapping` | object | `{}` | Mapping of Slack channel IDs to Anima names |

### Finding the Channel ID

Right-click a channel name in Slack, select "View channel details", and find the channel ID (starting with `C`) at the bottom. DM IDs start with `D`.

### Differences Between Modes

| Mode | Connection Direction | Public URL | Use Case |
|------|---------------------|------------|----------|
| `socket` | Server to Slack (WebSocket) | Not required | Servers behind NAT (recommended) |
| `webhook` | Slack to Server (HTTP POST) | Required | Public-facing servers |

## 4. Restart the Server

```bash
animaworks start
```

If the following appears in the startup log, the connection was successful:

```
INFO  animaworks.slack_socket: Slack Socket Mode connected
```

When disabled:

```
INFO  animaworks.slack_socket: Slack Socket Mode is disabled
```

## Message Flow

1. A Slack user sends a message in a mapped channel
2. Slack sends the event via WebSocket to `SlackSocketModeManager`
3. `anima_mapping` resolves the channel ID to an Anima name
4. `Messenger.receive_external()` places the message at `~/.animaworks/shared/inbox/{anima_name}/{msg_id}.json`
5. The Anima processes the inbox on its next run cycle (heartbeat/cron/manual)

## Related Files

| File | Role |
|------|------|
| `server/slack_socket.py` | SlackSocketModeManager implementation |
| `server/app.py:171-179` | Start/stop within lifespan |
| `core/messenger.py:150-175` | `receive_external()` -- inbox placement |
| `core/config/models.py:160-172` | `ExternalMessagingChannelConfig` model |
| `server/routes/webhooks.py:64-113` | Webhook endpoint (when mode=webhook) |
| `core/tools/slack.py` | Polling-based tools (send/messages/unreplied -- coexists with Socket Mode) |

## Troubleshooting

### Cannot Connect

- Verify that `SLACK_APP_TOKEN` starts with `xapp-`
- Confirm that Socket Mode is enabled in the Slack App settings
- Confirm that Event Subscriptions are enabled

### Messages Are Not Received

- Verify that the channel IDs in `anima_mapping` are correct
- Confirm that the Bot has been invited to the target channel
- Check the server logs for `"No anima mapping for channel"` messages

### Reconnection

- `slack-bolt`'s `AsyncSocketModeHandler` supports automatic reconnection
- WebSocket connections are periodically refreshed approximately every hour
- If rate limiting (429) occurs during long-running operation, restart the server to resolve it

## Limitations

- Socket Mode apps cannot be published to the Slack App Directory (intended for internal tools)
- Maximum concurrent WebSocket connections: 10 per app
- `apps.connections.open` rate limit: 1 per minute
- Processing of messages placed in the inbox depends on the Anima's next run cycle
