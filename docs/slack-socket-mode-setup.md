# Slack Socket Mode Setup Guide

Setup instructions for AnimaWorks to receive Slack messages in real time.

## Overview

Socket Mode receives events pushed from Slack over WebSocket.
No public URL is required; it works on servers behind NAT.

```
[Slack] ←WebSocket→ [SlackSocketModeManager] → Messenger.receive_external() → [Anima inbox]
                              ↑
                     Started in the background within server/app.py lifespan
                     Controlled by config.json external_messaging.slack
```

## Prerequisites

- The AnimaWorks server is running
- Communication-related optional dependencies are installed. Recommended:

  ```bash
  pip install "animaworks[communication]"
  ```

  This installs `slack-sdk` (Web API and webhook verification), `slack-bolt` (Socket Mode), and `aiohttp`. If any are missing, Socket Mode or the `slack` tool will not work.

## 1. Slack App Configuration (Slack admin UI)

Configure your app at https://api.slack.com/apps.

### Enabling Socket Mode (only when mode=socket)

1. In the left sidebar, choose **Socket Mode**
2. Turn **Enable Socket Mode** on
3. Generate an App-Level Token (scope: `connections:write`)
4. Save the generated `xapp-...` token

This step is not required for Webhook mode (`mode: "webhook"`).

### Event Subscriptions

1. In the left sidebar, choose **Event Subscriptions**
2. Turn **Enable Events** on
3. **Socket Mode**: No Request URL is needed
4. **Webhook mode**: Set the Request URL to `https://your-server/api/webhooks/slack/events` (the signature verification challenge is handled automatically)
5. Under **Subscribe to bot events**, add:

| Event | Description |
|-------|-------------|
| `message.channels` | Messages in public channels |
| `message.groups` | Messages in private channels |
| `message.im` | Direct messages |
| `message.mpim` | Group DMs |
| `app_mention` | @mentions |

### OAuth Scopes (Bot Token Scopes)

Under **OAuth & Permissions** in the left sidebar, add:

| Scope | Purpose |
|-------|---------|
| `channels:history` | Read public channel history |
| `channels:read` | List channels |
| `chat:write` | Send and update messages (`slack_send` / `slack_channel_post` / `slack_channel_update`) |
| `groups:history` | Read private channel history |
| `groups:read` | List private channels |
| `im:history` | Read DM history |
| `im:read` | List DMs |
| `im:write` | Open DMs |
| `mpim:history` | Read group DM history |
| `mpim:read` | List group DMs |
| `users:read` | Retrieve user information |
| `app_mentions:read` | Read @mentions |
| `reactions:write` | Add reactions (`slack_react`) |

Depending on workspace settings, posting with Anima display name and icon via `chat.postMessage` may require `chat:write.customize`.

### App Home

1. In the left sidebar, choose **App Home**
2. Enable the **Messages Tab**
3. Enable **Allow users to send Slash commands and messages from the messages tab**

### Installing to the workspace

1. In the left sidebar, choose **Install App**
2. Click **Install to Workspace**
3. Save the `xoxb-...` token shown after authorization

### Inviting the bot to a channel

Run `/invite @BotName` in each channel where you want to receive messages.

## 2. Credential configuration (AnimaWorks side)

Credentials are resolved in this order: `config.json` → vault → `shared/credentials.json` → environment variables.

Set the following keys in `~/.animaworks/shared/credentials.json`:

```json
{
  "SLACK_BOT_TOKEN": "xoxb-...",
  "SLACK_APP_TOKEN": "xapp-..."
}
```

| Key | Prefix | Purpose |
|-----|--------|---------|
| `SLACK_BOT_TOKEN` | `xoxb-` | Slack API calls (sending, fetching information) |
| `SLACK_APP_TOKEN` | `xapp-` | Establish Socket Mode WebSocket connection (only when mode=socket) |

You can also use environment variables or the `credentials` section of `config.json`. `SLACK_APP_TOKEN` is not required in Webhook mode.

**Per-Anima bot**: Use `SLACK_BOT_TOKEN__{anima_name}` and `SLACK_APP_TOKEN__{anima_name}` to give each Anima its own bot. Add these to vault or `shared/credentials.json`.

**Webhook mode with app_id_mapping**: Set Per-Anima signing secrets as `SLACK_SIGNING_SECRET__{anima_name}`.

## 3. config.json configuration

Control behavior under `external_messaging` in `~/.animaworks/config.json`.

### Minimal example (Slack Socket only)

```json
{
  "external_messaging": {
    "slack": {
      "enabled": true,
      "mode": "socket",
      "anima_mapping": {
        "C0ACT663B5L": "sakura"
      },
      "default_anima": "",
      "app_id_mapping": {}
    }
  }
}
```

### Additional top-level `external_messaging` fields

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `preferred_channel` | string | `"slack"` | Preferred outbound channel (`slack` / `chatwork`). Used for resolving Slack icon URLs for human notifications, etc. |
| `user_aliases` | object | `{}` | Human alias → contact info. On Slack ingest, **intent detection** uses `slack_user_id` (see below) |

Example `user_aliases` (alias names are arbitrary; `slack_user_id` is the Slack member ID `U...`):

```json
{
  "external_messaging": {
    "preferred_channel": "slack",
    "user_aliases": {
      "taro": { "slack_user_id": "U01234567", "chatwork_room_id": "" }
    },
    "slack": {
      "enabled": true,
      "mode": "socket",
      "anima_mapping": {},
      "default_anima": "sakura",
      "app_id_mapping": {}
    }
  }
}
```

If the channel message body **contains this ID** in the form `<@U01234567>`, it is treated like a bot mention with `intent="question"` (more likely to be processed immediately in the inbox). DMs are always `question` as before.

### `slack` subsection

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | bool | `false` | Enable or disable Slack ingestion |
| `mode` | string | `"socket"` | `"socket"` (recommended) or `"webhook"` |
| `anima_mapping` | object | `{}` | Slack channel ID → Anima name (shared bot) |
| `default_anima` | string | `""` | Fallback Anima when the channel is not in anima_mapping |
| `app_id_mapping` | object | `{}` | Slack API App ID → Anima name (multiple apps in Webhook mode) |

**Shared-bot mapping applies without reconnecting**: `server/slack_socket.py` calls `load_config()` on each message, so changes to `anima_mapping` / `default_anima` take effect from the next message onward (the WebSocket stays up).

### Finding the channel ID

In Slack, right-click the channel name → **View channel details** → the channel ID (starts with `C`) is at the bottom. DM IDs start with `D`.

### Mode comparison

| mode | Connection direction | Public URL | Use case |
|------|----------------------|------------|----------|
| `socket` | Server → Slack (WebSocket) | Not required | Server behind NAT (recommended) |
| `webhook` | Slack → Server (HTTP POST) | Required | Public server |

### Per-Anima bot (Socket Mode)

You can configure a dedicated Slack bot per Anima. Adding `SLACK_BOT_TOKEN__{anima_name}` and `SLACK_APP_TOKEN__{anima_name}` to vault or `shared/credentials.json` starts a Socket Mode connection for that Anima only. Per-Anima bots do not need channel mapping; all messages go to that Anima’s inbox.

```json
{
  "SLACK_BOT_TOKEN__sakura": "xoxb-...",
  "SLACK_APP_TOKEN__sakura": "xapp-..."
}
```

The list of Per-Anima bots is determined by scanning `SLACK_BOT_TOKEN__*` keys in vault / `credentials.json` (`server/slack_socket.SlackSocketModeManager._discover_per_anima_bots`).

Per-Anima bots and the shared bot (`SLACK_BOT_TOKEN` / `SLACK_APP_TOKEN`) can be used together. The shared bot routes by channel using `anima_mapping` and `default_anima`.

### Additional Webhook mode configuration

When `mode: "webhook"`, you need:

1. **Request URL**: In the Slack app’s Event Subscriptions, set `https://your-server/api/webhooks/slack/events`
2. **Signing secret**: Set `SLACK_SIGNING_SECRET` in `shared/credentials.json` or an environment variable

```json
{
  "SLACK_BOT_TOKEN": "xoxb-...",
  "SLACK_SIGNING_SECRET": "Signing Secret from the Slack admin UI"
}
```

| Key | Purpose |
|-----|---------|
| `SLACK_SIGNING_SECRET` | Verify webhook request signatures (replay protection) |

The Signing Secret is under **Basic Information** → **App Credentials** in the Slack app admin UI.

**Multiple Slack Apps (app_id_mapping)**: Create a dedicated Slack app per Anima and set `app_id_mapping` to `api_app_id → Anima name`. The api_app_id is under **Basic Information** in the Slack app admin UI (ID starting with `A`). In that case use Per-Anima signing secret `SLACK_SIGNING_SECRET__{anima_name}`.

## 4. Server restart and Slack connection updates

After first setup or major configuration changes:

```bash
animaworks start
```

If the startup log shows the following, the connection succeeded:

```
INFO  animaworks.slack_socket: Shared Slack bot registered (bot_uid=U...)
INFO  animaworks.slack_socket: Slack Socket Mode connected (1 handler(s))
```

With Per-Anima bots configured, you should also see `Per-Anima Slack bot registered: {name} (bot_uid=U...)`.

**When there is no shared token and only Per-Anima bots**, operation continues with Per-Anima connections only after `Shared Slack bot not configured; per-Anima bots only` even if shared bot registration fails.

When disabled, or when `mode: "webhook"`:

```
INFO  animaworks.slack_socket: Slack Socket Mode is disabled
```

In Webhook mode, Socket Mode does not start; events are received at the HTTP endpoint `/api/webhooks/slack/events`.

### Hot-reload Socket handlers without full restart

After adding/removing Per-Anima `SLACK_*__name` keys or updating credentials, you can rebuild only the Socket Mode handlers via the API (authenticated server):

- `POST /api/system/hot-reload/slack` — Slack only
- `POST /api/system/hot-reload` — config, credentials, Slack, and Animas together

Implementation: `server/reload_manager.py` → `SlackSocketModeManager.reload()` (diff for add/remove, keep existing connections while `connect_async` for new Per-Anima bots).

## Message flow

1. A Slack user sends a message in a mapped channel
2. Slack delivers the event via WebSocket (Socket Mode) or HTTP POST (Webhook)
3. **Deduplication**: The same message can produce both `message` and `app_mention` events. `ts` is recorded with a short TTL (~10 s); the second delivery is ignored (`_is_duplicate_ts` in `server/slack_socket.py`)
4. **call_human thread replies**: If the message is a thread reply and mapped via `route_thread_reply`, it is routed to the inbox of the Anima that sent the original notification (`core/notification/reply_routing.py`)
5. **Routing resolution**:
   - **Socket Mode Per-Anima bot**: All messages to that bot go directly to the corresponding Anima
   - **Socket Mode shared bot**: On each message, `anima_mapping.get(channel_id) or default_anima` (config hot-reloads)
   - **Webhook**: Resolve Anima with `app_id_mapping.get(api_app_id)` → otherwise `anima_mapping.get(channel_id) or default_anima`
6. **Thread context**: When `thread_ts` is present, a one-line parent summary and reply count are prepended as a `[Thread context]` block (`conversations.replies`, logic equivalent to fetching up to ~10 items)
7. **Body normalization**: Expand `<@U...>` to display names and convert Slack markup to plain text (`clean_slack_markup` in `core/tools/_slack_markdown.py`)
8. **Annotation**: Prepend a line indicating `[slack:DM]` or whether the channel message mentions the bot or an alias (`_build_slack_annotation`)
9. **intent**: Mentions of the bot `<@BOT>` or of a `slack_user_id` from `user_aliases` → `question`. Otherwise only DMs get `question` (`_detect_mention_intent` / `_detect_slack_intent`)
10. `Messenger.receive_external()` writes the message to `~/.animaworks/shared/inbox/{anima_name}/{msg_id}.json`
11. The Anima processes the inbox on its next run cycle (heartbeat / cron / manual)

## Slack tools (relationship to `core/tools/slack.py`)

Ingest (Socket/Webhook) is separate from **Slack Web API** calls: the entry point is `core/tools/slack.py`, with implementation split across `_slack_client.py`, `_slack_cache.py`, `_slack_markdown.py`, and `_slack_cli.py`.

### Schemas (`get_tool_schemas()`) and `dispatch()` responsibilities

- **`get_tool_schemas()`** returns only **`slack_channel_post`** and **`slack_channel_update`** (both gated actions; see permissions below).
- The following schema names are handled by **`dispatch(name, args)`** (e.g. `slack_send`). Agents mainly invoke these via **`use_tool`** with `tool_name: "slack"` and `action: "send"` etc., which maps to the `slack_send` path. **`ExternalToolDispatcher`** matches only names exposed by the module’s `get_tool_schemas()` through the registry, so direct `slack_send` calls do not go through the schema path—note that **`use_tool`** calls the module’s `dispatch` directly.

| Schema name (argument to `dispatch`) | Summary |
|-----------------------------------|---------|
| `slack_send` | Post to channel name or ID; optional `thread_ts`. Body via `md_to_slack_mrkdwn`. Can attach display name and icon via `resolve_anima_icon_identity` (`username` / `icon_url`) |
| `slack_messages` | Fetch channel history; upsert into SQLite cache and return from cache |
| `slack_search` | Keyword search in cache (optional channel filter) |
| `slack_unreplied` | Detect unreplied threads (cache-based; runs `auth_test` first) |
| `slack_channels` | List joined channels |
| `slack_react` | Add reaction (`channel`, `emoji`, `message_ts`) |
| `slack_channel_post` | Post by channel ID; body via `md_to_slack_mrkdwn`. Same display name/icon behavior as `slack_send`. Return value includes `ts` (for `slack_channel_update`) |
| `slack_channel_update` | Silent update of an existing message (`channel_id`, `ts`, `text`). Body only via `md_to_slack_mrkdwn` (does not override icon from the original post) |

### Token resolution

- **`_resolve_slack_token(args)`** looks up **`SLACK_BOT_TOKEN__{anima_name}`** in vault → `shared/credentials.json`, then falls back to the **`SlackClient` default** (`get_credential("slack", "slack", env_var="SLACK_BOT_TOKEN")`, i.e. the shared token).
- This is separate from **`SLACK_APP_TOKEN`** for Socket Mode. Sending, history, search, and other Web APIs need only the **Bot User Token (`xoxb-`)**.

### Gating and `EXECUTION_PROFILE`

`slack_channel_post` / `slack_channel_update` are **`gated: True`** in **`EXECUTION_PROFILE`** in `slack.py`. Without the following in **permissions.md** (or equivalent permission config), they are blocked:

- `slack_channel_post: yes`
- `slack_channel_update: yes`

`all: yes` alone does **not** open these gates (action-level grants are required, same as other tools).

### Message cache

Default directory: `~/.animaworks/cache/slack/` (`MessageCache` / SQLite).

### CLI

Subcommands from **`get_cli_guide()`** in `core/tools/_slack_cli.py`: **`channels`**, **`messages`**, **`send`**, **`search`**, **`unreplied`**. In the CLI, when **`ANIMAWORKS_ANIMA_DIR`** is set, **`SLACK_BOT_TOKEN__{name}`** is preferred per Anima. Display name and icon attachment apply to **`send` only** (`messages` etc. resolve the token only). **`react`**, fixed channel-ID post, and **update** are **not** in the CLI—use agent tools `slack_react`, `slack_channel_post`, and `slack_channel_update` (or `use_tool`).

Entry: `python -m core.tools.slack` or `animaworks-tool slack` (see `--help`).

## Related files

| File | Role |
|------|------|
| `server/slack_socket.py` | `SlackSocketModeManager` (Socket Mode, Per-Anima / shared bot, deduplication, thread context, mention/intent) |
| `server/app.py` (near lifespan) | Start/stop `SlackSocketModeManager` |
| `server/reload_manager.py` | `SlackSocketModeManager.reload()` when applying config/credentials |
| `server/routes/system.py` | `/api/system/hot-reload*` endpoints |
| `server/routes/webhooks.py` | Webhook (`/api/webhooks/slack/events`, signature verification; same shaping and intent as above) |
| `core/messenger.py` | `receive_external()` — inbox placement |
| `core/tooling/handler.py` | `use_tool` → direct `dispatch` on external modules (e.g. `slack` + `action` runs `slack_*`) |
| `core/tooling/dispatch.py` | `ExternalToolDispatcher` (core tools: registry matches only schema names from `get_tool_schemas()`) |
| `core/notification/reply_routing.py` | Route call_human thread replies to Animas |
| `core/config/schemas.py` | `UserAliasConfig`, `ExternalMessagingChannelConfig`, `ExternalMessagingConfig` |
| `core/tools/slack.py` | `get_tool_schemas` (channel_post/update only), `dispatch`, `EXECUTION_PROFILE`, re-exports |
| `core/tools/_anima_icon_url.py` | `resolve_anima_icon_identity` (display name/icon for `slack_send` / `slack_channel_post`) |
| `core/tools/_slack_client.py` | `SlackClient` (Web API, paging, 429 retries) |
| `core/tools/_slack_cache.py` | `MessageCache` (SQLite) |

## Troubleshooting

### Cannot connect

- Confirm `SLACK_APP_TOKEN` starts with `xapp-`
- Confirm Socket Mode is enabled in the Slack app settings
- Confirm Event Subscriptions are enabled
- Run `pip show slack-bolt aiohttp` to verify Socket dependencies are installed

### Messages are not received

- Verify channel IDs in `anima_mapping`
- If `default_anima` is set, confirm the fallback Anima is enabled
- Confirm the bot is invited to the target channel
- Check logs for `"No anima mapping for channel"` (Socket Mode) or `"No anima mapping for Slack channel %s and no default_anima"` (Webhook)

### Webhook mode signature error (400 Invalid signature)

- Confirm `SLACK_SIGNING_SECRET` (shared) or `SLACK_SIGNING_SECRET__{anima_name}` (with app_id_mapping) is set
- Confirm it matches the Signing Secret under **Basic Information** → **App Credentials**
- Check logs for `"SLACK_SIGNING_SECRET not configured"`

### ImportError in tools (slack-sdk)

- `pip install "animaworks[communication]"` or `pip install slack-sdk`

### Reconnection

- `slack-bolt`’s `AsyncSocketModeHandler` supports automatic reconnection
- WebSocket connections are refreshed on a ~1 hour cadence
- If rate limiting (429) appears during long runs, restarting the server can clear it
- To add Per-Anima bots only, use `POST /api/system/hot-reload/slack` to avoid a full restart

## Limitations

- Socket Mode apps cannot be listed in the Slack App Directory (internal tools)
- Maximum concurrent WebSocket connections: 10 per app
- `apps.connections.open` rate limit: 1 per minute
- Processing of messages placed in the inbox depends on the Anima’s next run cycle
