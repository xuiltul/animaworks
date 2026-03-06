# AnimaWorks REST API Reference

**[日本語版](api-reference.ja.md)**

> Last updated: 2026-03-06
> Base URL: `http://localhost:18500`
> See also: [spec.md](spec.md), [cli-reference.md](cli-reference.md)

---

## Authentication

| Method | Description |
|--------|-------------|
| Session cookie | `session_token` cookie obtained via `POST /api/auth/login` |
| localhost trust | When `trust_localhost: true` (default), requests from localhost skip auth |
| local_trust mode | When `auth.json` has `mode: "local_trust"`, all requests skip auth |

Unauthenticated endpoints: `/api/auth/login`, `/api/setup/*`, `/api/system/health`, `/api/webhooks/*`

---

## Table of Contents

1. [Authentication (Auth)](#1-authentication-auth)
2. [Anima Management](#2-anima-management)
3. [Chat](#3-chat)
4. [Shared Channels & DM](#4-shared-channels--dm)
5. [Memory](#5-memory)
6. [Sessions & History](#6-sessions--history)
7. [System](#7-system)
8. [Assets](#8-assets)
9. [Config](#9-config)
10. [Logs](#10-logs)
11. [User Management](#11-user-management)
12. [Tool Prompts](#12-tool-prompts)
13. [Setup](#13-setup)
14. [Webhooks](#14-webhooks)
15. [WebSocket](#15-websocket)
16. [Internal API](#16-internal-api)

---

## 1. Authentication (Auth)

### POST `/api/auth/login`

Log in and obtain a session cookie.

**Request:**

```json
{ "username": "string", "password": "string" }
```

**Response:** `200 OK`

```json
{ "username": "admin", "display_name": "Admin", "role": "owner" }
```

Set-Cookie: `session_token=...`

---

### POST `/api/auth/logout`

Destroy the current session.

**Response:** `200 OK` — `{ "status": "ok" }`

---

### GET `/api/auth/me`

Get the current authenticated user.

**Response:**

```json
{
  "username": "admin",
  "display_name": "Admin",
  "bio": "",
  "role": "owner",
  "auth_mode": "password",
  "has_password": true
}
```

---

## 2. Anima Management

### GET `/api/animas`

List all Animas.

**Response:**

```json
[
  {
    "name": "alice",
    "status": "running",
    "bootstrapping": false,
    "pid": 12345,
    "uptime_sec": 3600,
    "appearance": "anime",
    "supervisor": null,
    "speciality": "engineer",
    "role": "engineer",
    "model": "claude-opus-4-6"
  }
]
```

---

### GET `/api/animas/{name}`

Get detailed information about an Anima.

| Path Parameter | Type | Description |
|----------------|------|-------------|
| `name` | string | Anima name |

**Response:**

```json
{
  "status": { "enabled": true, "model": "claude-opus-4-6" },
  "identity": "# Alice\n...",
  "injection": "## Role\n...",
  "state": "Current task content",
  "pending": "Backlog content",
  "knowledge_files": ["topic1.md"],
  "episode_files": ["2026-03-05.md"],
  "procedure_files": ["deploy.md"]
}
```

---

### GET `/api/animas/{name}/config`

Get resolved model configuration for an Anima.

**Response:**

```json
{
  "anima": "alice",
  "model": "claude-opus-4-6",
  "execution_mode": "S",
  "config": { "max_turns": 200, "max_chains": 10 }
}
```

---

### POST `/api/animas/{name}/enable`

Enable an Anima.

**Response:** `200 OK` — `{ "name": "alice", "enabled": true }`

---

### POST `/api/animas/{name}/disable`

Disable an Anima (process stops in ~30 seconds).

**Response:** `200 OK` — `{ "name": "alice", "enabled": false }`

---

### POST `/api/animas/{name}/start`

Start an Anima process.

**Response:** `200 OK` — `{ "status": "started", "name": "alice" }`

If already running: `{ "status": "already_running", "current_status": "..." }`

---

### POST `/api/animas/{name}/stop`

Stop an Anima process.

**Response:** `200 OK` — `{ "status": "stopped", "name": "alice" }`

---

### POST `/api/animas/{name}/restart`

Restart an Anima process.

**Response:** `200 OK` — `{ "status": "restarted", "name": "alice", "pid": 12346 }`

---

### POST `/api/animas/{name}/interrupt`

Interrupt the running LLM session.

| Query Parameter | Type | Default | Description |
|----------------|------|---------|-------------|
| `thread_id` | string | null | Target thread ID |

**Response:** `200 OK` — IPC result or `{ "status": "timeout" }`

---

### POST `/api/animas/{name}/reload`

Hot-reload status.json without process restart.

**Response:** `200 OK` — IPC result

---

### POST `/api/animas/reload-all`

Hot-reload configuration for all Animas.

**Response:** `200 OK` — `{ "status": "ok", "results": { "alice": "..." } }`

---

### POST `/api/animas/{name}/trigger`

Manually trigger a heartbeat.

**Response:** `200 OK` — IPC result | `504` timeout

---

### GET `/api/animas/{name}/background-tasks`

List background tasks (async tool executions via submit).

**Response:** `200 OK` — `{ "tasks": [{ "task_id": "...", "status": "..." }] }`

---

### GET `/api/animas/{name}/background-tasks/{task_id}`

Get a specific background task's details.

---

### GET `/api/org/chart`

Get the organization chart (hierarchy).

| Query Parameter | Type | Default | Description |
|----------------|------|---------|-------------|
| `include_disabled` | bool | false | Include disabled Animas |
| `format` | string | "json" | `json` or `text` (ASCII tree) |

---

## 3. Chat

### POST `/api/animas/{name}/chat`

Send a non-streaming chat message.

**Request:**

```json
{
  "message": "Hello",
  "from_person": "human",
  "intent": "",
  "images": [{ "data": "base64...", "media_type": "image/png" }],
  "resume": null,
  "last_event_id": null,
  "thread_id": "default"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `message` | string | Required | Message body |
| `from_person` | string | Optional | Sender name (default: "human") |
| `intent` | string | Optional | Intent (delegation, report, question) |
| `images` | array | Optional | Image attachments (base64 + media_type) |
| `resume` | string | Optional | Session resume ID |
| `last_event_id` | string | Optional | SSE reconnection event ID |
| `thread_id` | string | Optional | Thread ID (default: "default") |

**Response:**

```json
{ "response": "Hello! How can I help?", "anima": "alice", "images": [] }
```

---

### POST `/api/animas/{name}/chat/stream`

Streaming chat via Server-Sent Events (SSE).

**Request:** Same as `POST /api/animas/{name}/chat`

**Response:** `text/event-stream`

```
event: text
data: {"text": "Hel", "emotion": "happy"}

event: tool_start
data: {"tool": "search_memory", "input": {"query": "..."}}

event: tool_end
data: {"tool": "search_memory", "result": "..."}

event: tool_detail
data: {"tool": "search_memory", "status": "running", "summary": "..."}

event: thinking
data: {"text": "Thinking..."}

event: done
data: {"response_id": "abc123", "emotion": "neutral"}

event: error
data: {"error": "Error message"}
```

---

### POST `/api/animas/{name}/greet`

Generate a greeting (without starting a conversation).

**Response:**

```json
{ "response": "Good morning!", "emotion": "happy", "cached": false, "anima": "alice" }
```

---

### GET `/api/animas/{name}/stream/active`

Get the currently active stream state.

| Query Parameter | Type | Default | Description |
|----------------|------|---------|-------------|
| `thread_id` | string | null | Thread ID |

---

### GET `/api/animas/{name}/stream/{response_id}/progress`

Get progress for a specific stream.

---

## 4. Shared Channels & DM

### GET `/api/channels`

List shared channels.

**Response:**

```json
[{ "name": "general", "message_count": 42, "last_post_ts": "2026-03-06T09:00:00Z" }]
```

---

### GET `/api/channels/{name}`

Get channel messages.

| Query Parameter | Type | Default | Description |
|----------------|------|---------|-------------|
| `limit` | int | 50 | Max messages |
| `offset` | int | 0 | Offset |

**Response:**

```json
{
  "channel": "general",
  "messages": [{ "from": "alice", "text": "Report", "ts": "..." }],
  "total": 100,
  "offset": 0,
  "limit": 50,
  "has_more": true
}
```

---

### POST `/api/channels/{name}`

Post a message to a channel.

**Request:** `{ "text": "Announcement", "from_name": "alice" }`

**Response:** `200 OK` — `{ "status": "ok", "channel": "general" }`

---

### GET `/api/channels/{name}/mentions/{anima}`

Get mentions for a specific Anima.

| Query Parameter | Type | Default | Description |
|----------------|------|---------|-------------|
| `limit` | int | 10 | Max results |

---

### GET `/api/dm`

List DM conversation pairs.

---

### GET `/api/dm/{pair}`

Get DM history (`pair` format: `alice-bob`).

| Query Parameter | Type | Default | Description |
|----------------|------|---------|-------------|
| `limit` | int | 50 | Max messages |

---

## 5. Memory

### GET `/api/animas/{name}/episodes`

List episode memory files.

**Response:** `{ "files": ["2026-03-05.md", "2026-03-04.md"] }`

---

### GET `/api/animas/{name}/episodes/{date}`

Get a specific episode.

**Response:** `{ "date": "2026-03-05", "content": "# 2026-03-05\n..." }`

---

### GET `/api/animas/{name}/knowledge`

List knowledge memory files.

### GET `/api/animas/{name}/knowledge/{topic}`

Get a specific knowledge entry.

### GET `/api/animas/{name}/procedures`

List procedure memory files.

### GET `/api/animas/{name}/procedures/{proc}`

Get a specific procedure.

---

### GET `/api/animas/{name}/conversation`

Get current conversation state.

**Response:**

```json
{
  "anima": "alice",
  "total_turn_count": 24,
  "raw_turns": 24,
  "compressed_turn_count": 5,
  "has_summary": true,
  "summary_preview": "Summary of conversation so far...",
  "total_token_estimate": 15000,
  "turns": [{ "role": "user", "content": "..." }]
}
```

---

### DELETE `/api/animas/{name}/conversation`

Clear conversation history.

**Response:** `{ "status": "cleared", "anima": "alice" }`

---

### POST `/api/animas/{name}/conversation/compress`

Compress conversation history.

---

### GET `/api/animas/{name}/memory/stats`

Get memory statistics.

**Response:**

```json
{
  "anima": "alice",
  "episodes": { "count": 20, "total_bytes": 45000 },
  "knowledge": { "count": 15, "total_bytes": 32000 },
  "procedures": { "count": 5, "total_bytes": 8000 }
}
```

---

## 6. Sessions & History

### GET `/api/animas/{name}/sessions`

List sessions (active conversation, archived sessions, threads).

### GET `/api/animas/{name}/conversation/history`

Get activity-log-based conversation history.

| Query Parameter | Type | Default | Description |
|----------------|------|---------|-------------|
| `limit` | int | 50 | Max entries |
| `before` | string | null | Get entries before this timestamp |
| `thread_id` | string | "default" | Thread ID |
| `strict_thread` | bool | false | Strict thread matching |

### GET `/api/animas/{name}/sessions/{session_id}`

Get archived session detail.

### GET `/api/animas/{name}/transcripts/{date}`

Get transcript for a date.

---

## 7. System

### GET `/api/system/health`

Health check (**no authentication required**).

**Response:** `200 OK` — `{ "status": "ok" }`

---

### GET `/api/system/status`

Get overall system status.

### GET `/api/system/connections`

Get WebSocket and process connection info.

### GET `/api/system/scheduler`

Get scheduler (heartbeat/cron) status.

### POST `/api/system/reload`

Reload Anima processes (detect additions/changes).

**Response:** `{ "added": 1, "refreshed": 2, "skipped_busy": 0, "removed": 0, "total": 5 }`

---

### GET `/api/activity/recent`

Get recent activity across all Animas.

| Query Parameter | Type | Default | Description |
|----------------|------|---------|-------------|
| `hours` | int | 48 | Time window (hours) |
| `anima` | string | null | Filter by Anima name |
| `offset` | int | 0 | Offset |
| `limit` | int | 200 | Max entries |
| `event_type` | string | null | Filter by event type |
| `grouped` | bool | false | Group by trigger |
| `group_limit` | int | 50 | Max groups |
| `group_offset` | int | 0 | Group offset |

---

### GET `/api/system/cost`

Get token usage and cost estimates.

| Query Parameter | Type | Default | Description |
|----------------|------|---------|-------------|
| `anima` | string | null | Specific Anima only |
| `days` | int | 30 | Number of days |

---

### POST `/api/system/hot-reload`

Hot-reload all configuration (config, credentials, Slack, Anima processes).

### POST `/api/system/hot-reload/slack`

Reload Slack connections only.

### POST `/api/system/hot-reload/credentials`

Reload credentials only.

### POST `/api/system/hot-reload/animas`

Reload Anima processes only.

---

### GET/POST `/api/system/log-level`

Get or set server log level.

**POST Request:** `{ "level": "DEBUG", "logger_name": "core.memory" }`

---

### POST `/api/settings/display-mode`

Change display mode.

**Request:** `{ "mode": "anime" }` or `{ "mode": "realistic" }`

---

### POST/GET `/api/system/frontend-logs`

Submit or view frontend logs.

---

## 8. Assets

### GET `/api/animas/{name}/assets`

List asset files.

### GET `/api/animas/{name}/assets/metadata`

Get asset metadata (expressions, animations, colors).

### GET `/api/animas/{name}/assets/{filename}`

Get an asset file (ETag / 304 support).

### GET `/api/animas/{name}/attachments/{filename}`

Get a chat attachment file.

### GET `/api/media/proxy`

Proxy external images (SSRF-protected).

| Query Parameter | Type | Description |
|----------------|------|-------------|
| `url` | string | Image URL to proxy |

---

### POST `/api/animas/{name}/assets/generate`

Generate character assets.

**Request:**

```json
{
  "prompt": "Custom prompt",
  "negative_prompt": "Exclusions",
  "steps": 28,
  "skip_existing": true,
  "image_style": "anime"
}
```

### POST `/api/animas/{name}/assets/generate-expression`

Generate an expression asset on demand.

**Request:** `{ "expression": "angry", "image_style": "anime" }`

### POST `/api/animas/{name}/assets/remake-preview`

Generate a remake preview (Vibe Transfer).

**Request:**

```json
{
  "style_from": "bob",
  "vibe_strength": 0.6,
  "vibe_info_extracted": 0.8,
  "image_style": "anime"
}
```

### POST `/api/animas/{name}/assets/remake-confirm`

Confirm a remake and replace existing assets.

### DELETE `/api/animas/{name}/assets/remake-preview`

Cancel a remake preview and restore backup.

---

## 9. Config

### GET `/api/system/config`

Get config.json contents (API keys masked).

### GET `/api/system/init-status`

Check initialization status.

---

## 10. Logs

### GET `/api/system/logs`

List log files.

### GET `/api/system/logs/stream`

Stream logs via SSE.

| Query Parameter | Type | Default | Description |
|----------------|------|---------|-------------|
| `file` | string | "animaworks.log" | Log filename |

### GET `/api/system/logs/{filename}`

Read log file contents.

| Query Parameter | Type | Default | Description |
|----------------|------|---------|-------------|
| `offset` | int | 0 | Start line |
| `limit` | int | 200 | Max lines |

### GET `/api/system/logs/file/read`

Read log file by path reference.

---

## 11. User Management

### GET `/api/users`

List users.

### POST `/api/users`

Add a user (owner only).

**Request:** `{ "username": "user1", "display_name": "User 1", "password": "pass", "bio": "" }`

### DELETE `/api/users/{username}`

Delete a user (owner only).

### PUT `/api/users/me/password`

Change own password.

**Request:** `{ "current_password": "old", "new_password": "new" }`

---

## 12. Tool Prompts

APIs for managing Anima tool descriptions, guides, and system prompt sections.

### GET `/api/tool-prompts/descriptions`

List all tool descriptions.

### GET/PUT `/api/tool-prompts/descriptions/{name}`

Get or update a tool description.

### GET `/api/tool-prompts/guides`

List all guides.

### GET/PUT `/api/tool-prompts/guides/{key}`

Get or update a guide.

### GET `/api/tool-prompts/sections`

List all system prompt sections.

### GET/PUT `/api/tool-prompts/sections/{key}`

Get or update a section.

### POST `/api/tool-prompts/preview/schema`

Preview tool schema.

**Request:** `{ "mode": "anthropic" }` — one of `anthropic`, `litellm`, `text`

### POST `/api/tool-prompts/preview/system-prompt`

Preview system prompt for an Anima.

**Request:** `{ "anima_name": "alice" }`

---

## 13. Setup

First-time setup APIs (**all unauthenticated**).

### GET `/api/setup/environment`

Get setup environment info.

**Response:**

```json
{
  "claude_code_available": true,
  "locale": "ja",
  "providers": ["anthropic", "openai"],
  "available_locales": ["ja", "en"]
}
```

### GET `/api/setup/detect-locale`

Detect locale from Accept-Language header.

### POST `/api/setup/validate-key`

Validate an API key.

**Request:** `{ "provider": "anthropic", "api_key": "sk-..." }`

### POST `/api/setup/complete`

Complete the setup process.

---

## 14. Webhooks

Receive events from external platforms (**no auth required; signature verification**).

### POST `/api/webhooks/slack/events`

Receive Slack Event API events (URL verification challenge supported).

### POST `/api/webhooks/chatwork`

Receive Chatwork webhook events.

---

## 15. WebSocket

### `ws://HOST:PORT/ws`

Main WebSocket for real-time dashboard and chat UI updates.

**Server → Client events:**

| Event | Description |
|-------|-------------|
| `anima_status` | Anima status change |
| `chat_response` | Chat response text |
| `tool_activity` | Tool usage updates |
| `heartbeat` | Periodic keep-alive (ping) |

**Client → Server:**

| Message | Description |
|---------|-------------|
| `{ "type": "pong" }` | Heartbeat response |

---

### `ws://HOST:PORT/ws/voice/{name}`

Voice chat WebSocket.

**Authentication:** Send `{ "type": "auth", "token": "SESSION_TOKEN" }` after connection (not required in localhost trust mode).

**Client → Server:**

| Format | Description |
|--------|-------------|
| binary | 16kHz mono 16-bit PCM audio data |
| `{ "type": "speech_end" }` | End of speech (triggers STT) |
| `{ "type": "interrupt" }` | Interrupt TTS playback (barge-in) |
| `{ "type": "config", ... }` | Configuration change |

**Server → Client:**

| Type | Format | Description |
|------|--------|-------------|
| `status` | JSON | Session state change |
| `transcript` | JSON | STT result text |
| `response_text` | JSON | Anima response text (chunked) |
| `tts_audio` | binary | TTS audio data |
| `tts_start` / `tts_done` | JSON | TTS start/end notification |
| `error` | JSON | Error notification |

---

## 16. Internal API

### POST `/api/internal/message-sent`

Notify about a message sent via CLI (for UI updates).

**Request:**

```json
{ "from_person": "alice", "to_person": "bob", "content": "Report", "message_id": "msg_123" }
```

### GET `/api/messages/{message_id}`

Get a saved message.

---

## Chat UI State

### GET `/api/chat/ui-state`

Get chat UI pane/tab state.

### PUT `/api/chat/ui-state`

Save chat UI state.

**Request:** `{ "state": { "version": 1, "active_anima": "alice" } }`
