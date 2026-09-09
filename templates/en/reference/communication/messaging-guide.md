# Complete Messaging Guide

Comprehensive guide for communicating with other Anima (team members).
Covers all procedures for sending, receiving, and managing message threads.

## send_message Tool — Parameter Reference

Use the `send_message` tool for sending messages (recommended).

### Parameter List

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `to` | string | MUST | Recipient. Resolution rules are under “Resolving `to` (unified outbound)” below. You can use Anima names, human aliases from `config.json`, `slack:USERID` / `chatwork:ROOMID`, or a bare Slack user ID (`U` followed by 8+ alphanumeric characters) |
| `content` | string | MUST | Message body |
| `intent` | string | MUST | Message intent. Allowed values only: `report` (progress/result report), `question` (inquiry requiring a reply). Use the `delegate_task` tool for task delegation. Use Board (`post_channel`) for acknowledgments, thanks, and FYI |
| `reply_to` | string | MAY | ID of the message being replied to (e.g. `20260215_093000_123456`) |
| `thread_id` | string | MAY | Thread ID. Specify when joining an existing thread |

### DM Limits (per run)

- Maximum **2 recipients** per run
- **No second message** to the same recipient (use Board for additional contact)
- Use Board (`post_channel`) for communication to 3 or more people

### Resolving `to` (unified outbound)

Recipients for `send_message` are resolved by `core/outbound.resolve_recipient` in the following **priority order** (strongest against spelling/casing drift first).

- **Leading and trailing whitespace** on the recipient string is trimmed before resolution.

| Priority | Condition | Result |
|----------|-----------|--------|
| 1 | **Exact** match with an existing Anima directory name (case-sensitive) | Internal Inbox |
| 2 | Key **matches** `external_messaging.user_aliases` in `config.json` (case-insensitive) | External (Slack or Chatwork). Uses `preferred_channel` first; if that side has no contact, falls back to the other configured channel |
| 3 | Starts with `slack:` (case-insensitive for the prefix; e.g. `slack:U0123456789`, `Slack:u06…`) | After the colon, trim; **normalize** the user ID to **uppercase**, then Slack DM |
| 4 | Starts with `chatwork:` (case-insensitive for the prefix) | Trim after the colon; Chatwork post to that room ID (room ID casing preserved) |
| 5 | **Bare Slack user ID**: starts with `U`, then **8+** alphanumeric characters (matches `^U[A-Z0-9]{8,}$`) | Slack DM (when you want ID only, no prefix). **Too-short strings** (e.g. `U12345`) do not match here and fall through to lower rules |
| 6 | **Case-insensitive** match with an existing Anima name | Internal Inbox (delivered under the canonical on-disk name) |
| 7 | None of the above | Unknown recipient (`RecipientNotFoundError`; low-level messages may mention known Anima or alias names) |

**Conflict between Anima name and alias**: If a string matches both a `user_aliases` key and an **exact** existing Anima directory name, **priority 1** always wins and delivery goes to the **internal Inbox** (the alias is not used).

#### When `send_message` recipient resolution fails

Even when `resolve_recipient` fails, the tool result does not return the raw exception. **Guidance** tailored to the session type is returned instead (`core/tooling/handler_comms.py`).

- **Human chat** (`chat`): Explains that `send_message` cannot target that recipient, that **replying in plain text reaches the human user**, and that `send_message` is for other Anima.
- **Non-chat** (heartbeat, cron, etc.): Explains that **`call_human`** should be used to reach humans, and `send_message` is for other Anima.

So low-level text like “Known animas: …” often does not appear to tool users. To fix configuration or aliases, check `external_messaging` in `config.json`.

#### Human aliases and `preferred_channel`

Each entry in `external_messaging.user_aliases` should set `slack_user_id` and/or `chatwork_room_id`. When `external_messaging.preferred_channel` is `slack`, Slack is chosen if a Slack ID exists; if the Slack ID is empty but Chatwork is set, delivery falls back to Chatwork (and vice versa when `preferred_channel` is `chatwork`). An alias with **neither** ID set fails resolution with an error.

#### Delivery after external routing (overview)

When resolved to an external route, `send_message` sends via API from `core/outbound.send_external` without going through the internal Messenger.

- **Try order**: First send on the resolved `channel` (`slack` or `chatwork`); on failure, follow `_build_channel_order` — if the peer has **both** Slack and Chatwork IDs, the other channel is tried in turn.
- **Slack**: Token used by `core/outbound._send_via_slack` is **`SLACK_BOT_TOKEN__{sending_anima_name}`** from Vault / shared credentials when present. Body is formatted with `md_to_slack_mrkdwn`. If `core/tools._anima_icon_url` resolves an icon URL, it is passed to `post_message`.
  - **`[Sender name]` prefix on the body**: Only when there is **no** per-Anima bot token, the body is prefixed with `[{sending_anima_name}] ` so the DM shows who sent it. **When a token exists**, no such prefix is added; sender is shown via `username` (Anima name) and `icon_url`.
- **Chatwork**: Post to the room via the sending Anima's own identity token (`CHATWORK_API_TOKEN__<anima_name>`). If a sending Anima name is known, prefix the body with `[Sender name] ` and format with `md_to_chatwork`.

### Basic Send Example

```
send_message(to="alice", content="Review complete. Three items to fix.", intent="report")
```

### Reply Example

Link your reply using the received message's `id` and `thread_id`:

```
send_message(
    to="alice",
    content="Understood. I'll respond by 3pm.",
    intent="report",
    reply_to="20260215_093000_123456",
    thread_id="20260215_090000_000000"
)
```

Aliases registered in `user_aliases` (e.g. `user`) can be used with `send_message` like internal Anima (routed to external channels):

```
send_message(to="user", content="Follow-up is done.", intent="report")
```

### When to Use Each intent

| intent | Use case | Example |
|--------|----------|---------|
| `report` | Progress or result report | Task completion report, status update to supervisor |
| `question` | Question requiring a reply | Clarification, consultation requiring a decision |

**Note**: Acknowledgments, thanks, and FYI (e.g. “Understood”, “Thank you”) cannot be sent via DM. Use Board (`post_channel`).

### Board vs DM Usage

| Use case | Tool | Example |
|----------|------|---------|
| Progress/result report | send_message (intent=report) | Task completion report to supervisor |
| Task delegation | delegate_task | Assign one durable task to a direct subordinate; inspect progress with task_tracker |
| Question/inquiry | send_message (intent=question) | Clarification request |
| Acknowledgment/thanks/FYI | post_channel (Board) | “Understood”, “Shared” |
| Communication to 3+ people | post_channel (Board) | Team-wide announcement |
| Second message to same recipient | post_channel (Board) | Sharing additional information |

## Thread Management

### Starting a New Thread

Omitting `thread_id` lets the system automatically set the message ID as the thread ID.
Do not specify `thread_id` when starting a new topic.

```
send_message(to="bob", content="I'd like to discuss the new project.", intent="question")
# → thread_id is auto-generated (same as message ID)
```

### Replying in an Existing Thread

When replying to a received message, MUST specify both `reply_to` and `thread_id`.

```
# Received message:
#   id: "20260215_093000_123456"
#   thread_id: "20260215_090000_000000"
#   content: "Please review"

send_message(
    to="alice",
    content="Review complete.",
    intent="report",
    reply_to="20260215_093000_123456",
    thread_id="20260215_090000_000000"
)
```

### Thread Management Rules

- MUST: Keep using the same `thread_id` for the same topic
- MUST: Set the original message's `id` to `reply_to` when replying
- SHOULD NOT: Mix different topics in an existing thread. Start a new thread for new topics
- MAY: Omit `thread_id` if unknown (treated as a new thread)

## Sending Messages via CLI

For when tools are unavailable or sending via Bash.

### Basic Syntax

```bash
animaworks send {sender_name} {recipient} "message content" [--intent report|question] [--reply-to ID] [--thread-id ID]
```

### Examples

```bash
# Basic send (intent optional; CLI allows sending without it)
animaworks send bob alice "Work complete. Please confirm." --intent report

# Thread reply
animaworks send bob alice "Understood" --intent report --reply-to 20260215_093000_123456 --thread-id 20260215_090000_000000
```

### Notes

- MUST: Wrap message content in double quotes
- SHOULD: Prefer the `send_message` tool when available (more reliable than CLI)
- Escape `"` in messages: `\"`

## Checking Received Messages

### Auto Delivery

When you receive a message, the system automatically notifies you of unread messages at heartbeat or conversation start.
Manual checking is usually unnecessary.

### Received Message Structure

Received messages include the following fields:

| Field | Description | Example |
|-------|-------------|---------|
| `id` | Unique message identifier | `20260215_093000_123456` |
| `thread_id` | Thread identifier | `20260215_090000_000000` |
| `reply_to` | Parent message ID | `20260215_085500_789012` |
| `from_person` | Sender name | `alice` |
| `to_person` | Recipient name (you) | `bob` |
| `type` | Message type | `message` (normal), `board_mention` (Board mention), `ack` (read receipt) |
| `content` | Message body | `Please review` |
| `intent` | Sender's intent | `report`, `question` |
| `timestamp` | Sent time | `2026-02-15T09:30:00` |

### Reply Obligation

- MUST: Reply to the sender when you receive an unread message
- MUST: Always respond to questions and requests
- SHOULD: Include next actions, not only “Understood”

## Receiving Messages from External Platforms

### The Server Receives Automatically

Messages from external platforms like Slack and Chatwork are **received continuously by the AnimaWorks server and automatically delivered to the target Anima's Inbox**. Animas do not need to maintain WebSocket connections or poll APIs themselves.

The server receives messages via the following methods (configured by administrators):

- **Socket Mode**: Real-time reception via Slack WebSocket
- **Webhook**: Reception via Slack Events API / Chatwork Webhook

Regardless of the method, messages are delivered to the Inbox in the same format.

### Identifying External Messages

Messages from external platforms differ from inter-Anima messages in the following ways:

| Field | Inter-Anima DM | External Message |
|-------|---------------|-----------------|
| `source` | `"anima"` | `"slack"`, `"chatwork"`, etc. |
| `from_person` | Anima name (e.g. `alice`) | `"slack:U12345..."` format |

### When External Messages Arrive

1. **Human DMs**: When a human sends a message to an Anima via Slack/Chatwork
2. **`call_human` replies**: When a human replies in the Slack thread of a `call_human` notification (details: `communication/call-human-guide.md`)
3. **Channel mentions**: When a message is posted in a Slack channel targeting an Anima

### Slack Messages: Immediate vs Deferred Processing

Slack messages are automatically classified for immediate processing or waiting until the next heartbeat:

| Condition | Processing timing | Reason |
|-----------|-------------------|--------|
| **@mention** (bot was mentioned) | **Immediate** | `intent="question"` is auto-assigned; processed immediately as actionable inbox work |
| **DM** (direct message to the bot) | **Immediate** | DMs are to the bot, so `intent="question"` is auto-assigned |
| **Channel message without mention** | **Next heartbeat** | `intent` is empty, so no immediate trigger; handled as unread on the scheduled heartbeat |

This avoids waking Anima for every channel message; they respond quickly only when explicitly addressed via @mention or DM.

### Responding to External Messages

When you receive an external message:

- For **`call_human` replies**: Respond via chat or `call_human`
- For **human DMs**: Reply via chat (Web UI), `send_message` to a registered human alias, or if you know the Slack user ID use `to="slack:U0123456789"` (or bare ID when the implementation accepts that form) — **`send_message` replies are possible** when they satisfy “Resolving `to`” above
- For **unknown senders**: Check the message `source` and `from_person`, and report to your supervisor if needed

## Message Body Best Practices

### Writing Good Messages

1. **Lead with the conclusion**: Let the recipient grasp the point in the first line
2. **Be specific**: Avoid vague wording; state numbers, deadlines, and scope clearly
3. **State actions explicitly**: Be clear about what you want the recipient to do
4. **State if a reply is needed**: When a response is required, say so (e.g. “Please reply with results”)

### Good vs Bad Examples

**Bad:**

```
Please check the data thing.
```

**Good:**

```
Please validate January 2026 sales data.
File: /shared/data/sales_202601.csv
Checks: missing values and amount field outliers
Deadline: today by 3pm
Please reply with results.
```

### Conveying Long Content

- SHOULD: If the body exceeds 500 characters, write the content to a file and include only the file path and summary in the message
- MUST: Place files in paths accessible to the recipient when referencing them

```
Created deploy procedure document.
File: ~/.animaworks/shared/docs/deploy-procedure-v2.md

Summary: Added 3 staging environment confirmation steps (see section 4.2).
Please review. Please reply with feedback.
```

## Common Failures and Fixes

### Wrong `intent`

**Symptom**: `Error: DM intent must be 'report' or 'question' only. Use Board (post_channel tool) for acknowledgments, thanks, or FYI.`

**Cause**: Omitted `intent`, or tried to send acknowledgment/thanks/FYI via DM

**Fix**: Always set `intent` to `report` or `question` for DMs. Use `delegate_task` for task delegation. Use Board (`post_channel`) for acknowledgments, thanks, and FYI

**Symptom**: `Error: intent='delegation' has been deprecated. Use the delegate_task tool to assign tasks to subordinates. send_message only supports 'report' and 'question' intents.`

**Cause**: Legacy `send_message(..., intent="delegation")`

**Fix**: Task delegation must use **`delegate_task` only**. `send_message` `intent` is `report` / `question` only

### Wrong recipient

**Symptom**: Unknown-recipient error, or message reaches the wrong party

**Cause**: `to` does not match resolution rules (Anima name spelling, alias not in `user_aliases`, Slack/Chatwork IDs not set, etc.)

**Fix**:

- For fastest delivery to an Anima, use the **same spelling as the directory name** (including case). If only casing differs, priority 6 still matches internal delivery
- If an **alias has the same name as an Anima**, an **exact** match to an Anima directory name **always** prefers internal delivery (if that is wrong, rename the Anima or change the alias)
- For humans, register aliases with `slack_user_id` / `chatwork_room_id` in `external_messaging.user_aliases` and verify `preferred_channel`
- If you know a Slack user ID, use `slack:USERID` or bare ID (`U` + **8+** alphanumerics) as `to`. **Short IDs** (e.g. `U12345`) are not accepted as Slack form and may be treated as unknown
- If the tool only returns hints like “reply directly in chat” or “use call_human”, see “When `send_message` recipient resolution fails” above and check `external_messaging` in `config.json` and session type
- If unsure, check org info with e.g. `search_memory(query="members", scope="knowledge")`

### Broken thread

**Symptom**: Reply not visible in conversation flow on the recipient side

**Cause**: Forgot to specify `reply_to` or `thread_id`

**Fix**: When replying, MUST set the original message's `id` to `reply_to` and the same `thread_id` as in the received message

### Message too long

**Symptom**: Recipient cannot grasp the point

**Fix**: Put the conclusion first; move details to a file. Use the message body as summary + file reference

### Second send to same recipient

**Symptom**: `Error: Message already sent to {to} in this run. Use Board for additional communication.`

**Cause**: Called `send_message` more than once to the same recipient in one run

**Fix**: Use Board (`post_channel`) for additional contact, or send in the next run (e.g. heartbeat)

### Sending to 3+ people

**Symptom**: `Error: Maximum {limit} DM recipients per run. Use Board (post_channel tool) for {limit}+ recipients.` (default `limit` is 2)

**Fix**: Use Board (`post_channel`) for communication to 3 or more people

### Forgetting to reply

**Symptom**: Recipient cannot track status; follow-up inquiry arrives

**Fix**: MUST reply to received messages. Even when you cannot act immediately, reply with e.g. “Acknowledged. I will respond by XX.”

## Sending Limits

System-wide rate limits apply to message sending.
Excessive sending can cause loops and failures; understand and follow these limits.

### Global Sending Limits (activity_log based)

| Limit | Default | Applies to |
|-------|---------|------------|
| Per hour | 30 messages | DM (`message_sent`) count |
| Per day | 100 messages | DM (`message_sent`) count |

Sending fails when limits are reached. `ack`, `error`, and `system_alert` types are not subject to limits.
Values can be changed in `config.json` via `heartbeat.max_messages_per_hour` / `heartbeat.max_messages_per_day`.

### Per-Run Limits

- **DM**: Maximum 2 recipients; 1 message per recipient
- **Board**: 1 post per channel per session (with cooldown)

### Cascade detection (round-trip limit between two parties)

Too many round-trips with the same party in a short time blocks sending.
Controlled by `heartbeat.depth_window_s` (time window) and `heartbeat.max_depth` (max depth) in `config.json`.

### When Limits Are Reached

1. Limits are computed from an `activity_log` sliding window
2. Hourly limit reached: Record send content in `current_state.md` and send in the next session
3. Daily limit reached: Send only essential messages; wait until the next day
4. Urgent contact needed: Use `call_human` (not subject to rate limits)

### Best Practices to Conserve Sending

- Combine multiple report items into one message
- Post acknowledgments, thanks, and FYI to Board (save DM quota)
- Consolidate regular info sharing into Board channel posts

## One-Round Rule

For DMs (`send_message`), **one topic per round-trip** is the principle.

### Rules

- MUST: Complete one topic in a single send–reply round
- MUST: If more than 3 round-trips are needed, move to a Board channel
- SHOULD: Include all needed information in the first message so follow-up questions are unnecessary

### Why the One-Round Rule

- More DM round-trips make hitting rate limits more likely
- Two-party message loops are throttled by **cascade detection** (sending is blocked when max depth is exceeded within the configured time window)
- Board posts can be read by other members and reduce duplicated information

### Exceptions

- Urgent blocker reports are exempt from count limits

## Communication Path Rules

Message recipients follow org structure:

| Situation | Recipient | Example |
|-----------|-----------|---------|
| Important progress or issue report | Supervisor | `send_message(to="manager", content="Task A complete", intent="report")` |
| Task instruction or delegation | Subordinate | `delegate_task(name="worker", instruction="Please create the report", deadline="1d")` |
| Peer coordination | Peer (same supervisor) | `send_message(to="peer", content="Please review", intent="question")` |
| Contact to another department | Via your supervisor | `send_message(to="manager", content="Need someone in dev (X) to check…", intent="question")` |

- MUST: Do not contact other department members directly. Go through your supervisor or theirs
- MAY: Communicate directly with peers (members with the same supervisor)

## Blocker Reports (MUST)

When any of the following occurs during task execution, report immediately to the requester via `send_message`.
Do not leave work idle in a “waiting” state without reporting.

- File/directory not found
- Access denied (insufficient permissions)
- Prerequisites not met
- Technical issue interrupting work
- Instructions unclear; you cannot decide

**Report to**: Requester (`send_message`)  
**Severe blocker** (delay of 30+ minutes expected): Also notify humans with `call_human`

### Blocker Report Example

```
send_message(
    to="manager",
    content="""[BLOCKER] Data aggregation task

Status: Specified file /shared/data/sales_202601.csv does not exist.
Impact: Cannot start aggregation.
Action needed: Please confirm the file path or place the file.""",
    intent="report"
)
```

## Required Elements for Request Messages (MUST)

When requesting work from another Anima, always include these five elements:

1. **Purpose** (why this work is needed)
2. **Scope** (file paths, resources)
3. **Expected outcome** (definition of done)
4. **Deadline**
5. **Whether a completion report is required**

Messages missing these elements force the recipient to ask for clarification, causing inefficient round-trips.
