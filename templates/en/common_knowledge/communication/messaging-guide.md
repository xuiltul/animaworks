# Messaging Guide

Comprehensive guide for communication with other Anima (team members).
Covers sending, receiving, and managing threads.

## send_message Tool — Parameter Reference

Use the `send_message` tool for sending messages (recommended).

<!-- AUTO-GENERATED:START tool_parameters -->
### Tool Parameter Reference (auto-generated)

#### `search_memory`

Search the anima's long-term memory (knowledge, episodes, procedures) by keyword.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | string | Yes | Search keyword |
| `scope` | `knowledge` \| `episodes` \| `procedures` \| `common_knowledge` \| `all` | No | Memory category to search |

#### `read_memory_file`

Read a file from the anima's memory directory by relative path.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `path` | string | Yes | Relative path within anima dir |

#### `write_memory_file`

Write or append to a file in the anima's memory directory.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `path` | string | Yes |  |
| `content` | string | Yes |  |
| `mode` | `overwrite` \| `append` | No |  |

#### `send_message`

Send a message to another anima.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `to` | string | Yes | Recipient anima name |
| `content` | string | Yes | Message content |
| `reply_to` | string | No | Message ID to reply to |
| `thread_id` | string | No | Thread ID |

<!-- AUTO-GENERATED:END -->

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `to` | string | MUST | Recipient Anima name (e.g. `alice`) |
| `content` | string | MUST | Message body |
| `reply_to` | string | MAY | ID of message you are replying to (e.g. `20260215_093000_123456`) |
| `thread_id` | string | MAY | Thread ID. Use when joining an existing thread |

### Basic Example

```
send_message(to="alice", content="Review complete. Three items to fix.")
```

### Replying Example

Use the received message's `id` and `thread_id` to link your reply:

```
send_message(
    to="alice",
    content="Understood. I'll fix by 3pm.",
    reply_to="20260215_093000_123456",
    thread_id="20260215_090000_000000"
)
```

## Thread Management

### Starting a New Thread

Omitting `thread_id` lets the system use the message ID as thread ID.
For a new topic, do not pass `thread_id`.

```
send_message(to="bob", content="I'd like to discuss the new project.")
# → thread_id is auto-generated (same as message ID)
```

### Replying in an Existing Thread

When replying, MUST include both `reply_to` and `thread_id`.

```
# Received message:
#   id: "20260215_093000_123456"
#   thread_id: "20260215_090000_000000"
#   content: "Please review"

send_message(
    to="alice",
    content="Review done.",
    reply_to="20260215_093000_123456",
    thread_id="20260215_090000_000000"
)
```

### Thread Rules

- MUST: Keep using the same `thread_id` for the same topic
- MUST: Set `reply_to` to the original message's `id` when replying
- SHOULD NOT: Mix different topics in one thread. Start a new thread for new topics
- MAY: Omit `thread_id` if unknown (creates a new thread)

## CLI Messaging

When tools are unavailable or sending via Bash:

### Syntax

```bash
python main.py send {your_name} <recipient> "message content"
```

### Examples

```bash
# Basic
python main.py send bob alice "Work complete. Please confirm."

# Thread reply
python main.py send bob alice "Understood" --reply-to 20260215_093000_123456 --thread-id 20260215_090000_000000
```

### Notes

- MUST: Wrap message in double quotes
- SHOULD: Prefer send_message when available (more reliable than CLI)
- Escape `"` inside messages: `\"`

## Receiving Messages

### Auto Delivery

When you receive a message, Heartbeat or chat start will show unread messages.
Manual checks are usually unnecessary.

### Received Message Fields

| Field | Description | Example |
|-------|-------------|---------|
| `id` | Unique message ID | `20260215_093000_123456` |
| `thread_id` | Thread ID | `20260215_090000_000000` |
| `reply_to` | Parent message ID | `20260215_085500_789012` |
| `from_person` | Sender | `alice` |
| `content` | Body | `Please review` |
| `timestamp` | Sent time | `2026-02-15T09:30:00` |

### Replying

- MUST: Reply to unread messages
- MUST: Respond to questions and requests
- SHOULD: Include next steps, not only "Understood"

## Message Best Practices

### Clear Messages

1. **Lead with conclusion**: Receiver gets the point in the first line
2. **Be specific**: Avoid vague wording; include numbers, deadlines, scope
3. **State actions**: Be explicit about what you want
4. **State if reply needed**: Write "Reply requested" when you need a response

### Good vs Bad

**Bad:**
```
Please check the data.
```

**Good:**
```
Please validate January 2026 sales data.
File: /shared/data/sales_202601.csv
Check: missing values and amount outliers
Deadline: today by 3pm
Please reply with results.
```

### Long Content

- SHOULD: If over ~500 chars, write to a file and send path + summary in the message
- MUST: Put files where the recipient can access them

```
Created deploy procedure.
File: ~/.animaworks/shared/docs/deploy-procedure-v2.md

Summary: Added 3 staging checks (section 4.2).
Please review. Reply requested.
```

## Common Mistakes and Fixes

### Wrong Recipient Name

**Symptom**: Message not received by intended party

**Cause**: `to` Anima name is incorrect

**Fix**: Anima names are case-sensitive. Use exact names from `identity.md`.
Check org: `search_memory(query="members", scope="knowledge")`

### Broken Thread

**Symptom**: Reply not visible in conversation flow

**Cause**: Missing `reply_to` or `thread_id`

**Fix**: For replies, MUST set `reply_to` to the original message `id` and use its `thread_id`

### Message Too Long

**Symptom**: Recipient misses the point

**Fix**: Put conclusion first; move details to files. Message = summary + reference

### Forgetting to Reply

**Symptom**: Recipient doesn't know status, asks again

**Fix**: MUST reply to received messages. If you can't act yet: "Checking. Will respond by XX."

## Sending Limits

Global rate limits apply to messages.
Avoid loops and overload by understanding these limits.

### Global Limits

| Limit | Value | Applies to |
|-------|-------|------------|
| Per hour | 30 messages | DM + Board posts combined |
| Per day | 100 messages | DM + Board posts combined |

`ack`, `error`, and `system_alert` are not counted.

### Per-Session Limits

- Board posts: 1 per channel per session (Heartbeat, chat, etc.)
- Duplicate DM to same recipient is blocked

### Hitting Limits

1. Limits are computed from activity_log sliding window
2. At hourly limit: wait for the next hour or retry in the next Heartbeat
3. At daily limit: send only essential messages; wait until next day
4. Urgent: use `call_human` (not subject to rate limits)

### Reducing Traffic

- Combine multiple updates into one message
- Avoid short replies like "OK"; include next actions
- Use Board for regular updates instead of many DMs

## One-Round Rule

For DMs, **one topic per round-trip** is the rule.

### Rules

- MUST: One topic should be handled in a single send-reply round
- MUST: If more than 3 round-trips are needed, move to a Board channel
- SHOULD: Include all needed info in the first message to avoid extra questions

### Why

- More DM round-trips increase rate limit usage
- Message loops between two parties are detected and throttled (4+ round-trips in 10 min)
- Board posts can be read by others and reduce duplicate info

### Exceptions

- One confirmation reply for task delegation (`intent: delegation`)
- Urgent blocker reports are not throttled

## Communication Paths

Route messages by org structure:

| Situation | Recipient | Example |
|-----------|-----------|---------|
| Important progress or issues | Supervisor | `send_message(to="manager", content="Task A done")` |
| Task instruction or check | Subordinate | `send_message(to="worker", content="Please create the report")` |
| Peer coordination | Peer (same supervisor) | `send_message(to="peer", content="Please review")` |
| Other department | Via your supervisor | `send_message(to="manager", content="Need dev team X to check...")` |

- MUST: Don't contact other departments directly. Go through your supervisor or theirs
- MAY: Communicate directly with peers (same supervisor)

## Blocker Reports (MUST)

When these occur during a task, report immediately to the assignee via `send_message`.
Do not leave them unreported.

- File/directory not found
- Access denied (permissions)
- Prerequisites not met
- Technical issue causing stoppage
- Instructions unclear and you cannot decide

Recipient: Assignee (send_message)
Severe blockers (expected delay 30+ min): Also notify humans with `call_human`

### Blocker Report Example

```
send_message(
    to="manager",
    content="""[BLOCKER] Data aggregation task

Status: File /shared/data/sales_202601.csv not found.
Impact: Cannot start aggregation.
Action needed: Please confirm path or provide file."""
)
```

## Required Elements for Request Messages (MUST)

When requesting work from another Anima, include these five elements:

1. **Purpose** (why this work is needed)
2. **Scope** (file paths, resources)
3. **Expected outcome** (definition of done)
4. **Deadline**
5. **Whether completion report is needed**

Missing any of these forces clarification messages and extra round-trips.
