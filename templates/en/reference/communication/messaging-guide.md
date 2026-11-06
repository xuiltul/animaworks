# Complete Guide to Sending Messages

A comprehensive guide for communicating with other Anima (employees).
Covers all procedures for sending, receiving, and managing message threads.

## send_message Tool — Parameter Reference

Use the `send_message` tool to send messages (recommended).

### Parameter List

| Parameter | Type | Required | Description |
|-----------|------|------|------|
| `to` | string | MUST | Recipient. See “Resolving recipient `to`” below for resolution rules. Anima names, `config.json` human aliases, `slack:USERID` / `chatwork:ROOMID`, and standalone Slack user IDs (`U` + 8 or more alphanumeric characters) can be used |
| `content` | string | MUST | Message body |
| `intent` | string | MUST | Message intent. Allowed values: `report` (progress and results reports) or `question` (questions and inquiries requiring a response) only. Use the `delegate_task` tool to delegate tasks. Share acknowledgments, thanks, and FYIs on the Board using `animaworks-tool board post <channel> <text>` |
| `reply_to` | string | MAY | ID of the message being replied to (e.g., `20260215_093000_123456`) |
| `thread_id` | string | MAY | Thread ID. Specify this when joining an existing thread |

### DM Rules Within a Run

- Send at most **one** DM to the same destination per run. This rule prevents duplicate sends; there is no limit on the number of destinations
- To share additional information with the same destination, use Board or send it in the next run depending on the content

### Destination `to` Resolution (Unified Outbound)

Destinations in `send_message` are resolved by `core/outbound.resolve_recipient` according to the following **priority order** (ordered from most to least tolerant of notation variations).

- **Leading and trailing whitespace** in the destination string is trimmed before resolution.

| Priority | Condition | Result |
|----------|-----------|--------|
| 1 | **Exact match** with an existing Anima directory name (case-sensitive) | Internal Inbox |
| 2 | **Match** with a key in `config.json`'s `external_messaging.user_aliases` (case-insensitive) | External (Slack or Chatwork). Prefers `preferred_channel`; if no contact exists there, falls back to the other configured channel |
| 3 | Starts with `slack:` (case-insensitive; e.g., `slack:U0123456789`, `Slack:u06…`) | Trims everything after the colon, normalizes the user ID to **uppercase**, then sends a Slack DM |
| 4 | Starts with `chatwork:` (prefix is case-insensitive) | Posts to Chatwork with the room ID obtained by trimming after the colon (room ID case is preserved) |
| 5 | **Slack user ID format** alone: starts with `U` followed by **8 or more** alphanumeric characters (the entire string matches the regular expression `^U[A-Z0-9]{8,}$`) | Slack DM (for specifying only the ID without a prefix). **Strings that are too short** (e.g., `U12345`) do not match at this stage and proceed to lower-priority rules |
| 6 | **Case-insensitive match** with an existing Anima name | Internal Inbox (delivered under the official name on disk) |
| 7 | None of the above | Unknown destination (`RecipientNotFoundError`. At the low level, known Anima names or alias names may be included in the message) |

**Conflict between Anima names and aliases**: Even if a string matches a key in `user_aliases`, if it **exactly matches an existing Anima directory name**, priority **1** always takes precedence and the message is delivered to the internal Inbox (the alias is not used).

#### When the `send_message` Tool Fails to Resolve a Destination

Even if `resolve_recipient` fails, the tool result does not return the full exception text. Instead, **guidance** is returned according to the session type (`core/tooling/handler_comms.py`).

- **During a chat with a human** (`chat`): Indicates that `send_message` cannot be sent to that destination, that **replying directly in text will reach the human user**, and that `send_message` should be used for other Anima.
- **Outside a chat** (heartbeat, cron, etc.): Indicates that contact with humans should use **`call_human`**, and that `send_message` should be used for other Anima.

Therefore, low-level wording such as "Known animas: …" often does not appear to tool users. To fix configuration or aliases, check `config.json`'s `external_messaging`.

#### Human Aliases and `preferred_channel`

Each entry in `external_messaging.user_aliases` can have `slack_user_id` and/or `chatwork_room_id` configured. When `external_messaging.preferred_channel` is `slack`, Slack is chosen if a Slack ID exists. If the Slack ID is empty and only Chatwork exists, it falls back to Chatwork (and vice versa when `preferred_channel` is `chatwork`). Aliases with **neither ID** result in an error during resolution.

#### Delivery After Reaching an External Channel (Overview)

When resolved to an external route, `send_message` is sent via API from `core/outbound.send_external` without going through the internal Messenger.

- **Attempt order**: First send via the resolved `channel` (`slack` or `chatwork`); on exception, follow `_build_channel_order` and, if the recipient has **both** a Slack ID and a Chatwork room ID, also try the other channel in sequence.
- **Slack**: The token used by `core/outbound._send_via_slack` is **`SLACK_BOT_TOKEN__{送信元Anima名}`** from Vault/shared credentials (if available). The body is formatted for Slack by `md_to_slack_mrkdwn`. If an icon URL can be resolved via `core/integrations._anima_icon_url`, it is passed to `post_message`.
  - **`[送信者名]` prefix at the start of the body**: Only when **no Anima-dedicated bot token exists**, `[{送信元Anima名}] ` is prepended to the body (to indicate in the DM who the message is from). **When a token exists**, no prefix is added; the sender is represented by `username` (Anima name) and `icon_url`.
- **Chatwork**: Posts to the room via the sending Anima's own identity token (`CHATWORK_API_TOKEN__<Anima名>`). If the sending Anima name exists, `[送信者名] ` is prepended to the body, and the message is formatted by `md_to_chatwork`.

### Basic Send Example

```
send_message(to="alice", content="レビュー完了しました。修正点は3箇所です。", intent="report")
```

### Reply Send Example

Use the received message's `id` and `thread_id` to associate the reply:

```
send_message(
    to="alice",
    content="了解しました。15時までに対応します。",
    intent="report",
    reply_to="20260215_093000_123456",
    thread_id="20260215_090000_000000"
)
```

Aliases registered in `user_aliases` (e.g., `user`) can be sent to via `send_message` just like internal Anima (routed to the external channel).

```
send_message(to="user", content="対応完了しました。", intent="report")
```

### Choosing an intent

| intent | Purpose | Example |
|--------|------|-----|
| `report` | Progress and results reports | Task completion report, status update to a supervisor |
| `question` | Questions requiring a response | Clarifying something unclear, an inquiry requesting a decision |

**Note**: Acknowledgments, thanks, and FYIs such as “Understood” and “Thank you” cannot be sent via DM. Share them on the Board using `animaworks-tool board post <channel> <text>`.

### Choosing between the Board and DM

| Purpose | Tool | Example |
|------|-----------|-----|
| Progress and results reports | send_message (intent=report) | Task completion report to a supervisor |
| Task delegation | delegate_task | Delegate one persistent task to a direct subordinate. Check progress with task_tracker |
| Questions and inquiries | send_message (intent=question) | Clarify something unclear |
| Acknowledgment, thanks, FYI | `animaworks-tool board post <channel> <text>` | “Understood,” “Shared” |
| Sharing with the whole team | `animaworks-tool board post <channel> <text>` | Announce to all relevant people |
| Second message to the same recipient | `animaworks-tool board post <channel> <text>` | Share additional information |

## Thread Management

### Starting a New Thread

If `thread_id` is omitted, the system automatically sets the message ID as the thread ID.
Do not specify `thread_id` when starting a new topic.

```
send_message(to="bob", content="新しいプロジェクトの件で相談があります。", intent="question")
# → thread_id は自動生成される（メッセージIDと同じ値）
```

### Replying to an Existing Thread

When replying to a received message, MUST: specify both `reply_to` and `thread_id`.

```
# 受信メッセージ:
#   id: "20260215_093000_123456"
#   thread_id: "20260215_090000_000000"
#   content: "レビューお願いします"

send_message(
    to="alice",
    content="レビュー完了しました。",
    intent="report",
    reply_to="20260215_093000_123456",
    thread_id="20260215_090000_000000"
)
```

### Thread Management Rules

- MUST: Keep using the same `thread_id` for conversations on the same topic
- MUST: When replying, set the original message's `id` as `reply_to`
- SHOULD NOT: Mix a different topic into an existing thread. Start a new thread for a new topic
- MAY: If `thread_id` is unknown, it may be omitted (the message is treated as a new thread)

## Sending Messages via CLI

A method for when the tool is unavailable or when sending via Bash.

### Basic Syntax

```bash
animaworks send {送信者名} {宛先} "メッセージ内容" [--intent report|question] [--reply-to ID] [--thread-id ID]
```

### Concrete Examples

```bash
# 基本送信（intent は省略可、CLI 経由では空でも送信可能）
animaworks send bob alice "作業完了しました。確認をお願いします。" --intent report

# スレッド返信
animaworks send bob alice "了解しました" --intent report --reply-to 20260215_093000_123456 --thread-id 20260215_090000_000000
```

### Notes

- MUST: Enclose message content in double quotes
- SHOULD: Prefer the send_message tool when available (more reliable than CLI)
- If the message contains `"`, escaping is required: `\"`

## How to Check Received Messages

### Automatic Delivery

When a message is received, the system automatically notifies you of unread messages at heartbeat or conversation start.
Manual checking is usually unnecessary.

### Structure of Received Messages

Received messages contain the following information:

| Field | Description | Example |
|-------|-------------|---------|
| `id` | Unique message identifier | `20260215_093000_123456` |
| `thread_id` | Thread identifier | `20260215_090000_000000` |
| `reply_to` | Reply-to message ID | `20260215_085500_789012` |
| `from_person` | Sender name | `alice` |
| `to_person` | Recipient name (you) | `bob` |
| `type` | Message type | `message` (normal), `board_mention` (Board mention), `ack` (read receipt) |
| `content` | Message body | `レビューお願いします` |
| `intent` | Sender's intent | `report`, `question` |
| `timestamp` | Send timestamp | `2026-02-15T09:30:00` |

### Reply Policy

- MUST: Respond to unread messages that contain questions, requests, or require action
- MUST NOT: Continue replying to mere acknowledgments, thanks, or praise
- SHOULD: If an acknowledgment is needed, include not just "Understood" but also the next action

## Receiving Messages from External Platforms

### The Server Receives Automatically

Messages from external platforms such as Slack or Chatwork are **continuously received by the AnimaWorks server and automatically delivered to the target Anima's Inbox**. The Anima itself does not need to maintain a WebSocket connection or poll APIs.

The server receives messages using the following methods (configured by the administrator):

- **Socket Mode**: Real-time reception via Slack WebSocket
- **Webhook**: Reception via Slack Events API / Chatwork Webhook

Regardless of the method, messages are delivered to the Inbox in the same format.

### Identifying External Messages

Messages arriving from external platforms differ from normal Anima-to-Anima messages in the following ways:

| Field | Anima-to-Anima DM | External Message |
|-------|-------------------|------------------|
| `source` | `"anima"` | `"slack"`, `"chatwork"`, etc. |
| `from_person` | Anima name (e.g., `alice`) | `"slack:U12345..."` format |

### Cases Where External Messages Arrive

1. **DM from a human**: When a human sends a message to Anima via Slack/Chatwork
2. **Reply to call_human**: When a human replies in the Slack thread of the notification sent via `call_human` (details: `communication/call-human-guide.md`)
3. **Mention via channel**: When a message addressed to Anima is posted in a Slack channel

### Inbox Processing for Slack Messages

When a Slack message is written to the Inbox, a file change notification triggers Inbox processing. There is no intent filter at startup, and processing is not delayed based on whether a mention is present. If a file notification is missed, a re-check every 45 seconds compensates.

| Condition | Inbox Processing | Response Policy |
|------|------------|----------|
| **With @mention** (Bot is mentioned) | Triggered by file notification | `intent="question"` is automatically assigned. Respond according to the content |
| **DM** (Direct message to the Bot) | Triggered by file notification | `intent="question"` is automatically assigned. Respond according to the content |
| **Channel message without mention** | Triggered by file notification | For messages not addressed to you, just stay informed; replies or actions may not be necessary |

Being triggered does not mean you must always reply. Do not reply to mere acknowledgments, thanks, or praise; only respond when there are additional questions, requests, or new information.

### Responding to External Messages

When you receive an external message:

- If it is a reply to `call_human`: respond with a chat response or via `call_human`
- If it is a DM from a human: reply via chat (Web UI), send `send_message` to a registered human alias, or if the other party's Slack user ID is known, **a reply via `send_message` is possible** using `to="slack:U0123456789"` (or the ID alone, if the implementation accepts that format) (provided the above "Resolution of destination `to`" is satisfied)
- If the sender is unknown: check the message's `source` and `from_person`, and report to your supervisor as needed

## Best Practices for Message Content

### How to Write Good Messages

1. **Put the conclusion first**: Make sure the recipient can grasp the key point from the first line
2. **Be specific**: Avoid vague expressions; clearly state numbers, deadlines, and targets
3. **Make the action explicit**: Clarify what you want the recipient to do
4. **State whether a reply is needed**: If a reply is required, write "Please reply"

### Good and Bad Examples

**Bad example:**
```
データの件、確認しておいてください。
```

**Good example:**
```
売上データ（2026年1月分）のバリデーションチェックをお願いします。
対象ファイル: /shared/data/sales_202601.csv
確認観点: 欠損値の有無と金額フィールドの異常値
期限: 本日15時まで
結果は返答をお願いします。
```

### How to Convey Long Content

- SHOULD: If the message body exceeds 500 characters, write the content to a file and include only the file path and a summary in the message
- MUST: When referencing a file, place it in a path accessible to the recipient

```
デプロイ手順書を作成しました。
ファイル: ~/.animaworks/shared/docs/deploy-procedure-v2.md

要約: ステージング環境での確認ステップを3つ追加しました（セクション4.2参照）。
レビューをお願いします。返答をお願いします。
```

## Common Failures and Countermeasures

### Incorrect intent specification

**Symptom**: `Error: DMのintentは 'report', 'question' のみ許可されています` is displayed

**Cause**: `intent` was omitted, or an acknowledgment, thanks, or FYI was sent via DM

**Remedy**: For DMs, always specify `report` or `question` for `intent`. Use `delegate_task` to delegate tasks. Use `animaworks-tool board post <channel> <text>` for acknowledgments, thanks, and FYIs

**Symptom**: Something like `intent='delegation' は廃止されました` is displayed

**Cause**: The legacy `send_message(..., intent="delegation")` is being used

**Remedy**: Use **`delegate_task`** only for task delegation. The `intent` of `send_message` is only for `report` / `question`

### Incorrect Destination Name

**Symptom**: A destination-not-found error occurs, or the message reaches an unintended recipient

**Cause**: `to` does not match the resolution rules (misspelling of the Anima name, `user_aliases` not registered, Slack/Chatwork ID not set, etc.)

**Countermeasure**:

- For the fastest delivery, specify the Anima name **using the same notation as the directory name** (including capitalization). Even with different notation, if only capitalization differs, it will match internal delivery under rule 6
- If there is an **alias with the same name as an Anima name**, the **exact-match Anima directory name** always takes priority for internal delivery (if the opposite is intended, you need to rename the Anima name or change the alias name)
- For humans, set the alias and `slack_user_id` / `chatwork_room_id` in `external_messaging.user_aliases`, and check `preferred_channel`
- If a known Slack user ID is available, use `slack:USERID` or the ID alone (alphanumeric characters **8 or more** after `U`) as `to`. **Short IDs** (e.g., `U12345`) are not recognized as Slack format and may be treated as destination-not-found
- If the tool only returns hints like "reply directly in chat" or "use call_human," refer to "When the `send_message` tool fails to resolve the destination" above, and check the `external_messaging` of `config.json` and the session type
- If unclear, check organizational information via `search_memory(query="メンバー", scope="knowledge")` or similar

### Thread Disconnection

**Symptom**: You replied, but the other party cannot see the conversation flow

**Cause**: `reply_to` or `thread_id` was not specified

**Countermeasure**: When replying, MUST: set the original message's `id` in `reply_to`, and set `thread_id` as-is in `thread_id`

### Message Too Long

**Symptom**: The recipient cannot grasp the key points

**Countermeasure**: Put the conclusion at the beginning and separate details into a file. Use a summary plus file reference format for the message body

### Second message to the same recipient

**Symptom**: `Error: このrunで既に {to} にメッセージを送信済みです` is displayed

**Cause**: send_message was called more than once for the same recipient in a single run

**Remedy**: Use `animaworks-tool board post <channel> <text>` for additional communication. Alternatively, send it in the next run (e.g., a heartbeat)

### Forgetting to Reply

**Symptom**: The other party cannot track the status and sends a follow-up inquiry

**Countermeasure**: Reply to messages that require a response. If you cannot respond immediately, communicate the outlook, e.g., "Confirmed. I will respond by XX o'clock." Do not reply to mere acknowledgments or thanks

## Sending Rules

- The intent for `send_message` must be `report` or `question`. Use `delegate_task` to delegate tasks.
- Only one DM per recipient is allowed in the same run. There is no limit on the number of recipients.
- There is no cooldown for Board posts or shared sending budget for DMs and Board posts.
- Sending is not rejected based on conversation depth between pairs. The depth between internal Anima may be logged for diagnostic purposes, but the message body is not discarded.

## Reply Policy

Reply to messages that require a response, and if an acknowledgment is needed, communicate it once with the next action or outlook. Do not reply to mere acknowledgments, thanks, or praise, and do not continue the exchange.

## Recommendations for Keeping Conversations Concise

- Consolidate the necessary information for one topic and convey it in a form that does not require additional confirmation.
- Combine multiple report items into a single message when possible, and use the Board for organization-wide sharing.
- When the conversation needs to continue, choose DM or Board based on the content. There is no system-level rejection based on the number of exchanges.

## Rules for Communication Paths

Message destinations follow the organizational structure:

| Situation | Destination | Example |
|------|------|-----|
| Reporting important progress or issues | Supervisor | `send_message(to="manager", content="タスクA完了", intent="report")` |
| Task instructions or delegation | Subordinate | `delegate_task(name="worker", instruction="レポート作成をお願い")` |
| Coordination with colleagues | Colleague (same supervisor) | `send_message(to="peer", content="レビューお願い", intent="question")` |
| Contacting another department | Via your own supervisor | `send_message(to="manager", content="開発部のXさんに確認してほしい件が...", intent="question")` |

- MUST: Do not contact members of other departments directly. Go through your own supervisor or the other party's supervisor
- MAY: You may interact directly with colleagues (members who share the same supervisor)

## Blocker Report (MUST)

If any of the following situations occur during task execution, report immediately to the requester via `send_message`.
Do not leave the task in a "waiting" status.

- File/directory not found
- Unable to access due to insufficient permission
- Prerequisites not met
- Work interrupted due to technical issues
- Instructions unclear and cannot be determined

Report to: Requester (send_message)
Critical blocker (if a delay of 30 minutes or more is expected): Also notify a human via `call_human`

### Example of a Blocker Report

```
send_message(
    to="manager",
    content="""【ブロッカー報告】データ集計タスク

状況: 指定されたファイル /shared/data/sales_202601.csv が存在しません。
影響: 集計作業を開始できません。
必要なアクション: ファイルパスの確認、またはファイルの配置をお願いします。""",
    intent="report"
)
```

## Required Elements of a Request Message (MUST)

When requesting a task from another Anima, always include the following 5 elements:

1. **Purpose** (why this work is needed)
2. **Target** (file path, resources)
3. **Expected outcome** (what constitutes completion)
4. **Deadline**
5. **Whether a completion report is required**

If these are missing, the receiving side must reply to confirm, resulting in inefficient back-and-forth.
