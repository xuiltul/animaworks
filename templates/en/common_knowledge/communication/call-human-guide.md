# call_human Guide — Notifying Humans and Receiving Replies

## Overview

`call_human` sends notifications to human administrators. **It is not one-way** — when a human replies in the Slack thread, the reply is automatically delivered to the sending Anima's Inbox.

## Sending

```
call_human(
    subject="Subject line",
    body="Body (include situation, what you tried, and what you need)",
    priority="normal"  # "normal" | "high" | "urgent"
)
```

| Parameter | Required | Description |
|-----------|----------|-------------|
| `subject` | MUST | Subject line (concise) |
| `body` | MUST | Body (include situation, attempts, and request) |
| `priority` | MAY | `normal` (default), `high`, `urgent` |

### When to Use

- High-urgency issues (data loss risk, security, service outage)
- Decisions beyond your authority
- Top-level Anima escalation when no supervisor exists

See `troubleshooting/escalation-flowchart.md` for detailed criteria.

### Rate Limits

- `call_human` is **exempt** from DM rate limits (30/h, 100/day)
- Can be sent without concern for limits even during emergencies

## Receiving Replies

### How It Works

1. `call_human` posts a message to Slack
2. The human replies in the message **thread**
3. The reply is automatically routed to the sending Anima's Inbox
4. The reply is processed in the next Inbox cycle (typically detected within 2 seconds)

### Reply Message Attributes

Received replies have these attributes:

- `source`: `"slack"` (reply via Slack)
- `from_person`: `"slack:U..."` format (Slack user ID)
- Processed like any other Inbox message

### Waiting for a Reply

If you need to wait for a reply after sending `call_human`:

1. Record the waiting state in `state/current_task.md`
2. The reply will arrive automatically in the next Inbox processing cycle
3. Even if no immediate reply comes, it will be delivered when the human responds

### Responding to a Reply

Once you receive a reply, you can respond as with any Inbox message. Note that since the sender is a Slack user, use chat response or another `call_human` rather than `send_message`.

## Notes

- Reply routing only works for notifications sent via **Bot Token mode** (`chat.postMessage`). Webhook mode notifications cannot receive replies (this is an admin configuration issue, not something Animas control)
- Reply mappings are retained for **7 days**. Replies to threads older than 7 days will not be delivered
- If no human reply arrives, you can send another `call_human` as a follow-up
