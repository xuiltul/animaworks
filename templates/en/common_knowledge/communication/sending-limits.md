# Message Send and Receive Operations Guide

This document summarizes the operations for message delivery, Inbox startup, and avoiding conversation loops. The previous send budget, depth rejection, Inbox cooldown / cascade / intent filters have been removed.

## Rules that remain in effect when sending

- The intent for `send_message` is `report` or `question`. Use `delegate_task` for task delegation.
- Only one DM to the same recipient per run. This guard prevents duplicate sends; there is no limit on the number of recipients.
- There are no send budgets per hour or per day, and no send refusals based on conversation depth between pairs.
- Conversation depth between internal Anima may be logged for diagnostic purposes. This is for observation only; it does not discard message contents or refuse to send.

## Responses to Acknowledgments, Thanks, and Praise

If the received content is only an acknowledgment, thanks, or praise, do not reply. Do not continue the exchange; only take necessary action if there are additional questions, requests, or new information.
For a work request, send an acknowledgment once, if needed, along with the outlook and next steps.

## Inbox Startup and Processing

Detect new JSON files in the Inbox via file change notifications, and start Inbox processing if there are unread messages.
Only one Inbox processing run can be active at a time. Messages that arrive during execution are processed together in the next run.
To guard against missed file notifications, re-check for unread messages every 45 seconds. On provider-side failures, wait for the recovery time of `rate_guard` and keep unread messages.

If there are a large number of messages, offloading to `state/overflow_inbox/` remains as capacity protection. This is not a suppression based on sender or intent; offloaded messages can be reviewed and processed later.

## Logging and Verification

`Messenger.send` records sent messages in the activity log. If conversations continue over a short period, diagnostic logs may be added, but the send result is unchanged.
The most recent `message_sent` / `channel_post` are used by `core/memory/priming/outbound.py` for priming.

If a message cannot be sent, check the actual delivery errors for destination resolution, permissions, and external channels. There is no need to wait for rate limits or conversation depth limits.

## Implementation Location

| Role | Module |
|------|------------|
| Recipient resolution and external delivery to Slack / Chatwork | `core/messaging/outbound.py` |
| Internal DM delivery, activity log recording, and depth diagnostic logging | `core/messaging/messenger.py` |
| DM deduplication and deduplication within Board runs | `core/tooling/handler_comms.py` |
| Inbox file wake, single execution, and provider backoff | `core/runtime/inbox_rate_limiter.py` |
| Inbox capacity protection via overflow | `core/anima/inbox_overflow.py` |
| Prompt injection for recent sends | `core/memory/priming/outbound.py` |
