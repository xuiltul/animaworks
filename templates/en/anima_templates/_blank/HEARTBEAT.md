# Heartbeat: {name}

## Active Hours
24 hours (server configuration timezone)

## Current Time
Use the value of the `現在時刻` field in the system prompt. Do not infer from history or schedules.

## Checklist
- Are there unread messages in the Inbox?
- Have any blockers occurred in ongoing tasks?
- Have new files been placed in my workspace?
- If nothing, do nothing (HEARTBEAT_OK)

## Notification Rules
- If deemed urgent, notify relevant parties
- If the supervisor is absent, communicate only the key points of results received from subordinates to the user via call_human
- Do not repeat the same notification within 24 hours
