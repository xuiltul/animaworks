# Heartbeat: {name}

## Active Hours
24 hours (server-configured time zone)

## Current Time
Use the value of the `現在時刻` field in the system prompt. Do not infer it from history or schedules.

## Checklist
- Check whether there are unread messages in the Inbox
- Check whether any blockers have arisen for in-progress tasks
- Check whether any new files have been placed in your workspace
- If there is nothing, do nothing (HEARTBEAT_OK)

## Notification Rules
- If something is deemed urgent, notify the relevant parties
- If there is no supervisor, report only the key points of deliverables received from subordinates to the user via call_human
- Do not repeat notifications about the same content within 24 hours
