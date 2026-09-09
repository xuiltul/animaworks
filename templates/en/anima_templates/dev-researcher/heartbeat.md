# Heartbeat: Dev Researcher

## Active Hours
24 hours (server timezone)

## Current Time
Use the `current_time` field from the system prompt. Do not infer it from history or schedules.

## Checklist
- Check whether a requested investigation exists
- Investigate read-only; do not modify the target files or data
- Report results in the agreed format
- If nothing needs attention, report HEARTBEAT_OK

## Notification Rules
- Always report investigation completion or blockers to the requester
- Do not repeat the same notification within 24 hours
