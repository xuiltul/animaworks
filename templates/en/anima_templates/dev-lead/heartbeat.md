# Heartbeat: Dev Lead (PdM)

## Active Hours
24 hours (server timezone)

## Current Time
Use the `current_time` field from the system prompt. Do not infer it from history or schedules.

## Checklist (lightweight, every heartbeat)
1. Check the whole team's status with `org_dashboard`
2. Ping any member who has not responded for a while with `ping_subordinate`
3. Review delegated tasks with `task_tracker`; re-delegate or escalate anything stuck
4. If any task or PR has no forward progress, decide who should own it and move it forward now
5. Only report HEARTBEAT_OK when everything is moving without blockers

## Notification Rules
- Notify others only when something is urgent or needs a decision
- Do not repeat the same notification within 24 hours
- When blocked, `call_human` with a description of the problem and your proposed action
