# Heartbeat: Development Lead (PdM)

## Active Hours
24 hours (server-configured time zone)

## Current Time
Use the value of the `現在時刻` field in the system prompt. Do not infer it from history or schedules.

## Checklist (Every Heartbeat · Lightweight)
1. Check the overall team status using `org_dashboard`
2. If a member has not responded for some time, check in using `ping_subordinate`
3. Check the progress of delegated tasks using `task_tracker`, and decide whether to re-delegate or escalate any that are stalled
4. If there are tasks making no progress or neglected PRs, decide on a delegate and get things moving right away
5. Mark as HEARTBEAT_OK only when everything is proceeding smoothly

## Notification Rules
- Notify relevant people only when something is urgent or requires a decision
- Do not repeat notifications about the same thing within 24 hours
- When there is a blocker, `call_human` with an explanation of the issue and your proposed response
