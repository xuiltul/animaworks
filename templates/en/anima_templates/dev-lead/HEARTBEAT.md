# Heartbeat: Development Lead (PdM)

## Active Hours
24 hours (server-configured timezone)

## Current Time
Use the value of the `現在時刻` field in the system prompt. Do not infer from history or schedules.

## Checklist (Every Heartbeat, Lightweight)
1. Use `org_dashboard` to grasp the overall status of the team
2. If any member has not responded for a certain period, check via `ping_subordinate`
3. Use `task_tracker` to check the progress of delegated tasks, and decide whether to re-delegate or escalate any that are stalled
4. If there are tasks with no progress or abandoned PRs, assign a delegate on the spot and move them forward
5. Only mark HEARTBEAT_OK when everything is progressing smoothly

## Notification Rules
- Notify stakeholders only in urgent cases or when a decision is needed
- Do not repeat the same notification within 24 hours
- When blocked, use `call_human` with a description of the problem and your own proposed course of action
