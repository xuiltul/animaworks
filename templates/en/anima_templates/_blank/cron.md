# Cron: {name}

<!--
=== Cron Format Specification ===

■ Basic Structure
  ## Task Name
  schedule: <5-field cron expression>
  type: llm | command
  (body or command/tool definition)

■ schedule: is required
  Write a `schedule:` line immediately after each task (## heading).
  If omitted, the task will not run.

■ 5-field cron expression
  schedule: min hour day month weekday
  ┌───── min (0-59)
  │ ┌───── hour (0-23)
  │ │ ┌───── day (1-31)
  │ │ │ ┌───── month (1-12)
  │ │ │ │ ┌───── weekday (0=Mon 〜 6=Sun)
  │ │ │ │ │
  * * * * *

■ Common Schedule Examples
  schedule: 0 9 * * *       # Every morning at 9:00
  schedule: */5 * * * *     # Every 5 minutes
  schedule: 0 9 * * 0-4    # Weekdays at 9:00 (Mon–Fri)
  schedule: 0 17 * * 4     # Every Friday at 17:00
  schedule: 0 2 * * *      # Daily at 2:00
  schedule: 30 12 1 * *    # 1st of each month at 12:30

■ Invalid Patterns
  ✗ ### cron expression     ← Use ## only for heading level
  ✗ schedule: 9am daily    ← Natural language not allowed; use 5-field cron expression
  ✗ (no schedule: line)    ← Must be specified

■ type values
  1. LLM type (type: llm) - Tasks requiring judgment or reasoning
  2. Command type (type: command) - Deterministic bash/tool execution

■ Options (command type only)
  skip_pattern: <regex>       — Skip LLM analysis when stdout matches
  trigger_heartbeat: false    — Do not trigger LLM analysis even when output exists

■ Detailed Reference
  → See common_skills/cron-management.md
-->

## Daily Planning
schedule: 0 9 * * *
type: llm
Review yesterday's progress from long-term memory and plan today's tasks.
Prioritize according to vision and goals.
Write results to state/current_task.md.

## Weekly Reflection
schedule: 0 17 * * 4
type: llm
Review this week's episodes/ and extract patterns to integrate into knowledge/.
(Memory consolidation = memory consolidation during sleep in neuroscience)

<!--
## Backup Execution
schedule: 0 2 * * *
type: command
command: /usr/local/bin/backup.sh

## Slack Notification
schedule: 0 9 * * 0-4
type: command
tool: slack_send
args:
  channel: "#general"
  message: "おはようございます！"
-->