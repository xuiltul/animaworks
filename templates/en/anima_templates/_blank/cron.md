# Cron: {name}

<!--
=== Cron Format Specification ===

■ Basic Structure
  ## Task name
  schedule: <5-field cron expression>
  type: llm | command
  (body or command/tool definition)

■ schedule: is required
  Write a `schedule:` line immediately after each task (## heading).
  Tasks without it will not run.

■ 5-field cron expression
  schedule: minute hour day month weekday
  ┌───── minute (0-59)
  │ ┌───── hour (0-23)
  │ │ ┌───── day (1-31)
  │ │ │ ┌───── month (1-12)
  │ │ │ │ ┌───── weekday (0=Mon … 6=Sun)
  │ │ │ │ │
  * * * * *

■ Common schedule examples
  schedule: 0 9 * * *       # Every morning 9:00
  schedule: */5 * * * *     # Every 5 minutes
  schedule: 0 9 * * 0-4     # Weekdays 9:00 (Mon-Fri)
  schedule: 0 17 * * 4      # Every Friday 17:00
  schedule: 0 2 * * *       # Daily 2:00
  schedule: 30 12 1 * *     # 1st of month 12:30

■ Invalid formats
  ✗ ### cron expression      ← Use ## only for headings
  ✗ schedule: every morning 9am   ← Natural language invalid; use 5-field cron
  ✗ (no schedule: line)    ← Must be specified

■ type options
  1. LLM type (type: llm) - Tasks requiring judgment or reasoning
  2. Command type (type: command) - Deterministic bash/tool execution

■ Options (command type only)
  skip_pattern: <regex>       — Skip LLM analysis when stdout matches
  trigger_heartbeat: false     — Do not trigger LLM analysis even on output

■ Detailed reference
  → See common_skills/cron-management.md
-->

## Daily Planning
schedule: 0 9 * * *
type: llm
Check yesterday's progress from long-term memory and plan today's tasks.
Prioritize based on mission and goals.
Write the result to state/current_task.md.

## Weekly Review
schedule: 0 17 * * 4
type: llm
Review this week's episodes/ and extract patterns to integrate into knowledge/.
(Memory consolidation = memory consolidation during sleep in neuroscience)

<!--
## Backup Run
schedule: 0 2 * * *
type: command
command: /usr/local/bin/backup.sh

## Slack Notification
schedule: 0 9 * * 0-4
type: command
tool: slack_send
args:
  channel: "#general"
  message: "Good morning!"
-->