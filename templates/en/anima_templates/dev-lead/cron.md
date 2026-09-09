# Cron: Dev Lead (PdM)

## Morning stand-up
schedule: 0 9 * * *
type: llm
Review the team's status and decide today's task allocation.
- Check for new issues and tasks
- Understand each member's current workload
- Assign tasks based on priority and each member's strengths
- Record the situation in state/daily_plan.md

## Full PR patrol
schedule: */15 * * * *
type: llm
Review all open PRs and determine what is missing to get each one merged.
- Check each PR's CI status, review status, and whether it has conflicts
- Re-delegate or escalate any PR that has stalled
- Only treat PRs that pass the quality gate (review complete and CI green) as merge candidates

## Weekly progress review
schedule: 0 17 * * 5
type: llm
Look back at the week's development results and extract lessons.
- Evaluate the quality of completed tasks and record team improvement points
- Detect technical debt and recurring problems, and plan countermeasures
- Fold the results into knowledge/ and send a weekly report to your supervisor
