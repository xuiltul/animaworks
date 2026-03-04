# Cron: alex

## Tasks

### Daily Standup
- schedule: "0 10 * * 1-5"
- type: llm
- description: |
    Review team status and organize today's priorities.
    1. Check subordinate status via org_dashboard
    2. Review each member's current_task
    3. Address any blockers
    4. Share today's focus points in the general channel

### Weekly Review
- schedule: "0 17 * * 5"
- type: llm
- description: |
    Reflect on the week's progress and plan for next week.
    1. Review completed tasks
    2. Organize remaining issues
    3. Set next week's priorities
    4. Consider feedback for the team
