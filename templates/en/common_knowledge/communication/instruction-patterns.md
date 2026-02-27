# Instruction Patterns

Patterns for giving clear, actionable instructions to subordinates and teammates.
Vague instructions cause rework and confusion. Use these patterns so others can act without guessing.

## Five Elements of a Clear Instruction

Include these five when giving instructions (MUST):

| Element | Description | MUST/SHOULD |
|---------|-------------|-------------|
| Purpose (why) | Reason and context for the work | MUST |
| Expected outcome (what) | What should exist when done | MUST |
| Deadline (when) | Completion deadline | MUST (or rough target) |
| Constraints (how) | Approach to use, what to avoid | SHOULD |
| When to report | On completion / progress / blockers | MUST |

## Good vs Bad Instructions

### Example 1: Data Aggregation Request

**Bad:**
```
send_message(
    to="alice",
    content="Please aggregate the sales data."
)
```
Problems: Which data, period, format, deadline are unclear.

**Good:**
```
send_message(
    to="alice",
    content="""Please do the monthly sales aggregation.

Purpose: For the 2/20 management meeting
Target: January 2026 sales data (/shared/data/sales_202601.csv)
Output: Department and product category summary (Markdown)
Location: /shared/reports/sales_summary_202601.md
Deadline: 2/18 (Tue) 17:00
Report: Please reply with a result summary when done."""
)
```

### Example 2: Investigation Request

**Bad:**
```
send_message(
    to="bob",
    content="Look into the API error."
)
```
Problems: Which API, which error, depth, report format unclear.

**Good:**
```
send_message(
    to="bob",
    content="""Please investigate GitHub API rate limit error (HTTP 403).

Context: Intermittent since 3pm yesterday, auto-deploy failing
Investigate:
1. Error frequency and pattern (log: /var/log/deploy/github-api.log)
2. Current rate limit config and usage
3. Mitigation options (retry strategy, token splitting, etc.)

Deadline: Today
Report: Summarize findings and recommendations in your reply. Report immediately if you find something critical."""
)
```

### Example 3: Review Request

**Bad:**
```
send_message(
    to="carol",
    content="Take a look at the code."
)
```

**Good:**
```
send_message(
    to="carol",
    content="""Please review the auth module code.

Target: ~/project/auth/token_manager.py (new)
Check:
- Security (token storage, invalidation)
- Error handling coverage
- Consistency with auth_handler.py

Deadline: Tomorrow morning
Report: Reply with 'LGTM' if fine, or specific items and reasons if changes needed."""
)
```

## Delegation Patterns

### Pattern 1: One-Off Task

Single task. Basic form.

```
send_message(
    to="alice",
    content="""[Request] Document update

Please update the API spec (/shared/docs/api-spec.md) with v2.1 changes.

Changes:
- Add pagination params to /api/users
- Add total_count to response
- See /shared/docs/changelog-v2.1.md for details

Deadline: 2/16 15:00
Reply when done."""
)
```

### Pattern 2: Recurring Task

Instruct a recurring task that should go into Heartbeat or cron.

```
send_message(
    to="bob",
    content="""[Recurring] Daily log monitoring

From now on, check the application log for anomalies every morning.

Target: /var/log/app/error.log
Check:
- ERROR/CRITICAL count in last 24h
- Any new error patterns

Reporting:
- No issues → no report
- 10+ ERROR or new pattern → report to me immediately
- Weekly summary every Friday

Add this to your Heartbeat checklist."""
)
```

### Pattern 3: Phased Task (with Milestones)

Break a large task into phases.

```
send_message(
    to="carol",
    content="""[Request] Notification feature — 3 phases

Please add the user notification feature.

Phase 1 (by 2/17): Design
- Define types (email/Slack/in-app) and priority
- Design data model
→ Reply with design. I'll approve before Phase 2.

Phase 2 (by 2/20): Implementation
- Implement based on approved design
- Include tests
→ Reply when done.

Phase 3 (by 2/21): Documentation
- API spec and user guide

Reply at the end of each phase.
Ask if unsure between phases."""
)
```

## Follow-Up Patterns

### Pre-Deadline Check

```
send_message(
    to="alice",
    content="""Checking on the sales aggregation.
Deadline is tomorrow 17:00 — how is it going?
Let me know if there are blockers."""
)
```

### Post-Deadline Follow-Up

```
send_message(
    to="bob",
    content="""Following up on the log monitoring results.
Deadline was today — what's the status?
Share any blockers; we can extend if needed."""
)
```

### Feedback After Deliverable

```
send_message(
    to="carol",
    content="""Reviewed the design. Overall good direction.

Changes requested:
1. Add exponential backoff for notification retries
2. Include i18n for notification templates in the design

Please send revised version by 2/18."""
)
```

## Escalation Criteria

When to handle yourself vs escalate to supervisor.

### Handle Yourself

- Work is clearly defined and within your scope
- Minor error with known fix
- Subordinate question you can answer
- Priorities are clear and no ambiguity

### Escalate

- MUST: Decision exceeds your authority (e.g. external API policy change)
- MUST: Incident affects multiple departments
- MUST: Deadline will be missed
- SHOULD: Unexpected situation, multiple valid options
- SHOULD: Subordinate escalated and you can't resolve

### Escalation Example

```
send_message(
    to="manager",
    content="""[Escalation] Deploy pipeline incident

Status: Auto-deploy stopped 12h due to GitHub API rate limit
Impact: Can't deploy v2.1 planned for today
Attempted: Longer retry interval, caching (no effect)
Needed: Decide whether to issue another GitHub token or switch to manual deploy

Please advise."""
)
```

## Instruction Templates

### Generic Task Request

```
[Request] {task_name}

{Background (1–2 lines)}

Work:
- {specific work 1}
- {specific work 2}

Deliverable: {what and where}
Deadline: {date/time}
Report: {on completion / progress / blockers}
```

### Investigation Request

```
[Investigation] {topic}

Background: {why needed}

Investigate:
1. {item 1}
2. {item 2}
3. {item 3}

Deadline: {date/time}
Report: Summarize findings and recommendations.
```

### Review Request

```
[Review] {subject}

Target: {path or resource}
Check:
- {point 1}
- {point 2}

Deadline: {date/time}
Report: Reply with "LGTM" or specific items and reasons.
```
