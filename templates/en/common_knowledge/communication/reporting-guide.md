# Reporting and Escalation Guide

> **Required**: Check the mandatory fields in `communication/message-quality-protocol.md` before reporting or escalating.

Reporting to your supervisor is the lifeblood of organizational operations.
Reporting at the right time and in the right format supports early problem detection and rapid decision-making.

## Tool Usage

Use the `send_message` tool for reporting. Observe the following constraints:

| Tool | Purpose | Notes |
|------|---------|-------|
| `send_message` | One-to-one reports to supervisor or colleagues | `intent="report"` required. Progress, results, decision requests |
| `post_channel` | Announcements to the whole team | Acknowledgments, thanks, FYI — use Board |
| `call_human` | Urgent notification to humans | Service outage, security incident, etc. |

**send_message constraints**:
- `intent` is required. Use `report` for reports, `question` for questions. Use delegate_task for task delegation
- Maximum 2 recipients per run; one message per recipient
- Use Board (post_channel) for additional follow-up

## When to Report

### Immediate Reports (MUST)

Report to your supervisor as soon as you become aware of any of the following:

| Situation | Reason | Example |
|-----------|--------|---------|
| Task completed | Supervisor needs to plan next actions | Deploy done, report created |
| Error or incident | Early response | API down, data inconsistency detected |
| Decision needed | Outside your authority | Policy change proposal, resource request |
| Deadline at risk | Supervisor needs to adjust schedule | Blocker, work stopped |
| Security concern | Urgent response needed | Suspicious access, suspected credential leak |

### Routine Reports (SHOULD)

| Situation | Frequency | Content |
|-----------|-----------|---------|
| Daily summary | Daily (end of day) | Today's work, tomorrow's plan |
| Weekly reflection | Weekly (Friday) | Week's work, issues, next week's plan |
| Long-running task progress | As needed (every 2–3 days) | Progress %, remaining work, blockers |

### When Not to Report

- Memory writes and maintenance (internal work)
- Routine work with no anomalies (Heartbeat `HEARTBEAT_OK` equivalent)
- Tasks where supervisor explicitly said "no report needed"

## Report Format

### SCANA Format

Structure reports with these five elements. Include all only as needed; choose based on context.

| Item | English | Description | MUST/MAY |
|------|---------|-------------|----------|
| Situation | Situation | What is happening now | MUST |
| Cause | Cause | Why (if known) | MAY |
| Action taken | Action taken | What you did | SHOULD |
| Next steps | Next steps | What you'll do / what you need decided | MUST |
| Appendix | Appendix | Related logs, file paths, numbers | MAY |

### Thread Continuation When Replying

When replying to a message from your supervisor, specify `reply_to` and `thread_id` to maintain conversation context.

```
send_message(
    to="manager",
    content="Understood. I will address this by 3pm.",
    intent="report",
    reply_to="20260215_093000_123456",
    thread_id="20260215_090000_000000"
)
```

### Format Example

```
send_message(
    to="manager",
    content="""[Completed] Monthly sales data aggregation

Situation: January 2026 sales data aggregation is complete.
Action taken: Aggregated by department and category, calculated MoM comparison.
Next steps: No action required. Please review the report.
Appendix: /shared/reports/sales_summary_202601.md""",
    intent="report"
)
```

## Report Templates by Type

### Completion Report

When a task has completed successfully.

```
send_message(
    to="manager",
    content="""[Completed] {task_name}

Situation: {task_name} is complete.
Deliverable: {file path or result summary}
Duration: {time spent}
Notes: {if any}""",
    intent="report"
)
```

**Example:**

```
send_message(
    to="manager",
    content="""[Completed] API spec v2.1 update

Situation: Reflected v2.1 changes in the API spec.
Deliverable: /shared/docs/api-spec.md (sections 3.2, 4.1 updated)
Duration: ~45 min
Notes: Added 3 pagination examples.""",
    intent="report"
)
```

### Error Report

When an error or incident has occurred.

```
send_message(
    to="manager",
    content="""[Error] {what happened}

Situation: {description of current state}
Cause: {cause, if known}
Impact: {what is affected, who is affected}
Action taken: {what you tried and the result}
Next steps: {proposed actions / decisions needed}""",
    intent="report"
)
```

**Example:**

```
send_message(
    to="manager",
    content="""[Error] Scheduled batch job failure

Situation: The 9:00 AM data sync batch has failed 3 times in a row.
Cause: External API response timeout (exceeds 30s). Likely provider-side issue.
Impact: Data since this morning not synced. Dashboard figures stuck at yesterday.
Action taken: Extended timeout to 60s and retried — still failed. Checked provider status page — no maintenance notice.
Next steps: Provider needs to be contacted. Please advise.""",
    intent="report"
)
```

### Decision Request (Escalation for Decision-Making)

When a decision is outside your authority.

```
send_message(
    to="manager",
    content="""[Decision needed] {topic}

Situation: {current situation}
Options:
A. {option A} — Pros: {benefits} / Cons: {drawbacks}
B. {option B} — Pros: {benefits} / Cons: {drawbacks}
My recommendation: {A or B} (reason: {why})

Please advise.""",
    intent="report"
)
```

**Example:**

```
send_message(
    to="manager",
    content="""[Decision needed] Log retention change

Situation: Disk usage at 85%, likely to exceed 90% within a week.
Options:
A. Shorten log retention 90→30 days — Pros: ~50GB freed immediately / Cons: Old logs unavailable for investigation
B. Add storage (500GB) — Pros: Keep logs / Cons: ~$50/month cost increase
C. Move old logs to archive storage — Pros: Keep logs + control cost / Cons: 2 days to implement
My recommendation: C (balances cost and data retention)

Please advise.""",
    intent="report"
)
```

### Progress Report

Mid-point report for long-running tasks.

```
send_message(
    to="manager",
    content="""[Progress] {task_name}

Progress: {completed work / overall %}
Remaining: {what's left}
Blockers: {if any, otherwise "none"}
ETA: {on track / behind schedule}""",
    intent="report"
)
```

**Example:**

```
send_message(
    to="manager",
    content="""[Progress] User notification feature implementation

Progress: Phase 1 (design) done, Phase 2 (implementation) ~70% complete
  - Email notification: done
  - Slack notification: done
  - In-app notification: in progress (~1 day left)
Remaining: In-app implementation + test code
Blockers: None
ETA: On track for 2/20 deadline.""",
    intent="report"
)
```

## Daily Summary Format

A one-day recap sent to your supervisor at end of day.

### Template

```
send_message(
    to="manager",
    content="""[Daily Summary] 2026-02-15

■ Done
- {completed task 1}
- {completed task 2}

■ In progress
- {task in progress} (progress: {XX}%, ETA: {deadline})

■ Issues / concerns
- {if any}

■ Tomorrow
- {plan 1}
- {plan 2}""",
    intent="report"
)
```

### Example

```
send_message(
    to="manager",
    content="""[Daily Summary] 2026-02-15

■ Done
- API spec v2.1 update (/shared/docs/api-spec.md)
- Fixed timezone bug in log monitor script

■ In progress
- User notification feature (70%, ETA 2/20)

■ Issues / concerns
- External API responses slowing (no impact yet, monitoring)

■ Tomorrow
- Finish in-app notification implementation
- Start test code""",
    intent="report"
)
```

## Urgent vs Routine Reports

### Urgent Report (Immediate, MUST)

Stop other work and report immediately when:

- **Service outage**: Incident affecting users
- **Data loss or corruption**: Risk of unrecoverable state
- **Security incident**: Suspicious access, suspected information leak
- **Deadline miss confirmed**: Committed deadline will not be met

MUST: Prefix urgent messages with "[URGENT]".
For service outage, data loss, or security incidents requiring immediate human response, also use `call_human`.

```
send_message(
    to="manager",
    content="""[URGENT] Production database latency

Situation: Production DB response time ~10x normal (300ms → 3000ms).
Impact: Web UI page load >10 seconds.
Action: Identifying slow queries. Will update within 10 min.""",
    intent="report"
)
```

### Routine Report (Next Heartbeat or Daily Summary)

These can be batched:

- Progress on on-track tasks
- Minor issues you've already resolved
- Improvement proposals and ideas

## Escalation Decision Flowchart

Decision flow for when to escalate:

```
1. Can you resolve within your authority?
   → Yes: Handle it, then report the result
   → No: Go to step 2

2. Is it urgent? (service impact, data loss risk)
   → Yes: Escalate immediately as [URGENT]
   → No: Go to step 3

3. Can you list clear options?
   → Yes: Report as [Decision needed] with options and recommendation
   → No: Report the situation as-is and ask for guidance
```

## Report Guidelines

### Do (MUST/SHOULD)

- MUST: Specify `intent="report"` in `send_message` (omission causes error)
- MUST: Distinguish fact from assumption. Use "likely", "possibly" for assumptions
- MUST: State impact clearly. What and who is affected
- SHOULD: Include what you tried and the result. Avoid duplicate suggestions from supervisor
- SHOULD: Propose next actions. Don't wait for direction only

### Avoid

- Long background before conclusion (put conclusion first)
- Vague reports ("something went wrong") with no concrete detail
- Mixing multiple unrelated topics in one message (split by topic)
- Hiding or downplaying problems (report accurately)
- Sending more than one `send_message` to the same recipient in one run (max 1; use Board for additional follow-up)
