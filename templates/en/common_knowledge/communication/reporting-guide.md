# Reporting and Escalation

Reporting to your supervisor is essential for org operations.
Reporting at the right time and in the right format supports fast decisions and early detection.

## When to Report

### Immediate Reports (MUST)

Report as soon as you become aware:

| Situation | Reason | Example |
|-----------|--------|---------|
| Task completed | Supervisor needs to plan next steps | Deploy done, report created |
| Error or incident | Early response | API down, data inconsistency |
| Decision needed | Outside your authority | Policy change, resource request |
| Deadline at risk | Supervisor needs to adjust schedule | Blocker, work stopped |
| Security concern | Urgent response needed | Suspicious access, credential leak |

### Routine Reports (SHOULD)

| Situation | Frequency | Content |
|-----------|-----------|---------|
| Daily summary | Daily (end of day) | Today’s work, tomorrow’s plan |
| Weekly reflection | Weekly (Friday) | Week’s work, issues, next week |
| Long-running task | Every 2–3 days | Progress, remaining work, blockers |

### When Not to Report

- Internal memory writes and maintenance
- Routine work with no anomalies (Heartbeat `HEARTBEAT_OK`)
- Tasks where supervisor said "no report needed"

## Report Format

### SCANA Format

Use these five elements as appropriate:

| Item | English | Description | MUST/MAY |
|------|---------|-------------|----------|
| Situation | Situation | What is happening now | MUST |
| Cause | Cause | Why (if known) | MAY |
| Action | Action taken | What you did | SHOULD |
| Next steps | Next steps | What you’ll do / what you need | MUST |
| Appendix | Appendix | Logs, paths, numbers | MAY |

### Example

```
send_message(
    to="manager",
    content="""[Completed] Monthly sales aggregation

Status: January 2026 sales aggregation done.
Action: Aggregated by department and category, added MoM comparison.
Next: No further action. Please review the report.
Appendix: /shared/reports/sales_summary_202601.md"""
)
```

## Report Templates

### Completion Report

```
send_message(
    to="manager",
    content="""[Completed] {task_name}

Status: {task_name} completed.
Deliverable: {path or summary}
Duration: {time}
Notes: {optional}"""
)
```

**Example:**

```
send_message(
    to="manager",
    content="""[Completed] API spec v2.1 update

Status: Updated API spec with v2.1 changes.
Deliverable: /shared/docs/api-spec.md (sections 3.2, 4.1)
Duration: ~45 min
Notes: Added 3 pagination examples."""
)
```

### Error Report

```
send_message(
    to="manager",
    content="""[Error] {what happened}

Status: {current state}
Cause: {cause, if known}
Impact: {what is affected}
Action: {what you tried}
Next: {next steps or decisions needed}"""
)
```

**Example:**

```
send_message(
    to="manager",
    content="""[Error] Scheduled batch failure

Status: Data sync batch has failed 3 times in a row since 9am.
Cause: External API timeout (30s). Likely provider-side issue.
Impact: Data since this morning not synced. Dashboard stuck at yesterday.
Action: Increased timeout to 60s and retried — still fails. Checked provider status page — no notice.
Next: Provider needs to be contacted. Please advise."""
)
```

### Decision Request

```
send_message(
    to="manager",
    content="""[Decision needed] {topic}

Status: {current situation}
Options:
A. {option A} — Pros: {benefits} / Cons: {drawbacks}
B. {option B} — Pros: {benefits} / Cons: {drawbacks}
Recommendation: {A or B} (reason: {why})

Please decide."""
)
```

**Example:**

```
send_message(
    to="manager",
    content="""[Decision needed] Log retention change

Status: Disk at 85%, likely >90% in a week.
Options:
A. 90→30 days retention — Pros: ~50GB freed / Cons: Old logs unavailable
B. Add 500GB storage — Pros: Keep logs / Cons: ~$50/month
C. Move old logs to archive storage — Pros: Keep logs, lower cost / Cons: 2 days to implement
Recommendation: C (balances cost and retention)

Please decide."""
)
```

### Progress Report

```
send_message(
    to="manager",
    content="""[Progress] {task_name}

Progress: {done so far / %}
Remaining: {what’s left}
Blockers: {or none}
ETA: {on track / late}"""
)
```

**Example:**

```
send_message(
    to="manager",
    content="""[Progress] User notification feature

Progress: Phase 1 (design) done, Phase 2 (implementation) ~70%
  - Email: done
  - Slack: done
  - In-app: in progress (~1 day left)
Remaining: In-app implementation + tests
Blockers: None
ETA: On track for 2/20."""
)
```

## Daily Summary Format

One-day recap for your supervisor.

### Template

```
send_message(
    to="manager",
    content="""[Daily Summary] 2026-02-15

■ Done
- {task 1}
- {task 2}

■ In progress
- {task} (progress: {XX}%, ETA: {date})

■ Issues
- {if any}

■ Tomorrow
- {plan 1}
- {plan 2}"""
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

■ Issues
- External API responses slower (no impact yet, monitoring)

■ Tomorrow
- Finish in-app notifications
- Start test suite"""
)
```

## Urgent vs Routine

### Urgent (Immediate, MUST)

Stop other work and report immediately for:

- Service outage affecting users
- Data loss or corruption risk
- Security incident
- Certain deadline miss

Use `【緊急】` (or equivalent) at the start:

```
send_message(
    to="manager",
    content="""[URGENT] Production DB latency

Status: DB response time ~10x normal (300ms → 3000ms).
Impact: Web UI load >10 seconds.
Action: Started identifying slow queries. Will update within 10 min."""
)
```

### Routine (Next Heartbeat or Daily Summary)

These can be batched:

- Progress on on-track tasks
- Minor issues you’ve already resolved
- Improvement ideas

## Escalation Flow

When deciding to escalate:

```
1. Can you fix it within your authority?
   → Yes: Fix, then report
   → No: Go to step 2

2. Is it urgent? (service/data/security impact)
   → Yes: Escalate immediately as [URGENT]
   → No: Go to step 3

3. Can you list clear options?
   → Yes: Report as [Decision needed] with options
   → No: Report situation and ask for guidance
```

## Report Guidelines

### Do (MUST/SHOULD)

- Distinguish fact from assumption (e.g. "likely", "possibly")
- State impact (who/what is affected)
- Include what you tried (avoid duplicate suggestions)
- Propose next actions (don’t only wait for direction)

### Avoid

- Long background before conclusion (put conclusion first)
- Vague reports ("something went wrong") with no detail
- Mixing unrelated topics in one message
- Softening or hiding problems (be accurate)
