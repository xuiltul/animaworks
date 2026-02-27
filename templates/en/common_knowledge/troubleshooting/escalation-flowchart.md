# Escalation Flowchart

Flowchart for deciding whether to fix a problem yourself or escalate.
Use when unsure; skip when the fix is clearly within your scope.

---

## Decision Flow

When something goes wrong, follow these steps in order.

### Step 1: Classify the Problem

Assign one type:

| Type | Description | Examples |
|------|-------------|----------|
| **A. Technical** | Tools or system behavior | Tool errors, permission denied, missing files |
| **B. Operational** | How to proceed or decide | Unclear spec, priority, blockers |
| **C. Interpersonal** | Coordination with other Anima | No reply, conflicting instructions, unclear owner |
| **D. Urgent** | Needs immediate action | Data at risk, security concern |

### Step 2: Assess Urgency

| Urgency | Criteria | Action |
|---------|----------|--------|
| **High** | Data loss or security risk if ignored | MUST: Report to supervisor immediately |
| **High** | Other Anima’s work completely blocked | MUST: Report to supervisor immediately |
| **Medium** | Your work blocked but you can do other tasks | SHOULD: Report to supervisor within 1 hour |
| **Low** | Slower but work can continue | MAY: Report in next Heartbeat |

**If urgency is High** → Go to Step 5 (Escalate).

### Step 3: Try to Solve It Yourself

Go through these steps. Stop when it’s resolved.

1. **Search memory**
   ```
   search_memory(query="keywords related to the problem", scope="all")
   ```
   - Look for similar past problems
   - Check procedures/ for steps

2. **Search common_knowledge**
   ```
   search_memory(query="keywords", scope="common_knowledge")
   ```
   - Check `troubleshooting/common-issues.md`

3. **Try alternatives**
   - Other approaches?
   - If permission issue: other paths or tools?

4. **Timebox self-resolution**
   - Technical: Escalate if not fixed in ~15 min
   - Operational: Escalate if you’re unsure (avoid bad decisions)
   - Interpersonal: One retry, then escalate

### Step 4: Choose Escalation Target

| Problem type | First contact | If that doesn’t help |
|--------------|----------------|---------------------|
| **A. Technical** | Peer (same speciality if applicable) | Supervisor |
| **B. Operational** | Supervisor | — |
| **C. Interpersonal** | Supervisor (mediation) | — |
| **D. Urgent** | Supervisor | — |

**Guidelines:**
- Peer: Same supervisor and relevant speciality
- Supervisor (MUST): Decision needed, other departments involved, or high urgency
- Other departments: Do not contact directly (MUST go via supervisor)

### Step 5: Escalate

Include these in your report (MUST):

1. **Situation**: What is happening
2. **Cause**: Likely cause (or "Investigating")
3. **Actions**: What you tried
4. **Request**: What you need (decision, permission, mediation, etc.)

---

## Escalation Message Templates

### Template 1: Blocker Report

```
send_message(
    to="supervisor_name",
    content="""[Blocked]

■ Situation
Task "Monthly report creation" is blocked.

■ Cause
No read permission for /data/sales/ where sales data lives.

■ Tried
- Checked permissions.md → /data/sales/ not allowed
- Searched for other data sources → none

■ Request
Please grant read access to /data/sales/."""
)
```

### Template 2: Decision Request

```
send_message(
    to="supervisor_name",
    content="""[Decision needed]

■ Situation
Task "Customer support flow improvement" has two approaches.

■ Options
A: Incremental changes (effort: low, risk: low, impact: medium)
B: Full redesign (effort: high, risk: medium, impact: high)

■ Recommendation
Recommend A. Reason: Current flow has limited issues; incremental change is enough.

■ Request
Please decide the approach."""
)
```

### Template 3: Technical Question to Peer

```
send_message(
    to="peer_name",
    content="""[Technical question]

Hitting Slack API rate limit.

■ Situation
- Trying to send 100+ messages in batch
- Getting 429 Too Many Requests around 50

■ Question
Do you have experience with Slack API rate limits?
Considering spacing out batches but unsure of a good interval."""
)
```

### Template 4: Urgent Report

```
send_message(
    to="supervisor_name",
    content="""[URGENT]

■ Situation
Auth errors keep coming from external API (XXX service).

■ Impact
- YYY task fully blocked
- ZZZ task may also be affected (same API)

■ Tried
- Retried 3 times → all failed
- Cannot validate API key myself

■ Request
Please check API key and investigate impact."""
)
```

---

## Example Scenarios

### Scenario 1: Unclear Instructions

**Situation**: Supervisor said "create a report" but period, format, and recipient are unclear.

**Good response**:
1. List what you can infer
2. Ask specific questions

```
send_message(
    to="supervisor_name",
    content="""About the report, need to confirm:

1. Period: This month?
2. Format: Same Markdown as last time?
3. Location: Save in knowledge/?

If this is correct, I'll proceed."""
)
```

**Bad**:
- Proceeding without clarifying
- Replying only "Instructions unclear" without questions

### Scenario 2: Conflicting Instructions

**Situation**: Supervisor said "prioritize A" but another Anima said "do B first."

**Good response**:
1. Prioritize your supervisor (MUST)
2. Report the situation to your supervisor

```
send_message(
    to="supervisor_name",
    content="""[Priority check]

Currently on task A per your instruction.
XXX asked me to prioritize task B.
I'll keep task A as priority unless you say otherwise.

Task B details: YYY (from XXX)"""
)
```

### Scenario 3: Repeated Errors

**Situation**: Chatwork API send failed 3 times.

**Good response**:
1. Log errors
2. Escalate after 3 failed retries
3. Start another unblocked task

```
# Log
write_memory_file(
    path="state/current_task.md",
    content="## Blocked\n\nChatwork API errors\n- 10:00 - 500 Internal Server Error\n- 10:05 - 500 Internal Server Error\n- 10:10 - 500 Internal Server Error\n\nReported to supervisor. Working on other tasks.",
    mode="overwrite"
)

# Report
send_message(
    to="supervisor_name",
    content="[Blocked] Chatwork API returned 500 three times in a row. Possible external outage. Waiting for recovery while working on other tasks."
)
```

### Scenario 4: Found Problem Outside Your Scope

**Situation**: You found inconsistency in data managed by another Anima.

**Good response**:
1. Record the finding
2. Report to your supervisor (do not contact other department directly)

```
send_message(
    to="supervisor_name",
    content="""[Info share]

Found an inconsistency during my work. Not my area but sharing.

■ Finding
/data/reports/monthly.md total and /data/sales/summary.md don't match.
- monthly.md: 1,234,567
- summary.md: 1,234,000

■ Context
Noticed while creating monthly report.

Not sure if action is needed. Up to you."""
)
```

---

## Escalation Checklist

Escalate (MUST) if any applies:

- [ ] A wrong decision could have serious impact
- [ ] Action needed is outside your permissions
- [ ] Other departments are affected
- [ ] No solution found in 15+ min
- [ ] Same issue happened 2+ times
- [ ] It involves security or data safety

You may (MAY) try to solve it yourself if:

- [ ] You’ve solved a similar issue before
- [ ] procedures/ has steps for it
- [ ] common_knowledge/ has guidance
- [ ] It stays within your permissions
- [ ] Failure would have limited impact
