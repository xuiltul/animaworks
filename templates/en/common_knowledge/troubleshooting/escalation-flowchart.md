# Escalation Flowchart

A flowchart for deciding whether to "solve it yourself" or "consult/report to someone" when a problem occurs.

Refer to this document when unsure how to proceed. This flowchart is unnecessary for problems you can clearly solve yourself.

---

## Decision Flowchart

When a problem occurs, follow these steps in order.

### Step 1: Identify the Problem Type

Classify the problem into one of the following:

| Type | Description | Examples |
|------|-------------|----------|
| **A. Technical** | Issues related to tools or system behavior | Tool errors, permission denied, missing files |
| **B. Operational** | Issues related to how to proceed with tasks or decisions | Unclear spec, priority judgment, blockers |
| **C. Interpersonal** | Issues related to coordination with other Anima | No reply, conflicting instructions, unclear owner |
| **D. Urgent** | Problems requiring immediate action | Data loss risk, security concern |

### Step 2: Assess Urgency

| Urgency | Criteria | Action |
|---------|----------|--------|
| **High** | Data loss or security risk if left unaddressed | MUST: Report to supervisor immediately |
| **High** | Other Anima's work is completely blocked | MUST: Report to supervisor immediately |
| **Medium** | Your work is blocked but you can work on other tasks | SHOULD: Report to supervisor within 1 hour |
| **Low** | Work efficiency drops but progress is possible | MAY: Report at next Heartbeat |

**If urgency is "High"** → Proceed to Step 5 (Escalate immediately)

### Step 3: Try to Solve It Yourself

Try self-resolution using the following steps. Stop when resolved at any step.

1. **Search memory**
   ```
   search_memory(query="keywords related to the problem", scope="all")
   ```
   - Check if you've experienced the same problem before
   - Check procedures/ for handling methods

2. **Search common knowledge**
   ```
   search_memory(query="keywords related to the problem", scope="common_knowledge")
   ```
   - Check if `troubleshooting/common-issues.md` has a matching problem

3. **Consider alternative approaches**
   - Think about alternative means to achieve the goal
   - If permission denied: try another path; if tool error: try another tool

4. **Time limit for self-resolution**
   - Technical: Escalate if not resolved within 15 minutes
   - Operational: Escalate immediately when unsure (avoid wrong decisions)
   - Interpersonal: Escalate after one retry

### Step 4: Determine Escalation Target

| Problem type | First contact | If that doesn't help |
|--------------|---------------|----------------------|
| **A. Technical** | Peer (when same specialty applies) | Supervisor |
| **B. Operational** | Supervisor | — |
| **C. Interpersonal** | Supervisor (request mediation) | — |
| **D. Urgent** | Supervisor (immediately) | — |

**Guidelines:**
- Peer consultation is OK when: same supervisor and problem relates to their specialty
- MUST report to supervisor when: business decision needed, other departments involved, or high urgency
- Do NOT contact other departments' Anima directly (MUST go via supervisor)

### Step 5: Execute Escalation

Include the following elements in your report message (MUST):

1. **Situation**: What is happening
2. **Cause**: What you think is the cause (or "Investigating" if unknown)
3. **Actions**: What you tried yourself
4. **Request**: What you need from the supervisor (decision, permission grant, mediation, etc.)

**send_message constraints (implementation compliance)**:
- `intent` is REQUIRED: one of `report` or `question`. Use delegate_task for task delegation. Do not omit
- Acknowledgments, thanks, and FYI are not allowed via DM. Use Board (post_channel)
- DM recipient limit per run is set by role/status.json (e.g. general/ops: 2, engineer: 5, manager: 10). One message per recipient. For more recipients than the limit, use Board
- DM and Board share the **same outbound budget** (hourly and 24-hour limits apply). See `communication/sending-limits.md` for details
- **Recipient**: Anima name, or human alias (if configured, delivers externally via Slack/Chatwork etc.)
- **During chat**: Reply to human users directly in text. Use send_message only for other Anima
- **Contacting humans** (unconfigured recipients): Use `call_human`
- For thread replies, specify `reply_to` and `thread_id` to preserve context
- For high urgency requiring immediate human action, consider `call_human` (subject, body, priority)

**post_channel (Board) constraints** (use when notifying 3+ people):
- Channels without metadata (general, ops, etc.) are available to all. Member-only channels allow posting only by members (ACL). If access denied, use `manage_channel(action="info", channel="channel_name")` to check members
- One post per channel per run. Repeated posts to the same channel require cooldown (`heartbeat.channel_post_cooldown_s` in config.json, default 300 seconds)
- Use `@name` in the body for mentions. Mentioned users receive DM notifications

**call_human parameters**:
- `subject` and `body` are required. `priority` is optional (`low` / `normal` / `high` / `urgent`, default `normal`)

---

## Escalation Message Templates

### Template 1: Blocker Report

```
send_message(
    to="supervisor_name",
    content="""[Blocked Report]

■ Situation
Task "Monthly report creation" is blocked.

■ Cause
No read permission for /data/sales/ where sales data is stored.

■ Tried
- Checked permissions.json → /data/sales/ not allowed
- Searched for alternative data sources → none found

■ Request
Please add read permission for /data/sales/.""",
    intent="report"
)
```

### Template 2: Decision Request

```
send_message(
    to="supervisor_name",
    content="""[Decision Request]

■ Situation
Task "Customer support flow improvement" has two possible approaches.

■ Options
A: Incremental changes to existing flow (effort: low, risk: low, impact: medium)
B: Full flow redesign (effort: high, risk: medium, impact: high)

■ My recommendation
Recommend A. Reason: Current flow issues are limited; incremental changes are sufficient.

■ Request
Please decide the approach.""",
    intent="question"
)
```

### Template 3: Technical Question to Peer

```
send_message(
    to="peer_name",
    content="""[Technical Question]

Hitting Slack API rate limit.

■ Situation
- Trying to send 100+ messages in batch
- Getting 429 Too Many Requests around the 50th message

■ Question
Do you have experience with Slack API rate limit workarounds?
Considering spacing out batch processing but unsure of appropriate interval.""",
    intent="question"
)
```

### Template 4: Urgent Report

For high urgency requiring immediate human action, also use `call_human`.

```
send_message(
    to="supervisor_name",
    content="""[Urgent Report]

■ Situation
Auth errors continue from external API (XXX service).

■ Impact
- YYY task fully stopped
- ZZZ task also uses same API; may be affected

■ Tried
- Retried 3 times → all failed
- Cannot verify API key validity myself

■ Request
Please verify API key and investigate impact scope.""",
    intent="report"
)
```

When immediate human notification is needed:
```
call_human(
    subject="[URGENT] External API auth errors continue",
    body="Auth errors continue from XXX service. YYY task stopped. Please verify API key.",
    priority="urgent"
)
```

---

## Common Escalation Scenarios

### Scenario 1: Unclear Instructions

**Situation**: Supervisor said "create a report" but target period, format, and recipient are unclear.

**Correct response**:
1. Organize what you can infer
2. Ask specific questions about unclear points

```
send_message(
    to="supervisor_name",
    content="""Regarding the report creation, need to confirm:

1. Target period: This month?
2. Format: Same Markdown format as last time?
3. Recipient: Save in knowledge/?

If the above is correct, I'll proceed.""",
    intent="question"
)
```

**Do NOT**:
- Proceed with your own interpretation without confirming
- Reply only "Instructions unclear" (without specific questions)
- Omit `intent` in send_message (one of `report` / `question` is required; use delegate_task for task delegation; omission causes errors)

### Scenario 2: Conflicting Instructions from Multiple Supervisors

**Situation**: Direct supervisor said "prioritize A" but another Anima asked "do B first."

**Correct response**:
1. Prioritize direct supervisor's instruction (MUST)
2. Report the situation to your direct supervisor

```
send_message(
    to="direct_supervisor_name",
    content="""[Priority Confirmation]

Currently working on task A per your instruction, but XXX asked me to prioritize task B.
I'll prioritize task A as instructed unless you say otherwise.

Task B details: YYY (request from XXX)""",
    intent="question"
)
```

### Scenario 3: Repeated Errors During Work

**Situation**: Chatwork API message send failed 3 times in a row.

**Correct response**:
1. Record error details
2. Escalate if not resolved after 3 retries
3. Start other unblocked tasks

```
# Record errors
write_memory_file(
    path="state/current_state.md",
    content="## Blocked\n\nChatwork API repeated errors\n- 1st: 10:00 - 500 Internal Server Error\n- 2nd: 10:05 - 500 Internal Server Error\n- 3rd: 10:10 - 500 Internal Server Error\n\nReported to supervisor. Working on other tasks.",
    mode="overwrite"
)

# Report to supervisor
send_message(
    to="supervisor_name",
    content="[Blocked Report] Chatwork API returned 500 three times in a row. Possible external outage. Waiting for recovery while working on other tasks.",
    intent="report"
)
```

### Scenario 4: Found Problem Outside Your Responsibility

**Situation**: Found inconsistency in data managed by another Anima during your work.

**Correct response**:
1. Record the finding
2. Report to your direct supervisor (do NOT contact other department's Anima directly)

```
send_message(
    to="supervisor_name",
    content="""[Info Share]

Found the following inconsistency during my work. Not my responsibility but sharing.

■ Finding
/data/reports/monthly.md total and /data/sales/summary.md don't match.
- monthly.md: ¥1,234,567
- summary.md: ¥1,234,000

■ Context
Noticed while referencing data during monthly report creation.

Whether to act is up to you.""",
    intent="report"
)
```

---

## Checklist When Unsure

Escalate (MUST) if any applies:

- [ ] A wrong decision could have irreversible impact
- [ ] Action needed is outside your permissions
- [ ] Other departments' Anima are affected
- [ ] No solution found in 15+ minutes
- [ ] Same problem occurred 2+ times
- [ ] Involves security or data safety

You may (MAY) try to solve it yourself if:

- [ ] You've solved a similar problem before
- [ ] procedures/ has handling steps
- [ ] common_knowledge/ has a solution
- [ ] Stays within your permissions
- [ ] Failure would have limited impact
