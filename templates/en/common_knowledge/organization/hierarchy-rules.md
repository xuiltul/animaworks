# Hierarchy Rules

AnimaWorks defines rules for communication between Anima based on organizational hierarchy.
This document defines rules for each relationship (supervisor, subordinate, peer, other departments).

## Basic Principles

- All internal communication MUST use `send_message` (messaging)
- Direct sharing of internal state is prohibited; convey information in your own words, compressed and interpreted
- Each message MUST have a clear purpose. Include "what you want done" and "what you want to convey" as MUST

## Communicating with Your Supervisor (Reports, Updates, Consultations)

Your supervisor is the Anima specified in the `supervisor` field.
It is displayed in the "Supervisor" section of the system prompt.

### MUST (Required)

- MUST report results when a task is completed
- MUST report promptly when problems or blockers occur during work
- MUST consult when decisions exceed your authority
- MUST ask for clarification when instructions from your supervisor are unclear

### SHOULD (Recommended)

- SHOULD report intermediate progress on long-running tasks
- When reporting problems, SHOULD include "what happened," "what you tried," and "proposal" as a set
- Keep reports concise. SHOULD put details in files and send references when needed

### Message Examples

Task completion report:
```
To bob: API endpoint implementation is complete.
- GET /api/users — List users
- POST /api/users — Create new user
- All tests pass.
Please assign the next task if any.
```

Problem report:
```
To bob: DB connection timeouts are occurring frequently.
- Occurrence: Since around 2:00 PM today
- What I tried: Increased connection pool size (no improvement)
- Hypothesis: Connection limit may have been reached
- Proposal: I would like to confirm whether we can increase max connection pool from 50 to 100.
```

Decision consultation:
```
To alice: I would like your decision on the caching strategy.
- Option A: Introduce Redis (fast but higher infra cost)
- Option B: File cache (low cost but limited speed)
- I recommend A, but cost decisions are up to you.
```

## Supervisor Tools

Anima with subordinates have dedicated tools for organizational management automatically enabled.

### Tool List

| Tool | Scope | Description | Main Parameters |
|------|-------|-------------|-----------------|
| `org_dashboard` | All subordinates (recursive) | Display process status, last activity, current tasks, and task count for all subordinates in a tree | None |
| `ping_subordinate` | All subordinates (recursive) | Liveness check for subordinates. Omit `name` for all at once, specify for a single Anima | `name` (optional) |
| `read_subordinate_state` | All subordinates (recursive) | Read subordinate's `state/current_task.md` and `state/pending.md` | `name` (required) |
| `delegate_task` | Direct subordinates only | Delegate task (add to subordinate queue + send DM + create tracking entry on your side) | `name`, `instruction`, `deadline` (required), `summary` (optional) |
| `task_tracker` | Your delegated tasks | Track progress of tasks delegated via `delegate_task` from descendant queue | `status` (optional: "all"/"active"/"completed", default "active") |
| `audit_subordinate` | All descendants (recursive) | Generate activity timeline or statistics summary. Omit `name` to audit all descendants at once (merged timeline) | `name` (optional), `mode` (optional: `"report"`/`"summary"`, default `"report"`), `hours` (optional: 1–168, default 24), `direct_only` (optional: boolean), `since` (optional: `"HH:MM"` start time today, takes precedence over hours) |
| `disable_subordinate` | All descendants (recursive) | Disable descendant (status.json enabled=false, process stops in ~30 seconds) | `name` (required), `reason` (optional) |
| `enable_subordinate` | All descendants (recursive) | Re-enable a disabled descendant | `name` (required) |
| `set_subordinate_model` | All descendants (recursive) | Change descendant's model (updates status.json; `restart_subordinate` required to apply) | `name`, `model` (required), `reason` (optional) |
| `set_subordinate_background_model` | All descendants (recursive) | Change descendant's background model (Heartbeat/Inbox/Cron). Empty string to clear. `restart_subordinate` required to apply | `name`, `model` (required), `credential`, `reason` (optional) |
| `restart_subordinate` | All descendants (recursive) | Restart descendant process (restart_requested flag, restarts in ~30 seconds) | `name` (required), `reason` (optional) |

`check_permissions` is available to all Anima (view your permission list).

### Task Delegation Flow

1. Execute `delegate_task(name="dave", instruction="...", deadline="...")`
   - `deadline` can be relative format (`30m`, `2h`, `1d`) or ISO8601 format (e.g., `2026-02-20`)
   - `summary` is optional (first 100 characters of instruction when omitted)
2. Task is automatically added to dave's task queue (`state/task_queue.jsonl`)
3. Task JSON is written to dave's `state/pending/` for immediate execution
4. DM is automatically sent to dave
5. Tracking entry is created in your queue (status="delegated")
6. Use `task_tracker(status="active")` to track in-progress delegated tasks (`status="all"` for all, `status="completed"` for completed only)

### Regular Status Checks (Should Be Done Periodically)

```
org_dashboard()                        # Display subordinate status in tree
read_subordinate_state(name="dave")    # Check dave's current and pending tasks
ping_subordinate()                     # Liveness check for all
audit_subordinate(name="dave")         # dave's activity timeline (last 24 hours)
audit_subordinate(name="dave", mode="summary", hours=168)  # dave's statistics summary (last 7 days)
audit_subordinate()                    # Merged timeline of all subordinates (last 24 hours)
audit_subordinate(direct_only=true)    # Merged timeline of direct subordinates only
audit_subordinate(since="09:00")       # All subordinates since 9:00 today
audit_subordinate(name="dave", since="13:00")  # dave since 13:00 today
```

#### audit_subordinate Modes

| Mode | Output | Use Case |
|------|--------|----------|
| `report` (default) | Chronological timeline with icons, timestamps, and content per event. Tool usage is aggregated at the bottom | Understanding "what happened today" |
| `summary` | Statistical overview: event counts, task status, communication peers, error details | Quantitative activity assessment |

When `name` is omitted and multiple Animas are targeted, `report` mode generates a **merged timeline** that interleaves events from all Animas in chronological order.

### Organization Expansion (Creating New Anima)

Anima that hold `skills/newstaff.md` can create new Anima with the `create_anima` tool.
Specify a character sheet (`character_sheet_content` or `character_sheet_path`).
After creation, the new Anima is automatically registered in config.json and activated on server reload.
If `supervisor` is omitted, the calling Anima is set as the supervisor.

## Communicating with Subordinates (Instructions, Confirmation)

Subordinates are Anima displayed in the "Subordinates" section of the system prompt.
All Anima that have you set as their `supervisor` are included.

### MUST (Required)

- Task instructions MUST include:
  - **Purpose**: Why this task is needed
  - **Expected deliverable**: What to produce or do
  - **Deadline and priority**: When and how urgent
- MUST provide feedback on completion when receiving reports from subordinates
- MUST regularly check subordinate status with `org_dashboard`

### SHOULD (Recommended)

- SHOULD assign tasks considering subordinate's speciality
- Be specific in instructions. SHOULD avoid vague expressions ("make it look good")
- When you have multiple subordinates, SHOULD clarify task dependencies before instructing

### MAY (Optional)

- MAY assign slightly challenging tasks for subordinate growth
- When coordination between subordinates is needed, MAY instruct with collaborators explicitly named

### Message Examples

Task instruction:
```
To dave: Please implement the user authentication API.
- Purpose: To enable login functionality from the frontend
- Deliverable: POST /api/auth/login endpoint (returns JWT)
- Spec: Email + password authentication. Return 401 on failure
- Deadline: By this Friday
- Related: eve is building the login screen on the frontend in parallel.
  You may coordinate directly with eve on API spec questions.
```

Confirmation and feedback:
```
To dave: I've reviewed the auth API implementation.
- Login and logout both work correctly
- One point: Token expiry does not appear to be set. Please set expiry to 24 hours
- Otherwise no issues. Please report again after the fix.
```

## Communicating with Peers (Coordination, Alignment)

Peers are Anima that share the same `supervisor`.
They are displayed in the "Peers" section of the system prompt.

### SHOULD (Recommended)

- SHOULD communicate directly with peers on related work
- SHOULD share in advance when your work affects a peer's work
- When working together, SHOULD clarify who is responsible for what

### MAY (Optional)

- MAY consult peers on matters related to their speciality
- MAY request reviews and opinions

### Message Examples

Coordination request:
```
To eve: Auth API implementation is complete, sharing with you.
- Endpoint: POST /api/auth/login
- Request: { "email": "...", "password": "..." }
- Response: { "token": "JWT string", "expires_in": 86400 }
- On error: 401 { "error": "Invalid credentials" }
Please use this in the login screen. Feel free to ask if you have questions.
```

Consultation:
```
To carol: I'd like to consult on the admin UI.
Regarding the table layout for the user list screen,
there are 20+ columns. How should we display them?
I'd appreciate your advice from a UX perspective.
```

## Communicating with Other Department Members

Other departments are Anima with a different `supervisor` from yours.
Anima not shown in the peers section are included.

### MUST (Required)

- As a rule, MUST NOT contact other department members directly
- When cross-department coordination is needed, MUST inform your supervisor
- Contact flows through your supervisor to the other department's supervisor, then to the target

### Communication Path

Example: dave (under bob) wants to contact frank (under carol)

```
Correct path:
  dave → bob (your supervisor) → alice (common supervisor) → carol (frank's supervisor) → frank

Incorrect:
  dave → frank (direct contact is prohibited)
```

### Message Example

Request to supervisor (when cross-department coordination is needed):
```
To bob: Regarding the customer data export feature,
I need to know the CSV format details that frank in sales needs.
Could you check via carol?
```

### Why Avoid Direct Contact

- Supervisors lose visibility over the overall picture of work
- Chain of command becomes confused, risking conflicting instructions
- Supervisors in each department lose the opportunity to judge priorities

## Top-Level Anima Contacting Humans

Top-level Anima (no `supervisor` set) have the responsibility to contact human administrators directly.
Use the `call_human` tool for contact.

### MUST (Required)

- MUST contact via `call_human` when problems, errors, or incidents are detected
- MUST report when decisions exceed your authority
- MUST report when you cannot resolve matters escalated from subordinates

### SHOULD (Recommended)

- SHOULD report when important tasks are completed
- SHOULD report when there are concerns about subordinate workload

### MAY (Optional)

- MAY skip reporting when routine checks find no issues
- MAY skip reporting when minor self-repairs are completed

### Message Examples

Problem report:
```
call_human:
  subject: "Production API response latency detected"
  body: |
    API response time has increased to 3x normal since around 2:00 PM.
    - Impact: All user-facing API endpoints
    - What I tried: Connection pool status check (no anomaly)
    - Hypothesis: DB server load may be high
    - Proposal: Please consider scaling up the DB server
  priority: high
```

Decision request:
```
call_human:
  subject: "New feature release approval request"
  body: |
    User authentication feature implementation is complete and all tests pass.
    Request approval for production release.
  priority: normal
```

## Emergency Exception Rules

The following exceptions apply to normal hierarchy rules.

### Definition of Emergency

Apply exception rules MAY when the following apply:

- System failure or service outage has occurred
- Security incident has occurred
- Supervisor has been unresponsive for an extended period
- Deadline is imminent and the normal path would not be in time

### Actions Permitted in Emergencies

- MAY contact other department members directly (MUST report to supervisor afterward)
- MAY skip a level and contact supervisor's supervisor directly (MUST report to direct supervisor as well)
- MAY make decisions that normally require consultation (MUST report and obtain approval afterward)

### Emergency Contact Message Example

```
To frank: [URGENT] dave here. Contacting you directly due to emergency.
User data inconsistency detected in production.
Can you check if customers have reported any issues?
I'm also reporting to bob and carol.
```

Follow-up report to supervisor:
```
To bob: [Follow-up report] I contacted frank (under carol) directly during the production incident.
- Reason: Needed urgent confirmation of customer data inconsistency impact
- Action: Asked frank to confirm customer inquiry status
- Result: 3 inquiries received, frank is handling them
Apologies for the post-facto report. Please review.
```

## Rule Priority Summary

| Priority | Rule | Level |
|----------|------|-------|
| 1 | Problem reports and escalation to supervisor | MUST |
| 2 | Task completion reports | MUST |
| 3 | Include purpose, deliverable, and deadline in task instructions | MUST |
| 4 | No direct contact with other departments | MUST |
| 5 | Direct coordination with peers | SHOULD |
| 6 | Intermediate progress reports | SHOULD |
| 7 | Emergency exception application | MAY |
