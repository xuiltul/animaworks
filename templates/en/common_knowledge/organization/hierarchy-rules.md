# Hierarchy Rules

AnimaWorks communication follows rules based on the org hierarchy.
This doc defines rules for each relationship (supervisor, subordinate, peer, other departments).

## Principles

- All internal communication uses `send_message` (messaging)
- Sharing internal state directly is not allowed; summarize and interpret in your own words
- Each message must have a clear purpose (what you want or what you need)

## Communicating with Your Supervisor

Your supervisor is the Anima in your `supervisor` field.
Shown in the system prompt "Supervisor" section.

### MUST

- Report task completion
- Report problems and blockers promptly
- Consult when decisions exceed your authority
- Ask for clarification when instructions are unclear

### SHOULD

- Report progress on long tasks
- When reporting problems, include what happened, what you tried, and what you suggest
- Keep reports short; put details in files and reference them

### Examples

Task completion:
```
To bob: API endpoints implemented.
- GET /api/users — list users
- POST /api/users — create user
- All tests pass.
Ready for next task.
```

Problem report:
```
To bob: DB timeouts are frequent.
- Started around 2pm
- Tried: larger pool (no change)
- Suspect: connection limit
- Suggestion: Increase max connections to 100. Please confirm.
```

Decision request:
```
To alice: Need decision on caching.
- Option A: Redis (fast, higher infra cost)
- Option B: File cache (lower cost, limited speed)
- I recommend A for speed; cost is your call.
```

## Supervisor Tools

Anima with subordinates get organizational tools automatically.

### Tool List

| Tool | Scope | Description |
|------|-------|-------------|
| `org_dashboard` | All subordinates (recursive) | Show process status, activity, tasks in a tree |
| `ping_subordinate` | All subordinates (recursive) | Check liveness (omit name for all) |
| `read_subordinate_state` | All subordinates (recursive) | Read subordinate `current_task.md` and `pending.md` |
| `delegate_task` | Direct subordinates only | Delegate (add to queue, send DM, track) |
| `task_tracker` | Your delegated tasks | Track delegated task progress |
| `disable_subordinate` | Direct subordinates | Disable subordinate (~30s to stop) |
| `enable_subordinate` | Direct subordinates | Re-enable |
| `set_subordinate_model` | Direct subordinates | Change model (needs restart_subordinate) |
| `restart_subordinate` | Direct subordinates | Restart process (~30s) |

`check_permissions` is available to all Anima.

### Delegation Flow

1. Run `delegate_task(name="dave", instruction="...", deadline="2026-02-20")`
2. Task is added to dave's queue
3. DM sent to dave (intent="delegation")
4. Tracking entry created in your queue
5. Use `task_tracker()` to monitor

### Regular Status Checks

```
org_dashboard()                        # Tree of all subordinates
read_subordinate_state(name="dave")    # dave's current and pending tasks
ping_subordinate()                     # Liveness check for all
```

## Communicating with Subordinates

Subordinates are Anima whose `supervisor` is your name (shown in the system prompt).

### MUST

- Include in task instructions:
  - **Purpose**: Why this task is needed
  - **Expected outcome**: What to produce
  - **Deadline and priority**: When and how urgent
- Give feedback after completion
- Regularly check status with `org_dashboard`

### SHOULD

- Consider speciality when assigning
- Be specific; avoid vague instructions
- If multiple subordinates, clarify dependencies before instructing

### MAY

- Assign slightly challenging tasks for growth
- Name collaborators when coordination is needed

### Examples

Task instruction:
```
To dave: Please implement user auth API.
- Purpose: Enable login from frontend
- Deliverable: POST /api/auth/login (returns JWT)
- Spec: Email+password auth. Return 401 on failure
- Deadline: Friday
- Note: eve is building the login UI. You can coordinate directly on API spec.
```

Feedback:
```
To dave: Auth API looks good.
- Login/logout work
- One fix: token expiry not set. Please add 24h expiry
- Otherwise fine. Report again after the fix.
```

## Communicating with Peers

Peers are Anima with the same `supervisor` (shown in the system prompt).

### SHOULD

- Work directly with peers on shared tasks
- Share when your work affects theirs
- Clarify responsibilities when working together

### MAY

- Ask peers about their speciality
- Request reviews and feedback

### Examples

Coordination:
```
To eve: Auth API is ready.
- Endpoint: POST /api/auth/login
- Request: { "email": "...", "password": "..." }
- Response: { "token": "JWT", "expires_in": 86400 }
- Error: 401 { "error": "Invalid credentials" }
Use this in the login screen. Ask if you need anything.
```

Consultation:
```
To carol: Need UX advice on the admin UI.
- User list table has 20+ columns
- How should we handle layout?
Would appreciate your input.
```

## Communicating with Other Departments

Other departments = Anima with a different `supervisor`.
They do not appear in your peers list.

### MUST

- Do not contact other departments directly
- When cross-department work is needed, tell your supervisor
- Chain: you → your supervisor → their supervisor → them

### Example Path

dave (bob) wants to contact frank (carol):

```
Correct:
  dave → bob (supervisor) → alice → carol (frank's supervisor) → frank

Wrong:
  dave → frank (direct — not allowed)
```

### Example Message

```
To bob: Need to know the CSV format for customer data export that frank (sales) needs.
Can you ask carol to check?
```

### Why No Direct Contact

- Supervisor loses visibility
- Risk of conflicting instructions
- Supervisors should control prioritization

## Top-Level Anima and Humans

Top-level Anima (no supervisor) use `call_human` to contact human admins.

### MUST

- Use `call_human` when they detect problems, errors, or incidents
- Report when decisions exceed their scope
- Report when they cannot resolve subordinate escalations

### SHOULD

- Report important completions
- Report concerns about subordinate workload

### MAY

- Skip report when routine checks find nothing
- Skip report for minor self-fixed issues

### Examples

Problem:
```
call_human:
  subject: "Production API latency detected"
  body: |
    API latency ~3x normal since 2pm.
    - Impact: All user-facing endpoints
    - Tried: connection pool check (OK)
    - Suspect: DB load
    - Suggestion: Consider DB scale-up
  priority: high
```

Decision request:
```
call_human:
  subject: "Feature release approval"
  body: |
    User auth feature done, all tests pass.
    Request approval for production release.
  priority: normal
```

## Emergency Exceptions

In emergencies, normal rules can be relaxed.

### Emergency Definition

- System or service outage
- Security incident
- Supervisor unreachable for a long time
- Deadline at risk and normal path is too slow

### Emergency Exceptions (MAY)

- Contact other departments directly (MUST report to supervisor afterward)
- Skip a level and contact supervisor’s supervisor (MUST report direct supervisor)
- Make decisions that normally need approval (MUST report and get approval afterward)

### Emergency Message Example

```
To frank: [URGENT] dave here. Contacting you directly due to emergency.
Production user data mismatch detected.
Can you check if customers have reported issues?
I'm also reporting to bob and carol.
```

Follow-up to supervisor:
```
To bob: [Follow-up] I contacted frank (carol’s team) directly during the prod incident.
- Reason: Needed fast check on customer impact
- Action: Asked frank about customer reports
- Result: 3 reports, frank is handling
Apologies for not going through you first.
```

## Rule Summary

| Priority | Rule | Level |
|----------|------|-------|
| 1 | Report problems and escalate to supervisor | MUST |
| 2 | Report task completion | MUST |
| 3 | Include purpose, outcome, deadline in task instructions | MUST |
| 4 | No direct contact with other departments | MUST |
| 5 | Direct coordination with peers | SHOULD |
| 6 | Report progress | SHOULD |
| 7 | Apply emergency exceptions only when justified | MAY |
