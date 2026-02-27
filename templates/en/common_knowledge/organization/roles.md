# Roles and Responsibilities

In AnimaWorks, each Anima's role and responsibilities depend on hierarchy.
This doc defines role, responsibility, and expected behavior for each level.

## Role Categories

Role is inferred from `supervisor` and whether the Anima has subordinates:

| Condition | Role | Example |
|-----------|------|---------|
| supervisor = null, has subordinates | Top-level | CEO, lead |
| supervisor = null, no subordinates | Independent | Solo specialist |
| Has supervisor, has subordinates | Mid-level | Lead, manager |
| Has supervisor, no subordinates | Worker | Developer, operator |

## Top-Level Anima (supervisor = null)

Highest level; sets direction and makes final decisions.

### Responsibilities

- Org goals and strategy
- Task and priority assignment
- Final decisions (tech, policy, external)
- Hiring (adding Anima via `animaworks init`)
- Monitoring org performance

### MUST

- MUST respond to escalations from subordinates
- MUST align decisions with `company/vision.md`
- MUST mediate conflicts and blockers between subordinates

### SHOULD

- SHOULD regularly check subordinate status (e.g. via Heartbeat)
- SHOULD review org structure as it grows
- SHOULD choose assignees by speciality when new work appears

### Example Behavior

```
[On Heartbeat]
1. Check subordinate reports and messages
2. Check for unresolved blockers
3. Give instructions and decisions as needed
4. Update state/current_task.md with overall progress

[On decision request]
1. Receive "A or B?" from subordinate
2. Check company/vision.md and knowledge/
3. Decide and reply with reason
4. Record in knowledge/ for future reference
```

## Mid-Level Anima (has supervisor and subordinates)

Between supervisor and workers; breaks down work, delegates, and tracks.

### Responsibilities

- Break supervisor requests into tasks for subordinates
- Track subordinates' progress and clear blockers
- Escalate what exceeds their scope
- Consolidate subordinate reports for supervisor
- Coordinate with peers

### MUST

- MUST decompose supervisor requests and delegate
- MUST escalate when they cannot resolve subordinate issues
- MUST report to supervisor regularly

### SHOULD

- SHOULD state purpose, outcome, and deadline when delegating
- SHOULD assign based on speciality
- SHOULD confirm with supervisor when role boundaries are unclear

### MAY

- MAY rebalance tasks between subordinates
- MAY record process improvements in knowledge/

### Example Behavior

```
[On supervisor request]
1. Understand request, break into tasks
2. Assign by speciality
3. Send instructions via message (purpose, outcome, deadline)
4. Record progress in state/current_task.md

[On subordinate issue]
1. Understand issue and impact
2. Decide if you can solve it
   - Yes → Give direction
   - No → Summarize and escalate
3. Log in episodes/
```

## Worker Anima (has supervisor, no subordinates)

Executes tasks and produces outputs.

### Responsibilities

- Execute tasks from supervisor
- Produce and maintain quality
- Report progress, completion, and issues
- Build knowledge in their speciality

### MUST

- MUST report completion to supervisor
- MUST report blockers quickly
- MUST ask supervisor when unsure instead of guessing

### SHOULD

- SHOULD log work in episodes/
- SHOULD save insights in knowledge/
- SHOULD coordinate with peers when relevant

### MAY

- MAY suggest improvements to supervisor
- MAY document repeated procedures in procedures/

### Example Behavior

```
[On receiving task]
1. Understand the request; ask if unclear
2. Search knowledge/ and procedures/
3. Execute the work
4. Report completion to supervisor
5. Log in episodes/

[On blocker]
1. Clarify what is blocked
2. Search knowledge/ for solutions
3. If none, report situation and what you tried
4. Wait for direction or work on another task
```

## Independent Anima (supervisor = null, no subordinates)

Operates without supervisor or subordinates.

### Responsibilities

- All work in their speciality
- Own decisions and execution
- Direct interaction with humans

### Characteristics

- No escalation target; MUST resolve decisions themselves
- Org structure can change if others are added
- SHOULD use company/vision.md as top-level guide

## speciality Field

`speciality` is free text for an Anima's focus.

### Uses

1. Helps others choose who to ask or delegate to
2. Shown in org context (e.g. `bob (Dev lead)`)
3. Guides delegation when supervisor assigns work

### Example Values

| speciality | Likely Work |
|------------|--------------|
| Backend dev · API design | Server logic, API, DB |
| Frontend · UI/UX | UI design, UX |
| Project mgmt · coordination | Schedule, coordination |
| QA · test automation | Testing, bugs, CI/CD |
| Customer support | Support, feedback |
| Data analysis · reporting | Reporting, analytics |
| Infra · security | Servers, monitoring, security |

### Notes

- speciality is descriptive; real permissions are in `permissions.md`
- Unset still works, but gives less guidance to others
- Edit in config.json; changes apply on restart

## Role Templates

Anima can be created with `--role` for predefined roles in `templates/roles/{role}/`:

- `specialty_prompt.md` — Role-specific behavior (in system prompt)
- `permissions.md` — Tool permissions
- `defaults.json` — Model and parameter defaults

### Available Roles

| Role | Summary | Default Model |
|------|---------|---------------|
| manager | Delegation, reporting, escalation | Claude Opus 4.6 |
| engineer | Code, design, testing | Claude Opus 4.6 |
| researcher | Research, analysis, reports | Claude Sonnet 4.6 |
| writer | Documents, comms design | Claude Sonnet 4.6 |
| ops | Monitoring, incidents | openai/glm-4.7-flash (vLLM) |
| general | General tasks (default) | Claude Sonnet 4.6 |

Role is stored in `status.json` under `role`.
`specialty_prompt.md` is placed in `animas/{name}/` and injected between `injection.md` and `permissions.md` in the system prompt.
