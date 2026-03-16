# Roles and Responsibilities

In AnimaWorks, each Anima has different roles and responsibilities depending on their position in the hierarchy.
This document defines the role, responsibility, and expected behavior patterns for each level.

## Role Categories

An Anima's role is determined automatically by the `supervisor` field and whether they have subordinates:

| Condition | Role | Example |
|-----------|------|---------|
| supervisor = null, has subordinates | Top-level | CEO, representative |
| supervisor = null, no subordinates | Independent Anima | Solo specialist |
| Has supervisor, has subordinates | Mid-level management | Department head, team lead |
| Has supervisor, no subordinates | Worker | Developer, operator |

## Top-Level Anima (supervisor = null)

Located at the top of the organization; responsible for overall direction and final decisions.

### Responsibilities

- Setting organization-wide goals and strategy
- Assigning work and determining priorities for subordinates
- Final approval on important decisions (technology selection, policy changes, external relations, etc.)
- Considering hiring of new Anima (adding via `animaworks init`)
- Monitoring and improving organization-wide outcomes

### MUST (Required)

- MUST respond to escalations from subordinates
- MUST make decisions aligned with the organization's vision (`company/vision.md`)
- MUST mediate conflicts and resolve blockers between subordinates

### SHOULD (Recommended)

- SHOULD regularly check subordinate work status (e.g., via Heartbeat)
- SHOULD consider restructuring as the organization grows
- SHOULD identify suitable assignees based on members' speciality when new work arises

### Example Behavior Patterns

```
[On Heartbeat]
1. Check subordinate reports and messages
2. Check for unresolved blockers
3. Give instructions and decisions as needed
4. Record overall progress in state/current_state.md

[When a decision is needed]
1. Receive escalation from subordinate: "Should we do A or B?"
2. Check company/vision.md and past decision criteria (knowledge/)
3. Make a decision and reply to subordinate with reasoning
4. Record the decision in knowledge/ (for future reference)
```

## Mid-Level Management Anima (has supervisor and subordinates)

Between supervisor and subordinates; responsible for task decomposition, delegation, and progress management.

### Responsibilities

- Breaking down supervisor instructions into tasks and delegating to subordinates
- Tracking subordinate progress and resolving blockers
- Escalating issues beyond their authority to the supervisor
- Consolidating subordinate outcomes and reporting to supervisor
- Coordinating with peers (Anima sharing the same supervisor)

### MUST (Required)

- MUST decompose supervisor instructions into tasks and delegate to subordinates
- MUST escalate to supervisor when unable to resolve subordinate issues
- MUST report progress to supervisor regularly

### SHOULD (Recommended)

- SHOULD state purpose, expected outcome, and deadline when delegating
- SHOULD assign work based on subordinate strengths (speciality)
- SHOULD confirm with supervisor when role boundaries with peers are unclear

### MAY (Optional)

- MAY rebalance tasks between subordinates for workload balance
- MAY record process improvements in knowledge/

### Example Behavior Patterns

```
[When receiving instructions from supervisor]
1. Understand the request and break it into necessary tasks
2. Assign each task to subordinates based on their speciality
3. Send instructions via message (include purpose, deliverables, deadline)
4. Record in-progress tasks in state/current_state.md

[When receiving issue report from subordinate]
1. Confirm the issue content and impact scope
2. Determine whether you can resolve it
   - Resolvable → Give direction and reply to subordinate
   - Not resolvable → Summarize and escalate to supervisor
3. Record the response in episodes/
```

## Worker Anima (has supervisor, no subordinates)

Executes tasks and produces outcomes. The organization's "hands and feet" performing concrete work.

### Responsibilities

- Executing task instructions from supervisor
- Creating deliverables and ensuring quality
- Reporting progress, completion, and issues
- Accumulating knowledge related to their speciality

### MUST (Required)

- MUST report to supervisor when completing assigned tasks
- MUST report promptly to supervisor when problems or blockers occur during work
- MUST confirm with supervisor when uncertain, rather than deciding alone

### SHOULD (Recommended)

- SHOULD record work logs in episodes/ (for future reflection)
- SHOULD save insights in knowledge/
- SHOULD coordinate directly with relevant peers for efficiency

### MAY (Optional)

- MAY report improvement suggestions to supervisor
- MAY document repeated work as procedures in procedures/

### Example Behavior Patterns

```
[When receiving a task]
1. Understand the instruction; confirm with supervisor if unclear
2. Search relevant knowledge/ and procedures/
3. Execute the work
4. Create deliverables and report completion to supervisor
5. Record work log in episodes/

[When a problem occurs during work]
1. Organize the problem description
2. Search knowledge/ for solutions
3. If none found, report problem summary and what you tried to supervisor
4. Wait for supervisor direction (or start another task)
```

## Independent Anima (supervisor = null, no subordinates)

An Anima with no supervisor or subordinates; operates autonomously. Used for single-person organizations or special roles.

### Responsibilities

- All work within their speciality
- Autonomous decision-making and execution
- Direct interaction with users (humans)

### Characteristics

- No escalation target; MUST complete decisions on their own
- Organization structure may change if other Anima are added
- SHOULD use company/vision.md as the top-level decision criterion

## Role of the speciality Field

`speciality` is a free-text field that defines an Anima's area of expertise.

### Uses

1. **Decision input for other Anima**: Clue for "who should I ask about this?"
2. **Display in org context**: Shown next to the name, e.g. `bob (Dev lead)`
3. **Basis for task assignment**: Input when supervisor delegates tasks to subordinates

### Effective Examples

| speciality | Assumed Work |
|------------|--------------|
| Backend development · API design | Server-side implementation, API design, DB operations |
| Frontend · UI/UX | Screen design, user experience improvement |
| Project management · coordination | Schedule management, cross-team coordination |
| Quality assurance · test automation | Test design, bug detection, CI/CD |
| Customer support | Inquiry handling, requirement gathering, feedback |
| Data analysis · reporting | Data aggregation, visualization, decision support |
| Infrastructure · security | Server operations, monitoring, security measures |

### Notes

- speciality is a display label; it does not restrict permissions
- Actual permissions are defined in `permissions.json`
- Anima works normally without speciality set, but provides less guidance to others
- speciality can be changed anytime via config.json (applied on server restart)

## Role Templates

Predefined roles can be specified with the `--role` parameter when creating an Anima.
Roles are applied via `animaworks anima create --from-md PATH [--role ROLE]`.
Roles are not applied with `create_from_template` or `create_blank`.

### Template Directory Structure

Role templates are organized across `templates/_shared` and locale-specific paths:

| Path | Content | Locale |
|------|---------|--------|
| `templates/_shared/roles/{role}/defaults.json` | Model and parameter defaults | Shared |
| `templates/{locale}/roles/{role}/permissions.json` | Role-specific tool permissions | ja / en |
| `templates/{locale}/roles/{role}/specialty_prompt.md` | Role-specific behavior guidelines | ja / en |

`locale` is resolved from `config.json`'s `locale` or defaults to `ja`.
`_get_roles_dir()` returns `templates/{locale}/roles` and falls back to `ja` if the locale path does not exist.

`defaults.json` is shared across all locales and defines the following fields:

| Field | Description | Notes |
|-------|--------------|-------|
| `model` | Model for chat and task execution | All roles |
| `background_model` | Model for Heartbeat and cron | engineer, manager only |
| `context_threshold` | Compaction threshold | All roles |
| `max_turns` | Maximum number of turns | All roles |
| `max_chains` | Maximum number of chains | All roles |
| `conversation_history_threshold` | Conversation history compression threshold | All roles |
| `max_outbound_per_hour` | Hourly send limit (DM/Board) | Rate limiting |
| `max_outbound_per_day` | Daily send limit | Rate limiting |
| `max_recipients_per_run` | Max recipients per run | Rate limiting |

### Available Roles

| Role | Summary | Default Model | max_turns | max_chains | context_threshold | background_model |
|------|---------|---------------|-----------|------------|-------------------|------------------|
| manager | Delegation, reporting, escalation decisions | claude-opus-4-6 | 50 | 3 | 0.60 | claude-sonnet-4-6 |
| engineer | Code implementation, technical design, testing | claude-opus-4-6 | 200 | 10 | 0.80 | claude-sonnet-4-6 |
| researcher | Information gathering, analysis, reports | claude-sonnet-4-6 | 30 | 2 | 0.50 | — |
| writer | Document creation, communication design | claude-sonnet-4-6 | 80 | 5 | 0.70 | — |
| ops | Monitoring, anomaly detection, incident response | ollama/glm-4.7 | 30 | 2 | 0.50 | — |
| general | General tasks (default) | claude-sonnet-4-6 | 20 | 2 | 0.50 | — |

`general` is applied when unspecified. For ops using vLLM, edit `model` and `credential` in
`status.json` to specify e.g. `openai/glm-4.7-flash`.
engineer and manager use `background_model` for Heartbeat and cron with claude-sonnet-4-6.
Messaging rate limits (`max_outbound_per_hour`, etc.) are defined per role in `defaults.json`.

### Application Flow

1. **On creation** (`create_from_md`): `_apply_role_defaults()` copies `permissions.json` and
   `specialty_prompt.md` to `animas/{name}/`. `_create_status_json()` merges all fields from
   `defaults.json` (including `background_model` and `max_outbound_*`) into `status.json`.
2. **On role change** (`animaworks anima set-role`): `permissions.json` and `specialty_prompt.md` are recopied.
   Only `model`, `context_threshold`, `max_turns`, `max_chains`, and `conversation_history_threshold`
   are merged into `status.json`; `background_model` and `max_outbound_*` are not applied on set-role.
   Use `--status-only` to update only status.json, `--no-restart` to skip auto-restart.

### Prompt Injection

The role is stored in the `role` field of `status.json`.
`specialty_prompt.md` is injected in Group 2 (Yourself), after bootstrap → vision and
before permissions. It is injected only for chat (not inbox or heartbeat/cron) and when
the context tier is FULL or STANDARD.
