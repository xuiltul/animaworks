# Machine Tool Usage Guide — Common Principles

## Overview

`animaworks-tool machine run` delegates heavy work such as code changes, investigation,
analysis, review, and testing to an external agent (cursor-agent / claude / codex / gemini).

Machine runs in an isolated execution environment without access to AnimaWorks infrastructure.
It cannot use memory, messaging, or organization information — all necessary context must be
included in the plan document.

## Meta-Pattern — Common to All Roles

All phases and all roles MUST follow these 4 steps:

```
1. Anima writes an instruction document (plan)
2. Machine executes based on the plan
3. Machine output is treated as a draft
4. Anima verifies the output and approves or revises
```

**Principles:**
- The initial instruction (what to tell machine to do) MUST be written by the Anima itself
- Intermediate artifact concretization (plan -> impl.plan etc.) may be delegated to machine
- Machine output is always a draft — NEVER pass it to the next phase without Anima approval
- If issues are found, revise the plan and re-delegate to machine, or fix it directly

## Status Management

Plan documents and artifacts carry their status as metadata within the document itself.
The framework does not manage status automatically — Anima decides transitions autonomously.

Place the following metadata block at the top of each document:

```markdown
status: draft | reviewed | approved
author: {anima_name}
date: {YYYY-MM-DD}
type: investigation | plan | impl-plan | review | test-plan | test-report
```

Transition criteria:
- `draft` -> `reviewed`: Anima has read and confirmed the content
- `reviewed` -> `approved`: Anima has approved quality and it may proceed to next phase
- Approval may be given by a superior role — follow organizational rules

## Plan Document Storage

**All plan documents MUST be saved in `state/plans/`.** Saving to `/tmp/` is prohibited
(lost on OS restart).

Naming convention: `{YYYY-MM-DD}_{task-summary}.{type}.md`

| type | Purpose | Example |
|------|---------|---------|
| `investigation` | Investigation report | `2026-03-27_login-bug.investigation.md` |
| `plan` | Implementation plan | `2026-03-27_fix-email-validation.plan.md` |
| `impl-plan` | Detailed implementation plan | `2026-03-27_fix-email-validation.impl-plan.md` |
| `review` | Review report | `2026-03-27_review-PR-42.review.md` |
| `test-plan` | Test plan | `2026-03-27_e2e-login-flow.test-plan.md` |
| `test-report` | Test result report | `2026-03-27_e2e-login-flow.test-report.md` |

Why `state/plans/`:
- **Persistence**: Survives OS restarts unlike `/tmp/`
- **Traceability**: Can verify what was instructed after the fact
- **Supervisor access**: Supervisors can review via `read_memory_file`

```
read_memory_file(path="../{subordinate_name}/state/plans/2026-03-27_fix-email-validation.plan.md")
```

## Machine Execution Commands

```bash
animaworks-tool machine run \
  "$(cat $ANIMAWORKS_ANIMA_DIR/state/plans/{plan_filename})" \
  -d /path/to/worktree
```

Specify engine explicitly:

```bash
animaworks-tool machine run -e cursor-agent \
  "$(cat $ANIMAWORKS_ANIMA_DIR/state/plans/{plan_filename})" \
  -d /path/to/worktree
```

Background execution (for heavy tasks):

```bash
animaworks-tool machine run --background \
  "$(cat $ANIMAWORKS_ANIMA_DIR/state/plans/{plan_filename})" \
  -d /path/to/worktree
```

## Rate Limits

- Session (chat): 5 runs / session
- Heartbeat: 2 runs / heartbeat
- `--background` results are available in `state/background_notifications/`

## Prohibited Actions

- Running `machine run` without a plan document is PROHIBITED (inline short instruction strings only are not allowed)
- Running with a plan that lacks both "Goal" and "Completion Criteria" is PROHIBITED
- Passing machine output to the next phase without verification is PROHIBITED (NEVER)
- Committing or pushing machine output without verification is PROHIBITED

## Machine Constraints

- Machine cannot access AnimaWorks infrastructure (memory, messaging, organization info)
- GitHub API operations (diff retrieval, comment posting, etc.) must be done by Anima and included in the plan
- Use `--background` for long-running tasks

## Role-Specific Workflow Guides

Refer to the workflow guide corresponding to your role:

| Role | Guide | Overview |
|------|-------|----------|
| PdM | `operations/machine/workflow-pdm.md` | Investigation -> Plan creation |
| Engineer | `operations/machine/workflow-engineer.md` | Concretization -> Implementation |
| Reviewer | `operations/machine/workflow-reviewer.md` | Code review -> Meta-review |
| Tester | `operations/machine/workflow-tester.md` | Test design -> Execution -> Result verification |

Role assignments can be found in `injection.md` or `specialty_prompt.md`.
If you hold multiple roles, refer to all applicable guides.

## When Using Team Design Templates

If `injection.md` references `team-design/development/{role}/machine.md`, **that path takes precedence over this file**. Each role’s `machine.md` is self-contained, including base rules and how to write prompts.

This file is the shared guide for standalone Anima that do not use the team-design templates.
