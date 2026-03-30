# Development Full Team — Overview

## Four-role layout

| Role | Responsibility | Suggested `--role` | Example `speciality` | Details |
|------|------------------|---------------------|----------------------|---------|
| **PdM** | Investigation, planning, decisions | `manager` | `pdm` | `development/pdm/` |
| **Engineer** | Implementation and implementation verification | `engineer` | `backend`, `fullstack` | `development/engineer/` |
| **Reviewer** | Code review (static verification) | `engineer` | `code-review` | `development/reviewer/` |
| **Tester** | Test design and execution (dynamic verification) | `engineer` | `testing`, `qa` | `development/tester/` |

Collapsing into one Anima risks context bloat, self-review blind spots, and serial bottlenecks.

Each role directory has `injection.template.md` (injection skeleton), `machine.md` (machine usage patterns), and `checklist.md` (quality checklist).

> Core principles: `team-design/guide.md`

## Handoff chain

```
PdM → investigation.md/plan.md (approved) → delegate_task
  → Engineer → impl.plan.md → implementation → implementation verification
    → Reviewer (static verification) ─┐
    → Tester   (dynamic verification) ─┤ ← can run in parallel
      └─ fixes needed → back to Engineer
      └─ both APPROVE → PdM → call_human → human merge decision
```

### Handoff documents

| From → To | Document | Condition |
|-----------|----------|-----------|
| PdM → Engineer | `plan.md` | `status: approved` |
| Engineer → Reviewer/Tester | Implementation diff + `plan.md` | After implementation verification |
| Reviewer → Engineer | `review.md` | `status: approved` |
| Tester → Engineer/PdM | `test-report.md` | `status: approved` |

### Operating rules

- **Worktree**: Engineer creates after receiving plan.md (`{task-id}/{summary}`). Machine runs with `-d /path/to/worktree`. Merge and cleanup after completion are Engineer’s job.
- **Fix cycle**: Critical → full re-review / Warning → diff-only check / Still unresolved after 3 rounds → escalate to PdM.
- **Machine failure**: Record in `current_state.md` → reassess on next heartbeat.

## Scaling

| Scale | Composition | Notes |
|-------|-------------|-------|
| Small | PdM + Engineer (Reviewer combined) | Self-review risk accepted |
| Medium | Four roles as in this template | Standard |
| Large | PdM 1 + multiple Engineers + 1–2 Reviewer/Testers each | PdM delegates by module with `delegate_task` |
