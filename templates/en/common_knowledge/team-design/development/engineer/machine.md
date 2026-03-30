# Engineer — Machine Usage Patterns

## Base rules

1. **Write the plan first** — No inline-only runs. Pass a plan file.
2. **Output is draft** — Verify machine output and set `status: approved` before the next phase.
3. **Storage**: `state/plans/{YYYY-MM-DD}_{summary}.{type}.md` (no `/tmp/`)
4. **Rate limits**: chat 5/session, heartbeat 2
5. **Machine has no infra access** — Include memory, messaging, org context in the plan.

---

## Overview

Engineer **receives plan.md, delegates concretization and implementation to machine, and owns two verification checkpoints**.

- Detailed implementation plan (impl.plan.md) → machine; Engineer verifies
- Code implementation → machine; Engineer verifies
- Verification twice: when approving impl.plan.md and when reviewing implementation output

---

## Phase 1: Concretization (plan.md → impl.plan.md)

### Step 1: Read plan.md

From PdM, confirm:

- [ ] `status: approved`
- [ ] Goal and completion criteria clear
- [ ] No technical contradictions in approach
- [ ] Open questions resolved with PdM before proceeding

### Step 2: Delegate concretization to machine

Ask machine to produce impl.plan.md from plan.md.

```bash
animaworks-tool machine run \
  "$(cat $ANIMAWORKS_ANIMA_DIR/state/plans/{plan.md})" \
  -d /path/to/worktree
```

Save as `state/plans/{date}_{summary}.impl-plan.md` (`status: draft`).

**Granularity**: One machine call = one focus. Large plan.md may be split by module across multiple runs.

### Step 3: Verify impl.plan.md (checkpoint 1)

- [ ] Aligns with plan.md goals and approach
- [ ] Target file paths exist and are correct
- [ ] Changes technically sound
- [ ] Dependencies and order correct
- [ ] Consistent with existing code
- [ ] Rollback possible

Fix as needed and set `status: approved`.

## Phase 2: Implementation

### Step 4: Run implementation on machine

Pass approved impl.plan.md to machine.

```bash
animaworks-tool machine run \
  "$(cat $ANIMAWORKS_ANIMA_DIR/state/plans/{impl-plan.md})" \
  -d /path/to/worktree
```

**Notes:**
- impl.plan.md should list files to change; stating “do not change files not listed” reduces scope creep
- Prefer excerpts over whole files for context (too much context hurts quality)

### Step 5: Verify implementation output (checkpoint 2)

- [ ] `git diff` matches impl.plan.md
- [ ] No unintended files changed
- [ ] Tests run and pass
- [ ] Respects plan.md constraints and completion criteria

### Step 6: Next steps

- **Pass**: Request review/test from Reviewer/Tester
- **Fail**: Revise impl.plan.md and re-run machine, or fix yourself

---

## impl.plan.md template

```markdown
# Detailed implementation plan: {task name}

status: draft
author: {anima_name}
date: {YYYY-MM-DD}
type: impl-plan
source: state/plans/{original plan.md}

## Goal (from plan.md)

{Copy from plan.md}

## Target files

{Create/Modify/Delete — do not change files not listed}

| File | Action | Summary |
|------|--------|---------|
| {path1} | Create | {why} |
| {path2} | Modify | {what} |
| {path3} | Delete | {why} |

## Implementation steps

{Phases; each step runnable by machine at a time}

### Phase 1: {prep}

- [ ] **Step 1.1**: {change}
  - File: `{path}`
  - Detail: {context}

### Phase 2: {core}

- [ ] **Step 2.1**: {change}
  - File: `{path}`
  - Detail: {context}

### Phase 3: {tests / cleanup}

- [ ] **Step 3.1**: {tests}
  - File: `{path}`

## Dependencies

| From | To | Reason |
|------|-----|--------|
| Step 2.1 | Step 1.1 | {reason} |

## Constraints (from plan.md)

{Copy}

## Completion criteria (from plan.md)

{Copy}

## Rollback plan

1. {restore}
2. {verify after restore}
```

---

## Constraints

- NEVER start from `plan.md` without `status: approved`
- NEVER run implementation on machine without approved impl.plan.md
- NEVER commit/push without verifying implementation output
- MUST fix issues found in verification before moving on
