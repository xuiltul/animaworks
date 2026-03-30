# Machine Workflow — Engineer

## Role Definition

Engineer **receives plan.md, delegates concretization and implementation to machine,
and serves as the quality gate at 2 verification checkpoints**.

- Implementation detail plan (impl.plan.md) concretization -> delegate to machine, Engineer verifies
- Code implementation -> delegate to machine, Engineer verifies
- Verification happens twice: at impl.plan.md approval and at implementation output review

> Prerequisite: Understand the meta-pattern and common principles in `operations/machine/tool-usage.md`.

## Phase 1: Concretization (plan.md -> impl.plan.md)

### Step 1: Read plan.md

Read the plan.md received from PdM and confirm:

- [ ] `status: approved` is present
- [ ] Goal and completion criteria are clear
- [ ] Implementation approach has no technical contradictions
- [ ] Clarify any questions with PdM before proceeding

### Step 2: Delegate concretization to machine

Use plan.md as input to request machine generate an impl.plan.md.

```bash
animaworks-tool machine run \
  "Based on the following plan.md, create an impl.plan.md.
  Detail specific changes per file, dependencies, and implementation order.

  $(cat $ANIMAWORKS_ANIMA_DIR/state/plans/{plan.md})" \
  -d /path/to/worktree
```

Save result as `state/plans/{date}_{summary}.impl-plan.md` (`status: draft`).

### Step 3: Verify impl.plan.md (Checkpoint 1)

Engineer reads impl.plan.md and confirms:

- [ ] Consistent with plan.md goals and approach
- [ ] File paths are accurate (files actually exist)
- [ ] Changes are technically sound
- [ ] Dependencies and implementation order are correct
- [ ] Compatible with existing codebase
- [ ] Design allows rollback

Fix issues directly, then change to `status: approved`.

## Phase 2: Implementation

### Step 4: Delegate implementation to machine

Pass the `status: approved` impl.plan.md to machine for implementation.

```bash
animaworks-tool machine run \
  "$(cat $ANIMAWORKS_ANIMA_DIR/state/plans/{impl-plan.md})" \
  -d /path/to/worktree
```

### Step 5: Verify implementation output (Checkpoint 2)

Verify machine output against impl.plan.md:

- [ ] `git diff` shows changes consistent with impl.plan.md
- [ ] No unintended files were modified
- [ ] Tests are runnable and all pass
- [ ] No violations of plan.md constraints
- [ ] plan.md completion criteria are met

### Step 6: Process results

- **Pass**: Request review/testing from Reviewer/Tester
- **Issues found**: Revise impl.plan.md and re-delegate to machine, or fix directly

## Implementation Detail Plan Template (impl.plan.md)

```markdown
# Implementation Detail Plan: {Task Name}

status: draft
author: {anima_name}
date: {YYYY-MM-DD}
type: impl-plan
source: state/plans/{original plan.md}

## Goal (inherited from plan.md)

{Copy goal from plan.md}

## Target Files

| File | Action | Change Summary |
|------|--------|---------------|
| {path/to/file1} | Create | {purpose} |
| {path/to/file2} | Modify | {change summary} |
| {path/to/file3} | Delete | {reason} |

## Implementation Steps

### Phase 1: {Preparation}

- [ ] **Step 1.1**: {specific change}
  - File: `{path}`
  - Details: {additional context}

- [ ] **Step 1.2**: {specific change}
  - File: `{path}`
  - Details: {additional context}

### Phase 2: {Core Implementation}

- [ ] **Step 2.1**: {specific change}
  - File: `{path}`
  - Details: {additional context}

### Phase 3: {Testing & Cleanup}

- [ ] **Step 3.1**: {test creation}
  - File: `{path}`

## Dependencies

| From | To | Reason |
|------|-----|--------|
| Step 2.1 | Step 1.1 | {reason} |

## Constraints (inherited from plan.md)

{Copy constraints from plan.md}

## Completion Criteria (inherited from plan.md)

{Copy completion criteria from plan.md}

## Rollback Plan

1. {recovery steps if issues arise}
2. {verification steps after recovery}
```

## Constraints

- NEVER start work based on a plan.md without `status: approved`
- NEVER delegate implementation to machine without `status: approved` on impl.plan.md
- NEVER commit or push machine implementation output without verification
- Issues found during verification MUST be fixed before proceeding to next phase
