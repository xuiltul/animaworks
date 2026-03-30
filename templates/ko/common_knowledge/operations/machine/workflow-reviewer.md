# Machine Workflow — Reviewer

## Role Definition

Reviewer **delegates the entire review to machine, then verifies the correctness
of the review results (meta-review)**.

- Designing review perspectives -> Reviewer's own judgment
- Conducting code review -> delegate to machine
- Verifying review result correctness -> Reviewer's own judgment

Machine excels at fast static analysis, pattern detection, and requirements compliance checks,
but determining design decision validity and context-aware severity assessment is the Reviewer's
responsibility.

> Prerequisite: Understand the meta-pattern and common principles in `operations/machine/tool-usage.md`.

## Workflow

### Step 1: Create a review plan (Reviewer writes this)

Create a plan document defining review perspectives, targets, and criteria.

```bash
write_memory_file(path="state/plans/{date}_{summary}.review.md", content="...")
```

Before creating, gather the following on the Anima side:
- `git diff` or PR diff content
- Related Issue / plan.md requirements
- Existing code structure of the target (via `search_memory` or direct reading as needed)

### Step 2: Delegate review to machine

```bash
animaworks-tool machine run \
  "$(cat $ANIMAWORKS_ANIMA_DIR/state/plans/{review_plan})" \
  -d /path/to/worktree
```

Save result to `state/plans/{date}_{summary}.review.md` by appending or overwriting (`status: draft`).

### Step 3: Verify review results (Meta-review)

Reviewer reads review.md and confirms:

- [ ] Are findings factually accurate (no false positives)?
- [ ] Are any important issues missed?
- [ ] Are severity ratings (Critical/Warning/Info) appropriate?
- [ ] Consistency with plan.md completion criteria and constraints
- [ ] Consistency with coding standards
- [ ] Any additional findings from Reviewer's own perspective?

Reviewer fixes/supplements as needed, then changes to `status: approved`.

### Step 4: Feedback

Send the approved review.md to Engineer.
If changes are needed, clearly specify action items.

## Review Plan Template (review.plan.md)

```markdown
# Review Plan: {Review Target Summary}

status: draft
author: {anima_name}
date: {YYYY-MM-DD}
type: review

## Review Perspectives

{Specify perspectives to cover}

- [ ] Requirements compliance: Does it meet plan.md completion criteria?
- [ ] Code quality: Readability, maintainability, naming conventions
- [ ] Safety: Any security concerns?
- [ ] Performance: N+1 queries, unnecessary loops, etc.
- [ ] Testing: Test coverage, edge cases

## Target

{Diff information for the review target}

- Branch / PR: {info}
- Changed files:
  - {file1}
  - {file2}

## Issue / plan.md Requirements

{Related Issue URL, summary of plan.md completion criteria}

## Diff Information

{git diff output or key change excerpts}

## Verification Commands

{Commands to run during review}

- `git diff {base}..{head}`
- `{test execution command}`
- `{lint execution command}`

## Output Format

Output review results in the following format:

- **Critical**: Issues that must be fixed
- **Warning**: Issues that should be fixed
- **Info**: Informational notes and improvement suggestions
```

## Review Result Template (review.md)

```markdown
# Review Result: {Review Target Summary}

status: draft
author: {anima_name}
date: {YYYY-MM-DD}
type: review

## Overall Verdict

{APPROVE / REQUEST_CHANGES / COMMENT}

## Findings

### Critical (Must Fix)

| # | File | Line | Finding | Recommended Fix |
|---|------|------|---------|----------------|
| 1 | {path} | {line} | {content} | {fix suggestion} |

### Warning (Should Fix)

| # | File | Line | Finding | Recommended Fix |
|---|------|------|---------|----------------|
| 1 | {path} | {line} | {content} | {fix suggestion} |

### Info (Informational)

| # | File | Line | Finding |
|---|------|------|---------|
| 1 | {path} | {line} | {content} |

## Requirements Compliance Check

| Completion Criterion | Met | Notes |
|---------------------|-----|-------|
| {criterion1} | Yes / No | {notes} |
| {criterion2} | Yes / No | {notes} |

## Additional Comments

{Reviewer's own observations and notes}
```

## Constraints

- Review plan (what perspectives to review from) MUST be written by Reviewer
- NEVER pass machine review results directly to Engineer — Reviewer must verify first
- NEVER send a review.md without `status: approved` as feedback to Engineer
- Diff information and Issue requirements must be gathered by Anima and included in the plan (machine cannot access GitHub API)
