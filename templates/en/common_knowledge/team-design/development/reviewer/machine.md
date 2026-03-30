# Reviewer — Machine Usage Patterns

## Base rules

1. **Write the plan first** — No inline-only runs.
2. **Output is draft** — Verify and set `status: approved` before handoff.
3. **Storage**: `state/plans/{YYYY-MM-DD}_{summary}.{type}.md` (no `/tmp/`)
4. **Rate limits**: chat 5/session, heartbeat 2
5. **Machine has no infra access** — Put context in the plan.

---

## Overview

Reviewer **delegates review to machine and validates the review (meta-review)**.

- Review focus design → Reviewer
- Performing code review → machine
- Validating review quality → Reviewer

Machine is strong at static analysis and pattern checks; **design judgment and context-aware severity** stay with Reviewer.

---

## Workflow

### Step 1: Write a review plan (Reviewer)

Clarify focus, targets, and criteria.

```bash
write_memory_file(path="state/plans/{date}_{summary}.review.md", content="...")
```

Prepare on Anima side before running:
- `git diff` or PR diff
- Related Issue / plan.md requirements
- Existing structure (`search_memory` or read files)

### Step 2: Run review on machine

```bash
animaworks-tool machine run \
  "$(cat $ANIMAWORKS_ANIMA_DIR/state/plans/{review_plan})" \
  -d /path/to/worktree
```

Append or save to `state/plans/{date}_{summary}.review.md` (`status: draft`).

### Step 3: Meta-review

- [ ] Findings factual (no false positives)
- [ ] No important misses
- [ ] Critical/Warning/Info severity appropriate
- [ ] Matches plan.md completion/constraints
- [ ] Matches coding standards
- [ ] Any extra findings from Reviewer’s own lens

Edit and set `status: approved`.

### Step 4: Feedback

Send approved review.md to Engineer with concrete action items when changes are required.

---

## Review plan template (review.plan.md)

```markdown
# Review plan: {target summary}

status: draft
author: {anima_name}
date: {YYYY-MM-DD}
type: review

## Focus

- [ ] Requirements: plan.md completion criteria
- [ ] Code quality: readability, maintainability, naming
- [ ] Safety: security issues
- [ ] Performance: N+1, unnecessary loops
- [ ] Tests: coverage, edge cases

## Target

- Branch / PR: {info}
- Files:
  - {file1}
  - {file2}

## Issue / plan.md

{Issue URL, plan.md summary}

## Diff

{Diff excerpt — not whole files; “diff + ~10 lines context” works well}

## Commands

- `git diff {base}..{head}`
- `{test command}`
- `{lint command}`

## Required output format

**Critical**: must-fix (file, line, suggestion)  
**Warning**: should-fix  
**Info**: notes

Outputs not following this format are invalid.
```

## Review result template (review.md)

```markdown
# Review result: {target summary}

status: draft
author: {anima_name}
date: {YYYY-MM-DD}
type: review

## Verdict

{APPROVE / REQUEST_CHANGES / COMMENT}

## Findings

### Critical

| # | File | Line | Issue | Suggested fix |
|---|------|------|-------|---------------|
| 1 | {path} | {line} | {text} | {fix} |

### Warning

| # | File | Line | Issue | Suggested fix |
|---|------|------|-------|---------------|
| 1 | {path} | {line} | {text} | {fix} |

### Info

| # | File | Line | Issue |
|---|------|------|-------|
| 1 | {path} | {line} | {text} |

## Requirements check

| Completion criterion | Met | Notes |
|---------------------|-----|-------|
| {c1} | Yes / No | |
| {c2} | Yes / No | |

## Additional comments

{Reviewer notes}
```

---

## Constraints

- Review plan (what to check) MUST be written by Reviewer.
- NEVER forward machine review to Engineer without meta-review.
- NEVER send review.md without `status: approved`.
- Diff and Issue context must be gathered by Anima (machine has no GitHub API access).
