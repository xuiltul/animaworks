# Tester — Machine Usage Patterns

## Base rules

1. **Write the plan first** — No inline-only runs.
2. **Output is draft** — Verify and set `status: approved` before handoff.
3. **Storage**: `state/plans/{YYYY-MM-DD}_{summary}.{type}.md` (no `/tmp/`)
4. **Rate limits**: chat 5/session, heartbeat 2
5. **Machine has no infra access** — Put context in the plan.

---

## Overview

Tester **owns test strategy, machine-based execution, and judgment of results**.

- Strategy and focus → Tester
- Detailed test cases → machine; Tester verifies
- Execution → machine
- Interpreting results → Tester

---

## Test types

Depends on project and system. Choose strategy from plan.md and system characteristics.

| Type | Summary | Examples |
|------|---------|----------|
| **E2E** | User flows in browser/automation | Login, checkout |
| **Component** | Single component in environment | API endpoint, batch |
| **Integration** | Multiple components together | DB → API → UI |
| **Regression** | Existing behavior not broken | Full existing suite |

---

## Workflow

### Step 1: Write test plan (Tester)

From plan.md and implementation, outline strategy, focus, scenarios.

```bash
write_memory_file(path="state/plans/{date}_{summary}.test-plan.md", content="...")
```

### Step 2: Delegate test-case detail to machine

```bash
animaworks-tool machine run \
  "$(cat $ANIMAWORKS_ANIMA_DIR/state/plans/{test-plan.md})" \
  -d /path/to/worktree
```

Save as `state/plans/{date}_{summary}.test-cases.md` (`status: draft`).

### Step 3: Verify test cases

- [ ] plan.md completion covered
- [ ] Happy and unhappy paths
- [ ] Edge cases (boundary, empty, large data)
- [ ] Executable given environment assumptions
- [ ] High-priority scenarios covered

Fix and set `status: approved`.

### Step 4: Run tests on machine

```bash
animaworks-tool machine run \
  "Run the following test cases and report results.
  For each case record Pass/Fail and evidence (output, screenshot info, etc.).

  $(cat $ANIMAWORKS_ANIMA_DIR/state/plans/{test-cases.md})" \
  -d /path/to/worktree
```

**Notes:**
- State environment preconditions (services, data) in the plan — machine cannot guess.
- State that code outside test targets must not be changed.

Save as `state/plans/{date}_{summary}.test-report.md` (`status: draft`).

### Step 5: Verify test report

- [ ] Every case has a result
- [ ] Fail causes identified (implementation vs env vs test design)
- [ ] False positives ruled out
- [ ] False negatives considered
- [ ] Gaps needing more tests

Add notes and set `status: approved`.

### Step 6: Feedback

- All pass → report pass to Engineer/PdM
- Failures → report repro steps, expected vs actual to Engineer
- Need more tests → return to Step 1

---

## Test plan template (test.plan.md)

```markdown
# Test plan: {target summary}

status: draft
author: {anima_name}
date: {YYYY-MM-DD}
type: test-plan
source: state/plans/{original plan.md}

## Purpose

{What this test validates}

## Strategy

{Types and why}

- E2E: {flows}
- Component: {targets}

## Focus

- [ ] Functional requirements
- [ ] Error handling
- [ ] Performance (if needed)
- [ ] Security (if needed)

## Targets

- {t1}
- {t2}

## Out of scope

{What not to test — avoids machine generating out-of-scope tests}

## Environment

{Concrete — machine cannot infer}

- Runtime: {browser / container / local}
- Preconditions: {data, services}
- Command: {how to run tests}

## Scenario outline

### Happy path

1. {s1}
2. {s2}

### Unhappy path

1. {s1}
2. {s2}

## Pass criteria

{Definition of “test pass”}
```

## Test cases template (test-cases.md)

```markdown
# Test cases: {target summary}

status: draft
author: {anima_name}
date: {YYYY-MM-DD}
type: test-cases

## Cases

### Happy path

| # | Scenario | Preconditions | Steps | Expected | Priority |
|---|----------|---------------|-------|----------|----------|
| 1 | {name} | {pre} | {steps} | {exp} | High |

### Unhappy path

| # | Scenario | Preconditions | Steps | Expected | Priority |
|---|----------|---------------|-------|----------|----------|
| 1 | {name} | {pre} | {steps} | {exp} | High |

### Edge cases

| # | Scenario | Preconditions | Steps | Expected | Priority |
|---|----------|---------------|-------|----------|----------|
| 1 | {name} | {pre} | {steps} | {exp} | Low |
```

## Test report template (test-report.md)

```markdown
# Test report: {target summary}

status: draft
author: {anima_name}
date: {YYYY-MM-DD}
type: test-report

## Summary

- When: {YYYY-MM-DD HH:MM}
- Total: {N}
- Pass: {N} / Fail: {N} / Skip: {N}
- Overall: {pass / fail}

## Details

| # | Scenario | Result | Evidence | Notes |
|---|----------|--------|----------|-------|
| 1 | {name} | Pass | {log} | |

## Failures

| # | Scenario | Repro | Expected | Actual | Severity |
|---|----------|-------|------------|--------|----------|
| 1 | {name} | {steps} | {exp} | {act} | Critical |

## Tester notes

{Analysis and recommendations}
```

---

## Constraints

- Test strategy and focus MUST be written by Tester.
- test-cases.md may be machine-generated but stays draft until Tester approves.
- NEVER declare pass without verifying the test report.
- MUST investigate unclear failures before final judgment.
- NEVER send test-report.md without `status: approved`.
