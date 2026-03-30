# Machine Workflow — Tester

## Role Definition

Tester is responsible for **designing test strategy, delegating test execution to machine,
and verifying/judging test results**.

- Test strategy and perspective design -> Tester's own judgment
- Test case concretization -> delegate to machine, Tester verifies
- Test execution -> delegate to machine
- Test result interpretation and judgment -> Tester's own judgment

> Prerequisite: Understand the meta-pattern and common principles in `operations/machine/tool-usage.md`.

## Test Types

Test types depend on the project and target system.
Tester should determine the appropriate test strategy based on plan.md and system characteristics.

Representative test types:

| Type | Overview | Examples |
|------|----------|----------|
| **E2E Test** | Verify actual user flows via browser automation, etc. | Login flow, purchase flow |
| **Component Test** | Verify individual component behavior within execution environment | API endpoints, batch processing |
| **Integration Test** | Verify interaction between multiple components | DB -> API -> Frontend |
| **Regression Test** | Verify existing features are not broken by changes | Run existing test suite |

## Workflow

### Step 1: Create a test plan (Tester writes this)

Read plan.md and implementation details, then plan test strategy, perspectives, and scenario overview.

```bash
write_memory_file(path="state/plans/{date}_{summary}.test-plan.md", content="...")
```

### Step 2: Delegate test case concretization to machine

```bash
animaworks-tool machine run \
  "$(cat $ANIMAWORKS_ANIMA_DIR/state/plans/{test-plan.md})" \
  -d /path/to/worktree
```

Save result as `state/plans/{date}_{summary}.test-cases.md` (`status: draft`).

### Step 3: Verify test cases

Tester reads test-cases.md and confirms:

- [ ] Are plan.md completion criteria covered by test cases?
- [ ] Are both happy path and error cases included?
- [ ] Are edge cases considered (boundary values, empty values, large data, etc.)?
- [ ] Are test cases practically executable (realistic environment and preconditions)?
- [ ] Are high-priority scenarios covered?

Fix as needed, then change to `status: approved`.

### Step 4: Delegate test execution to machine

Based on approved test-cases.md, delegate test execution to machine.

```bash
animaworks-tool machine run \
  "Execute the following test cases and report results.
  For each test case, record Pass/Fail with evidence (output, screenshot info, etc.).

  $(cat $ANIMAWORKS_ANIMA_DIR/state/plans/{test-cases.md})" \
  -d /path/to/worktree
```

Save result as `state/plans/{date}_{summary}.test-report.md` (`status: draft`).

### Step 5: Verify test results

Tester reads test.report.md and judges:

- [ ] Are results recorded for all test cases?
- [ ] Are failure causes identified (implementation issue vs. environment issue vs. test case issue)?
- [ ] Are there any false positives (reported as Fail but actually fine)?
- [ ] Any possibility of false negatives (reported as Pass but actually broken)?
- [ ] Do test results reveal areas needing additional testing?

Tester adds their own observations, then changes to `status: approved`.

### Step 6: Feedback

- All tests Pass -> Report pass to Engineer / PdM
- Failures found -> Report specific defect information to Engineer (reproduction steps, expected vs. actual)
- Additional testing needed -> Return to Step 1

## Test Plan Template (test.plan.md)

```markdown
# Test Plan: {Test Target Summary}

status: draft
author: {anima_name}
date: {YYYY-MM-DD}
type: test-plan
source: state/plans/{original plan.md}

## Test Objective

{What this test verifies}

## Test Strategy

{Test types to use and rationale}

- E2E Test: {target flows}
- Component Test: {target components}

## Test Perspectives

{What to focus testing on}

- [ ] Functional requirements compliance
- [ ] Error handling
- [ ] Performance (if applicable)
- [ ] Security (if applicable)

## Test Targets

{Files, endpoints, screens to test}

- {target1}
- {target2}

## Test Environment

{Environment information needed for test execution}

- Execution environment: {browser / container / local, etc.}
- Preconditions: {test data, service state, etc.}
- Execution command: {how to run tests}

## Test Scenario Overview

{High-level scenarios -- details to be concretized by machine}

### Happy Path

1. {scenario1}
2. {scenario2}

### Error Cases

1. {scenario1}
2. {scenario2}

## Pass Criteria

{What constitutes "test passed"}
```

## Test Cases Template (test-cases.md)

```markdown
# Test Cases: {Test Target Summary}

status: draft
author: {anima_name}
date: {YYYY-MM-DD}
type: test-plan

## Test Case List

### Happy Path

| # | Scenario | Preconditions | Steps | Expected Result | Priority |
|---|----------|--------------|-------|-----------------|----------|
| 1 | {name} | {preconditions} | {steps} | {expected} | High |
| 2 | {name} | {preconditions} | {steps} | {expected} | Medium |

### Error Cases

| # | Scenario | Preconditions | Steps | Expected Result | Priority |
|---|----------|--------------|-------|-----------------|----------|
| 1 | {name} | {preconditions} | {steps} | {expected} | High |

### Edge Cases

| # | Scenario | Preconditions | Steps | Expected Result | Priority |
|---|----------|--------------|-------|-----------------|----------|
| 1 | {name} | {preconditions} | {steps} | {expected} | Low |
```

## Test Report Template (test.report.md)

```markdown
# Test Result Report: {Test Target Summary}

status: draft
author: {anima_name}
date: {YYYY-MM-DD}
type: test-report

## Overall Results

- Execution date: {YYYY-MM-DD HH:MM}
- Total tests: {N}
- Pass: {N} / Fail: {N} / Skip: {N}
- Verdict: {Pass / Fail}

## Detailed Results

| # | Scenario | Result | Evidence | Notes |
|---|----------|--------|----------|-------|
| 1 | {name} | Pass | {output/logs} | |
| 2 | {name} | Fail | {error details} | {root cause analysis} |

## Defect List (for Failures)

| # | Scenario | Reproduction Steps | Expected | Actual | Severity |
|---|----------|-------------------|----------|--------|----------|
| 1 | {name} | {steps} | {expected} | {actual} | Critical |

## Tester Observations

{Tester's own analysis, additional observations, recommendations}
```

## Constraints

- Test plan (test strategy, perspectives) MUST be written by Tester
- test-cases.md generated by machine remains a draft until Tester verifies and approves
- NEVER judge test results as "passed" without verification
- If failure cause is unclear, investigate further before making a judgment (MUST)
- NEVER send a test.report.md without `status: approved` as feedback
