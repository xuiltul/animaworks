# Tester — Quality checklist

Checklist for test quality, with or without machine.

---

## Test plan (test.plan.md)

### Strategy

- [ ] Purpose aligned with plan.md goals and completion criteria
- [ ] Rationale for chosen test types stated
- [ ] Scope clear (what is / isn’t tested)

### Coverage

- [ ] Functional requirements
- [ ] Error handling
- [ ] Regression of existing behavior
- [ ] Non-functional (performance, security) if needed

### Feasibility

- [ ] Environment preconditions realistic
- [ ] Test data preparation described
- [ ] Run commands accurate

---

## Test cases (test-cases.md)

### Categories

- [ ] Happy path: main flows covered
- [ ] Unhappy path: invalid input, permissions, resource limits
- [ ] Edge: boundaries, empty, max, special chars

### Per case

- [ ] Preconditions clear (test independence)
- [ ] Steps reproducible
- [ ] Expected result concrete and checkable
- [ ] Priorities sensible

### plan.md mapping

- [ ] Every completion criterion mapped to cases
- [ ] Constraints verified by tests

---

## Test report (test-report.md)

### Completeness

- [ ] Every case has a result
- [ ] Evidence for each (logs, output, screenshot info)
- [ ] Skip reasons recorded

### Failure analysis

- [ ] Cause class correct (implementation / env / test)
- [ ] False positives identified
- [ ] False negatives considered
- [ ] Repro steps concrete

### Bug report quality

- [ ] Third party can reproduce
- [ ] Expected vs actual clear
- [ ] Severity reasonable

---

## Feedback quality

### To Engineer

- [ ] Concrete repro for failures
- [ ] Expected vs actual stated
- [ ] How to retest after fix

### To PdM

- [ ] Summary (counts, overall pass/fail)
- [ ] Impact if major issues
- [ ] Whether more testing is needed
