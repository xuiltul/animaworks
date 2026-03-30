# Engineer — Quality checklist

Checklist for Engineer artifact quality, with or without machine.

---

## impl.plan.md

### Consistency with plan.md

- [ ] Goals from plan.md reflected accurately
- [ ] Every completion criterion maps to steps
- [ ] No steps violate constraints
- [ ] No scope creep beyond plan.md

### Technical soundness

- [ ] Paths correct and verified to exist
- [ ] Changes fit Stack and patterns
- [ ] Dependencies and order correct
- [ ] Consistent with existing structure

### Executability

- [ ] Each step concrete enough for machine
- [ ] Rollback plan present
- [ ] Test create/update steps included

---

## Implementation output

### Code quality

- [ ] Meets project coding standards
- [ ] Type hints appropriate
- [ ] Error handling adequate
- [ ] No debug noise or dead code

### Tests

- [ ] New code has unit tests where appropriate
- [ ] All existing tests pass
- [ ] Edge cases (empty, boundary, large data) covered

### Design fit

- [ ] `git diff` matches impl.plan.md
- [ ] No unintended file changes
- [ ] plan.md completion criteria satisfied
- [ ] plan.md constraints respected

### Safety

- [ ] Breaking changes (API, schema) approved in plan.md if applicable
- [ ] No security issues (secrets in code, missing validation)
- [ ] No obvious performance regressions (N+1, unnecessary loops)

---

## Handoff to Reviewer/Tester

- [ ] Checkpoint 2 verification done
- [ ] Summary of changes clear for reviewers
- [ ] Link to plan.md included
- [ ] Test commands documented
