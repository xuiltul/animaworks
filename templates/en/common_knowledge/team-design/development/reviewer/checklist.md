# Reviewer — Quality checklist

Checklist for review quality, with or without machine.

---

## Coverage

### Requirements

- [ ] All plan.md completion criteria met
- [ ] No violations of plan.md constraints
- [ ] No out-of-scope changes

### Code quality

- [ ] Naming matches project conventions
- [ ] Single responsibility per function (no huge functions)
- [ ] No undue duplication (DRY)
- [ ] No debug leftovers, commented dead code, unused imports

### Design

- [ ] Fits existing architecture/patterns
- [ ] Appropriate abstraction level
- [ ] Dependency direction sound (no bad cycles)
- [ ] Public API changes consider backward compatibility

### Security

- [ ] External input validated
- [ ] No SQLi/XSS-class issues
- [ ] No hardcoded secrets
- [ ] File ops safe from path traversal

### Performance

- [ ] No N+1 or obvious hot loops
- [ ] Large-data/memory considered
- [ ] Async used where appropriate

### Tests

- [ ] New code has tests where needed
- [ ] Happy and unhappy paths
- [ ] Tests not flaky (timing, external deps)

---

## Meta-review

### Machine output

- [ ] False positives removed
- [ ] False negatives supplemented
- [ ] Severity (Critical/Warning/Info) reasonable
- [ ] Findings evidence-based

### Feedback quality

- [ ] Critical items have concrete fixes
- [ ] Rationale stated (“why it matters”)
- [ ] Constructive tone
- [ ] Reproducible references (file, line)
