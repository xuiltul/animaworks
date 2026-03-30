# PdM — Quality checklist

Checklist for PdM to ensure artifact quality, with or without machine.

---

## Investigation report (investigation.md)

### Completeness

- [ ] Answers the investigation goal comprehensively
- [ ] Facts vs assumptions are clearly separated
- [ ] Impact assessment included
- [ ] Quantitative data where relevant (files, lines, affected users, etc.)

### Reliability

- [ ] Sources (code, logs, docs) cited
- [ ] “Unknown” / “needs more research” called out
- [ ] No reliance on obsolete specs

---

## Plan (plan.md)

### Goal clarity

- [ ] Goal in 1–3 sentences
- [ ] Completion objectively verifiable
- [ ] Scope (in / out) clear

### Implementation approach

- [ ] Grounded in investigation
- [ ] Technically feasible (or Engineer consulted)
- [ ] Consistent with existing code/architecture
- [ ] Phasable if needed

### Constraints

- [ ] Coding standards referenced
- [ ] Backward compatibility considered if needed
- [ ] Performance requirements stated if needed

### Completion criteria

- [ ] Each criterion objectively verifiable
- [ ] Testing approach included
- [ ] Reviewer/Tester can verify

### Risk management

- [ ] Major risks listed with mitigations
- [ ] Rollback considered

---

## Delegation quality

### Before `delegate_task`

- [ ] `status: approved` on plan.md
- [ ] Completion criteria actionable for Engineer
- [ ] No open questions left unaddressed
- [ ] Deadline set

### Feedback handling

- [ ] All Reviewer/Tester feedback reviewed
- [ ] Response plan for Critical items
- [ ] Re-verification requested after fixes
