# Legal Full Team — Overview

## Three-role layout

| Role | Responsibility | Suggested `--role` | Example `speciality` | Details |
|------|----------------|---------------------|----------------------|---------|
| **Legal Director** | Analysis planning, contract scan, judgment, final sign-off | `manager` | `legal-director` | `legal/director/` |
| **Legal Verifier** | Independent verification, optimism-bias detection, carry-forward verification | `researcher` | `legal-verifier` | `legal/verifier/` |
| **Precedent Researcher** | Statutes, case law, and industry-standard evidence | `general` | `legal-researcher` | `legal/researcher/` |

Putting every step in one Anima invites self-review blind spots (optimism bias), lost findings (silent drop), and context bloat.

Each role directory has `injection.template.md` (injection skeleton), `machine.md` (machine usage patterns for that role only), and `checklist.md` (quality checklist).

> Core principles: `team-design/guide.md`

## Handoff chain

```
Director → analysis-plan.md (approved) + carry-forward tracker reference
  → machine full-contract scan → Director verifies
    → audit-report.md (reviewed)
      → Verifier (independent verification) ─┐
      → Researcher (evidence verification) ───┤ ← can run in parallel
        └─ findings → send back to Director
        └─ both APPROVE → Director → update carry-forward tracker → call_human → human final review
```

### Handoff documents

| From → To | Document | Condition |
|-----------|----------|-----------|
| Director → Verifier/Researcher | `audit-report.md` + `analysis-plan.md` | `status: reviewed` |
| Verifier → Director | `verification-report.md` | `status: approved` |
| Researcher → Director | `precedent-report.md` | `status: approved` |

### Operating rules

- **Fix cycle**: Critical → full re-verification (re-engage Verifier and Researcher) / Warning → diff-only check / Still unresolved after 3 rounds → escalate to humans
- **carry-forward tracker**: Track findings across all versions of the matter. Silent drop (a prior finding vanishing without mention) is forbidden
- **Machine failure**: Record in `current_state.md` → reassess on next heartbeat

## Scaling

| Scale | Composition | Notes |
|-------|-------------|-------|
| Solo | Director covers all roles (quality via checklists) | NDA checks, standard-form contract review |
| Pair | Director + Verifier | SPA amended-version review, medium-risk contracts |
| Full team | Three roles as in this template | SPA first-pass review, M&A DD, high-risk matters |

## Mapping to the development team

| Development role | Legal role | Why it maps |
|------------------|------------|-------------|
| PdM (investigate, plan, decide) | Director (analysis plan, judgment) | Sets what to analyze |
| Engineer (implementation) | Director + machine (contract scan) | Director runs analysis via machine; no separate “implementer” Anima |
| Reviewer (static verification) | Verifier (independent verification) | Core separation of execution vs verification |
| Tester (dynamic verification) | Researcher (evidence verification) | Backs claims like “industry standard” and case law with real data |

## Carry-forward Tracker — finding tracking table

Update this table whenever the contract version changes. It manages every finding from the last review and structurally prevents silent drop.

### Tracking rules

- Track every finding from the last review in this table
- On each new version, update status for every row
- Any item not “resolved” must be mentioned again in the next review
- Silent drop (disappearance without mention) is forbidden

### Template

```markdown
# Finding tracking table: {matter-name}

| # | First seen | Item | Initial risk | v1 status | v2 status | v3 status | Residual risk now |
|---|------------|------|--------------|-----------|-----------|-----------|-------------------|
| C-1 | {date} | {finding} | Critical | {status} | {status} | — | {risk} |
| C-2 | {date} | {finding} | Critical | {status} | {status} | — | {risk} |
| H-1 | {date} | {finding} | High | {status} | {status} | — | {risk} |
| M-1 | {date} | {finding} | Medium | {status} | {status} | — | {risk} |

Status legend:
- Open: unchanged since last round
- Resolved: fixed and risk removed
- Partially resolved: improved but residual risk remains (detail in residual risk column)
- Worsened: new risk from the change, or existing risk increased
```
