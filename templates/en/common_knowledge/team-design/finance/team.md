# Finance Full Team — Team Overview

## Four-role structure

| Role | Responsibility | Recommended `--role` | Example `speciality` | Details |
|------|----------------|----------------------|----------------------|---------|
| **Finance Director** | Analysis planning, financial judgment, numerical verification, final approval | `manager` | `cfo`, `finance-director` | `finance/director/` |
| **Financial Auditor** | Independent verification, assumption checks, Data Lineage verification | `researcher` | `financial-auditor` | `finance/auditor/` |
| **Data Analyst** | Source extraction, structuring, multi-stage verification | `general` | `data-analyst` | `finance/analyst/` |
| **Market Data Collector** | External market data, benchmarks, reference prices | `general` | `market-data` | `finance/collector/` |

Putting the entire flow in one Anima invites self-review blind spots (interpretation optimism bias), lost variances (silent drop), and context bloat.

Each role directory has `injection.template.md` (injection.md skeleton), `machine.md` (machine usage patterns, where applicable), and `checklist.md` (quality checklist).

> For core principles: `team-design/guide.md`

## Handoff chain

```
Analyst (source extraction) + Collector (external data) ← may run in parallel
  → Director → analysis-plan.md (approved) → machine runs analysis
    → analysis-report.md (reviewed)
      → Auditor (independent verification)
        └─ Issues found → send back to Director
        └─ APPROVE → Director → update Variance Tracker → call_human → human final review
```

### Handoff documents

| From → To | Document | Condition |
|-----------|----------|------------|
| Analyst/Collector → Director | Source data + extraction verification results | Verified |
| Director → Auditor | `analysis-report.md` + `analysis-plan.md` | `status: reviewed` |
| Auditor → Director | `audit-report.md` | `status: approved` |

### Operating rules

- **Fix cycle**: Critical → full re-verification (re-engage Auditor) / Warning → delta check only / if not resolved in three round-trips → escalate to human
- **Variance Tracker**: Track material variances across months. Flagged variances must not vanish from the next report without mention (silent drop is forbidden)
- **Data Lineage Rule**: Every figure in analysis-report.md must trace to source data. Estimates must carry an “estimate” marker
- **On machine failure**: Record in `current_state.md` → reassess on next heartbeat

## Scaling

| Scale | Composition | Notes |
|-------|---------------|-------|
| Solo | Director covers all roles (quality via checklist) | Routine monthly reporting, single-entity analysis |
| Pair | Director + Auditor | Analysis with material judgment, multi-entity comparison |
| Trio | Director + Auditor + Analyst (Collector combined) | High data volume |
| Full team | Four roles as in this template | Consolidated analysis, large engagements, portfolio review |

## Mapping to development and legal teams

| Development role | Legal team role | Finance team role | Why |
|------------------|-----------------|-------------------|-----|
| PdM (research, plan, judgment) | Director (analysis plan, judgment) | Director (analysis plan, judgment) | Decides *what* to analyze |
| Engineer (implementation) | Director + machine | Director + machine | Director runs analysis via machine; no separate “engineer” Anima |
| Reviewer (static verification) | Verifier (independent verification) | Auditor (independent verification) | Core split between *doing* and *verifying* |
| Tester (dynamic verification) | Researcher (evidence gathering) | Collector (external data) | External information for corroboration |
| — | — | Analyst (data extraction) | Finance-specific: accurate extraction and structuring from sources |

## Monthly Variance Tracker

Track material variances across months. Structurally prevent prior flagged variances from disappearing from the next report without mention (silent drop).

### Tracking rules

- When a material variance (above threshold) is detected, register it in this table
- On the next analysis, update status for every line item
- Any item not “resolved” must be mentioned again in the next report
- silent drop (disappearance without mention) is forbidden

### Template

```markdown
# Monthly variance tracker: {subject name}

| # | First seen month | Account | Initial variance % | Month M | Month M+1 | Month M+2 | Residual risk |
|---|------------------|---------|---------------------|---------|-----------|-----------|----------------|
| V-1 | {month} | {account} | {variance %} | {status} | {status} | — | {risk assessment} |
| V-2 | {month} | {account} | {variance %} | {status} | {status} | — | {risk assessment} |

Status legend:
- Resolved: Root cause known, action complete, risk removed
- Watch: Cause known but recurrence risk remains (include watch period and criteria)
- Investigating: Root cause not yet identified
- Worsening: Variance widened or new risk emerged
```
