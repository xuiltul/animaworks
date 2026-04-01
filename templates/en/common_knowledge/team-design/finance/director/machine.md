# Finance Director — machine usage patterns

## Ground rules

1. **Write the plan document first** — Do not run on a short inline instruction string only. Pass a plan file
2. **Output is draft** — Always validate machine output yourself before `status: approved` for the next step
3. **Storage**: `state/plans/{YYYY-MM-DD}_{summary}.{type}.md` (no `/tmp/`)
4. **Rate limits**: chat 5/session, heartbeat 2
5. **Machine cannot reach infra** — Put memory, messaging, and org context into the plan document

---

## Overview

Finance Director combines PdM (planning and judgment) and Engineer (execution).

- Analysis plan (`analysis-plan.md`) → written by Director
- Analysis execution → delegated to machine; validated by Director
- Risk and interpretation → decided by Director
- Two verification passes: after machine analysis and when integrating Auditor feedback

---

## Phase 1: Analysis plan (PdM equivalent)

### Step 1: Review the Variance Tracker

If a prior analysis exists, load the Variance Tracker and understand status for every variance.

### Step 2: Create analysis-plan.md (Director writes it)

Create a plan that states purpose, subject, scope, perspective, and carry-forward items.

```bash
write_memory_file(path="state/plans/{date}_{subject}.analysis-plan.md", content="...")
```

**The “analysis perspective,” “scope,” and “carry-forward from prior” sections are core Director judgment — machine must never write them (NEVER).**

### Step 3: Approve analysis-plan.md

Review the plan you wrote and set `status: approved`.

## Phase 2: Analysis execution (Engineer equivalent)

### Step 4: Run analysis via machine

Pass analysis-plan.md and source data as inputs.

```bash
animaworks-tool machine run \
  "$(cat $ANIMAWORKS_ANIMA_DIR/state/plans/{analysis-plan.md})" \
  -d /path/to/workspace
```

Save the result as `state/plans/{date}_{subject}.analysis-report.md` (`status: draft`).

**When submitting analysis:**
- Include source data (from Analyst/Collector) in the plan (machine cannot read memory)
- Include every Variance Tracker line item (ask machine to update status for each)
- Specify the output format

### Step 5: Validate analysis-report.md

Read analysis-report.md and validate against `director/checklist.md`:

- [ ] Every Variance Tracker item is covered (no silent drop)
- [ ] All figures verified programmatically (assert for identities and consistency)
- [ ] Interpretations and assumptions sufficiently supported
- [ ] Estimates marked “estimate”

If issues exist, fix Director-side and set `status: reviewed`.

### Step 6: Delegate

Hand `analysis-report.md` with `status: reviewed` to Auditor via `delegate_task`.

## Phase 3: Integration and final judgment

### Step 7: Integrate feedback

After receiving Auditor `audit-report.md`:

- Re-check items challenged on assumptions against historical data
- Re-verify sources for Data Lineage issues
- Recalculate if numerical accuracy is questioned
- Bring Variance Tracker to latest status

### Step 8: Final report

Set consolidated analysis-report.md to `status: approved` and report to humans via `call_human`.

---

## Analysis plan template (analysis-plan.md)

```markdown
# Financial analysis plan: {subject name}

status: draft
author: {anima name}
date: {YYYY-MM-DD}
type: analysis-plan

## Purpose

{What to clarify — 1–3 sentences}

## Period and entities

| Subject | Period | Source data |
|---------|--------|-------------|
| {entity / subject} | {period} | {data files} |

## Carry-forward from prior analysis

| # | Prior variance | Prior status | Check this time |
|---|----------------|--------------|-----------------|
| V-1 | {detail} | Watch | {Resolved? Not worsened?} |
| ... | ... | ... | ... |

(If first analysis, state “first analysis” explicitly)

## Analysis perspective (scope)

{Set by Director judgment}

1. {Perspective 1}
2. {Perspective 2}
3. {Perspective 3}

## Out of scope

- {Exclusions}

## Output format

- Key metrics summary
- Outliers and material variances list
- Account-level analysis (if applicable)
- Variance Tracker status updates
- Recommended actions

## Deadline

{deadline}
```

## Analysis report template (analysis-report.md)

```markdown
# Financial analysis report: {subject name}

status: draft
author: {anima name}
date: {YYYY-MM-DD}
type: analysis-report
source: state/plans/{original analysis-plan.md}

## Overall assessment

{1–3 sentence summary of the analysis}

## Key metrics summary

| Metric | Current | Prior | Change % | Assessment |
|--------|---------|-------|----------|------------|
| {name} | {value} | {value} | {%} | {OK / watch / alert} |

## Outliers and material variances

| # | Item | Change % | Risk | Basis | Recommended action |
|---|------|----------|------|-------|-------------------|
| 1 | {name} | {%} | Critical | {basis} | {action} |
| 2 | {name} | {%} | High | {basis} | {action} |

## Account-level analysis

### {Account name}

- **Figures**: {current} / {prior} / change {%}
- **Analysis**: {drivers and context}
- **Assumptions**: {if any}
- **Recommended action**: {concrete action}

Repeat per account as needed.

## Variance Tracker status updates

| # | Prior variance | Prior status | Current status | Residual risk |
|---|----------------|--------------|----------------|---------------|
| V-1 | {detail} | Watch | {resolved / watch / worsening} | {assessment} |

## Numerical verification summary

{Summary of programmatic checks (assert, etc.)}

## Recommended actions

{Prioritized concrete actions}
```

---

## Constraints

- analysis-plan.md MUST be written by Director
- Interpretation and judgment on analysis-report MUST be finalized by Director (treat machine output as draft)
- NEVER send analysis-report.md to Auditor without `status: reviewed`
- NEVER let Variance Tracker items vanish without mention (NEVER — silent drop forbidden)
- Verify all figures programmatically before reporting (MUST — do not trust LLM arithmetic)
