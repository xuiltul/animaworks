# Legal Director — machine usage patterns

## Ground rules

1. **Write the plan document first** — Do not run on a short inline prompt string. Pass a plan file.
2. **Output is draft** — Always validate machine output yourself before `status: approved` for the next step.
3. **Save location**: `state/plans/{YYYY-MM-DD}_{summary}.{type}.md` (no `/tmp/`)
4. **Rate limits**: chat 5/session, heartbeat 2
5. **Machine has no infra access** — Put memory, messaging, and org context into the plan document.

---

## Overview

Legal Director combines PdM (planning and judgment) and Engineer (execution).

- Analysis plan (`analysis-plan.md`) → Director writes
- Contract scan execution → delegate to machine; Director validates
- Final risk rating → Director decides
- Two validation passes: machine scan output, then Verifier/Researcher feedback

---

## Phase 1: Analysis plan (PdM-like)

### Step 1: Review the carry-forward tracker

If there was a prior review, read the carry-forward tracker and understand status for every finding.

### Step 2: Create `analysis-plan.md` (Director writes)

Create a plan that states purpose, subject, lenses, and carry-forward items.

```bash
write_memory_file(path="state/plans/{date}_{matter-name}.analysis-plan.md", content="...")
```

**The “analysis lenses,” “scope,” and “carry-forward” sections in `analysis-plan.md` are core Director judgment — NEVER have machine write them.**

### Step 3: Approve `analysis-plan.md`

Review the `analysis-plan.md` you wrote and set `status: approved`.

## Phase 2: Contract scan (Engineer-like)

### Step 4: Run contract scan on machine

Pass `analysis-plan.md` as input and ask machine for a full-clause analysis of the contract.

```bash
animaworks-tool machine run \
  "$(cat $ANIMAWORKS_ANIMA_DIR/state/plans/{analysis-plan.md})" \
  -d /path/to/workspace
```

Save the result as `state/plans/{date}_{matter-name}.audit-report.md` (`status: draft`).

**When submitting the scan:**
- Include the full contract text in the plan (machine cannot read memory)
- Include every carry-forward finding (ask machine to update status per item)
- Specify output format (e.g. risk matrix)

### Step 5: Validate `audit-report.md`

Read `audit-report.md` and check against `director/checklist.md`:

- [ ] Every carry-forward finding is covered (no silent drop)
- [ ] Each clause has legal basis for its risk rating
- [ ] “Acceptable” / “no further negotiation” decisions state why
- [ ] Any downgrade from the prior review has a stated reason

Fix issues yourself, then set `status: reviewed`.

### Step 6: Delegate

Hand `audit-report.md` with `status: reviewed` to Verifier and Researcher via `delegate_task`.

## Phase 3: Integration and final judgment

### Step 7: Integrate feedback

After `verification-report.md` and `precedent-report.md`:

- Revisit risk for items flagged for optimism bias
- Fold Researcher evidence into `audit-report.md`
- Update carry-forward tracker to latest status

### Step 8: Final report

Set consolidated `audit-report.md` to `status: approved` and report to humans with `call_human`.

---

## Template: analysis plan (`analysis-plan.md`)

```markdown
# Legal analysis plan: {matter-name}

status: draft
author: {anima-name}
date: {YYYY-MM-DD}
type: analysis-plan

## Purpose

{What to clarify — 1–3 sentences}

## Subject documents

| Document | Version | Received |
|----------|---------|----------|
| {name} | {version} | {date} |

## Carry-forward from prior review

| # | Prior finding | Prior risk | What to verify this round |
|---|---------------|------------|---------------------------|
| C-1 | {content} | Critical | {Fixed? Residual risk?} |
| ... | ... | ... | ... |

(If no prior review, state “Initial analysis”.)

## Analysis lenses (scope)

{Set by Director’s judgment}

1. {lens-1}
2. {lens-2}
3. {lens-3}

## Out of scope

- {excluded}

## Deliverable format

- Risk matrix (item / risk / basis / recommended action)
- carry-forward tracker status updates
- Negotiation priority ordering
- Draft email (if needed)

## Deadline

{deadline}
```

## Template: audit report (`audit-report.md`)

```markdown
# Audit report: {matter-name}

status: draft
author: {anima-name}
date: {YYYY-MM-DD}
type: audit-report
source: state/plans/{source analysis-plan.md}

## Executive summary

{Overall contract risk — 1–3 sentences}

## Risk matrix

| # | Clause | Risk | Basis | Recommended action | Negotiation priority |
|---|--------|------|-------|---------------------|----------------------|
| 1 | {clause name / number} | Critical | {legal basis} | {concrete action} | Highest |
| 2 | {clause name / number} | High | {legal basis} | {concrete action} | High |
| ... | ... | ... | ... | ... | ... |

## Clause-by-clause analysis

### {clause name}

- **Text**: {quote}
- **Risk**: {Critical/High/Medium/Low}
- **Analysis**: {legal analysis}
- **Recommended action**: {negotiation point}

(Repeat for each clause.)

## Carry-forward status updates

| # | Prior finding | Prior risk | This round | Residual risk |
|---|---------------|------------|------------|---------------|
| C-1 | {content} | Critical | {resolved/partial/open/worsened} | {assessment} |
| ... | ... | ... | ... | ... |

## Pre-sign checklist

{What must be fixed before signature vs acceptable after}

## Additional notes

{Director notes}
```

---

## Constraints

- `analysis-plan.md` MUST be written by Director
- Risk ratings in `audit-report.md` MUST be finalized by Director (treat machine output as draft)
- NEVER hand `audit-report.md` to Verifier / Researcher without `status: reviewed`
- NEVER let a tracked carry-forward finding disappear without mention (NEVER — silent drop forbidden)
