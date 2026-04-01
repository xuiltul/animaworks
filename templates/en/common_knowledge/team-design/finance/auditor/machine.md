# Financial Auditor — machine usage patterns

## Ground rules

1. **Write the plan document first** — Do not run on a short inline instruction string only. Pass a plan file
2. **Output is draft** — Always validate machine output yourself before `status: approved` for the next step
3. **Storage**: `state/plans/{YYYY-MM-DD}_{summary}.{type}.md` (no `/tmp/`)
4. **Rate limits**: chat 5/session, heartbeat 2
5. **Machine cannot reach infra** — Put memory, messaging, and org context into the plan document

---

## Overview

Auditor **delegates verification scans to machine and validates that those scans are sound (meta-verification)**.

- Verification design → Auditor decides
- Recalculation, Variance Tracker reconciliation, Data Lineage tracing → delegate to machine
- Validity of findings → Auditor decides
- Assumption soundness (Assumption Challenge) → Auditor decides

Machine can recalculate, diff, and trace Data Lineage quickly; building downside scenarios and judging assumptions remains Auditor’s job.

---

## Workflow

### Step 1: Write the verification plan (Auditor writes it)

Create a plan that states perspective, subject, and criteria.

```bash
write_memory_file(path="state/plans/{date}_{subject}.audit-plan.md", content="...")
```

Before creating it, gather on the Anima side:
- Director’s `analysis-report.md` and `analysis-plan.md`
- Variance Tracker (for comparison with prior)
- Source data (if needed for independent verification)

### Step 2: Run verification scan via machine

```bash
animaworks-tool machine run \
  "$(cat $ANIMAWORKS_ANIMA_DIR/state/plans/{audit-plan.md})" \
  -d /path/to/workspace
```

Append or write `state/plans/{date}_{subject}.audit-report.md` (`status: draft`).

### Step 3: Meta-verify the results

Read audit-report.md and confirm:

- [ ] Findings are factual (no false positives)
- [ ] Variance Tracker reconciliation has no gaps
- [ ] Data Lineage traces are accurate
- [ ] Assumption Challenge coverage for Director’s assumptions
- [ ] Any additional points from your own review

Edit and supplement yourself; set `status: approved`.

### Step 4: Feedback

Send approved audit-report.md to Director.
If issues exist, include disconfirming data and recommended fixes.

---

## Verification plan template (audit-plan.md)

```markdown
# Verification plan: {subject summary}

status: draft
author: {anima name}
date: {YYYY-MM-DD}
type: audit-plan

## Verification focus

- [ ] Assumption Challenge: disconfirm Director’s assumptions and interpretations
- [ ] Variance Tracker tracking: all prior variances reflected in this report
- [ ] Data Lineage: every figure traces to source data
- [ ] Numerical accuracy: independent recalculation of key metrics
- [ ] Accounting identities: balance sheet and trial balance consistency

## Subjects

- analysis-report.md: {path}
- analysis-plan.md: {path}
- Variance Tracker: {path}
- Source data: {path / location}

## Required output format

Produce findings in the following structure. **Output not following this format is invalid.**

- **Critical**: Must fix (weak assumptions, silent drop, numerical inconsistency)
- **Warning**: Should fix (insufficient evidence, incomplete Data Lineage)
- **Info**: Informational or improvement suggestions
```

## Verification report template (audit-report.md)

```markdown
# Verification report: {subject name}

status: draft
author: {anima name}
date: {YYYY-MM-DD}
type: audit-report

## Overall verdict

{APPROVE / REQUEST_CHANGES / COMMENT}

## Assumption Challenge results

| # | Item | Director’s view | Auditor view | Disconfirming data | Recommendation |
|---|------|-----------------|--------------|-------------------|----------------|
| 1 | {item} | {seasonal, etc.} | {concern} | {historical data} | {fix} |

## Variance Tracker leakage detection

| # | Prior variance | Prior status | Treatment in this report | Verdict |
|---|----------------|--------------|---------------------------|---------|
| 1 | {variance} | {status} | {mentioned / silent drop} | {OK / NG} |

## Data Lineage verification

| # | Figure | Source info | Verdict |
|---|--------|-------------|---------|
| 1 | {metric} | {traceable / unknown} | {OK / NG} |

## Independent recalculation

| # | Metric | Director value | Auditor recalc | Delta | Verdict |
|---|--------|----------------|----------------|-------|---------|
| 1 | {metric} | {value} | {value} | {delta} | {OK / NG} |

## Auditor comments

{Additional analysis, observations, recommendations}
```

---

## Constraints

- Verification plan (what to check) MUST be written by Auditor
- NEVER pass machine verification output straight to Director — Auditor must meta-verify
- NEVER send audit-report.md without `status: approved` to Director
- Assumption Challenge MUST be done by Auditor, not delegated wholesale to machine
