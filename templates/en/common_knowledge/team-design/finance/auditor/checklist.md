# Financial Auditor — quality checklist

Checklist for Financial Auditor to ensure verification quality, whether or not machine is used.

---

## A. Assumption Challenge gate

- [ ] Listed all Director assumptions (“seasonal,” “temporary,” “industry average,” etc.)
- [ ] Confirmed every “seasonal” judgment with at least 12 months of history
- [ ] Every “temporary” judgment has a tracking window and recurrence criteria
- [ ] Considered downside (sensitivity) scenarios for optimistic forecasts
- [ ] Every “industry average” / “standard” claim has concrete benchmark data
- [ ] Items downgraded to lower risk vs. prior have sufficient justification

---

## B. Data Lineage verification

- [ ] Every figure in analysis-report traces to source data
- [ ] Estimates marked “estimate”
- [ ] No orphan calculations (figures not tied to sources)
- [ ] Full Variance Tracker reflected in analysis-report (silent drop detection)
- [ ] No prior Critical item missing mention this period (silent drop detection)

---

## C. Numerical accuracy

- [ ] Independent recalculation of key metrics
- [ ] Accounting identities (balance sheet, trial balance) hold
- [ ] Reviewed Director’s verification results (assert, etc.)
- [ ] Alternative calculation path as a cross-check where useful
- [ ] Change rates correct (numerator/denominator not swapped)
- [ ] Rounding consistency checked

---

## D. Cross-entity reconciliation (consolidated / group only)

- [ ] Intercompany balances eliminated
- [ ] Elimination entries complete (no gaps)
- [ ] No double counting
- [ ] FX rates and translation method consistent

---

## E. Meta-verification quality

### Machine verification output

- [ ] Removed false positives
- [ ] Supplemented false negatives
- [ ] Severity (Critical/Warning/Info) appropriate
- [ ] Findings fact-based (not speculation)

### Feedback quality

- [ ] Critical items include disconfirming data and recommended fixes
- [ ] Assumption Challenge includes historical or downside evidence
- [ ] Constructive tone (issue + improvement)
- [ ] silent drop citations reference the prior report
