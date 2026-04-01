# Finance Director — quality checklist

Checklist for Finance Director to ensure deliverable quality, whether or not machine is used.

---

## A. Consistency with prior analysis (Variance Tracker gate)

- [ ] All material variances from the prior analysis are listed
- [ ] Each variance has status: resolved / watch / investigating / worsening
- [ ] “Watch” items include watch period and decision criteria
- [ ] “Investigating” items include progress toward root cause
- [ ] Variance Tracker status is updated for every row (zero silent drops)

---

## B. Financial statement integrity

- [ ] Balance sheet identity holds (assets = liabilities + equity)
- [ ] Trial balance debits equal credits
- [ ] Cash flow ending cash reconciles to balance sheet cash
- [ ] Subtotals and totals match sum of line items
- [ ] No transcription errors from source data (Data Lineage check)
- [ ] Periods (dates) are correct

---

## C. Numerical verification

- [ ] All calculations verified programmatically (not relying on LLM arithmetic)
- [ ] Key identities and consistency checked with assert (or equivalent)
- [ ] Reporting only after all checks pass
- [ ] Rounding and scale applied consistently
- [ ] FX rates and source dates recorded where applicable
- [ ] If mismatches found, root cause found before reporting

---

## D. Analysis-type-specific checks

Depending on analysis type, verify applicable items:

### D-1. Monthly trial balance

- [ ] Prior-period comparisons (prior month, prior-year same month) included
- [ ] All material moves above threshold detected and reported
- [ ] Drivers explained (seasonality, one-offs, structural change)
- [ ] Outliers judged against objective criteria

### D-2. Portfolio valuation

- [ ] Valuation date and method stated
- [ ] Comparison to benchmarks / reference indicators included
- [ ] P&L on valuation accurate
- [ ] Analysis by asset class included where relevant

### D-3. Consolidated / group

- [ ] Intercompany transactions eliminated correctly
- [ ] FX rates and translation method stated
- [ ] Legal-entity data reconciles to consolidated figures
- [ ] No double counting

### D-4. Cash flow

- [ ] Assumptions for cash forecast stated
- [ ] Consistency with repayment schedules checked
- [ ] Liquidity shortfall risk assessed

---

## E. Output quality

- [ ] Every figure traces to source (file + field)
- [ ] Estimates marked “estimate”
- [ ] Plain language for non-finance readers
- [ ] Recommended actions for every risk item
- [ ] No factual errors in names, periods, amounts, units

---

## F. Delegation quality

### Before requesting Auditor

- [ ] analysis-report.md has `status: reviewed`
- [ ] analysis-plan.md attached
- [ ] Variance Tracker at latest status
- [ ] Numerical verification (assert, etc.) summarized
- [ ] Auditor can access sources and prior reports
- [ ] Deadline set

### After feedback

- [ ] All Auditor feedback reviewed
- [ ] Assumption challenges re-checked with historical data
- [ ] Data Lineage issues re-checked at source
- [ ] Final Variance Tracker status updated
