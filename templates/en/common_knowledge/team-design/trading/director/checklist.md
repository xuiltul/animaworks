# Strategy Director — quality checklist

Checklist for Strategy Director to protect strategy quality and PDCA rotation, with or without machine.

---

## A. P&L monitoring gate

- [ ] Did you confirm cumulative P&L?
- [ ] Did you confirm drawdown is within `{max_drawdown_pct}`?
- [ ] If drawdown is ≥80% of threshold, did you consider mitigation?
- [ ] Did you assess P&L trend (improving/flat/worsening)?
- [ ] Did you check aggregate portfolio risk across strategies?

---

## B. Strategy quality gate

- [ ] Is there a large gap between in-sample and out-of-sample backtest performance?
- [ ] Did you run parameter sensitivity and confirm small changes do not swing results wildly?
- [ ] Did you assess edge persistence (recent win rate vs all-time)?
- [ ] Are slippage and fees realistic?
- [ ] Is survivorship bias ruled out?
- [ ] Is look-ahead bias ruled out?

---

## C. Risk management gate

- [ ] Are all position sizes within `{position_limit}`?
- [ ] Is leverage within `{leverage_limit}`?
- [ ] Did you check correlation concentration (multiple positions same direction)?
- [ ] Did you check liquidity risk (thin books)?
- [ ] Did every strategy define stop-loss rules?
- [ ] Did you check max concurrent positions?

---

## D. PDCA progress gate

- [ ] Were all prior Check findings reflected in Act?
- [ ] Is performance-tracker updated with latest status (zero silent drops)?
- [ ] Did you review open items on ops-issue-tracker?
- [ ] Did Act decisions (continue/adjust/stop) state rationale?
- [ ] Is the next Check date set?

---

## E. Delegation quality

### Before Engineer / Analyst requests

- [ ] Is strategy-plan.md `status: approved`?
- [ ] Are risk parameters explicit?
- [ ] Are completion criteria concrete and testable?
- [ ] Did you set a deadline?
- [ ] Does strategy-plan.md include carry-forward from Performance Tracker?

### Responding to Auditor feedback

- [ ] Did you review every Auditor finding?
- [ ] Did you run root cause analysis for P&L divergence?
- [ ] Did you decide actions for ops-health issues?
- [ ] Did you finalize performance-tracker / ops-issue-tracker updates?
