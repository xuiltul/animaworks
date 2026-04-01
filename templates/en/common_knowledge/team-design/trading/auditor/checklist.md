# Risk Auditor — quality checklist

Checklist for Risk Auditor to protect verification quality, with or without machine.
This checklist targets three failure modes common in trading teams.

---

## A. P&L verification gate (“ignoring losses”)

- [ ] Did you compute P&L independently of Director/Engineer and reconcile?
- [ ] Did you confirm drawdown is within `{max_drawdown_pct}`?
- [ ] If drawdown is ≥80% of threshold, did you warn Director early?
- [ ] Did you assess win-rate trend (improving/flat/worsening)?
- [ ] Did you detect edge decay (recent win rate vs all-time gap)?
- [ ] Did you challenge optimistic language (“still fine,” “temporary,” “within expectations”)?
- [ ] For strategies Director said to “continue,” did you consider worst case if stop were needed?
- [ ] Did you confirm backtest vs live P&L gap is within `{pl_divergence_threshold}`?

---

## B. Execution verification gate (“not verifying live trading”)

- [ ] Did you confirm bot process is running (PID, last log timestamp)?
- [ ] Did you confirm exchange/data APIs are healthy?
- [ ] Did you confirm orders actually fill (not spinning empty)?
- [ ] Are dry-run and production clearly separated?
- [ ] Are rejects/timeouts not abnormally high?
- [ ] Is slippage within expected bounds?
- [ ] Is bot downtime within `{max_downtime}`?

---

## C. Asset tracking gate (“forgetting wallet tracking”)

- [ ] Did you reconcile all exchange/wallet balances?
- [ ] Is expected vs actual gap within `{balance_tolerance}`?
- [ ] Are deposit/withdrawal/transfer logs accurate?
- [ ] Did you run a position inventory (symbols, qty, marks)?
- [ ] Can you explain balance change since last check (P&L + flows = delta)?

---

## D. Risk compliance gate

- [ ] Are all positions within `{position_limit}`?
- [ ] Is leverage within `{leverage_limit}`?
- [ ] Did you check correlation concentration (same-direction positions)?
- [ ] Do stop-loss rules fire when they should?
- [ ] Did you check max concurrent positions?

---

## E. Carry-forward gate

- [ ] Is every performance-tracker row updated?
- [ ] Were all prior “open” items mentioned again (zero silent drops)?
- [ ] Is every ops-issue-tracker row updated?
- [ ] Are newly detected issues registered in trackers?
- [ ] Are status transitions logical (no “resolved” → “open” regression)?

---

## F. Meta-verification quality

### Verifying machine verification

- [ ] Did you sample-check P&L calculations?
- [ ] Are reconciliation numbers accurate?
- [ ] Did you remove false positives?
- [ ] Did you catch false negatives?

### Feedback quality

- [ ] Do Critical findings include numeric evidence and recommended actions?
- [ ] Do optimism findings include worst-case scenarios?
- [ ] Do carry-forward gaps cite prior report sections?
- [ ] Are performance-review.md and ops-health-report.md complete?
