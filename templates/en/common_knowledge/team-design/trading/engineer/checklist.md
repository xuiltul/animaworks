# Trading Engineer — quality checklist

Checklist for Trading Engineer to protect implementation and backtest quality, with or without machine.

---

## A. Implementation quality

### Alignment with strategy-plan.md

- [ ] Does implementation reflect strategy-plan.md goals?
- [ ] Are risk parameters (drawdown threshold, position caps, etc.) wired in?
- [ ] Do completion criteria map to impl.plan.md steps?
- [ ] Are there out-of-scope changes?

### Code quality

- [ ] Does code follow project conventions?
- [ ] Is error handling adequate (API retries, position protection)?
- [ ] Is randomness seeded for reproducibility?
- [ ] Is logging sufficient (entries/exits/P&L/errors)?
- [ ] No leftover debug code or dead comments?

### Tests

- [ ] Are unit tests added for new code?
- [ ] Do all existing tests pass?
- [ ] Are edge cases covered (missing data, API failure, divide-by-zero, etc.)?

---

## B. Backtest integrity

- [ ] Is look-ahead bias ruled out (no future data)?
- [ ] Is survivorship bias ruled out (delistings, etc.)?
- [ ] Is slippage realistic?
- [ ] Are fees accurate per venue?
- [ ] Are initial capital and lot sizes realistic?
- [ ] Is in-sample vs out-of-sample split sensible?
- [ ] Enough trades for statistical significance?

---

## C. Infra health

- [ ] Is the bot process running as expected?
- [ ] Are API connections (exchange, data) healthy?
- [ ] Is latency acceptable?
- [ ] Any disk/memory issues?
- [ ] Is log rotation configured?
- [ ] Are dry-run and production clearly separated?

---

## D. Handoff quality

### Before reporting to Auditor / Director

- [ ] Is backtest-report.md complete (summary, conditions, criteria check, reproduction)?
- [ ] Did you set `status: reviewed` (self-verified)?
- [ ] Is check vs strategy-plan.md completion criteria documented?
- [ ] Are reproduction steps (commands/scripts) included?
- [ ] Are known limitations stated?
