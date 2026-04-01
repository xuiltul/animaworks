# Risk Auditor — machine usage guide

Guide for Risk Auditor to run P&L calculations and operational health scans with machine.

> **Important**: Use machine for *data collection and computation*. *Risk judgment and APPROVE/REJECT* stay with Auditor. Do not forward machine output to Director unverified.

---

## Basic rules

1. **Plan first**: Before machine, be clear what to verify
2. **Output is draft**: Do not treat machine output as final; Auditor adds judgment
3. **File paths**: Save plans under `state/plans/` and pass them as machine instructions
4. **Rate limits**: Obey machine concurrency limits
5. **Sensitive data**: Do not pass API keys or passwords to machine

---

## Phase 1: P&L verification (machine)

### Use cases and instruction templates

#### 1-A: Independent P&L calculation

```
From the trade log below, compute P&L independently:

Log file: {path}
Period: {start–end}
Compute:
- Total P&L (fees included)
- Win rate (wins/losses/breakevens)
- Profit factor
- Max drawdown (amount, %, duration)
- Max consecutive losses
- Sharpe ratio

Compare to: {values reported by Director/Engineer}
Output: computed results + diff list vs reported values
```

#### 1-B: Expected vs actual divergence

```
Compare backtest expectations to live performance:

Backtest: {path to backtest-report.md}
Live: {path to trade log}
Compare:
- P&L gap %
- Win-rate gap
- Max drawdown gap
- Trade frequency gap
- Slippage actual vs assumed

Output: divergence summary + flag items exceeding `{pl_divergence_threshold}`
```

---

## Phase 2: Operational health scan (machine)

#### 2-A: Process and connectivity

```
Check health of the services below:

Targets:
- {bot 1}: process up + last log timestamp
- {bot 2}: process up + last log timestamp
- API: {exchange} API response check

Output: per-service status (up/down/no response) + last activity time
```

#### 2-B: Balance reconciliation

```
Reconcile balances below:

Targets:
- {exchange/wallet 1}: expected {amount} vs actual
- {exchange/wallet 2}: expected {amount} vs actual
- Total portfolio: expected total {amount} vs actual

Tolerance: {balance_tolerance}

Output: reconciliation + flag items exceeding tolerance
```

#### 2-C: Fill verification

```
Reconcile orders and fills for the period below:

Period: last {period}
Target: {bot/strategy name}

Checks:
- Orders placed vs filled count
- Partial fills and impact
- Rejects/timeouts
- Expected slippage vs actual

Output: fill summary + anomaly flags
```

---

## Phase 3: Carry-forward verification (machine)

#### 3-A: Tracker consistency

```
Verify consistency of the trackers below:

performance-tracker: {path}
ops-issue-tracker: {path}

Checks:
- Were all items “open” last review mentioned again (silent drop detection)?
- Are status transitions logical (e.g. no “resolved” → “open” regression)?
- Do new rows have first-seen date and severity?

Output: consistency result + silent-drop list
```

---

## Verification report templates

### performance-review.md

```markdown
# Performance verification report: {strategy}

status: draft
author: {anima name}
date: {YYYY-MM-DD}
type: performance-review

## Verification summary
| Item | Director/Engineer reported | Auditor independent | Gap | Verdict |
|------|------------------------------|---------------------|-----|---------|
| P&L | {value} | {value} | {diff} | {OK/investigate} |
| Win rate | {%} | {%} | {diff} | {OK/investigate} |
| Max DD | {%} | {%} | {diff} | {OK/investigate} |

## Optimism bias check
{Verification of phrases like “still fine,” “temporary,” etc.}

## Carry-forward verification
{Tracking of prior findings; silent drops yes/no}

## Overall: {APPROVE / REQUEST_CHANGES}
{Rationale}
```

### ops-health-report.md

```markdown
# Ops health report: {subject}

status: draft
author: {anima name}
date: {YYYY-MM-DD}
type: ops-health-report

## Service status
| Service | Status | Last activity | Notes |
|---------|--------|---------------|-------|
| {bot} | {up/down} | {time} | {notes} |

## Balance reconciliation
| Account/wallet | Expected | Actual | Diff | Verdict |
|----------------|----------|--------|------|---------|
| {name} | {amount} | {amount} | {diff} | {OK/investigate} |

## Fill verification
| Strategy | Orders | Fills | Fill rate | Notes |
|----------|--------|-------|-----------|-------|
| {name} | {N} | {N} | {%} | {notes} |

## ops-issue-tracker updates
{New issues + status updates on existing}

## Overall: {APPROVE / REQUEST_CHANGES}
{Rationale}
```
