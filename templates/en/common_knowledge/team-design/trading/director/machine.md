# Strategy Director — machine usage guide

Guide for Strategy Director to run market scans and quantitative evaluation with machine.

> **Important**: Write `strategy-plan.md` yourself (do not have machine write it). Use machine for *analysis execution*; *judgment* stays with Director.

---

## Basic rules

1. **Plan first**: Before machine, be clear what to analyze
2. **Output is draft**: Do not treat machine output as final; Director validates
3. **File paths**: Save plans under `state/plans/` and pass them as machine instructions
4. **Rate limits**: Obey machine concurrency limits
5. **Sensitive data**: Do not pass API keys or passwords to machine

---

## Phase 1: Strategy design (Director writes)

Do not use machine. Director creates `strategy-plan.md` using the template below.

### strategy-plan.md template

```markdown
# Strategy plan: {strategy name}

status: draft
author: {anima name}
date: {YYYY-MM-DD}
type: strategy-plan

## Strategy objective
{What to achieve — 1–3 sentences}

## Target markets
| Asset class | Symbol/pair | Timeframe | Exchange/platform |
|-------------|-------------|-----------|-------------------|
| {class} | {symbol} | {timeframe} | {exchange} |

## Risk parameters
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Max drawdown | {%} | {rationale} |
| Position cap | {amount or lots} | {rationale} |
| Leverage cap | {multiple} | {rationale} |
| Stop-loss rules | {conditions} | {rationale} |

## Carry-forward from Performance Tracker
| # | Prior issue | Prior status | This cycle’s plan |
|---|-------------|--------------|-------------------|
| S-{N} | {issue} | {status} | {plan} |

## Strategy hypothesis
{Edge hypothesis — why you expect profit}

## Validation plan
1. {validation step 1}
2. {validation step 2}

## Completion criteria
- [ ] Backtest meets {conditions}
- [ ] Paper-trade for {period}
- [ ] Auditor APPROVE

## Deadline
{deadline}
```

---

## Phase 2: Market scan and quantitative analysis (machine)

Director delegates market data collection and quantitative analysis to machine.

### Use cases and instruction templates

#### 2-A: Market environment scan

```
For the markets below, collect and analyze data for the last {period}:

Targets: {symbol/pair list}
Analysis items:
- Price trend (direction, volatility)
- Volume changes
- Correlation structure changes
- Key events (economic indicators, regulatory changes, etc.)

Output format:
- Per-symbol summary (3 lines max)
- Risk factor list
- Notable outliers
```

#### 2-B: Quantitative validation of backtest results

```
Quantitatively validate the following backtest results:

Target: {path to backtest-report.md}
Checks:
- In-sample vs out-of-sample performance gap
- Parameter sensitivity (±{N}% variation impact)
- Max drawdown period and recovery period
- Trade frequency and fee impact

Output: summary + overfitting risk (high/medium/low)
```

---

## Phase 3: PDCA analysis (machine)

### Use: data collection for Check phase

```
Analyze recent {period} performance for the strategy below:

Target: {strategy name / log path}
Items:
- P&L path (daily/weekly)
- Win-rate trend (improving/flat/worsening)
- Drawdown path
- Edge decay signals (recent win rate vs all-time)

Output: PDCA Check report (input for Act decisions)
```

> **Note**: Machine output is *Check* data. *Act* decisions (continue/adjust/stop) are Director’s.
