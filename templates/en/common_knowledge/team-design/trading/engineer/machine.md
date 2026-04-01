# Trading Engineer — machine usage guide

Guide for Trading Engineer to implement bots, run backtests, and change infra with machine.

> **Important**: Do not run machine without a plan. Always create an MD plan first and pass it as the instruction.

---

## Basic rules

1. **Plan first**: Create an impl.plan.md-equivalent plan before machine
2. **Output is draft**: Do not ship machine output to production unchanged; Engineer validates
3. **File paths**: Save plans under `state/plans/` and pass them as machine instructions
4. **Workspace**: Always pass a workspace directory to machine (`-d {path}`)
5. **Rate limits**: Obey machine concurrency limits
6. **No direct prod changes**: Do not have machine change production directly

---

## Phase 1: Implementation plan (Engineer writes)

After receiving strategy-plan.md, capture technical details in a plan document.

### impl.plan.md template

```markdown
# Implementation plan: {strategy/task name}

status: draft
author: {anima name}
date: {YYYY-MM-DD}
type: impl-plan
source: {path to strategy-plan.md}

## Overview
{Technical approach to strategy-plan.md goals — 1–3 sentences}

## Implementation steps
1. {Step 1: concrete work}
2. {Step 2: concrete work}
3. ...

## Test plan
- [ ] {Test 1: unit}
- [ ] {Test 2: integration}
- [ ] {Test 3: dry-run}

## Risk parameter implementation
| Parameter | Value in strategy-plan.md | How implemented |
|-----------|---------------------------|-----------------|
| Max drawdown | {value} | {where} |
| Position cap | {value} | {where} |
| Stop-loss rules | {conditions} | {where} |

## Rollback plan
{Recovery steps if something goes wrong}
```

---

## Phase 2: Bot implementation (machine)

### Use cases and instruction templates

#### 2-A: Strategy logic

```
Implement strategy logic per the plan below:

Plan: {path to impl.plan.md}
Workspace: {repo path}

Requirements:
- {language/framework}
- Fixed random seed (reproducibility)
- Logging (entries/exits/P&L)
- Error handling (API retries, position protection)

Tests:
- Add unit tests
- All existing tests must pass
```

#### 2-B: Backtest run

```
Run a backtest for the strategy below:

Strategy: {code path}
Data: {data path or fetch method}
Period: {start–end}
Parameters: {parameter set}

Backtest settings:
- Slippage: {assumption}
- Fees: {exchange fee rate}
- Initial capital: {amount}

Output:
- P&L path
- Win rate, profit factor
- Max drawdown (amount, %, duration)
- Trade count and average holding period
- Sharpe ratio
```

---

## Phase 3: Infra changes (machine)

### Use: deploy and config changes

```
Apply the infra changes below:

Plan: {path to impl.plan.md}
Target: {server/service}

Changes:
- {change 1}
- {change 2}

Verification:
- {how to verify after change}

Rollback:
- {how to recover if something fails}
```

> **Note**: Production changes must be validated in dry-run. Do not pass production API keys to machine.

---

## backtest-report.md template

```markdown
# Backtest report: {strategy name}

status: draft
author: {anima name}
date: {YYYY-MM-DD}
type: backtest-report
source: {path to strategy-plan.md}

## Summary
| Metric | Value |
|--------|-------|
| Period | {start–end} |
| P&L | {amount} |
| Win rate | {%} |
| Profit factor | {value} |
| Max drawdown | {%} ({amount}) |
| Sharpe | {value} |
| Trades | {count} |

## Backtest conditions
| Setting | Value |
|---------|-------|
| Slippage | {assumption} |
| Fees | {rate} |
| Initial capital | {amount} |
| Random seed | {value} |

## Check vs strategy-plan.md completion criteria
- [ ] {criterion 1}: {result}
- [ ] {criterion 2}: {result}

## Known limitations
- {limitation 1}

## Reproduction
{Commands/steps to reproduce the backtest}
```
