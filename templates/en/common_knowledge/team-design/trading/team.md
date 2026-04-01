# Trading Full Team — Team Overview

## Four-role structure

| Role | Responsibility | Recommended `--role` | Example `speciality` | Details |
|------|----------------|----------------------|----------------------|---------|
| **Strategy Director** | Strategy design, risk limits, PDCA oversight, final judgment | `manager` | `trading-director` | `trading/director/` |
| **Market Analyst** | Market analysis, signal generation, model development | `researcher` | `market-analyst` | `trading/analyst/` |
| **Trading Engineer** | Bot implementation, backtests, execution stack, data pipelines | `engineer` | `trading-engineer` | `trading/engineer/` |
| **Risk Auditor** | Independent P&L verification, ops health audit, carry-forward tracking | `engineer` or `ops` | `risk-auditor` | `trading/auditor/` |

Putting the entire flow in one Anima invites P&L optimism bias (downplaying losses), missing operational verification (not noticing bot stops), and lost issue tracking (unreviewed wallets left unaddressed).

Each role directory has `injection.template.md` (injection.md skeleton), `machine.md` (machine usage patterns), and `checklist.md` (quality checklist).

> For core principles: `team-design/guide.md`

## Handoff chain

```
Director → strategy-plan.md (approved) + performance-tracker reference
  → delegate_task
    → Engineer: bot implementation / backtest (machine)
    → Analyst: market analysis / signal validation (machine)
      → backtest-report.md + market-analysis.md (reviewed)
        → Auditor (P&L verification + ops health audit) ← independent verification
          └─ Issues found → send back to Director
          └─ APPROVE → Director → update performance-tracker → upstream report / call_human
```

### Handoff documents

| From → To | Document | Condition |
|-----------|----------|------------|
| Director → Engineer | `strategy-plan.md` | `status: approved` |
| Director → Analyst | `strategy-plan.md` (analysis perspective instructions) | `status: approved` |
| Engineer → Auditor | `backtest-report.md` | `status: reviewed` |
| Analyst → Director | `market-analysis.md` | `status: reviewed` |
| Auditor → Director | `performance-review.md` + `ops-health-report.md` | `status: approved` |

### Operating rules

- **Fix cycle**: Critical (drawdown threshold exceeded, bot stopped, asset mismatch) → immediate action + Auditor re-verification / Warning → delta check only / if not resolved in three round-trips → escalate to human
- **Performance Tracker**: Track P&L, win rate, Sharpe, and max DD across strategy versions. Prior flagged issues must not vanish from the next review without mention (silent drop is forbidden)
- **Ops Issue Tracker**: Track operational issues with carry-forward. silent drop is forbidden
- **PDCA cycle**: Plan = Director (strategy design), Do = Engineer (implementation) + Analyst (analysis), Check = Auditor (independent verification), Act = Director (judgment and correction instructions)
- **On machine failure**: Record in `current_state.md` → reassess on next heartbeat

## Scaling

| Scale | Composition | Notes |
|-------|---------------|-------|
| Solo | Director covers all roles (quality via checklist) | Paper-trade validation, single strategy |
| Pair | Director + Auditor | Medium-risk live ops with few strategies |
| Trio | Director + Engineer + Auditor | Bot development phase (Analyst covered by Director) |
| Full team | Four roles as in this template | Production ops with multiple strategies |

## Mapping to development and legal teams

| Development role | Legal team role | Trading team role | Why |
|------------------|-----------------|-------------------|-----|
| PdM (research, plan, judgment) | Director (analysis plan, judgment) | Director (strategy design, PDCA judgment) | Decides *what* to do |
| Engineer (implementation) | Director + machine | Engineer (bot implementation, backtest) | Writes code and ships working systems |
| Reviewer (static verification) | Verifier (independent verification) | Auditor (P&L verification + ops health) | Core split between *doing* and *verifying* |
| Tester (dynamic verification) | Researcher (evidence gathering) | Analyst (market analysis, signal quality) | External-data backing and quality checks |

## Strategy Performance Tracker — strategy performance tracking table

Update this table on each strategy version change. Structurally prevent prior flagged issues from disappearing from the next review without mention (silent drop).

### Tracking rules

- Add a row on each strategy version change (including parameter changes)
- If drawdown exceeds threshold `{max_drawdown_pct}`, report to Director immediately
- Any issue not marked “ongoing monitoring” must be mentioned again in the next review
- silent drop (disappearance without mention) is forbidden

### Template

```markdown
# Strategy performance tracker: {strategy name}

| # | Period | Version | P&L | Win rate | Sharpe | Max DD | Status | Notes |
|---|--------|---------|-----|----------|--------|--------|--------|-------|
| S-1 | {start–end} | {v1} | {amount} | {%} | {value} | {%} | {assessment} | {notes} |
| S-2 | {start–end} | {v2} | {amount} | {%} | {value} | {%} | {assessment} | {notes} |

Status legend:
- Live: Running in production
- Paper trade: Under validation
- Parameter tuning: Improvement cycle in progress
- Stopped: Stopped due to threshold breach or edge decay
- Retired: Strategy discarded
```

## Ops Issue Tracker — operational issue tracking table

Track operational issues with carry-forward. Structurally prevent silent drop.

### Tracking rules

- When you detect an operational issue (bot stop, API outage, asset mismatch, etc.), register it in this table
- On the next Heartbeat / review, update status for every line item
- Any item not “resolved” must be mentioned again next time
- silent drop (disappearance without mention) is forbidden

### Template

```markdown
# Ops issue tracker: {team name}

| # | Detected | Issue | Severity | Status | Owner | Resolved | Notes |
|---|----------|-------|----------|--------|-------|----------|-------|
| O-1 | {date} | {description} | Critical | {state} | {owner} | {date} | {notes} |
| O-2 | {date} | {description} | Warning | {state} | {owner} | {date} | {notes} |

Status legend:
- Open: Detected, not started
- In progress: Fix underway
- Resolved: Issue removed
- Recurred: Resolved once but happened again
- Ongoing monitoring: Mitigated for now; root cause not addressed
```
