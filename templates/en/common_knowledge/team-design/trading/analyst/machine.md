# Market Analyst — machine usage guide

Guide for Market Analyst to run large-scale data analysis and model validation with machine.

> **Important**: Use machine for *data collection and computation*. *Signal interpretation and confidence* stay with Analyst.

---

## Basic rules

1. **Plan first**: Before machine, be clear what to analyze
2. **Output is draft**: Do not paste machine output straight into market-analysis.md; Analyst validates
3. **File paths**: Save plans under `state/plans/` and pass them as machine instructions
4. **Rate limits**: Obey machine concurrency limits
5. **Sensitive data**: Do not pass API keys or passwords to machine

---

## Phase 1: Data collection and preprocessing (machine)

### Use cases and instruction templates

#### 1-A: Price data and statistics

```
For the symbols/pairs below, collect data for the last {period} and compute basic stats:

Targets: {symbol/pair list}
Fields:
- OHLCV
- Volatility ({period} moving average)
- Return distribution (normality tests included)
- Correlation matrix (among targets)

Output: stats summary table + outlier flags
```

#### 1-B: Sentiment analysis

```
Analyze sentiment for {subject} from the sources below:

Sources: {news feed / social / on-chain}
Subject: {symbol/theme}
Period: last {period}
Items:
- Sentiment score over time
- Keyword frequency changes
- Abnormal sentiment spikes

Output: sentiment summary + key event list
```

---

## Phase 2: Model validation (machine)

#### 2-A: Backtest vs live divergence

```
Compare backtest and live results for the strategy below:

Backtest: {path}
Live: {path}
Compare:
- Win-rate gap
- P&L distribution gap
- Slippage actual vs assumed
- Fill-rate gap

Output: divergence summary + severity (within tolerance / investigate / stop)
```

#### 2-B: Walk-forward analysis

```
Run walk-forward analysis for the strategy parameters below:

Strategy: {name/code path}
Parameters: {list}
Train window: {period}
Test window: {period}
Step: {rolling window}

Output: performance per window + overfit ratio
```

---

## Phase 3: Analysis report support (machine)

### market-analysis.md template (Analyst adds final judgment)

```markdown
# Market analysis report: {subject}

status: draft
author: {anima name}
date: {YYYY-MM-DD}
type: market-analysis

## Executive summary
{1–3 sentence conclusion}

## Subject and period
| Symbol/pair | Timeframe | Period |
|-------------|-----------|--------|
| {symbol} | {timeframe} | {period} |

## Signals

| # | Signal | Direction | Confidence | Statistical support | Assumptions |
|---|--------|-----------|------------|---------------------|-------------|
| 1 | {signal} | {buy/sell/neutral} | {high/medium/ref} | {support} | {assumption} |

## Risk factors
- {risk 1}
- {risk 2}

## Assumptions and limits
- {Assumption: when this analysis applies}
- {Limit: cases this analysis does not cover}
```
