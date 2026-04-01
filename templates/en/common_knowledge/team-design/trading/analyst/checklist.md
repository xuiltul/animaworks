# Market Analyst — quality checklist

Checklist for Market Analyst to protect signal and data quality, with or without machine.

---

## A. Signal quality

- [ ] Does each signal include statistical support (p-value, CI, sample size, etc.)?
- [ ] Is confidence stated in three tiers (high / medium / reference)?
- [ ] Did you assess false positive rate?
- [ ] Did you check hit rate for similar past signals?
- [ ] Are signals backed by multiple independent sources?
- [ ] Does analysis account for regime (trend/range/volatility)?

---

## B. Data quality

- [ ] Did you verify source reliability?
- [ ] Did you check data freshness (last update)?
- [ ] Did you document missing/outlier handling?
- [ ] Is the sample period sufficient for the timeframe?
- [ ] Are cross-source contradictions resolved or explained?

---

## C. Model validation

- [ ] Did you check backtest vs live performance gap?
- [ ] Did walk-forward analysis show overfitting?
- [ ] Did you assess robustness to regime change?
- [ ] Did you run parameter sensitivity (small changes should not swing results)?
- [ ] Is survivorship bias ruled out?
- [ ] Is look-ahead bias ruled out?

---

## D. Report quality

- [ ] Does market-analysis.md follow the format?
- [ ] Does the executive summary state the conclusion clearly?
- [ ] Does every signal list confidence and statistical support?
- [ ] Are assumptions and limits explicit?
- [ ] Are risk factors listed?
- [ ] For “high” confidence signals, is backing strong enough (otherwise lower confidence)?

---

## E. Verifying machine output

### When delegating data collection and computation to machine

- [ ] Did you spot-check calculations (sampling OK)?
- [ ] Are outliers handled correctly?
- [ ] Is the fetched range/period as instructed?
- [ ] Are units and scales for generated stats correct?
- [ ] Did you add Analyst judgment before finalizing market-analysis.md?
