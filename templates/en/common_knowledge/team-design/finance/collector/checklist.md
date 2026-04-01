# Market Data Collector — quality checklist

Checklist for Market Data Collector to ensure collection quality.

---

## A. Data quality

- [ ] Source URL, API name, or provider recorded
- [ ] Retrieval timestamp accurate
- [ ] Freshness adequate for the analysis (not stale without disclosure)
- [ ] Units and currency stated
- [ ] No values filled in by guess or estimate

---

## B. Completeness

- [ ] All items Director requested were collected
- [ ] No unintended gaps in time series
- [ ] Unavailable items stated explicitly (with reason)
- [ ] Cross-source checks performed where possible

---

## C. Accuracy

- [ ] When sources disagree, all values listed
- [ ] If FX conversion needed, rate and as-of date recorded
- [ ] Consistency checked vs. prior collections (no unexplained large jumps)
- [ ] For outliers, distinguished source issue vs. genuine market move

---

## D. Reporting quality

- [ ] Data structured for Director’s analysis
- [ ] Counts, period, and sources stated
- [ ] Inter-source conflicts reported
- [ ] Limitations stated (lag, coverage, precision bounds)
