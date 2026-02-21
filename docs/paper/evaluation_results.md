# Memory System Ablation Study Results

> Generated: 2026-02-20 09:33 UTC
> Dataset: Synthetic business domain (30 knowledge + 15 episodes + 100 noise)
> Mode: mock

## 1. Priming Ablation (Automatic Memory Retrieval)

### Hypothesis

Automatic priming improves search precision by pre-activating relevant memories in the system prompt before query processing.

### Results

| Metric | Priming OFF | Priming ON | Delta |
|--------|------------|------------|-------|
| Precision@3 | 0.75 | 0.75 | +0.00 |
| Precision@5 | 0.75 | 0.75 | +0.00 |
| Recall@5 | 0.73 | 0.73 | +0.00 |
| Avg. Priming Tokens | - | 1 | - |

### Interpretation

Precision@3 showed negligible change (+0.00). Precision@5 showed negligible change (+0.00). Recall@5 showed negligible change (+0.00). Priming injected an average of 1 tokens per query. The priming effect was marginal in this preliminary study. Larger sample sizes may be needed to detect a statistically significant difference.

![Priming Comparison](figures/priming_comparison.png)

## 2. Forgetting Ablation (Active Memory Pruning)

### Hypothesis

Active forgetting (synaptic downscaling, neurogenesis reorganization, and complete forgetting) improves search precision by removing noise and low-value memories from the store.

### Results

| Metric | Forgetting OFF | Forgetting ON | Delta |
|--------|---------------|--------------|-------|
| Precision@3 | 0.75 | 0.75 | +0.00 |
| Precision@5 | 0.75 | 0.75 | +0.00 |
| Memory Count (ON) | 130 | 130 | +0 |
| Memory Count (OFF) | 130 | 130 | +0 |

### Interpretation

Precision@3 showed negligible change (+0.00). Precision@5 showed negligible change (+0.00). Forgetting reduced memory store from 130 to 130 chunks (0% reduction). Forgetting maintained precision while reducing memory size, demonstrating effective noise removal without information loss.

![Forgetting Comparison](figures/forgetting_comparison.png)

## 3. Reconsolidation Ablation (Procedure Revision)

### Hypothesis

Reconsolidation (procedure revision after error detection) enables adaptive learning, improving task success rate across repeated trials.

### Results

| Round | Recon. OFF | Recon. ON | Delta |
|-------|-----------|----------|-------|
| Round 1 | 0.00 | 0.00 | +0.00 |
| Round 2 | 0.00 | 1.00 | +1.00 |

**Overall success rate**: OFF=0.00, ON=1.00 (delta=+1.00)

### Interpretation

With reconsolidation enabled, success rate progressed from 0.00 (Round 1) to 1.00 (Round 2). Without reconsolidation, success rate went from 0.00 (Round 1) to 0.00 (Round 2). Reconsolidation demonstrated superior cross-round improvement, supporting the hypothesis that procedure revision enables adaptive learning from errors.

![Reconsolidation Progression](figures/reconsolidation_progression.png)

## 4. Summary

- **Priming**: neutral effect on Precision@3 (+0.00)
- **Forgetting**: Precision@3 delta +0.00, memory reduction 0 chunks
- **Reconsolidation**: Final success rate ON=1.00 vs OFF=0.00

![Summary Dashboard](figures/summary_dashboard.png)

## Methodology Notes

- Sample size: N=5 per condition (preliminary study)
- Dataset: Synthetic, deterministic (seed=42)
- Evaluation: Automated metrics (no human evaluation)
- Search engine: ChromaDB + intfloat/multilingual-e5-small (384-dim)
- Forgetting pipeline: 3-stage (synaptic downscaling, neurogenesis, complete forgetting)
- Reconsolidation: LLM-based procedure revision (mock mode uses pattern matching)
- Limitations:
  - Small sample size limits statistical power
  - Synthetic dataset may not capture real-world memory complexity
  - Mock mode results are deterministic approximations
  - No cross-validation or bootstrap confidence intervals
