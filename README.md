# GPBoost: Structured Variance in Cortisol Prediction

An experiment comparing naive gradient boosting against **GPBoost** (Sigrist, 2022) on synthetic longitudinal cortisol data with grouped random effects.

## Motivation

Standard GBDTs (LightGBM, XGBoost) assume row independence. When observations are grouped — e.g., repeated measurements per person — individual-level baseline shifts become unexplained noise. GPBoost combines GBDTs with Gaussian Processes to explicitly model this structured variance.

## Experiment

Synthetic cortisol data is generated from a hierarchical model:

```
y_ij = f(x_ij) + b_i + ε_ij
```

- **f(x)** — fixed effects: diurnal rhythm, stress, sleep, age
- **b_i ~ N(0, σ²_b)** — per-donor random effect (the structured variance)
- **ε_ij ~ N(0, σ²_ε)** — residual noise

Two models are compared:

| Model | Sees donor ID | Handles unseen donors |
|---|---|---|
| Naive GBDT | ✗ | ✗ |
| GPBoost | ✓ (grouped RE) | ✓ (RBF kernel) |

Performance is evaluated separately on **seen vs. unseen donors** — the split where GPBoost's GP kernel is expected to show the clearest advantage.

## Goals

- [ ] Generate synthetic grouped cortisol data
- [ ] Benchmark LightGBM against grouped data
- [ ] Implement GPBoost with grouped random effects
- [ ] Evaluate GP kernel generalization on held-out donors

## References

- Sigrist, F. (2022). [GPBoost Python Package](https://github.com/pfriedrich/GPBoost)
- Sigrist, F. (2020). Gaussian Process Boosting. *Journal of Machine Learning Research*
