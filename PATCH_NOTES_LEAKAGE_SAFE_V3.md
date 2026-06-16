# Leakage-safe patch v3

## Why this patch exists

This patch extends the leakage-safe rating evaluation from v2 with an **ablation study**. The goal is to test whether the rating regression truly benefits from using both sentiment probability features, or whether one sentiment model dominates.

Run from repository root:

```bash
python rating_prediction/fixed_experiment.py
```

## What v3 adds on top of v2

1. **Ablation study for sentiment probability features**
   - `LR prob only`: rating regression uses only the Logistic Regression positive probability.
   - `NB prob only`: rating regression uses only the Naive Bayes / ComplementNB positive probability.
   - `LR + NB probs`: rating regression uses both probability features.

2. **Each feature setup is tested with both rating models**
   - Linear Regression.
   - Ridge Regression.

3. **Baselines are kept**
   - Dummy Mean.
   - Dummy Median.
   - Direct TF-IDF → Ridge.

The ablation is still leakage-safe because the probability features are generated using OOF predictions for train and train-only models for test.

## Reproduced metrics

### Sentiment OOF metrics

| Model | OOF Accuracy | OOF F1 |
|---|---:|---:|
| LR_balanced | 0.9373 | 0.8938 |
| ComplementNB | 0.9339 | 0.8863 |

### Leakage-safe rating metrics with ablation

| Feature Set | Model | MAE | RMSE | R² | Within-1-Star |
|---|---|---:|---:|---:|---:|
| LR + NB probs | Linear Regression | 0.5017 | 0.8718 | 0.7384 | 85.85% |
| LR + NB probs | Ridge Regression | 0.5019 | 0.8719 | 0.7383 | 85.85% |
| LR prob only | Linear Regression | 0.5151 | 0.8727 | 0.7378 | 84.90% |
| LR prob only | Ridge Regression | 0.5154 | 0.8727 | 0.7378 | 84.90% |
| NB prob only | Linear Regression | 0.5210 | 0.9212 | 0.7079 | 85.60% |
| NB prob only | Ridge Regression | 0.5212 | 0.9212 | 0.7079 | 85.60% |
| Direct TF-IDF | Direct TFIDF + Ridge | 0.6497 | 0.9388 | 0.6966 | 75.83% |
| Trivial baseline | Dummy Median | 1.1510 | 2.0566 | -0.4560 | 71.15% |
| Trivial baseline | Dummy Mean | 1.5162 | 1.7044 | 0.0000 | 6.07% |

## Key interpretation

The best setting remains **LR + NB probability features with Linear Regression**. The ablation shows:

- LR probability alone is already strong: R² ≈ 0.738, but MAE is worse than using both features.
- NB probability alone has competitive Within-1 but lower R², meaning it captures broad polarity but explains less continuous rating variance.
- Combining LR + NB gives the best MAE and R², so keeping both features is justified.
- Ridge and Linear remain nearly identical because the sentiment feature space is only 1–2 dimensions.

## Recommended wording for report/slide

> We added an ablation study to test the contribution of each sentiment probability feature. LR-only and NB-only features both contain useful polarity information, but combining both probabilities gives the best MAE and R². This supports the design choice of using both sentiment models as compact features for rating regression.

## Defense caveat

The method should still be described as **sentiment-informed rating prediction on polar reviews {1,2,4,5}**, not a full 1-to-5 rating predictor. Rating 3 was removed to create the binary sentiment task, so neutral reviews are outside the current evaluation scope.
