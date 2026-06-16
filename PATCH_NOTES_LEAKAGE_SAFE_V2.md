# Leakage-safe patch v2

## Why this patch exists

The first implementation produced sentiment probabilities on the full dataset after fitting sentiment models on a train split. That makes many rating-regression features in-sample predictions and weakens the validity of the reported rating metrics.

This patch keeps the original project structure but adds a stricter evaluation script:

```bash
python rating_prediction/fixed_experiment.py
```

## What v2 fixes/adds

1. **Train/test split before vectorization**
   - `TfidfVectorizer.fit(...)` is called only on the training text.
   - Test text is transformed with the train-fitted vectorizer.

2. **Out-of-fold sentiment features for rating training**
   - Training sentiment probabilities are generated with 5-fold OOF prediction.
   - Each training row receives a probability from a model that did not train on that row.
   - Test sentiment probabilities come from sentiment models trained only on the full training split.

3. **Negation words are preserved**
   - `no`, `nor`, and `not` are removed from the stopword list.

4. **More honest baselines**
   - `Dummy Mean` is kept as the R² reference.
   - `Dummy Median` is added because it is a fairer floor for Within-1-Star on the polar, imbalanced rating distribution.
   - `Direct TFIDF + Ridge` is added as a direct text-to-rating baseline.

5. **From-scratch model files are safer**
   - `sentiment_model/LogisticRegression.py` now has an intercept, numerical clipping, and optional `class_weight="balanced"`.
   - `sentiment_model/NaiveBayes.py` is vectorized and also includes a from-scratch `ComplementNaiveBayes` class.

## Reproduced metrics

### Sentiment OOF metrics

| Model | OOF Accuracy | OOF F1 |
|---|---:|---:|
| LR_balanced | 0.9373 | 0.8938 |
| ComplementNB | 0.9339 | 0.8863 |

### Leakage-safe rating metrics

| Model | MAE | RMSE | R² | Within-1-Star |
|---|---:|---:|---:|---:|
| Sentiment OOF + Linear | 0.5017 | 0.8718 | 0.7384 | 85.85% |
| Sentiment OOF + Ridge | 0.5019 | 0.8719 | 0.7383 | 85.85% |
| Direct TFIDF + Ridge | 0.6497 | 0.9388 | 0.6966 | 75.83% |
| Dummy Median | 1.1510 | 2.0566 | -0.4560 | 71.15% |
| Dummy Mean | 1.5162 | 1.7044 | 0.0000 | 6.07% |

## Recommended wording for report/slide

> To avoid in-sample leakage, we re-evaluated the rating model with a leakage-safe stacking setup. TF-IDF is fitted only on the training split, and training sentiment probabilities are generated using out-of-fold prediction. Under this stricter setup, the sentiment-informed model still obtains MAE ≈ 0.50 and R² ≈ 0.74 on polar reviews {1, 2, 4, 5}. Within-1-Star is 85.85%, compared with a median-constant baseline of 71.15%, so we interpret Within-1 as a useful but less decisive metric than MAE/R².

## Defense caveat

The method should be described as **sentiment-informed rating prediction on polar reviews**, not as a full 1-to-5 rating predictor. Rating 3 was removed to create the binary sentiment task, so the model does not currently cover neutral reviews.
