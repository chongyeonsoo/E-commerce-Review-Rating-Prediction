"""
Leakage-safe experiment for E-commerce Review Rating Prediction.

What this fixes compared with the original pipeline:
1. Split data before vectorization.
2. Fit TF-IDF on train only.
3. Generate train sentiment probabilities with out-of-fold predictions.
4. Generate test sentiment probabilities from sentiment models trained only on train.
5. Add regression baselines: DummyRegressor mean/median and direct TF-IDF -> Ridge.
6. Add ablation study for LR-only, NB-only, and LR+NB sentiment probability features.
7. Keep negation words (no/not/nor) out of the stopword list.

Run from repository root:
    python rating_prediction/fixed_experiment.py
"""

from __future__ import annotations

import os
import re
import string
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from sklearn.dummy import DummyRegressor
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import ComplementNB, MultinomialNB


RANDOM_STATE = 42
OUTPUT_DIR = "outputs_fixed"
TEXT_COL = "Review Text"
RATING_COL = "Rating"


@dataclass
class FixedExperimentResult:
    sentiment_metrics: pd.DataFrame
    rating_metrics: pd.DataFrame
    predictions: pd.DataFrame


def parse_rating(series: pd.Series) -> pd.Series:
    """Convert strings like 'Rated 5 out of 5 stars' to numeric rating."""
    return series.astype(str).str.extract(r"(\d+)", expand=False).astype(float)


def clean_text_for_tfidf(text: str) -> str:
    """Light text cleaner. Vectorizer handles tokenization and stopwords."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_and_prepare_data(csv_path: str = "Amazon_Reviews.csv", drop_neutral: bool = True) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8", engine="python")
    df = df.dropna(subset=[TEXT_COL, RATING_COL]).drop_duplicates().copy()
    df[RATING_COL] = parse_rating(df[RATING_COL])
    df = df.dropna(subset=[RATING_COL]).copy()
    if drop_neutral:
        df = df[df[RATING_COL] != 3].copy()
    df["Cleaned_Text_Fixed"] = df[TEXT_COL].apply(clean_text_for_tfidf)
    df["Sentiment_Label"] = (df[RATING_COL] >= 4).astype(int)
    df["Review_Length_Fixed"] = df["Cleaned_Text_Fixed"].str.split().str.len()
    return df.reset_index(drop=True)


def evaluate_rating(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 1, 5)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
        "Within_1_Star": float((np.abs(y_true.to_numpy() - y_pred) <= 1.0).mean()),
    }


def make_vectorizer() -> TfidfVectorizer:
    # Keep negation terms because they are crucial for sentiment.
    stop_words = sorted(set(ENGLISH_STOP_WORDS) - {"no", "nor", "not"})
    return TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.90,
        stop_words=stop_words,
    )


def make_sentiment_models() -> Dict[str, object]:
    return {
        "LR_balanced": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="liblinear",
            random_state=RANDOM_STATE,
        ),
        # ComplementNB is often more robust than MultinomialNB on imbalanced text data.
        "ComplementNB": ComplementNB(alpha=0.1),
    }


def get_positive_proba(model, X) -> np.ndarray:
    proba = model.predict_proba(X)
    classes = list(model.classes_)
    pos_idx = classes.index(1)
    return proba[:, pos_idx]


def make_oof_sentiment_features(X_train, y_train: pd.Series, X_test) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    models = make_sentiment_models()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    train_features = pd.DataFrame(index=np.arange(X_train.shape[0]))
    test_features = pd.DataFrame(index=np.arange(X_test.shape[0]))
    metric_rows = []

    for model_name, base_model in models.items():
        oof_proba = np.zeros(X_train.shape[0], dtype=float)
        oof_pred = np.zeros(X_train.shape[0], dtype=int)

        for tr_idx, val_idx in cv.split(X_train, y_train):
            model = make_sentiment_models()[model_name]
            model.fit(X_train[tr_idx], y_train.iloc[tr_idx])
            oof_proba[val_idx] = get_positive_proba(model, X_train[val_idx])
            oof_pred[val_idx] = (oof_proba[val_idx] >= 0.5).astype(int)

        final_model = make_sentiment_models()[model_name]
        final_model.fit(X_train, y_train)
        test_proba = get_positive_proba(final_model, X_test)

        col = "lr_1_oof" if model_name.startswith("LR") else "nb_1_oof"
        train_features[col] = oof_proba
        test_features[col.replace("_oof", "_test")] = test_proba

        metric_rows.append({
            "Model": model_name,
            "OOF_Accuracy": accuracy_score(y_train, oof_pred),
            "OOF_F1": f1_score(y_train, oof_pred),
        })

    # Align train/test feature names for regression.
    test_features.columns = [c.replace("_test", "_oof") for c in test_features.columns]
    return train_features, test_features, pd.DataFrame(metric_rows)


def run_fixed_experiment(csv_path: str = "Amazon_Reviews.csv", drop_neutral: bool = True) -> FixedExperimentResult:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_and_prepare_data(csv_path=csv_path, drop_neutral=drop_neutral)

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df[RATING_COL],
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    vectorizer = make_vectorizer()
    X_train_tfidf = vectorizer.fit_transform(train_df["Cleaned_Text_Fixed"])
    X_test_tfidf = vectorizer.transform(test_df["Cleaned_Text_Fixed"])

    y_sent_train = train_df["Sentiment_Label"]
    y_rating_train = train_df[RATING_COL]
    y_rating_test = test_df[RATING_COL]

    X_rating_train, X_rating_test, sentiment_metrics = make_oof_sentiment_features(
        X_train_tfidf, y_sent_train, X_test_tfidf
    )

    # Ablation study: test which sentiment probability feature actually contributes.
    # The columns below are leakage-safe because they were generated via OOF for train
    # and train-only models for test.
    feature_sets = {
        "LR prob only": ["lr_1_oof"],
        "NB prob only": ["nb_1_oof"],
        "LR + NB probs": ["lr_1_oof", "nb_1_oof"],
    }
    sentiment_rating_models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
    }

    metric_rows = []
    pred_df = test_df[[TEXT_COL, RATING_COL, "Sentiment_Label"]].copy()
    pred_df = pred_df.rename(columns={RATING_COL: "Actual_Rating"})

    # Proper trivial baselines. These do not use text or sentiment features.
    baseline_models = {
        "Dummy Mean": DummyRegressor(strategy="mean"),
        # Median is an honest floor for Within-1-Star on this polar, imbalanced dataset.
        "Dummy Median": DummyRegressor(strategy="median"),
    }
    for name, model in baseline_models.items():
        model.fit(np.zeros((len(y_rating_train), 1)), y_rating_train)
        y_pred = np.clip(model.predict(np.zeros((len(y_rating_test), 1))), 1, 5)
        metric = evaluate_rating(y_rating_test, y_pred)
        metric_rows.append({
            "Feature_Set": "Trivial baseline",
            "Model": name,
            **metric,
        })
        safe = name.lower().replace(" + ", "_").replace(" ", "_")
        pred_df[f"Predicted_{safe}"] = y_pred
        pred_df[f"Residual_{safe}"] = pred_df["Actual_Rating"] - y_pred
        pred_df[f"Within1_{safe}"] = pred_df[f"Residual_{safe}"].abs() <= 1

    # Direct text baseline: text -> rating without the sentiment-probability layer.
    direct_model = Ridge(alpha=1.0)
    direct_model.fit(X_train_tfidf, y_rating_train)
    y_pred = np.clip(direct_model.predict(X_test_tfidf), 1, 5)
    metric = evaluate_rating(y_rating_test, y_pred)
    metric_rows.append({
        "Feature_Set": "Direct TF-IDF",
        "Model": "Direct TFIDF + Ridge",
        **metric,
    })
    pred_df["Predicted_direct_tfidf_ridge"] = y_pred
    pred_df["Residual_direct_tfidf_ridge"] = pred_df["Actual_Rating"] - y_pred
    pred_df["Within1_direct_tfidf_ridge"] = pred_df["Residual_direct_tfidf_ridge"].abs() <= 1

    # Sentiment-probability ablations: LR only vs NB only vs both features.
    for feature_name, cols in feature_sets.items():
        X_train_subset = X_rating_train[cols]
        X_test_subset = X_rating_test[cols]
        for model_name, model in sentiment_rating_models.items():
            model.fit(X_train_subset, y_rating_train)
            y_pred = np.clip(model.predict(X_test_subset), 1, 5)
            metric = evaluate_rating(y_rating_test, y_pred)
            metric_rows.append({
                "Feature_Set": feature_name,
                "Model": model_name,
                **metric,
            })
            safe_feature = feature_name.lower().replace(" + ", "_").replace(" ", "_").replace("-", "_")
            safe_model = model_name.lower().replace(" ", "_")
            safe = f"{safe_feature}_{safe_model}"
            pred_df[f"Predicted_{safe}"] = y_pred
            pred_df[f"Residual_{safe}"] = pred_df["Actual_Rating"] - y_pred
            pred_df[f"Within1_{safe}"] = pred_df[f"Residual_{safe}"].abs() <= 1

    rating_metrics = pd.DataFrame(metric_rows).sort_values(["MAE", "RMSE"]).reset_index(drop=True)

    sentiment_metrics.to_csv(os.path.join(OUTPUT_DIR, "sentiment_oof_metrics.csv"), index=False)
    rating_metrics.to_csv(os.path.join(OUTPUT_DIR, "rating_fixed_metrics.csv"), index=False)
    pred_df.to_csv(os.path.join(OUTPUT_DIR, "rating_fixed_predictions.csv"), index=False)

    print("\n--- SENTIMENT OOF METRICS ---")
    print(sentiment_metrics.to_string(index=False))
    print("\n--- LEAKAGE-SAFE RATING METRICS ---")
    print(rating_metrics.to_string(index=False))
    print(f"\nSaved outputs to: {OUTPUT_DIR}/")

    return FixedExperimentResult(sentiment_metrics, rating_metrics, pred_df)


if __name__ == "__main__":
    run_fixed_experiment()
