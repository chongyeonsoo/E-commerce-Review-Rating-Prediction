from sentiment_model.LogisticRegression import Logisticregression
from sentiment_model.NaiveBayes import MultinomialNaiveBayes
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import pandas as pd


def model_pipeline(X, y):
    result = {}
    rows = []

    print("TRAIN MODEL WITH OUT-OF-FOLD")

    model_general = {
        "LOGISTIC REGRESSION": Logisticregression(lr=0.1, max_iter=1000, class_weight= "balanced"),
        "NAIVE BAYES":         MultinomialNaiveBayes(alpha=0.1),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in model_general.items():
        print(f"\nMODEL {name}")

        oof_proba = np.zeros(X.shape[0])
        oof_pred  = np.zeros(X.shape[0], dtype=int)

        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            print(f"  Fold {fold + 1}/5")

            if name == "LOGISTIC REGRESSION":
                m = Logisticregression(lr=0.01, max_iter=1000, class_weight= "balanced")
            else:
                m = MultinomialNaiveBayes(alpha=0.001)

            m.fit(X[train_idx], y.iloc[train_idx])
            proba = m.predict_proba(X[val_idx])
            if proba.ndim == 2:
                proba = proba[:, 1]
            else:
                proba = proba
            oof_proba[val_idx] = proba
            oof_pred[val_idx]  = (proba>= 0.5).astype(int)

        # Train lại trên toàn bộ data để dùng cho production
        model.fit(X, y)

        acc = accuracy_score(y, oof_pred)
        f1  = f1_score(y, oof_pred)

        result[name] = {
            "model":       model,
            "accuracy":    acc,
            "f1":          f1,
            "probability": oof_proba,
        }

        rows.append({
            "Model":      name,
            "Accuracy":   round(acc, 4),
            "F1":         round(f1, 4),
            "Prob Shape": oof_proba.shape,
        })

    df_model_sentiment = pd.DataFrame(rows)
    return result, df_model_sentiment