from sentiment_model.LogisticRegression import Logisticregression
from sentiment_model.NaiveBayes import MultinomialNaiveBayes
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
def model_pipeline(X, y):
    result= {}
    print("SPLIT TRAIN TEST AND TRAIN MODEL")
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size= 0.8, random_state= 42)
    model1 = Logisticregression(lr = 0.01, max_iter= 1000)
    model2 = MultinomialNaiveBayes(alpha = 0.001)
    model_general = {
        "LOGISTIC REGRESSION": model1,
        "NAIVE BAYES": model2
    }
    for name, model in model_general.items():
        print(f"MODEL {name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob_test = model.predict_proba(X)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        result[name] = {
            "model": model,
            "accuracy": acc,
            "f1": f1,
            "probability": y_prob_test
        } 
        rows = []

    for model_name, info in result.items():
        rows.append({
            "Model": model_name,
            "Accuracy": round(info["accuracy"], 4),
            "F1": round(info["f1"], 4),
            "Prob Shape": info["probability"].shape
    })

    df_model_sentiment = pd.DataFrame(rows)
    return result, df_model_sentiment