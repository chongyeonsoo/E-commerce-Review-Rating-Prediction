from data_loader.load_data import load_dataset
from data_loader.loader_pipeline import loader_pipeline
from preprocessing.preprocessing_pipeline import preprocessing_pipeline
from extraction_feature.feature_pipeline import extraction_pipeline
from sentiment_model.model_pipeline import model_pipeline
from rating_prediction.rating_pipeline import rating_prediction_pipeline
from sklearn.metrics import accuracy_score
import pandas as pd
def main():
    data = load_dataset('Amazon_Reviews.csv')
    df = loader_pipeline(data.copy())
    data = preprocessing_pipeline(df.copy())
    df, X_tfidf, y_sentiment = extraction_pipeline(data.copy())
    print(df.head(10))
    result, df_model_sentiment = model_pipeline(X_tfidf, y_sentiment)
    print(df_model_sentiment.head(10))
    lr_pro = result['LOGISTIC REGRESSION']['probability']
    df['lr_pro'] = lr_pro.flatten()
    df['lr_0'] = 1 - df['lr_pro']
    df['lr_1'] = df['lr_pro']
    nb_pro = result['NAIVE BAYES']['probability']
    nb_pro_0 = nb_pro[:, 0]
    nb_pro_1 = nb_pro[:, 1]
    df['nb_0'] = nb_pro_0.flatten()
    df['nb_1'] = nb_pro_1.flatten()
    print(df.head(10))

    # RATING PREDICTION + VISUALIZE + METRICS
    rating_result = rating_prediction_pipeline(df)
    print(rating_result['metrics'])

if __name__ == "__main__":
    main()