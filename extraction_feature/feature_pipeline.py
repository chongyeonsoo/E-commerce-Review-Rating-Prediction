from extraction_feature.feature import create_sentiment_label, extract_tfidf_features, create_tfidf_dataframe
def extraction_pipeline(df):
    df = create_sentiment_label(df.copy())
    X_tfidf, vectorizer = extract_tfidf_features(df)
    y_sentiment = df['Sentiment_Label']
    return df, X_tfidf, y_sentiment