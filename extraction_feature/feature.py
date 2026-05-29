import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def create_sentiment_label(df):
    df = df.copy()
    df = df[df['Rating'] != 3]
    df['Sentiment_Label'] = df['Rating'].apply(
        lambda x: 1 if x >= 4 else 0
    )
    return df
# TF-IDF FEATURE EXTRACTION
def extract_tfidf_features(df, text_column='Cleaned_Review Text'):

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.90
    )

    X_tfidf = vectorizer.fit_transform(df[text_column])

    print("\n--- TF-IDF FEATURE EXTRACTION ---")
    print(f"TF-IDF matrix shape: {X_tfidf.shape}")

    return X_tfidf, vectorizer


# CONVERT TF-IDF MATRIX TO DATAFRAME
def create_tfidf_dataframe(X_tfidf, vectorizer):

    feature_names = vectorizer.get_feature_names_out()

    tfidf_df = pd.DataFrame(
        X_tfidf.toarray(),
        columns=feature_names
    )

    print("TF-IDF DataFrame created!")
    print(f"TF-IDF DataFrame shape: {tfidf_df.shape}")

    return tfidf_df
