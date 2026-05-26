import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack, csr_matrix
from gensim.models import Word2Vec

def create_sentiment_label(df):

    df = df.copy()

    df = df[df['Rating'] != 3]

    df['Sentiment'] = df['Rating'].apply(
        lambda x: 1 if x >= 4 else 0
    )

    print("Created column: Sentiment")
    print(df['Sentiment'].value_counts())

    return df


def extract_tfidf_features(df, text_column='Cleaned_Review Text'):

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.90
    )

    X_tfidf = vectorizer.fit_transform(df[text_column])

    print("TF-IDF feature extraction completed!")
    print(f"TF-IDF feature shape: {X_tfidf.shape}")

    return X_tfidf, vectorizer


def train_word2vec_model(
    df,
    token_column='Token_Review Text',
    vector_size=100,
    window=5,
    min_count=2,
    workers=4,
    sg=1
):

    sentences = df[token_column].tolist()

    word2vec_model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=sg
    )

    print("Word2Vec training completed!")
    print(f"Word2Vec vector size: {vector_size}")

    return word2vec_model



def average_word2vec(tokens, model, vector_size):


    valid_vectors = []

    for word in tokens:
        if word in model.wv:
            valid_vectors.append(model.wv[word])

    if len(valid_vectors) == 0:
        return np.zeros(vector_size)

    return np.mean(valid_vectors, axis=0)


def extract_word2vec_features(
    df,
    word2vec_model,
    token_column='Token_Review Text',
    vector_size=100
):

    X_word2vec = np.array([
        average_word2vec(tokens, word2vec_model, vector_size)
        for tokens in df[token_column]
    ])

    print("Word2Vec feature extraction completed!")
    print(f"Word2Vec feature shape: {X_word2vec.shape}")

    return X_word2vec


def combine_all_features(X_tfidf, X_word2vec, df):


    X_numeric = df[['Review_Length']].values

    X_word2vec_sparse = csr_matrix(X_word2vec)
    X_numeric_sparse = csr_matrix(X_numeric)

    X_combined = hstack([
        X_tfidf,
        X_word2vec_sparse,
        X_numeric_sparse
    ])

    print("Combined TF-IDF + Word2Vec + numerical features completed!")
    print(f"Combined feature shape: {X_combined.shape}")

    return X_combined


def split_dataset(X, y, test_size=0.2, random_state=42):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print("Train-test split completed!")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    return X_train, X_test, y_train, y_test


def feature_extraction_pipeline(df):

    print("\n--- FEATURE EXTRACTION ---")
    df = create_sentiment_label(df)

    X_tfidf, tfidf_vectorizer = extract_tfidf_features(
        df,
        text_column='Cleaned_Review Text'
    )

    word2vec_model = train_word2vec_model(
        df,
        token_column='Token_Review Text',
        vector_size=100,
        window=5,
        min_count=2,
        workers=4,
        sg=1
    )

    X_word2vec = extract_word2vec_features(
        df,
        word2vec_model,
        token_column='Token_Review Text',
        vector_size=100
    )


    X = combine_all_features(X_tfidf, X_word2vec, df)
    y = df['Sentiment']

    X_train, X_test, y_train, y_test = split_dataset(X, y)

    return X_train, X_test, y_train, y_test, tfidf_vectorizer, word2vec_model, df