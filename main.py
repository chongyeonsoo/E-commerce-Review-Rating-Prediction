from load_data import *
from preprocessing import *
from visualization import *
from feature_extraction import *


def main():

    # LOAD DATASET
    print("--- LOAD DATASET ---")
    df = load_dataset("Amazon_Reviews.csv")

    # CLEAN NUMERIC
    print("CLEAN NUMERIC COLUMNS")
    df = clean_numeric_columns(df)

    # BASIC INFO
    print("\n--- DATASET INFO ---")
    basic_info(df)

    # REMOVE NaN + DUPLICATE
    print("REMOVE MISSING VALUES & DUPLICATES")
    df = clean_missing_duplicate(df)

    # CREATE REVIEW LENGTH
    df['Review_Length'] = df['Review Text'].apply(lambda x: len(str(x).split()))
    print("Created column: Review_Length")

    # VISUALIZATION
    print("\n--- VISUALIZATION ---")
    visualize_rating_distribution(df)
    visualize_review_length(df)
    visualize_average_length(df)
    visualize_heatmap(df)
    generate_wordcloud(df)

    # TEXT PREPROCESSING
    print("TEXT PREPROCESSING")
    df = clean_text(df, 'Review Text')

    # TOP WORDS VISUALIZATION
    print("TOP WORDS VISUALIZATION")
    top_words_visualization(df)
    top_words_positive_negative(df)

    # FINAL CLEAN DATASET
    print("\n--- FINAL CLEAN DATASET ---")
    clean_df = create_final_dataset(df)
    print(clean_df.head())

    # FEATURE EXTRACTION
    X_train, X_test, y_train, y_test, tfidf_vectorizer, word2vec_model, model_df = feature_extraction_pipeline(df)


if __name__ == "__main__":
    main()





