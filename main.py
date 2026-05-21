from load_data import *
from preprocessing import *
from visualization import *


def main():

    # LOAD DATASET
    df = load_dataset("Amazon_Reviews.csv")

    # CLEAN NUMERIC
    df = clean_numeric_columns(df)

    # BASIC INFO
    basic_info(df)

    # REMOVE NaN + DUPLICATE
    df = clean_missing_duplicate(df)

    # VISUALIZATION
    visualize_rating_distribution(df)

    visualize_review_length(df)

    visualize_average_length(df)

    visualize_heatmap(df)

    generate_wordcloud(df)

    # PREPROCESSING
    df = preprocess_reviews(df)

    # TOP WORDS
    top_words_visualization(df)
    top_words_positive_negative(df)

    # FINAL DATASET
    clean_df = create_final_dataset(df)

    print(clean_df.head())

    

if __name__ == "__main__":
    main()