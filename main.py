from load_data import *
from preprocessing import *
from visualization import *

def main():

    # LOAD DATASET
    print("\n--- LOAD DATASET ---")
    df = load_dataset("D:\\NLP\\Amazon_Reviews.csv")

    # CLEAN NUMERIC COLUMNS
    print("\n--- CLEAN NUMERIC COLUMNS ---")
    df = clean_numeric_columns(df)

    # BASIC DATASET INFO
    print("\n--- DATASET INFO BEFORE CLEANING ---")
    basic_info(df)

    # REMOVE MISSING VALUES AND DUPLICATES
    print("\n--- REMOVE MISSING VALUES & DUPLICATES ---")
    df = clean_missing_duplicate(df)

    # CREATE REVIEW LENGTH COLUMN
    print("\n--- CREATE REVIEW LENGTH ---")
    df['Review_Length'] = df['Review Text'].apply(
        lambda x: len(str(x).split())
    )
    print("Created column: Review_Length")

    # BASIC VISUALIZATION
    print("\n--- BASIC VISUALIZATION ---")

    visualize_rating_distribution(df)

    visualize_review_length(df)

    visualize_average_length(df)

    visualize_heatmap(df)

    generate_wordcloud(df)

    # TEXT PREPROCESSING
    print("\n--- TEXT PREPROCESSING ---")
    df = clean_text(df, 'Review Text')

    # TEXT VISUALIZATION AFTER CLEANING
    print("\n--- TEXT VISUALIZATION ---")

    top_words_visualization(df)

    top_words_positive_negative(df)

    # FINAL CLEAN DATASET
    print("\n--- FINAL CLEAN DATASET ---")
    clean_df = create_final_dataset(df)
    print(clean_df.head())


if __name__ == "__main__":
    main()