from load_data import *
from preprocessing import *
from visualization import *


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

    df['Review_Length'] = df['Review Text'].apply(lambda x: len(str(x).split()))
    print("Created column: Review_Length")

    # VISUALIZATION
    print("\n--- VISUALIZATION ---")

    ## Rating distribution
    visualize_rating_distribution(df)

    ## Review length distribution
    visualize_review_length(df)

    ## Average review length by rating
    visualize_average_length(df)

    ## Heatmap
    visualize_heatmap(df)

    ## WordCloud
    generate_wordcloud(df)

    # PREPROCESSING
    print("TEXT PREPROCESSING")
    df = clean_text(df, 'Review Text')

    ## POSITIVE VS NEGATIVE WORDCLOUD
    print("TOP WORDS VISUALIZATION")
    top_words_visualization(df)
    ## WordCloud Positive vs Negative (Top 20)
    top_words_positive_negative(df)

    # FINAL DATASET
    print("\n--- FINAL CLEAN DATASET ---")
    clean_df = create_final_dataset(df)
    print(clean_df.head())

    

if __name__ == "__main__":
    main()





