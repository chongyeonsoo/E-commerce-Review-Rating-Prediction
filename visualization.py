import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from wordcloud import WordCloud
from collections import Counter

# RATING DISTRIBUTION
def visualize_rating_distribution(df):

    plt.figure(figsize=(10,6))
    sns.countplot(data=df, x='Rating',palette='Blues')
    plt.title('Distribution of Ratings')
    plt.xlabel('Rating (Stars)')
    plt.ylabel('Count')

    plt.show()

# REVIEW LENGTH DISTRIBUTION
def visualize_review_length(df):

    plt.figure(figsize=(10,6))
    sns.histplot(df['Review_Length'], bins=50, kde=True, color='royalblue')
    plt.title('Review Length Distribution')
    plt.xlabel('Review Length')
    plt.ylabel('Frequency')
    plt.xlim(0, df['Review_Length'].quantile(0.95))

    plt.show()

# AVERAGE REVIEW LENGTH BY RATING
def visualize_average_length(df):

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Rating', y='Review_Length', estimator=np.mean, palette='Blues', errorbar=None)
    plt.title('Average Review Length by Rating')
    plt.xlabel('Rating (Stars)')
    plt.ylabel('Average Review Length')

    plt.show()



def visualize_heatmap(df):

    plt.figure(figsize=(8, 6))
    numeric_df = df.select_dtypes(include=[np.number])
    correlation_matrix = numeric_df.corr()
    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap='Blues',
        fmt=".2f",
        linewidths=.5
    )
    plt.title('Correlation Heatmap')

    plt.show()


def generate_wordcloud(df):

    text_corpus = " ".join(df['Review Text'].dropna().astype(str).tolist())
    wordcloud = WordCloud(width=1000, height=500, background_color='white', colormap='Blues').generate(text_corpus)
    plt.figure(figsize=(10,6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud')
    plt.show()

# TOP 20 FREQUENT WORDS
def top_words_visualization(df):

    all_words = " ".join(
        df['Cleaned_Review Text']
        .dropna()
        .astype(str)
        .tolist()
    ).split()

    word_counts = Counter(all_words)

    top_words_df = pd.DataFrame(
        word_counts.most_common(20),
        columns=['Word', 'Frequency']
    )

    plt.figure(figsize=(12, 8))

    sns.barplot(
        data=top_words_df,
        x='Frequency',
        y='Word',
        palette='Blues_r'
    )

    plt.title('Top 20 Most Frequent Words')
    plt.xlabel('Frequency')
    plt.ylabel('Words')

    plt.show()

# POSITIVE VS NEGATIVE WORDCLOUD

def top_words_positive_negative(df):

    ### Phan chia du lieu
    df_cleaned = df.dropna(subset=['Cleaned_Review Text'])
    df_positive = df_cleaned[df_cleaned['Rating'] >= 4]
    df_negative = df_cleaned[df_cleaned['Rating'] <= 2]

    ### Xu ly du lieu tich cuc (Positive)
    pos_corpus = " ".join(df_positive['Cleaned_Review Text']).split()
    pos_counts = Counter(pos_corpus)
    top_20_pos_dict = dict(pos_counts.most_common(20))

    ### Xu ly du lieu tieu cuc (Negative)
    neg_corpus = " ".join(df_negative['Cleaned_Review Text']).split()
    neg_counts = Counter(neg_corpus)
    top_20_neg_dict = dict(neg_counts.most_common(20))

    ### Tao WordCloud tich cuc
    wc_pos = WordCloud(
        width=800, height=600, 
        background_color='white', 
        colormap='Blues' 
    ).generate_from_frequencies(top_20_pos_dict) 

    ### Tao WordCloud tieu cuc
    wc_neg = WordCloud(
        width=800, height=600, 
        background_color='white', 
        colormap='Blues'
    ).generate_from_frequencies(top_20_neg_dict)

    plt.figure(figsize=(20, 10))
    ### WordCloud tich cuc
    plt.subplot(1, 2, 1) 
    plt.imshow(wc_pos, interpolation='bilinear')
    plt.axis('off') 
    plt.title('Top 20 Positive Words (4–5)', fontsize=20, color='royalblue', fontweight='bold')
    ### WordCloud tieu cuc
    plt.subplot(1, 2, 2)
    plt.imshow(wc_neg, interpolation='bilinear')
    plt.axis('off')
    plt.title('Top 20 Negative Words (1–2)', fontsize=20, color='darkred', fontweight='bold')
    plt.tight_layout() 
    plt.show()

