import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from wordcloud import WordCloud
from collections import Counter


print("\n--- VISUALIZATION ---")
def visualize_rating_distribution(df):

    plt.figure(figsize=(10,6))
    sns.countplot(data=df, x='Rating', palette='Blues')
    plt.title('Phân bố Rating của Review')
    plt.xlabel('Rating (Stars)')
    plt.ylabel('Số lượng')
    plt.show()


def visualize_review_length(df):

    plt.figure(figsize=(10,6))
    df['Review_Length'] = df['Review Text'].apply(lambda x: len(str(x).split()))
    sns.histplot(df['Review_Length'], bins=50, kde=True, color='royalblue')
    plt.title('Phân bố độ dài của Review (Số lượng từ)')
    plt.xlabel('Độ dài review')
    plt.ylabel('Tần suất')
    plt.xlim(0, df['Review_Length'].quantile(0.95)) 
    plt.show()


def visualize_average_length(df):

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Rating', y='Review_Length', estimator=np.mean, palette='Blues', errorbar=None)
    plt.title('Độ dài Review trung bình theo từng mức Rating')
    plt.xlabel('Rating (Stars)')
    plt.ylabel('Độ dài trung bình (Số lượng từ)')
    plt.show()


def visualize_heatmap(df):

    plt.figure(figsize=(8, 6))
    numeric_df = df.select_dtypes(include=[np.number])
    correlation_matrix = numeric_df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='Blues', fmt=".2f", linewidths=.5)
    plt.title('Ma trận tương quan (Heatmap) giữa các biến số')
    plt.show()


def generate_wordcloud(df):

    text_corpus = " ".join(df['Review Text'].dropna().astype(str).tolist())
    wordcloud = WordCloud(width=1000, height=500, background_color='white', colormap='Blues').generate(text_corpus)
    plt.figure(figsize=(10,6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud')
    plt.show()


def top_words_visualization(df):

    all_words = " ".join(df['cleaned_review'].dropna().astype(str).tolist()).split()
    word_counts = Counter(all_words)
    top_words_df = pd.DataFrame(word_counts.most_common(20), columns=['Word', 'Frequency'])

    plt.figure(figsize=(12, 8))
    sns.barplot(data=top_words_df, x='Frequency', y='Word', palette='Blues_r')
    plt.title('Top 20 từ xuất hiện nhiều nhất')
    plt.xlabel('Tần suất xuất hiện')
    plt.ylabel('Từ vựng')
    plt.show()

def top_words_positive_negative(df):

    ### Phan chia du lieu
    df_cleaned = df.dropna(subset=['cleaned_review'])
    df_positive = df_cleaned[df_cleaned['Rating'] >= 4]
    df_negative = df_cleaned[df_cleaned['Rating'] <= 2]

    ### Xu ly du lieu tich cuc (Positive)
    pos_corpus = " ".join(df_positive['cleaned_review']).split()
    pos_counts = Counter(pos_corpus)
    top_20_pos_dict = dict(pos_counts.most_common(20))

    ### Xu ly du lieu tieu cuc (Negative)
    neg_corpus = " ".join(df_negative['cleaned_review']).split()
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
    plt.title('Top 20 Negative Words (1–2)', fontsize=20, color='royalblue', fontweight='bold')
    plt.tight_layout() 
    plt.show()