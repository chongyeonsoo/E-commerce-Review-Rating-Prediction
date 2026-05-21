# ============================================================
# PROJECT   : E-commerce Review Rating Prediction
# DATASET   : Amazon_Reviews.csv
# AUTHOR    : Nhóm 5
# ============================================================

# Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string

from bs4 import BeautifulSoup
from collections import Counter
from wordcloud import WordCloud

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load Dataset
print("--- LOAD DATASET ---")

## Doc file dataset
try:
    df = pd.read_csv('Amazon_Reviews.csv', encoding='utf-8', engine='python')
    print("Tải dữ liệu thành công!")
except:
    df = pd.read_csv('Amazon_Reviews.csv', encoding='latin-1', engine='python')
    print("Tải dữ liệu bằng encoding latin1 thành công!")

# Tien xu ly nhanh cot Rating (Chuyen "Rated 5 out of 5 stars" thanh so 5)
df['Rating'] = df['Rating'].astype(str).str.extract(r'(\d+)', expand=False).astype(float)
# Tien xu ly nhanh cot Review Count (Chuyen "82 reviews" thanh so 82)
df['Review Count'] = df['Review Count'].astype(str).str.extract(r'(\d+)', expand=False).astype(float)
## In du lieu
print("5 dòng đầu tiên của dữ liệu: ")
print(df.head())
## Kich thuoc du lieu
print(f"Kích thước tập dữ liệu (Shape): {df.shape}")
## Thong tin du lieu
print("\nThông tin dữ liệu (Info):")
df.info()
## Thong ke du lieu
print("\nThống kê mô tả (Describe):")
print(df.describe())
## Kiem tra du lieu bi thieu
print("\nSố lượng Missing Values:")
print(df.isnull().sum())
## Kiem tra du lieu trung lap
print(f"\nSố lượng Duplicate Values: {df.duplicated().sum()}")
# Xu ly Missing và Duplicates 
df = df.dropna(subset=['Review Text', 'Rating']) 
df = df.drop_duplicates()
print(f"Kích thước sau khi xóa NaN và Duplicates: {df.shape}")

# Visualization
print("\n--- VISUALIZATION ---")
df['Review_Length'] = df['Review Text'].apply(lambda x: len(str(x).split()))

## Countplot Rating 
plt.figure(figsize=(10,6))
sns.countplot(data=df, x='Rating', palette='Blues')
plt.title('Phân bố Rating của Review')
plt.xlabel('Rating (Stars)')
plt.ylabel('Số lượng')
plt.show()

## Histogram do dai review
plt.figure(figsize=(10,6))
sns.histplot(df['Review_Length'], bins=50, kde=True, color='royalblue')
plt.title('Phân bố độ dài của Review (Số lượng từ)')
plt.xlabel('Độ dài review')
plt.ylabel('Tần suất')
plt.xlim(0, df['Review_Length'].quantile(0.95)) 
plt.show()

## Do dai review trung binh theo rating
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='Rating', y='Review_Length', estimator=np.mean, palette='Blues', errorbar=None)
plt.title('Độ dài Review trung bình theo từng mức Rating')
plt.xlabel('Rating (Stars)')
plt.ylabel('Độ dài trung bình (Số lượng từ)')
plt.show()

## Heatmap tương quan giua cac bien so
plt.figure(figsize=(8, 6))
numeric_df = df.select_dtypes(include=[np.number])
correlation_matrix = numeric_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='Blues', fmt=".2f", linewidths=.5)
plt.title('Ma trận tương quan (Heatmap) giữa các biến số')
plt.show()

## WordCloud
text_corpus = " ".join(df['Review Text'].dropna().astype(str).tolist())
wordcloud = WordCloud(width=1000, height=500, background_color='white', colormap='Blues').generate(text_corpus)
plt.figure(figsize=(10,6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud')
plt.show()


# Text Preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    ## Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    ## Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    ## Remove punctuation & special characters
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    ## Remove numbers
    text = re.sub(r'\d+', '', text)
    
    ## Tokenize
    tokens = word_tokenize(text)
    
    ## Remove stopwords và Lemmatization
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    return " ".join(cleaned_tokens)

## Ap dung ham lam sach 
print("--- Text Preprocessing ---")
df['cleaned_review'] = df['Review Text'].apply(clean_text)
print("Hoàn tất Text Preprocessing!")

## Top 20 tu xuat hien nhieu nhat (sau Preprocessing)
from collections import Counter
all_words = " ".join(df['cleaned_review'].dropna().astype(str).tolist()).split()
word_counts = Counter(all_words)
top_words_df = pd.DataFrame(word_counts.most_common(20), columns=['Word', 'Frequency'])

plt.figure(figsize=(12, 8))
sns.barplot(data=top_words_df, x='Frequency', y='Word', palette='Blues_r')
plt.title('Top 20 từ xuất hiện nhiều nhất')
plt.xlabel('Tần suất xuất hiện')
plt.ylabel('Từ vựng')
plt.show()

## WordCloud Positive vs Negative (Top 20)
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


print("\n--- FINAL CLEAN DATASET ---")
clean_df = df[['Rating', 'cleaned_review', 'Review_Length']]
print("Hiển thị 5 dòng đầu của dữ liệu sạch:")
print(clean_df.head())







































































