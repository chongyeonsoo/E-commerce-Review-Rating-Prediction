import re
import string

import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

stop_words = set(stopwords.words('english'))

lemmatizer = WordNetLemmatizer()

print("--- TEXT PREPROCESSING ---")
def clean_text(text):

    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove HTML
    text = re.sub(r'<.*?>', '', text)

    # Remove URL
    text = re.sub(r'http\\S+|www\\S+', '', text)

    # Remove punctuation
    text = text.translate(
        str.maketrans('', '', string.punctuation)
    )

    # Remove numbers
    text = re.sub(r'\\d+', '', text)

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords
    tokens = [
        word for word in tokens
        if word not in stop_words
    ]

    # Lemmatization
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return " ".join(tokens)


def preprocess_reviews(df):

    df['cleaned_review'] = df['Review Text'].apply(clean_text)

    return df


def create_final_dataset(df):

    print("\n--- FINAL CLEAN DATASET ---")
    clean_df = df[['Rating', 'cleaned_review', 'Review_Length']]
    print("Hiển thị 5 dòng đầu của dữ liệu: ")
    print(clean_df.head())

    return clean_df