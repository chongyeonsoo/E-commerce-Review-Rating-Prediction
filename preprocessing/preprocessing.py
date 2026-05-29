import re
import string
import pandas as pd
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
# DOWNLOAD NLTK DATA
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def clean_text(df, column_name):

    print(f"\n--- PREPROCESSING COLUMN: {column_name} ---")

    ## Create coloum name
    cleaned_column = f"Cleaned_{column_name}"
    token_column = f"Token_{column_name}"

    ## Text cleaning function
    def preprocess_single_text(text):

        ### Handle non-string values
        if not isinstance(text, str):
            return "", []
        
        ### Lowercase
        text = text.lower()

        ### Remove HTML tags
        text = re.sub(r'<.*?>', '', text)

        ### Remove URLS
        text = re.sub(r'http\S+|www\S+', '', text)

        ### REemove numbers
        text = re.sub(r'\d+', '', text)

        ### Remove punctuation
        text = text.translate(
            str.maketrans('', '', string.punctuation)
        )

        ### Remove special characters
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        ### Tokenization
        tokens = word_tokenize(text)

        ### Remove stopwords
        tokens = [
            word for word in tokens
            if word not in stop_words
        ]

        ### Remove emty tokens
        tokens = [
            word for word in tokens
            if word.strip() != ''
        ]

        ### Lemmatization
        tokens = [
            lemmatizer.lemmatize(word, pos='v')
            for word in tokens
        ]

        ### Join tokens
        cleaned_text = " ".join(tokens)

        return cleaned_text, tokens

    results = df[column_name].apply(preprocess_single_text)
    df[cleaned_column] = results.apply(lambda x: x[0])
    df[token_column] = results.apply(lambda x: x[1])

    print(f"Created column: {cleaned_column}")
    print(f"Created column: {token_column}")

    return df


def create_final_dataset(df):

    # Chọn các cột cần thiết
    df['Review Length'] = df['Token_Review Text'].apply(len)

    return df
