import pandas as pd

# Load Dataset
print("--- LOAD DATASET ---")
def load_dataset(file_path):

    try:
        df = pd.read_csv('Amazon_Reviews.csv', encoding='utf-8', engine='python')
        print("Tải dữ liệu thành công!")
    except:
        df = pd.read_csv('Amazon_Reviews.csv', encoding='latin-1', engine='python')
        print("Tải dữ liệu bằng encoding latin1 thành công!")

    return df

def clean_numeric_columns(df):

    # Tien xu ly nhanh cot Rating (Chuyen "Rated 5 out of 5 stars" thanh so 5)
    df['Rating'] = df['Rating'].astype(str).str.extract(r'(\d+)', expand=False).astype(float)
    # Tien xu ly nhanh cot Review Count (Chuyen "82 reviews" thanh so 82)
    df['Review Count'] = df['Review Count'].astype(str).str.extract(r'(\d+)', expand=False).astype(float)

    return df


def basic_info(df):

    print("\n--- DATASET INFO ---")

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


def clean_missing_duplicate(df):

    df = df.dropna(subset=['Review Text', 'Rating']) 
    df = df.drop_duplicates()
    print(f"Kích thước sau khi xóa NaN và Duplicates: {df.shape}")

    return df