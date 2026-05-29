from preprocessing.preprocessing import clean_text,  create_final_dataset
def preprocessing_pipeline(df):
    df = clean_text(df, 'Review Text')
    steps = [create_final_dataset]
    for step in steps:
        df = step(df)
    return df