from data_loader.load_data import clean_numeric_columns, clean_missing_duplicate, basic_info
def loader_pipeline(df):

    steps = [
        clean_numeric_columns,
        basic_info,
        clean_missing_duplicate
    ]
    for step in steps:

        df = step(df)

    return df
 