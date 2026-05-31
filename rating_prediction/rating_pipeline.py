from rating_prediction.rating import run_rating_prediction


def rating_prediction_pipeline(df):
    """
    Pipeline cho task: Rating prediction + visualize + metrics.
    Input cần là dataframe đã có sentiment probability từ 2 model:
        - lr_pro hoặc lr_1
        - nb_1
    """
    result = run_rating_prediction(
        df=df,
        feature_cols=None,
        target_col='Rating',
        test_size=0.2,
        random_state=42,
        output_dir='outputs'
    )
    return result
