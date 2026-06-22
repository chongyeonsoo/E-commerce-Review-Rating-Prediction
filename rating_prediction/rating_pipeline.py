from rating_prediction.rating import run_rating_prediction, error_analysis_by_rating


def rating_prediction_pipeline(df):
    result = run_rating_prediction(
        df=df,
        target_col='Rating',
        test_size=0.2,
        random_state=42,
        output_dir='outputs'
    )
    error_analysis_by_rating(
        pred_df=result['predictions'],
        best_combination=result['best_combination'],
        output_dir='outputs'
    )

    return result
