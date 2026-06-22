import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def prepare_rating_data(df, target_col='Rating'):
    data = df.copy()

    lr_col = 'lr_1' if 'lr_1' in data.columns else ('lr_pro' if 'lr_pro' in data.columns else None)
    nb_col = 'nb_1' if 'nb_1' in data.columns else None

    if lr_col is None and nb_col is None:
        raise ValueError("Không tìm thấy cột sentiment nào (lr_1, lr_pro, nb_1)")

    feature_sets = {}
    if lr_col:
        feature_sets['LR'] = [lr_col]
    if nb_col:
        feature_sets['NB'] = [nb_col]

    data[target_col] = pd.to_numeric(data[target_col], errors='coerce')
    all_feature_cols = list({col for cols in feature_sets.values() for col in cols})
    for col in all_feature_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    data = data.dropna(subset=all_feature_cols + [target_col]).reset_index(drop=True)

    print(f"\n--- RATING PREDICTION DATA ---")
    print(f"Dataset shape: {data.shape}")
    print(f"Feature sets: {list(feature_sets.keys())}")

    return data, feature_sets, target_col


def evaluate_rating(y_true, y_pred):
    y_pred = np.clip(y_pred, 1, 5)
    y_true = np.array(y_true)
    return {
        'MAE':          round(float(mean_absolute_error(y_true, y_pred)), 4),
        'RMSE':         round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        'R2':           round(float(r2_score(y_true, y_pred)), 4),
        'Within_1_Star': f"{float((np.abs(y_true - y_pred) <= 1.0).mean()) * 100:.2f}%",
    }


def run_rating_prediction(
    df,
    target_col='Rating',
    test_size=0.2,
    random_state=42,
    output_dir='outputs'
):
    os.makedirs(output_dir, exist_ok=True)

    data, feature_sets, target_col = prepare_rating_data(df, target_col)

    y = data[target_col]

    train_idx, test_idx = train_test_split(
        data.index,
        test_size=test_size,
        random_state=random_state,
        stratify=y if y.nunique() > 1 else None
    )

    y_train = y.loc[train_idx]
    y_test  = y.loc[test_idx]

    rating_models = {
        'Linear Regression': LinearRegression(),
        'SVR':               SVR(kernel='rbf', C=1.0, epsilon=0.1),
    }

    metric_rows = []
    pred_df = data.loc[test_idx].copy()
    pred_df['Actual_Rating'] = y_test.values

    for feature_name, cols in feature_sets.items():
        X_train = data.loc[train_idx, cols]
        X_test  = data.loc[test_idx,  cols]

        for model_name, model in rating_models.items():
            model.fit(X_train, y_train)
            y_pred = np.clip(model.predict(X_test), 1, 5)

            metrics = evaluate_rating(y_test, y_pred)
            combination = f"{feature_name} + {model_name}"

            metric_rows.append({
                'Combination':   combination,
                'MAE':           metrics['MAE'],
                'RMSE':          metrics['RMSE'],
                'R2':            metrics['R2'],
                'Within_1_Star': metrics['Within_1_Star'],
            })

            safe = combination.lower().replace(' + ', '_').replace(' ', '_')
            pred_df[f'Predicted_{safe}'] = y_pred
            pred_df[f'Residual_{safe}']  = pred_df['Actual_Rating'] - y_pred
            pred_df[f'Within1_{safe}']   = pred_df[f'Residual_{safe}'].abs() <= 1.0

    metrics_df = pd.DataFrame(metric_rows).sort_values('MAE').reset_index(drop=True)

    print("\n--- RATING PREDICTION METRICS ---")
    print(metrics_df.to_string(index=False))

    metrics_path = os.path.join(output_dir, 'rating_prediction_metrics.csv')
    pred_path    = os.path.join(output_dir, 'rating_prediction_predictions.csv')
    metrics_df.to_csv(metrics_path, index=False)
    pred_df.to_csv(pred_path, index=False)

    _visualize_metrics(metrics_df, output_dir)

    best_row = metrics_df.iloc[0]
    print(f"\nBest: {best_row['Combination']}"
          f" | MAE={best_row['MAE']:.4f}"
          f", RMSE={best_row['RMSE']:.4f}"
          f", R2={best_row['R2']:.4f}"
          f", Within-1-Star={best_row['Within_1_Star']}")

    return {
        'metrics':          metrics_df,
        'predictions':      pred_df,
        'best_combination': best_row['Combination'],
        'metrics_path':     metrics_path,
        'predictions_path': pred_path,
    }


def _visualize_metrics(metrics_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for metric in ['MAE', 'RMSE']:
        plt.figure(figsize=(10, 5))
        plt.bar(metrics_df['Combination'], metrics_df[metric])
        plt.title(f'Rating Prediction - {metric} by Combination')
        plt.xlabel('Sentiment Model + Rating Model')
        plt.ylabel(metric)
        plt.xticks(rotation=15, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'rating_{metric.lower()}_comparison.png'), dpi=300)
        plt.close()
def error_analysis_by_rating(pred_df, best_combination, output_dir='outputs'):
    os.makedirs(output_dir, exist_ok=True)

    safe = best_combination.lower().replace(' + ', '_').replace(' ', '_')
    pred_col     = f'Predicted_{safe}'
    residual_col = f'Residual_{safe}'

    if pred_col not in pred_df.columns:
        print(f"[WARNING] Không tìm thấy cột: {pred_col}")
        return None, None

    tmp = pred_df.copy()
    tmp['Abs_Error'] = tmp[residual_col].abs()

    summary = (
        tmp.groupby('Actual_Rating')
        .agg(
            Count=('Abs_Error', 'size'),
            MAE=('Abs_Error', 'mean'),
            Mean_Predicted=(pred_col, 'mean')
        )
        .reset_index()
    )
    summary['MAE']            = summary['MAE'].round(4)
    summary['Mean_Predicted'] = summary['Mean_Predicted'].round(4)

    print("\n--- ERROR ANALYSIS BY RATING ---")
    print(summary.to_string(index=False))

    csv_path = os.path.join(output_dir, 'error_analysis_by_rating.csv')
    summary.to_csv(csv_path, index=False)

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(summary['Actual_Rating'], summary['MAE'])
    axes[0].set_title(f'MAE by Rating Level - {best_combination}')
    axes[0].set_xlabel('Actual Rating')
    axes[0].set_ylabel('MAE')
    axes[0].set_xticks([1, 2, 3, 4, 5])
    for i, val in enumerate(summary['MAE']):
        axes[0].text(summary['Actual_Rating'].iloc[i], val + 0.01,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    axes[1].bar(summary['Actual_Rating'], summary['Count'])
    axes[1].set_title('Sample Count by Rating Level')
    axes[1].set_xlabel('Actual Rating')
    axes[1].set_ylabel('Count')
    axes[1].set_xticks([1, 2, 3, 4, 5])
    for i, val in enumerate(summary['Count']):
        axes[1].text(summary['Actual_Rating'].iloc[i], val,
                     str(val), ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'error_analysis_by_rating.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()

    return fig_path, summary