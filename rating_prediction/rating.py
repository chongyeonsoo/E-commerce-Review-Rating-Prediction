import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def prepare_rating_data(df, feature_cols=None, target_col='Rating'):
    """
    Chuẩn bị dữ liệu cho Rating Prediction.
    Theo note của leader: dùng xác suất sentiment của 2 model đã lưu trong dataframe.
    Mặc định dùng xác suất positive của Logistic Regression và Naive Bayes:
        - lr_pro hoặc lr_1
        - nb_1
    """
    data = df.copy()

    # Nếu main.py chỉ lưu lr_pro thì dùng lr_pro; nếu có lr_1 thì ưu tiên lr_1.
    if feature_cols is None:
        if 'lr_1' in data.columns:
            lr_col = 'lr_1'
        elif 'lr_pro' in data.columns:
            lr_col = 'lr_pro'
        else:
            lr_col = None

        feature_cols = []
        if lr_col is not None:
            feature_cols.append(lr_col)
        if 'nb_1' in data.columns:
            feature_cols.append('nb_1')

    missing_cols = [col for col in feature_cols + [target_col] if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing columns for rating prediction: {missing_cols}")

    # Đảm bảo target và feature là numeric.
    data[target_col] = pd.to_numeric(data[target_col], errors='coerce')
    for col in feature_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    data = data.dropna(subset=feature_cols + [target_col]).copy()

    X = data[feature_cols]
    y = data[target_col]

    print("\n--- RATING PREDICTION DATA ---")
    print(f"Feature columns: {feature_cols}")
    print(f"Target column: {target_col}")
    print(f"Dataset shape for rating prediction: {data.shape}")

    return data, X, y, feature_cols


def train_rating_models(X_train, y_train, random_state=42):
    """
    Chỉ dùng 2 model theo yêu cầu:
    1. Linear Regression: baseline dễ giải thích, khớp paper.
    2. Ridge Regression: regularized linear model, ổn định hơn khi feature probability tương quan.
    """
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0, random_state=random_state)
    }

    for name, model in models.items():
        print(f"Training rating model: {name}")
        model.fit(X_train, y_train)

    return models


def evaluate_rating_models(models, X_test, y_test):
    rows = []
    predictions = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)

        # Rating thật nằm trong [1, 5], nên clip để output hợp lý hơn khi visualize/report.
        y_pred = np.clip(y_pred, 1, 5)
        predictions[name] = y_pred

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        rows.append({
            'Model': name,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2
        })

    metrics_df = pd.DataFrame(rows).sort_values(by='MAE').reset_index(drop=True)
    print("\n--- RATING PREDICTION METRICS ---")
    print(metrics_df)

    return metrics_df, predictions


def add_predictions_to_dataframe(data, X_test, y_test, predictions):
    pred_df = data.loc[X_test.index].copy()
    pred_df['Actual_Rating'] = y_test.values

    for name, y_pred in predictions.items():
        safe_name = name.lower().replace(' ', '_')
        pred_df[f'Predicted_Rating_{safe_name}'] = y_pred
        pred_df[f'Residual_{safe_name}'] = pred_df['Actual_Rating'] - y_pred

    return pred_df


def visualize_rating_metrics(metrics_df, output_dir='outputs'):
    os.makedirs(output_dir, exist_ok=True)

    # MAE comparison
    plt.figure(figsize=(8, 5))
    plt.bar(metrics_df['Model'], metrics_df['MAE'])
    plt.title('Rating Prediction - MAE Comparison')
    plt.xlabel('Model')
    plt.ylabel('MAE')
    plt.tight_layout()
    path_mae = os.path.join(output_dir, 'rating_prediction_mae.png')
    plt.savefig(path_mae, dpi=300, bbox_inches='tight')
    plt.close()

    # RMSE comparison
    plt.figure(figsize=(8, 5))
    plt.bar(metrics_df['Model'], metrics_df['RMSE'])
    plt.title('Rating Prediction - RMSE Comparison')
    plt.xlabel('Model')
    plt.ylabel('RMSE')
    plt.tight_layout()
    path_rmse = os.path.join(output_dir, 'rating_prediction_rmse.png')
    plt.savefig(path_rmse, dpi=300, bbox_inches='tight')
    plt.close()

    return path_mae, path_rmse


def visualize_actual_vs_predicted(pred_df, best_model_name, output_dir='outputs'):
    os.makedirs(output_dir, exist_ok=True)
    safe_name = best_model_name.lower().replace(' ', '_')
    pred_col = f'Predicted_Rating_{safe_name}'

    plt.figure(figsize=(7, 6))
    plt.scatter(pred_df['Actual_Rating'], pred_df[pred_col], alpha=0.35)
    plt.plot([1, 5], [1, 5], linestyle='--')
    plt.title(f'Actual vs Predicted Rating - {best_model_name}')
    plt.xlabel('Actual Rating')
    plt.ylabel('Predicted Rating')
    plt.xlim(0.8, 5.2)
    plt.ylim(0.8, 5.2)
    plt.tight_layout()
    path_scatter = os.path.join(output_dir, f'actual_vs_predicted_{safe_name}.png')
    plt.savefig(path_scatter, dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(pred_df[f'Residual_{safe_name}'], bins=30)
    plt.axvline(0, linestyle='--')
    plt.title(f'Residual Distribution - {best_model_name}')
    plt.xlabel('Actual Rating - Predicted Rating')
    plt.ylabel('Frequency')
    plt.tight_layout()
    path_residual = os.path.join(output_dir, f'residual_{safe_name}.png')
    plt.savefig(path_residual, dpi=300, bbox_inches='tight')
    plt.close()

    return path_scatter, path_residual


def run_rating_prediction(
    df,
    feature_cols=None,
    target_col='Rating',
    test_size=0.2,
    random_state=42,
    output_dir='outputs'
):
    data, X, y, feature_cols = prepare_rating_data(df, feature_cols, target_col)

    # Không stratify trực tiếp như classification vì target là số rời rạc nhưng regression vẫn có thể split thường.
    # Nếu muốn giữ phân phối rating ổn định, stratify=y cũng dùng được vì rating là 1-5.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if y.nunique() > 1 else None
    )

    models = train_rating_models(X_train, y_train, random_state=random_state)
    metrics_df, predictions = evaluate_rating_models(models, X_test, y_test)
    pred_df = add_predictions_to_dataframe(data, X_test, y_test, predictions)

    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, 'rating_prediction_metrics.csv')
    pred_path = os.path.join(output_dir, 'rating_prediction_predictions.csv')
    metrics_df.to_csv(metrics_path, index=False)
    pred_df.to_csv(pred_path, index=False)

    visualize_rating_metrics(metrics_df, output_dir=output_dir)
    best_model_name = metrics_df.iloc[0]['Model']
    visualize_actual_vs_predicted(pred_df, best_model_name, output_dir=output_dir)

    print(f"\nSaved metrics to: {metrics_path}")
    print(f"Saved predictions to: {pred_path}")
    print(f"Best rating model by MAE: {best_model_name}")

    return {
        'models': models,
        'metrics': metrics_df,
        'predictions': pred_df,
        'feature_cols': feature_cols,
        'best_model': best_model_name,
        'metrics_path': metrics_path,
        'predictions_path': pred_path
    }
