from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
def visualize_sentiment(result, y, output_dir='outputs'):
    os.makedirs(output_dir, exist_ok=True)
    y = np.array(y)

    for name, info in result.items():
        proba = info['probability']
        pred  = (proba >= 0.5).astype(int)
        safe  = name.lower().replace(' ', '_')

        # 1. Confusion Matrix
        cm = confusion_matrix(y, pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, cmap='Blues')
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(['Negative', 'Positive'])
        ax.set_yticklabels(['Negative', 'Positive'])
        ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
        ax.set_title(f'Confusion Matrix - {name}')
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'confusion_matrix_{safe}.png'), dpi=300)
        plt.close()

        # 2. ROC-AUC
        fpr, tpr, _ = roc_curve(y, proba)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {name}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'roc_auc_{safe}.png'), dpi=300)
        plt.close()

    # 3. Bar chart so sánh metrics
    metrics_rows = [
        {
            'Model':    name,
            'Accuracy': info['accuracy'],
            'F1':       info['f1'],
        }
        for name, info in result.items()
    ]
    metrics_df = pd.DataFrame(metrics_rows)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(metrics_df))
    width = 0.35
    ax.bar([i - width/2 for i in x], metrics_df['Accuracy'], width, label='Accuracy')
    ax.bar([i + width/2 for i in x], metrics_df['F1'],       width, label='F1')
    ax.set_xticks(list(x))
    ax.set_xticklabels(metrics_df['Model'])
    ax.set_title('Sentiment Model Comparison')
    ax.set_ylabel('Score')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sentiment_model_comparison.png'), dpi=300)
    plt.close()