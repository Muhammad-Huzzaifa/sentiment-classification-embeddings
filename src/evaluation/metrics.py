"""Evaluation metrics."""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)
from src.config import RESULTS_DIR


def evaluate_and_save_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str
) -> Dict[str, float]:
    """Computes metrics and saves confusion matrix.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        model_name: Name used in output files.

    Returns:
        Metric dictionary.
    """
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    filepath = RESULTS_DIR / f"cm_{model_name.replace(' ', '_').lower()}.png"
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    metrics = {
        'Model': model_name,
        'Accuracy': acc,
        'Precision': precision,
        'Recall': recall,
        'F1': f1
    }

    print(f"\n--- {model_name} ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    return metrics


def save_metrics_summary(
    metrics_list: List[Dict[str, float]],
    output_filename: str = "summary.csv"
) -> None:
    """Saves metric rows as CSV.

    Args:
        metrics_list: Rows produced by evaluation.
        output_filename: Target file name.
    """
    df = pd.DataFrame(metrics_list)
    filepath = RESULTS_DIR / output_filename
    df.to_csv(filepath, index=False)
    print(f"\nMetrics saved to {filepath}")
    print(df)
