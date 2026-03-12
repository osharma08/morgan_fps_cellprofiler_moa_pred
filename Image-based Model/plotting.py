"""
plotting.py

This module provides utilities for saving and visualizing model
evaluation results, including:

1. Confusion matrix (cumulative across folds).
2. ROC curves (one-vs-rest per MOA, averaged across folds).
3. Saving per-fold, summary, and per-MOA averaged metrics as CSV files.

All plots and metrics are saved in experiment-specific directories
based on the configuration (cell line, aggregation, normalization,
split strategy, MOA condition).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc


def make_experiment_dir(config):
    """Create and return path for saving results based on config."""
    experiment_name = (
        f"{config['cell_line']}_{config['aggregation']}_"
        f"{config['normalization']}_{config['split_strategy']}_"
        f"{config['moa_condition']}"
    )
    save_dir = os.path.join(config["output_dir"], experiment_name)
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def plot_confusion_matrix(cm, int_to_label, config, normalize=True):
    """Plot and save cumulative confusion matrix."""
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    save_dir = make_experiment_dir(config)

    fig, ax = plt.subplots(figsize=(12, 8))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[int_to_label[i] for i in range(len(int_to_label))]
    )
    disp.plot(cmap='Blues', xticks_rotation=90,
              values_format=".2f" if normalize else "d", ax=ax)
    ax.set_xlabel('Predicted label', fontsize=14)
    ax.set_ylabel('True label', fontsize=14)
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.title.set_size(16)
    plt.title(f"Confusion Matrix ({config['cell_line']})", fontsize=16)
    plt.tight_layout()

    fig_path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(fig_path, dpi=400)
    plt.close()
    print(f"Saved confusion matrix → {fig_path}")


def plot_average_roc(y_true_folds, y_prob_folds, int_to_label, config):
    """Plot one-vs-rest ROC curves averaged across folds."""
    n_classes = len(int_to_label)
    mean_fpr = np.linspace(0, 1, 100)

    tpr_per_class = {i: [] for i in range(n_classes)}
    aucs_per_class = {i: [] for i in range(n_classes)}

    for y_true, y_prob in zip(y_true_folds, y_prob_folds):
        # One-hot encode
        y_true_bin = np.zeros((y_true.size, n_classes))
        y_true_bin[np.arange(y_true.size), y_true] = 1

        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            aucs_per_class[i].append(roc_auc)

            # Interpolate TPR
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tpr_per_class[i].append(interp_tpr)

    save_dir = make_experiment_dir(config)
    plt.figure(figsize=(10, 8))

    for i in range(n_classes):
        mean_tpr = np.mean(tpr_per_class[i], axis=0)
        mean_auc = np.mean(aucs_per_class[i])
        std_auc = np.std(aucs_per_class[i])
        plt.plot(mean_fpr, mean_tpr,
                 label=f"{int_to_label[i]} (AUC = {mean_auc:.2f} ± {std_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--", alpha=0.7)
    plt.xlabel("False Positive Rate", fontsize=14)
    plt.ylabel("True Positive Rate", fontsize=14)
    plt.title(f"ROC Curves ({config['cell_line']})", fontsize=16)
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()

    fig_path = os.path.join(save_dir, "roc_curves.png")
    plt.savefig(fig_path, dpi=400)
    plt.close()
    print(f"Saved ROC curves → {fig_path}")


def save_metrics(metrics_df, mean_metrics, std_metrics, config, fold_reports, int_to_label):
    """
    Save per-fold, summary, and per-MOA averaged metrics as CSV files.
    """
    save_dir = make_experiment_dir(config)

    # --- Save per-fold macro metrics
    metrics_path = os.path.join(save_dir, "fold_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)

    # --- Save summary (macro average)
    summary_path = os.path.join(save_dir, "summary_metrics.csv")
    summary_df = pd.DataFrame({
        "metric": mean_metrics.index,
        "mean": mean_metrics.values,
        "std": std_metrics.values
    })
    summary_df.to_csv(summary_path, index=False)

    # --- Save per-MOA averages ---
    per_class_metrics = {label: {"precision": [], "recall": [], "f1": []}
                         for label in int_to_label.values()}

    for rep in fold_reports:
        for moa in int_to_label.values():
            if moa in rep:
                per_class_metrics[moa]["precision"].append(rep[moa]["precision"])
                per_class_metrics[moa]["recall"].append(rep[moa]["recall"])
                per_class_metrics[moa]["f1"].append(rep[moa]["f1-score"])

    per_class_avg = []
    for moa, vals in per_class_metrics.items():
        per_class_avg.append({
            "moa": moa,
            "precision": np.mean(vals["precision"]) if vals["precision"] else np.nan,
            "recall": np.mean(vals["recall"]) if vals["recall"] else np.nan,
            "f1": np.mean(vals["f1"]) if vals["f1"] else np.nan
        })

    per_class_df = pd.DataFrame(per_class_avg)
    per_class_path = os.path.join(save_dir, "per_moa_metrics.csv")
    per_class_df.to_csv(per_class_path, index=False)

    print(f"Saved fold metrics → {metrics_path}")
    print(f"Saved summary metrics → {summary_path}")
    print(f"Saved per-MOA metrics → {per_class_path}")
