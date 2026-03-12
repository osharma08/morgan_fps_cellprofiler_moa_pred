# evaluate.py
"""
Evaluation utilities for MOA classification models.

Supports:
- Internal test set evaluation (from split)
- External test set evaluation (from independent file) #for this project we used 2689 compounds from Probe&Drug

Outputs:
- Confusion matrix (CSV + PNG)
- Per-class precision/recall/F1 (CSV)
- Overall accuracy/F1/ROC-AUC (CSV)
- ROC curves (multi-class, one-vs-rest)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, auc, accuracy_score
)
from tabulate import tabulate

from preprocess import get_morgan_fingerprint


def _save_confusion_matrix(y_true, y_pred, label_dict, title, out_path):
    """Save confusion matrix as PNG (no CSV)."""
    inv_label_dict = {v: k for k, v in label_dict.items()}
    present_labels = sorted(np.unique(y_true))
    present_classes = [inv_label_dict[l] for l in present_labels]

    conf_matrix = confusion_matrix(y_true, y_pred, labels=present_labels)
    conf_matrix_percent = conf_matrix.astype("float") / conf_matrix.sum(axis=1, keepdims=True) * 100

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        conf_matrix_percent, annot=True, fmt=".1f", cmap="Blues",
        xticklabels=present_classes, yticklabels=present_classes
    )
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title(f"Confusion Matrix (%): {title}")
    plt.tight_layout()
    plt.savefig(f"{out_path}_confusion_matrix.png", dpi=400)
    plt.close()


def _save_roc_curves(y_true, y_prob, label_dict, title, out_path):
    """Save ROC curves (one-vs-rest)."""
    from sklearn.preprocessing import label_binarize

    classes = sorted(label_dict.values())
    y_true_bin = label_binarize(y_true, classes=classes)

    plt.figure(figsize=(8, 6))
    for i, class_id in enumerate(classes):
        if class_id not in np.unique(y_true):
            continue
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2,
                 label=f"{list(label_dict.keys())[list(label_dict.values()).index(class_id)]} (AUC={roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves: {title}")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{out_path}_roc.png", dpi=400)
    plt.close()


from data_splitting import apply_moa_condition

def evaluate_test_set(test_df, model, label_dict, out_prefix="testset",
                      smiles_col="SMILES_protonated_7.4",
                      title="Test Set Evaluation",
                      moa_condition="as_is"):
    """
    Evaluate a test set (internal or external) against a trained model.

    Parameters
    ----------
    test_df : pd.DataFrame
        Must contain moa_label + SMILES.
    model : keras.Model
        Trained model.
    label_dict : dict
        MOA → integer label mapping from training.
    out_prefix : str
        Prefix for saving results (e.g., "internal" or "external").
    smiles_col : str
        Column name with SMILES.
    title : str
        Plot title.
    moa_condition : str, default="as_is"
        MOA merge strategy used in training — must match so labels align.

    Returns
    -------
    metrics_df : pd.DataFrame
        Per-class metrics (precision, recall, F1).
    """
    test_df = test_df.copy()

    # --- apply same MOA merge as training ---
    test_df = apply_moa_condition(test_df, moa_col="moa_label", condition=moa_condition)

    # Fingerprints
    test_df = get_morgan_fingerprint(test_df, smiles_col)
    test_df = test_df[["moa_label", "morgan_fps"]]
    test_df["labels"] = test_df["moa_label"].map(label_dict)
    test_df = test_df.dropna(subset=["labels"])  # drop unseen MOAs

    # Features + labels
    x_test = np.array(test_df["morgan_fps"].tolist(), dtype=np.int32)
    y_true = test_df["labels"].to_numpy(dtype=np.int32)

    # Predictions
    y_prob = model.predict(x_test)
    y_pred = np.argmax(y_prob, axis=1)

    # Per-class metrics
    rows = []
    for category, label in label_dict.items():
        if label in y_true:
            precision = precision_score(y_true, y_pred, labels=[label], average=None)[0]
            recall = recall_score(y_true, y_pred, labels=[label], average=None)[0]
            f1 = f1_score(y_true, y_pred, labels=[label], average=None)[0]
        else:
            precision = recall = f1 = np.nan
        rows.append([category, precision, recall, f1])

    metrics_df = pd.DataFrame(rows, columns=["MOA", "Precision", "Recall", "F1"])
    metrics_df.to_csv(f"{out_prefix}_metrics.csv", index=False)

    # Overall metrics
    overall = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
    }
    try:
        overall["roc_auc_macro_ovr"] = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except Exception:
        pass
    pd.Series(overall).to_csv(f"{out_prefix}_overall_scores.csv", header=False)

    # Print table
    from tabulate import tabulate
    printable = [[r[0],
                  f"{r[1]:.4f}" if not np.isnan(r[1]) else "N/A",
                  f"{r[2]:.4f}" if not np.isnan(r[2]) else "N/A",
                  f"{r[3]:.4f}" if not np.isnan(r[3]) else "N/A"]
                 for r in rows]
    print(f"\n=== Evaluation: {title} ===")
    print(tabulate(printable, headers=["Category", "Precision", "Recall", "F1"], tablefmt="grid"))

    # Save confusion matrix + ROC
    _save_confusion_matrix(y_true, y_pred, label_dict, title, f"{out_prefix}")
    _save_roc_curves(y_true, y_prob, label_dict, title, f"{out_prefix}")

    return metrics_df
