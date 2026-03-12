"""
training.py

This module defines the neural network architecture and the training
pipeline for CellProfiler-based drug screening profiles. It supports:

1. Building a feed-forward neural network with focal loss.
2. Performing k-fold cross-validation (train/val splits) while keeping
   a fixed external test set from NPZ files.
3. Applying optional StandardScaler normalization on training folds.
4. Selecting features using mutual information (SelectKBest).
5. Handling class imbalance via scikit-learn's class_weight.
6. Returning fold metrics, averaged metrics, confusion matrix, and ROC data.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import SparseTopKCategoricalAccuracy
import os
import joblib
from utils import sparse_focal_loss


def build_model(input_size, num_classes, lr=1e-4):
    """
    Define and compile the neural network model.

    Parameters
    ----------
    input_size : int
        Number of input features.
    num_classes : int
        Number of output classes (MOAs).
    lr : float, default=1e-4
        Learning rate for Adam optimizer.

    Returns
    -------
    model : tensorflow.keras.Model
        Compiled Keras model.
    """
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_size,), kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(64, activation='relu', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer=Adam(lr),
        loss=sparse_focal_loss(alpha=0.5, gamma=1.2),
        metrics=['accuracy', SparseTopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
    )
    return model


def run_training(config, data_dict):
    """
    Run k-fold cross-validation on trainval_df and evaluate on fixed test_df.

    Returns
    -------
    dict
        {
            "metrics_df": pd.DataFrame of per-fold macro metrics,
            "mean_metrics": pd.Series, average macro metrics across folds,
            "std_metrics": pd.Series, std dev of macro metrics across folds,
            "cumulative_cm": np.ndarray, summed confusion matrix across folds,
            "y_true_folds": list of np.ndarray, test labels per fold,
            "y_prob_folds": list of np.ndarray, test probs per fold,
            "int_to_label": dict, index→MOA mapping,
            "fold_reports": list of dicts (per-fold classification_report),
            "per_class_df": pd.DataFrame, mean precision/recall/F1 per MOA across folds
        }
    """
    trainval_df = data_dict["trainval_df"]
    test_df = data_dict["test_df"]
    feature_cols = data_dict["feature_cols"]

    # Extract feature arrays and labels
    X_trainval = trainval_df[feature_cols].to_numpy()
    y_trainval = trainval_df["moa"].to_numpy()
    X_test = test_df[feature_cols].to_numpy()
    y_test = test_df["moa"].to_numpy()

    # Encode labels
    all_labels = np.unique(np.concatenate([y_trainval, y_test]))
    label_to_int = {label: idx for idx, label in enumerate(all_labels)}
    int_to_label = {idx: label for label, idx in label_to_int.items()}
    y_trainval_enc = np.array([label_to_int[l] for l in y_trainval])
    y_test_enc = np.array([label_to_int[l] for l in y_test])

    # Stratified CV setup
    skf = StratifiedKFold(n_splits=config["n_folds"], shuffle=True, random_state=42)

    fold_metrics = []
    fold_reports = []
    y_true_folds, y_prob_folds = [], []
    cumulative_cm = np.zeros((len(all_labels), len(all_labels)), dtype=np.int32)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_trainval, y_trainval_enc), 1):
        print(f"\n=== Fold {fold}/{config['n_folds']} ===")

        X_tr, X_val = X_trainval[train_idx], X_trainval[val_idx]
        y_tr, y_val = y_trainval_enc[train_idx], y_trainval_enc[val_idx]

        # --- Normalization ---
        if config["normalization"] == "standard":
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_val = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)
        else:
            # Already normalized in preprocessing (DMSO-based)
            X_tr, X_val, X_test_scaled = X_tr, X_val, X_test

        # --- Feature selection ---
        selector = SelectKBest(mutual_info_classif, k=config["k_features"])
        X_tr_sel = selector.fit_transform(X_tr, y_tr)
        X_val_sel = selector.transform(X_val)
        X_test_sel = selector.transform(X_test_scaled)
        
        # --- Class weights ---
        class_weights = compute_class_weight(class_weight="balanced",
                                             classes=np.unique(y_tr), y=y_tr)
        class_weight_dict = {i: w for i, w in enumerate(class_weights)}

        # --- Model ---
        model = build_model(X_tr_sel.shape[1], len(all_labels), lr=config["learning_rate"])
        model.fit(X_tr_sel, y_tr,
                  validation_data=(X_val_sel, y_val),
                  batch_size=config["batch_size"],
                  epochs=config["epochs"],
                  class_weight=class_weight_dict,
                  verbose=0)

        # --- Save model for this fold ---
        save_dir = os.path.join(config["output_dir"], "models")
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir,
                                  f"{config['cell_line']}_{config['aggregation']}_{config['normalization']}_"
                                  f"{config['split_strategy']}_{config['moa_condition']}_fold{fold}.h5")
        model.save(model_path)
        print(f"✅ Saved model for fold {fold} → {model_path}")

        # --- Predictions ---
        y_test_probs = model.predict(X_test_sel)
        y_test_preds = np.argmax(y_test_probs, axis=1)

        # --- Macro metrics ---
        acc = accuracy_score(y_test_enc, y_test_preds)
        prec = precision_score(y_test_enc, y_test_preds, average="macro", zero_division=0)
        rec = recall_score(y_test_enc, y_test_preds, average="macro", zero_division=0)
        f1 = f1_score(y_test_enc, y_test_preds, average="macro", zero_division=0)

        print(f"Test Accuracy={acc:.3f}, Precision={prec:.3f}, Recall={rec:.3f}, F1={f1:.3f}")
        fold_metrics.append({"fold": fold, "acc": acc, "precision": prec, "recall": rec, "f1": f1})

        # --- Per-class metrics ---
        report = classification_report(y_test_enc, y_test_preds,
                                       target_names=[int_to_label[i] for i in range(len(all_labels))],
                                       output_dict=True, zero_division=0)
        fold_reports.append(report)

        # Store for ROC/CM
        y_true_folds.append(y_test_enc)
        y_prob_folds.append(y_test_probs)
        cm = confusion_matrix(y_test_enc, y_test_preds, labels=range(len(all_labels)))
        cumulative_cm += cm

    # --- Aggregate per-fold macro metrics ---
    metrics_df = pd.DataFrame(fold_metrics)
    mean_metrics = metrics_df.mean(numeric_only=True)
    std_metrics = metrics_df.std(numeric_only=True)

    print("\n=== Average across folds (Test Set) ===")
    for k in ["acc", "precision", "recall", "f1"]:
        print(f"{k.upper()}: {mean_metrics[k]:.3f} ± {std_metrics[k]:.3f}")

    # --- Aggregate per-class metrics across folds ---
    per_class_metrics = {label: {"precision": [], "recall": [], "f1": []} for label in all_labels}
    for rep in fold_reports:
        for moa in all_labels:
            per_class_metrics[moa]["precision"].append(rep[moa]["precision"])
            per_class_metrics[moa]["recall"].append(rep[moa]["recall"])
            per_class_metrics[moa]["f1"].append(rep[moa]["f1-score"])

    per_class_avg = {
        moa: {
            "precision": np.mean(per_class_metrics[moa]["precision"]),
            "recall": np.mean(per_class_metrics[moa]["recall"]),
            "f1": np.mean(per_class_metrics[moa]["f1"]),
        }
        for moa in per_class_metrics
    }
    per_class_df = pd.DataFrame(per_class_avg).T
    per_class_df.index.name = "moa"

    print("\n=== Per-MOA average metrics (Test Set, across folds) ===")
    print(per_class_df.round(3))

    return {
        "metrics_df": metrics_df,
        "mean_metrics": mean_metrics,
        "std_metrics": std_metrics,
        "cumulative_cm": cumulative_cm,
        "y_true_folds": y_true_folds,
        "y_prob_folds": y_prob_folds,
        "int_to_label": int_to_label,
        "fold_reports": fold_reports,
        "per_class_df": per_class_df
    }
