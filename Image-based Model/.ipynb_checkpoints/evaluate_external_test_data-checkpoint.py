"""
evaluate_external_test_data.py

Module to evaluate a trained model on an external dataset (e.g., Jump Pilot).

Steps:
1. Load trained model (.h5) and label encoder or mapping.
2. Preprocess external data (already done using preprocess_ext_test_data.py).
3. Extract X (features) and y (true labels, if available).
4. Predict with model.
5. Compute metrics (accuracy, precision, recall, F1, ROC-AUC).
6. Plot confusion matrix + ROC curves.
7. Save results and per-MOA averages using plotting utilities.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report, confusion_matrix
)
from tensorflow.keras.models import load_model
from plotting import (
    plot_confusion_matrix, plot_average_roc, save_metrics, make_experiment_dir
)


def evaluate_external_dataset(model_path, data_dict, label_encoder, config):
    """
    Evaluate a trained model on preprocessed external data.

    Parameters
    ----------
    model_path : str
        Path to trained model (.h5 file).
    data_dict : dict
        Output from preprocess_ext_test_data.load_and_prepare_external(),
        containing:
        - df : preprocessed DataFrame
        - feature_cols : list of feature column names
        - non_feature_cols : list of metadata columns
    label_encoder : LabelEncoder or mapping dict
        Used to encode/decode MOA labels.
    config : dict
        Experiment configuration (from config.py).

    Returns
    -------
    dict
        Evaluation summary containing:
        - metrics_df : DataFrame with overall metrics
        - per_class_df : DataFrame with per-MOA metrics
    """

    # --- Load model ---
    print(f"Loading trained model from: {model_path}")
    model = load_model(model_path, compile=False)

    # --- Prepare data ---
    df = data_dict["df"]
    feature_cols = data_dict["feature_cols"]

    X = df[feature_cols].values
    y_true = None
    if "moa" in df.columns and df["moa"].notna().any():
        y_true = df["moa"].astype(str).values
        print(f"Found {len(np.unique(y_true))} unique MOA labels in test data.")

    # --- Encode labels if available ---
    if y_true is not None:
        if hasattr(label_encoder, "transform"):
            y_enc = label_encoder.transform(y_true)
            int_to_label = {i: lbl for i, lbl in enumerate(label_encoder.classes_)}
        elif isinstance(label_encoder, dict):
            label_to_int = label_encoder
            int_to_label = {v: k for k, v in label_to_int.items()}
            y_enc = np.array([label_to_int.get(lbl, -1) for lbl in y_true])
            mask = y_enc != -1
            X = X[mask]
            y_enc = y_enc[mask]
        else:
            raise ValueError("Unsupported label_encoder format.")
    else:
        y_enc, int_to_label = None, None
        print("⚠️ No MOA labels found — evaluation limited to prediction output only.")

    # --- Standard normalization if required ---
    if config["normalization"] == "standard":
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    # --- Prediction ---
    print("Running model predictions...")
    y_prob = model.predict(X, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    results_dir = make_experiment_dir(config)

    # --- Metrics (if ground truth exists) ---
    if y_enc is not None:
        acc = accuracy_score(y_enc, y_pred)
        prec_macro = precision_score(y_enc, y_pred, average="macro", zero_division=0)
        rec_macro = recall_score(y_enc, y_pred, average="macro", zero_division=0)
        f1_macro = f1_score(y_enc, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_enc, y_pred, average="weighted", zero_division=0)
        try:
            roc_auc = roc_auc_score(y_enc, y_prob, multi_class="ovr", average="macro")
        except ValueError:
            roc_auc = np.nan

        print(f"\n✅ External Dataset Evaluation:")
        print(f"Accuracy: {acc:.3f}")
        print(f"F1-macro: {f1_macro:.3f}")
        print(f"ROC-AUC (macro): {roc_auc:.3f}")

        # --- Confusion Matrix ---
        cm = confusion_matrix(y_enc, y_pred)
        plot_confusion_matrix(cm, int_to_label, config, normalize=True)

        # --- ROC Curves ---
        plot_average_roc([y_enc], [y_prob], int_to_label, config)

        # --- Per-class metrics ---
        report = classification_report(
            y_enc, y_pred, target_names=int_to_label.values(),
            output_dict=True, zero_division=0
        )

        metrics_df = pd.DataFrame([{
            "Accuracy": acc,
            "Precision_macro": prec_macro,
            "Recall_macro": rec_macro,
            "F1_macro": f1_macro,
            "F1_weighted": f1_weighted,
            "ROC_AUC_macro": roc_auc
        }])

        # Convert classification report to DataFrame
        per_class_df = (
            pd.DataFrame(report)
            .T.reset_index()
            .rename(columns={"index": "MOA"})
        )
        per_class_df = per_class_df[per_class_df["MOA"].isin(int_to_label.values())]

        # Save metrics
        metrics_df.to_csv(os.path.join(results_dir, "external_overall_metrics.csv"), index=False)
        per_class_df.to_csv(os.path.join(results_dir, "external_per_moa_metrics.csv"), index=False)

        print(f"Saved evaluation metrics → {results_dir}")

        return {
            "metrics_df": metrics_df,
            "per_class_df": per_class_df,
            "confusion_matrix": cm
        }

    else:
        # If no ground truth, save predictions only
        df["Predicted_MOA"] = [
            list(int_to_label.values())[i] if int_to_label else str(i)
            for i in y_pred
        ]
        df.to_csv(os.path.join(results_dir, "external_predictions.csv"), index=False)
        print(f"⚠️ No true labels — saved predictions only → {results_dir}")

        return {"predictions": df, "metrics_df": None}
