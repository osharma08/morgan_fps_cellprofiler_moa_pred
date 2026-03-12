"""
preprocess_external_test_data.py

This script prepares an *external* dataset (e.g., Jump Pilot plates)
for evaluation using a previously trained model.

It reuses the same logic as `preprocessing.py`, except:
- It does NOT load or apply NPZ splits.
- It does NOT perform variance filtering (to preserve model feature alignment).
- It outputs a single processed dataset (X, y) ready for inference/evaluation.
"""

import os
import pandas as pd
import numpy as np
from config import CONFIG
from utils import apply_moa_condition


def load_and_prepare_external(config, training_feature_file=None):
    """
    Preprocess external dataset (no splitting, no variance filtering).

    Steps:
    1. Load plate CSVs from data_root/cell_line.
    2. Clean non-feature columns.
    3. Aggregate to well-level (mean or median).
    4. Merge with metadata.
    5. Apply MOA merge rules.
    6. Normalize per plate (if selected).
    7. Optionally align features with the training feature list.
    8. Return final dataframe and feature column list.

    Parameters
    ----------
    config : dict
        Configuration from config.py.
    training_feature_file : str, optional
        Path to .npy file containing the list of feature columns
        used during model training (for alignment).

    Returns
    -------
    dict
        {
            "df": processed DataFrame,
            "feature_cols": list of feature columns,
            "non_feature_cols": list of metadata columns
        }
    """

    # --- Paths ---
    folder = os.path.join(config["data_root"], config["cell_line"])
    files = [f"P{i}_aligned.csv" for i in range(1, 5)]
    metadata_file = os.path.join(folder, config["metadata_file"].format(cell_line=config["cell_line"]))

    # --- Step 1: Load and aggregate plates ---
    all_plates = []
    for i, file in enumerate(files, start=1):
        plate_path = os.path.join(folder, file)
        print(f"Reading plate: {file}")
        if not os.path.exists(plate_path):
            print(f"⚠️ Skipping missing file: {plate_path}")
            continue

        df = pd.read_csv(plate_path)

        # Remove non-feature columns
        df = df.loc[:, ~df.columns.str.startswith((
            "Channel_", "Count_", "Frame_", "FileName", "ImageNumber",
            "Median", "StDev_", "ExecutionTime_", "Threshold_", "Width_",
            "URL_", "Scaling", "Series", "ModuleError", "PathName_",
            "ProcessingStatus", "ImageSet_", "MD5Digest_", "Group", "Height"
        ))]
        df = df.loc[:, ~((df.columns.str.startswith("Metadata_")) &
                         (~df.columns.isin(["Metadata_Site", "Metadata_Well"])))]
        df = df.loc[:, ~df.columns.str.contains(r"(_X$|_Y$|_Z$)")]
        df = df.loc[:, ~df.columns.str.contains("_Location")]

        # Aggregate to well-level
        if config["aggregation"]:
            df = df.groupby("Metadata_Well").agg(config["aggregation"]).reset_index()

        df["PlateID"] = f"P{i}"
        all_plates.append(df)

    if not all_plates:
        raise FileNotFoundError("No valid plate CSV files found in folder.")

    merged_df = pd.concat(all_plates, ignore_index=True)

    # --- Step 2: Merge with metadata ---
    metadata_df = pd.read_excel(metadata_file)
    metadata_df = metadata_df.drop_duplicates(subset=["plate", "well"])
    metadata_df = metadata_df.rename(columns={
        "plate": "PlateID",
        "well": "Metadata_Well",
        "compound": "compound",
        "moa": "moa",
        "smiles": "smiles",
        "smiles_protonated": "smiles_protonated"
    })

    annotated_df = pd.merge(
        merged_df,
        metadata_df[["PlateID", "Metadata_Well", "compound", "moa", "smiles", "smiles_protonated"]],
        on=["PlateID", "Metadata_Well"],
        how="left"
    )
    print("✅ Reached end of Step 2")


    # --- Step 3: Filter to DMSO + compounds with MOA ---
    filtered_df = annotated_df[
        (annotated_df["compound"].fillna("").str.upper() == "DMSO") |
        (annotated_df["moa"].notna() & (annotated_df["moa"].str.strip() != ""))
    ].copy()

    # --- Step 4: Apply MOA merging ---
    print("Before MOA condition:", filtered_df["moa"].unique())

    filtered_df = apply_moa_condition(filtered_df, moa_col="moa", condition=config["moa_condition"])

    print("After MOA condition:", filtered_df["moa"].unique())


    # --- Step 5: (Removed) Variance filtering ---
    # Skipped to preserve feature consistency with the pretrained model.

    # --- Step 6: Normalization ---
    non_feature_cols = ["PlateID", "Metadata_Well", "compound", "moa", "smiles", "smiles_protonated"]
    feature_cols = [
        c for c in filtered_df.columns
        if c not in non_feature_cols and np.issubdtype(filtered_df[c].dtype, np.number)
    ]

    if config["normalization"] in ["dmso_zscore", "dmso_mad"]:
        normalized_df = []
        for plate, df_plate in filtered_df.groupby("PlateID"):
            dmso = df_plate[df_plate["compound"].str.upper() == "DMSO"]
            if dmso.empty:
                print(f"⚠️ No DMSO wells in {plate}. Skipping normalization.")
                normalized_df.append(df_plate)
                continue

            if config["normalization"] == "dmso_zscore":
                mu, sigma = dmso[feature_cols].mean(), dmso[feature_cols].std(ddof=0)
                sigma = sigma.replace(0, np.nan)
                df_plate.loc[:, feature_cols] = (df_plate[feature_cols] - mu) / sigma
            elif config["normalization"] == "dmso_mad":
                med = dmso[feature_cols].median()
                mad = (dmso[feature_cols] - med).abs().median()
                mad = mad.replace(0, np.nan)
                df_plate.loc[:, feature_cols] = (df_plate[feature_cols] - med) / mad

            df_plate.loc[:, feature_cols] = df_plate[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
            normalized_df.append(df_plate)

        filtered_df = pd.concat(normalized_df, ignore_index=True)

    # --- Step 7: Optional alignment with training feature list ---
    if training_feature_file and os.path.exists(training_feature_file):
        training_features = np.load(training_feature_file, allow_pickle=True)
        missing = [f for f in training_features if f not in filtered_df.columns]
        extra = [f for f in filtered_df.columns if f not in training_features and f not in non_feature_cols]

        if missing:
            print(f"⚠️ Adding {len(missing)} missing features as zeros to match training features.")
            for col in missing:
                filtered_df[col] = 0.0
        if extra:
            print(f"⚠️ Dropping {len(extra)} extra features not used in training.")
            filtered_df = filtered_df.drop(columns=extra)

        # Reorder columns to match training
        filtered_df = filtered_df[non_feature_cols + list(training_features)]
        feature_cols = list(training_features)

    print(f"Preprocessing complete. Final shape: {filtered_df.shape}")
    print(f"Number of features: {len(feature_cols)}")

    return {
        "df": filtered_df,
        "feature_cols": feature_cols,
        "non_feature_cols": non_feature_cols
    }


