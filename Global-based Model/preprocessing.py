"""
preprocessing.py

This module handles all preprocessing steps for CellProfiler-based
drug screening data. It prepares the dataset for training by:

1. Loading raw plate CSVs for the selected cell line.
2. Cleaning non-feature columns.
3. Aggregating site-level data to well-level (mean/median).
4. Merging with compound metadata (compound, MOA, SMILES).
5. Filtering to retain only DMSO controls and compounds with known MOA.
6. Applying MOA merge rules (as defined in config).
7. Performing variance-based feature filtering.
8. Normalizing features (DMSO-based z-score/MAD if chosen).
9. Splitting into train/validation/test based on NPZ splits.

The output is a dictionary containing train/val/test sets and metadata
ready for model training.
"""

import os
import pandas as pd
import numpy as np
from utils import variance_filter, apply_moa_condition
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

def load_and_prepare(config):
    """
    Load, clean, aggregate, annotate, normalize and split dataset.

    Parameters
    ----------
    config : dict
        Dictionary of configuration options (from config.py).
        Keys include:
            - "cell_line": str, one of {"OVCAR3", "U2OS", "HEPG2"}
            - "aggregation": str or None, {"mean", "median", None}
            - "normalization": str, {"standard", "dmso_zscore", "dmso_mad", None}
            - "moa_condition": str, MOA merging strategy
            - "split_strategy": str, {"structure", "stratified"}
            - "split_files": dict mapping strategy → NPZ path
            - "data_root": base path to cell line folders
            - "metadata_file": str template for metadata filename

    Returns
    -------
    dict
        {
            "trainval_df": pd.DataFrame
                Training + validation wells, before CV split.
            "test_df": pd.DataFrame
                Fixed test wells from NPZ split.
            "feature_cols": list of str
                Names of feature columns used for ML.
            "non_feature_cols": list of str
                Names of metadata/annotation columns.
        }
    """
    # --- Paths ---
    folder = os.path.join(config["data_root"], config["cell_line"])
    files = [f"P{i}.csv" for i in range(1, 5)]   # <-- FIXED
    metadata_file = os.path.join(folder, config["metadata_file"].format(cell_line=config["cell_line"]))
    split_file = os.path.join(folder, config["split_files"][config["split_strategy"]])


    # --- Step 1: Load plates and aggregate ---
    all_plates = []
    for i, file in enumerate(files, start=1):
        df = pd.read_csv(os.path.join(folder, file))

        # Cleaning: remove non-feature columns
        df = df.loc[:, ~df.columns.str.startswith(("Channel_", "Count_", "Frame_", "FileName", "ImageNumber", "Median", "StDev_", 
                                               "ExecutionTime_", "Threshold_", "Width_", "URL_", "Scaling", "Series", "ModuleError",
                                              "PathName_", "ProcessingStatus", "ImageSet_", "MD5Digest_", "Group", "Height"))]
        df = df.loc[:, ~((df.columns.str.startswith("Metadata_")) &
                         (~df.columns.isin(["Metadata_Site", "Metadata_Well"])))]
        df = df.loc[:, ~df.columns.str.contains(r"(_X$|_Y$|_Z$)")]
        df = df.loc[:, ~df.columns.str.contains("_Location")]


        # Aggregate to well-level
        if config["aggregation"]:
            df = df.groupby("Metadata_Well").agg(config["aggregation"]).reset_index()

        # Add plate ID
        df["PlateID"] = f"P{i}"
        all_plates.append(df)

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

    # --- Step 3: Filter to DMSO + compounds with MOA ---
    filtered_df = annotated_df[
        (annotated_df["compound"].fillna("").str.upper() == "DMSO") |
        (annotated_df["moa"].notna() & (annotated_df["moa"].str.strip() != ""))
    ].copy()

    # --- Step 4: Apply MOA merging rule ---
    filtered_df = apply_moa_condition(filtered_df, moa_col="moa", condition=config["moa_condition"])

    # --- Step 5: Variance-based filtering ---
    non_feature_cols = ["PlateID", "Metadata_Well", "compound", "moa", "smiles", "smiles_protonated"]
    
    # Keep only numeric features for variance filtering
    feature_cols = [
        c for c in filtered_df.columns 
        if c not in non_feature_cols and np.issubdtype(filtered_df[c].dtype, np.number)]
    
    filtered_df, dropped = variance_filter(filtered_df, feature_cols)
    '''
    # ✅ Save final feature columns after variance filtering
    col_save_path = os.path.join(folder, f"{config['cell_line']}_final_features.txt")
    with open(col_save_path, "w") as f:
        for col in filtered_df.columns:
            if col not in non_feature_cols:
                f.write(col + "\n")
    
    print(f"Saved final feature list → {col_save_path}")
    print(f"Number of features after variance filtering: {len([c for c in filtered_df.columns if c not in non_feature_cols])}")
    '''
    # Recompute feature_cols AFTER variance filter
    feature_cols = [
        c for c in filtered_df.columns 
        if c not in non_feature_cols and np.issubdtype(filtered_df[c].dtype, np.number)
    ]
    
    # --- Step 6: Normalization (DMSO-based per plate) ---
    if config["normalization"] in ["dmso_zscore", "dmso_mad"]:
        normalized_df = []
    
        for plate, df_plate in filtered_df.groupby("PlateID"):
            dmso = df_plate[df_plate["compound"].str.upper() == "DMSO"]
    
            if dmso.empty:
                print(f"⚠️ No DMSO wells found in {plate}, skipping normalization for this plate.")
                normalized_df.append(df_plate)
                continue
    
            if config["normalization"] == "dmso_zscore":
                mu, sigma = dmso[feature_cols].mean(), dmso[feature_cols].std(ddof=0)
                sigma = sigma.replace(0, np.nan)   # avoid div by 0
                df_plate.loc[:, feature_cols] = (df_plate[feature_cols] - mu) / sigma
            
            elif config["normalization"] == "dmso_mad":
                med = dmso[feature_cols].median()
                mad = (dmso[feature_cols] - med).abs().median()
                mad = mad.replace(0, np.nan)       # avoid div by 0
                df_plate.loc[:, feature_cols] = (df_plate[feature_cols] - med) / mad
                        
            # After normalization, replace NaN/inf with 0
            df_plate.loc[:, feature_cols] = df_plate[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
            
                
            normalized_df.append(df_plate)
    
        filtered_df = pd.concat(normalized_df, ignore_index=True)
    
    # NOTE: If "standard", StandardScaler will be applied later in training.py


    # --- Step 7: Load split from NPZ ---
    splits = np.load(split_file, allow_pickle=True)
    train_smiles, val_smiles, test_smiles = splits["train"], splits["val"], splits["test"]

    # Fixed test set
    test_df = filtered_df[filtered_df["smiles_protonated"].isin(test_smiles)].copy()

    # Merge train+val for CV
    trainval_df = filtered_df[
        filtered_df["smiles_protonated"].isin(train_smiles) |
        filtered_df["smiles_protonated"].isin(val_smiles)
    ].copy()


     # --- Step 8: Add Morgan fingerprints AFTER split ---
    def add_morgan(df, smiles_col="smiles_protonated", n_bits=2048, radius=2):
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
        fps = []
        for smi in df[smiles_col]:
            if pd.isna(smi):
                fps.append(np.zeros(n_bits, dtype=np.float32))
                continue
            mol = Chem.MolFromSmiles(str(smi))
            if mol is not None:
                fp = gen.GetFingerprint(mol)
                arr = np.zeros((n_bits,), dtype=np.int8)
                DataStructs.ConvertToNumpyArray(fp, arr)
                fps.append(arr.astype(np.float32))
            else:
                fps.append(np.zeros(n_bits, dtype=np.float32))
        fps_df = pd.DataFrame(fps, index=df.index,
                              columns=[f"morgan_{i}" for i in range(n_bits)],
                              dtype=np.float32)
        return pd.concat([df, fps_df], axis=1)

    trainval_df = add_morgan(trainval_df)
    test_df = add_morgan(test_df)

    # --- Step 9: Recompute feature_cols (CellProfiler + Morgan) ---
    feature_cols = [
        c for c in trainval_df.columns
        if c not in non_feature_cols and np.issubdtype(trainval_df[c].dtype, np.number)
    ]

    return {
        "trainval_df": trainval_df,
        "test_df": test_df,
        "feature_cols": feature_cols,
        "non_feature_cols": non_feature_cols
    }