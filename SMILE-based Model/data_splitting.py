# data_splitting.py
# -*- coding: utf-8 -*-
"""
Data splitting utilities for MOA modeling.

Provides two modes of splitting:
1. Structure-based clustering (LeaderPicker + cluster threshold)
2. Random stratified split by MOA label

Includes MOA merging rules so splits respect merged labels.
"""
import os
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from rdkit import DataStructs, Chem
from rdkit.SimDivFilters import rdSimDivPickers
from rdkit.Chem import rdFingerprintGenerator

# --- MOA merge rules ---
MOA_TK, MOA_EGFR, MOA_VEGFR = "TYROSINE KINASE", "EGFR", "VEGFR"

def apply_moa_condition(df: pd.DataFrame, moa_col: str = "moa_label", condition: str = "as_is") -> pd.DataFrame:
    """
    Apply MOA merge conditions.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with a moa_col.
    moa_col : str, default="moa_label"
        Column with MOA labels.
    condition : str, default="as_is"
        One of:
        - "as_is"  → keep all MOAs unchanged
        - "merge_tk_egfr_vegfr" → merge TK, EGFR, VEGFR into TYROSINE KINASE
        - "merge_tk_egfr_keep_vegfr" → merge TK+EGFR, keep VEGFR separate
        - "merge_tk_vegfr_keep_egfr" → merge TK+VEGFR, keep EGFR separate
        - "merge_all_kinases" → merge AKT, AURORA KINASE, CDK, EGFR, JAK, VEGFR, TK into TYROSINE KINASE
        - "merge_st_vs_tk" → merge into 2 categories:
              SERINE/THREONINE PROTEIN KINASE: {AKT, AURORA KINASE, CDK, MTOR}
              TYROSINE KINASE: {JAK, EGFR, VEGFR, TK}

    Returns
    -------
    pd.DataFrame
        Copy of df with updated MOA labels.
    """
    mapping = {k: k for k in df[moa_col].astype(str).unique()}

    if condition == "merge_tk_egfr_vegfr":
        for k in [MOA_TK, MOA_EGFR, MOA_VEGFR]:
            mapping[k] = MOA_TK

    elif condition == "merge_tk_egfr_keep_vegfr":
        for k in [MOA_TK, MOA_EGFR]:
            mapping[k] = MOA_TK
        mapping[MOA_VEGFR] = MOA_VEGFR

    elif condition == "merge_tk_vegfr_keep_egfr":
        for k in [MOA_TK, MOA_VEGFR]:
            mapping[k] = MOA_TK
        mapping[MOA_EGFR] = MOA_EGFR

    elif condition == "merge_all_kinases":
        for k in ["AKT", "AURORA KINASE", "CDK", "EGFR", "JAK", "VEGFR", MOA_TK]:
            mapping[k] = MOA_TK

    elif condition == "merge_st_vs_tk":
        for k in ["AKT", "AURORA KINASE", "CDK", "MTOR"]:
            mapping[k] = "SERINE/THREONINE PROTEIN KINASE"
        for k in ["JAK", "EGFR", "VEGFR", MOA_TK]:
            mapping[k] = "TYROSINE KINASE"

    out = df.copy()
    out[moa_col] = out[moa_col].astype(str).map(mapping).fillna(out[moa_col].astype(str))
    return out



# --- Morgan fingerprint generator ---
def get_morgan_fingerprint(df: pd.DataFrame, smiles_column: str) -> pd.DataFrame:
    """
    Compute Morgan fingerprints (radius=2, 2048 bits) for molecules.

    Adds 'morgan_fps' column.
    """
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = []
    for smi in df[smiles_column]:
        mol = Chem.MolFromSmiles(smi)
        fps.append(morgan_gen.GetFingerprint(mol) if mol is not None else None)
    df = df.copy()
    df["morgan_fps"] = fps
    return df


# --- Core clustering function ---
def cluster_data(dataset: pd.DataFrame, threshold: float):
    """
    Cluster within each MOA using LeaderPicker.

    Leaders (representatives) → set='tr'
    Others → set='val' initially
    """
    def assign_points_to_clusters(picked_indices, fingerprints):
        clusters = defaultdict(list)
        for i, idx in enumerate(picked_indices):
            clusters[i].append(idx)

        similarities = np.zeros((len(picked_indices), len(fingerprints)))
        for i, centroid_idx in enumerate(picked_indices):
            similarities[i, :] = DataStructs.BulkTanimotoSimilarity(fingerprints[centroid_idx], fingerprints)
            similarities[i, i] = 0

        best_matches = np.argmax(similarities, axis=0)
        for i, idx in enumerate(best_matches):
            if i not in picked_indices:
                clusters[idx].append(i)
        return clusters

    grouped_data = dataset.groupby("moa_label")
    for _, group_df in grouped_data:
        group_df = group_df.copy()
        group_df["set"] = "val"

        fingerprints = group_df["morgan_fps"].tolist()
        leader_picker = rdSimDivPickers.LeaderPicker()
        fps = [fp for fp in fingerprints if fp is not None]

        picked_indices = leader_picker.LazyBitVectorPick(fps, len(fps), threshold)
        representative_indices = group_df.iloc[picked_indices]
        representative_compounds = representative_indices["name"]

        # mark leaders as training
        group_df.loc[group_df["name"].isin(representative_compounds), "set"] = "tr"

        # assign clusters
        clusters = assign_points_to_clusters(picked_indices, fingerprints)
        group_df["cluster"] = None
        for cluster_idx, compound_indices in clusters.items():
            group_df.loc[group_df.index.isin(compound_indices), "cluster"] = cluster_idx

        dataset.loc[group_df.index, ["set", "cluster"]] = group_df[["set", "cluster"]].values


def save_split_smiles(df_train, df_val, df_test, smiles_col, split_choice, moa_condition, threshold=None, output_dir="outputs"):
    """
    Save train/val/test SMILES arrays into a single .npz file.

    Parameters
    ----------
    df_train, df_val, df_test : pd.DataFrame
        Splits containing a SMILES column.
    smiles_col : str
        Column containing SMILES strings.
    split_choice : str
        "structure" or "stratified".
    moa_condition : str
        MOA merging strategy used.
    threshold : float, optional
        Threshold for structure-based clustering.
    output_dir : str
        Directory to save .npz file.
    """
    os.makedirs(output_dir, exist_ok=True)

    tag = f"{split_choice}_{moa_condition}"
    if split_choice == "structure" and threshold is not None:
        tag += f"_thr{threshold:.2f}"

    out_path = f"{output_dir}/{tag}_smiles.npz"

    np.savez(out_path,
             train=df_train[smiles_col].values,
             val=df_val[smiles_col].values,
             test=df_test[smiles_col].values)

    print(f"✅ Saved SMILES splits → {out_path}")

# --- Structure split ---
def process_dataset_across_thresholds(
    file_path: str,
    smiles_col: str = "SMILES_protonated_7.4",
    thresholds=[0.65],
    output_dir: str = "outputs",
    moa_condition: str = "as_is",
):
    """
    Cluster-based split across thresholds.

    Steps:
    1. Load Excel dataset
    2. Drop NaN/duplicates in SMILES
    3. Compute fingerprints
    4. Apply MOA merging
    5. Cluster per MOA, assign leaders→tr, others→val
    6. Split val into val/test **stratified by MOA**
    7. Save splits as CSV

    Parameters
    ----------
    file_path : str
        Path to Excel dataset.
    smiles_col : str
        Column with protonated SMILES.
    thresholds : list of float
        Cluster thresholds to evaluate.
    output_dir : str
        Directory to save CSVs.
    moa_condition : str
        MOA merge condition.

    Returns
    -------
    dict
        { threshold_key: DataFrame }
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    datasets = {}
    base_df = pd.read_excel(file_path)
    base_df.dropna(subset=[smiles_col], inplace=True)
    base_df.drop_duplicates(subset=[smiles_col], inplace=True)

    base_df = get_morgan_fingerprint(base_df, smiles_col)
    base_df = apply_moa_condition(base_df, moa_col="moa_label", condition=moa_condition)

    for threshold in thresholds:
        df_copy = base_df.copy()
        cluster_data(df_copy, threshold=threshold)

        # --- NEW: split val into val/test stratified per MOA ---
        from sklearn.model_selection import train_test_split
        val_df = df_copy[df_copy["set"] == "val"].copy()

        if not val_df.empty:
            new_val_idx = []
            new_test_idx = []
            for moa, group in val_df.groupby("moa_label"):
                if len(group) == 1:
                    # if only 1 compound → keep in val
                    new_val_idx.extend(group.index.tolist())
                else:
                    val_idx, test_idx = train_test_split(
                        group.index, test_size=0.5, random_state=42, shuffle=True
                    )
                    new_val_idx.extend(val_idx)
                    new_test_idx.extend(test_idx)

            df_copy.loc[new_val_idx, "set"] = "val"
            df_copy.loc[new_test_idx, "set"] = "test"

        # naming with threshold + moa_condition
        thr_str = str(threshold).replace(".", "")
        name = f"structure_thr{threshold}_sim{1-threshold}_{moa_condition}"
        datasets[name] = df_copy

        # save with descriptive names
        df_copy[df_copy["set"] == "tr"].to_csv(f"{output_dir}/train_{name}.csv", index=False)
        df_copy[df_copy["set"] == "val"].to_csv(f"{output_dir}/val_{name}.csv", index=False)
        df_copy[df_copy["set"] == "test"].to_csv(f"{output_dir}/test_{name}.csv", index=False)

        df_train = df_copy[df_copy["set"] == "tr"].copy()
        df_val   = df_copy[df_copy["set"] == "val"].copy()
        df_test  = df_copy[df_copy["set"] == "test"].copy()

    save_split_smiles(df_train, df_val, df_test,
                  smiles_col, split_choice="structure",
                  moa_condition=moa_condition, threshold=threshold,
                  output_dir=output_dir)

    
    return datasets


def split_stratified(
    df: pd.DataFrame,
    label_col: str = "moa_label",
    smiles_col: str = "SMILES_protonated_7.4",
    test_size: float = 0.2,
    val_size: float = 0.1,
    seed: int = 42,
    output_dir: str = "outputs",
    moa_condition: str = "as_is",
):
    """
    Stratified split by MOA label.

    Ensures *every* MOA appears in train, val, test 
    (if that MOA has >=3 compounds).
    For very small MOAs (1–2 compounds), keeps them in train/val.

    Also computes Morgan fingerprints via `get_morgan_fingerprint`.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with MOA labels + SMILES.
    label_col : str
        Column with MOA labels.
    smiles_col : str
        Column with SMILES strings.
    test_size : float
        Fraction of test set.
    val_size : float
        Fraction of validation set.
    seed : int
        Random seed.
    output_dir : str
        Where to save CSVs.
    moa_condition : str
        MOA merge condition.

    Returns
    -------
    (df_train, df_val, df_test)
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # --- apply MOA merge ---
    df = apply_moa_condition(df, moa_col=label_col, condition=moa_condition)
    df = df.reset_index(drop=True)

    # --- compute Morgan fingerprints ---
    df = get_morgan_fingerprint(df, smiles_col)

    # --- stratified split by MOA ---
    train_idx, val_idx, test_idx = [], [], []

    for moa, group in df.groupby(label_col):
        n = len(group)
        if n == 1:
            train_idx.extend(group.index)
        elif n == 2:
            idx_tr, idx_val = train_test_split(group.index, test_size=0.5, random_state=seed)
            train_idx.extend(idx_tr); val_idx.extend(idx_val)
        else:
            idx_temp, idx_test = train_test_split(
                group.index, test_size=test_size, random_state=seed, stratify=group[label_col]
            )
            val_rel = val_size / (1 - test_size)
            idx_train, idx_val = train_test_split(
                idx_temp, test_size=val_rel, random_state=seed, stratify=df.loc[idx_temp, label_col]
            )
            train_idx.extend(idx_train); val_idx.extend(idx_val); test_idx.extend(idx_test)

    # assign splits
    df_train, df_val, df_test = df.loc[train_idx].copy(), df.loc[val_idx].copy(), df.loc[test_idx].copy()
    df_train["set"], df_val["set"], df_test["set"] = "tr", "val", "test"

    # save
    tag = f"stratified_{moa_condition}"
    df_train.to_csv(f"{output_dir}/train_{tag}.csv", index=False)
    df_val.to_csv(f"{output_dir}/val_{tag}.csv", index=False)
    df_test.to_csv(f"{output_dir}/test_{tag}.csv", index=False)
    pd.concat([df_train, df_val, df_test]).to_csv(f"{output_dir}/all_with_sets_{tag}.csv", index=False)

    save_split_smiles(df_train, df_val, df_test,
                  smiles_col="SMILES_protonated_7.4",
                  split_choice="stratified",
                  moa_condition=moa_condition,
                  output_dir=output_dir)


    return df_train, df_val, df_test



def prepare_data(input_data: pd.DataFrame):
    """
    Convert DataFrame with 'set' column into numpy arrays.

    Parameters
    ----------
    input_data : pd.DataFrame
        Must contain ['morgan_fps', 'labels', 'set'].

    Returns
    -------
    tuple
        (x_tr, y_tr, x_val, y_val, x_test, y_test)
    """
    x_tr = np.array(input_data[input_data['set'] == 'tr']['morgan_fps'].tolist(), dtype=np.int32)
    x_val = np.array(input_data[input_data['set'] == 'val']['morgan_fps'].tolist(), dtype=np.int32)
    x_test = np.array(input_data[input_data['set'] == 'test']['morgan_fps'].tolist(), dtype=np.int32)

    y_tr = input_data[input_data['set'] == 'tr']['labels'].to_numpy(dtype=np.int32)
    y_val = input_data[input_data['set'] == 'val']['labels'].to_numpy(dtype=np.int32)
    y_test = input_data[input_data['set'] == 'test']['labels'].to_numpy(dtype=np.int32)

    print("      Dataset     |     Shape")
    print("------------------------------------")
    print(f"Training set      | {x_tr.shape}")
    print(f"Training labels   | {y_tr.shape}")
    print(f"Validation set    | {x_val.shape}")
    print(f"Validation labels | {y_val.shape}")
    print(f"Test set          | {x_test.shape}")
    print(f"Test labels       | {y_test.shape}")
    print("------------------------------------")

    return x_tr, y_tr, x_val, y_val, x_test, y_test
