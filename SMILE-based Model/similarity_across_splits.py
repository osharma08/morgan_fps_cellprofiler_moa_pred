# similarity_across_splits.py
"""
Compute and visualize chemical similarity (Tanimoto) and scaffold overlap
between train/val/test/external splits, per MOA.

Outputs journal-quality bar plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold


def compute_murcko_scaffold(mol):
    """Return Bemis–Murcko scaffold SMILES for a molecule."""
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold)
    except Exception:
        return None


def compute_max_similarity_to_train(test_fp, train_fps):
    """Compute max Tanimoto similarity of a test fp to training fps."""
    if not train_fps:
        return np.nan
    sims = DataStructs.BulkTanimotoSimilarity(test_fp, train_fps)
    return max(sims) if sims else np.nan


def _prepare_fps_scaffolds(df, label_dict, smiles_col="SMILES_protonated_7.4"):
    """Prepare molecules, fingerprints, scaffolds per MOA."""
    df = df.copy()
    df["labels"] = df["moa_label"].map(label_dict)
    df["mol"] = df[smiles_col].apply(Chem.MolFromSmiles)
    df["rdkit_fp"] = df["mol"].apply(
        lambda mol: AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048) if mol else None
    )
    df["scaffold"] = df["mol"].apply(compute_murcko_scaffold)
    return df.dropna(subset=["rdkit_fp"])


# --- Main overlap analysis ---
def analyze_train_other_overlap(train_df, other_df, label_dict, other_name="Other"):
    """
    Analyze max similarity & scaffold overlap between train and another split.
    Ensures all MOAs appear, even if missing in other set.
    """
    train_df = _prepare_fps_scaffolds(train_df, label_dict)
    other_df = _prepare_fps_scaffolds(other_df, label_dict)

    records = []
    for moa, label in label_dict.items():
        moa_train = train_df[train_df["labels"] == label]
        moa_other = other_df[other_df["labels"] == label]

        if moa_other.empty:
            # MOA not in "other" set → include with zeros
            records.append({
                "MOA": moa,
                "Mean_Max_Similarity": 0.0,
                "Scaffold_Overlap": 0.0,
                "Num_Train": len(moa_train),
                f"Num_{other_name}": 0
            })
            continue

        # Compute normally
        train_fps = list(moa_train["rdkit_fp"])
        train_scaffolds = set(moa_train["scaffold"].dropna().tolist())

        max_sims = [
            compute_max_similarity_to_train(fp, train_fps)
            for fp in moa_other["rdkit_fp"]
        ]
        mean_max_sim = np.nanmean(max_sims) if len(max_sims) > 0 else 0.0

        test_scaffolds = set(moa_other["scaffold"].dropna().tolist())
        overlap = len(test_scaffolds & train_scaffolds) / len(test_scaffolds) if test_scaffolds else 0.0

        records.append({
            "MOA": moa,
            "Mean_Max_Similarity": mean_max_sim,
            "Scaffold_Overlap": overlap,
            "Num_Train": len(moa_train),
            f"Num_{other_name}": len(moa_other)
        })

    return pd.DataFrame(records)



# --- Plotting ---
def plot_overlap_bars(results_dict, output_dir=".", prefix="overlap"):
    """
    Plot grouped bar charts for max Tanimoto similarity and scaffold overlap.

    Parameters
    ----------
    results_dict : dict
        Keys = comparison names (e.g., "Train vs Val"),
        Values = DataFrames from analyze_train_other_overlap.
    output_dir : str
        Directory to save plots.
    prefix : str
        Filename prefix.
    """
    palette = {
        "Mean_Max_Similarity": "cornflowerblue",
        "Scaffold_Overlap": "darkorange"
    }

    for name, df in results_dict.items():
        if df.empty:
            continue

        melted = df.melt(
            id_vars="MOA",
            value_vars=["Mean_Max_Similarity", "Scaffold_Overlap"],
            var_name="Metric",
            value_name="Value"
        )
        melted["Metric"] = melted["Metric"].replace({
            "Mean_Max_Similarity": "Max Tanimoto Similarity",
            "Scaffold_Overlap": "Scaffold Overlap"
        })

        plt.figure(figsize=(12, 5))
        ax = sns.barplot(
            data=melted,
            x="MOA", y="Value", hue="Metric",
            palette={
                "Max Tanimoto Similarity": "cornflowerblue",
                "Scaffold Overlap": "darkorange"
            },
            edgecolor="black", linewidth=0.8
        )
        plt.xticks(rotation=90, fontsize=12)
        plt.yticks(fontsize=12)
        plt.xlabel("Mechanism of Action", fontsize=14)
        plt.ylabel("Value", fontsize=14)
        plt.legend(
            title="", fontsize=10, loc="upper center",
            bbox_to_anchor=(0.5, 1.18), ncol=2, frameon=False
        )
        plt.ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{prefix}_{name.replace(' ', '_')}.png", dpi=400)
        plt.close()
