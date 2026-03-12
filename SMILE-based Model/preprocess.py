# preprocess.py
# -*- coding: utf-8 -*-
"""
Preprocessing utilities for MOA modeling:
- Morgan fingerprint generation
- Descriptor calculation
- Lipinski Rule of 5 checks
- Descriptor & Lipinski visualization
- MOA cluster analysis vs thresholds
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors
from rdkit.Chem import rdFingerprintGenerator
from rdkit.SimDivFilters import rdSimDivPickers


def get_morgan_fingerprint(df: pd.DataFrame, smiles_column: str) -> pd.DataFrame:
    """
    Calculate Morgan fingerprints (radius=2, 2048 bits) for molecules in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing SMILES.
    smiles_column : str
        Name of the column with SMILES strings.

    Returns
    -------
    pd.DataFrame
        DataFrame with an added 'morgan_fps' column containing RDKit ExplicitBitVect fingerprints.
    """
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    morgan_fps = []
    for smiles in df[smiles_column]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fp = morgan_gen.GetFingerprint(mol)
            morgan_fps.append(fp)
        else:
            morgan_fps.append(None)

    df['morgan_fps'] = morgan_fps
    return df

def plot_moa_distribution(df: pd.DataFrame, moa_col: str = "moa_label",
                          output_path: str = "moa_distribution_hist.png"):
    """
    Plot histogram (bar chart) of number of compounds per MOA.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with MOA labels.
    moa_col : str, default="moa_label"
        Column containing MOA labels.
    output_path : str
        Path to save the figure.
    """
    import seaborn as sns
    import matplotlib.pyplot as plt

    counts = df[moa_col].value_counts().sort_index()

    plt.figure(figsize=(10, 6))
    sns.barplot(x=counts.index, y=counts.values, color="grey", edgecolor="black")
    plt.xticks(rotation=90, ha="right", fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("Mechanism of Action", fontsize=14)
    plt.ylabel("Number of Compounds", fontsize=14)
    plt.title("MOA Distribution (All Compounds)", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=400)
    plt.show()


def calculate_descriptors(smiles: str):
    """
    Calculate basic physicochemical descriptors from a SMILES string.

    Descriptors:
    - TPSA
    - Num Rotatable Bonds
    - Num H-bond Acceptors
    - Num H-bond Donors
    - LogP
    - Molecular Weight

    Parameters
    ----------
    smiles : str
        Molecule SMILES string.

    Returns
    -------
    tuple
        (TPSA, Rot_Bonds, H_Accpt, H_Donors, LogP, MolWt)
        Returns None for all if SMILES is invalid.
    """
    try:
        if pd.isna(smiles) or not isinstance(smiles, str):
            return None, None, None, None, None, None

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None, None, None, None, None

        tpsa = Descriptors.TPSA(mol)
        num_rotatable_bonds = Descriptors.NumRotatableBonds(mol)
        num_h_acceptors = Descriptors.NumHAcceptors(mol)
        num_h_donors = Descriptors.NumHDonors(mol)
        log_p = Descriptors.MolLogP(mol)
        mol_wt = Descriptors.MolWt(mol)

        return tpsa, num_rotatable_bonds, num_h_acceptors, num_h_donors, log_p, mol_wt

    except Exception as e:
        print(f"Descriptor calculation failed for {smiles}: {e}")
        return None, None, None, None, None, None



def plot_descriptor_histograms_grid(data: pd.DataFrame, target_col: str, 
                                    output_path: str = "descriptor_histograms.png"):
    """
    Plot histogram grid of molecular descriptors per MOA class.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing descriptor columns and a target_col.
    target_col : str
        Column with MOA labels.
    color_dict : dict
        Mapping {MOA: color}.
    output_path : str
        Path to save PNG.
    """
    import matplotlib.patches as mpatches

    descriptors = ['TPSA', 'Rot_Bonds', 'H_Accpt', 'H_Donors', 'LogP', 'Mol.Wt']
    x_limits = {
        'TPSA': (0, 600),
        'Rot_Bonds': (0, 40),
        'H_Accpt': (0, 30),
        'H_Donors': (0, 30),
        'LogP': (-10, 10),
        'Mol.Wt': (0, 1000)
    }

    color_dict = {
        "AKT": "skyblue",
        "AURORA KINASE": "hotpink",
        "CDK": "darkgreen",
        "EGFR": "orange",
        "HDAC": "purple",
        "HMGCR": "brown",
        "JAK": "lightgreen",
        "MTOR": "red",
        "PARP": "dimgray",
        "TOPOISOMERASE": "navy",
        "TUBULIN": "olive",
        "TYROSINE KINASE": "pink",
        "VEGFR": "yellow"
        }
    
    drug_classes = sorted(data[target_col].unique())
    fig, axs = plt.subplots(len(drug_classes), len(descriptors), figsize=(32, 36), sharey='col')

    for i, drug_class in enumerate(drug_classes):
        subset = data[data[target_col] == drug_class]
        for j, descriptor in enumerate(descriptors):
            ax = axs[i, j]
            hist, bins = np.histogram(subset[descriptor], bins=20)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            total = len(subset)
            ax.bar(bin_centers, hist / total * 100,
                   width=np.diff(bins),
                   color=color_dict.get(drug_class, 'gray'),
                   edgecolor='black',
                   alpha=0.7)
            ax.set_xlim(x_limits[descriptor])
            ax.tick_params(axis='both', labelsize=14)
            if i == len(drug_classes) - 1:
                ax.set_xlabel(descriptor, fontsize=14)

    legend_patches = [mpatches.Patch(color=color, label=cls) for cls, color in color_dict.items()]
    fig.legend(handles=legend_patches, loc='upper center', ncol=len(color_dict) // 2, fontsize=18)

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig(output_path, dpi=400, bbox_inches='tight')
    plt.show()


def apply_lipinskis_rule_to_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """
    Apply Lipinski's Rule of 5 filters to a dataset.
    Adds a boolean column 'Ro5_check'.
    """
    data['Ro5_check'] = (
        (data['Mol.Wt'] <= 500) &
        (data['LogP'] <= 5) &
        (data['H_Donors'] <= 5) &
        (data['H_Accpt'] <= 10)
    )
    return data


def plot_lipinski_pie(df: pd.DataFrame, output_path: str = "lipinski_rule_pie.png"):
    """
    Plot overall Lipinski Rule of 5 pass/fail pie chart.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'Ro5_check' column.
    output_path : str
        File path to save plot.
    """
    ro5_counts = df['Ro5_check'].value_counts()
    labels = ['Pass', 'Fail']
    sizes = [ro5_counts.get(True, 0), ro5_counts.get(False, 0)]
    colors = ['lightskyblue', 'hotpink']

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    plt.title("Lipinski's Rule of Five", fontsize=14)
    plt.tight_layout()
    fig.savefig(output_path, dpi=400)
    plt.show()


def plot_lipinski_pie_per_moa(df: pd.DataFrame, output_path: str = "lipinski_pie_per_moa.png"):
    """
    Plot per-MOA Lipinski Rule of 5 pass/fail pie charts.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'moa_label' and 'Ro5_check' columns.
    output_path : str
        File path to save plot.
    """
    grouped_data = df.groupby('moa_label')
    colors = ['lightskyblue', 'hotpink']

    num_classes = len(grouped_data)
    num_rows = math.ceil(num_classes / 4)
    num_cols = 4

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, num_rows * 3))
    axes = axes.flatten()

    for ax, (label, group) in zip(axes, grouped_data):
        ro5_counts = group['Ro5_check'].value_counts()
        sizes = [ro5_counts.get(True, 0), ro5_counts.get(False, 0)]
        ax.pie(sizes, labels=['Pass', 'Fail'], colors=colors, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        ax.set_title(f'{label}', fontsize=14, pad=9)

    for i in range(len(grouped_data), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.savefig(output_path, dpi=400)
    plt.show()


def plot_cluster_vs_threshold(data_frame: pd.DataFrame, column_name: str,
                              num_rows: int, num_cols: int,
                              main_title: str, figsize: tuple,
                              output_path: str = "moa_cluster_analysis.png"):
    """
    Plot #clusters vs threshold curves for each group in column_name.

    Parameters
    ----------
    data_frame : pd.DataFrame
        DataFrame with 'morgan_fps'.
    column_name : str
        Column to group by (e.g., 'moa_label').
    num_rows : int
        Rows in subplot grid.
    num_cols : int
        Columns in subplot grid.
    main_title : str
        Overall title.
    figsize : tuple
        Figure size.
    output_path : str
        Path to save figure.
    """
    plt.figure(figsize=figsize)
    plt.suptitle(main_title, fontsize=14)
    grouped_data = data_frame.groupby(column_name)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)

    for i, (group_key, group_df) in enumerate(grouped_data):
        fingerprints = group_df['morgan_fps'].tolist()
        leader_picker = rdSimDivPickers.LeaderPicker()
        results = []
        for thresh in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1):
            picked_indices = leader_picker.LazyBitVectorPick(fingerprints, len(fingerprints), thresh)
            results.append((thresh, len(picked_indices)))

        row, col = i // num_cols, i % num_cols
        ax = axes[row, col]
        ax.plot([x[0] for x in results], [x[1] for x in results], marker='o', color='dimgrey')
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Clusters')
        ax.set_title(f'{group_key}')
        ax.grid(True)

    for j in range(i + 1, num_rows * num_cols):
        row, col = j // num_cols, j % num_cols
        fig.delaxes(axes[row, col])

    plt.tight_layout()
    plt.savefig(output_path, dpi=400, bbox_inches='tight')
    plt.show()
