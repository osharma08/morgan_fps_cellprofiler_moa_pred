"""
utils.py

Utility functions for preprocessing, MOA label handling, and custom loss
functions used in CellProfiler drug screening workflows.

Includes:
1. Variance-based feature filtering.
2. MOA merging strategies (e.g., collapsing EGFR/VEGFR into TK).
3. Sparse categorical focal loss for imbalanced classification.
"""

import numpy as np
import pandas as pd
import tensorflow as tf


# --- Variance filtering ---
def variance_filter(df, feature_cols, low=0.001, high=10000):
    """
    Drop features with low variance, extremely high variance, or NaN values.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing features + metadata.
    feature_cols : list of str
        Columns to evaluate variance on.
    low : float, default=0.001
        Threshold below which variance is considered too low.
    high : float, default=10000
        Threshold above which variance is considered too high.

    Returns
    -------
    df_filtered : pd.DataFrame
        Dataframe with problematic features removed.
    dropped : list of str
        Names of columns dropped due to low/high variance or NaNs.
    """
    feature_std = df[feature_cols].std()
    low_std = feature_std[feature_std < low].index
    high_std = feature_std[feature_std > high].index
    na_cols = df[feature_cols].columns[df[feature_cols].isnull().any()]

    cols_to_drop = set(low_std).union(high_std).union(na_cols)
    df_filtered = df.drop(columns=cols_to_drop)

    return df_filtered, list(cols_to_drop)


# --- MOA merging ---
MOA_TK, MOA_EGFR, MOA_VEGFR = "TYROSINE KINASE", "EGFR", "VEGFR"

def apply_moa_condition(df, moa_col: str = "moa", condition: str = "as_is"):
    """
    Apply MOA merge conditions to collapse related MOA categories.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing a MOA column.
    moa_col : str, default="moa"
        Column name containing MOA labels.
    condition : str, default="as_is"
        One of:
        - "as_is" → keep all MOAs unchanged
        - "merge_tk_egfr_vegfr" → merge TK, EGFR, VEGFR into TYROSINE KINASE
        - "merge_tk_egfr_keep_vegfr" → merge TK+EGFR, keep VEGFR separate
        - "merge_tk_vegfr_keep_egfr" → merge TK+VEGFR, keep EGFR separate
        - "merge_all_kinases" → merge AKT, AURORA KINASE, CDK, EGFR, JAK, VEGFR, TK into TYROSINE KINASE
        - "merge_st_vs_tk" → merge into 2 broad groups:
              SERINE/THREONINE PROTEIN KINASE: {AKT, AURORA KINASE, CDK, MTOR}
              TYROSINE KINASE: {JAK, EGFR, VEGFR, TK}

    Returns
    -------
    pd.DataFrame
        Copy of input dataframe with updated MOA labels.
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


# --- Custom focal loss ---
def sparse_focal_loss(gamma=1.2, alpha=0.5):
    """
    Sparse categorical focal loss for imbalanced multi-class classification.

    Parameters
    ----------
    gamma : float, default=1.2
        Modulating factor to down-weight easy examples.
    alpha : float, default=0.5
        Class-balancing weight.

    Returns
    -------
    function
        Loss function that can be passed to Keras model.compile().
    """
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_true = tf.one_hot(y_true, depth=tf.shape(y_pred)[-1])
        y_pred = tf.clip_by_value(y_pred, 1e-8, 1. - 1e-8)
        ce = -y_true * tf.math.log(y_pred)
        fl = alpha * tf.pow(1 - y_pred, gamma) * ce
        return tf.reduce_sum(fl, axis=1)
    return loss
