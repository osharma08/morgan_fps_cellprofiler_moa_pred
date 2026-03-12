# config.py

CONFIG = {
    # --- User choices ---
    "cell_line": "NIHOVCAR3",     # options: "NIHOVCAR3", "U2OS", "HEPG2_new"

    # Aggregation method
    "aggregation": "mean",      # options: "mean", "median"

    # Normalization method
    # - None → no normalization
    # - "standard" → StandardScaler
    # - "dmso_zscore" → (x - mean(DMSO)) / std(DMSO)
    # - "dmso_mad" → (x - median(DMSO)) / MAD(DMSO)
    "normalization": "dmso_zscore",

    # MOA merging strategy
    # options: "as_is", "merge_tk_egfr_vegfr",
    #          "merge_tk_egfr_keep_vegfr", "merge_tk_vegfr_keep_egfr",
    #          "merge_all_kinases", "merge_st_vs_tk"
    "moa_condition": "as_is",

    # Splitting strategy: choose one
    # - "structure" → scaffold-based split
    # - "stratified" → stratified MOA-based split
    "split_strategy": "structure",

    # Paths to NPZ files for both strategies (must exist in data/<cell_line>/)
    "split_files": {
        "structure": "Splitting_Smiles/structure_as_is_thr0.65_smiles.npz",
        "stratified": "Splitting_Smiles/stratified_as_is_smiles.npz"
    },

    # --- Training hyperparameters ---
    "batch_size": 16,
    "epochs": 1000,
    "learning_rate": 1e-4, 
    "n_folds": 5,          # for cross-validation
    "k_features": 300,     # number of features to select (mutual info)

    # --- Paths ---
    "data_root": "./Data_cp_features/",  # where all cell line folders live
    "metadata_file": "{cell_line}_metadata.xlsx",
    "output_dir": "./results/"           # results will be saved here
}
