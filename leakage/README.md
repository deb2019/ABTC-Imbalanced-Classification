# Single-feature leakage sensitivity

`leakage_sensitivity.py` recomputes Supplementary Tables S12–S14 from `main_results_long.csv` (mean score of every method on every dataset of the main experiment, with the dataset's primary metric). No model is retrained.

Two exclusion sets are used (Supplementary Table S11): 21 datasets with a single feature of ROC-AUC = 1.000, and a wider set adding two datasets with a single-feature AUC above 0.99 and a target-related feature name. Wilcoxon p-values use the normal approximation with tie correction; Holm's correction is applied within each metric family and, as a check, over all 32 tests.

Output: `leakage_sensitivity_results.xlsx`.
