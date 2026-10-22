# Ablation and revision experiments

`ABTC_Ablation_Final.ipynb` (and the identical script `abtc_ablation_final.py`) produces the results of Supplementary Sections S1–S5:

| Section | Question | Sheets of `ABTC_ablation_summary.xlsx` |
|---|---|---|
| S2 | Joint (α, τ) search versus single-threshold search | `1_identity_check`, `2_PathA_*`, `3_ridge_ties` |
| S3 | Effect of isotonic calibration | `2_PathA_*` (configurations A6–A8), `2b_prob_quality_*` |
| S4 | Cost-driven path: tuned α versus the Bayes rule α = 0 | `5_PathB_costs`, `5_PathB_tests`, `5_PathB_threshold_shift` |
| S5 | Computational cost | `4_timing_parallel_run_ms`, `4b_timing_sequential_ms` |

Settings are in the configuration cell: data folders (`DATA_DIRS`), the dataset list, `COST_DATASETS` (same as `config/cost_datasets.csv`) and the statistics options. Every setting that changes the results enters a configuration hash; the published workbook has hash `b1ed3987` (`CODE_VERSION = 'v4-2026-10-22'`). Results are cached per dataset under `ABTC_ablation_results/`, so a stopped run can be restarted.

`0_dataset_info` lists, for every data file, N, d, the number of positives and the single-feature leakage screen (features with AUC ≥ 0.97).
