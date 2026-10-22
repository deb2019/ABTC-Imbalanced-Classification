# Changelog

## 1.2.0 (2026-10-23)

Release accompanying the revised manuscript. The main-experiment code and results are unchanged. Changes are listed relative to the 1.0.0 code base.

**Added**
- `ablation/`: notebook, script and summary workbook of the revision experiments (Supplementary S1–S5): joint (α, τ) search versus single-threshold search, calibration, cost-driven path, timing.
- `leakage/`: single-feature leakage sensitivity analysis (Supplementary Tables S12–S14).
- `abtc.effective_threshold` and the vectorised `abtc.joint_alpha_tau_search_fast` (same 540 candidates as Path A).
- `config/cost_datasets.csv` (the 17 Path B datasets and their cost ratios).
- `results/cost_sensitive/cost_sensitive_results.csv`/`.xlsx`: the cost-driven results of the main experiment in tidy form (mean and SD in separate columns, data file, cost ratio, flag for the 17 datasets of Table 6), replacing `Cost-Sensitive_Result_Table.xlsx`.
- `data/dataset_catalogue.csv`, replacing `dataset_sources.csv` and `Complete Data Description.csv`: data file, N, d, target and class counts of the files actually used, plus source, licence and the resampling, leakage and duplicate flags.
- `CHANGELOG.md`; authors and DOI in `CITATION.cff`.

**Corrected**
- Documentation of α (`prior_adjust.py`, `search_config.yml`): α = 0 is no adjustment and, for a minority positive class, a larger α raises the adjusted probability. The docstring example was corrected. The formula and its implementation are unchanged.
- Unit tests that asserted the opposite direction of α, and a Path B test whose claim does not hold in general, were replaced by tests of properties that hold exactly (α = 0 identity, effective-threshold identity, optimality of Path B over its α grid).
- End-to-end regression test: the reference generator and the test now run the same single-split pipeline (`src/abtc/ci.py`) and the threshold is chosen on a validation part of the training data. Stale reference files were removed; regenerate them with `scripts/generate_golden.py`.
- `notebooks/02_Reproduce_Public_Results.ipynb` renamed `02_Synthetic_Pipeline_Demo.ipynb` and described as a demonstration on synthetic data; it never reproduced the paper's tables.
- `scripts/reproduce_all.py` replaced by `scripts/run_main_experiment.py`, which states that it runs one configured dataset and stops on failure (`check=True`).
- README and `prior_adjust.py`: the Bayes-rule interpretation of α now states its assumptions (true posterior, change of class prior only); the preprocessing difference between the main experiment (scaling before CV) and the ablation code (within each training subset) is stated.
- `env/abtc_env.yml`: scipy, openpyxl, pytest and jupyterlab added.
- README: timing figures replaced by measured values; PR-AUC removed from the Path A objectives (it does not depend on the threshold); statements about search efficiency replaced by the effective-threshold result.
- `config/metrics_policy.csv`: primary metric of the American Indian drug-use dataset set to F2 as in the paper.

## 1.1.0

Earlier release, archived at https://doi.org/10.5281/zenodo.20728733 (see the GitHub release notes).

## 1.0.0

Single-layer grid search: 27α × 20τ for Path A and 27α with analytical τ* for Path B; main experiment of the original submission.
