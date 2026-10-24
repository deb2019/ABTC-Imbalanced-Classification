# ABTC: Adaptive Bayesian Thresholding Classifier

Code, configuration and results for the paper *Adaptive Bayesian Thresholding for Metric and Cost Driven Decisions in Imbalanced Binary Classification* (Roy D., Roy A., Sahu B., Roy U.).

- Repository: https://github.com/deb2019/ABTC-Imbalanced-Classification
- Archived release v1.2.0 (this version): https://doi.org/10.5281/zenodo.23610933
- Earlier release v1.1.0: https://doi.org/10.5281/zenodo.20728733

ABTC selects a decision threshold for a binary classifier in three steps: isotonic calibration of the predicted probabilities, a prior adjustment of the calibrated probability controlled by an exponent α, and a threshold τ chosen for a target metric (Path A) or set from misclassification costs (Path B).

---

## Method in brief

**Prior adjustment.** With π the positive-class proportion in the training data and L = log((1 − π)/π),

```
p_adj = (p / π^α) / (p / π^α + (1 − p) / (1 − π)^α)      i.e.   logit(p_adj) = logit(p) + α·L
```

- α = 0 leaves p unchanged. If p is the true posterior of the training population, α = 1 gives the posterior under equal class priors (class-conditional distributions unchanged).
- For a minority positive class (π < 0.5), a larger α raises p_adj, so the positive class is predicted more often.

**Effective threshold.** p_adj ≥ τ is the same decision as p ≥ t(α, τ) = σ(logit τ − α·L). Every (α, τ) pair is therefore one threshold on the calibrated probability (`abtc.effective_threshold`; Supplementary Proposition S1).

**Path A (metric-driven).** Joint grid over 27 values of α in [0.1, 3.0] and 20 values of τ in [0.01, 0.99] (540 candidates), maximising F1, F2, G-mean, MCC or balanced accuracy on a validation split.

**Path B (cost-driven).** τ* = C_FP / (C_FP + C_FN); α is searched over the same 27 values (27 candidates). If the calibrated probability equals the true posterior P(Y = 1 | x) and the deployment population differs from the training population at most in its class prior, then α = 0 gives the Bayes rule for the stated costs when the priors agree, and a non-zero α corresponds to assuming a different deployment prior (Supplementary Proposition S2). Isotonic calibration does not guarantee these assumptions; with a calibrated score the rule is optimal only among rules that use that score.

**Search cost.** The original implementation builds one confusion matrix per candidate (median 5.5 s per fold over 20 datasets). `joint_alpha_tau_search_fast` evaluates the same 540 candidates with sorted scores and binary search (median 8 ms per fold). Supplementary Table S8 gives the full timing.

---

## Repository layout

```
ABTC/
├─ src/abtc/                    ABTC library (pip install -e .)
│  ├─ calibration.py            isotonic calibration
│  ├─ prior_adjust.py           alpha_adjust_posterior, effective_threshold
│  ├─ thresholding.py           Path A / Path B searches, vectorised Path A
│  ├─ metrics.py                F-beta, G-mean, MCC, balanced accuracy, PR-AUC
│  └─ ci.py                     single-split pipeline used only by the regression tests
├─ experiments/
│  ├─ ABTC_V2_Final.py          main experiment of the paper (Tables 4–6, Figs. 5–7)
│  └─ base_learner_experiment.py
├─ ablation/                    revision experiments (Supplementary S1–S5)
│  ├─ ABTC_Ablation_Final.ipynb
│  ├─ abtc_ablation_final.py    same code as the notebook, as a script
│  └─ ABTC_ablation_summary.xlsx
├─ leakage/                     single-feature leakage sensitivity (Supplementary S7)
│  ├─ leakage_sensitivity.py
│  ├─ main_results_long.csv
│  └─ leakage_sensitivity_results.xlsx
├─ config/
│  ├─ search_config.yml         ABTC grid
│  ├─ cost_datasets.csv         the 17 Path B datasets and their cost ratios
│  ├─ metrics_policy.csv        primary metric per dataset for the regression tests
│  ├─ metrics_defaults.yml
│  └─ test_datasets.csv         target columns for the regression tests
├─ data/
│  └─ dataset_catalogue.csv     all datasets: data file, N, d, class counts, source, licence
├─ results/                     main-experiment results (per metric family, cost-driven, statistics)
├─ figures/                     figures of the paper
├─ notebooks/                   quick start and a synthetic pipeline demonstration
├─ scripts/                     generate_golden.py, run_main_experiment.py, nb_normalize.py
├─ tests/                       unit tests and the end-to-end regression test
├─ env/abtc_env.yml             Conda environment
├─ CHANGELOG.md
└─ CITATION.cff
```

---

## Installation

```bash
conda env create -f env/abtc_env.yml
conda activate abtc
python -m pip install -e .
python -c "import abtc; print(abtc.__version__)"
```

---

## Quick start

```python
import numpy as np
from abtc import alpha_adjust_posterior, joint_alpha_tau_search, joint_alpha_tau_search_fast, effective_threshold, fbeta

# p_val: calibrated validation probabilities, y_val: labels, pi: positive-class proportion in training
alpha, tau, score, n_eval = joint_alpha_tau_search(
    y_val, p_val, pi, adjust_fn=alpha_adjust_posterior,
    metric_fn=lambda yt, yp: fbeta(yt, yp, beta=2.0))          # 540 candidates

alpha, tau, score, n_eval = joint_alpha_tau_search_fast(y_val, p_val, pi, metric="F2")   # same candidates, vectorised
t = effective_threshold(alpha, tau, pi)                          # equivalent threshold on p
y_hat = (p_test >= t).astype(int)
```

Path B:

```python
from abtc import alpha_search_analytical_tau
alpha, tau_star, neg_cost, n_eval = alpha_search_analytical_tau(
    y_val, p_val, pi, adjust_fn=alpha_adjust_posterior, cost_fn=10.0, cost_fp=1.0,
    metric_fn=lambda yt, yp: -(1.0 * ((yt == 0) & (yp == 1)).sum() + 10.0 * ((yt == 1) & (yp == 0)).sum()))
```

---

## Reproducing the paper

**Main experiment** (Tables 4–6): `experiments/ABTC_V2_Final.py` runs one dataset per call. Set the data file in `load_dataset()`, the target column and `PRIMARY_METRIC` (or `COST_FP` / `COST_FN` for a cost dataset) at the top of the script, then run it (`python scripts/run_main_experiment.py` runs it and stops with an error if the run fails). The primary metric of each dataset is given by the per-metric result files in `results/result_metricwise/`; the cost datasets and ratios are in `config/cost_datasets.csv`. Protocol: 10-fold stratified CV (`random_state = 42`), a stratified 20 % validation split inside each training fold (`random_state = 42 + fold`), `CalibratedClassifierCV(LogisticRegression(liblinear, max_iter = 1000), cv = 3, method = 'isotonic')`. In this script the features are standardised once on the whole dataset before the CV split. GHOST, OTLP and OECV are simplified re-implementations on a common base model; their settings are listed in Supplementary Table S9.

**Ablation and revision experiments** (Supplementary S1–S5): open `ablation/ABTC_Ablation_Final.ipynb`, set the data folder in the configuration cell and run all cells. Finished datasets are cached, so an interrupted run can be restarted. The published output is `ablation/ABTC_ablation_summary.xlsx` (configuration hash b1ed3987). Unlike the main experiment, this code fits imputation and scaling inside each training subset (`SCALE_WITHIN_FOLD = True`). The ablation results are not a re-run of the main experiment and do not replace its numbers.

**Leakage sensitivity** (Supplementary Tables S12–S14): `cd leakage && python leakage_sensitivity.py`. It reads `main_results_long.csv` and needs no data files.

**Synthetic demonstration:** `notebooks/02_Synthetic_Pipeline_Demo.ipynb` runs the pipeline on synthetic data generated with `make_classification`; it does not reproduce the paper's tables.

---

## Data

`data/dataset_catalogue.csv` lists every dataset with its source, licence and the data file used in the experiments. N, d, target and class counts are those of the files used, after preprocessing. For 19 datasets (`resampled_from_source = True`) the files are resampled versions of the source data with a different class ratio. The columns `single_feature_auc_1`, `wider_leakage_set` and `duplicate_pair` mark the datasets discussed in Supplementary S7. Three catalogue entries were not among the 175 data files of the ablation runs (`in_ablation_runs = False`).

The source datasets are available from the repositories listed in the catalogue (UCI, Kaggle, Zenodo, Mendeley Data, GitHub, Zindi, GeeksforGeeks, ICPSR). Data obtained under a restricted-use agreement (ICPSR/NAHDAP) cannot be redistributed.

---

## Tests

```bash
pytest -q
```

Unit tests cover the threshold rules, the prior adjustment (including the effective-threshold identity) and the vectorised search. The end-to-end regression test runs only when CSV files are placed in `tests/data/`: create the reference file once with `python scripts/generate_golden.py`, then run `pytest`. This test uses a single split (`src/abtc/ci.py`) and does not reproduce the paper tables.

---

## Licence and citation

MIT licence (see `LICENSE`). Please cite the paper and this software as described in `CITATION.cff`.
