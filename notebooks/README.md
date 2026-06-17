# Notebooks

- `01_ABTC_Quickstart.ipynb`
- `02_Reproduce_Public_Results.ipynb`
- `03_Build_Figures.ipynb`


### Reproducibility levels
- **Fast CI check (seconds–minutes):** `python scripts/generate_golden.py`
  Produces `tests/golden/expected_ci.json` on small single-split runs. Purpose: catch regressions. 
  Not intended to match paper tables.
- **Paper reproduction (hours):** run `notebooks/02_Reproduce_Public_Results.ipynb`
  Implements KFOLD=10, RANDOM_STATE=42, 3-fold isotonic calibration in-train, and the exact 
  τ-selection policy (metric vs cost). Outputs match the paper within tolerance.

