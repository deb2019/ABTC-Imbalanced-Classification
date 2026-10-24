# Known change of class prior

`ABTC_PriorShift.ipynb` (and the identical script `abtc_priorshift.py`) produces Supplementary Section S5 (Tables S8–S9).

Test folds are resampled by class to a new positive rate π′ = m·π₁ (m = 0.25, 0.5, 1, 2, 4; five repeats), which is pure label shift. On the same calibrated probabilities five rules are compared by misclassification cost per test case: R0 Bayes rule ignoring the shift (α = 0); R1 ABTC with α set from the known π′ (α = 1 − logit π′ / logit π₁); R2 cost-minimising threshold tuned on the unshifted validation split; R3 ABTC Path B with α tuned on validation; R4 a threshold re-tuned on validation data resampled to π′ (needs labels from the new population).

Data loading, preprocessing and the calibrated model are those of `../ablation`. Put the notebook in the same folder as the ablation notebook (or set `DATA_DIRS`) and run all cells; results are cached per dataset. The published output is `ABTC_priorshift_summary.xlsx` (configuration hash d509de80).
