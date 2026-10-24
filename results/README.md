# Results

Outputs of the main experiment (`experiments/ABTC_V2_Final.py`).

- `result_metricwise/All dataset result of <metric> metric.xlsx`: mean, standard deviation and median of every method on every dataset whose primary metric is F2, F1, MCC or balanced accuracy. These files define the primary metric of each dataset.
- `statistical_result/Overall Friedman Result.xlsx`: Friedman average ranks per domain and metric.
- `cost_sensitive/cost_sensitive_results.csv` (and `.xlsx`): total misclassification cost (mean and standard deviation over 10 folds) of ABTC Path B, ABTC Path A (with the dataset's primary metric), cost-sensitive SVM and Balanced RF on the cost datasets. It has 18 rows; `in_main_table6 = False` marks Haberman Survival, which is not in Table 6 of the paper (17 datasets). It replaces the earlier `Cost-Sensitive_Result_Table.xlsx`.
- `base_learner_result/`: results of `experiments/base_learner_experiment.py`.

Results of the revision experiments are in `../ablation/ABTC_ablation_summary.xlsx` and `../leakage/leakage_sensitivity_results.xlsx`.
