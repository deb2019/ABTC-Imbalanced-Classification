"""
Single-feature leakage screen: sensitivity of the main comparisons (Supplementary Tables S14-S16).

Input : main_results_long.csv  - mean score of every method on every dataset of the main experiment
        (columns ds, clf, mean, std, metric; metric = the dataset's primary metric).
Output: leakage_sensitivity_results.xlsx and a printed summary.

No model is retrained. The flagged datasets are those of Supplementary Table S13:
  S1    - a single feature with ROC-AUC = 1.000 (max(AUC, 1-AUC), four decimals)
  WIDER - two further datasets with a single-feature AUC above 0.99 and a target-related feature name
Wilcoxon p-values use the normal approximation with tie correction (zero differences discarded).
"""
import re
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare, rankdata

CLF = ['ABTC', 'OTLP', 'OECV', 'Balanced RF', 'q*-Classifier', 'PAN', 'GHOST', 'Cost-Sens. SVM', 'NBC (0.5)']
S1 = ['Audit risk Prediction', 'Adult Autism Prediction', 'Prediction of Credit Card High and Low Transactions',
      'Credit Risk Prediction', 'Prediction of Drug Overdose', 'Prediction of Significance of Earthquake',
      'Electrical Grid Stability dataset predicting Stable or Unstable', 'Healthcare dataset Prediction of Adherence',
      'Maintenance Predictive For Industrial Machines', 'Predicting Income on census dataset',
      'Prediction of high value trade', 'Company Market Value Prediction', 'Google Play Store Reviews',
      'Predicting High value customer in ecommerce', 'Letterbox Movie Classification Dataset',
      'Netflix Movie Recommendation', 'Stroke Risk', 'High-Speed Train Bogie Vibration & Fault Diagnosis',
      'Predicting Low and high Startup Funding', 'Depression Prediction', 'Prediction of High Admission']
WIDER = ['Loan Fraud Prediction', 'Predicting fraudulent Loan application',
         'Whether the client subscribed to a term deposit']
SETTINGS = {'All': [], 'Without S1': S1, 'Without wider set': S1 + WIDER}
FAMILIES = ['F2', 'F1', 'MCC', 'BalancedAcc']


def norm(s):
    return re.sub(r'[^a-z0-9]', '', str(s).lower())


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); n = len(p); adj = np.empty(n); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (n - r) * p[i])); adj[i] = run
    return adj


def wilcoxon_p(d):
    d = np.asarray(d, float)
    if not np.any(np.abs(d) > 1e-12):
        return 1.0
    return float(wilcoxon(d, zero_method='wilcox', method='approx').pvalue)


def avg_ranks(T):
    return pd.Series(np.array([rankdata(-r) for r in T.values]).mean(0), index=T.columns)


R = pd.read_csv('main_results_long.csv')
W = R.pivot_table(index=['metric', 'ds'], columns='clf', values='mean')[CLF]
for lst in (S1, WIDER):
    missing = [x for x in lst if x not in set(R.ds)]
    assert not missing, missing

# ---- Table S16: within each primary-metric family (each dataset once per family) ----
fam_rows, fam_tests = [], []
for m in FAMILIES:
    Tm = W.xs(m, level='metric')
    for sc, ex in SETTINGS.items():
        T = Tm[~Tm.index.isin(ex)].dropna()
        rk = avg_ranks(T)
        for c in CLF[1:]:
            d = (T['ABTC'] - T[c]).values
            fam_tests.append(dict(family=m, setting=sc, baseline=c, n=len(T), wins=int((d > 1e-9).sum()),
                                  ties=int((np.abs(d) <= 1e-9).sum()), losses=int((d < -1e-9).sum()),
                                  p=wilcoxon_p(d)))
        fam_rows.append(dict(family=m, setting=sc, n=len(T), ABTC_rank=round(rk['ABTC'], 2),
                             ABTC_position=int((rk < rk['ABTC']).sum()) + 1,
                             friedman_p=friedmanchisquare(*T.values.T).pvalue))
ft = pd.DataFrame(fam_tests)
ft['p_holm_family'] = ft.groupby(['family', 'setting']).p.transform(lambda s: holm(s.values))
ft['p_holm_all32'] = ft.groupby('setting').p.transform(lambda s: holm(s.values))
fr = pd.DataFrame(fam_rows)
fr['beaten_holm_family'] = [int((ft[(ft.family == r.family) & (ft.setting == r.setting)].p_holm_family < .05).sum())
                            for r in fr.itertuples()]
fr['beaten_holm_all32'] = [int((ft[(ft.family == r.family) & (ft.setting == r.setting)].p_holm_all32 < .05).sum())
                           for r in fr.itertuples()]

# ---- Tables S12-S13: exploratory, each dataset name counted once (normalised names) ----
Wd = W.copy(); Wd['key'] = [norm(ds) for _, ds in Wd.index]
ds_rank_rows, ds_tests = [], []
for sc, ex in SETTINGS.items():
    exk = {norm(x) for x in ex}
    T = Wd[~Wd.key.isin(exk)].dropna()
    ranks = pd.DataFrame([rankdata(-r) for r in T[CLF].values], columns=CLF, index=T.key).groupby(level=0).mean()
    means = T.groupby('key')[CLF].mean()
    rk = ranks.mean()
    ds_rank_rows.append(dict(setting=sc, n=len(ranks), friedman_p=friedmanchisquare(*ranks.values.T).pvalue,
                             **{c: round(rk[c], 2) for c in CLF}))
    rows = []
    for c in CLF[1:]:
        d = (means['ABTC'] - means[c]).values
        rows.append(dict(setting=sc, baseline=c, n=len(d), wins=int((d > 1e-9).sum()),
                         ties=int((np.abs(d) <= 1e-9).sum()), losses=int((d < -1e-9).sum()), p=wilcoxon_p(d)))
    df = pd.DataFrame(rows); df['p_holm'] = holm(df.p); ds_tests.append(df)
dr = pd.DataFrame(ds_rank_rows); dt = pd.concat(ds_tests)

pd.set_option('display.width', 220)
print('Table S14 (ranks, dataset level)\n', dr.to_string(index=False))
print('\nTable S15 (Wilcoxon, dataset level)\n', dt.round(4).to_string(index=False))
print('\nTable S16 (per family)\n', fr.to_string(index=False))
with pd.ExcelWriter('leakage_sensitivity_results.xlsx') as w:
    dr.to_excel(w, sheet_name='S12_ranks_dataset_level', index=False)
    dt.to_excel(w, sheet_name='S13_wilcoxon_dataset_level', index=False)
    fr.to_excel(w, sheet_name='S14_family_summary', index=False)
    ft.to_excel(w, sheet_name='S14_family_tests', index=False)
    pd.DataFrame({'dataset': S1 + WIDER, 'set': ['S1'] * len(S1) + ['wider'] * len(WIDER)}).to_excel(
        w, sheet_name='flagged_datasets', index=False)
