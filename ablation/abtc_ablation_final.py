# # ABTC — Ablation & Revision Experiments (Discover AI, major revision)
# 
# One notebook for the reviewer questions that need new numbers:
# 
# | Reviewer point | What this notebook produces |
# |---|---|
# | **R2-Q2** Why joint (α, τ) instead of one threshold? | ABTC vs equal-budget 1-D threshold search, equivalence check, TOST equivalence test, ridge figure |
# | **R2-Q3** When is the cost threshold Bayes-optimal after adjustment? | Path B: tuned α vs pure Bayes rule (α = 0), calibrated vs uncalibrated |
# | **R2-Q4 / R1-3b** Computational cost | Sequential timing benchmark: original 540-loop vs fast implementation vs 1-D search; model fit time |
# | **R2-Q1** Where does the gain come from? | Calibrated vs the *same models* uncalibrated (A7, A6), and vs a single LR (A8, OTLP-like) |
# | **R1-3c** Calibration assumption | Bayes rule on calibrated vs uncalibrated posteriors (B2 vs B4); Brier score / log loss |
# 
# **Key identity.** logit(P_adj) = logit(P) + α·L with L = log((1−π₁)/π₁), so
# P_adj ≥ τ ⇔ P ≥ t(α, τ) = σ(logit τ − α·L). Every (α, τ) pair is one effective threshold.
# 
# ### Configurations (all share the same fitted models in each fold — only the threshold rule differs)
# 
# **Path A (metric-driven)** — run for F1, F2, G-Mean, MCC, Balanced Accuracy:
# 
# | ID | Probabilities | Threshold rule | Candidates |
# |---|---|---|---|
# | A1 | calibrated (isotonic, 3 models) | **ABTC Path A**, joint (α, τ) grid | 27 × 20 = 540 |
# | A2 | calibrated | τ only, α = 0, same 20 τ values | 20 |
# | A3 | calibrated | τ only, uniform grid, equal budget | 540 |
# | A4 | calibrated | τ only over the 540 *induced* thresholds — **must equal A1** | 540 |
# | A5 | calibrated | τ only, exhaustive (every distinct score + "all negative") | n_val + 1 |
# | A6 | **same 3 models, uncalibrated** (average) | τ only, uniform grid | 540 |
# | A7 | **same 3 models, uncalibrated** | ABTC Path A, joint (α, τ) grid | 540 |
# | A8 | **single LR** on the whole training part (OTLP/GHOST base model) | τ only, uniform grid | 540 |
# 
# A1 vs A7 and A3 vs A6 isolate **calibration alone** (identical base models); A6 vs A8 isolates the
# 3-model averaging. Test-fold Brier score and log loss are reported for the three probability types.
# 
# **Path B (cost-driven)** — only for datasets in `COST_DATASETS`, cost C_FP : C_FN = 1 : r:
# 
# | ID | Probabilities | Rule |
# |---|---|---|
# | B1 | calibrated | **ABTC Path B** as in the paper: τ* = 1/(1+r), α tuned over [0.1, 3.0] |
# | B2 | calibrated | Pure Bayes rule: α = 0, τ* = 1/(1+r), nothing tuned |
# | B3 | calibrated | τ only, exhaustive cost-minimising threshold (incl. "all negative") |
# | B4 | same 3 models, uncalibrated | Pure Bayes rule, α = 0 |
# | B5 | calibrated | **Revised Path B**: α grid that also contains α = 0 |
# 
# **Statistics.** Mean difference with 95 % bootstrap CI; Wilcoxon signed-rank (location of the
# differences); Holm within each table *and* across all reported Path A tests; equivalence of the **mean**
# difference within ±`EQUIV_MARGIN` by a mean-based TOST (paired t) together with the 90 % bootstrap CI.
# 
# **Protocol** as in `ABTC_V2_Final.py`: 10-fold stratified CV (`random_state=42`), 20 % stratified
# validation split (`random_state=42+fold`), `CalibratedClassifierCV(LogisticRegression(liblinear,
# max_iter=1000), cv=3, method='isotonic')`. By default, median imputation and standard scaling are
# inside a Pipeline, so they are re-fit on every training subset (also inside the calibration CV).
# 
# **How to use:** edit the configuration cell (section 1), then *Run All*. Results are saved per
# dataset in a folder tied to the current settings, so if the run stops, run again — finished datasets
# with the same settings and unchanged data are skipped.

# %%
# ## 1. Configuration — edit only this cell

# %%
# Folders searched for the CSV files, in this order ('.' = the folder this notebook is in).
# A file is taken from the first folder where it is found. Add more folders if needed,
# e.g. r'C:\Users\user\ABTC_data'.
DATA_DIRS = [r'.', r'./datasets']
OUT_DIR  = r'./ABTC_ablation_results'

# ---------------------------------------------------------------------------------------------
# DATASETS: ('file.csv', 'target column')   — add / remove lines freely.
# Optional 3rd item = primary metric for that dataset, e.g. ('pima_diabetes.csv', 'Outcome', 'F2').
# If no 3rd item is given, DEFAULT_PRIMARY_METRIC is used. Allowed: 'F1','F2','G-Mean','MCC','BalancedAcc'
# Duplicate file names are skipped automatically.
# ---------------------------------------------------------------------------------------------
DATASETS = [
    # ---------------- Medical ----------------
    ('Adult_Depression_Preprocessed.csv',                   'Label'),
    ('autism_adult_binary_preprocessed.csv',                'ASD'),
    ('autism_child_binary_preprocessed.csv',                'ASD'),
    ('transfusion_binary_preprocessed.csv',                 'Donated'),
    ('bone_marrow_preprocessed.csv',                        'survival_status'),
    ('Breast_Cancer_Preprocessed.csv',                      'Status'),
    ('CTG_preprocessed_scaled.csv',                         'NSP'),
    ('cervical_cancer.csv',                                 'Biopsy'),
    ('CKD_Preprocessed.csv',                                'Target'),
    ('ckd_binary_preprocessed_all_scaled.csv',              'class'),
    ('cirrhosis_preprocessed_binary.csv',                   'Status'),
    ('Diabetes_Preprocessed.csv',                           'Diabetes'),
    ('preprocessed_thyroid_data.csv',                       'Recurred'),
    ('DiseaseDiagnosis_Preprocessed.csv',                   'has_disease'),
    ('DIA_combined_preprocessed.csv',                       'Label'),
    ('haberman_preprocessed_final.csv',                     'Target'),
    ('Health_Diabetes_Preprocessed_Imbalanced.csv',         'Label'),
    ('Preprocessed_Healthcare_Adherence_Imbalanced.csv',    'Label'),
    ('Healthy_Lifestyle_Imbalanced.csv',                    'Label'),
    ('Heart_Disease_Preprocessed_Imbalanced.csv',           'Label'),
    ('hepatitis_preprocessed.csv',                          'class'),
    ('HCV_data_imb.csv',                                    'Target'),
    ('LengthOfStay_Binary_Imbalanced_50K.csv',              'Label'),
    ('ICU_Death_Prediction_Preprocessed.csv',               'target'),
    ('Malaria_preprocessed_scaled.csv',                     'Label'),
    ('InfraredThermo_Preprocessed.csv',                     'is_fever'),
    ('nhs_highly_imbalanced_scaled.csv',                    'Target_High_Admission'),
    ('ovarian_dataset_smaller.csv',                         'RiskLabel'),
    ('parkinsons_highly_imbalanced.csv',                    'status'),
    ('PCOS_Preprocessed.csv',                               'PCOS'),
    ('preprocessed_primary_tumor.csv',                      'Target_Lung'),
    ('imbalanced_depression_dataset.csv',                   'Target_Depression'),
    ('imbalanced_schizophrenia_dataset.csv',                'Target_Schizophrenia'),
    ('appendicitis_preprocessed_binary.csv',                'Severity'),
    ('preprocessed_stunting_sampled_10k.csv',               'Stunting_binary'),
    ('support2_preprocessed_binary.csv',                    'target'),
    ('Thoracic_Surgery_Preprocessed.csv',                   'Label'),
    ('preprocessed_diabetes_dataset_imbalanced.csv',        'Outcome'),
    ('imbalanced_typhoid_hospitalized.csv',                 'Hospitalized'),
    ('pre_alzheimers_disease.csv',                          'Diagnosis'),
    ('pre_anxiety_depression_Imb.csv',                      'Chronic_Illnesses'),
    ('prepro_cancer.csv',                                   'Diagnosis'),
    ('preprocessed_Disease_Symptom.csv',                    'Outcome Variable'),
    ('pre_lung_cancer.csv',                                 'PULMONARY_DISEASE'),
    ('pre_ovarian_cancer_binary_16k.csv',                   'RiskLabel_Binary'),
    ('pre_stroke_risk.csv',                                 'At_Risk'),
    ('pima_diabetes.csv',                                   'Outcome'),

    # ---------------- Fraud Detection ----------------
    ('Bank_Transactions_Preprocessed.csv',                  'IsFraud'),
    ('preprocessed_credit_fraud.csv',                       'is_fraud'),
    ('CreditCard_50000_492.csv',                            'Class'),
    ('Fraud_Transaction_With_Synthetic_IsFraud.csv',        'IsFraud'),
    ('Final_Healthcare.csv',                                'PotentialFraud'),
    ('Shill_Bidding_Preprocessed.csv',                      'Label'),
    ('preprocessed_airbnb_fraud.csv',                       'fraud'),
    ('Preprocessed_Bank_Fraud.csv',                         'targets'),
    ('Preprocessed_Fraud_Train.csv',                        'Class'),
    ('preprocessed_training_data.csv',                      'fraud'),
    ('Preprocessed_Fraud_Insurance.csv',                    'fraud_reported'),
    ('preprocessed_insu_response.csv',                      'Response'),
    ('Pre_Loan_Fraud_Dataset.csv',                          'Fraudulent'),

    # ---------------- Cyber Security ----------------
    ('AndroidAdware_Binary_Preprocessed_Final.csv',         'Label'),
    ('ClaMP_Imbalanced.csv',                                'class'),
    ('AndroidMalware_Imbalanced.csv',                       'Result'),
    ('imbalanced_malware_dataset.csv',                      'Target'),
    ('Network_Anomaly_preprocessed.csv',                    'Label'),
    ('Cybersecurity_WebAttack_Preprocessed_30k.csv',        'Label'),
    ('phishing_imbalanced_40k.csv',                         'label'),
    ('rt_iot_binary_preprocessed_5037.csv',                 'target'),
    ('TUANDROMD_Preprocessed.csv',                          'Label'),
    ('cyber_class_pre.csv',                                 'Target'),
    ('prep_Cyber.csv',                                      'sus'),
    ('pre_link_phishing_imb.csv',                           'status'),
    ('pre_healthcare_ransomware.csv',                       'data_stolen'),
    ('pre_phishing_imb.csv',                                'Result'),
    ('pre_phishing_50Kdata.csv',                            'Phising'),
    ('Phishing_Imb_Pre.csv',                                'label'),
    ('teen_cyber_risk_pre_reduce.csv',                      'Target'),

    # ---------------- Socio ----------------
    ('adult_preprocessed.csv',                              'income'),
    ('AQI_RespDeath_Binary_Imbalanced.csv',                 'Label'),
    ('tomshardware_imb.csv',                                'buzz_label'),
    ('communities_binary_imbalanced.csv',                   'CrimeLevel'),
    ('Crime_imb_preprocessed.csv',                          'HighCrime'),
    ('final_binary_corrected_contraceptive.csv',            'HighUse'),
    ('cannabis_preprocessed.csv',                           'Cannabis_Used'),
    ('Earthquake_Preprocessed.csv',                         'is_significant'),
    ('DrugOverdose_Preprocessed.csv',                       'high_risk'),
    ('Mental_Health_Workforce_imb.csv',                     'Label'),
    ('Himalaya_Expedition.csv',                             'death'),
    ('IncomeClassification_Preprocessed.csv',               'income'),
    ('AIDRUG_Data.csv',                                     'Q51C'),
    ('StartupFunding_Preprocessed.csv',                     'is_high_funding'),
    ('preprocessed_small_dataset.csv',                      'Target_Short_Walk'),
    ('MentalHealth_Imb.csv',                                'mental_health_issue'),
    ('News_Aggregator_10k_Imbalanced.csv',                  'Label'),
    ('Student_Info_Imbalanced.csv',                         'Label'),
    ('Overstimulation_Imb.csv',                             'Overstimulated'),
    ('ozone_preprocessed.csv',                              'target'),
    ('page_blocks_imbalanced_Preprocessed.csv',             'Target'),
    ('rice_binary_more_imbalanced.csv',                     'Class'),
    ('SmokingCancer_Preprocessed.csv',                      'has_cancer'),
    ('preprocessed_lung_cancer_dataset.csv',                'Target_Cancer'),
    ('SocialMediaEngagement_Preprocessed.csv',              'engaged'),
    ('stress_detection_more_imbalanced.csv',                'Stress_Binary'),
    ('Traffic_Flow_Top100_Features.csv',                    'target'),
    ('preprocessed_tree_health_dataset.csv',                'Health_Status'),
    ('preprocessed_us_census_income_imbalanced.csv',        'High_Income'),
    ('vk_more_imbalanced_scaled_cleaned.csv',               'target'),
    ('bankruptcy_company_pre.csv',                          'Bankrupt'),
    ('personality_processed.csv',                           'Target'),
    ('preprocessed_flood.csv',                              'FloodRisk'),
    ('flood_risk_imbalanced_pro.csv',                       'Flood_Occurred'),
    ('hotel_bookings_preprocessed.csv',                     'target'),
    ('personality_binary_preprocessed.csv',                 'target'),
    ('fetal_health_preprocessed.csv',                       'binary_target'),

    # ---------------- Predictive Maintenance ----------------
    ('ai4i2020_preprocessed.csv',                           'Machine failure'),
    ('Annealing_Binary_Preprocessed.csv',                   'Label'),
    ('Helicopter_Anomaly_dataset.csv',                      'Lavel'),
    ('aps_failure_reduced_30k.csv',                         'class'),
    ('pop_failures_converted_positive_class.csv',           'outcome'),
    ('grid_stability_preprocessed.csv',                     'stabf_binary'),
    ('HTRU_2_preprocessed.csv',                             'Class'),
    ('ionosphere_preprocessed_corrected.csv',               'target'),
    ('MITBIH_ECG_Binary_Preprocessed.csv',                  'Label'),
    ('pre_train_fault.csv',                                 'condition'),
    ('pre_pm_sensor_imb.csv',                               'fail'),
    ('preprocessed_maintenance_data.csv',                   'Failures'),
    ('pre_conveyor_fault.csv',                              'Fault'),

    # ---------------- Finance and Marketing ----------------
    #   Credit Card Approval
    ('preprocessed_imbalanced_credit_data.csv',             'target'),
    ('Credit_data_cleaned_ML.csv',                          'Label'),
    ('Credit_preprocessed.csv',                             'Target'),
    ('CCCD_Prepro_IMB.csv',                                 'Class'),
    ('German_preproces.csv',                                'Category'),
    ('cc_app_imb.csv',                                      'ApprovalStatus'),
    ('crx_imb.csv',                                         'ApprovalStatus'),
    ('pre_credit_use.csv',                                  'Target'),
    #   Loan Approval
    ('preprocessed_loan_approval.csv',                      'Loan_Status'),
    ('preprocessed_loan.csv',                               'good_bad_flag'),
    ('pre_loan_data.csv',                                   'loan_status'),
    ('ready_loan_data.csv',                                 'loan_status'),
    ('pre_Loan_dataApp.csv',                                'Loan_Status'),
    ('preprocessed_default_fin.csv',                        'Defaulted'),
    #   Insurance
    ('pre_IMB_Insu.csv',                                    'insuranceclaim'),
    ('pre_ins_fraud.csv',                                   'fraud_reported'),
    ('caravan_scaled_pre.csv',                              'CARAVAN'),
    ('pre_ins_imb.csv',                                     'insuranceclaim'),
    #   Marketing and Churn Prediction
    ('pre_bank_marketing.csv',                              'success'),
    ('clickstream_binary_imb.csv',                          'purchase_proxy'),
    ('pre_telco_churn.csv',                                 'Churn'),
    ('pre_customer_churn.csv',                              'Churn'),
    ('customer_churn_preprocessed.csv',                     'Churn'),
    ('online_shoppers_preprocessed.csv',                    'Revenue'),
    #   Finance Other
    ('audit_risk_preprocessed.csv',                         'Risk'),
    ('pre_binancecoin_binary.csv',                          'target'),
    ('ad_preprocessed.csv',                                 'target'),
    ('NIFTY500_Imbalanced.csv',                             'target'),
    ('pre_high_value_cust.csv',                             'HighValueCustomer'),
    ('prepro_trade_invest.csv',                             'High_Investment'),
    ('TurkishCF_Preprocessed.csv',                          'Label'),
    ('preprocessed_financial_distress.csv',                 'Distress'),
    ('preprocessed_financial_budget.csv',                   'HasBudget'),
    ('preprocessed_income_dataset.csv',                     'HighIncome'),
    # ('preprocessed_financial_distress.csv', 'Distress'),   # duplicate entry — removed
    ('preprocessed_student_overspending.csv',               'VeryHighOverspend'),
    ('pre_companies_marketvalue.csv',                       'HighMarketValue'),

    # ---------------- Recommendation ----------------
    ('drug_reviews.csv',                                    'label'),
    ('imdb_final_preprocessed.csv',                         'target'),
    ('tripadvisor_review_preprocessed.csv',                 'target'),
    ('prep_airlines_reviews.csv',                           'Target'),
    ('pca_23_review_data.csv',                              'sentiment'),
    ('pre_airline_dataset.csv',                             'Target'),
    ('pre_market_comments.csv',                             'Target'),
    ('Pre_Netflix_Rating.csv',                              'Target'),
    ('pre_google_play_reviews.csv',                         'Target'),
    ('pre_letterbox_dataset.csv',                           'Target'),
    ('pre__music_dataset.csv',                              'Target'),
    ('pre_womens_clothing_data.csv',                        'Recommended'),
]

DEFAULT_PRIMARY_METRIC = 'F2'

# ---------------------------------------------------------------------------------------------
# COST_DATASETS (Path B, Table 6): 'file.csv': r   means   C_FP : C_FN = 1 : r
# File names verified against the dataset catalogue (old appendix Table A1).
# ---------------------------------------------------------------------------------------------
COST_DATASETS = {
    'pima_diabetes.csv':                  5,   # PIMA Indian
    'HCV_data_imb.csv':                  10,   # Hepatitis C
    'cirrhosis_preprocessed_binary.csv': 25,   # Cirrhosis Patient
    'preprocessed_airbnb_fraud.csv':      5,   # Airbnb Fraud Detection
    'CreditCard_50000_492.csv':          10,   # Credit Card Fraud
    'Preprocessed_Fraud_Insurance.csv':  15,   # Insurance Fraud
    'AndroidMalware_Imbalanced.csv':      5,   # Android Malware
    'Phishing_Imb_Pre.csv':               5,   # Malicious URL Detection (Predicting Malicious URL)
    'rt_iot_binary_preprocessed_5037.csv': 3,  # Network Traffic Samples
    'rice_binary_more_imbalanced.csv':    3,   # Rice Cammeo and Osmancik
    'cannabis_preprocessed.csv':          5,   # Cannabis Addiction
    'DrugOverdose_Preprocessed.csv':     10,   # Drug Overdose
    'SmokingCancer_Preprocessed.csv':    25,   # Smoking and Cancer Risk
    'aps_failure_reduced_30k.csv':        2,   # Air Pressure Failure
    'HTRU_2_preprocessed.csv':            3,   # Anomaly Pulser System
    'ai4i2020_preprocessed.csv':          5,   # Machine Failure (CNC)
    'prep_airlines_reviews.csv':          3,   # Airline Reviews
}

# ---------------------------------------------------------------------------------------------
# Experiment settings (defaults reproduce the paper's ABTC configuration)
# ---------------------------------------------------------------------------------------------
KFOLD, RANDOM_STATE, VAL_SIZE = 10, 42, 0.20
N_ALPHA, N_TAU = 27, 20
ALPHA_RANGE, TAU_RANGE = (0.1, 3.0), (0.01, 0.99)
METRICS = ['F1', 'F2', 'G-Mean', 'MCC', 'BalancedAcc']

SCALE_WITHIN_FOLD  = True    # True: imputation + scaling inside a Pipeline, fit on training data only (no leakage).
                             # False: global imputation + scaling before CV, exactly as in the original code
N_JOBS             = -1      # parallel datasets; set 1 if you get memory errors
TIME_ORIGINAL_LOOP = True    # time the original sklearn 540-evaluation loop (adds ~20 s per dataset); False to skip
TIMING_METRIC      = 'F2'
LEAKAGE_CHECK      = True    # flag single features that almost perfectly predict the target
LEAKAGE_AUC        = 0.97    # ... i.e. a single feature with AUC ≥ this value
EQUIV_MARGIN       = 0.005   # equivalence margin for the MEAN difference (metric scale): ±0.005 = ±0.5 percentage points
N_BOOT             = 10000   # bootstrap resamples for confidence intervals
TIMING_BENCH_N     = 20      # datasets used for the dedicated SEQUENTIAL timing benchmark (section 5.5); 0 to skip
TIE_TOL            = 1e-12
RIDGE_FIGURE       = ('pima_diabetes.csv', 'Outcome', 'F2')   # dataset used for the ridge figure; None to skip

# %%
# ## 2. Imports and core functions

# %%
import os, time, glob, warnings, traceback
import numpy as np
import pandas as pd
from math import sqrt
from scipy.stats import wilcoxon, friedmanchisquare, rankdata, ttest_1samp
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.base import clone
import hashlib, json
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (precision_score, recall_score, f1_score, confusion_matrix,
                             matthews_corrcoef, balanced_accuracy_score, roc_auc_score,
                             brier_score_loss, log_loss)
from joblib import Parallel, delayed

warnings.filterwarnings('ignore')
pd.set_option('display.width', 220); pd.set_option('display.max_columns', 40)

DATA_DIRS = [os.path.abspath(os.path.expanduser(d)) for d in DATA_DIRS]

def locate(fname):
    for d in DATA_DIRS:
        p = os.path.join(d, fname)
        if os.path.isfile(p):
            return p
    return None
OUT_DIR  = os.path.abspath(os.path.expanduser(OUT_DIR))
CODE_VERSION = 'v4-2026-10-22'
# Every setting that changes results goes into a fingerprint; results live in folds_<fingerprint>,
# so results from different settings / code versions are never mixed.
_cfg = dict(code=CODE_VERSION, kfold=KFOLD, seed=RANDOM_STATE, val=VAL_SIZE, na=N_ALPHA, nt=N_TAU,
            ar=ALPHA_RANGE, tr=TAU_RANGE, metrics=METRICS, scale=SCALE_WITHIN_FOLD, tie=TIE_TOL)
CONFIG_HASH = hashlib.md5(json.dumps(_cfg, sort_keys=True, default=str).encode()).hexdigest()[:8]
FOLD_DIR = os.path.join(OUT_DIR, f'folds_{CONFIG_HASH}')
os.makedirs(FOLD_DIR, exist_ok=True)
with open(os.path.join(FOLD_DIR, 'config.json'), 'w') as fh:
    json.dump(_cfg, fh, indent=1, default=str)
print('Results folder for this configuration:', FOLD_DIR)
ALPHA_LIST = np.linspace(*ALPHA_RANGE, N_ALPHA)
ALPHA_LIST_B5 = np.concatenate([[0.0], ALPHA_LIST])
TAU_LIST   = np.linspace(*TAU_RANGE, N_TAU)
TAU_UNIF   = np.linspace(*TAU_RANGE, N_ALPHA * N_TAU)

# normalise the dataset list: drop duplicates, attach primary metric
_seen, DS = set(), []
for item in DATASETS:
    f, t = item[0], item[1]
    pm = item[2] if len(item) > 2 else DEFAULT_PRIMARY_METRIC
    if f in _seen:
        continue
    _seen.add(f); DS.append((f, t, pm))
PRIMARY = {f: pm for f, _, pm in DS}
print(f'{len(DS)} datasets ({len(DATASETS) - len(DS)} duplicates skipped) | '
      f'{sum(f in COST_DATASETS for f, _, _ in DS)} of them are Path B cost datasets')

# ---------- fast confusion counts and metrics ----------
def counts_at(p, y, thr):
    """TP, FP, FN, TN for predictions (p >= t) at every t in thr."""
    pos = np.sort(p[y == 1]); neg = np.sort(p[y == 0]); thr = np.atleast_1d(thr)
    TP = len(pos) - np.searchsorted(pos, thr, side='left')
    FP = len(neg) - np.searchsorted(neg, thr, side='left')
    return TP.astype(float), FP.astype(float), (len(pos) - TP).astype(float), (len(neg) - FP).astype(float)

def metric_from_counts(TP, FP, FN, TN, metric, cfp=1.0, cfn=10.0):
    """Same definitions as ABTC_V2_Final.py. Higher is better (cost is returned negated)."""
    with np.errstate(divide='ignore', invalid='ignore'):
        prec = np.where(TP + FP > 0, TP / (TP + FP), 0.0)
        rec  = np.where(TP + FN > 0, TP / (TP + FN), 0.0)
        spec = np.where(TN + FP > 0, TN / (TN + FP), 0.0)
        if metric == 'F1':
            d = 2 * TP + FP + FN; return np.where(d > 0, 2 * TP / np.where(d > 0, d, 1), 0.0)
        if metric == 'F2':
            d = 4 * prec + rec; return np.where(prec + rec > 0, 5 * prec * rec / np.where(d > 0, d, 1), 0.0)
        if metric == 'G-Mean':
            return np.sqrt(rec * spec)
        if metric == 'MCC':
            d = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
            return np.where(d > 0, (TP * TN - FP * FN) / np.where(d > 0, d, 1), 0.0)
        if metric == 'BalancedAcc':
            return 0.5 * (rec + spec)
        if metric == 'cost':
            return -(cfp * FP + cfn * FN)
    raise ValueError(metric)

def score_preds(y, yhat, metric, cfp=1.0, cfn=10.0):
    y = np.asarray(y); yhat = np.asarray(yhat)
    c = [np.array([float(np.sum((yhat == a) & (y == b)))]) for a, b in [(1, 1), (1, 0), (0, 1), (0, 0)]]
    v = float(metric_from_counts(*c, metric, cfp, cfn)[0])
    return -v if metric == 'cost' else v          # cost reported as a positive number

def first_best(scores, tol=TIE_TOL):
    """First maximum (= strict '>' loop of the original code) and how many candidates tie with it."""
    best = np.max(scores); tied = np.flatnonzero(scores >= best - tol)
    return int(tied[0]), int(len(tied)), float(best)

def adjust_posterior(p, alpha, pi):
    """Exactly the ABTC formula of ABTC_V2_Final.py (with its 1e-12 guards)."""
    num = p / (pi ** alpha + 1e-12)
    return num / (num + (1 - p) / ((1 - pi) ** alpha + 1e-12) + 1e-12)

def logit(x):
    x = np.clip(x, 1e-15, 1 - 1e-15); return np.log(x / (1 - x))

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def induced_threshold(alpha, tau, pi):
    """t = sigma(logit(tau) - alpha * log((1-pi)/pi))  — threshold on the unadjusted posterior."""
    return sigmoid(logit(tau) - alpha * np.log((1 - pi) / pi))

# %%
# ### Threshold-selection rules

# %%
def select_abtc_A(p_val, y_val, pi, metric):
    """ABTC Path A: alpha-major loop over (alpha, tau) on the adjusted posterior."""
    s = np.empty(N_ALPHA * N_TAU)
    for i, a in enumerate(ALPHA_LIST):
        s[i * N_TAU:(i + 1) * N_TAU] = metric_from_counts(*counts_at(adjust_posterior(p_val, a, pi), y_val, TAU_LIST), metric)
    k, n_tied, best = first_best(s)
    a, t = ALPHA_LIST[k // N_TAU], TAU_LIST[k % N_TAU]
    return dict(alpha=a, tau=t, t_eff=float(induced_threshold(a, t, pi)), n_tied=n_tied, val=best, grid=s)

def select_induced(p_val, y_val, pi, metric):
    A, T = np.meshgrid(ALPHA_LIST, TAU_LIST, indexing='ij')
    cand = induced_threshold(A, T, pi).ravel()
    k, n_tied, best = first_best(metric_from_counts(*counts_at(p_val, y_val, cand), metric))
    return dict(t_eff=float(cand[k]), n_tied=n_tied, val=best)

def select_grid(p_val, y_val, grid, metric, cfp=1.0, cfn=10.0):
    k, n_tied, best = first_best(metric_from_counts(*counts_at(p_val, y_val, grid), metric, cfp, cfn))
    return dict(t_eff=float(grid[k]), n_tied=n_tied, val=best)

def select_exhaustive(p_val, y_val, metric, cfp=1.0, cfn=10.0):
    """Every distinct validation score as a cut (p >= cut), PLUS +inf = predict everything negative.
    All-positive is the lowest distinct score. The chosen cut is moved to the midpoint with the next
    lower distinct score so that test points are not placed exactly on the boundary."""
    u = np.unique(p_val)
    cand = np.append(u, np.inf)
    k, n_tied, best = first_best(metric_from_counts(*counts_at(p_val, y_val, cand), metric, cfp, cfn))
    if k == len(u):
        t = np.inf
    else:
        t = u[k] if k == 0 else 0.5 * (u[k] + u[k - 1])
    return dict(t_eff=float(t), n_tied=n_tied, val=best)

def select_abtc_B(p_val, y_val, pi, cfp, cfn, alphas):
    tau = cfp / (cfp + cfn)
    s = np.array([metric_from_counts(*counts_at(adjust_posterior(p_val, a, pi), y_val, [tau]), 'cost', cfp, cfn)[0]
                  for a in alphas])
    k, n_tied, best = first_best(s)
    return dict(alpha=alphas[k], tau=tau, t_eff=float(induced_threshold(alphas[k], tau, pi)), n_tied=n_tied, val=best)

# ---------- original sklearn loop (copied from ABTC_V2_Final.py): timing + fidelity check ----------
def _f2_orig(y_true, y_pred):
    prec = precision_score(y_true, y_pred, zero_division=0); rec = recall_score(y_true, y_pred)
    return 0.0 if (prec + rec) == 0 else 5 * prec * rec / (4 * prec + rec)

def _gmean_orig(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape != (2, 2): return 0.0
    TN, FP, FN, TP = cm.ravel()
    return sqrt((TP / (TP + FN) if TP + FN else 0.0) * (TN / (TN + FP) if TN + FP else 0.0))

_ORIG = {'F1': lambda a, b: f1_score(a, b, zero_division=0), 'F2': _f2_orig, 'G-Mean': _gmean_orig,
         'MCC': matthews_corrcoef, 'BalancedAcc': balanced_accuracy_score}

def select_original_loop(p_val, y_val, pi, metric):
    best, ba, bt = -np.inf, None, None
    for a in ALPHA_LIST:
        p_adj = adjust_posterior(p_val, a, pi)
        for t in TAU_LIST:
            s = _ORIG[metric](y_val, (p_adj >= t).astype(int))
            if s > best: best, ba, bt = s, a, t
    return ba, bt

# %%
# ## 3. Data loading and the per-dataset runner

# %%
def load_xy(fname, target):
    """Reads the CSV and encodes it. Missing values are KEPT (imputed later inside the fold
    pipeline) unless SCALE_WITHIN_FOLD=False, which reproduces the original global preprocessing."""
    path = locate(fname)
    if path is None:
        raise FileNotFoundError(f'{fname} not found in {DATA_DIRS}')
    df = pd.read_csv(path, low_memory=False)
    if target not in df.columns:
        raise KeyError(f"target column '{target}' not found")
    y = pd.to_numeric(df[target], errors='coerce')
    keep = y.notna()
    df, y = df[keep], y[keep].astype(int).values
    if set(np.unique(y)) - {0, 1}:
        raise ValueError(f'target is not coded 0/1: values {np.unique(y)[:6]}')
    X = df.drop(columns=[target])
    obj = X.select_dtypes(exclude=[np.number]).columns
    X = X.drop(columns=[c for c in obj if X[c].nunique() > 50])      # ID/date-like text columns
    if X.select_dtypes(exclude=[np.number]).shape[1] > 0:
        X = pd.get_dummies(X, drop_first=True)
    X = X.astype(float)
    X = X.loc[:, X.nunique() > 1]                                     # constant columns
    cols = X.columns.tolist(); X = X.values
    if not SCALE_WITHIN_FOLD:                                         # original behaviour
        X = StandardScaler().fit_transform(SimpleImputer(strategy='median').fit_transform(X))
    return X, y, cols

def data_signature(fname):
    p = locate(fname); st = os.stat(p)
    return f'{st.st_size}-{int(st.st_mtime)}'

def base_estimator():
    lr = LogisticRegression(solver='liblinear', max_iter=1000)
    if SCALE_WITHIN_FOLD:   # imputation + scaling are re-fit on every training subset (also inside the calibration CV)
        return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), lr)
    return lr

def leakage_scan(X, y, cols, thr=None):
    """Diagnostic only: AUC of each single feature (NaN filled with the median just for this scan)."""
    thr = LEAKAGE_AUC if thr is None else thr
    out = []
    for j, c in enumerate(cols):
        x = X[:, j]
        if np.isnan(x).any(): x = np.where(np.isnan(x), np.nanmedian(x), x)
        if np.unique(x).size < 2: continue
        auc = roc_auc_score(y, x); auc = max(auc, 1 - auc)
        if auc >= thr: out.append((c, round(auc, 4)))
    return out

def fold_split(Xs, y, fold_iter):
    for fold, (tr, te) in fold_iter:
        Xi, Xv, yi, yv = train_test_split(Xs[tr], y[tr], test_size=VAL_SIZE, stratify=y[tr],
                                          random_state=RANDOM_STATE + fold)
        yield fold, Xi, Xv, Xs[te], yi, yv, y[te]

class UncalibratedEnsemble:
    """The SAME base models that CalibratedClassifierCV(cv=3) fits (same 3 stratified splits, same
    estimator), averaged WITHOUT the isotonic step. Calibrated vs this ensemble isolates the effect
    of calibration alone; ensemble/training-size differences are identical in both."""
    def fit(self, X, y):
        cv = StratifiedKFold(n_splits=3)            # what CalibratedClassifierCV(cv=3) uses for classifiers
        self.models_ = [clone(base_estimator()).fit(X[tr], y[tr]) for tr, _ in cv.split(X, y)]
        return self
    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.models_], axis=0)

def fit_models(Xi, yi):
    t0 = time.perf_counter()
    cal = CalibratedClassifierCV(base_estimator(), cv=3, method='isotonic').fit(Xi, yi)
    fit_time = time.perf_counter() - t0
    unc = UncalibratedEnsemble().fit(Xi, yi)                 # same models, no calibration
    single = clone(base_estimator()).fit(Xi, yi)             # one LR on all of Xi (= OTLP/GHOST base model)
    return cal, unc, single, fit_time

def check_fold_counts(yi, yv, yte):
    if min(np.bincount(yi, minlength=2)) < 3:
        raise ValueError(f'a fold has <3 samples of one class in the model-training part ({np.bincount(yi)})')
    if yv.sum() < 1 or yte.sum() < 1:
        raise ValueError('a fold has no positive sample in validation or test')

def run_dataset(fname, target):
    stem = os.path.splitext(fname)[0]
    out_csv = os.path.join(FOLD_DIR, stem + '.csv'); out_info = os.path.join(FOLD_DIR, stem + '.info')
    sig = data_signature(fname)
    _ck = lambda v: '' if v is None or str(v) in ('', 'nan') else f'{float(v):g}'   # '5', 5 and '5.0' compare equal
    cost_key = _ck(COST_DATASETS.get(fname))
    if os.path.exists(out_csv) and os.path.exists(out_info):
        old = pd.read_csv(out_info).iloc[0]
        if str(old.get('data_sig')) == sig and _ck(old.get('cost_key')) == cost_key:
            return dict(dataset=fname, status='done (cached, same config & data)')
    try:
        Xs, y, cols = load_xy(fname, target)
        info = dict(dataset=fname, n=len(y), n_features=Xs.shape[1], n_pos=int(y.sum()), pi=float(y.mean()),
                    data_sig=sig, cost_key=cost_key, config_hash=CONFIG_HASH)
        if min(np.bincount(y)) < KFOLD:
            raise ValueError(f'too few minority samples ({min(np.bincount(y))}) for {KFOLD}-fold CV')
        leaks = leakage_scan(Xs, y, cols) if LEAKAGE_CHECK else []
        rows = []
        folds = enumerate(StratifiedKFold(n_splits=KFOLD, shuffle=True, random_state=RANDOM_STATE).split(Xs, y))
        for fold, Xi, Xv, Xte, yi, yv, yte in fold_split(Xs, y, folds):
            check_fold_counts(yi, yv, yte)
            pi = float(np.mean(yi))
            cal, unc, single, fit_time = fit_models(Xi, yi)
            p_val, p_te = cal.predict_proba(Xv)[:, 1], cal.predict_proba(Xte)[:, 1]       # calibrated
            q_val, q_te = unc.predict_proba(Xv)[:, 1], unc.predict_proba(Xte)[:, 1]       # same models, uncalibrated
            s_val, s_te = single.predict_proba(Xv)[:, 1], single.predict_proba(Xte)[:, 1] # single LR
            base = dict(dataset=fname, fold=fold, pi=pi, n_val=len(yv), n_val_pos=int(yv.sum()), fit_time=fit_time)

            # probability quality on the test fold (calibration check)
            for cfg, pr in [('C_cal', p_te), ('C_uncal_ens', q_te), ('C_single_lr', s_te)]:
                pr_c = np.clip(pr, 1e-15, 1 - 1e-15)
                rows.append(dict(base, path='C', metric='Brier', config=cfg, test=brier_score_loss(yte, pr)))
                rows.append(dict(base, path='C', metric='LogLoss', config=cfg, test=log_loss(yte, pr_c, labels=[0, 1])))

            # ------------------------------ Path A ------------------------------
            for m in METRICS:
                def rec(cfg, yhat, d, dt, **extra):
                    rows.append(dict(base, path='A', metric=m, config=cfg, test=score_preds(yte, yhat, m),
                                     val=d.get('val'), alpha=d.get('alpha', np.nan), tau=d.get('tau', np.nan),
                                     t_eff=d.get('t_eff'), n_tied=d.get('n_tied'), sel_time=dt, **extra))
                t0 = time.perf_counter(); a1 = select_abtc_A(p_val, yv, pi, m); dt = time.perf_counter() - t0
                y1 = (adjust_posterior(p_te, a1['alpha'], pi) >= a1['tau']).astype(int)
                extra = {}
                if TIME_ORIGINAL_LOOP and m == TIMING_METRIC:
                    t0 = time.perf_counter(); oa, ot = select_original_loop(p_val, yv, pi, m)
                    o_val = float(metric_from_counts(*counts_at(adjust_posterior(p_val, oa, pi), yv, [ot]), m)[0])
                    extra = dict(orig_loop_time=time.perf_counter() - t0,
                                 orig_match=float(np.isclose(oa, a1['alpha']) and np.isclose(ot, a1['tau'])),
                                 orig_same_val=float(abs(o_val - a1['val']) <= 1e-9))
                rec('A1', y1, a1, dt, **extra)

                for cfg, fn in [('A2', lambda: select_grid(p_val, yv, TAU_LIST, m)),
                                ('A3', lambda: select_grid(p_val, yv, TAU_UNIF, m)),
                                ('A5', lambda: select_exhaustive(p_val, yv, m))]:
                    t0 = time.perf_counter(); d = fn(); dt = time.perf_counter() - t0
                    rec(cfg, (p_te >= d['t_eff']).astype(int), d, dt)

                t0 = time.perf_counter(); a4 = select_induced(p_val, yv, pi, m); dt = time.perf_counter() - t0
                y4 = (p_te >= a4['t_eff']).astype(int)
                rec('A4', y4, a4, dt, agree_A1=float(np.mean(y4 == y1)))

                t0 = time.perf_counter(); a6 = select_grid(q_val, yv, TAU_UNIF, m); dt = time.perf_counter() - t0
                rec('A6', (q_te >= a6['t_eff']).astype(int), a6, dt)

                t0 = time.perf_counter(); a7 = select_abtc_A(q_val, yv, pi, m); dt = time.perf_counter() - t0
                rec('A7', (adjust_posterior(q_te, a7['alpha'], pi) >= a7['tau']).astype(int), a7, dt)

                t0 = time.perf_counter(); a8 = select_grid(s_val, yv, TAU_UNIF, m); dt = time.perf_counter() - t0
                rec('A8', (s_te >= a8['t_eff']).astype(int), a8, dt)

            # ------------------------------ Path B ------------------------------
            if fname in COST_DATASETS:
                cfp, cfn = 1.0, float(COST_DATASETS[fname]); tau_star = cfp / (cfp + cfn)
                def recB(cfg, yhat, d, dt):
                    rows.append(dict(base, path='B', metric='cost', config=cfg, cost_ratio=f'1:{cfn:g}',
                                     test=score_preds(yte, yhat, 'cost', cfp, cfn),
                                     val=-d['val'] if 'val' in d else np.nan, alpha=d.get('alpha', np.nan),
                                     tau=d.get('tau', np.nan), t_eff=d.get('t_eff'), n_tied=d.get('n_tied', np.nan),
                                     sel_time=dt))
                t0 = time.perf_counter(); b1 = select_abtc_B(p_val, yv, pi, cfp, cfn, ALPHA_LIST); dt = time.perf_counter() - t0
                recB('B1', (adjust_posterior(p_te, b1['alpha'], pi) >= tau_star).astype(int), b1, dt)
                recB('B2', (p_te >= tau_star).astype(int), dict(alpha=0.0, tau=tau_star, t_eff=tau_star), 0.0)
                t0 = time.perf_counter(); b3 = select_exhaustive(p_val, yv, 'cost', cfp, cfn); dt = time.perf_counter() - t0
                recB('B3', (p_te >= b3['t_eff']).astype(int), b3, dt)
                recB('B4', (q_te >= tau_star).astype(int), dict(alpha=0.0, tau=tau_star, t_eff=tau_star), 0.0)
                t0 = time.perf_counter(); b5 = select_abtc_B(p_val, yv, pi, cfp, cfn, ALPHA_LIST_B5); dt = time.perf_counter() - t0
                recB('B5', (adjust_posterior(p_te, b5['alpha'], pi) >= tau_star).astype(int), b5, dt)

        pd.DataFrame(rows).to_csv(out_csv, index=False)
        info.update(status='ok', leakage=('; '.join(f'{c} (AUC={a})' for c, a in leaks) if leaks else ''))
        pd.DataFrame([info]).to_csv(out_info, index=False)
        return info
    except Exception as e:
        tb = traceback.extract_tb(e.__traceback__)[-1]
        return dict(dataset=fname, status='ERROR', error=f'{type(e).__name__}: {e}  [line {tb.lineno}: {tb.line}]')

# %%
# ## 4. Run all datasets
# Results are stored in `OUT_DIR/folds_<fingerprint>`. The fingerprint changes whenever a setting or the
# code version changes, so old and new results are never mixed. Re-running skips a dataset only if it was
# finished with the same settings AND the CSV file is unchanged (size + modification time) AND the same cost ratio.

# %%
t_start = time.time()
print('Searching for CSV files in:'); [print('   ', d, '' if os.path.isdir(d) else '  <-- folder does not exist') for d in DATA_DIRS]
print('OUT_DIR =', OUT_DIR)
missing = [f for f, _, _ in DS if locate(f) is None]
if missing:
    print(f'WARNING: {len(missing)} of {len(DS)} files not found in any DATA_DIRS folder:'); [print('   ', m) for m in missing]
todo = [(f, t) for f, t, _ in DS if f not in missing]
if not todo:
    found = sorted(f for d in DATA_DIRS if os.path.isdir(d) for f in os.listdir(d) if f.lower().endswith('.csv'))[:15]
    raise FileNotFoundError(f'None of the DATASETS files were found in {DATA_DIRS}.\n'
                            f'Some CSV files that ARE there: {found}\nCheck DATA_DIRS and the file names in DATASETS.')
print(f'{len(todo)} datasets found and will be run.')

try:
    status = Parallel(n_jobs=N_JOBS, verbose=5)(delayed(run_dataset)(f, t) for f, t in todo)
except Exception as e:                         # e.g. a Windows multiprocessing problem
    print(f'Parallel run failed ({type(e).__name__}: {e}) -> running sequentially instead')
    status = [run_dataset(f, t) for f, t in todo]
status = pd.DataFrame(status)
if 'error' not in status: status['error'] = ''
errors = status[status.status == 'ERROR']
print(f'\nFinished {len(status) - len(errors)}/{len(todo)} datasets in {(time.time() - t_start) / 60:.1f} min')
if len(errors):
    print(f'\nERRORS ({len(errors)}):'); print(errors[['dataset', 'error']].to_string(index=False))
errors.to_csv(os.path.join(OUT_DIR, 'errors.csv'), index=False)

# %%
# ## 5. Analysis
# ### 5.0 Load results, dataset info and leakage warnings

# %%
fold_files = glob.glob(os.path.join(FOLD_DIR, '*.csv'))
if not fold_files:
    raise RuntimeError(f'No result files in {FOLD_DIR}.\n'
                       'Section 4 did not finish any dataset: look at its output (missing files / ERRORS list), '
                       'or open errors.csv in OUT_DIR. Fix the cause and run Section 4 again before this cell.')
folds = pd.concat([pd.read_csv(f) for f in fold_files], ignore_index=True)
info  = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(FOLD_DIR, '*.info'))], ignore_index=True)
folds = folds[folds.dataset.isin(PRIMARY)]            # only datasets in the current DATASETS list
info  = info[info.dataset.isin(PRIMARY)]
A = folds[folds.path == 'A'].copy(); B = folds[folds.path == 'B'].copy(); C = folds[folds.path == 'C'].copy()
A['primary'] = A.dataset.map(PRIMARY)
print(f'{A.dataset.nunique()} datasets in Path A, {B.dataset.nunique()} in Path B')
if len(set(A.primary)) == 1:
    print(f'NOTE: every dataset uses the same primary metric ({A.primary.iloc[0]}). The PRIMARY table is then just that '
          'metric, not a domain-specific primary-metric analysis. Add the 3rd item in DATASETS to fix this.')
leaky = info[info.leakage.fillna('') != ''][['dataset', 'leakage']]
if len(leaky):
    print('\nPOSSIBLE TARGET LEAKAGE (a single feature with AUC ≥ LEAKAGE_AUC) — please inspect these:')
    print(leaky.to_string(index=False))
info.to_csv(os.path.join(OUT_DIR, 'dataset_info.csv'), index=False)

NAMES = {'A1': 'ABTC (α,τ) 540, calibrated', 'A2': 'τ-only α=0, 20, calibrated',
         'A3': 'τ-only uniform 540, calibrated', 'A4': 'τ-only induced 540, calibrated',
         'A5': 'τ-only exhaustive, calibrated', 'A6': 'τ-only uniform 540, same models UNcalibrated',
         'A7': 'ABTC (α,τ) 540, same models UNcalibrated', 'A8': 'τ-only uniform 540, single LR (OTLP-like)',
         'B1': 'ABTC Path B (α tuned), calibrated', 'B2': 'Bayes rule α=0, calibrated',
         'B3': 'τ-only cost-tuned, calibrated', 'B4': 'Bayes rule α=0, same models UNcalibrated',
         'B5': 'Revised Path B (α grid incl. 0), calibrated'}

# ---------- statistics helpers ----------
def holm(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); adj = np.empty(n); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (n - r) * p[i])); adj[i] = run
    return adj

def boot_ci(d, n=N_BOOT, seed=0):
    rng = np.random.default_rng(seed); d = np.asarray(d)
    m = d[rng.integers(0, len(d), (n, len(d)))].mean(axis=1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)

def safe_wilcoxon(d, alternative='two-sided'):
    d = np.asarray(d)
    if np.allclose(d, 0): return 1.0
    try: return wilcoxon(d, zero_method='zsplit', alternative=alternative).pvalue
    except ValueError: return 1.0

def boot_ci_level(d, level, n=N_BOOT, seed=0):
    rng = np.random.default_rng(seed); d = np.asarray(d)
    m = d[rng.integers(0, len(d), (n, len(d)))].mean(axis=1)
    a = (1 - level) / 2 * 100
    return np.percentile(m, a), np.percentile(m, 100 - a)

def tost_mean(d, margin):
    """Equivalence of the MEAN paired difference within ±margin: two one-sided paired t-tests.
    p_TOST < 0.05  <=>  the 90% CI of the mean difference lies inside (−margin, +margin)."""
    d = np.asarray(d, float)
    if np.allclose(d, d[0]):                      # zero variance
        return 0.0 if abs(d[0]) < margin else 1.0
    p_lo = ttest_1samp(d, -margin, alternative='greater').pvalue
    p_hi = ttest_1samp(d,  margin, alternative='less').pvalue
    return max(p_lo, p_hi)

def compare(tab, ref, others, higher_better=True, margin=EQUIV_MARGIN, family=''):
    """Paired comparison over datasets.
    * mean_diff with 95% bootstrap CI (positive = ref better)
    * W/T/L and two-sided Wilcoxon signed-rank (tests the location of the differences, NOT the mean)
    * p_holm_table: Holm within this table only (the global Holm over all reported tests is added later)
    * Equivalence of the MEAN difference within ±margin: mean-based TOST (paired t) and the 90% bootstrap CI;
      'equivalent' is True only if p_TOST < 0.05 AND the 90% CI lies inside ±margin."""
    out = []
    for c in others:
        d = (tab[ref] - tab[c]).values
        if not higher_better: d = -d
        lo95, hi95 = boot_ci_level(d, 0.95); lo90, hi90 = boot_ci_level(d, 0.90)
        out.append(dict(family=family, comparison=f'{ref} vs {c}', other=NAMES.get(c, c), n=len(d),
                        mean_ref=tab[ref].mean(), mean_other=tab[c].mean(), mean_diff=d.mean(),
                        ci95_low=lo95, ci95_high=hi95, ci90_low=lo90, ci90_high=hi90,
                        W=int((d > 1e-12).sum()), T=int((np.abs(d) <= 1e-12).sum()), L=int((d < -1e-12).sum()),
                        p_wilcoxon=safe_wilcoxon(d), p_TOST_mean=tost_mean(d, margin) if margin > 0 else np.nan))
    out = pd.DataFrame(out); out['p_holm_table'] = holm(out.p_wilcoxon)
    out['equivalent'] = (out.p_TOST_mean < 0.05) & (out.ci90_low > -margin) & (out.ci90_high < margin)
    return out

def friedman(tab, higher_better=True):
    v = tab.values
    r = np.array([rankdata(-x if higher_better else x) for x in v]).mean(axis=0)
    stat, p = friedmanchisquare(*v.T)
    return pd.Series(r, index=tab.columns).sort_values(), stat, p

EXCEL = {}   # every table also goes into one Excel file at the end

# %%
# ### 5.1 Sanity checks — A1 ≡ A4 (equivalence identity) and fidelity with the original code

# %%
a1 = A[A.config == 'A1'].set_index(['dataset', 'fold', 'metric']).test
a4 = A[A.config == 'A4'].set_index(['dataset', 'fold', 'metric']).test.reindex(a1.index)
chk = A[A.config == 'A4'].groupby('metric').agg(mean_prediction_agreement=('agree_A1', 'mean'),
                                                min_prediction_agreement=('agree_A1', 'min'))
chk['identical_test_score_%'] = pd.Series(np.isclose(a1.values, a4.values), index=a1.index).groupby(level='metric').mean() * 100
print(chk.round(4))
if 'orig_match' in A:
    o = A[(A.config == 'A1') & A.orig_match.notna()]
    mism = o[o.orig_match < 1]
    print(f"\nOriginal sklearn loop chooses the same (α*, τ*) in {o.orig_match.mean() * 100:.2f}% of folds "
          f"({len(mism)} mismatching folds).")
    if len(mism) and 'orig_same_val' in o:
        print(f"In {mism.orig_same_val.mean() * 100:.1f}% of the mismatching folds the original choice has exactly the "
              "same validation score, i.e. the mismatch is only a floating-point tie between equally good pairs.")
        mism[['dataset', 'fold', 'metric', 'alpha', 'tau', 'orig_same_val']].to_csv(
            os.path.join(OUT_DIR, 'original_loop_mismatches.csv'), index=False)
EXCEL['1_identity_check'] = chk.reset_index()

# %%
# ### 5.2 Path A — main ablation (R2-Q2) and calibration effect (R2-Q1)
# * **A1 vs A2/A3/A5**: does joint (α, τ) beat a single threshold?  (`equivalent` = TOST within ±EQUIV_MARGIN)
# * **A1 vs A7** and **A3 vs A6**: effect of isotonic calibration.

# %%
ds = A.groupby(['dataset', 'metric', 'config']).test.mean().unstack('config')
CFG_A = ['A1', 'A2', 'A3', 'A5', 'A6', 'A7', 'A8']
ALL_TESTS = []
for m in METRICS + ['PRIMARY']:
    tab = (A[A.metric == A.primary].groupby(['dataset', 'config']).test.mean().unstack('config')
           if m == 'PRIMARY' else ds.xs(m, level='metric'))[CFG_A].dropna()
    c = pd.concat([compare(tab, 'A1', ['A2', 'A3', 'A5', 'A8'], family=m),      # α: joint search vs single threshold
                   compare(tab, 'A1', ['A7'], family=m),                         # calibration, with ABTC
                   compare(tab, 'A3', ['A6'], family=m),                         # calibration, τ-only (same models)
                   compare(tab, 'A6', ['A8'], family=m)])                        # 3-model average vs single LR
    if m != 'PRIMARY': ALL_TESTS.append(c)
    r, fs, fp = friedman(tab)
    print(f'\n==================== {m}   (n = {len(tab)} datasets) ====================')
    print(tab.agg(['mean', 'std', 'median']).T.rename(index=NAMES).round(4))
    print(c.drop(columns=['family']).round(4).to_string(index=False))
    print(f'Friedman χ² = {fs:.2f}, p = {fp:.3g} | average ranks: ' + ', '.join(f'{k}={v:.2f}' for k, v in r.items()))
    EXCEL[f'2_PathA_{m}'] = c
    EXCEL[f'2_PathA_{m}_means'] = tab.agg(['mean', 'std', 'median']).T.rename(index=NAMES).reset_index()

# Holm over ALL Path A tests reported above (5 metrics x 7 comparisons); PRIMARY is a re-selection, not counted twice
ALL_TESTS = pd.concat(ALL_TESTS, ignore_index=True)
ALL_TESTS['p_holm_global'] = holm(ALL_TESTS.p_wilcoxon)
ALL_TESTS['different_global'] = ALL_TESTS.p_holm_global < 0.05
print('\n===== All Path A tests with Holm correction across every reported comparison =====')
print(ALL_TESTS[['family', 'comparison', 'mean_diff', 'ci95_low', 'ci95_high', 'W', 'T', 'L', 'p_wilcoxon',
                 'p_holm_global', 'p_TOST_mean', 'equivalent']].round(4).to_string(index=False))
EXCEL['2_PathA_ALL_tests_globalHolm'] = ALL_TESTS
ds.reset_index().to_csv(os.path.join(OUT_DIR, 'pathA_dataset_means.csv'), index=False)

# %%
# ### 5.2b Probability quality — does isotonic calibration actually improve the probabilities?
# Brier score and log loss on the test folds (lower is better): calibrated vs the *same* 3 models
# uncalibrated vs a single LR.

# %%
if len(C):
    cq = C.groupby(['dataset', 'metric', 'config']).test.mean().unstack('config')
    for mm in ['Brier', 'LogLoss']:
        t = cq.xs(mm, level='metric').dropna()
        NAMES.update({'C_cal': 'calibrated (isotonic)', 'C_uncal_ens': 'same models uncalibrated', 'C_single_lr': 'single LR'})
        cc = compare(t, 'C_cal', ['C_uncal_ens', 'C_single_lr'], higher_better=False, margin=0, family=mm)
        print(f'\n---- {mm} (positive mean_diff = calibrated is better) ----')
        print(t.agg(['mean', 'median']).T.round(4))
        print(cc.drop(columns=['family', 'p_TOST_mean', 'equivalent']).round(4).to_string(index=False))
        EXCEL[f'2b_prob_quality_{mm}'] = cc

# %%
# ### 5.3 Non-identifiability: how many (α, τ) grid pairs tie with the optimum?

# %%
ridge = A[A.config == 'A1'].groupby('metric').agg(
    median_tied_pairs=('n_tied', 'median'), mean_tied_pairs=('n_tied', 'mean'),
    pct_folds_with_ties=('n_tied', lambda s: (s > 1).mean() * 100),
    median_alpha_star=('alpha', 'median'), median_effective_threshold=('t_eff', 'median'))
print(ridge.round(3)); EXCEL['3_ridge_ties'] = ridge.reset_index()

# %%
# ### 5.4 Computational cost (R2-Q4, R1-3b) — milliseconds per fold
# **Indicative only**: these times come from the main run, where many datasets ran in parallel and
# competed for the CPU. Use the dedicated sequential benchmark (next cell) for numbers in the paper.
# `A1 original loop` = the original 540-evaluation loop with sklearn metric calls (as in the paper's code).

# %%
tm = A[A.metric == TIMING_METRIC].pivot_table(index='dataset', columns='config', values='sel_time', aggfunc='mean') * 1000
tm = tm.rename(columns={c: f'{c} selection' for c in tm.columns})
if 'orig_loop_time' in A:
    tm['A1 original loop'] = A[(A.config == 'A1') & (A.metric == TIMING_METRIC)].groupby('dataset').orig_loop_time.mean() * 1000
tm['model fit (calibrated LR)'] = A[A.config == 'A1'].groupby('dataset').fit_time.mean() * 1000
timing = tm.agg(['median', 'mean', 'max']).T.round(3)
print(timing); EXCEL['4_timing_parallel_run_ms'] = timing.reset_index()

# %%
# ### 5.4b Dedicated SEQUENTIAL timing benchmark (use these numbers in the paper)
# One process, nothing else running in the notebook: for the first `TIMING_BENCH_N` datasets (all folds),
# each selection rule is timed on the same validation scores, repeated 3 times (median kept).
# Note: the gap between the original loop and the fast version is mostly implementation
# (≈540 sklearn metric calls vs. one sort + vectorised counting), not a different algorithm —
# both evaluate exactly the same 540 candidates and return the same (α*, τ*).

# %%
def _t(fn, rep=3):
    ts = []
    for _ in range(rep):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return float(np.median(ts))

bench = []
if TIMING_BENCH_N:
    done = sorted(set(A.dataset))[:TIMING_BENCH_N]
    tgt = {f: t for f, t, _ in DS}
    for f in done:
        Xs, y, _ = load_xy(f, tgt[f])
        folds_ = enumerate(StratifiedKFold(KFOLD, shuffle=True, random_state=RANDOM_STATE).split(Xs, y))
        for fold, Xi, Xv, Xte, yi, yv, yte in fold_split(Xs, y, folds_):
            pi = float(yi.mean())
            t0 = time.perf_counter(); cal = CalibratedClassifierCV(base_estimator(), cv=3, method='isotonic').fit(Xi, yi)
            fit_t = time.perf_counter() - t0
            pv = cal.predict_proba(Xv)[:, 1]; m = TIMING_METRIC
            bench.append(dict(dataset=f, fold=fold, n_val=len(yv),
                              fit_calibrated_LR=fit_t,
                              ABTC_original_loop=_t(lambda: select_original_loop(pv, yv, pi, m), rep=1),
                              ABTC_fast=_t(lambda: select_abtc_A(pv, yv, pi, m)),
                              tau_only_540=_t(lambda: select_grid(pv, yv, TAU_UNIF, m)),
                              tau_only_exhaustive=_t(lambda: select_exhaustive(pv, yv, m))))
    bench = pd.DataFrame(bench)
    bsum = (bench.drop(columns=['fold', 'n_val']).groupby('dataset').mean() * 1000).agg(['median', 'mean', 'max']).T.round(3)
    print(f'Sequential benchmark on {bench.dataset.nunique()} datasets (ms per fold, {TIMING_METRIC}):'); print(bsum)
    bench.to_csv(os.path.join(OUT_DIR, 'timing_sequential_raw.csv'), index=False)
    EXCEL['4b_timing_sequential_ms'] = bsum.reset_index()

# %%
# ### 5.5 Path B (R2-Q3, R1-3c) — total misclassification cost on the test folds (lower is better)
# * **B1 vs B2**: does tuning α help over the pure Bayes rule?
# * **B2 vs B4**: does the Bayes rule need calibrated probabilities? (B4 = the same 3 models, uncalibrated)
# * **B5**: revised Path B (α = 0 allowed).

# %%
if len(B):
    bt = B.groupby(['dataset', 'cost_ratio', 'config']).test.agg(['mean', 'std']).unstack('config')
    bt.columns = [f'{c} {s}' for s, c in bt.columns]
    bt = bt[[f'{c} {s}' for c in ['B1', 'B2', 'B3', 'B4', 'B5'] for s in ['mean', 'std']]]
    print(bt.round(2))
    bm = B.groupby(['dataset', 'config']).test.mean().unstack('config')
    cb = pd.concat([compare(bm, 'B1', ['B2', 'B3', 'B4', 'B5'], higher_better=False, margin=0),
                    compare(bm, 'B2', ['B4'], higher_better=False, margin=0),
                    compare(bm, 'B5', ['B2'], higher_better=False, margin=0)])
    cb['p_holm_all_B'] = holm(cb.p_wilcoxon)
    print('\n(positive mean_diff = first method has LOWER cost)')
    print(cb.drop(columns=['family', 'p_TOST_mean', 'equivalent']).round(4).to_string(index=False))
    shift = B[B.config == 'B1'].groupby('dataset').agg(tau_star=('tau', 'first'), median_alpha_star=('alpha', 'median'),
                                                        median_effective_threshold=('t_eff', 'median'))
    print('\nPath B: effective threshold after α-tuning vs the Bayes threshold τ*'); print(shift.round(4))
    EXCEL['5_PathB_costs'] = bt.reset_index(); EXCEL['5_PathB_tests'] = cb.drop(columns=['p_TOST_mean', 'equivalent'])
    EXCEL['5_PathB_threshold_shift'] = shift.reset_index()

# %%
# ### 5.6 Figure for the response letter — the (α, τ) score surface is a ridge
# White curves are iso-threshold lines t(α, τ) = constant; the validation score is constant along them.

# %%
import matplotlib.pyplot as plt

def plot_ridge(fname, target, metric='F2', fold=0):
    Xs, y, _ = load_xy(fname, target)
    folds_ = list(enumerate(StratifiedKFold(KFOLD, shuffle=True, random_state=RANDOM_STATE).split(Xs, y)))
    _, Xi, Xv, _, yi, yv, _ = next(fold_split(Xs, y, [folds_[fold]]))
    pi = float(np.mean(yi)); cal = fit_models(Xi, yi)[0]
    S = select_abtc_A(cal.predict_proba(Xv)[:, 1], yv, pi, metric)['grid'].reshape(N_ALPHA, N_TAU)
    Aa, Tt = np.meshgrid(ALPHA_LIST, TAU_LIST, indexing='ij')
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    im = ax.pcolormesh(TAU_LIST, ALPHA_LIST, S, shading='nearest', cmap='viridis')
    cs = ax.contour(Tt, Aa, induced_threshold(Aa, Tt, pi), levels=8, colors='white', linewidths=0.8)
    ax.clabel(cs, fmt='t=%.2f', fontsize=7)
    ax.set_xlabel('τ'); ax.set_ylabel('α'); fig.colorbar(im, label=f'validation {metric}')
    ax.set_title(f'{os.path.splitext(fname)[0]} (π₁ = {pi:.3f}), fold {fold + 1}')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f'ridge_{os.path.splitext(fname)[0]}_{metric}.png'), dpi=300)
    plt.show()

if RIDGE_FIGURE and locate(RIDGE_FIGURE[0]):
    plot_ridge(*RIDGE_FIGURE)

# %%
# ### 5.7 Save every table into one Excel file

# %%
xlsx = os.path.join(OUT_DIR, 'ABTC_ablation_summary.xlsx')
try:
    with pd.ExcelWriter(xlsx) as w:
        if 'status' in globals():
            status.to_excel(w, sheet_name='0_run_status', index=False)
        info.to_excel(w, sheet_name='0_dataset_info', index=False)
        for k, v in EXCEL.items():
            v.to_excel(w, sheet_name=k[:31], index=False)
    print('Saved', xlsx)
except Exception as e:      # openpyxl missing -> CSV files instead
    for k, v in EXCEL.items():
        v.to_csv(os.path.join(OUT_DIR, f'{k}.csv'), index=False)
    print(f'Excel export failed ({e}); tables saved as CSV in {OUT_DIR}')
