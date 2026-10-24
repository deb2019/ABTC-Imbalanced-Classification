# %%
# ## 1. Configuration — edit only this cell
DATA_DIRS = [r'.', r'./datasets']
OUT_DIR   = r'./ABTC_priorshift_results'

SHIFT_FACTORS      = [0.25, 0.5, 1.0, 2.0, 4.0]   # deployment prior pi' = m * pi_1 (clipped to [0.005, 0.95])
N_REPEAT           = 5        # random test subsamples per (fold, m)
DEFAULT_COST_RATIO = 5        # C_FP : C_FN = 1 : r for datasets not in COST_DATASETS
ONLY_COST_DATASETS = False    # True: run only the 17 cost datasets (fast check)
N_JOBS             = -1
KFOLD, RANDOM_STATE, VAL_SIZE = 10, 42, 0.20
N_ALPHA, ALPHA_RANGE = 27, (0.1, 3.0)            # ABTC Path B alpha grid (rule R3)
PI_CLIP = (0.005, 0.95)

# %%
# ## 2. Dataset lists (same as the ablation notebook)
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

# %%
# ## 3. Code
import os, json, hashlib, warnings, traceback
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from joblib import Parallel, delayed
warnings.filterwarnings('ignore')
pd.set_option('display.width', 220); pd.set_option('display.max_columns', 40)

DATA_DIRS = [os.path.abspath(os.path.expanduser(d)) for d in DATA_DIRS]
OUT_DIR = os.path.abspath(os.path.expanduser(OUT_DIR))
CODE_VERSION = 'ps1-2026-10-24'
_cfg = dict(code=CODE_VERSION, shifts=SHIFT_FACTORS, rep=N_REPEAT, r=DEFAULT_COST_RATIO, k=KFOLD,
            seed=RANDOM_STATE, val=VAL_SIZE, na=N_ALPHA, ar=ALPHA_RANGE, clip=PI_CLIP)
CONFIG_HASH = hashlib.md5(json.dumps(_cfg, sort_keys=True, default=str).encode()).hexdigest()[:8]
FOLD_DIR = os.path.join(OUT_DIR, f'folds_{CONFIG_HASH}'); os.makedirs(FOLD_DIR, exist_ok=True)
ALPHA_LIST = np.linspace(*ALPHA_RANGE, N_ALPHA)
RULES = ['R0', 'R1', 'R2', 'R3', 'R4']
NAMES = {'R0': 'Bayes rule, shift ignored (alpha = 0)', 'R1': 'ABTC with known prior (alpha from pi\')',
         'R2': 'Threshold tuned on validation', 'R3': 'ABTC Path B (alpha tuned on validation)',
         'R4': 'Re-tuned on source validation data resampled to the known prior'}

_seen, DS = set(), []
for item in DATASETS:
    f, t = item[0], item[1]
    if f in _seen: continue
    _seen.add(f); DS.append((f, t))
if ONLY_COST_DATASETS:
    DS = [d for d in DS if d[0] in COST_DATASETS]
print(f'{len(DS)} datasets | results folder: {FOLD_DIR}')

def locate(fname):
    for d in DATA_DIRS:
        p = os.path.join(d, fname)
        if os.path.isfile(p): return p
    return None

def load_xy(fname, target):
    """Identical to the ablation notebook (SCALE_WITHIN_FOLD = True)."""
    path = locate(fname)
    if path is None: raise FileNotFoundError(f'{fname} not found in {DATA_DIRS}')
    df = pd.read_csv(path, low_memory=False)
    y = pd.to_numeric(df[target], errors='coerce'); keep = y.notna()
    df, y = df[keep], y[keep].astype(int).values
    if set(np.unique(y)) - {0, 1}: raise ValueError('target is not coded 0/1')
    X = df.drop(columns=[target])
    obj = X.select_dtypes(exclude=[np.number]).columns
    X = X.drop(columns=[c for c in obj if X[c].nunique() > 50])
    if X.select_dtypes(exclude=[np.number]).shape[1] > 0: X = pd.get_dummies(X, drop_first=True)
    X = X.astype(float); X = X.loc[:, X.nunique() > 1]
    return X.values, y

def base_estimator():
    return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                         LogisticRegression(solver='liblinear', max_iter=1000))

def logit(x):
    x = np.clip(x, 1e-15, 1 - 1e-15); return np.log(x / (1 - x))

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def adjust_posterior(p, alpha, pi):
    num = p / (pi ** alpha + 1e-12)
    return num / (num + (1 - p) / ((1 - pi) ** alpha + 1e-12) + 1e-12)

def cost_at(p, y, thr, r):
    """Total cost C_FP*FP + C_FN*FN (C_FP = 1, C_FN = r) of the rule p >= t, for every t in thr."""
    pos = np.sort(p[y == 1]); neg = np.sort(p[y == 0]); thr = np.atleast_1d(thr)
    TP = len(pos) - np.searchsorted(pos, thr, side='left')
    FP = len(neg) - np.searchsorted(neg, thr, side='left')
    return FP + r * (len(pos) - TP)

def min_cost_threshold(p, y, r):
    """Exhaustive cost-minimising threshold (every distinct score + 'all negative'), first minimum,
    moved to the midpoint with the next lower score - as rule B3 of the ablation notebook."""
    u = np.unique(p); cand = np.append(u, np.inf)
    c = cost_at(p, y, cand, r); k = int(np.argmin(c))
    if k == len(u): return np.inf
    return u[k] if k == 0 else 0.5 * (u[k] + u[k - 1])

def path_b_alpha(p, y, pi, r):
    tau = 1.0 / (1.0 + r)
    c = np.array([cost_at(adjust_posterior(p, a, pi), y, [tau], r)[0] for a in ALPHA_LIST])
    return ALPHA_LIST[int(np.argmin(c))]

def resample_to(y, pi_new, rng):
    """Indices of a class-wise subsample of y with positive rate pi_new (as close as possible)."""
    i1 = np.flatnonzero(y == 1); i0 = np.flatnonzero(y == 0)
    n1, n0 = len(i1), len(i0)
    if pi_new < n1 / (n1 + n0):          # remove positives
        k1, k0 = int(round(pi_new * n0 / (1 - pi_new))), n0
    else:                                # remove negatives
        k1, k0 = n1, int(round(n1 * (1 - pi_new) / pi_new))
    k1, k0 = min(k1, n1), min(k0, n0)
    if k1 < 1 or k0 < 1: return None
    return np.concatenate([rng.choice(i1, k1, replace=False), rng.choice(i0, k0, replace=False)])

def run_dataset(fname, target):
    stem = os.path.splitext(fname)[0]; out = os.path.join(FOLD_DIR, stem + '.csv')
    r = float(COST_DATASETS.get(fname, DEFAULT_COST_RATIO))
    p = locate(fname)
    sig = f'{os.stat(p).st_size}-{int(os.stat(p).st_mtime)}-{r:g}' if p else ''
    if os.path.exists(out) and os.path.exists(out + '.sig') and open(out + '.sig').read() == sig:
        return dict(dataset=fname, status='done (cached)')
    try:
        X, y = load_xy(fname, target)
        if min(np.bincount(y)) < KFOLD: raise ValueError('too few minority samples for 10-fold CV')
        tau = 1.0 / (1.0 + r); rows = []
        skf = StratifiedKFold(n_splits=KFOLD, shuffle=True, random_state=RANDOM_STATE)
        for fold, (tr, te) in enumerate(skf.split(X, y)):
            Xi, Xv, yi, yv = train_test_split(X[tr], y[tr], test_size=VAL_SIZE, stratify=y[tr],
                                              random_state=RANDOM_STATE + fold)
            if min(np.bincount(yi, minlength=2)) < 3: raise ValueError('a fold has <3 samples of one class')
            cal = CalibratedClassifierCV(base_estimator(), cv=3, method='isotonic').fit(Xi, yi)
            pv = cal.predict_proba(Xv)[:, 1]; pt = cal.predict_proba(X[te])[:, 1]; yt = y[te]
            pi1 = float(yi.mean())
            t_R2 = min_cost_threshold(pv, yv, r)                                    # tuned, old prior
            t_R3 = float(sigmoid(logit(tau) - path_b_alpha(pv, yv, pi1, r) * np.log((1 - pi1) / pi1)))
            for m in SHIFT_FACTORS:
                pi_new = float(np.clip(m * pi1, *PI_CLIP))
                alpha = 1.0 - logit(pi_new) / logit(pi1) if abs(logit(pi1)) > 1e-9 else 0.0
                # R1 threshold: logit t = logit tau* + logit pi1 - logit pi'   (= alpha form, also when pi1 = 0.5)
                t_R1 = float(sigmoid(logit(tau) + logit(pi1) - logit(pi_new)))
                for k in range(N_REPEAT):
                    rng = np.random.default_rng([RANDOM_STATE, fold, int(round(m * 1000)), k])
                    it = resample_to(yt, pi_new, rng); iv = resample_to(yv, pi_new, rng)
                    if it is None: continue
                    t_R4 = min_cost_threshold(pv[iv], yv[iv], r) if iv is not None else t_R2
                    ps, ys = pt[it], yt[it]
                    thr = {'R0': tau, 'R1': t_R1, 'R2': t_R2, 'R3': t_R3, 'R4': t_R4}
                    for rule, t in thr.items():
                        rows.append(dict(dataset=fname, fold=fold, m=m, rep=k, pi_train=pi1, pi_new=pi_new,
                                         alpha=alpha, n_test=len(ys), n_pos=int(ys.sum()), rule=rule,
                                         threshold=t, cost=float(cost_at(ps, ys, [t], r)[0]) / len(ys)))
        pd.DataFrame(rows).to_csv(out, index=False)
        open(out + '.sig', 'w').write(sig)
        return dict(dataset=fname, status='ok', rows=len(rows))
    except Exception as e:
        return dict(dataset=fname, status='error', error=f'{type(e).__name__}: {e}')

# %%
# ## 4. Run
status = Parallel(n_jobs=N_JOBS, verbose=5)(delayed(run_dataset)(f, t) for f, t in DS)
status = pd.DataFrame(status)
print(status.status.value_counts())
if (status.status == 'error').any():
    print(status[status.status == 'error'].to_string(index=False))

# %%
# ## 5. Analysis
files = [os.path.join(FOLD_DIR, os.path.splitext(f)[0] + '.csv') for f, _ in DS]
R = pd.concat([pd.read_csv(f) for f in files if os.path.exists(f)], ignore_index=True)
D = R.groupby(['dataset', 'm', 'rule']).cost.mean().unstack('rule')[RULES]   # mean cost per case
D = D.reset_index()
D['cost_dataset'] = D.dataset.isin(list(COST_DATASETS))

def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); n = len(p); adj = np.empty(n); run = 0.0
    for r_, i in enumerate(o):
        run = max(run, min(1.0, (n - r_) * p[i])); adj[i] = run
    return adj

def safe_wilcoxon(d):
    d = np.asarray(d, float)
    if np.allclose(d, 0): return 1.0
    try: return float(wilcoxon(d, zero_method='zsplit').pvalue)
    except ValueError: return 1.0

summ, tests = [], []
for subset, mask in [('all datasets', D.dataset.notna()), ('17 cost datasets', D.cost_dataset)]:
    for m in SHIFT_FACTORS:
        T = D[mask & (D.m == m)].dropna(subset=RULES)
        row = dict(subset=subset, m=m, n_datasets=len(T))
        for rr in RULES: row[f'{rr} mean cost x100'] = round(100 * T[rr].mean(), 3)
        summ.append(row)
        for other in ['R0', 'R2', 'R3', 'R4']:
            d = (T[other] - T['R1']).values          # > 0: R1 (ABTC with known prior) has the lower cost
            tests.append(dict(subset=subset, m=m, comparison=f'R1 vs {other}', n=len(d),
                              R1_lower=int((d > 1e-12).sum()), ties=int((np.abs(d) <= 1e-12).sum()),
                              R1_higher=int((d < -1e-12).sum()),
                              mean_cost_R1=T['R1'].mean(), mean_cost_other=T[other].mean(),
                              median_ratio_R1_over_other=float(np.median(T['R1'] / T[other].replace(0, np.nan))),
                              p_wilcoxon=safe_wilcoxon(d)))
S = pd.DataFrame(summ); TT = pd.DataFrame(tests)
TT['p_holm_all'] = holm(TT.p_wilcoxon)
print('Mean cost per test case x100 (lower is better)'); print(S.to_string(index=False))
print('\nR1 = ABTC with the known deployment prior, against the other rules')
print(TT.round(4).to_string(index=False))

with pd.ExcelWriter(os.path.join(OUT_DIR, 'ABTC_priorshift_summary.xlsx')) as w:
    pd.DataFrame([_cfg | dict(config_hash=CONFIG_HASH)]).to_excel(w, sheet_name='0_config', index=False)
    status.to_excel(w, sheet_name='0_run_status', index=False)
    S.to_excel(w, sheet_name='1_mean_cost', index=False)
    TT.to_excel(w, sheet_name='2_tests_R1', index=False)
    D.to_excel(w, sheet_name='3_per_dataset', index=False)
    pd.DataFrame([{'rule': k, 'description': v} for k, v in NAMES.items()]).to_excel(w, sheet_name='rules', index=False)
print('Saved', os.path.join(OUT_DIR, 'ABTC_priorshift_summary.xlsx'))
