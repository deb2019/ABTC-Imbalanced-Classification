"""
============================================================================
 Base-Learner Experiment — ABTC(LR) vs ABTC(RF) vs q*(RF) vs Balanced RF
============================================================================
Purpose: test whether ABTC's advantage is a base-learner artifact or comes
from the ABTC mechanism itself. Run on a REPRESENTATIVE SUBSET of datasets
(not all 174) using the SAME CV protocol and seeds as the main experiment.

Four configurations compared, each on its dataset's PRIMARY metric:
  1. ABTC(LR)      - ABTC framework, Logistic Regression base learner
  2. ABTC(RF)      - ABTC framework, Random Forest base learner
  3. q*(RF)        - q*-Classifier (RF quantile classifier, O'Brien & Ishwaran 2019)
  4. Balanced RF   - BalancedRandomForestClassifier (resampling ensemble)

ABTC(LR), q*(RF) and Balanced RF numbers also exist in the main results; here
they are recomputed on the subset so the head-to-head is on identical data.
Only ABTC(RF) is genuinely new.

Fill in DATASETS at the bottom: ('filename.csv', 'target_col', 'PRIMARY_METRIC').
============================================================================
"""

import os, csv, warnings
from math import sqrt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import (
    f1_score, matthews_corrcoef, balanced_accuracy_score,
    average_precision_score, confusion_matrix
)
from imblearn.ensemble import BalancedRandomForestClassifier

warnings.filterwarnings("ignore")

# ---------------- CONFIG (match the main experiment) ----------------
RANDOM_STATE = 42
KFOLD        = 10
N_TREES      = 300
N_ALPHA      = 27
N_TAU        = 20
ALPHA_RANGE  = (0.1, 3.0)
TAU_RANGE    = (0.01, 0.99)
OUTPUT_CSV   = "base_learner_results.csv"
DATA_ROOT    = "."
DEFAULT_METRIC = "F2"

# ---------------- METRIC HELPERS ----------------
def f2_score_custom(y_true, y_pred):
    tp = np.sum((y_true==1)&(y_pred==1)); fp = np.sum((y_true==0)&(y_pred==1))
    fn = np.sum((y_true==1)&(y_pred==0))
    if tp==0 and (fp>0 or fn>0): return 0.0
    prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
    rec  = tp/(tp+fn) if (tp+fn)>0 else 0.0
    if prec==0 and rec==0: return 0.0
    b2=4.0; den=b2*prec+rec
    return (1+b2)*(prec*rec)/den if den>0 else 0.0

def gmean(y_true, y_pred):
    cm=confusion_matrix(y_true,y_pred)
    if cm.shape!=(2,2): return 0.0
    tn,fp,fn,tp=cm.ravel()
    sens=tp/(tp+fn) if (tp+fn) else 0.0
    spec=tn/(tn+fp) if (tn+fp) else 0.0
    return float(sqrt(sens*spec))

def metric_value(y_true, y_pred, name):
    if name=="F1": return f1_score(y_true,y_pred,zero_division=0)
    if name=="F2": return f2_score_custom(y_true,y_pred)
    if name=="G-Mean": return gmean(y_true,y_pred)
    if name=="MCC": return matthews_corrcoef(y_true,y_pred)
    if name=="BalancedAcc": return balanced_accuracy_score(y_true,y_pred)
    raise ValueError(name)

# ---------------- ABTC (generic base learner) ----------------
class ABTC(BaseEstimator, ClassifierMixin):
    def __init__(self, base="LR", objective="F2"):
        self.base=base; self.objective=objective
    def fit(self, X, y):
        y=np.asarray(y).astype(int)
        self.pi_=float(np.mean(y))
        if self.base=="LR":
            base=LogisticRegression(solver="liblinear", max_iter=1000)
        else:
            base=RandomForestClassifier(n_estimators=N_TREES, n_jobs=-1,
                                        random_state=RANDOM_STATE)
        self.cal_=CalibratedClassifierCV(base, cv=3, method="isotonic")
        self.cal_.fit(X,y)
        self.pos_=list(self.cal_.classes_).index(1)
        return self
    def _raw(self, X): return self.cal_.predict_proba(X)[:,self.pos_]
    def _adj(self, p, a):
        num=p/(self.pi_**a+1e-12)
        den=num+(1-p)/((1-self.pi_)**a+1e-12)
        return num/(den+1e-12)
    def select_threshold(self, Xv, yv):
        p=self._raw(Xv)
        alphas=np.linspace(*ALPHA_RANGE,N_ALPHA); taus=np.linspace(*TAU_RANGE,N_TAU)
        best=-np.inf; ba=alphas[0]; bt=0.5
        for a in alphas:
            pa=self._adj(p,a)
            for t in taus:
                s=metric_value(yv,(pa>=t).astype(int),self.objective)
                if s>best: best,ba,bt=s,a,t
        self.a_,self.t_=ba,bt; return self
    def predict(self, X): return (self._adj(self._raw(X),self.a_)>=self.t_).astype(int)

# ---------------- q* (RF quantile classifier) ----------------
class QStar(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        y=np.asarray(y).astype(int); self.pi_=float(np.mean(y))
        self.rf_=RandomForestClassifier(n_estimators=N_TREES, oob_score=True,
                  bootstrap=True, n_jobs=-1, random_state=RANDOM_STATE)
        self.rf_.fit(X,y)
        pos=list(self.rf_.classes_).index(1)
        oob=self.rf_.oob_decision_function_[:,pos]; oob=oob[~np.isnan(oob)]
        self.t_=float(np.quantile(oob,1.0-self.pi_)) if len(oob) else 0.5
        self.pos_=pos; return self
    def predict(self, X):
        return (self.rf_.predict_proba(X)[:,self.pos_]>=self.t_).astype(int)

# ---------------- per-dataset evaluation ----------------
def resolve_target(df, target):
    return target if (target and target in df.columns) else df.columns[-1]

def evaluate(path, target, objective):
    df=pd.read_csv(path); tgt=resolve_target(df,target)
    X=StandardScaler().fit_transform(df.drop(columns=[tgt]).values)
    y=df[tgt].values.astype(int)
    skf=StratifiedKFold(n_splits=KFOLD, shuffle=True, random_state=RANDOM_STATE)
    scores={m:[] for m in ["ABTC(LR)","ABTC(RF)","q*(RF)","Balanced RF"]}
    for fold,(tr,te) in enumerate(skf.split(X,y)):
        Xtr,ytr=X[tr],y[tr]; Xte,yte=X[te],y[te]
        Xi,Xv,yi,yv=train_test_split(Xtr,ytr,test_size=0.2,stratify=ytr,
                                     random_state=RANDOM_STATE+fold)
        # ABTC(LR)
        m=ABTC(base="LR",objective=objective).fit(Xi,yi); m.select_threshold(Xv,yv)
        scores["ABTC(LR)"].append(metric_value(yte,m.predict(Xte),objective))
        # ABTC(RF)
        m=ABTC(base="RF",objective=objective).fit(Xi,yi); m.select_threshold(Xv,yv)
        scores["ABTC(RF)"].append(metric_value(yte,m.predict(Xte),objective))
        # q*(RF)  -- trains on full training fold (OOB threshold)
        m=QStar().fit(Xtr,ytr)
        scores["q*(RF)"].append(metric_value(yte,m.predict(Xte),objective))
        # Balanced RF (threshold 0.5, native)
        brf=BalancedRandomForestClassifier(n_estimators=N_TREES, n_jobs=-1,
              random_state=RANDOM_STATE).fit(Xtr,ytr)
        pos=list(brf.classes_).index(1)
        pred=(brf.predict_proba(Xte)[:,pos]>=0.5).astype(int)
        scores["Balanced RF"].append(metric_value(yte,pred,objective))
    row={"dataset":os.path.basename(path),"objective":objective}
    for m,v in scores.items():
        row[f"{m}_mean"]=float(np.nanmean(v)); row[f"{m}_std"]=float(np.nanstd(v,ddof=1)) if len(v)>1 else 0.0
    return row

# ---------------- driver (resumable) ----------------
def run_all(datasets):
    cols=["dataset","objective"]
    for m in ["ABTC(LR)","ABTC(RF)","q*(RF)","Balanced RF"]:
        cols+=[f"{m}_mean",f"{m}_std"]
    done=set()
    if os.path.exists(OUTPUT_CSV):
        done=set(pd.read_csv(OUTPUT_CSV)["dataset"])
        print(f"Resuming: {len(done)} done.")
    f=open(OUTPUT_CSV,"a",newline=""); w=csv.DictWriter(f,fieldnames=cols)
    if not done: w.writeheader(); f.flush()
    for i,entry in enumerate(datasets,1):
        if len(entry)==3: fn,target,obj=entry
        else: fn,target=entry; obj=DEFAULT_METRIC
        base=os.path.basename(fn)
        if base in done: print(f"[{i}] SKIP {base}"); continue
        try:
            r=evaluate(os.path.join(DATA_ROOT,fn),target,obj); w.writerow(r); f.flush()
            print(f"[{i}] OK {base:38s} obj={obj:4s} "
                  f"ABTC-LR={r['ABTC(LR)_mean']:.3f} ABTC-RF={r['ABTC(RF)_mean']:.3f} "
                  f"q*RF={r['q*(RF)_mean']:.3f} BRF={r['Balanced RF_mean']:.3f}")
        except Exception as e:
            print(f"[{i}] FAIL {base}: {type(e).__name__}: {e}")
    f.close(); print("Done ->", OUTPUT_CSV)

# ---------------- DATASETS: fill in ----------------
DATASETS = [
#    Medical Dataset      
     ('pima_diabetes.csv',                  'Outcome'), 
     ('parkinsons_highly_imbalanced.csv',                  'status'),
     ('cirrhosis_preprocessed_binary.csv',      'Status'),
     ('preprocessed_thyroid_data.csv',            'Recurred'),
     ('haberman_preprocessed_final.csv',                  'Target'),

 #    Fraud Detection Dataset
     ('preprocessed_airbnb_fraud.csv',                  'fraud'), 
     ('Preprocessed_Fraud_Insurance.csv',                  'fraud_reported'), 
     ('Bank_Transactions_Preprocessed.csv',                  'IsFraud'), 
 
#    Cyber Security Dataset
     ('cyber_class_pre.csv',                  'Target'),
     ('Cybersecurity_WebAttack_Preprocessed_30k.csv',                  'Label'),
     ('pre_phishing_imb.csv',                  'Result'),

#    Socio Dataset
     ('Overstimulation_Imb.csv',                  'Overstimulated'),
     ('stress_detection_more_imbalanced.csv',                  'Stress_Binary'),
     ('MentalHealth_Imb.csv',                  'mental_health_issue'),
     ('StartupFunding_Preprocessed.csv',                  'is_high_funding'),  

#    Predictive Maintenance  Dataset
     ('Helicopter_Anomaly_dataset.csv',                  'Lavel'), 
     ('aps_failure_reduced_30k.csv',                  'class'),
     ('MITBIH_ECG_Binary_Preprocessed.csv',                  'Label'),
   

#    Finance and Marketing  Dataset
    #  Loan Approval
     ('preprocessed_loan_approval.csv',                  'Loan_Status'), 
    #  Insurance
     ('pre_IMB_Insu.csv',                  'insuranceclaim'), 
    #  Marketing and Churn Prediction
     ('clickstream_binary_imb.csv',                  'purchase_proxy'), 
     ('pre_telco_churn.csv',                  'Churn'), 
    #  Finance Other
     ('audit_risk_preprocessed.csv',                  'Risk'), 
     ('pre_companies_marketvalue.csv',                  'HighMarketValue'),

#    Recommendation  Dataset
    ('imdb_final_preprocessed.csv',                  'target'), 
    ('pre_google_play_reviews.csv',                  'Target'), 
    ('pre_womens_clothing_data.csv',                  'Recommended')

] 

if __name__=="__main__":
    if not DATASETS: print("Fill in DATASETS, then re-run.")
    else: run_all(DATASETS)
