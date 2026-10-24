"""
ABTC - Adaptive Bayesian Thresholding Classifier
=================================================
ABTC V1: Single-Layer Search with Expanded α Range

Search Configuration:
- Path A (Metric-Driven): 27α × 20τ = 540 evaluations
- Path B (Cost-Sensitive): 27α = 27 evaluations (τ is analytical)

This is the script that produced the main results of the paper (Tables 4-6, Figs. 5-7);
it is kept unchanged so that those results can be reproduced. It runs one dataset per call:
set the data file in load_dataset(), the target column (default 'Outcome') and
PRIMARY_METRIC / COST_FP / COST_FN, then run it.
Preprocessing note: features are standardised once on the whole dataset before the CV
split (StandardScaler().fit_transform(X) in main()). The ablation code (ablation/) fits
imputation and scaling inside each training subset instead.

Baselines: GHOST, OTLP and OECV below are simplified re-implementations that share ABTC's
base model (LogisticRegression, liblinear) and validation split; their exact settings are
listed in Supplementary Table S11. 'PR-AUC' and 'G-Mean' are accepted as objectives by the
code but were not used as primary metrics in the paper (F2, F1, MCC, Balanced accuracy, cost).

Authors: Debashis Roy, Anandarup Roy, Bibhuprasad Sahu, Utpal Roy
"""

import re
import numpy as np
import pandas as pd
import random
from math import sqrt
from scipy.stats import wilcoxon, friedmanchisquare, rankdata
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    average_precision_score, balanced_accuracy_score, matthews_corrcoef
)
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, ClassifierMixin
import warnings

warnings.filterwarnings('ignore')

# ================== CONFIGURATION ==================
PRIMARY_METRIC = 'F2'       # 'F1','F2','G-Mean','MCC','BalancedAcc','PR-AUC','cost'
COST_FP        = 1.0        # Cost of False Positive
COST_FN        = 10.0       # Cost of False Negative
RANDOM_STATE   = 42
KFOLD          = 10
N_BOOTSTRAP    = 1000

# ============== EVALUATION FUNCTIONS ===============
def f2_score_custom(y_true, y_pred):
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred)
    if (prec + rec) == 0:
        return 0.0
    β = 2.0
    return (1+β**2)*(prec*rec)/(β**2*prec + rec)

def gmean(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape != (2,2):
        return 0.0
    TN,FP,FN,TP = cm.ravel()
    sens = TP/(TP+FN) if (TP+FN) else 0.0
    spec = TN/(TN+FP) if (TN+FP) else 0.0
    return sqrt(sens*spec)

def cost_metric(y_true, y_pred, cost_fp=1.0, cost_fn=10.0):
    """Calculate total misclassification cost"""
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape != (2,2):
        return float('inf')
    TN, FP, FN, TP = cm.ravel()
    return cost_fp * FP + cost_fn * FN

def evaluate_classifier(y_true, y_proba, threshold=0.5, cost_fp=1.0, cost_fn=10.0):
    y_pred = (y_proba >= threshold).astype(int)
    return {
        'F1':          f1_score(y_true, y_pred, zero_division=0),
        'F2':          f2_score_custom(y_true, y_pred),
        'G-Mean':      gmean(y_true, y_pred),
        'MCC':         matthews_corrcoef(y_true, y_pred),
        'BalancedAcc': balanced_accuracy_score(y_true, y_pred),
        'PR-AUC':      average_precision_score(y_true, y_proba),
        'cost':        cost_metric(y_true, y_pred, cost_fp, cost_fn)
    }

def cliffs_delta(lst1, lst2):
    n1, n2 = len(lst1), len(lst2)
    if n1==0 or n2==0:
        return 0.0
    more = sum(x>y for x in lst1 for y in lst2)
    less = sum(x<y for x in lst1 for y in lst2)
    return (more-less)/(n1*n2)

def bootstrap_ci(data, num_iterations=1000, ci=0.95):
    boot = []
    for _ in range(num_iterations):
        sample = np.random.choice(data, size=len(data), replace=True)
        boot.append(np.median(sample))
    lo = np.percentile(boot, (1-ci)/2*100)
    hi = np.percentile(boot, (1+ci)/2*100)
    return lo, hi


# ============== ABTC CLASSIFIER (VERSION 1) ==============

class ABTCClassifier(BaseEstimator, ClassifierMixin):
    """
    Adaptive Bayesian Thresholding Classifier (ABTC)
    ================================================
    
    A model-agnostic framework that combines Bayesian posterior recalibration
    with optimally-derived, metric- or cost-based threshold selection.
    
    Parameters
    ----------
    base_model : estimator, default=None
        Base probabilistic classifier. If None, uses LogisticRegression.
        
    calibration_method : str, default='isotonic'
        Calibration method: 'isotonic' or 'sigmoid' (Platt scaling).
        
    objective : str, default='F1'
        Optimization objective. Options: 'F1', 'F2', 'G-Mean', 'MCC', 
        'BalancedAcc', 'PR-AUC', 'cost'.
        - If 'cost': Uses Path B (τ calculated analytically, tune only α)
        - Otherwise: Uses Path A (tune both α and τ jointly)
        
    cost_fp : float, default=1.0
        Cost of false positive (used when objective='cost').
        
    cost_fn : float, default=10.0
        Cost of false negative (used when objective='cost').
        
    n_alpha : int, default=27
        Number of α values in the search grid.
        Default 27 provides Δα ≈ 0.11 over [0.1, 3.0].
        
    n_tau : int, default=20
        Number of τ values in the search grid.
        Default 20 provides Δτ ≈ 0.05 over [0.01, 0.99].
        
    alpha_range : tuple, default=(0.1, 3.0)
        Range for posterior adjustment parameter α.
        
    tau_range : tuple, default=(0.01, 0.99)
        Range for decision threshold τ (used only in Path A).
        
    verbose : bool, default=True
        Whether to print optimization details.
    
    Attributes
    ----------
    pi_ : float
        Estimated minority class prior from training data.
        
    best_alpha : float
        Optimal posterior adjustment parameter.
        
    best_tau : float
        Optimal decision threshold.
        
    optimization_path : str
        'Path A' (metric-driven) or 'Path B' (cost-sensitive).
        
    n_evaluations_ : int
        Number of evaluations performed during optimization.
    
    Notes
    -----
    Grid Configuration:
    - Path A: n_alpha × n_tau evaluations (default: 27 × 20 = 540)
    - Path B: n_alpha evaluations (default: 27, since τ is analytical)
    
    Examples
    --------
    >>> # Metric-driven optimization (Path A)
    >>> abtc = ABTCClassifier(objective='F2')
    >>> abtc.fit(X_train, y_train)
    >>> abtc.select_threshold(X_val, y_val)
    >>> predictions = abtc.predict(X_test)
    >>>
    >>> # Cost-sensitive optimization (Path B)
    >>> abtc_cost = ABTCClassifier(objective='cost', cost_fp=1.0, cost_fn=10.0)
    """
    
    def __init__(self,
                 base_model=None,
                 calibration_method='isotonic',
                 objective='F1',
                 cost_fp=1.0,
                 cost_fn=10.0,
                 n_alpha=27,
                 n_tau=20,
                 alpha_range=(0.1, 3.0),
                 tau_range=(0.01, 0.99),
                 verbose=True):
        
        self.base_model         = base_model or LogisticRegression(solver='liblinear', max_iter=1000)
        self.calibration_method = calibration_method
        self.objective          = objective
        self.cost_fp            = cost_fp
        self.cost_fn            = cost_fn
        self.n_alpha            = n_alpha
        self.n_tau              = n_tau
        self.alpha_range        = alpha_range
        self.tau_range          = tau_range
        self.verbose            = verbose
        
        # Build search grids
        self.alpha_list = np.linspace(alpha_range[0], alpha_range[1], n_alpha).tolist()
        self.tau_list   = np.linspace(tau_range[0], tau_range[1], n_tau).tolist()

        # Learned parameters
        self.pi_               = None
        self.calibrated        = None
        self.best_alpha        = None
        self.best_tau          = None
        self.optimization_path = None
        self.n_evaluations_    = 0

    def fit(self, X, y):
        """
        Step 1: Train Calibrated Base Model
        Step 2: Calculate minority class prior
        """
        # Step 2: Calculate π₁ = (1/N) Σ I[yᵢ=1]
        self.pi_ = np.mean(y)
        
        # Step 1: Train and calibrate base model
        self.calibrated = CalibratedClassifierCV(
            self.base_model, 
            cv=3, 
            method=self.calibration_method
        )
        self.calibrated.fit(X, y)
        return self

    def _raw_proba(self, X):
        """Get calibrated probability P(Y=1|x)"""
        return self.calibrated.predict_proba(X)[:,1]

    def _adjust_posterior(self, p, α):
        """
        Apply posterior adjustment with parameter α:
        p_adj = (p/π₁^α) / (p/π₁^α + (1-p)/(1-π₁)^α)
        """
        numerator = p / (self.pi_**α + 1e-12)
        denominator = numerator + ((1-p) / ((1-self.pi_)**α + 1e-12))
        return numerator / (denominator + 1e-12)

    def _compute_score(self, y_true, y_pred):
        """Compute the objective metric"""
        if self.objective == 'cost':
            # For cost, lower is better, so return negative
            return -cost_metric(y_true, y_pred, self.cost_fp, self.cost_fn)
        else:
            return {
                'F1':          f1_score(y_true, y_pred, zero_division=0),
                'F2':          f2_score_custom(y_true, y_pred),
                'G-Mean':      gmean(y_true, y_pred),
                'MCC':         matthews_corrcoef(y_true, y_pred),
                'BalancedAcc': balanced_accuracy_score(y_true, y_pred),
                'PR-AUC':      average_precision_score(y_true, y_pred)
            }[self.objective]

    def select_threshold(self, X_val, y_val):
        """
        Step 3: Determine Optimal Parameters (α*, τ*)
        
        - If objective='cost': Uses Path B (τ calculated analytically, tune only α)
        - Otherwise: Uses Path A (tune both α and τ jointly)
        
        Uses single-layer grid search with expanded α range.
        """
        p_raw = self._raw_proba(X_val)
        self.n_evaluations_ = 0

        if self.objective == 'cost':
            self._select_threshold_path_b(p_raw, y_val)
        else:
            self._select_threshold_path_a(p_raw, y_val)

    def _select_threshold_path_b(self, p_raw, y_val):
        """
        PATH B: Cost-Sensitive Optimization
        -----------------------------------
        Step 3.B.1: Calculate τ* analytically using cost ratio
                    τ* = C_FP / (C_FP + C_FN)
        Step 3.B.2: Tune only α with τ* fixed
        
        Evaluations: n_alpha (default: 27)
        """
        self.optimization_path = 'Path B'
        
        # Step 3.B.1: Calculate τ* analytically from cost ratio
        cost_ratio = self.cost_fp / self.cost_fn
        self.best_tau = cost_ratio / (1.0 + cost_ratio)
        
        if self.verbose:
            print(f"  [Path B] Analytical τ* = {self.best_tau:.4f} (from cost ratio {cost_ratio:.4f})")
        
        # Step 3.B.2: Grid search over α with τ* fixed
        best_score = -np.inf
        best_alpha = None
        
        for α in self.alpha_list:
            p_adj = self._adjust_posterior(p_raw, α)
            y_pred = (p_adj >= self.best_tau).astype(int)
            score = self._compute_score(y_val, y_pred)
            self.n_evaluations_ += 1
            if score > best_score:
                best_score = score
                best_alpha = α
        
        self.best_alpha = best_alpha
        
        if self.verbose:
            print(f"  [Path B] α* = {best_alpha:.4f}, Evals = {self.n_evaluations_}")

    def _select_threshold_path_a(self, p_raw, y_val):
        """
        PATH A: Metric-Driven Optimization
        ----------------------------------
        Step 3.A: Grid search over (α, τ) jointly
        
        Evaluations: n_alpha × n_tau (default: 27 × 20 = 540)
        """
        self.optimization_path = 'Path A'
        
        best_score = -np.inf
        best_alpha = None
        best_tau = None
        
        # Grid search over (α, τ)
        for α in self.alpha_list:
            p_adj = self._adjust_posterior(p_raw, α)
            for τ in self.tau_list:
                y_pred = (p_adj >= τ).astype(int)
                score = self._compute_score(y_val, y_pred)
                self.n_evaluations_ += 1
                if score > best_score:
                    best_score = score
                    best_alpha = α
                    best_tau = τ
        
        self.best_alpha = best_alpha
        self.best_tau = best_tau
        
        if self.verbose:
            print(f"  [Path A] α* = {best_alpha:.4f}, τ* = {best_tau:.4f}, Evals = {self.n_evaluations_}")

    def predict_proba(self, X):
        """
        Step 4: Apply posterior adjustment
        p_adj = (p_raw/π₁^α*) / (p_raw/π₁^α* + (1-p_raw)/(1-π₁)^α*)
        """
        p_raw = self._raw_proba(X)
        return self._adjust_posterior(p_raw, self.best_alpha)

    def predict(self, X):
        """
        Step 4: Final ABTC prediction rule
        ŷ = 1 if p_adj ≥ τ*, else 0
        """
        return (self.predict_proba(X) >= self.best_tau).astype(int)

    def get_params_summary(self):
        """Get summary of learned parameters and configuration"""
        return {
            'objective': self.objective,
            'optimization_path': self.optimization_path,
            'n_alpha': self.n_alpha,
            'n_tau': self.n_tau,
            'pi': self.pi_,
            'alpha': self.best_alpha,
            'tau': self.best_tau,
            'n_evaluations': self.n_evaluations_
        }


# ============== BASELINE CLASSIFIERS ==============

class GHOSTThresholdClassifier:
    def __init__(self, base_model=None):
        self.base_model = base_model or LogisticRegression(solver='liblinear', max_iter=1000)
        self.pi_        = None
        self.threshold_ = 0.5
    def fit(self, X, y):
        self.base_model.fit(X,y)
        self.pi_ = np.mean(y)
    def predict_proba(self, X):
        return self.base_model.predict_proba(X)[:,1]
    def select_threshold(self, X_val, y_val):
        p = self.predict_proba(X_val)
        w = sqrt((1-self.pi_)/(self.pi_+1e-12))
        best_s, best_t = -1, 0.5
        for t in np.linspace(0.01,0.99,99):
            y_pred = (p>=t).astype(int)
            cm = confusion_matrix(y_val, y_pred)
            if cm.shape!=(2,2): continue
            TN,FP,FN,TP = cm.ravel()
            sens = TP/(TP+FN) if TP+FN else 0
            spec = TN/(TN+FP) if TN+FP else 0
            denom = w*w*sens + spec
            s = (1+w*w)*(sens*spec)/denom if denom else 0
            if s>best_s:
                best_s,best_t = s,t
        self.threshold_ = best_t

class OTLPThresholdClassifier:
    def __init__(self, base_model=None, metric='F1'):
        self.base_model = base_model or LogisticRegression(solver='liblinear', max_iter=1000)
        self.metric     = metric
        self.threshold_ = 0.5
    def fit(self, X, y):
        self.base_model.fit(X,y)
    def predict_proba(self, X):
        return self.base_model.predict_proba(X)[:,1]
    def select_threshold(self, X_val, y_val):
        p = self.predict_proba(X_val)
        best_s, best_t = -np.inf, 0.5
        for t in np.linspace(0.01,0.99,99):
            y_pred = (p>=t).astype(int)
            s = {
                'F1': f1_score(y_val,y_pred,zero_division=0),
                'F2': f2_score_custom(y_val,y_pred),
                'G-Mean': gmean(y_val,y_pred),
                'MCC': matthews_corrcoef(y_val,y_pred),
                'BalancedAcc': balanced_accuracy_score(y_val,y_pred)
            }[self.metric]
            if s>best_s:
                best_s,best_t = s,t
        self.threshold_ = best_t

class PosteriorAdjustmentClassifier:
    def __init__(self, base_model=None):
        self.base_model = base_model or LogisticRegression(solver='liblinear', max_iter=1000)
        self.pi_        = 0.5
    def fit(self, X, y):
        self.base_model.fit(X,y)
        self.pi_ = np.mean(y)
    def predict_proba(self, X):
        raw = self.base_model.predict_proba(X)[:,1]
        num = raw/(self.pi_+1e-12)
        den = num + ((1-raw)/(1-self.pi_+1e-12))
        return num/den

class OECVClassifier:
    def __init__(self, base_model=None, generations=30, pop_size=20):
        self.base_model = base_model or LogisticRegression(solver='liblinear', max_iter=1000)
        self.generations = generations
        self.pop_size = pop_size
        self.tau = 0.5
    def fit(self, X,y):
        self.base_model.fit(X,y)
    def predict_proba(self, X):
        return self.base_model.predict_proba(X)[:,1]
    def fitness(self, tau, y_true, p):
        return f2_score_custom(y_true, (p>=tau).astype(int))
    def select_threshold(self, X_val, y_val):
        p = self.predict_proba(X_val)
        pop = list(np.random.uniform(0.1,0.9,size=self.pop_size))
        for _ in range(self.generations):
            scores = [self.fitness(t, y_val, p) for t in pop]
            top = sorted(zip(scores,pop), key=lambda x:-x[0])[:self.pop_size//2]
            pop = [t for _,t in top]
            while len(pop)<self.pop_size:
                c = min(max(random.choice(pop) + np.random.uniform(-0.05,0.05),0.01),0.99)
                pop.append(c)
        self.tau = pop[0]



# ============== q*-CLASSIFIER (Random Forest Quantile Classifier) ==============
class QStarClassifier:
    """
    q*-Classifier (O'Brien & Ishwaran, 2019):
    A Random Forest *quantile* classifier for class-imbalanced data.

    The decision threshold is NOT 0.5 and this is NOT GaussianNB. The threshold
    is set to the (1 - pi) quantile of the forest's out-of-bag (OOB) positive-
    class probabilities, where pi is the minority (positive) prevalence, so the
    predicted-positive rate matches the true prevalence.

    Reference: O'Brien, R. and Ishwaran, H. (2019). "A random forests quantile
    classifier for class imbalanced data." Pattern Recognition, 90, 232-249.
    """
    def __init__(self, n_estimators=300, random_state=RANDOM_STATE):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.threshold_ = 0.5

    def fit(self, X, y):
        y = np.asarray(y).astype(int)
        self.pi_ = float(np.mean(y))  # minority (positive) prevalence
        self.rf_ = RandomForestClassifier(
            n_estimators=self.n_estimators,
            oob_score=True, bootstrap=True,
            n_jobs=-1, random_state=self.random_state
        )
        self.rf_.fit(X, y)
        pos_idx = list(self.rf_.classes_).index(1)
        oob = self.rf_.oob_decision_function_[:, pos_idx]
        oob = oob[~np.isnan(oob)]
        # quantile threshold: (1 - pi) quantile of OOB scores
        self.threshold_ = float(np.quantile(oob, 1.0 - self.pi_)) if len(oob) else 0.5
        return self

    def predict_proba(self, X):
        pos_idx = list(self.rf_.classes_).index(1)
        return self.rf_.predict_proba(X)[:, pos_idx]

    def select_threshold(self, X_val, y_val):
        # threshold already set from OOB during fit; nothing to tune on val
        return self


# ============== UTILITY: LOAD DATASET =============
def load_dataset():
    # Replace with your actual file path
    return pd.read_csv('pima_diabetes.csv')


# ============== MAIN =============================
def main():
    print("="*70)
    print("ABTC EXPERIMENT - Version 1")
    print("="*70)
    print(f"Primary Objective: {PRIMARY_METRIC}")
    if PRIMARY_METRIC == 'cost':
        print(f"Cost Configuration: C_FP={COST_FP}, C_FN={COST_FN}")
    print(f"K-Fold CV: {KFOLD}")
    print("="*70)
    
    # Configuration for ABTC
    N_ALPHA = 27  # Number of α values (expanded range)
    N_TAU = 20    # Number of τ values
    
    print("\n" + "="*70)
    print("ABTC SEARCH CONFIGURATION")
    print("="*70)
    print(f"\n  α grid: {N_ALPHA} values in [0.1, 3.0], Δα ≈ {2.9/(N_ALPHA-1):.3f}")
    print(f"  τ grid: {N_TAU} values in [0.01, 0.99], Δτ ≈ {0.98/(N_TAU-1):.3f}")
    print(f"\n  Path A evaluations: {N_ALPHA} × {N_TAU} = {N_ALPHA * N_TAU}")
    print(f"  Path B evaluations: {N_ALPHA} (τ is analytical)")
    print("="*70)

    # 1) Load & scale
    df = load_dataset()
    X = df.drop(columns=['Outcome']).values
    y = df['Outcome'].values
    Xs = StandardScaler().fit_transform(X)
    
    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution: {np.sum(y==0)} negative, {np.sum(y==1)} positive")
    print(f"Imbalance ratio: 1:{np.sum(y==0)/max(np.sum(y==1),1):.2f}")

    # 2) Precompute folds + inner splits
    outer = list(StratifiedKFold(
        n_splits=KFOLD, shuffle=True, random_state=RANDOM_STATE
    ).split(Xs, y))

    inner_splits = []
    for fold, (tr, te) in enumerate(outer):
        Xtr, ytr = Xs[tr], y[tr]
        Xi, Xv, yi, yv = train_test_split(
            Xtr, ytr, test_size=0.2, stratify=ytr,
            random_state=RANDOM_STATE + fold
        )
        inner_splits.append((tr, te, Xi, Xv, yi, yv))

    # 3) Build classifier dictionary
    classifiers = {
        'NBC (0.5)':       GaussianNB(),
        'q*-Classifier':   QStarClassifier(n_estimators=300, random_state=RANDOM_STATE),
        'Cost-Sens. SVM':  SVC(probability=True, class_weight='balanced', random_state=RANDOM_STATE),
        'Balanced RF':     BalancedRandomForestClassifier(random_state=RANDOM_STATE),
        'PAN':             PosteriorAdjustmentClassifier(),
        'GHOST':           GHOSTThresholdClassifier(),
        'OTLP':            OTLPThresholdClassifier(metric=PRIMARY_METRIC if PRIMARY_METRIC!='cost' else 'F1'),
        'OECV':            OECVClassifier(),
        'ABTC':            ABTCClassifier(
                               base_model=LogisticRegression(solver='liblinear', max_iter=1000),
                               calibration_method='isotonic',
                               objective=PRIMARY_METRIC,
                               cost_fp=COST_FP,
                               cost_fn=COST_FN,
                               n_alpha=N_ALPHA,
                               n_tau=N_TAU,
                               alpha_range=(0.1, 3.0),
                               tau_range=(0.01, 0.99),
                               verbose=True
                           )
    }

    all_scores = {name: [] for name in classifiers}

    # 4) Run outer folds
    print("\n" + "="*70)
    print("RUNNING K-FOLD CROSS-VALIDATION")
    print("="*70)
    
    for fold in range(KFOLD):
        tr, te, Xi, Xv, yi, yv = inner_splits[fold]
        Xte, yte = Xs[te], y[te]
        print(f"\n>>> Fold {fold+1}/{KFOLD}")

        for name, clf in classifiers.items():
            
            if name == 'ABTC':
                clf.fit(Xi, yi)
                clf.select_threshold(Xv, yv)
                proba = clf.predict_proba(Xte)
                thresh = clf.best_tau
                
            elif name in ('GHOST','OTLP','OECV'):
                clf.fit(Xi, yi)
                clf.select_threshold(Xv, yv)
                proba = clf.predict_proba(Xte)
                if proba.ndim>1: proba = proba[:,1]
                thresh = getattr(clf, 'threshold_', getattr(clf, 'tau', 0.5))
                print(f"  [{name}] τ={thresh:.3f}")
                
            elif name == 'q*-Classifier':
                clf.fit(Xi, yi)
                clf.select_threshold(Xv, yv)
                proba = clf.predict_proba(Xte)
                if proba.ndim>1: proba = proba[:,1]
                thresh = clf.threshold_
                print(f"  [q*] tau_q={thresh:.3f}")

            elif name == 'PAN':
                clf.fit(Xi, yi)
                proba = clf.predict_proba(Xte)
                thresh = 0.5

            else:
                clf.fit(Xi, yi)
                proba = clf.predict_proba(Xte)
                if proba.ndim>1: proba = proba[:,1]
                thresh = 0.5

            all_scores[name].append(
                evaluate_classifier(yte, proba, threshold=thresh, 
                                  cost_fp=COST_FP, cost_fn=COST_FN)
            )

    # 5) Reporting
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    # All metrics summary
    all_metrics = ['F1','F2','G-Mean','MCC','BalancedAcc','PR-AUC','cost']
    print("\n=== All Metrics (mean ± std) ===")
    rows = []
    for nm, scs in all_scores.items():
        entry = {'Classifier': nm}
        for m in all_metrics:
            vals = [s[m] for s in scs]
            entry[m] = f"{np.nanmean(vals):.3f} ± {np.nanstd(vals):.3f}"
        rows.append(entry)
    print(pd.DataFrame(rows).to_string(index=False))

    # Primary metric detailed summary
    pm = PRIMARY_METRIC
    print(f"\n=== Primary Metric [{pm}] Detailed Summary ===")
    summary = []
    for nm, scs in all_scores.items():
        vals = [s[pm] for s in scs]
        summary.append({
            'Classifier': nm,
            'Mean':   f"{np.nanmean(vals):.4f}",
            'Std':    f"{np.nanstd(vals):.4f}",
            'Median': f"{np.nanmedian(vals):.4f}",
            'Min':    f"{np.nanmin(vals):.4f}",
            'Max':    f"{np.nanmax(vals):.4f}"
        })
    print(pd.DataFrame(summary).to_string(index=False))

    # Wilcoxon test
    print(f"\n=== Wilcoxon Signed-Rank Test (ABTC vs Others on {pm}) ===")
    base = [s[pm] for s in all_scores['ABTC']]
    results = []
    for nm, scs in all_scores.items():
        if nm == 'ABTC': continue
        oth = [s[pm] for s in scs]
        
        if pm == 'cost':
            stat, p = wilcoxon(base, oth, alternative='less', zero_method='zsplit')
        else:
            stat, p = wilcoxon(base, oth, alternative='greater', zero_method='zsplit')
        
        δ = cliffs_delta(base, oth)
        lo, hi = bootstrap_ci(np.array(base)-np.array(oth), num_iterations=N_BOOTSTRAP)
        results.append({
            'Classifier': nm,
            'p-value': f"{p:.4f}",
            'δ': f"{δ:.2f}",
            'CI': f"[{lo:.3f},{hi:.3f}]",
            'Signif': 'Yes' if p<0.05 else 'No'
        })
    print(pd.DataFrame(results).to_string(index=False))

    # Friedman test
    print(f"\n=== Friedman Test on {pm} ===")
    data = np.array([[s[pm] for s in scs] for scs in all_scores.values()]).T
    data = data[~np.isnan(data).any(axis=1)]
    
    if pm == 'cost':
        stat, p = friedmanchisquare(*data.T)
        ranks = np.mean([rankdata(r) for r in data], axis=0)
    else:
        stat, p = friedmanchisquare(*data.T)
        ranks = np.mean([rankdata(-r) for r in data], axis=0)
    
    df_r = pd.DataFrame({'Classifier': list(classifiers.keys()), 'AvgRank': ranks})
    print(f"Chi-square = {stat:.3f}, p-value = {p:.4f}")
    print(df_r.sort_values('AvgRank').to_string(index=False))
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    
    # Print ABTC configuration summary
    abtc = classifiers['ABTC']
    print(f"\nABTC Final Configuration:")
    print(f"  Objective:          {abtc.objective}")
    print(f"  Optimization Path:  {abtc.optimization_path}")
    print(f"  α grid size:        {abtc.n_alpha}")
    print(f"  τ grid size:        {abtc.n_tau}")
    print(f"  Final α*:           {abtc.best_alpha:.4f}")
    print(f"  Final τ*:           {abtc.best_tau:.4f}")
    print(f"  Evaluations (last): {abtc.n_evaluations_}")

if __name__=='__main__':
    main()
