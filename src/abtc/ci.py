"""
Small single-split pipeline used only by the regression tests (scripts/generate_golden.py and
tests/test_policy_and_end_to_end.py). It is NOT the paper protocol (10-fold CV, see
experiments/ABTC_V2_Final.py); it exists so that both scripts compute exactly the same thing.

Split: 70 % train / 30 % test (stratified, seed 42); a stratified 20 % validation part of the
training data is used to choose the threshold, so the test part is never used for selection.
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

from .metrics import fbeta, gmean, mcc, balanced_acc, pr_auc
from .thresholding import tau_cost, sweep_metric_threshold

SEED = 42


def metric_fn_from_name(metric):
    nm = str(metric).lower()
    if nm == "f2": return lambda yt, yp: fbeta(yt, yp, 2.0)
    if nm == "f1": return lambda yt, yp: fbeta(yt, yp, 1.0)
    if nm in ("g-mean", "gmean"): return gmean
    if nm == "mcc": return mcc
    if nm in ("balancedacc", "balanced accuracy", "balance accuracy", "balacc"): return balanced_acc
    return lambda yt, yp: fbeta(yt, yp, 2.0)


def run_case(X, y, metric="F2", cost_ratio=None, seed=SEED):
    """Fit LR + isotonic calibration, choose tau on validation data, report test metrics."""
    y = np.asarray(y).astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
    Xfit, Xval, yfit, yval = train_test_split(Xtr, ytr, test_size=0.2, stratify=ytr, random_state=seed)
    scaler = StandardScaler().fit(Xfit)
    base = LogisticRegression(max_iter=2000, random_state=seed).fit(scaler.transform(Xfit), yfit)
    s_fit = base.predict_proba(scaler.transform(Xfit))[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(s_fit, yfit)
    p_val = iso.transform(base.predict_proba(scaler.transform(Xval))[:, 1])
    p_te = iso.transform(base.predict_proba(scaler.transform(Xte))[:, 1])
    if str(metric).lower() == "cost":
        tau = tau_cost(float(cost_ratio or 10.0), 1.0)
    else:
        tau, _ = sweep_metric_threshold(yval, p_val, metric_fn_from_name(metric))
    yhat = (p_te >= tau).astype(int)
    return dict(tau=float(tau), F1=float(fbeta(yte, yhat, 1.0)), F2=float(fbeta(yte, yhat, 2.0)),
                GMean=float(gmean(yte, yhat)), MCC=float(mcc(yte, yhat)),
                BalAcc=float(balanced_acc(yte, yhat)), PRAUC=float(pr_auc(yte, p_te)))
