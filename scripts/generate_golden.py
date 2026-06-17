# scripts/generate_golden.py
import sys, json, pathlib, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

ROOT = pathlib.Path(__file__).resolve().parents[1]
# ensure local package is importable even if -e . not installed yet
sys.path.insert(0, str(ROOT / "src"))

from abtc.metrics import fbeta, gmean, mcc, balanced_acc, pr_auc
from abtc.thresholding import tau_cost, sweep_metric_threshold

DATA = ROOT / "tests" / "data"
GOLD = ROOT / "tests" / "golden" / "expected.json"
POL  = ROOT / "config" / "metrics_policy.csv"
TEST = ROOT / "config" / "test_datasets.csv"
SEED = 42

def nrm(s):  # normalize strings: strip & remove BOM
    return str(s).strip().replace("\ufeff", "")

def resolve_policy(name):
    metric, beta, cr = "F2", 2, None
    if POL.exists():
        pol = pd.read_csv(POL, encoding="utf-8-sig")
        r = pol[pol["dataset_name"].apply(nrm) == nrm(name)]
        if len(r):
            m   = nrm(r.iloc[0].get("primary_metric","")) or metric
            b   = nrm(r.iloc[0].get("beta",""))
            cr0 = nrm(r.iloc[0].get("cost_ratio",""))
            beta = int(float(b)) if b not in ("","nan") else (beta if m.lower().startswith("f") else None)
            cr   = float(cr0) if cr0 not in ("","nan") else None
            metric = m
    return metric, beta, cr

def resolve_target(name, csv_filename):
    target = "y"
    if TEST.exists():
        df = pd.read_csv(TEST, encoding="utf-8-sig")
        # optional: if you later add a csv_file column, this will prefer it
        if "csv_file" in df.columns:
            r = df[df["csv_file"].apply(lambda s: nrm(s).lower()) == csv_filename.lower()]
            if len(r) and "target_col" in r.columns:
                t = nrm(r.iloc[0]["target_col"])
                if t: return t
        # fallback by dataset_name (your case)
        r = df[df["dataset_name"].apply(nrm) == nrm(name)]
        if len(r) and "target_col" in r.columns:
            t = nrm(r.iloc[0]["target_col"])
            if t: return t
    return target

def calibrate_probs(Xtr, ytr, Xte, seed=SEED):
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
    base = LogisticRegression(max_iter=2000, random_state=seed).fit(Xtr_s, ytr)
    s_tr = base.predict_proba(Xtr_s)[:,1]; s_te = base.predict_proba(Xte_s)[:,1]
    ir = IsotonicRegression(out_of_bounds="clip").fit(s_tr, ytr)
    return ir.transform(s_te)

def metric_fn_from_name(metric):
    nm = metric.lower()
    if nm=="f2": return lambda yt, yp: fbeta(yt, yp, 2.0)
    if nm=="f1": return lambda yt, yp: fbeta(yt, yp, 1.0)
    if nm in ("g-mean","gmean"): return gmean
    if nm=="mcc": return mcc
    if nm in ("balancedacc","balanced accuracy","balacc"): return balanced_acc
    return lambda yt, yp: fbeta(yt, yp, 2.0)

def main():
    out = {}
    for csv in sorted(DATA.glob("*.csv")):
        name = csv.stem.replace("_"," ").replace("and","&")
        df = pd.read_csv(csv, encoding="utf-8-sig")
        tgt = resolve_target(name, csv.name)
        assert tgt in df.columns, f"{csv.name}: expected target column '{tgt}'. Check config/test_datasets.csv"
        X = df.drop(columns=[tgt]).values
        y = df[tgt].values.astype(int)

        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)
        pte = calibrate_probs(Xtr, ytr, Xte, seed=SEED)

        metric, beta, cost_ratio = resolve_policy(name)
        if metric.lower() == "cost":
            tau = tau_cost(float(cost_ratio or 10.0), 1.0, float(ytr.mean()))
        else:
            tau, _ = sweep_metric_threshold(yte, pte, metric_fn_from_name(metric))

        yhat = (pte >= tau).astype(int)
        rec = dict(
            tau=float(tau),
            F1=float(fbeta(yte, yhat, 1.0)),
            F2=float(fbeta(yte, yhat, 2.0)),
            GMean=float(gmean(yte, yhat)),
            MCC=float(mcc(yte, yhat)),
            BalAcc=float(balanced_acc(yte, yhat)),
            PRAUC=float(pr_auc(yte, pte)),
        )
        out[name] = rec
        print(f"[golden] {name} (tgt={tgt}, metric={metric})")

    GOLD.parent.mkdir(parents=True, exist_ok=True)
    with open(GOLD, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {len(out)} datasets to {GOLD}")

if __name__ == "__main__":
    main()
