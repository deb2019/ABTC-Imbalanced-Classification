import json, pathlib, pytest, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from abtc.metrics import fbeta, gmean, mcc, balanced_acc, pr_auc

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "data"
GOLD = ROOT / "tests" / "golden" / "expected.json"

def calibrate_probs(Xtr, ytr, Xte, seed=42):
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
    base = LogisticRegression(max_iter=2000, random_state=seed).fit(Xtr_s, ytr)
    s_tr = base.predict_proba(Xtr_s)[:,1]; s_te = base.predict_proba(Xte_s)[:,1]
    ir = IsotonicRegression(out_of_bounds="clip").fit(s_tr, ytr)
    return ir.transform(s_te)

@pytest.mark.parametrize("csv_path", sorted(DATA.glob("*.csv")))
def test_against_golden(csv_path):
    with open(GOLD, "r", encoding="utf-8") as f:
        gold = json.load(f)
    name = csv_path.stem.replace("_"," ").replace("and","&")
    assert name in gold, f"Golden missing for {name}. Run: python scripts/generate_golden.py"
    df = pd.read_csv(csv_path)
    X = df.drop(columns=['y']).values; y = df['y'].values.astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    pte = calibrate_probs(Xtr, ytr, Xte, seed=42)
    yhat = (pte >= 0.5).astype(int)  # metrics-only check
    got = dict(
        F1=float(fbeta(yte, yhat, 1.0)),
        F2=float(fbeta(yte, yhat, 2.0)),
        GMean=float(gmean(yte, yhat)),
        MCC=float(mcc(yte, yhat)),
        BalAcc=float(balanced_acc(yte, yhat)),
        PRAUC=float(pr_auc(yte, pte)),
    )
    exp = gold[name]
    rtol, atol = 5e-2, 1e-3
    for k in ["F1","F2","GMean","MCC","BalAcc","PRAUC"]:
        assert abs(got[k] - exp[k]) <= max(atol, rtol * max(1.0, abs(exp[k]))), f"{name}:{k} got={got[k]} exp={exp[k]}"
