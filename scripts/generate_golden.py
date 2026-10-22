# scripts/generate_golden.py
# Writes tests/golden/expected.json from the CSVs in tests/data (single split, see src/abtc/ci.py).
import sys, json, pathlib, numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
# ensure local package is importable even if -e . not installed yet
sys.path.insert(0, str(ROOT / "src"))

from abtc.ci import run_case

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

def main():
    out = {}
    for csv in sorted(DATA.glob("*.csv")):
        name = csv.stem.replace("_"," ").replace("and","&")
        df = pd.read_csv(csv, encoding="utf-8-sig")
        tgt = resolve_target(name, csv.name)
        assert tgt in df.columns, f"{csv.name}: expected target column '{tgt}'. Check config/test_datasets.csv"
        X = df.drop(columns=[tgt]).values
        y = df[tgt].values.astype(int)

        metric, beta, cost_ratio = resolve_policy(name)
        rec = run_case(X, y, metric, cost_ratio, seed=SEED)
        out[name] = rec
        print(f"[golden] {name} (tgt={tgt}, metric={metric})")

    GOLD.parent.mkdir(parents=True, exist_ok=True)
    with open(GOLD, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {len(out)} datasets to {GOLD}")

if __name__ == "__main__":
    main()
