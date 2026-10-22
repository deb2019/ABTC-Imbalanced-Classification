"""End-to-end regression test: recompute the single-split pipeline (src/abtc/ci.py) for every CSV in
tests/data and compare with tests/golden/expected.json written by scripts/generate_golden.py."""
import json, pathlib, sys
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from abtc.ci import run_case
from generate_golden import resolve_policy, resolve_target  # same policy and target resolution

DATA = ROOT / "tests" / "data"
GOLD = ROOT / "tests" / "golden" / "expected.json"
CSVS = sorted(DATA.glob("*.csv"))


@pytest.mark.skipif(not CSVS, reason="no CSVs in tests/data (see README, 'Fast CI check')")
@pytest.mark.parametrize("csv_path", CSVS)
def test_against_golden(csv_path):
    assert GOLD.exists(), "Run: python scripts/generate_golden.py"
    gold = json.loads(GOLD.read_text(encoding="utf-8"))
    name = csv_path.stem.replace("_", " ").replace("and", "&")
    assert name in gold, f"Golden missing for {name}. Run: python scripts/generate_golden.py"
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    tgt = resolve_target(name, csv_path.name)
    X = df.drop(columns=[tgt]).values; y = df[tgt].values.astype(int)
    metric, _, cost_ratio = resolve_policy(name)
    got = run_case(X, y, metric, cost_ratio)
    for k in ["tau", "F1", "F2", "GMean", "MCC", "BalAcc", "PRAUC"]:
        assert got[k] == pytest.approx(gold[name][k], rel=1e-6, abs=1e-9), f"{name}:{k}"
