"""Run experiments/ABTC_V2_Final.py once (one dataset, as configured at the top of that script:
data file in load_dataset(), target column, PRIMARY_METRIC / costs). Exits with an error if the run fails."""
import subprocess, sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
exp = ROOT / "experiments" / "ABTC_V2_Final.py"
if not exp.exists():
    sys.exit("Missing experiments/ABTC_V2_Final.py")
print(">>> Running ABTC_V2_Final.py")
subprocess.run([sys.executable, str(exp)], check=True)
