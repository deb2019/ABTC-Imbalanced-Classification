import subprocess, sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
exp  = ROOT / "experiments" / "ABTC_V2_Final.py"

if not exp.exists():
    print("Missing experiments/ABTC_V2_Final.py")
    sys.exit(1)

print(">>> Running ABTC_V2_Final.py")
subprocess.run([sys.executable, str(exp)], check=False)
