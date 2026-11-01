"""One command to regenerate and verify every result of the SUSCON XIII paper.

    python reproduce.py            # full run, about 6 minutes on a laptop
    python reproduce.py --verify   # re-check claims on existing outputs only

Steps
  1. circular_msw_pipeline.py   9-stage pipeline        -> outputs/pipeline/
  2. analysis.run_all           8 validation modules    -> outputs/  (30 tables)
  3. verification/extra_checks  V2 pooled-mean baseline, data and unit audit
  4. compare outputs/ with reference/paper_run_1f5883c/ (the run behind the paper)
  5. verification/verify_claims every paper claim -> VERIFICATION_REPORT.md

Exit code 0 = PASS, 1 = FAIL.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "outputs")
REF = os.path.join(ROOT, "reference", "paper_run_1f5883c")
DATA = os.path.join(ROOT, "Waste_Management_and_Recycling_India.csv")
DATA_SHA256 = "66a7ed50099d0f568298100d29879566ce685933d82a8ae5b1348b41ae5275f7"

STEPS = [
    ("pipeline (9 stages)", [sys.executable, "circular_msw_pipeline.py",
                             "--data", "Waste_Management_and_Recycling_India.csv",
                             "--out", os.path.join("outputs", "pipeline"), "--no-figures"]),
    ("validation modules (8)", [sys.executable, "-m", "analysis.run_all"]),
    ("verification extras", [sys.executable, os.path.join("verification", "extra_checks.py")]),
]


def sha256(path: str, normalise_newlines: bool = False) -> str:
    with open(path, "rb") as f:
        b = f.read()
    if normalise_newlines:
        b = b.replace(b"\r\n", b"\n")
    return hashlib.sha256(b).hexdigest()


def compare_with_paper_run():
    identical, numeric, differ = [], [], []
    for name in sorted(os.listdir(REF)):
        ref, new = os.path.join(REF, name), os.path.join(OUT, name)
        if not os.path.exists(new):
            differ.append(f"{name} (missing)")
            continue
        if sha256(ref, True) == sha256(new, True):
            identical.append(name)
            continue
        a, b = pd.read_csv(ref), pd.read_csv(new)
        try:
            pd.testing.assert_frame_equal(a, b, check_dtype=False, rtol=1e-6, atol=1e-9)
            numeric.append(name)
        except AssertionError:
            differ.append(name)
    return identical, numeric, differ


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", action="store_true", help="skip regeneration, re-check claims only")
    args = ap.parse_args()
    os.chdir(ROOT)
    env = dict(os.environ, PYTHONHASHSEED="0", PYTHONUTF8="1", MPLBACKEND="Agg")

    got = sha256(DATA, normalise_newlines=True)
    print(f"dataset sha256 (LF-normalised): {got}")
    if got != DATA_SHA256:
        print("ERROR: dataset does not match the file the paper was computed on.")
        return 1

    timings = {}
    if not args.verify:
        t_all = time.time()
        for i, (label, cmd) in enumerate(STEPS, 1):
            print(f"\n[{i}/{len(STEPS) + 2}] {label}\n" + "-" * 60, flush=True)
            t0 = time.time()
            r = subprocess.run(cmd, cwd=ROOT, env=env)
            timings[label] = round(time.time() - t0, 1)
            if r.returncode != 0:
                print(f"ERROR: step '{label}' exited with {r.returncode}")
                return 1
        total = round(time.time() - t_all, 1)

        print(f"\n[4/{len(STEPS) + 2}] compare with the paper's run\n" + "-" * 60)
        identical, numeric, differ = compare_with_paper_run()
        print(f"byte-identical: {len(identical)}  numerically equal: {len(numeric)}  different: {len(differ)}")
        for d in differ:
            print(f"   differs: {d}")
        import numpy, scipy, sklearn, ortools  # noqa: E401  (versions for the manifest)
        manifest = dict(
            run_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            python=sys.version, platform=platform.platform(),
            packages={m.__name__: m.__version__ for m in (numpy, pd, scipy, sklearn, ortools)},
            dataset_sha256_lf=got, step_seconds=timings, total_seconds=total,
            reference_files=len(os.listdir(REF)), identical_to_paper_run=len(identical),
            numerically_equal_to_paper_run=len(identical) + len(numeric), different_from_paper_run=differ,
        )
        with open(os.path.join(OUT, "run_manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

    print(f"\n[{len(STEPS) + 2}/{len(STEPS) + 2}] verify paper claims\n" + "-" * 60, flush=True)
    r = subprocess.run([sys.executable, os.path.join("verification", "verify_claims.py")], cwd=ROOT, env=env)
    print("\nReport: VERIFICATION_REPORT.md   Claim table: outputs/claims_check.csv")
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
