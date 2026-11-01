"""Run every reviewer-response analysis end to end and write all artefacts to
`outputs/`. Reproduces all numbers cited in REVIEWER_RESPONSE.md.

    python -m analysis.run_all
"""
from __future__ import annotations

import time

from analysis import (sd_model, pci_mci, lca_uncertainty, rl_policy,
                      forecasting_plus, routing, mcdm_compare, ablations)

STAGES = [
    ("SD model + calibration + identifiability", sd_model.run),
    ("PCI operationalization + MCI positioning", pci_mci.run),
    ("LCA inventory + uncertainty + sensitivity", lca_uncertainty.run),
    ("RL policy: reward sweep + CVaR + OPE", rl_policy.run),
    ("Forecasting: walk-forward + conformal + SHAP", forecasting_plus.run),
    ("Routing: OR-Tools vs greedy", routing.run),
    ("MCDM comparison + rank stability", mcdm_compare.run),
    ("Ablations + uncertainty propagation", ablations.run),
]


def main():
    t0 = time.time()
    for i, (name, fn) in enumerate(STAGES, 1):
        print(f"\n{'='*70}\n[{i}/{len(STAGES)}] {name}\n{'='*70}")
        fn()
    print(f"\nAll reviewer-response analyses complete in {time.time()-t0:.0f}s. "
          f"Artefacts in outputs/.")


if __name__ == "__main__":
    main()
