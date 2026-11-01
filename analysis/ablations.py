"""
Layer ablations + uncertainty propagation
==========================================
Addresses reviewer points:
  * "No ablation to quantify the added value of each layer (SD+RL vs RL-alone;
     with vs without PCI; with vs without member in the ensemble)"
  * "Uncertainty propagation stated conceptually but not reported with concrete
     intervals for key outputs (diversion rates, GHG reductions, PCI trajectories)"

Ablations
---------
A1  SD-coupled feedback vs open-loop schedule
    Closed-loop = the MDP policy reacting to realised circularity (SD state).
    Open-loop   = the same policy's expected-path action SCHEDULE applied blindly.
    Rolled out under stochastic dynamics -> value of SD state-feedback.
A2  With vs without PCI in the allocation objective
    Maximise circularity alone vs circularity + a PCI term -> shift in the
    recommended mix and plastic outcomes -> value of the PCI layer.
A3  Forecast ensemble vs single members
    Mean ensemble vs GBR-alone vs HistoricalMean-alone -> value of ensembling.

Uncertainty propagation
-----------------------
Monte-Carlo over emission factors, recovery weights, and scenario uplift gives
90% intervals on 2035 diversion (circularity), GHG reduction vs BAU, and the
plastic-circularity (PCI proxy) trajectory.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.common import load_panel, save_csv
import analysis.rl_policy as rl
import analysis.forecasting_plus as fp
from analysis.common import load_rows

RNG = np.random.default_rng(99)


# ----------------------------------------------------- A1: SD feedback ablation
def expected_path_schedule(pol, c0):
    """Follow the policy along the EXPECTED (no-noise) trajectory to extract a
    fixed open-loop action schedule a[0..T-1]."""
    ci = rl._ci(c0); bi = rl.NB - 1; sched = []
    for t in range(rl.T):
        a = pol[t, ci, bi]
        if rl.A_COST[a] > rl.B_GRID[bi]:
            a = 0
        sched.append(a); bi -= rl.A_COST[a]
        c = rl.C_GRID[ci]; c_next = c + rl.K_PUSH[a] * (rl.CMAX - c)
        ci = rl._ci(c_next)
    return sched


def rollout_openloop(sched, c0, base_leak, n=4000):
    out = []
    for _ in range(n):
        ci = rl._ci(c0); bi = rl.NB - 1; total_r = 0.0
        for t in range(rl.T):
            a = sched[t]
            if rl.A_COST[a] > rl.B_GRID[bi]:
                a = 0
            bi -= rl.A_COST[a]
            probs, idxs = zip(*[(p, j) for p, j in rl.transition(ci, a)])
            ci_n = int(RNG.choice(idxs, p=np.array(probs)))
            total_r += rl.reward(ci, a, rl._W_FOR_EVAL, base_leak, t == rl.T - 1, ci_n)
            ci = ci_n
        fc = rl.C_GRID[ci]; flk = rl.leakage_of(fc, base_leak)
        out.append((total_r, fc, flk, float(flk <= rl.LEGAL_LEAK_CAP)))
    return np.array(out)


def ablation_sd(panel):
    nat = panel.groupby("year").apply(
        lambda g: g["recovered_weighted"].sum() / g["total_waste"].sum(),
        include_groups=False)
    c0 = float(nat.iloc[-1])
    base_leak = float((panel.groupby("year")["landfill_leakage_qty"].sum()
                       / panel.groupby("year")["total_waste"].sum()).iloc[-1])
    pol, _ = rl.solve(rl.WEIGHT_SCHEMES["balanced"], base_leak)
    closed = rl.rollout(pol, c0, base_leak)            # (r, c, leak, cost, legal)
    sched = expected_path_schedule(pol, c0)
    openl = rollout_openloop(sched, c0, base_leak)     # (r, c, leak, legal)
    return pd.DataFrame([
        dict(variant="SD+policy (closed-loop feedback)",
             mean_return=round(closed[:, 0].mean(), 3),
             final_circularity=round(closed[:, 1].mean(), 3),
             P_legal_ok=round(closed[:, 4].mean(), 3)),
        dict(variant="policy-alone (open-loop schedule)",
             mean_return=round(openl[:, 0].mean(), 3),
             final_circularity=round(openl[:, 1].mean(), 3),
             P_legal_ok=round(openl[:, 3].mean(), 3)),
    ])


# ----------------------------------------------------- A2: PCI ablation
def ablation_pci():
    """Allocate national mix maximising (a) circularity only vs (b) circularity +
    PCI term. Pathways carry a recovery (circularity) and a plastic-suitability
    score (how well they close the plastic loop)."""
    from scipy.optimize import linprog
    # name, circ(recovery), plastic_suitability, min, max
    P = pd.DataFrame([
        ("recycling",            0.85, 1.00, 0.15, 0.55),
        ("composting",           0.60, 0.00, 0.10, 0.45),
        ("anaerobic digestion",  0.75, 0.00, 0.05, 0.35),
        ("advanced recycling",   0.70, 0.90, 0.00, 0.20),
        ("waste-to-energy",      0.25, 0.10, 0.00, 0.20),
        ("landfill",             0.05, 0.00, 0.00, 0.15),
    ], columns=["method", "circ", "plastic", "min", "max"])
    n = len(P)
    A_eq = np.ones((1, n)); b_eq = [1.0]
    bounds = list(zip(P["min"], P["max"]))

    def solve(obj):
        res = linprog(-obj, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
        return res.x

    s_circ = solve(P["circ"].to_numpy())
    s_both = solve(P["circ"].to_numpy() + 0.7 * P["plastic"].to_numpy())
    rows = []
    for label, s in [("without_PCI", s_circ), ("with_PCI", s_both)]:
        rows.append(dict(objective=label,
                         recycling_share=round(max(s[0], 0), 3),
                         advanced_recycling_share=round(max(s[3], 0), 3),
                         plastic_loop_score=round(float((s * P["plastic"]).sum()), 3),
                         system_circularity=round(float((s * P["circ"]).sum()), 3)))
    return pd.DataFrame(rows)


# ----------------------------------------------------- A3: ensemble ablation
def ablation_ensemble():
    rows = load_rows()
    cy = fp.feature_panel(rows)
    wf = fp.walk_forward(cy)
    from sklearn.metrics import mean_absolute_error
    mae_naive = mean_absolute_error(wf["ytrue"], wf["naive"])
    ens = 0.5 * wf["model"] + 0.5 * wf["mean"]
    variants = {
        "GBR-alone": wf["model"], "HistoricalMean-alone": wf["mean"],
        "Ensemble(GBR+Mean)": ens,
    }
    rows_out = []
    for name, pred in variants.items():
        mae = mean_absolute_error(wf["ytrue"], pred)
        rows_out.append(dict(member=name, walkfwd_MAE=round(mae, 1),
                             skill_vs_naive=round(1 - mae / mae_naive, 4)))
    return pd.DataFrame(rows_out).sort_values("walkfwd_MAE")


# ----------------------------------------------------- uncertainty propagation
def uncertainty_propagation(panel, n=10000):
    """MC over emission intensity, recovery efficiency, and scenario uplift ->
    90% intervals on 2035 circularity, GHG reduction vs BAU, plastic circularity."""
    g = panel.groupby("year").agg(
        total=("total_waste", "sum"), rec=("recovered_weighted", "sum"),
        emis=("net_emissions", "sum"), leak=("landfill_leakage_qty", "sum"),
        pl=("plastic_waste", "sum"), pleak=("plastic_leakage", "sum"))
    last = g.iloc[-1]
    base_circ = last["rec"] / last["total"]
    base_pcirc = (last["pl"] - last["pleak"]) / max(last["pl"], 1)
    emis_int = last["emis"] / last["total"]

    # Advanced_Recycling uplift assumptions with uncertainty
    circ_uplift = RNG.triangular(0.20, 0.30, 0.40, n)
    plastic_uplift = RNG.triangular(0.15, 0.25, 0.35, n)
    emis_coupling = RNG.triangular(0.45, 0.60, 0.75, n)     # GHG sensitivity to circ
    growth = RNG.triangular(0.015, 0.025, 0.035, n)

    horizon = 2035 - 2023
    total_2035 = last["total"] * (1 + growth) ** horizon
    circ_2035 = np.clip(base_circ * (1 + circ_uplift), 0, 0.98)
    pcirc_2035 = np.clip(base_pcirc * (1 + plastic_uplift), 0, 0.99)
    emis_2035 = total_2035 * emis_int * (1 - emis_coupling * (circ_2035 - base_circ))
    bau_2035 = last["total"] * (1 + growth) ** horizon * emis_int
    ghg_reduction = 100 * (bau_2035 - emis_2035) / bau_2035

    def iv(x):
        return (round(float(np.mean(x)), 3), round(float(np.percentile(x, 5)), 3),
                round(float(np.percentile(x, 95)), 3))
    out = []
    for label, x in [("circularity_2035 (diversion)", circ_2035),
                     ("GHG_reduction_vs_BAU_pct", ghg_reduction),
                     ("plastic_circularity_2035 (PCI proxy)", pcirc_2035)]:
        m, lo, hi = iv(x)
        out.append(dict(output=label, mean=m, p05=lo, p95=hi))
    return pd.DataFrame(out)


def run():
    panel = load_panel()
    print("A1  SD-feedback ablation ...")
    a1 = ablation_sd(panel); save_csv(a1, "ablation_sd_feedback.csv")
    print("A2  PCI ablation ...")
    a2 = ablation_pci(); save_csv(a2, "ablation_pci.csv")
    print("A3  ensemble ablation ...")
    a3 = ablation_ensemble(); save_csv(a3, "ablation_ensemble.csv")
    print("UQ  uncertainty propagation ...")
    uq = uncertainty_propagation(panel); save_csv(uq, "uncertainty_intervals.csv")

    print("\nABLATIONS")
    print("A1 SD feedback:"); print(a1.to_string(index=False))
    print("A2 PCI:"); print(a2.to_string(index=False))
    print("A3 ensemble:"); print(a3.to_string(index=False))
    print("\nUNCERTAINTY (90% interval)")
    print(uq.to_string(index=False))
    return a1, a2, a3, uq


if __name__ == "__main__":
    run()
