"""
Plastic Circularity Index (PCI): operationalization, weight calibration,
and positioning vs the Material Circularity Indicator (MCI)
=======================================================================
Addresses reviewer points:
  * "PCI weights (alpha,beta,gamma,delta) and polymer-grade flows not operationalized;
     risk of arbitrariness without a data-backed calibration protocol"
  * "no discussion of the Material Circularity Indicator (MCI); positioning PCI
     relative to MCI and plastics leakage indicators is needed"
  * "data collection plan to estimate collection/recycling/substitution/leakage"

PCI definition
--------------
For each city-year plastic stream, four polymer-grade flow rates in [0,1]:
    collection    c  -- share of generated plastic that is formally collected
    recycling     r  -- share recycled into secondary material
    substitution  s  -- quality of substitution (recyclate displacing virgin)
    leakage       l  -- share landfilled / lost from the material loop
    PCI = alpha*c + beta*r + gamma*s + delta*(1 - l),   alpha+beta+gamma+delta = 1

Flow estimation from THIS dataset (+ literature anchors)
--------------------------------------------------------
    recycling r  = Recycling Rate (%) / 100              [observed column]
    leakage   l  = landfilled plastic mass / total plastic mass  [observed]
    collection c = Municipal Efficiency Score / 10        [observed proxy, anchored
                   to CPCB ~60% formal plastic collection in metros]
    substitution s = literature value (recyclate->virgin displacement efficiency
                   for Indian recycled plastic ~0.35); carried as an explicit
                   assumption and stress-tested in the value-sensitivity scan.

Weight calibration -- THREE protocols reported side by side (answers "arbitrariness"):
    1. Equal        : alpha=beta=gamma=delta=0.25 (neutral default)
    2. AHP/expert   : eigenvector of a pairwise-comparison matrix + consistency ratio
    3. Entropy      : Shannon-entropy objective weights from the observed flow matrix
Then a Dirichlet Monte-Carlo over the weight simplex gives PCI distributions and
city-rank stability (Spearman) -- quantifying weight sensitivity.

MCI positioning
---------------
Ellen MacArthur simplified Material Circularity Indicator for the plastic stream:
    V   = 1 - recycled_input_fraction        (virgin feedstock)
    W   = 1 - recycling_rate                  (unrecovered at end of life)
    LFI = (V + W) / 2                          (linear flow index)
    MCI = max(0, 1 - LFI * F)                  F = 0.9 utility factor (default)
PCI and MCI are correlated across cities (construct validity) but PCI adds the
collection and substitution dimensions MCI omits -- shown via Pearson/Spearman and
the residual (what PCI captures beyond MCI).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from analysis.common import load_rows, save_csv

RNG = np.random.default_rng(7)

SUBSTITUTION_BASE = 0.35          # literature anchor (Indian recyclate displacement)
RECYCLED_INPUT_FRACTION = 0.10    # virgin-dominated feedstock in India (~10% recyclate)
MCI_UTILITY_F = 0.9


# --------------------------------------------------------------- flow matrix
def plastic_flows(rows: pd.DataFrame) -> pd.DataFrame:
    pl = rows[rows.waste_type == "plastic"].copy()

    def agg(g):
        tot = g["quantity"].sum()
        land = g.loc[g.method == "landfill", "quantity"].sum()
        return pd.Series({
            "recycling": g["recycling_rate"].mean() / 100.0,
            "leakage": land / tot if tot else 0.0,
            "collection": g["municipal_efficiency"].mean() / 10.0,
            "plastic_tons": tot,
        })

    f = pl.groupby(["city", "year"]).apply(agg, include_groups=False).reset_index()
    f["substitution"] = SUBSTITUTION_BASE
    return f


def pci(flows: pd.DataFrame, w: dict) -> pd.Series:
    return (w["alpha"] * flows["collection"] + w["beta"] * flows["recycling"]
            + w["gamma"] * flows["substitution"] + w["delta"] * (1 - flows["leakage"]))


# ---------------------------------------------------------- weight protocols
def ahp_weights():
    """Pairwise-comparison matrix (recycling>leakage-reduction>collection>substitution)
    -> priority eigenvector + consistency ratio (CR<0.1 acceptable)."""
    # order: collection, recycling, substitution, leakage(=1-l)
    M = np.array([
        [1,   1/3, 3,   1/2],
        [3,   1,   5,   2  ],
        [1/3, 1/5, 1,   1/4],
        [2,   1/2, 4,   1  ],
    ], float)
    w, v = np.linalg.eig(M)
    k = np.argmax(w.real)
    pri = np.abs(v[:, k].real); pri /= pri.sum()
    lam = w.real[k]; n = M.shape[0]
    ci = (lam - n) / (n - 1)
    ri = {4: 0.90}[n]
    cr = ci / ri
    return dict(alpha=pri[0], beta=pri[1], gamma=pri[2], delta=pri[3]), cr


def entropy_weights(flows: pd.DataFrame):
    """Objective Shannon-entropy weights from the observed flow matrix."""
    cols = ["collection", "recycling", "substitution", "leakage_inv"]
    X = pd.DataFrame({
        "collection": flows["collection"], "recycling": flows["recycling"],
        "substitution": flows["substitution"], "leakage_inv": 1 - flows["leakage"],
    })
    P = X / X.sum(axis=0)
    P = P.replace(0, 1e-12)
    k = 1.0 / np.log(len(X))
    E = -k * (P * np.log(P)).sum(axis=0)        # entropy per criterion
    d = 1 - E                                    # divergence
    # substitution is constant -> zero divergence -> floor it so it keeps a weight
    d = d.clip(lower=1e-6)
    w = d / d.sum()
    return dict(alpha=float(w["collection"]), beta=float(w["recycling"]),
                gamma=float(w["substitution"]), delta=float(w["leakage_inv"]))


# ------------------------------------------------------------- MCI (EMF)
def mci(flows: pd.DataFrame) -> pd.Series:
    V = 1 - RECYCLED_INPUT_FRACTION
    W = 1 - flows["recycling"]
    lfi = (V + W) / 2.0
    return (1 - lfi * MCI_UTILITY_F).clip(lower=0)


# ---------------------------------------------------------------------- main
def run():
    rows = load_rows()
    flows = plastic_flows(rows)

    # city-level mean flows (average over years)
    cf = flows.groupby("city").agg(
        collection=("collection", "mean"), recycling=("recycling", "mean"),
        substitution=("substitution", "mean"), leakage=("leakage", "mean")).reset_index()

    weq = dict(alpha=.25, beta=.25, gamma=.25, delta=.25)
    wahp, cr = ahp_weights()
    went = entropy_weights(flows)

    wtab = pd.DataFrame([
        dict(protocol="Equal", **{k: round(v, 4) for k, v in weq.items()}, note=""),
        dict(protocol="AHP_expert", **{k: round(v, 4) for k, v in wahp.items()},
             note=f"consistency_ratio={cr:.3f} ({'OK' if cr < 0.1 else 'revise'})"),
        dict(protocol="Entropy_objective", **{k: round(v, 4) for k, v in went.items()},
             note="data-driven"),
    ])
    save_csv(wtab, "pci_weight_protocols.csv")

    for name, w in [("Equal", weq), ("AHP", wahp), ("Entropy", went)]:
        cf[f"PCI_{name}"] = pci(cf, w).round(4)
    cf["MCI"] = mci(cf).round(4)
    save_csv(cf.round(4).sort_values("PCI_AHP", ascending=False), "pci_city_scores.csv")

    # --- weight sensitivity: Dirichlet MC over the simplex ---
    n_mc = 4000
    W = RNG.dirichlet(np.ones(4), size=n_mc)        # uniform over simplex
    pci_mc = np.zeros((n_mc, len(cf)))
    base = cf[["collection", "recycling", "substitution", "leakage"]].to_numpy()
    for i in range(n_mc):
        a, b, g, d = W[i]
        pci_mc[i] = a * base[:, 0] + b * base[:, 1] + g * base[:, 2] + d * (1 - base[:, 3])
    pci_summary = pd.DataFrame({
        "city": cf["city"],
        "PCI_mean": pci_mc.mean(0).round(4),
        "PCI_p05": np.percentile(pci_mc, 5, axis=0).round(4),
        "PCI_p95": np.percentile(pci_mc, 95, axis=0).round(4),
    }).sort_values("PCI_mean", ascending=False)
    save_csv(pci_summary, "pci_weight_sensitivity.csv")

    # rank stability: Spearman of each MC ranking vs the mean ranking
    mean_rank = pd.Series(-pci_mc.mean(0)).rank().values
    rhos = [spearmanr(pd.Series(-pci_mc[i]).rank().values, mean_rank).statistic
            for i in range(0, n_mc, 4)]
    rank_stab = float(np.mean(rhos))

    # --- substitution value sensitivity ---
    svals = [0.1, 0.2, 0.35, 0.5, 0.7]
    srows = []
    for s in svals:
        tmp = cf.copy(); tmp["substitution"] = s
        srows.append(dict(substitution=s, mean_PCI_AHP=round(pci(tmp, wahp).mean(), 4)))
    save_csv(pd.DataFrame(srows), "pci_substitution_sensitivity.csv")

    # --- PCI vs MCI ---
    pr, pp = pearsonr(cf["PCI_AHP"], cf["MCI"])
    sr, sp = spearmanr(cf["PCI_AHP"], cf["MCI"])
    # residual: PCI variance not explained by MCI (what PCI adds)
    b1, b0 = np.polyfit(cf["MCI"], cf["PCI_AHP"], 1)
    resid = cf["PCI_AHP"] - (b1 * cf["MCI"] + b0)
    # correlate residual with collection+substitution (the dims MCI ignores)
    extra = 0.5 * (cf["collection"] + cf["substitution"])
    r_extra, _ = pearsonr(resid, extra)
    cmp = pd.DataFrame([
        dict(metric="Pearson r (PCI,MCI)", value=round(pr, 3), p=round(pp, 4)),
        dict(metric="Spearman rho (PCI,MCI)", value=round(sr, 3), p=round(sp, 4)),
        dict(metric="R2 of MCI explaining PCI", value=round(pr ** 2, 3), p=np.nan),
        dict(metric="corr(PCI-residual, collection+substitution)",
             value=round(r_extra, 3), p=np.nan),
        dict(metric="PCI weight-rank stability (Spearman)", value=round(rank_stab, 3), p=np.nan),
        dict(metric="AHP consistency ratio", value=round(cr, 3), p=np.nan),
    ])
    save_csv(cmp, "pci_vs_mci.csv")

    print("\nPCI / MCI SUMMARY")
    print(f"  AHP weights: alpha={wahp['alpha']:.2f} beta={wahp['beta']:.2f} "
          f"gamma={wahp['gamma']:.2f} delta={wahp['delta']:.2f} (CR={cr:.3f})")
    print(f"  Entropy weights: alpha={went['alpha']:.2f} beta={went['beta']:.2f} "
          f"gamma={went['gamma']:.2f} delta={went['delta']:.2f}")
    print(f"  PCI vs MCI: Pearson r={pr:.3f} (R2={pr**2:.2f}), Spearman={sr:.3f}")
    print(f"  PCI captures beyond MCI: residual corr w/ collection+substitution = {r_extra:.3f}")
    print(f"  Weight-perturbation rank stability (Spearman) = {rank_stab:.3f}")
    return cmp


if __name__ == "__main__":
    run()
