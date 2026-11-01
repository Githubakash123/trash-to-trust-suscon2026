"""
MCDM cross-method comparison and rank-stability analysis
========================================================
Addresses reviewer points:
  * "Limited comparison to alternative MCDM methods beyond a brief mention;
     missing rank-stability analysis across criteria/weights specific to Indian
     policy contexts"

Ranks the 7 candidate technologies with FOUR MCDM methods on the SAME
literature-based decision matrix used in the pipeline (`TECHNOLOGIES`):
    TOPSIS  (distance to ideal)        -- pipeline default
    SAW/WSM (weighted sum, vector-norm)
    WPM     (weighted product)
    VIKOR   (compromise ranking, v=0.5)
Then:
  * pairwise rank agreement (Spearman, Kendall) across methods
  * Monte-Carlo rank stability under Dirichlet weight perturbation:
    P(top-1 unchanged) and mean Spearman vs the base ranking, per method.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from analysis.common import save_csv
import circular_msw_pipeline as cmp

RNG = np.random.default_rng(3)

CRIT = list(cmp.TECH_CRITERIA.keys())
SPEC = cmp.TECH_CRITERIA
COLS = [SPEC[c][0] for c in CRIT]
BENEFIT = np.array([SPEC[c][1] == "max" for c in CRIT])
TECH = cmp.TECHNOLOGIES["technology"].tolist()
MAT = cmp.TECHNOLOGIES[COLS].to_numpy(float)
W0 = np.array([cmp.TECH_WEIGHTS[c] for c in CRIT]); W0 = W0 / W0.sum()


def _norm_minmax(M):
    """Min-max normalise to [0,1] with benefit/cost direction."""
    out = np.zeros_like(M)
    for j in range(M.shape[1]):
        col = M[:, j]; lo, hi = col.min(), col.max()
        rng = hi - lo if hi > lo else 1.0
        out[:, j] = (col - lo) / rng if BENEFIT[j] else (hi - col) / rng
    return out


def topsis(M, w):
    norm = M / np.sqrt((M ** 2).sum(axis=0))
    v = norm * w
    ib = np.where(BENEFIT, v.max(0), v.min(0))
    iw = np.where(BENEFIT, v.min(0), v.max(0))
    db = np.sqrt(((v - ib) ** 2).sum(1)); dw = np.sqrt(((v - iw) ** 2).sum(1))
    return dw / (db + dw)


def saw(M, w):
    return (_norm_minmax(M) * w).sum(1)


def wpm(M, w):
    N = _norm_minmax(M) + 1e-6
    return np.prod(N ** w, axis=1)


def vikor(M, w, v=0.5):
    fstar = np.where(BENEFIT, M.max(0), M.min(0))
    fminus = np.where(BENEFIT, M.min(0), M.max(0))
    rng = np.where(fstar != fminus, fstar - fminus, 1.0)
    d = (fstar - M) / rng * np.where(BENEFIT, 1, -1)
    # generic: contribution = w * (f* - f)/(f*-f-) regardless of direction
    d = w * np.abs((fstar - M) / rng)
    S = d.sum(1); R = d.max(1)
    def nrm(x):
        return (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else np.zeros_like(x)
    Q = v * nrm(S) + (1 - v) * nrm(R)
    return -Q                                  # higher = better (invert: low Q best)


METHODS = {"TOPSIS": topsis, "SAW": saw, "WPM": wpm, "VIKOR": vikor}


def ranking(scores):
    """Return rank vector (1=best) for a score (higher=better)."""
    order = np.argsort(-scores)
    rank = np.empty(len(scores), int)
    rank[order] = np.arange(1, len(scores) + 1)
    return rank


def run():
    base_scores = {m: f(MAT, W0) for m, f in METHODS.items()}
    base_rank = {m: ranking(s) for m, s in base_scores.items()}

    rank_tab = pd.DataFrame({"technology": TECH})
    for m in METHODS:
        rank_tab[f"rank_{m}"] = base_rank[m]
    rank_tab = rank_tab.sort_values("rank_TOPSIS").reset_index(drop=True)
    save_csv(rank_tab, "mcdm_rankings.csv")

    # pairwise agreement
    arows = []
    ms = list(METHODS)
    for i in range(len(ms)):
        for j in range(i + 1, len(ms)):
            a, b = ms[i], ms[j]
            rho = spearmanr(base_rank[a], base_rank[b]).statistic
            tau = kendalltau(base_rank[a], base_rank[b]).statistic
            arows.append(dict(method_A=a, method_B=b,
                              spearman=round(rho, 3), kendall=round(tau, 3),
                              same_top1=bool(base_rank[a].argmin() == base_rank[b].argmin())))
    save_csv(pd.DataFrame(arows), "mcdm_pairwise_agreement.csv")

    # Monte-Carlo rank stability under Dirichlet weight perturbation
    n_mc = 3000
    conc = W0 * 50                              # concentration ~ moderate perturbation
    Wd = RNG.dirichlet(conc, size=n_mc)
    srows = []
    for m, f in METHODS.items():
        top1_keep = 0; rhos = []
        base_top = base_rank[m].argmin()
        for i in range(n_mc):
            r = ranking(f(MAT, Wd[i]))
            top1_keep += int(r.argmin() == base_top)
            rhos.append(spearmanr(r, base_rank[m]).statistic)
        srows.append(dict(method=m,
                          base_top1=TECH[base_top],
                          P_top1_stable=round(top1_keep / n_mc, 3),
                          mean_spearman_vs_base=round(float(np.mean(rhos)), 3)))
    stab = pd.DataFrame(srows)
    save_csv(stab, "mcdm_rank_stability.csv")

    print("\nMCDM COMPARISON")
    print(rank_tab.to_string(index=False))
    print("\n  Rank stability under weight perturbation:")
    print(stab.to_string(index=False))
    mean_rho = np.mean([r["spearman"] for r in arows])
    print(f"  Mean cross-method Spearman agreement = {mean_rho:.3f}")
    return rank_tab, stab


if __name__ == "__main__":
    run()
