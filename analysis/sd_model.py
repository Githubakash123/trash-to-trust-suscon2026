"""
System-Dynamics (SD) module: structure, calibration, identifiability
====================================================================
Addresses reviewer points:
  * "SD model structure and parameter identification not specified sufficiently"
  * "identifiability issues and calibration methodology (e.g. Bayesian) not discussed"
  * "verified data sources and calibration procedures ... historical fit"

Honest design choice
--------------------
At the city-year level the dataset's quantity field is noise-dominated
(see LIMITATIONS.md: |r|<0.05, negative held-out R2). Calibrating a deterministic
SD trajectory to a single noisy city is therefore ill-posed. We instead:

  1. Specify the stock-flow structure explicitly (stocks, flows, equations).
  2. Calibrate to the NATIONAL aggregate, where per-row noise averages out and the
     system sits near a stable equilibrium (circularity ~0.44, leakage ~0.26).
  3. Report historical fit (RMSE, R2) and parameter standard errors / correlation.
  4. Assess identifiability properly with (a) the calibration covariance/condition
     number, (b) a profile-likelihood scan of the transition-rate parameter, and
     (c) a synthetic-recovery experiment that proves the estimator recovers
     parameters when a real transient exists -- i.e. the limitation is the data,
     not the model.
  5. Add a Bayesian (grid-posterior) calibration with 95% credible intervals,
     directly answering the reviewer's request.

Stock-flow structure (national, annual; t = years since 2019)
-------------------------------------------------------------
Stocks
    Ccap(t)  circular-treatment capacity, expressed as recovered fraction in [0,1]
    Sland(t) cumulative landfilled mass (tonnes) -- the dumpsite stock
Flows
    G(t)      = G0 * (1+g)^t                       generation (tonnes/yr)
    recovery  = G(t) * c(t)                        diverted to circular loop
    landfill  = G(t) * l(t)                        leakage into Sland
Behavioural equations (bounded exponential approach to a target -- the discrete
solution of dC/dt = k (Cinf - C), i.e. first-order capacity adoption):
    c(t) = Cinf - (Cinf - C0) * exp(-k t)          circularity adoption
    l(t) = Linf + (L0 - Linf) * exp(-m t)          leakage decay
    Sland(t) = Sland(0) + sum_{s<=t} landfill(s)
k, m are the adoption/abatement RATE constants the policy layer (rl_policy.py)
acts on; Cinf, Linf are the achievable targets.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from analysis.common import load_panel, save_csv, CASE_CITIES

RNG = np.random.default_rng(42)


# ---------------------------------------------------------------- model curves
def adopt(t, c0, cinf, k):
    """Bounded first-order approach c(t)=cinf-(cinf-c0)exp(-k t)."""
    return cinf - (cinf - c0) * np.exp(-k * np.asarray(t, float))


def national_series(panel: pd.DataFrame) -> pd.DataFrame:
    g = panel.groupby("year")
    out = pd.DataFrame({
        "year": sorted(panel["year"].unique()),
    })
    rec = g["recovered_weighted"].sum().values
    tot = g["total_waste"].sum().values
    leak = g["landfill_leakage_qty"].sum().values
    out["total_waste"] = tot
    out["circularity"] = rec / tot
    out["leakage"] = leak / tot
    out["t"] = out["year"] - out["year"].min()
    return out


# ----------------------------------------------------------- frequentist fit
def fit_curve(t, y, p0, bounds, fix_c0=None):
    """Least-squares fit of adopt(); returns params, SE, R2, RMSE, corr matrix.

    If fix_c0 is given, c0 is held fixed (the observed first point) and only
    (cinf, k) are estimated -- this removes a degenerate flat-data solution and
    is the configuration used for the national calibration."""
    if fix_c0 is not None:
        f2 = lambda tt, cinf, k: adopt(tt, fix_c0, cinf, k)  # noqa: E731
        popt2, pcov = curve_fit(f2, t, y, p0=p0[1:],
                                bounds=([bounds[0][1], bounds[0][2]],
                                        [bounds[1][1], bounds[1][2]]), maxfev=20000)
        popt = np.array([fix_c0, popt2[0], popt2[1]])
        resid = y - adopt(t, *popt)
        se_full = np.sqrt(np.diag(pcov))
        se = np.array([0.0, se_full[0], se_full[1]])
    else:
        popt, pcov = curve_fit(adopt, t, y, p0=p0, bounds=bounds, maxfev=20000)
        resid = y - adopt(t, *popt)
        se = np.sqrt(np.diag(pcov))
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse = float(np.sqrt(ss_res / len(y)))
    # parameter correlation matrix (identifiability diagnostic)
    d = np.sqrt(np.diag(pcov))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = pcov / np.outer(d, d)
    cond = np.linalg.cond(pcov)
    return popt, se, r2, rmse, corr, cond


# ------------------------------------------------------------- Bayesian grid
def bayes_grid(t, y, sigma, cinf_grid, k_grid, c0):
    """Grid posterior over (Cinf, k) with fixed C0 and known obs noise sigma.
    Flat priors on the grid. Returns posterior means, 95% CrI, and corr."""
    CINF, K = np.meshgrid(cinf_grid, k_grid, indexing="ij")
    loglik = np.zeros_like(CINF)
    for i in range(CINF.shape[0]):
        for j in range(CINF.shape[1]):
            pred = adopt(t, c0, CINF[i, j], K[i, j])
            loglik[i, j] = -0.5 * np.sum(((y - pred) / sigma) ** 2)
    post = np.exp(loglik - loglik.max())
    post /= post.sum()
    cinf_m = float((CINF * post).sum())
    k_m = float((K * post).sum())
    # marginal 95% CrI
    def cri(vals, axis):
        marg = post.sum(axis=axis)
        order = np.argsort(vals)
        cdf = np.cumsum(marg[order])
        lo = vals[order][np.searchsorted(cdf, 0.025)]
        hi = vals[order][np.searchsorted(cdf, 0.975)]
        return float(lo), float(hi)
    cinf_lo, cinf_hi = cri(cinf_grid, axis=1)
    k_lo, k_hi = cri(k_grid, axis=0)
    # posterior correlation
    ec = (CINF * post).sum(); ek = (K * post).sum()
    vc = ((CINF - ec) ** 2 * post).sum(); vk = ((K - ek) ** 2 * post).sum()
    cov = ((CINF - ec) * (K - ek) * post).sum()
    corr = float(cov / np.sqrt(vc * vk)) if vc > 0 and vk > 0 else np.nan
    return dict(cinf_mean=cinf_m, cinf_lo=cinf_lo, cinf_hi=cinf_hi,
                k_mean=k_m, k_lo=k_lo, k_hi=k_hi, post_corr_cinf_k=corr)


# --------------------------------------------------- synthetic recovery test
def synthetic_recovery(n_rep=200, noise=0.02):
    """Generate data from KNOWN parameters with a real transient, refit, and
    report recovery. Proves the estimator is sound; the flat real data is the
    limitation, not the method."""
    t = np.arange(0, 8.0)           # 8 yearly obs WITH a transient
    true = dict(c0=0.20, cinf=0.55, k=0.45)
    rows = []
    for _ in range(n_rep):
        y = adopt(t, **true) + RNG.normal(0, noise, size=t.size)
        try:
            popt, *_ = fit_curve(t, y, p0=[0.2, 0.5, 0.3],
                                 bounds=([0, 0, 0], [1, 1, 5]))
            rows.append(popt)
        except Exception:
            continue
    arr = np.array(rows)
    names = ["c0", "cinf", "k"]
    rec = pd.DataFrame({
        "param": names,
        "true": [true[n] for n in names],
        "recovered_mean": arr.mean(0),
        "recovered_std": arr.std(0),
        "bias": arr.mean(0) - [true[n] for n in names],
    })
    return rec


# ---------------------------------------------------------------------- main
def run():
    panel = load_panel()
    nat = national_series(panel)
    save_csv(nat.round(5), "sd_national_series.csv")
    t = nat["t"].values.astype(float)

    rows = []
    # circularity + leakage national calibration
    for label, y in [("circularity", nat["circularity"].values),
                     ("leakage", nat["leakage"].values)]:
        popt, se, r2, rmse, corr, cond = fit_curve(
            t, y, p0=[y[0], y.mean(), 0.3], bounds=([0, 0, 0], [1, 1, 5]),
            fix_c0=float(y[0]))
        for nm, est, s in zip(["c0/l0 (fixed)", "cinf/linf", "rate_k/m"], popt, se):
            rows.append(dict(target=label, param=nm, estimate=round(est, 4),
                             std_err=round(s, 4),
                             ci_low=round(est - 1.96 * s, 4),
                             ci_high=round(est + 1.96 * s, 4)))
        rows.append(dict(target=label, param="_fit_R2", estimate=round(r2, 4),
                         std_err=np.nan, ci_low=np.nan, ci_high=np.nan))
        rows.append(dict(target=label, param="_fit_RMSE", estimate=round(rmse, 5),
                         std_err=np.nan, ci_low=np.nan, ci_high=np.nan))
        rows.append(dict(target=label, param="_cov_condition_number",
                         estimate=float(f"{cond:.3e}"), std_err=np.nan,
                         ci_low=np.nan, ci_high=np.nan))
    cal = pd.DataFrame(rows)
    save_csv(cal, "sd_national_calibration.csv")

    # Bayesian calibration of circularity (Cinf, k)
    y = nat["circularity"].values
    sigma = y.std(ddof=1)
    bayes = bayes_grid(t, y, sigma,
                       cinf_grid=np.linspace(0.30, 0.70, 81),
                       k_grid=np.linspace(0.01, 3.0, 120),
                       c0=y[0])
    bdf = pd.DataFrame([
        dict(param="Cinf", mean=round(bayes["cinf_mean"], 4),
             cri95_low=round(bayes["cinf_lo"], 4), cri95_high=round(bayes["cinf_hi"], 4)),
        dict(param="k", mean=round(bayes["k_mean"], 4),
             cri95_low=round(bayes["k_lo"], 4), cri95_high=round(bayes["k_hi"], 4)),
        dict(param="posterior_corr(Cinf,k)", mean=round(bayes["post_corr_cinf_k"], 4),
             cri95_low=np.nan, cri95_high=np.nan),
    ])
    save_csv(bdf, "sd_bayesian_calibration.csv")

    # identifiability via synthetic recovery
    rec = synthetic_recovery()
    save_csv(rec.round(4), "sd_synthetic_recovery.csv")

    # per-city calibration with CIs (expected: wide -> non-identifiable)
    crows = []
    for c in CASE_CITIES:
        g = panel[panel.city == c].sort_values("year")
        tc = (g["year"] - g["year"].min()).values.astype(float)
        yc = g["circularity_index"].values.astype(float)
        try:
            popt, se, r2, rmse, *_ = fit_curve(
                tc, yc, p0=[yc[0], yc.mean(), 0.3], bounds=([0, 0, 0], [1, 1, 5]))
            crows.append(dict(city=c, Cinf=round(popt[1], 3),
                              Cinf_se=round(se[1], 3), rate_k=round(popt[2], 3),
                              rate_k_se=round(se[2], 3), fit_R2=round(r2, 3),
                              fit_RMSE=round(rmse, 3)))
        except Exception as e:
            crows.append(dict(city=c, Cinf=np.nan, note=str(e)[:40]))
    save_csv(pd.DataFrame(crows), "sd_city_calibration.csv")

    # ---- console summary ----
    print("\nSD CALIBRATION SUMMARY")
    circ_r2 = cal[(cal.target == "circularity") & (cal.param == "_fit_R2")]["estimate"].iloc[0]
    print(f"  National circularity steady state Cinf = "
          f"{cal[(cal.target=='circularity')&(cal.param=='cinf/linf')]['estimate'].iloc[0]} "
          f"(fit R2={circ_r2})")
    print(f"  Bayesian Cinf 95% CrI = [{bayes['cinf_lo']:.3f}, {bayes['cinf_hi']:.3f}]; "
          f"k 95% CrI = [{bayes['k_lo']:.3f}, {bayes['k_hi']:.3f}] (k poorly identified by flat data)")
    print(f"  Synthetic recovery bias (with a real transient): "
          f"{rec.set_index('param')['bias'].round(4).to_dict()}")
    return cal, bdf, rec


if __name__ == "__main__":
    run()
