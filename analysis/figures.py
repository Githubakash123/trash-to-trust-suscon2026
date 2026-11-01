"""
Publication figures for the reviewer-response analyses.
Reads outputs/*.csv and writes 300-dpi PNG + PDF to figures/ (prefix `rev_`).

    python -m analysis.figures
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.common import OUT, FIG

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True,
})
C = dict(blue="#2c6fbb", green="#2e8b57", orange="#e08a2e", red="#c0392b",
         grey="#7f8c8d", purple="#7d5ba6")


def _o(name):
    return pd.read_csv(os.path.join(OUT, name))


def _save(fig, name):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"   -> figures/{name}.png|.pdf")


def fig_sd():
    s = _o("sd_national_series.csv")
    rec = _o("sd_synthetic_recovery.csv")
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot(s["year"], s["circularity"], "o-", color=C["green"], label="circularity")
    ax[0].plot(s["year"], s["leakage"], "s--", color=C["red"], label="leakage")
    ax[0].set_title("(a) National indicators near steady state")
    ax[0].set_xlabel("Year"); ax[0].set_ylabel("Index"); ax[0].set_ylim(0, 0.6)
    ax[0].legend(frameon=False)
    x = np.arange(len(rec)); w = 0.38
    ax[1].bar(x - w/2, rec["true"], w, label="true", color=C["grey"])
    ax[1].bar(x + w/2, rec["recovered_mean"], w, yerr=rec["recovered_std"],
              capsize=3, label="recovered", color=C["blue"])
    ax[1].set_xticks(x); ax[1].set_xticklabels(rec["param"])
    ax[1].set_title("(b) Synthetic recovery (bias < 0.01)")
    ax[1].set_ylabel("Parameter value"); ax[1].legend(frameon=False)
    _save(fig, "rev_fig1_sd_identifiability")


def fig_pci():
    cf = _o("pci_city_scores.csv")
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.scatter(cf["MCI"], cf["PCI_AHP"], c=C["purple"], s=28, alpha=0.8)
    b1, b0 = np.polyfit(cf["MCI"], cf["PCI_AHP"], 1)
    xs = np.linspace(cf["MCI"].min(), cf["MCI"].max(), 50)
    ax.plot(xs, b1*xs+b0, "--", color=C["grey"])
    r = np.corrcoef(cf["MCI"], cf["PCI_AHP"])[0, 1]
    ax.set_xlabel("Material Circularity Indicator (MCI)")
    ax.set_ylabel("Plastic Circularity Index (PCI, AHP)")
    ax.set_title(f"PCI vs MCI: correlated but complementary\nPearson r={r:.2f} (R²={r**2:.2f})")
    _save(fig, "rev_fig2_pci_vs_mci")


def fig_lca():
    d = _o("lca_impact_intervals.csv").sort_values("ghg_net_mean")
    fig, ax = plt.subplots(figsize=(7, 4))
    y = np.arange(len(d))
    err = np.vstack([d["ghg_net_mean"]-d["ghg_net_p05"], d["ghg_net_p95"]-d["ghg_net_mean"]])
    colors = [C["green"] if v < 0 else C["red"] for v in d["ghg_net_mean"]]
    ax.barh(y, d["ghg_net_mean"], xerr=err, capsize=4, color=colors, alpha=0.85)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels(d["pathway"])
    ax.set_xlabel("Net life-cycle GHG (kg CO₂e / t), 90% interval")
    ax.set_title("LCA impact intervals by pathway (Monte-Carlo)")
    _save(fig, "rev_fig3_lca_intervals")


def fig_rl():
    sw = _o("rl_reward_weight_sensitivity.csv")
    risk = _o("rl_risk_sensitivity.csv")
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(len(sw))
    ax[0].bar(x, sw["final_circularity"], color=C["blue"], alpha=0.85)
    ax[0].set_xticks(x); ax[0].set_xticklabels(sw["weight_scheme"], rotation=30, ha="right")
    ax[0].set_ylabel("Final circularity (2035)")
    ax[0].set_title("(a) Reward-weight sensitivity")
    for xi, (c, co) in enumerate(zip(sw["final_circularity"], sw["mean_cost"])):
        ax[0].text(xi, c+0.005, f"${co:.1f}", ha="center", fontsize=7, color=C["grey"])
    # risk: P_legal nominal vs stress for the two policies
    piv = risk.pivot(index="policy", columns="environment", values="P_legal_ok")
    pol = piv.index.tolist(); xx = np.arange(len(pol)); w = 0.38
    ax[1].bar(xx-w/2, piv["nominal"], w, label="nominal", color=C["green"], alpha=0.85)
    ax[1].bar(xx+w/2, piv["stress"], w, label="stress", color=C["orange"], alpha=0.85)
    ax[1].set_xticks(xx); ax[1].set_xticklabels(pol, rotation=15, ha="right")
    ax[1].set_ylabel("P(legal compliance)"); ax[1].set_ylim(0, 1)
    ax[1].set_title("(b) Robustness: risk-averse holds up under stress")
    ax[1].legend(frameon=False)
    _save(fig, "rev_fig4_rl_policy")


def fig_forecast():
    comp = _o("forecast_walkforward_comparison.csv")
    pi = _o("forecast_prediction_intervals.csv").iloc[0]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    x = np.arange(len(comp))
    ax[0].bar(x, comp["skill_vs_naive"], color=[C["green"] if s > 0 else C["red"]
              for s in comp["skill_vs_naive"]], alpha=0.85)
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set_xticks(x); ax[0].set_xticklabels(comp["method"], rotation=20, ha="right")
    ax[0].set_ylabel("Skill vs naive (1 − MAE/MAE_naive)")
    ax[0].set_title("(a) Walk-forward skill")
    cov = pi["empirical_coverage_on_final_year"]
    ax[1].bar(["target", "achieved"], [0.90, cov], color=[C["grey"], C["blue"]], alpha=0.85)
    ax[1].set_ylim(0, 1); ax[1].set_ylabel("90% PI coverage")
    ax[1].set_title(f"(b) Conformal PI\n(width ≈ {pi['rel_width_vs_mean']*100:.0f}% of mean)")
    _save(fig, "rev_fig5_forecasting")


def fig_routing():
    s = _o("routing_savings_summary.csv")
    fig, ax = plt.subplots(figsize=(5.6, 4))
    x = np.arange(len(s)); w = 0.38
    ax.bar(x-w/2, s["greedy_km"], w, label="greedy", color=C["grey"], alpha=0.85)
    ax.bar(x+w/2, s["ortools_km"], w, label="OR-Tools", color=C["green"], alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(s["problem"])
    ax.set_ylabel("Total distance (km)")
    ax.set_title("Routing: OR-Tools vs greedy baseline")
    for xi, (g, sv) in enumerate(zip(s["greedy_km"], s["distance_saving_pct"])):
        ax.text(xi, g+400, f"−{sv:.0f}%", ha="center", color=C["red"], fontsize=9)
    ax.legend(frameon=False)
    _save(fig, "rev_fig6_routing")


def fig_mcdm():
    st = _o("mcdm_rank_stability.csv")
    fig, ax = plt.subplots(figsize=(5.6, 4))
    x = np.arange(len(st)); w = 0.38
    ax.bar(x-w/2, st["P_top1_stable"], w, label="P(top-1 stable)", color=C["blue"], alpha=0.85)
    ax.bar(x+w/2, st["mean_spearman_vs_base"], w, label="mean Spearman", color=C["green"], alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(st["method"]); ax.set_ylim(0.8, 1.01)
    ax.set_ylabel("Stability"); ax.set_title("MCDM rank stability under weight perturbation")
    ax.legend(frameon=False, loc="lower right")
    _save(fig, "rev_fig7_mcdm_stability")


def fig_uncertainty():
    u = _o("uncertainty_intervals.csv")
    fig, ax = plt.subplots(figsize=(7, 3.4))
    y = np.arange(len(u))
    err = np.vstack([u["mean"]-u["p05"], u["p95"]-u["mean"]])
    ax.errorbar(u["mean"], y, xerr=err, fmt="o", color=C["blue"], capsize=5, ms=7)
    ax.set_yticks(y); ax.set_yticklabels(u["output"])
    ax.set_xlabel("Value (90% interval)")
    ax.set_title("Propagated uncertainty on key 2035 outputs")
    for yi, (m, lo, hi) in enumerate(zip(u["mean"], u["p05"], u["p95"])):
        ax.text(hi, yi, f"  {m:.3g} [{lo:.3g}, {hi:.3g}]", va="center", fontsize=8)
    ax.margins(x=0.25)
    _save(fig, "rev_fig8_uncertainty")


def run():
    os.makedirs(FIG, exist_ok=True)
    for fn in [fig_sd, fig_pci, fig_lca, fig_rl, fig_forecast, fig_routing,
               fig_mcdm, fig_uncertainty]:
        fn()
    print("All reviewer-response figures written to figures/ (rev_fig1..8).")


if __name__ == "__main__":
    run()
