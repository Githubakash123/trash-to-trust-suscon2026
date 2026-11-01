"""Verification-only computations that the paper relies on but the original
analysis code does not write to a table.

None of these change the paper's code or its outputs. They re-use the exact
walk-forward folds of ``analysis.forecasting_plus`` and add what the paper's
text asserts:

  E1  V2 - a pooled (all-cities) mean baseline on the same expanding-window
      folds, plus walk-forward R^2 for every method (paper Sections 3.5, 4.4).
  E2  Data audit - |Pearson r| between tonnage and every covariate, the
      within-city lag-1 autocorrelation (paper Section 4.2), and the tests
      quoted in ERRATA.md showing the panel is synthetic (KS vs uniform,
      disposal-vs-type independence, between-city ANOVA, landfill offsets).
  E3  Unit audit - the unit of the tonnage column the MAE is computed on
      (paper Table 3 labels it t/year).

    python verification/extra_checks.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, r2_score

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from analysis.common import DATA, load_rows, save_csv  # noqa: E402
from analysis.forecasting_plus import COVARS, feature_panel, walk_forward  # noqa: E402


def pooled_mean_baseline(cy: pd.DataFrame, wf: dict) -> np.ndarray:
    """For each walk-forward target year, predict every city with the mean of
    all strictly earlier city-years (all cities pooled)."""
    out = []
    for ty in wf["tyear"]:
        out.append(cy.loc[cy.year < ty, "total_waste"].mean())
    return np.array(out)


def e1_baselines(cy: pd.DataFrame) -> pd.DataFrame:
    wf = walk_forward(cy)
    y = wf["ytrue"]
    preds = {
        "GBR + covariates": wf["model"],
        "City historical mean": wf["mean"],
        "Naive persistence": wf["naive"],
        "Pooled mean (all cities)": pooled_mean_baseline(cy, wf),
    }
    mae_naive = mean_absolute_error(y, wf["naive"])
    rows = []
    for name, p in preds.items():
        mae = mean_absolute_error(y, p)
        rows.append(dict(method=name, walkfwd_MAE=round(mae, 1),
                         walkfwd_R2=round(r2_score(y, p), 4),
                         skill_vs_persistence=round(1 - mae / mae_naive, 4),
                         n_predictions=len(y),
                         test_years="-".join(str(t) for t in sorted(set(wf["tyear"])))))
    df = pd.DataFrame(rows).sort_values("walkfwd_MAE").reset_index(drop=True)
    best_naive = df[df.method != "GBR + covariates"].iloc[0]
    gbr = df[df.method == "GBR + covariates"].iloc[0]
    df["skill_vs_strongest_naive"] = (1 - df["walkfwd_MAE"] / best_naive["walkfwd_MAE"]).round(4)
    df["strongest_naive"] = best_naive["method"]
    df["gbr_beats_strongest_naive"] = bool(gbr["walkfwd_MAE"] < best_naive["walkfwd_MAE"])
    return df


def e2_audit(rows: pd.DataFrame, cy: pd.DataFrame) -> pd.DataFrame:
    out = []
    for c in COVARS:
        if c in ("lag1_total_waste", "year"):
            continue
        r = cy[["total_waste", c]].corr().iloc[0, 1]
        out.append(dict(test="pearson_r_cityyear_tonnage_vs", variable=c, value=round(r, 4)))
    for c in ["recycling_rate", "population_density", "municipal_efficiency",
              "cost_per_ton", "awareness_campaigns"]:
        r = rows[["quantity", c]].corr().iloc[0, 1]
        out.append(dict(test="pearson_r_row_tonnage_vs", variable=c, value=round(r, 4)))
    ac = (cy.sort_values(["city", "year"]).groupby("city")["total_waste"]
          .apply(lambda s: s.autocorr(lag=1)))
    out.append(dict(test="lag1_autocorrelation_mean_over_cities", variable="total_waste",
                    value=round(float(ac.mean()), 4)))

    # Is the panel synthetic? (ERRATA.md, "what the dataset is")
    raw = pd.read_csv(DATA)
    q = raw["Waste Generated (Tons/Day)"]
    ks = stats.kstest(q, "uniform", args=(q.min(), q.max() - q.min()))
    out.append(dict(test="ks_pvalue_tonnage_vs_uniform", variable="quantity", value=round(ks.pvalue, 4)))
    ct = pd.crosstab(raw["Waste Type"], raw["Disposal Method"])
    chi = stats.chi2_contingency(ct)
    out.append(dict(test="chi2_pvalue_disposal_independent_of_waste_type", variable="Disposal Method",
                    value=round(chi.pvalue, 4)))
    for wt, m in [("Plastic", "Composting"), ("E-Waste", "Composting"), ("Hazardous", "Recycling")]:
        out.append(dict(test="impossible_rows", variable=f"{wt}->{m}", value=int(ct.loc[wt, m])))
    an = stats.f_oneway(*[g.circularity_index.values for _, g in
                          load_panel_from(rows).groupby("city")])
    out.append(dict(test="anova_pvalue_circularity_across_cities", variable="circularity_index",
                    value=round(an.pvalue, 4)))
    for c, (lat, lon) in TRUE_CITY_COORDS.items():
        r = raw[raw["City/District"] == c].iloc[0]
        llat, llon = (float(x) for x in r["Landfill Location (Lat, Long)"].split(","))
        out.append(dict(test="landfill_km_from_city_centre", variable=c,
                        value=round(haversine_km(lat, lon, llat, llon), 0)))
    return pd.DataFrame(out)


# Approximate city-centre coordinates (public gazetteer values).
TRUE_CITY_COORDS = {"Mumbai": (19.076, 72.878), "Delhi": (28.614, 77.209),
                    "Chennai": (13.083, 80.271), "Indore": (22.720, 75.858)}


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return float(2 * 6371.0 * np.arcsin(np.sqrt(a)))


def load_panel_from(rows: pd.DataFrame) -> pd.DataFrame:
    import circular_msw_pipeline as cmp
    return cmp.build_city_year_panel(rows)


def e3_units() -> pd.DataFrame:
    header = pd.read_csv(DATA, nrows=0).columns.tolist()
    col = [h for h in header if h.startswith("Waste Generated")][0]
    return pd.DataFrame([dict(source_column=col,
                              aggregation="sum over the 5 waste streams per city-year",
                              annualised_anywhere_in_code=False,
                              implied_unit="tonnes/day (summed over streams)")])


def run():
    rows = load_rows()
    cy = feature_panel(rows)
    e1 = e1_baselines(cy); save_csv(e1, "verification_v2_baselines.csv")
    e2 = e2_audit(rows, cy); save_csv(e2, "verification_data_audit.csv")
    e3 = e3_units(); save_csv(e3, "verification_units.csv")
    print(e1.to_string(index=False))
    return e1, e2, e3


if __name__ == "__main__":
    run()
