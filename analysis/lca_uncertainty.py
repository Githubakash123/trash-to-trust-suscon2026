"""
LCA inventory with uncertainty + sensitivity
============================================
Addresses reviewer points:
  * "no audited LCA inventories, process parameters, or facility-level data;
     impact results not reported with uncertainty ranges"
  * "technology performance parameters (capture efficiencies, LHV, emission
     factors) with uncertainty ranges and sensitivity results"
  * "Uncertainty propagation stated conceptually but not reported with concrete
     intervals"

What this does
--------------
1. Publishes an explicit LCA inventory: every parameter gets a low/mode/high
   (triangular) range plus a source tag (`lca_inventory.csv`).
2. Computes net life-cycle GHG MECHANISTICALLY for the two energy/disposal
   pathways the reviewer named:
       Waste-to-energy: net = EF_process - LHV*eff_elec*grid_EF (energy credit)
       Landfill+gas:    net = (1-capture)*L0*MCF*GWP_CH4
   and propagates parameter uncertainty by Monte Carlo.
3. For material pathways (recycling, composting, AD, advanced) propagates
   triangular literature ranges directly.
4. Reports per-pathway net-GHG and cost 90% intervals (`lca_impact_intervals.csv`)
   and a one-at-a-time tornado sensitivity for WtE (`lca_tornado_wte.csv`).

Parameter sources (see METHODOLOGY.md References): EEA net-emissions-per-option;
IPCC 2006 Vol.5 Ch.3 (landfill CH4, L0, MCF) & Ch.5; Sharma & Chandel 2017
(Mumbai MSW LCA); CEA India grid emission factor 2022-23 (~0.71-0.85 kgCO2/kWh);
ACS EST 2020 organics; Asian MSW treatment cost review 2017.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.common import save_csv

RNG = np.random.default_rng(2024)
N_MC = 20000
GWP_CH4 = 28.0   # IPCC AR5 100-yr


# (param, low, mode, high, unit, source)
INVENTORY = pd.DataFrame([
    # Waste-to-energy mechanistic params
    ("WtE", "LHV_MJ_per_kg",      7.5,  9.0, 10.5, "MJ/kg", "Indian mixed MSW LHV"),
    ("WtE", "eff_electric",       0.16, 0.21, 0.26, "frac",  "mass-burn net elec eff"),
    ("WtE", "grid_EF_kgCO2_kWh",  0.71, 0.79, 0.90, "kgCO2/kWh", "CEA India 2022-23"),
    ("WtE", "EF_process_kgCO2_t", 850., 990., 1150., "kgCO2/t", "fossil-C combustion"),
    ("WtE", "cost_usd_t",         90.,  110., 140.,  "USD/t", "Asian MSW cost review"),
    ("WtE", "recovery_pct",       18.,  25.,  32.,   "%",     "energy recovery only"),
    # Landfill + gas-capture mechanistic params
    ("Landfill", "L0_kgCH4_t",    50.,  62.,  75.,   "kgCH4/t", "IPCC L0 (DOC-based)"),
    ("Landfill", "MCF",           0.6,  0.8,  1.0,   "frac",  "methane correction factor"),
    ("Landfill", "gas_capture",   0.0,  0.30, 0.55,  "frac",  "Indian semi-aerobic capture"),
    ("Landfill", "cost_usd_t",    40.,  60.,  80.,   "USD/t", "Asian MSW cost review"),
    ("Landfill", "recovery_pct",  2.,   5.,   10.,   "%",     "marginal recovery"),
    # Triangular net-GHG (kgCO2e/t) for material/biological pathways
    ("Mechanical recycling", "ghg_net_kgCO2e_t", -650., -500., -350., "kgCO2e/t", "virgin-displacement credit"),
    ("Mechanical recycling", "cost_usd_t",       95.,   120.,  150.,  "USD/t", "Asian cost review"),
    ("Mechanical recycling", "recovery_pct",     78.,   85.,   90.,   "%",    "MRF yield"),
    ("Composting", "ghg_net_kgCO2e_t",           -70.,  -41.,  -10.,  "kgCO2e/t", "EEA/ACS organics"),
    ("Composting", "cost_usd_t",                 20.,   30.,   45.,   "USD/t", "Asian cost review"),
    ("Composting", "recovery_pct",               50.,   60.,   70.,   "%",    "compost yield"),
    ("Anaerobic digestion", "ghg_net_kgCO2e_t",  -170., -123., -80.,  "kgCO2e/t", "biogas credit"),
    ("Anaerobic digestion", "cost_usd_t",        55.,   70.,   90.,   "USD/t", "Asian cost review"),
    ("Anaerobic digestion", "recovery_pct",      68.,   75.,   82.,   "%",    "digestate+energy"),
    ("Advanced recycling", "ghg_net_kgCO2e_t",   -50.,  50.,   180.,  "kgCO2e/t", "pyrolysis range"),
    ("Advanced recycling", "cost_usd_t",         160.,  200.,  260.,  "USD/t", "chemical recycling"),
    ("Advanced recycling", "recovery_pct",       55.,   70.,   80.,   "%",    "monomer yield"),
], columns=["pathway", "param", "low", "mode", "high", "unit", "source"])


def tri(p):
    """Triangular draws for one inventory row."""
    return RNG.triangular(p["low"], p["mode"], p["high"], size=N_MC)


def _row(pathway, param):
    return INVENTORY[(INVENTORY.pathway == pathway) & (INVENTORY.param == param)].iloc[0]


def wte_net_ghg(draws=None):
    lhv = tri(_row("WtE", "LHV_MJ_per_kg"))
    eff = tri(_row("WtE", "eff_electric"))
    grid = tri(_row("WtE", "grid_EF_kgCO2_kWh"))
    efp = tri(_row("WtE", "EF_process_kgCO2_t"))
    kwh_per_t = lhv * 1000.0 / 3.6 * eff        # MJ/kg -> kWh/t * efficiency
    credit = kwh_per_t * grid
    return efp - credit                          # kgCO2e/t (positive = net emit)


def landfill_net_ghg():
    l0 = tri(_row("Landfill", "L0_kgCH4_t"))
    mcf = tri(_row("Landfill", "MCF"))
    cap = tri(_row("Landfill", "gas_capture"))
    return (1 - cap) * l0 * mcf * GWP_CH4         # kgCO2e/t


def run():
    save_csv(INVENTORY, "lca_inventory.csv")

    # Monte-Carlo net GHG per pathway (keys match the inventory `pathway` labels)
    ghg = {"WtE": wte_net_ghg(), "Landfill": landfill_net_ghg()}
    for pw in ["Mechanical recycling", "Composting", "Anaerobic digestion",
               "Advanced recycling"]:
        ghg[pw] = tri(_row(pw, "ghg_net_kgCO2e_t"))

    cost = {}
    for pw in ["WtE", "Landfill", "Mechanical recycling", "Composting",
               "Anaerobic digestion", "Advanced recycling"]:
        cost[pw] = tri(_row(pw, "cost_usd_t"))

    DISPLAY = {"WtE": "Waste-to-energy"}

    rows = []
    for pw in ghg:
        g, c = ghg[pw], cost[pw]
        rows.append(dict(
            pathway=DISPLAY.get(pw, pw),
            ghg_net_mean=round(float(g.mean()), 1),
            ghg_net_p05=round(float(np.percentile(g, 5)), 1),
            ghg_net_p95=round(float(np.percentile(g, 95)), 1),
            cost_mean=round(float(c.mean()), 1),
            cost_p05=round(float(np.percentile(c, 5)), 1),
            cost_p95=round(float(np.percentile(c, 95)), 1),
        ))
    impact = pd.DataFrame(rows).sort_values("ghg_net_mean").reset_index(drop=True)
    save_csv(impact, "lca_impact_intervals.csv")

    # Tornado: one-at-a-time effect on WtE net GHG (vary each param low<->high,
    # others at mode), measured as swing in kgCO2e/t.
    base = dict(lhv=_row("WtE", "LHV_MJ_per_kg")["mode"],
                eff=_row("WtE", "eff_electric")["mode"],
                grid=_row("WtE", "grid_EF_kgCO2_kWh")["mode"],
                efp=_row("WtE", "EF_process_kgCO2_t")["mode"])

    def wte_point(lhv, eff, grid, efp):
        return efp - (lhv * 1000 / 3.6 * eff) * grid

    base_val = wte_point(**base)
    tor = []
    for nm, key, prow in [("LHV", "lhv", _row("WtE", "LHV_MJ_per_kg")),
                          ("eff_electric", "eff", _row("WtE", "eff_electric")),
                          ("grid_EF", "grid", _row("WtE", "grid_EF_kgCO2_kWh")),
                          ("EF_process", "efp", _row("WtE", "EF_process_kgCO2_t"))]:
        lo = dict(base); lo[key] = prow["low"]
        hi = dict(base); hi[key] = prow["high"]
        v_lo, v_hi = wte_point(**lo), wte_point(**hi)
        tor.append(dict(parameter=nm, low_value=round(v_lo, 1), high_value=round(v_hi, 1),
                        swing_kgCO2e_t=round(abs(v_hi - v_lo), 1)))
    tornado = pd.DataFrame(tor).sort_values("swing_kgCO2e_t", ascending=False)
    save_csv(tornado, "lca_tornado_wte.csv")

    print("\nLCA UNCERTAINTY SUMMARY (net kgCO2e/t, 90% interval)")
    for _, r in impact.iterrows():
        print(f"  {r['pathway']:22s} {r['ghg_net_mean']:8.0f}  "
              f"[{r['ghg_net_p05']:.0f}, {r['ghg_net_p95']:.0f}]")
    print(f"  WtE base net GHG = {base_val:.0f} kgCO2e/t; top driver: "
          f"{tornado.iloc[0]['parameter']} (swing {tornado.iloc[0]['swing_kgCO2e_t']:.0f})")
    return impact, tornado


if __name__ == "__main__":
    run()
