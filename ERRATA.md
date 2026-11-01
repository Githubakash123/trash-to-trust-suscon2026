# Errata: where the paper and the code disagree

This file lists every place where the SUSCON XIII full paper (version of 4 September 2026) says something that the code in this repository does not do, or describes a regenerated number with the wrong unit or scope. Each entry has an ID. `verification/verify_claims.py` checks the claims that carry it, and `VERIFICATION_REPORT.md` shows which ones.

Numbers that regenerate exactly and are described correctly are not listed here. There are 32 of them (see the report).

## Before anything else: what the dataset is

The 850-row Kaggle panel (`Waste_Management_and_Recycling_India.csv`, MIT licence, Krishna Yadu) does not measure Indian municipal performance. An audit of every quantitative field found the following:

- Tonnage has no relationship with any covariate: max |r| = 0.043 at row level. Its distribution fits a uniform over its observed range (KS p = 0.89).
- `Disposal Method` is independent of `Waste Type` (χ² p = 0.55). The panel contains physically impossible rows: 39 plastic and 40 e-waste records sent to composting, and 35 hazardous records sent to recycling.
- The 34 cities cannot be told apart by circularity (ANOVA p = 0.57).
- The landfill coordinates fall far from their cities: Mumbai's is 607 km away, Delhi's 1,437 km, Chennai's 1,823 km and Indore's 1,970 km.

All of these figures are recomputed on every run by `verification/extra_checks.py` (`outputs/verification_data_audit.csv`).

Treat the panel as a **testbed with known-noise ground truth**: it shows how much an unvalidated pipeline over-claims. Do not read it as evidence about Indian cities. The paper's Section 4.2 says the tonnage column fails its audit. The fallback it gives ("composition, treatment shares and index dynamics survive") does not hold either, because those fields are synthetic too.

## Units, scope and rounding

| ID | Paper says | Code does | Correction |
|---|---|---|---|
| E-UNIT | Table 3: walk-forward MAE in **t/year**; conformal half-width "11,432 t" | The source column is `Waste Generated (Tons/Day)`. A city-year value is the sum over the 5 streams. Nothing annualises it. | Units are **t/day (summed over five streams)**. The same error is in the IndoML abstract's Table 1. |
| E-FOLDS | 4.3: MAE "for the final held-out year across 34 cities" | MAE is pooled over the three walk-forward test years 2021–2023: **102 city-years** | "…across 102 city-year predictions (test years 2021–2023)". The 68/34 split applies only to the conformal interval. |
| E-D4 | 4.2: correlations with every covariate below 0.05 | True at row level (max 0.043). At the city-year level the forecasts use, max \|r\| = 0.147 (cost per tonne). | State the level: "at the record level". |
| E-ROUND | Table 8: headline falls "by a factor of about 3.8" | 29.93 % / 7.773 % = **3.85** | "about 3.9", or "nearly fourfold", which the text already says |

## Methods described but not implemented

| ID | Paper says | Code does |
|---|---|---|
| E-GRID | Table 1: CEA grid factor 0.7117 kgCO₂/kWh "applied directly; no transfer required" | `lca_inventory.csv` samples the WtE grid factor as triangular(0.71, 0.79, 0.90). Table 4's WtE interval comes from that sampling. |
| E-LCA | 3.3: 23 inventory parameters propagated to every pathway | 23 parameters are sampled (correct). Only WtE and landfill are built up mechanistically. Mechanical recycling, composting, AD and advanced recycling are each **a single assumed net factor** sampled from a triangular range (e.g. recycling −650/−500/−350). Their Table 4 rows restate those assumptions; they are not inventory results. |
| E-FUZZY | 3.6: **fuzzy** TOPSIS with triangular numbers from the Monte Carlo percentiles; criteria include a social / informal-livelihood criterion; ranking "inherits LCA uncertainty" (Table 2) | Classic crisp TOPSIS on a fixed literature matrix (`TECHNOLOGIES`). The criteria are circularity, emissions, scalability, cost and maturity. There is no social criterion, and no input comes from the LCA Monte Carlo. Table 6's numbers are correct for crisp TOPSIS. |
| E-RL | 3.7: actions are segregation campaigns, capacity additions, informal-sector integration and tariff changes. 4.9: the policy "dominates no-action, random and expert-heuristic baselines" | Actions are four intensity levels `["none", "low", "med", "high"]` on one circularity state. No baseline comparison is computed anywhere. The OPE numbers (−0.442, 0.155, 0.161) are correct. |
| E-SD | 3.4: four stocks: unmanaged waste, treatment capacity, municipal budget, informal workforce, coupled by a sigmoid pressure function | Two stocks, `Ccap(t)` and `Sland(t)`, following first-order exponential adoption. The synthetic-recovery biases (V7) are correct for this model. |
| E-GP | 3.8 / 4.11 / 4.15: **preemptive** goal programming over diversion, GHG, cost, land take and **informal employment**; "all five goals attained, 60/25/5/5/5" | No goal programme exists in the code. Stage 07 is a single-objective LP (maximise circularity subject to an emissions cap and share bounds), and it gives 55/10/35 today. The 60/25/5/5/5 allocation and the "five goals met" table survive only as pre-pipeline Colab outputs (`reference/legacy_colab_a70d27d/`). The goals in that table are emissions, circularity, leakage, cost and landfill, not informal employment. Its own solution is 55/33/5/5/2, not 60/25/5/5/5. The equity conclusion in 4.15 therefore has no computational support. |
| E-INDORE | Table 9: Indore allocation 40/35/25 | There is no Indore-specific allocation. 40/35/25 is the **national** LP result of the pre-pipeline Colab run (`reference/legacy_colab_a70d27d/07_optimized_allocation.csv`). The current code gives 55/10/35. |
| E-ROUTE | 4.11 / 6.1: "collection routing" for a city, 31,010 → 23,570 km, 24.0 %, 7.0 tCO₂ | The numbers regenerate. The network is **34 cities (inter-city)**, with the depot at the node nearest the centroid, using haversine distances between the dataset's landfill coordinates, which are random points. This is not municipal collection, and the saving is OR-Tools versus a greedy heuristic, not versus current practice. OR-Tools runs on a time limit, so the optimised distance can vary slightly on a much slower or faster machine. |
| E-V8 | 4.12: the 7.8 % GHG reduction is "propagated through the coupled chain, with emission factors, substitution credits and parameter uncertainty carried forward" | `ablations.uncertainty_propagation` computes `GHG reduction = coupling × (circularity gain)`, where **coupling ~ triangular(0.45, 0.60, 0.75) is an assumption**. No LCA emission factor or substitution credit enters. 7.8 % ≈ 0.60 × (0.561 − 0.432). The interval reflects the spread of the assumed coupling, not of the chain. The ~fourfold gap between diversion and climate is therefore set by an assumed number, not found. |
| E-V1 | 4.5: R² = 0.26 from a single random split | The figure exists only in a pre-pipeline Colab output (`reference/legacy_colab_a70d27d/quantity_prediction_model_metrics.csv`: R² 0.2627, MAE 1,886). No code in this repository, or in the source commit, produces it, so the split, features and target cannot be checked. The *validated* side of V1 holds: under walk-forward evaluation the boosted model's R² is −0.23. |

## Code notes (no paper claim depends on them)

- `circular_msw_pipeline.py` line 213: `net_emissions_row = gross_emissions - 0.0 * recovered_weighted`, so net equals gross for all 850 rows.
- The pipeline's stage 04 always falls back to a correlation matrix ("ARDL requires more time-series observations"). There is no ARDL or PMG estimate behind the "ramp rates bounded by estimated adjustment speeds" in 3.8 and Table 2.

## What holds up

Once the units are corrected, V2 (pooled-mean baseline), V3 (conformal coverage 0.912), V4 (accounting flip), V5 (rank stability), V6 (OPE errors) and V7 (synthetic-recovery bias) all regenerate exactly. The V2 reversal is computed **in this repository** (`verification/extra_checks.py`) on the paper's own walk-forward folds: pooled mean MAE 5,408.7 against boosting 5,869.2. It does not depend on the companion manuscript.
