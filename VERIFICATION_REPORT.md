# Verification report: SUSCON XIII full paper

*Trash to Trust: Layer-Wise Validation of an Integrated Decision Framework for Circular and Equitable Municipal Solid Waste Management in India* (A. Kundu)

**Result: PASS** - 61 of 61 claims landed in their documented status.

PASS means every regenerated number matches the paper within its rounding and every known discrepancy is the one documented in `ERRATA.md`. It does not mean every sentence of the paper is correct.

| Status | Claims |
|---|---:|
| REPRODUCED | 32 |
| REPRODUCED_WITH_CORRECTION | 16 |
| LEGACY_ONLY | 2 |
| NOT_SUPPORTED | 11 |

Run: 2026-11-01T11:50:04+00:00 - Python 3.14.6 on Windows-11-10.0.26300-SP0 - 5.1 min. Byte-identical to the paper's run: 30 of 30 tables; numerically equal (rtol 1e-6): 30.

## Claim by claim

| ID | Section | Claim | Paper | Regenerated | Status | Erratum |
|---|---|---|---|---|---|---|
| D1 | 3.1 | Panel size: city-stream-year records | 850 | 850 | REPRODUCED |  |
| D2 | 3.1 | Number of cities | 34 | 34 | REPRODUCED |  |
| D3 | 3.1 | Years covered | 2019-2023 | 2019-2023 | REPRODUCED |  |
| D4 | 4.2 | Tonnage |r| with every covariate below 0.05 (row level) | 0.05 | 0.0429 | REPRODUCED_WITH_CORRECTION | E-D4 |
| D5 | 4.2 | No year-to-year persistence (mean within-city lag-1 autocorrelation <= 0) | 0 | -0.1599 | REPRODUCED |  |
| D6 | Table 1 | CEA grid factor 0.7117 kgCO2/kWh 'applied directly' | 0.7117 fixed | 0.71/0.79/0.9 | NOT_SUPPORTED | E-GRID |
| F1 | Table 3 | Walk-forward MAE, naive persistence | 7629 | 7,628.8 | REPRODUCED_WITH_CORRECTION | E-UNIT |
| F2 | Table 3 | Walk-forward MAE, city historical mean | 6543 | 6,543.3 | REPRODUCED_WITH_CORRECTION | E-UNIT |
| F3 | Table 3 | Walk-forward MAE, gradient boosting | 5869 | 5,869.2 | REPRODUCED_WITH_CORRECTION | E-UNIT |
| F4 | Table 3 | Walk-forward MAE, ensemble (boosting + city mean) | 5867 | 5,867.4 | REPRODUCED_WITH_CORRECTION | E-UNIT |
| F5 | Table 3 | Skill vs persistence, city mean (%) | 14.2 | 14.23 | REPRODUCED |  |
| F6 | 4.3 | Skill vs persistence, gradient boosting (%) | 23.1 | 23.07 | REPRODUCED |  |
| F7 | 4.3 | Skill of gradient boosting vs city mean (%) | 10.3 | 10.3021 | REPRODUCED |  |
| F8 | 4.3 | MAE evaluated on 'the final held-out year across 34 cities' (n predictions) | 34 | 102 | NOT_SUPPORTED | E-FOLDS |
| F9 | Table 3 note | Split-conformal 90% half-width (t) | 11432 | 11,431.8 | REPRODUCED_WITH_CORRECTION | E-UNIT |
| F10 | Table 3 note | Split-conformal empirical coverage | 0.912 | 0.912 | REPRODUCED |  |
| F11 | Table 3 note | Calibration / evaluation city-years | 68/34 | 68/34 | REPRODUCED |  |
| F12 | 4.4 / V2 | Pooled mean is the strongest naive baseline | Pooled mean (all cities) | Pooled mean (all cities) | REPRODUCED |  |
| F13 | 4.4 / V2 | No learned model beats the strongest naive baseline | False | False | REPRODUCED |  |
| F14 | 4.4 / V2 | Learned model has negative walk-forward R2 (upper bound) | 0 | -0.2304 | REPRODUCED |  |
| F15 | 4.5 / V1 | R2 = 0.26 from a single random train-test split | 0.26 | 0.2627 | LEGACY_ONLY | E-V1 |
| L1 | Table 4 | Net GHG mechanical recycling (kgCO2e/t), mean / p05 / p95 | -500.1 / -602.2 / -397.2 | -500.1 / -602.2 / -397.2 | REPRODUCED_WITH_CORRECTION | E-LCA |
| L2 | Table 4 | Net GHG anaerobic digestion (kgCO2e/t), mean / p05 / p95 | -124.2 / -155.2 / -94 | -124.2 / -155.2 / -94 | REPRODUCED_WITH_CORRECTION | E-LCA |
| L3 | Table 4 | Net GHG composting (kgCO2e/t), mean / p05 / p95 | -40.4 / -60.9 / -19.7 | -40.4 / -60.9 / -19.7 | REPRODUCED_WITH_CORRECTION | E-LCA |
| L4 | Table 4 | Net GHG advanced recycling (kgCO2e/t), mean / p05 / p95 | 59.8 / -15.4 / 140.9 | 59.8 / -15.4 / 140.9 | REPRODUCED_WITH_CORRECTION | E-LCA |
| L5 | Table 4 | Net GHG waste-to-energy (kgCO2e/t), mean / p05 / p95 | 576.1 / 440.2 / 711.5 | 576.1 / 440.2 / 711.5 | REPRODUCED |  |
| L6 | Table 4 | Net GHG landfill (kgCO2e/t), mean / p05 / p95 | 998.4 / 686.4 / 1,360.0 | 998.4 / 686.4 / 1,360.0 | REPRODUCED |  |
| L7 | 3.3 | Uncertain inventory parameters propagated by Monte Carlo | 23 | 23 | REPRODUCED |  |
| M1 | Table 5 / V4 | Rank 1 under facility-gate, cost-weighted accounting | Decentralized composting | Decentralized composting | REPRODUCED |  |
| M2 | Table 5 / V4 | Rank 1 under system-wide accounting | Mechanical plastic recycling | Mechanical plastic recycling | REPRODUCED |  |
| M3 | Table 6 / V5 | Fuzzy TOPSIS: P(rank 1 retained), mean Spearman vs base | 0.99 / 0.982 | 0.99 / 0.982 | REPRODUCED_WITH_CORRECTION | E-FUZZY |
| M4 | Table 6 / V5 | SAW: P(rank 1 retained), mean Spearman vs base | 0.999 / 0.957 | 0.999 / 0.957 | REPRODUCED |  |
| M5 | Table 6 / V5 | WPM: P(rank 1 retained), mean Spearman vs base | 0.999 / 0.975 | 0.999 / 0.975 | REPRODUCED |  |
| M6 | Table 6 / V5 | VIKOR: P(rank 1 retained), mean Spearman vs base | 0.959 / 0.935 | 0.959 / 0.935 | REPRODUCED |  |
| M7 | 4.8 | All four methods agree on the leading alternative | True | True | REPRODUCED |  |
| M8 | 3.6 | Fuzzy TOPSIS with triangular numbers from Monte Carlo percentiles | True | False | NOT_SUPPORTED | E-FUZZY |
| M9 | 3.6 | Ranking includes a social / informal-livelihood criterion | True | False | NOT_SUPPORTED | E-FUZZY |
| R1 | 4.9 / V6 | On-policy Monte Carlo value of learned policy | -0.442 | -0.442 | REPRODUCED |  |
| R2 | 4.9 / V6 | WIS absolute error | 0.155 | 0.155 | REPRODUCED |  |
| R3 | 4.9 / V6 | FQE absolute error | 0.161 | 0.161 | REPRODUCED |  |
| R4 | 4.9 | Policy dominates no-action, random and expert-heuristic baselines | True | False | NOT_SUPPORTED | E-RL |
| R5 | 3.7 | Actions = segregation, capacity, informal integration, tariff | 4 named levers | ["none", "low", "med", "high"] | NOT_SUPPORTED | E-RL |
| S1 | 4.10 / V7 | Synthetic-recovery bias, c0 | -0.003 | -0.0025 | REPRODUCED |  |
| S2 | 4.10 / V7 | Synthetic-recovery bias, cinf | 0.003 | 0.0026 | REPRODUCED |  |
| S3 | 4.10 / V7 | Synthetic-recovery bias, k | 0.008 | 0.0081 | REPRODUCED |  |
| S4 | 3.4 | SD stocks: unmanaged waste, capacity, budget, informal workforce | 4 stocks | Ccap, Sland | NOT_SUPPORTED | E-SD |
| A1 | 4.11 | Goal-programming allocation recycling/composting/AD/incineration/landfill (%) | 60/25/5/5/5 | 60/25/5/5/5 | LEGACY_ONLY | E-GP |
| A2 | 3.8 / 4.15 | GP goals include informal employment | includes informal employment | G1 emissions <= 65% of baseline; G2 circularity >= 65% of waste; G3 leakage <= 40% of baseline; G4 cost <= 115% of baseline; G5 landfill <= 15% of waste | NOT_SUPPORTED | E-GP |
| A3 | 3.8 | Allocation solved as preemptive goal programming | True | False | NOT_SUPPORTED | E-GP |
| A4 | Table 9 | Indore allocation shares 40/35/25 | 40/35/25 | 55/10/35 | NOT_SUPPORTED | E-INDORE |
| A5 | 4.11 | Routing distance greedy -> optimised (km) | 31010 / 23570 | 31,009.9 / 23,570.3 | REPRODUCED_WITH_CORRECTION | E-ROUTE |
| A6 | 4.11 | Routing distance reduction (%) | 24 | 24 | REPRODUCED_WITH_CORRECTION | E-ROUTE |
| A7 | 4.11 | CO2 avoided per cycle (t) | 7 | 6.9784 | REPRODUCED_WITH_CORRECTION | E-ROUTE |
| P1 | Table 7 | Circularity index 2023 baseline | 0.432 | 0.4318 | REPRODUCED |  |
| P2 | Table 7 | Circularity 2035 mean / p05 / p95 | 0.561 / 0.531 / 0.591 | 0.561 / 0.531 / 0.591 | REPRODUCED |  |
| P3 | Table 7 | Plastic circularity 2035 mean / p05 / p95 | 0.842 / 0.796 / 0.888 | 0.842 / 0.796 / 0.888 | REPRODUCED |  |
| P4 | Table 7 / V8 | GHG reduction vs BAU 2035 (%), mean / p05 / p95 | 7.8 / 5.7 / 10.1 | 7.773 / 5.674 / 10.095 | REPRODUCED_WITH_CORRECTION | E-V8 |
| P5 | 4.12 | Diversion improvement 0.432 -> 0.561 (%) | 30 | 29.9286 | REPRODUCED |  |
| P6 | Table 8 / V8 | Headline falls by a factor of about 3.8 | 3.8 | 3.8503 | REPRODUCED_WITH_CORRECTION | E-ROUND |
| P7 | Methods | GHG response = assumed coupling x circularity gain (triangular low, mode, high) | propagated through LCA factors | 0.45, 0.60, 0.75 | NOT_SUPPORTED | E-V8 |
| X1 | 3.10 | End-to-end runtime (minutes), informational | 4 | 5.1 | REPRODUCED |  |

Evidence file for each row: `outputs/claims_check.csv`. Errata: `ERRATA.md`.
