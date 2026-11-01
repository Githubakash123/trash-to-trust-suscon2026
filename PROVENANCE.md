# Provenance

## Dataset

| | |
|---|---|
| File | `Waste_Management_and_Recycling_India.csv` (850 rows × 13 columns) |
| Source | Kaggle, *Waste Management and Recycling in Indian Cities*, Krishna Yadu: https://www.kaggle.com/datasets/krishnayadav456wrsty/waste-management-and-recycling-in-indian-cities |
| Licence | MIT (from the Kaggle dataset metadata, checked 2026-11-01). Redistributed here unchanged. |
| SHA-256 (LF line endings) | `66a7ed50099d0f568298100d29879566ce685933d82a8ae5b1348b41ae5275f7` |
| Checked by | `reproduce.py`, which stops if the file differs |

The data are synthetic in every quantitative field: tonnage, disposal method and coordinates (see ERRATA.md). Use the panel as a testbed for the validation protocol, not as a measurement of Indian cities. If you cite it, cite the Kaggle page above.

## Code

The analysis code is copied **unchanged** from the commit that produced the paper's numbers: commit `1f5883c` (21 June 2026) in the author's private working repository. Git blob IDs are content hashes, so `git ls-files -s` in this repository shows the same IDs:

| File | git blob (identical in source commit 1f5883c) |
|---|---|
| `circular_msw_pipeline.py` | `ac9541952d73328c1ef889f8460c9a8825354248` |
| `circular_msw_figures.py` | `d8c856e2b64324b1c03588f62087117cf1e033c2` |
| `analysis/ablations.py` | `f3578201dfe969525d476b7fb232814a85907e8a` |
| `analysis/common.py` | `9abb9aaf646091bbd0d5424e4d0c610406784f07` |
| `analysis/figures.py` | `b2f79df5e6e904a7a33b2727f1a648810b1bc9dd` |
| `analysis/forecasting_plus.py` | `5cb56c09c466b1f2ef0b30396dc365eabec46d61` |
| `analysis/lca_uncertainty.py` | `8146ac469b50143ddc766472ce63e732fa5151cf` |
| `analysis/mcdm_compare.py` | `af5110d9eaecfe9aa1731593d2fa0da9170a4781` |
| `analysis/pci_mci.py` | `e39c626f26e980c65a0f1f0a7454a6010bdb7c7b` |
| `analysis/rl_policy.py` | `6aa8fee891a43af44f76552415b371f96443daa9` |
| `analysis/routing.py` | `79ed641b474de43b64aefa984003394e7d128cef` |
| `analysis/run_all.py` | `6359853d7343f45edc11e589a7ad9e4aaff771c1` |
| `analysis/sd_model.py` | `fb93f17eff665ef8402e523c6c88d3dcdab10cae` |
| `Waste_Management_and_Recycling_India.csv` | `f734a83e501766f98816382cc99d65e1ccbc9e8c` |

Added for this repository, and not part of the paper's run: `reproduce.py`, `verification/`, `notebooks/`, `.github/`, and the documentation files.

`circular_msw_figures.py` and `analysis/figures.py` are kept for completeness. The paper has no figures, so `reproduce.py` does not call them.

## Reference outputs

| Folder | What it is |
|---|---|
| `reference/paper_run_1f5883c/` | The 30 analysis tables committed with the code in `1f5883c`. These are the tables the paper's numbers were taken from. `reproduce.py` compares each fresh run against them. |
| `reference/legacy_colab_a70d27d/` | Outputs from the earlier Google Colab notebook (commit `a70d27d`, 20 June 2026). **No code in this repository produces them.** They are kept because the paper quotes three numbers that exist only here: R² = 0.26, the 60/25/5/5/5 allocation, and 40/35/25. The panel file in this folder also contains a spurious year-1970 row per city. |
