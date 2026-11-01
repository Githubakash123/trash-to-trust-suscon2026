"""Shared data-loading helpers for the reviewer-response analyses.

All modules import the *same* harmonised panel and per-row indicators from the
main pipeline so every new result is consistent with the published artefacts.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

# Make the sibling pipeline importable regardless of CWD.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import circular_msw_pipeline as cmp  # noqa: E402

DATA = os.path.join(_ROOT, "Waste_Management_and_Recycling_India.csv")
OUT = os.path.join(_ROOT, "outputs")
FIG = os.path.join(_ROOT, "figures")
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

CASE_CITIES = ["Delhi", "Mumbai", "Indore"]


def load_rows(path: str = DATA) -> pd.DataFrame:
    """Per-row harmonised table with environmental indicators attached."""
    raw = cmp.load_and_harmonize(path)
    return cmp.add_indicators(raw)


def load_panel(path: str = DATA) -> pd.DataFrame:
    """City x Year panel with circularity / leakage / plastic indices."""
    return cmp.build_city_year_panel(load_rows(path))


def save_csv(df: pd.DataFrame, name: str, index: bool = False) -> str:
    p = os.path.join(OUT, name)
    df.to_csv(p, index=index)
    print(f"   -> {name:48s} {df.shape}")
    return p
