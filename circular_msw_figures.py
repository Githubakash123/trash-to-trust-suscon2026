"""
Interactive figures for the Circular MSW study (Plotly -> standalone HTML)
=========================================================================
Generates the interactive HTML figures listed in the paper's figure index,
reading the CSV artefacts produced by `circular_msw_pipeline.py`.

    fig01  dataset inventory (bar)
    fig04  circular waste-flow Sankey (waste type -> method -> fate)
    fig11  waste-type composition treemap
    fig12  yearly waste-management trends
    fig13  geospatial high-risk map (scatter_geo / scatter_mapbox fallback)
    fig14  next-year city forecast
    fig15  prediction-model metrics
    fig17  waste-type -> method sunburst
    fig19  high-risk priority ranking

Usage
-----
    pip install plotly
    python circular_msw_pipeline.py --data Waste_Management_and_Recycling_India.csv --out outputs
    python circular_msw_figures.py --out outputs --figdir figures
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Plotly is required:  pip install plotly") from exc


# --- risk model used by the geospatial + priority figures --------------------
RISK_WEIGHTS = dict(landfill=0.5, plastic_leakage=0.3, emissions=0.2)


def _read(out_dir: str, name: str) -> pd.DataFrame:
    path = os.path.join(out_dir, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing pipeline output: {path}. Run circular_msw_pipeline.py first.")
    return pd.read_csv(path)


def _save(fig, fig_dir: str, name: str):
    html = os.path.join(fig_dir, name + ".html")
    fig.write_html(html, include_plotlyjs="cdn")
    print(f"  -> {name}.html")
    # Static PNG too, if a kaleido engine is installed (optional).
    try:
        fig.write_image(os.path.join(fig_dir, name + ".png"), scale=2)
    except Exception:
        pass


# ----------------------------------------------------------------------------
def fig01_dataset_inventory(ind: pd.DataFrame, fig_dir: str):
    inv = (ind.groupby("waste_type")["quantity"].sum()
           .sort_values(ascending=False).reset_index())
    fig = px.bar(inv, x="waste_type", y="quantity", text_auto=".2s",
                 color="waste_type", title="Dataset inventory — total quantity by waste type")
    fig.update_layout(showlegend=False, xaxis_title="Waste type", yaxis_title="Tons")
    _save(fig, fig_dir, "fig01_dataset_inventory")


def fig04_waste_flow_sankey(ind: pd.DataFrame, fig_dir: str):
    """waste type -> disposal method -> fate (recovered / leaked)."""
    d = ind.copy()
    d["fate"] = np.where(d["is_landfill_or_leakage"], "Leaked / landfilled", "Recovered loop")

    labels, idx = [], {}

    def node(name):
        if name not in idx:
            idx[name] = len(labels)
            labels.append(name)
        return idx[name]

    src, tgt, val = [], [], []
    for (wt, m), q in d.groupby(["waste_type", "method"])["quantity"].sum().items():
        src.append(node(wt.title())); tgt.append(node(m.title())); val.append(q)
    for (m, f), q in d.groupby(["method", "fate"])["quantity"].sum().items():
        src.append(node(m.title())); tgt.append(node(f)); val.append(q)

    fig = go.Figure(go.Sankey(
        node=dict(label=labels, pad=15, thickness=18,
                  color=["#2c7fb8" if l in ("Recovered loop",) else
                         "#d95f0e" if l == "Leaked / landfilled" else "#41ab5d"
                         for l in labels]),
        link=dict(source=src, target=tgt, value=val)))
    fig.update_layout(title_text="Circular waste-flow: type → method → fate", font_size=11)
    _save(fig, fig_dir, "fig04_circular_waste_flow_sankey")


def fig11_treemap(summary: pd.DataFrame, fig_dir: str):
    fig = px.treemap(summary, path=["waste_type"], values="total_waste_tons",
                     color="estimated_emissions_tco2e", color_continuous_scale="OrRd",
                     title="Waste-type composition (size = tons, colour = emissions)")
    _save(fig, fig_dir, "fig11_waste_type_composition_treemap")


def fig12_yearly_trends(panel: pd.DataFrame, fig_dir: str):
    yr = panel.groupby("year").agg(total_waste=("total_waste", "sum"),
                                   recovered=("recovered_weighted", "sum"),
                                   emissions=("net_emissions", "sum")).reset_index()
    fig = go.Figure()
    fig.add_bar(x=yr["year"], y=yr["total_waste"], name="Total waste")
    fig.add_bar(x=yr["year"], y=yr["recovered"], name="Recovered")
    fig.add_scatter(x=yr["year"], y=yr["emissions"], name="Emissions (tCO2e)",
                    yaxis="y2", mode="lines+markers")
    fig.update_layout(title="Yearly waste-management trends", barmode="group",
                      yaxis_title="Tons", yaxis2=dict(title="tCO2e", overlaying="y", side="right"))
    _save(fig, fig_dir, "fig12_yearly_waste_management_trends")


def _risk_table(ind: pd.DataFrame) -> pd.DataFrame:
    d = ind.copy()
    d["landfill_q"] = d["quantity"].where(d["is_landfill_or_leakage"], 0.0)
    d["plastic_leak_q"] = d["quantity"].where(d["is_plastic"] & d["is_landfill_or_leakage"], 0.0)
    g = d.groupby("city").agg(
        landfill=("landfill_q", "sum"),
        plastic_leak=("plastic_leak_q", "sum"),
        emissions=("gross_emissions", "sum"),
        latitude=("latitude", "first"),
        longitude=("longitude", "first"),
        total=("quantity", "sum"),
    ).reset_index()
    for c in ["landfill", "plastic_leak", "emissions"]:
        m = g[c].max() or 1
        g[c + "_n"] = g[c] / m
    g["risk_score"] = (RISK_WEIGHTS["landfill"] * g["landfill_n"]
                       + RISK_WEIGHTS["plastic_leakage"] * g["plastic_leak_n"]
                       + RISK_WEIGHTS["emissions"] * g["emissions_n"])
    return g.sort_values("risk_score", ascending=False)


def fig13_geo_risk(ind: pd.DataFrame, fig_dir: str):
    g = _risk_table(ind).dropna(subset=["latitude", "longitude"])
    if g.empty:
        print("  -- fig13 skipped (no coordinates)")
        return
    fig = px.scatter_geo(g, lat="latitude", lon="longitude", size="total",
                         color="risk_score", hover_name="city", scope="asia",
                         color_continuous_scale="YlOrRd",
                         title="Geospatial high-risk map (landfill + leakage + emissions)")
    fig.update_geos(center=dict(lat=22, lon=80), projection_scale=3)
    _save(fig, fig_dir, "fig13_geospatial_high_risk_map")


def fig14_forecast(forecast: pd.DataFrame, fig_dir: str):
    top = forecast.sort_values("forecast_waste_tons", ascending=False).head(15)
    fig = go.Figure()
    fig.add_bar(y=top["city"], x=top["latest_waste_tons"], name="Latest", orientation="h")
    fig.add_bar(y=top["city"], x=top["forecast_waste_tons"], name="Forecast", orientation="h")
    fig.update_layout(title="Next-year city waste forecast (top 15)", barmode="group",
                      xaxis_title="Tons", yaxis=dict(autorange="reversed"))
    _save(fig, fig_dir, "fig14_next_year_city_forecast")


def fig15_metrics(comparison: pd.DataFrame, fig_dir: str):
    c = comparison.sort_values("backtest_MAE")
    fig = go.Figure()
    fig.add_bar(x=c["method"], y=c["backtest_MAE"], name="Backtest MAE (tons)")
    fig.add_scatter(x=c["method"], y=c["skill_vs_naive"], name="Skill vs naive",
                    yaxis="y2", mode="lines+markers")
    fig.update_layout(title="Forecast model backtest — MAE and skill vs naive baseline",
                      yaxis_title="MAE (tons)",
                      yaxis2=dict(title="Skill (1 - MAE/MAE_naive)", overlaying="y",
                                  side="right", zeroline=True))
    _save(fig, fig_dir, "fig15_prediction_model_metrics")


def fig17_sunburst(ind: pd.DataFrame, fig_dir: str):
    s = ind.groupby(["waste_type", "method"])["quantity"].sum().reset_index()
    s["waste_type"] = s["waste_type"].str.title()
    s["method"] = s["method"].str.title()
    fig = px.sunburst(s, path=["waste_type", "method"], values="quantity",
                      title="Waste type → disposal method sunburst")
    _save(fig, fig_dir, "fig17_waste_type_method_sunburst")


def fig05b_topsis_sensitivity(out_dir: str, fig_dir: str):
    s = _read(out_dir, "05b_topsis_sensitivity.csv")
    score_cols = [c for c in s.columns if c.startswith("score_")]
    fig = go.Figure()
    for c in score_cols:
        fig.add_bar(x=s["technology"], y=s[c], name=c.replace("score_", ""))
    fig.update_layout(title="TOPSIS technology ranking — weighting-scheme sensitivity",
                      barmode="group", yaxis_title="Closeness coefficient",
                      xaxis_tickangle=-25)
    _save(fig, fig_dir, "fig05b_topsis_sensitivity")


def fig19_risk_ranking(ind: pd.DataFrame, fig_dir: str):
    g = _risk_table(ind).head(20)
    fig = px.bar(g.sort_values("risk_score"), x="risk_score", y="city",
                 orientation="h", color="risk_score", color_continuous_scale="Reds",
                 title="High-risk priority ranking (top 20 cities)")
    _save(fig, fig_dir, "fig19_high_risk_priority_ranking")


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Generate interactive figures from pipeline outputs")
    ap.add_argument("--out", default="outputs", help="folder with pipeline CSV outputs")
    ap.add_argument("--figdir", default="figures", help="destination for HTML/PNG figures")
    args = ap.parse_args()
    os.makedirs(args.figdir, exist_ok=True)

    ind = _read(args.out, "02_detailed_indicators.csv")
    panel = _read(args.out, "03_city_year_panel.csv")
    forecast = _read(args.out, "next_year_city_forecast.csv")
    comparison = _read(args.out, "forecast_model_backtest_comparison.csv")
    wt_summary = (ind.groupby("waste_type").agg(
        total_waste_tons=("quantity", "sum"),
        estimated_emissions_tco2e=("gross_emissions", "sum")).reset_index())

    print("Generating interactive figures:")
    fig01_dataset_inventory(ind, args.figdir)
    fig05b_topsis_sensitivity(args.out, args.figdir)
    fig04_waste_flow_sankey(ind, args.figdir)
    fig11_treemap(wt_summary, args.figdir)
    fig12_yearly_trends(panel, args.figdir)
    fig13_geo_risk(ind, args.figdir)
    fig14_forecast(forecast, args.figdir)
    fig15_metrics(comparison, args.figdir)
    fig17_sunburst(ind, args.figdir)
    fig19_risk_ranking(ind, args.figdir)
    print(f"\nDone. Figures in: {os.path.abspath(args.figdir)}")


if __name__ == "__main__":
    main()
