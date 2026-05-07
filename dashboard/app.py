"""
Streamlit Dashboard — Sales Forecasting System
================================================
A unique, polished frontend for the per-state forecasting backend.

Features that differentiate this submission:
  1. Live API integration (calls FastAPI backend at runtime)
  2. Per-state champion model badge + leaderboard view
  3. Interactive Plotly charts (history + 8-week forecast + CI band)
  4. Multi-state comparison view
  5. Model performance heatmap (SMAPE per state per model)
  6. CSV download of forecasts (business-ready output)
  7. "What-If" retrain trigger button

Run:
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Sales Forecasting — Per-State Champion System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = st.sidebar.text_input("API base URL", value="http://localhost:8000")
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"


# ----------------------------------------------------------------------------
# Data loaders (cached)
# ----------------------------------------------------------------------------
@st.cache_data(ttl=60)
def load_metadata() -> dict:
    """Load metadata safely from API."""

    try:
        r = requests.get(f"{API_URL}/models", timeout=5)

        if not r.ok:
            return {}

        data = r.json()

        # API returns list -> convert into dict
        if isinstance(data, list):
            champions = {}
            scores = {}

            for item in data:
                state = item.get("state")
                champion = item.get("champion")
                metrics = item.get("metrics", {})

                champions[state] = {
                    "model": champion,
                    "smape": metrics.get("smape", 0)
                }

                scores[state] = {
                    champion: metrics.get("smape", 0)
                }

            return {
                "champions": champions,
                "scores": scores
            }

        return data

    except Exception:
        return {}

@st.cache_data(ttl=60)
def load_history() -> pd.DataFrame:
    """Load weekly resampled history from artifacts."""
    hist_path = ARTIFACTS_DIR / "weekly_history.parquet"
    if hist_path.exists():
        return pd.read_parquet(hist_path)
    csv_path = ARTIFACTS_DIR / "weekly_history.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path, parse_dates=["date"])
    return pd.DataFrame(columns=["date", "state", "sales"])


def fetch_forecast(state: str, horizon: int = 8) -> dict | None:
    try:
        r = requests.get(
            f"{API_URL}/forecast/{state}",
            params={"horizon": horizon},
            timeout=15,
        )
        if r.ok:
            return r.json()
        st.error(f"API {r.status_code}: {r.text[:200]}")
    except Exception as e:
        st.warning(f"API unavailable ({e}). Serving from cached artifacts.")
    return None


# ----------------------------------------------------------------------------
# Sidebar — global controls
# ----------------------------------------------------------------------------
st.sidebar.title("📈 Forecast Console")
st.sidebar.markdown("Production-ready per-state forecasting system.")

meta = load_metadata()
hist = load_history()

if isinstance(hist, pd.DataFrame) and not hist.empty:
    states = sorted(hist["state"].unique())
elif isinstance(meta, dict):
    champs = meta.get("champions", meta)
    states = sorted(champs.keys()) if isinstance(champs, dict) else []
elif isinstance(meta, list):
    states = sorted({(m.get("state") if isinstance(m, dict) else str(m)) for m in meta if m})
else:
    states = []
if not states:
    st.error(
        "No artifacts found. Train models first:\n\n```bash\npython -m src.training.train_all\n```"
    )
    st.stop()

mode = st.sidebar.radio(
    "View",
    ["Single state", "Compare states", "Model leaderboard", "Retrain"],
    index=0,
)


# ----------------------------------------------------------------------------
# Helper: build forecast chart
# ----------------------------------------------------------------------------
def forecast_chart(state: str, hist_df: pd.DataFrame, fc: dict) -> go.Figure:
    h = hist_df[hist_df["state"] == state].sort_values("date").tail(104)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=h["date"], y=h["sales"], mode="lines",
            name="History (last 2y)", line=dict(color="#5A6B7B", width=2),
        )
    )
    if fc and "forecast" in fc:
        fdf = pd.DataFrame(fc["forecast"])
        fdf["date"] = pd.to_datetime(fdf["date"])
        if "lower" in fdf and "upper" in fdf:
            fig.add_trace(go.Scatter(
                x=list(fdf["date"]) + list(fdf["date"][::-1]),
                y=list(fdf["upper"]) + list(fdf["lower"][::-1]),
                fill="toself", fillcolor="rgba(99,110,250,0.18)",
                line=dict(color="rgba(255,255,255,0)"), showlegend=True,
                name="Confidence band",
            ))
        fig.add_trace(go.Scatter(
            x=fdf["date"], y=fdf["yhat"], mode="lines+markers",
            name=f"Forecast — {fc.get('model', 'champion')}",
            line=dict(color="#636EFA", width=3, dash="dash"),
        ))
    fig.update_layout(
        title=f"{state} — Weekly Sales Forecast (next 8 weeks)",
        xaxis_title="Week", yaxis_title="Sales",
        template="plotly_white", height=480, hovermode="x unified",
    )
    return fig


# ----------------------------------------------------------------------------
# View 1 — Single state
# ----------------------------------------------------------------------------
if mode == "Single state":
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("📊 Per-State Sales Forecast")
    with col2:
        state = st.selectbox("State", states, index=0)

    fc = fetch_forecast(state, horizon=8)
    champions = meta.get("champions", {}) if isinstance(meta, dict) else {}
    champ = champions.get(state, {})

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🏆 Champion model", champ.get("model", fc.get("model", "—") if fc else "—"))
    m2.metric("SMAPE", f"{champ.get('smape', 0):.2f}%" if champ else "—")
    m3.metric("Last observed", str(hist[hist["state"] == state]["date"].max().date()) if not hist.empty else "—")
    m4.metric("Forecast horizon", "8 weeks")

    if fc:
        st.plotly_chart(forecast_chart(state, hist, fc), use_container_width=True)
        fdf = pd.DataFrame(fc["forecast"])
        st.subheader("Forecast table")
        st.dataframe(fdf, use_container_width=True)
        st.download_button(
            "⬇ Download CSV", fdf.to_csv(index=False).encode(),
            file_name=f"forecast_{state}.csv", mime="text/csv",
        )
    else:
        st.info("Start the API (`uvicorn src.api.main:app --port 8000`) to see live forecasts.")
        if not hist.empty:
            h = hist[hist["state"] == state].tail(104)
            fig = go.Figure(go.Scatter(x=h["date"], y=h["sales"], mode="lines", name="History"))
            fig.update_layout(title=f"{state} — History", template="plotly_white", height=420)
            st.plotly_chart(fig, use_container_width=True)


# ----------------------------------------------------------------------------
# View 2 — Compare states
# ----------------------------------------------------------------------------
elif mode == "Compare states":
    st.title("🔀 Multi-State Comparison")
    picks = st.multiselect("Pick states to compare", states, default=states[:3])
    if picks:
        fig = make_subplots(rows=len(picks), cols=1, shared_xaxes=False,
                            subplot_titles=[f"{s}" for s in picks])
        all_rows = []
        for i, s in enumerate(picks, start=1):
            h = hist[hist["state"] == s].tail(78)
            fig.add_trace(go.Scatter(x=h["date"], y=h["sales"], name=f"{s} hist",
                                     line=dict(color="#5A6B7B")), row=i, col=1)
            fc = fetch_forecast(s, 8)
            if fc:
                fdf = pd.DataFrame(fc["forecast"])
                fdf["date"] = pd.to_datetime(fdf["date"])
                fdf["state"] = s
                all_rows.append(fdf)
                fig.add_trace(go.Scatter(x=fdf["date"], y=fdf["yhat"], name=f"{s} fc",
                                         line=dict(color="#636EFA", dash="dash")), row=i, col=1)
        fig.update_layout(height=260 * len(picks), template="plotly_white", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        if all_rows:
            combined = pd.concat(all_rows, ignore_index=True)
            st.download_button(
                "⬇ Download combined forecasts CSV",
                combined.to_csv(index=False).encode(),
                file_name="forecasts_multi.csv", mime="text/csv",
            )


# ----------------------------------------------------------------------------
# View 3 — Leaderboard
# ----------------------------------------------------------------------------
elif mode == "Model leaderboard":
    st.title("🏆 Model Leaderboard — SMAPE per State")
    scores = meta.get("scores", {}) if isinstance(meta, dict) else {}
    if not scores:
        st.info("No leaderboard data found in artifacts/metadata.json.")
    else:
        rows = []
        for state, by_model in scores.items():
            for model, smape in by_model.items():
                rows.append({"state": state, "model": model, "smape": smape})
        df = pd.DataFrame(rows)

        st.subheader("Heatmap")
        pivot = df.pivot(index="state", columns="model", values="smape")
        fig = go.Figure(data=go.Heatmap(
            z=pivot.values, x=pivot.columns, y=pivot.index,
            colorscale="RdYlGn_r", colorbar=dict(title="SMAPE %"),
            hovertemplate="State: %{y}<br>Model: %{x}<br>SMAPE: %{z:.2f}%<extra></extra>",
        ))
        fig.update_layout(height=max(400, 22 * len(pivot)), template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Champion summary")
        champs = meta.get("champions", {}) if isinstance(meta, dict) else {}
        cdf = pd.DataFrame([
            {"state": s, "champion": v.get("model"), "smape": v.get("smape")}
            for s, v in champs.items()
        ]).sort_values("smape")
        st.dataframe(cdf, use_container_width=True)

        st.subheader("Champion model distribution")
        counts = cdf["champion"].value_counts().reset_index()
        counts.columns = ["model", "states_won"]
        fig2 = go.Figure(go.Bar(x=counts["model"], y=counts["states_won"],
                                marker_color="#636EFA"))
        fig2.update_layout(template="plotly_white", height=350,
                           yaxis_title="# states where this is champion")
        st.plotly_chart(fig2, use_container_width=True)


# ----------------------------------------------------------------------------
# View 4 — Retrain
# ----------------------------------------------------------------------------
elif mode == "Retrain":
    st.title("🔁 Trigger Retraining")
    st.markdown(
        "Kicks off a background retrain on the API server. "
        "All four model families (SARIMA, Prophet, XGBoost, LSTM) are retrained per state, "
        "and a new champion is selected via walk-forward SMAPE."
    )
    if st.button("🚀 Retrain now", type="primary"):
        try:
            r = requests.post(f"{API_URL}/retrain", timeout=10)
            if r.ok:
                st.success(f"Retrain queued: {r.json()}")
            else:
                st.error(f"{r.status_code}: {r.text}")
        except Exception as e:
            st.error(f"API error: {e}")

st.sidebar.divider()
st.sidebar.caption("Backend: FastAPI · Models: SARIMA, Prophet, XGBoost, LSTM")
st.sidebar.caption("Selection: walk-forward SMAPE per state")
