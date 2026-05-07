# Streamlit Dashboard — Frontend

A polished, interactive frontend for the per-state forecasting backend.

## Run

```bash
# 1. Train models (creates artifacts/)
python -m src.training.train_all

# 2. Start the API
uvicorn src.api.main:app --port 8000

# 3. In another terminal, start the dashboard
streamlit run dashboard/app.py
```

Open <http://localhost:8501>.

## Views

1. **Single state** — champion model badge, SMAPE, history + 8-week forecast chart with confidence band, downloadable CSV.
2. **Compare states** — side-by-side multi-state forecast subplots and combined CSV export.
3. **Model leaderboard** — heatmap of SMAPE × state × model, champion summary table, and bar chart of which model wins where.
4. **Retrain** — one-click trigger for the FastAPI background retrain endpoint.

## Why this stands out

- Calls the live REST API (proves the backend works end-to-end)
- Shows *per-state champion selection*, not just one global model
- Heatmap makes the "which algorithm wins where" story instantly visible
- Business-ready CSV downloads for stakeholders
