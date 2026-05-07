# Time Series Forecasting System with API

End-to-end time-series forecasting system for predicting weekly sales across US states using historical sales data.

The system trains multiple forecasting algorithms, compares their performance, automatically selects the best-performing model, and serves predictions through a FastAPI REST API.

---

# Objective

Forecast the next 8 weeks of sales for each state while:

* handling missing dates and missing values
* capturing trend and seasonality
* comparing multiple forecasting models
* automatically selecting the best model
* serving predictions through an API

---

# Implemented Models

The project trains and compares the following forecasting models:

1. SARIMA
2. Facebook Prophet
3. XGBoost with lag features
4. LSTM (PyTorch)

Different states may show different forecasting behavior, so the system automatically selects the best-performing model for each state.

---

# Feature Engineering

Implemented feature engineering includes:

* lag features: t-1, t-7, t-30
* rolling mean
* rolling standard deviation
* month feature
* holiday flag
* calendar-based features

---

# Data Processing

The preprocessing pipeline:

* handles missing dates and missing values
* parses mixed date formats
* converts data into weekly frequency
* resamples data using weekly intervals

---

# Validation Strategy

Time-series validation logic is used to avoid data leakage.

* historical data is used for training
* latest 8 weeks are used for validation

Evaluation metrics:

* SMAPE
* MAE
* RMSE

---

# REST API

The system exposes predictions through FastAPI REST endpoints.

Available endpoints include:

| Method | Endpoint            | Description             |
| ------ | ------------------- | ----------------------- |
| GET    | `/health`           | Health check            |
| GET    | `/states`           | List supported states   |
| GET    | `/models`           | Best model and metrics  |
| GET    | `/forecast/{state}` | Forecast for one state  |
| GET    | `/forecast`         | Forecast for all states |
| POST   | `/retrain`          | Retrain all models      |

---

# Project Structure

```text
forecasting_project/
├── data/
├── src/
│   ├── api/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── training/
│   └── utils/
├── tests/
├── artifacts/
├── Dockerfile
├── Makefile
├── requirements.txt
└── README.md
```

---

# Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Training

Train all forecasting models:

```bash
python -m src.training.train_all
```

---

# Running the API

Start the FastAPI server:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Swagger API documentation:

```text
http://localhost:8000/docs
```

---

# Example Forecast Response

```json
{
  "state": "Example State",
  "model": "XGBoost",
  "horizon_weeks": 8,
  "metrics": {
    "smape": 4.12,
    "mae": 18000000,
    "rmse": 23000000
  },
  "forecast": [
    {
      "date": "2023-12-10",
      "yhat": 521000000
    }
  ]
}
```

---

# Testing

Run tests:

```bash
pytest -q
```

---

# Docker Support

Build Docker image:

```bash
docker build -t forecasting-api .
```

Run Docker container:

```bash
docker run -p 8000:8000 forecasting-api
```

---

# Tech Stack

* Python
* Pandas
* NumPy
* Statsmodels
* Prophet
* XGBoost
* PyTorch
* FastAPI
* Docker
* Pytest

---

# Summary

This project demonstrates:

* end-to-end time-series forecasting
* feature engineering for sequential data
* multiple forecasting model comparison
* automated model selection
* REST API integration
* production-oriented ML workflow
