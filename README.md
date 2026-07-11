# Stock Price Prediction

A full-stack web app for stock price prediction using **ARIMA** and **LSTM** models. Built with React (Vite), Tailwind CSS, and a Python FastAPI backend.

## Stack

- **Frontend:** React 19, Vite, Tailwind CSS, Recharts
- **Backend:** FastAPI, yfinance, statsmodels (ARIMA), TensorFlow (LSTM)

## Setup

### Backend (Python)

```bash
cd backend
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
# source venv/bin/activate
pip install -r requirements.txt
# If install fails (e.g. "Unknown compiler" on Windows), use pre-built wheels only:
# pip install --only-binary :all: -r requirements.txt
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API will be at **http://localhost:8000**. Docs: http://localhost:8000/docs

### Frontend (React)

```bash
cd frontend
npm install
npm run dev
```

App will be at **http://localhost:5173**.

## Usage

1. Enter a stock ticker (e.g. AAPL, MSFT, GOOGL) or pick one of the suggestions.
2. Choose **ARIMA** or **LSTM** and the number of forecast days.
3. Click **History** to load past prices only, or **Predict** to run the model and see historical + forecast on the chart.

## Project structure

```
stock price prediction/
├── backend/
│   ├── main.py           # FastAPI app, routes, yfinance data
│   ├── requirements.txt
│   └── models/
│       ├── arima_model.py # ARIMA fit & forecast
│       └── lstm_model.py  # LSTM train & forecast
├── frontend/
│   ├── src/
│   │   ├── App.jsx       # Main UI & chart
│   │   ├── api.js        # API client
│   │   └── index.css     # Tailwind
│   └── package.json
└── README.md
```

## Notes

- Stock data is from **Yahoo Finance** (yfinance). Use valid tickers (e.g. AAPL, MSFT).
- **ARIMA** needs at least 30 points; **LSTM** needs more (default 60+). Both use 2 years of history.
- Predictions are for educational purposes only, not financial advice.
