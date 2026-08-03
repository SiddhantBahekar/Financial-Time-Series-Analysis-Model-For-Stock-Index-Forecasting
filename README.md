# Stock Price Prediction & Financial Analytics Dashboard

A full-stack web application for financial time-series analysis and stock price forecasting using **Statistical (ARIMA)**, **Machine Learning (GBRT)**, and **Deep Learning (LSTM)** models, featuring an automated **Inverse-RMSE Weighted Ensemble** and holdout backtesting engine.

Built with **React 19**, **Vite**, **Tailwind CSS**, **Recharts**, and a **Python FastAPI** backend powered by **yfinance**, **statsmodels**, **scikit-learn**, and **TensorFlow / Keras**.

---

## Key Features

- **Multi-Model Forecasting Engine:**
  - **Auto (Inverse-RMSE Ensemble):** Evaluates models on a holdout period and blends predictions proportionally to model accuracy.
  - **GBRT (Gradient Boosted Regression Trees):** High-speed tabular machine learning model trained on lagged returns and rolling technical statistics.
  - **ARIMA (AutoRegressive Integrated Moving Average):** Classical econometric time-series model with automated AIC grid search and stationarity verification.
  - **LSTM (Long Short-Term Memory):** Deep neural network architecture designed to capture non-linear temporal dynamics over 60-day historical sequences.
- **Automated Holdout Backtesting:** Evaluates out-of-sample accuracy on historical data with metrics including **RMSE**, **MAE**, **MAPE**, **sMAPE**, **$R^2$ Score**, and **Directional Accuracy (%)**.
- **Financial Market Analytics:** Computes key quantitative market indicators including annualized returns, daily and annualized volatility, Sharpe ratio, maximum drawdown, skewness, kurtosis, and 52-week price bounds.
- **Uncertainty Bounds & Forecast Summaries:** Provides multi-horizon out-of-sample forecasts (5 to 90 business days ahead) along with 95% confidence intervals based on backtest residual standard deviations.
- **Interactive Visual Dashboard:** Built with React 19 and Recharts, featuring live API health connection monitoring, stock quick-select tags (`AAPL`, `MSFT`, `GOOGL`, `AMZN`, `META`, `TSLA`, `NVDA`, `NIFTY 50`), custom step adjustments, and responsive dark-theme design.

---

## Architecture & How the Project Works

```
                     ┌──────────────────────────────────────┐
                     │         Yahoo Finance API           │
                     └──────────────────┬───────────────────┘
                                        │ (Historical Daily OHLCV Data)
                                        ▼
                     ┌──────────────────────────────────────┐
                     │          FastAPI Backend             │
                     │          (Python 3.10+)              │
                     └──────────────────┬───────────────────┘
                                        │
         ┌──────────────────────────────┼──────────────────────────────┐
         ▼                              ▼                              ▼
┌─────────────────┐           ┌──────────────────┐           ┌──────────────────┐
│  ARIMA Model    │           │    GBRT Model    │           │    LSTM Model    │
│ (statsmodels)   │           │  (scikit-learn)  │           │   (TensorFlow)   │
└────────┬────────┘           └────────┬─────────┘           └────────┬─────────┘
         │                             │                              │
         └─────────────────────────────┼──────────────────────────────┘
                                       │
                                       ▼
                     ┌──────────────────────────────────────┐
                     │  Holdout Backtest & Stats Engine     │
                     │             (stats.py)               │
                     └──────────────────┬───────────────────┘
                                       │ (Metrics, Backtests & Predictions)
                                       ▼
                     ┌──────────────────────────────────────┐
                     │        React 19 Frontend Dashboard   │
                     │     (Vite + Tailwind + Recharts)     │
                     └──────────────────────────────────────┘
```

### End-to-End Workflow

1. **Data Fetching:** The backend receives requests with a stock ticker (e.g., `AAPL`, `MSFT`, `^NSEI`) and fetches up to 2 years of daily price data using `yfinance`.
2. **Data Preprocessing & Feature Engineering:**
   - **ARIMA:** Checks series stationarity using the **Augmented Dickey-Fuller (ADF)** test and performs a grid search over $(p, d, q)$ parameter combinations to minimize the Akaike Information Criterion (AIC).
   - **GBRT:** Transforms closing price series into stationary log returns, percentage returns, multi-period price/return lags ($t-1 \dots t-20$), and rolling statistics (mean and standard deviation over 5, 10, and 20-day windows).
   - **LSTM:** Scales closing prices between $[0, 1]$ using `MinMaxScaler` and constructs 3D sequence tensors of length 60 (`seq_length=60`).
3. **Model Fitting & Multi-Step Forecasting:**
   - Fits selected model on historical data.
   - For multi-step out-of-sample business-day forecasting ($H$ steps ahead), predictions are generated iteratively, dynamically recalculating feature vectors or rolling sequence inputs at each time step.
4. **Holdout Backtesting:**
   - A holdout dataset (60 to 90 trading days) is reserved from the end of the time series.
   - The model is trained on data prior to the holdout, and its out-of-sample predictions are compared against true ground-truth prices to calculate quantitative validation metrics.
5. **Inverse-RMSE Ensemble Blending (`Auto` Mode):**
   - When **Auto** mode is selected, both ARIMA and GBRT undergo holdout backtesting.
   - Weights are computed inversely proportional to their Root Mean Squared Errors:
     $$w_{\text{arima}} = \frac{1}{\text{RMSE}_{\text{ARIMA}}}, \quad w_{\text{gbrt}} = \frac{1}{\text{RMSE}_{\text{GBRT}}}$$
   - The final forecast is constructed as a weighted linear combination:
     $$\hat{y}_{\text{ensemble}} = \frac{w_{\text{arima}} \cdot \hat{y}_{\text{arima}} + w_{\text{gbrt}} \cdot \hat{y}_{\text{gbrt}}}{w_{\text{arima}} + w_{\text{gbrt}}}$$
6. **API Response & Visualization:** The backend aggregates series metrics, backtest evaluations, and forecast arrays into structured JSON responses for immediate rendering in the React frontend.

---

## Machine Learning & Statistical Models Detailed

### 1. Auto (Inverse-RMSE Ensemble)
- **Concept:** Combines predictions from complementary algorithms (statistical ARIMA + tabular GBRT) to reduce model variance and bias.
- **Weighting Mechanism:** Out-of-sample holdout RMSE values determine each model's contribution. Models with lower historical forecasting error receive higher weights.
- **Fallback Logic:** If holdout backtesting fails due to short history or data constraints, the engine automatically selects the best single performing model.

### 2. Gradient Boosted Regression Trees (GBRT)
- **Module:** `backend/models/gbt_model.py`
- **Algorithm:** `sklearn.ensemble.HistGradientBoostingRegressor`
- **Feature Set:**
  - Lags at $t-1, t-2, t-3, t-5, t-10, t-20$ trading days for price, percentage return, and log return.
  - Rolling 5, 10, and 20-day mean and standard deviation of close prices and returns.
- **Hyperparameters:** Loss = `squared_error`, max depth = 6, learning rate = 0.05, max iterations = 400.
- **Forecasting:** Recursive multi-step forecasting where predicted values update lag structures step-by-step.

### 3. ARIMA (AutoRegressive Integrated Moving Average)
- **Module:** `backend/models/arima_model.py`
- **Algorithm:** `statsmodels.tsa.arima.model.ARIMA`
- **Order Selection:** Automated AIC minimization over parameter bounds $p \in [0, 3]$, $d \in \{0, 1\}$, $q \in [0, 3]$.
- **Stationarity Testing:** Uses Augmented Dickey-Fuller (`adfuller`) test ($p\text{-value} < 0.05$).
- **Forecasting:** Direct linear out-of-sample projection over future business days (`freq="B"`).

### 4. LSTM (Long Short-Term Memory Network)
- **Module:** `backend/models/lstm_model.py`
- **Framework:** TensorFlow / Keras (`tf.keras`)
- **Network Architecture:**
  - Layer 1: LSTM (50 units, `return_sequences=True`)
  - Regularization: Dropout (rate = 0.2)
  - Layer 2: LSTM (50 units, `return_sequences=False`)
  - Regularization: Dropout (rate = 0.2)
  - Output Layer: Dense (1 unit, linear activation)
- **Training Details:** Optimized using Adam with Mean Squared Error (MSE) loss and `EarlyStopping` callback (patience = 10 epochs on validation split).
- **Sequence Processing:** Formats data into 60-day rolling sequence windows with `MinMaxScaler` normalizations.

---

## Tech Stack

### Frontend
- **Framework:** React 19
- **Build Tool:** Vite
- **Styling:** Tailwind CSS (Dark Mode Design Tokens)
- **Visualization:** Recharts (Responsive Line Charts & Custom Tooltips)

### Backend
- **Framework:** FastAPI (Uvicorn ASGI Server)
- **Data Ingestion:** `yfinance`
- **Time-Series Analysis:** `statsmodels`
- **Machine Learning & Preprocessing:** `scikit-learn`, `numpy`, `pandas`
- **Deep Learning:** `TensorFlow` / `Keras`

---

## Project Structure

```
stock price prediction/
├── backend/
│   ├── main.py              # FastAPI server, endpoints & CORS configuration
│   ├── stats.py             # Quantitative financial statistics & backtest metrics engine
│   ├── requirements.txt     # Python dependencies
│   └── models/
│       ├── arima_model.py   # ARIMA model fit, order selection & forecasting
│       ├── gbt_model.py     # GBRT feature engineering, training & recursive forecasting
│       └── lstm_model.py    # TensorFlow/Keras LSTM neural network architecture
├── frontend/
│   ├── src/
│   │   ├── App.jsx          # Main UI layout, controls, metric cards & chart components
│   │   ├── api.js           # API client for backend endpoints
│   │   ├── index.css        # Tailwind CSS styles & modern aesthetics
│   │   └── main.jsx         # React application entry point
│   ├── package.json         # Frontend dependencies & Vite scripts
│   └── vite.config.js       # Vite configuration
└── README.md                # Project documentation
```

---

## Setup & Installation

### Prerequisites
- Python 3.10+ installed
- Node.js (v18+) and `npm` installed

### 1. Backend Setup (Python FastAPI)

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# (Optional Windows note: If compiler issues occur, install pre-built binaries):
# pip install --only-binary :all: -r requirements.txt

# Run FastAPI server
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

- API Base URL: `http://localhost:8000`
- Interactive OpenAPI Docs: `http://localhost:8000/docs`

### 2. Frontend Setup (React + Vite)

```bash
cd frontend

# Install node dependencies
npm install

# Start Vite development server
npm run dev
```

- Web App URL: `http://localhost:5173`

---

## API Endpoints Reference

| Method | Endpoint | Description | Query / Body Parameters |
| :--- | :--- | :--- | :--- |
| `GET` | `/api/health` | Backend status check | None |
| `GET` | `/api/stock/{ticker}` | Fetch raw historical closing price series | `period` (default `"2y"`) |
| `POST` | `/api/predict` | Run model prediction, backtest & metric analysis | `{ "ticker": "AAPL", "model": "auto", "steps": 30 }` |

---

## Usage Instructions

1. Enter a valid ticker symbol (e.g. `AAPL`, `MSFT`, `GOOGL`, `AMZN`, `META`, `TSLA`, `NVDA`) or click a suggested tag (including `NIFTY 50` / `^NSEI`).
2. Select a forecasting model from the dropdown:
   - **Auto (best):** Ensemble of ARIMA + GBRT weighted by inverse RMSE.
   - **GBRT:** High-speed tree ensemble with rolling technical indicators.
   - **ARIMA:** Statistical auto-regressive time-series forecasting.
   - **LSTM:** Deep learning recurrent neural network.
3. Specify the number of forecast days (5 to 90 days).
4. Click **History** to display historical price data only, or **Predict** to execute model forecasting and display historical prices, out-of-sample forecasts, backtest metrics, and market statistics cards.

---

## Disclaimer

This application is created for **educational, analytical, and research purposes only**. Financial market predictions involve inherent risk and uncertainty. Forecasts generated by these models should not be construed as financial or investment advice.
