"""
FastAPI backend for stock price prediction.
Uses ARIMA and LSTM models.
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from models.arima_model import predict_arima
from models.lstm_model import predict_lstm
from models.gbt_model import predict_gbrt, backtest_gbrt

from stats import compute_backtest_metrics, compute_series_stats, summarize_forecast

app = FastAPI(title="Stock Price Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionRequest(BaseModel):
    ticker: str
    model: str  # "arima", "lstm", "gbrt", or "auto"
    steps: int = 30


def fetch_stock_data(ticker: str, period: str = "2y") -> pd.Series:
    """Fetch historical closing prices from Yahoo Finance."""
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    if df.empty or len(df) < 60:
        raise HTTPException(
            status_code=400,
            detail=f"Not enough data for {ticker}. Try a valid ticker (e.g. AAPL, MSFT, GOOGL).",
        )
    return df["Close"]


@app.get("/api/stock/{ticker}")
def get_stock_data(
    ticker: str,
    period: str = Query("2y", description="1mo, 3mo, 6mo, 1y, 2y, 5y"),
):
    """Get historical stock data for a ticker."""
    try:
        series = fetch_stock_data(ticker.upper(), period)
        return {
            "ticker": ticker.upper(),
            "dates": [d.strftime("%Y-%m-%d") for d in series.index],
            "values": series.tolist(),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict")
def predict(request: PredictionRequest):
    """Run ARIMA or LSTM prediction for a ticker."""
    ticker = request.ticker.upper()
    model_name = request.model.lower()
    steps = max(5, min(90, request.steps))

    try:
        series = fetch_stock_data(ticker, period="2y")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    def _backtest_arima(s: pd.Series, test_size: int = 90):
        s = pd.Series(s).dropna().astype(float)
        if s.shape[0] < test_size + 60:
            return None
        train = s.iloc[:-test_size]
        test = s.iloc[-test_size:]
        tmp = predict_arima(train, steps=test_size, order=None)
        pred = np.asarray(tmp["forecast_values"], dtype=float)
        true = test.to_numpy(dtype=float)
        n = min(true.shape[0], pred.shape[0])
        return {
            "test_dates": [d.strftime("%Y-%m-%d") for d in test.index[:n]],
            "y_true": true[:n],
            "y_pred": pred[:n],
        }

    def _backtest_lstm(s: pd.Series, test_size: int = 60):
        # LSTM is slower; keep a smaller holdout and fewer epochs.
        s = pd.Series(s).dropna().astype(float)
        if s.shape[0] < test_size + 120:
            return None
        train = s.iloc[:-test_size]
        test = s.iloc[-test_size:]
        tmp = predict_lstm(train, steps=test_size, seq_length=60, epochs=20)
        pred = np.asarray(tmp["forecast_values"], dtype=float)
        true = test.to_numpy(dtype=float)
        n = min(true.shape[0], pred.shape[0])
        return {
            "test_dates": [d.strftime("%Y-%m-%d") for d in test.index[:n]],
            "y_true": true[:n],
            "y_pred": pred[:n],
        }

    try:
        backtest = None
        result = None
        if model_name == "auto":
            # Evaluate ARIMA and GBRT on a recent holdout and, when both are healthy,
            # build a weighted ensemble (more stable, usually slightly more accurate).
            bt_arima = _backtest_arima(series, test_size=90)
            bt_gbrt = None
            try:
                bt_g = backtest_gbrt(series, test_size=90)
                bt_gbrt = {
                    "test_dates": bt_g["test_dates"],
                    "y_true": bt_g["y_true"],
                    "y_pred": bt_g["y_pred"],
                }
            except Exception:
                bt_gbrt = None

            def rmse_of(bt):
                if not bt or not isinstance(bt, dict):
                    return np.inf
                y_true = bt.get("y_true")
                y_pred = bt.get("y_pred")
                if y_true is None or y_pred is None:
                    return np.inf
                try:
                    m = compute_backtest_metrics(y_true, y_pred)
                except ValueError:
                    # If anything inside metrics complains about array truth values, treat as unusable.
                    return np.inf
                return m.get("rmse", np.inf) or np.inf

            rm_arima = rmse_of(bt_arima)
            rm_gbrt = rmse_of(bt_gbrt)

            use_ensemble = (
                np.isfinite(rm_arima)
                and np.isfinite(rm_gbrt)
                and rm_arima > 0
                and rm_gbrt > 0
                and bt_arima is not None
                and bt_gbrt is not None
            )

            if use_ensemble:
                # Inverse-RMSE weights: put more weight on the more accurate model.
                w_arima = 1.0 / rm_arima
                w_gbrt = 1.0 / rm_gbrt
                w_sum = w_arima + w_gbrt

                # Ensemble backtest
                y_true = np.asarray(bt_arima.get("y_true"), dtype=float)
                y_pred_a = np.asarray(bt_arima.get("y_pred"), dtype=float)
                y_pred_g = np.asarray(bt_gbrt.get("y_pred"), dtype=float)
                n_bt = min(y_true.shape[0], y_pred_a.shape[0], y_pred_g.shape[0])
                y_true = y_true[:n_bt]
                y_pred_a = y_pred_a[:n_bt]
                y_pred_g = y_pred_g[:n_bt]
                y_pred_ens = (w_arima * y_pred_a + w_gbrt * y_pred_g) / w_sum
                test_dates = bt_arima.get("test_dates", [])[:n_bt]

                backtest = {
                    "test_dates": test_dates,
                    "y_true": y_true,
                    "y_pred": y_pred_ens,
                }

                # Ensemble forecast: weighted average of ARIMA + GBRT forecasts
                res_arima = predict_arima(series, steps=steps, order=None)
                res_gbrt = predict_gbrt(series, steps=steps)
                fa = np.asarray(res_arima["forecast_values"], dtype=float)
                fg = np.asarray(res_gbrt["forecast_values"], dtype=float)
                n_f = min(fa.shape[0], fg.shape[0])
                fa = fa[:n_f]
                fg = fg[:n_f]
                f_ens = (w_arima * fa + w_gbrt * fg) / w_sum
                fd = res_arima["forecast_dates"][:n_f]

                result = {
                    "historical_dates": res_arima["historical_dates"],
                    "historical_values": res_arima["historical_values"],
                    "forecast_dates": fd,
                    "forecast_values": f_ens.tolist(),
                    "ensemble_components": {
                        "arima_rmse": rm_arima,
                        "gbrt_rmse": rm_gbrt,
                    },
                }
                model_name = "ensemble"
            else:
                # Fallback: pick the single better model by RMSE (or whichever exists).
                if rm_gbrt <= rm_arima:
                    model_name = "gbrt"
                    backtest = bt_gbrt
                else:
                    model_name = "arima"
                    backtest = bt_arima

        if model_name == "arima":
            result = result or predict_arima(series, steps=steps, order=None)
            backtest = backtest or _backtest_arima(series, test_size=90)
        elif model_name == "lstm":
            result = predict_lstm(series, steps=steps)
            backtest = _backtest_lstm(series, test_size=60)
        elif model_name == "gbrt":
            result = predict_gbrt(series, steps=steps)
            bt_g = backtest_gbrt(series, test_size=90)
            backtest = {
                "test_dates": bt_g["test_dates"],
                "y_true": bt_g["y_true"],
                "y_pred": bt_g["y_pred"],
            }
        elif model_name == "ensemble":
            # result/backtest are fully built in the auto branch; just ensure they exist.
            if result is None or backtest is None:
                raise HTTPException(
                    status_code=500,
                    detail="Ensemble result unavailable; try a specific model.",
                )
        else:
            raise HTTPException(
                status_code=400,
                detail="model must be 'arima', 'lstm', 'gbrt', or 'auto'",
            )

        # Metrics + stats bundle for dashboard
        series_stats = compute_series_stats(series)
        backtest_metrics = None
        if backtest and isinstance(backtest, dict):
            y_true_bt = backtest.get("y_true")
            y_pred_bt = backtest.get("y_pred")
            if y_true_bt is not None and y_pred_bt is not None:
                try:
                    backtest_metrics = compute_backtest_metrics(y_true_bt, y_pred_bt)
                except ValueError:
                    # If metrics computation fails (e.g. ambiguous array truth), skip backtest stats.
                    backtest_metrics = None

        residual_std = backtest_metrics.get("error_std") if backtest_metrics else None
        forecast_summary = summarize_forecast(np.asarray(result.get("forecast_values", []), dtype=float), residual_std=residual_std)

        result["ticker"] = ticker
        result["model"] = model_name
        result["stats"] = {
            "series": series_stats,
            "backtest": backtest_metrics,
            "forecast": forecast_summary,
        }
        # Keep payload small: only include dates for backtest, not full arrays unless needed later
        if backtest and backtest_metrics and isinstance(backtest, dict):
            test_dates = backtest.get("test_dates")
            if test_dates is None:
                test_dates = []
            y_true_bt = backtest.get("y_true")
            if y_true_bt is None:
                y_true_bt = []
            result["stats"]["backtest"]["test_window"] = {
                "n": int(len(y_true_bt)),
                "start": test_dates[0] if len(test_dates) > 0 else None,
                "end": test_dates[-1] if len(test_dates) > 0 else None,
            }

        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
def health():
    return {"status": "ok"}
