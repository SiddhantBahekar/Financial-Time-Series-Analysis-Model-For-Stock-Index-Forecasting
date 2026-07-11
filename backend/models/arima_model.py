"""
ARIMA model for stock price prediction.
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import warnings

warnings.filterwarnings("ignore")


def is_stationary(series: pd.Series) -> bool:
    """Check if series is stationary using Augmented Dickey-Fuller test."""
    result = adfuller(series.dropna())
    return result[1] < 0.05


def fit_arima(series: pd.Series, order: tuple = (5, 1, 0)) -> ARIMA:
    """
    Fit ARIMA model on closing price series.
    order: (p, d, q) - default (5,1,0) works well for many stock series.
    """
    model = ARIMA(series, order=order)
    fitted = model.fit()
    return fitted


def select_arima_order(
    series: pd.Series,
    max_p: int = 3,
    max_q: int = 3,
    d_options=(0, 1),
) -> tuple:
    """
    Small AIC grid-search to pick a reasonable ARIMA(p,d,q).
    Keeps search tiny to stay fast on API calls.
    """
    s = pd.Series(series).dropna().astype(float)
    best_order = (1, 1, 0)
    best_aic = np.inf
    for d in d_options:
        for p in range(0, max_p + 1):
            for q in range(0, max_q + 1):
                if p == 0 and d == 0 and q == 0:
                    continue
                try:
                    fitted = ARIMA(s, order=(p, d, q)).fit()
                    aic = float(fitted.aic)
                    if np.isfinite(aic) and aic < best_aic:
                        best_aic = aic
                        best_order = (p, d, q)
                except Exception:
                    continue
    return best_order


def predict_arima(
    series: pd.Series,
    steps: int = 30,
    order: tuple | None = None,
) -> dict:
    """
    Fit ARIMA and return historical + forecast.
    Returns dict with dates and values for plot.
    """
    series_clean = series.dropna()
    if len(series_clean) < 30:
        raise ValueError("Need at least 30 data points for ARIMA")

    if order is None:
        order = select_arima_order(series_clean)

    model = ARIMA(series_clean, order=order)
    fitted = model.fit()

    # Forecast future steps
    forecast = fitted.forecast(steps=steps)
    last_date = series_clean.index[-1]
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=steps,
        freq="B",
    )  # business days

    return {
        "historical_dates": [d.strftime("%Y-%m-%d") for d in series_clean.index],
        "historical_values": series_clean.tolist(),
        "forecast_dates": [d.strftime("%Y-%m-%d") for d in future_dates],
        "forecast_values": forecast.tolist(),
        "order": list(order),
    }
