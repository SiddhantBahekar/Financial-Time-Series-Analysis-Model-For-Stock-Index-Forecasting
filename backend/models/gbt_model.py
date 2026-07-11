from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


DEFAULT_LAGS = [1, 2, 3, 5, 10, 20]
DEFAULT_WINDOWS = [5, 10, 20]


def _make_feature_frame(
    close: pd.Series,
    lags: List[int] = DEFAULT_LAGS,
    windows: List[int] = DEFAULT_WINDOWS,
) -> pd.DataFrame:
    s = pd.Series(close).astype(float).copy()
    df = pd.DataFrame({"close": s})

    # Log returns are typically more stationary than price levels; still keep price features too.
    df["ret1"] = df["close"].pct_change()
    df["logret1"] = np.log(df["close"]).diff()

    for lag in lags:
        df[f"close_lag_{lag}"] = df["close"].shift(lag)
        df[f"ret1_lag_{lag}"] = df["ret1"].shift(lag)
        df[f"logret1_lag_{lag}"] = df["logret1"].shift(lag)

    for w in windows:
        df[f"roll_mean_{w}"] = df["close"].rolling(w).mean()
        df[f"roll_std_{w}"] = df["close"].rolling(w).std(ddof=1)
        df[f"roll_ret_mean_{w}"] = df["ret1"].rolling(w).mean()
        df[f"roll_ret_std_{w}"] = df["ret1"].rolling(w).std(ddof=1)

    # Target: next business day's close
    df["y_next"] = df["close"].shift(-1)
    return df


def _train_model(X: np.ndarray, y: np.ndarray) -> HistGradientBoostingRegressor:
    # Solid default for tabular time-series features without heavy tuning
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        max_depth=6,
        learning_rate=0.05,
        max_iter=400,
        l2_regularization=0.0,
        random_state=42,
    )
    model.fit(X, y)
    return model


def backtest_gbrt(
    series: pd.Series,
    test_size: int = 90,
    lags: List[int] = DEFAULT_LAGS,
    windows: List[int] = DEFAULT_WINDOWS,
) -> Dict[str, object]:
    s = pd.Series(series).dropna()
    if s.shape[0] < (max(lags) + max(windows) + test_size + 5):
        raise ValueError("Not enough data for GBRT backtest; try a longer period.")

    df = _make_feature_frame(s, lags=lags, windows=windows).dropna()
    if df.shape[0] <= test_size + 10:
        raise ValueError("Not enough clean feature rows for GBRT backtest.")

    feature_cols = [c for c in df.columns if c not in ("y_next",)]
    X = df[feature_cols].to_numpy(dtype=float)
    y = df["y_next"].to_numpy(dtype=float)

    X_train, y_train = X[:-test_size], y[:-test_size]
    X_test, y_test = X[-test_size:], y[-test_size:]

    model = _train_model(X_train, y_train)
    y_pred = model.predict(X_test)

    return {
        "y_true": y_test,
        "y_pred": y_pred,
        "feature_cols": feature_cols,
        "test_dates": [d.strftime("%Y-%m-%d") for d in df.index[-test_size:]],
    }


def predict_gbrt(
    series: pd.Series,
    steps: int = 30,
    lags: List[int] = DEFAULT_LAGS,
    windows: List[int] = DEFAULT_WINDOWS,
    train_window: Optional[int] = 504,  # ~2 years trading days
) -> Dict[str, object]:
    s = pd.Series(series).dropna().astype(float)
    if s.shape[0] < (max(lags) + max(windows) + 30):
        raise ValueError("Need more history for GBRT (try a longer period).")

    if train_window is not None and s.shape[0] > train_window:
        s_train = s.iloc[-train_window:]
    else:
        s_train = s

    df = _make_feature_frame(s_train, lags=lags, windows=windows).dropna()
    feature_cols = [c for c in df.columns if c not in ("y_next",)]
    X = df[feature_cols].to_numpy(dtype=float)
    y = df["y_next"].to_numpy(dtype=float)

    model = _train_model(X, y)

    # Recursive multi-step forecast
    history = s.copy()
    forecasts: List[float] = []
    for _ in range(steps):
        # Build features for the latest point and drop only the target column.
        df_last = _make_feature_frame(history, lags=lags, windows=windows).iloc[[-1]]
        df_last = df_last[feature_cols]
        if df_last.isnull().any(axis=None):
            # As a safety net, if any feature is still NaN, back off gracefully.
            raise ValueError("Unable to build features for forecast; insufficient recent data.")
        X_last = df_last.to_numpy(dtype=float)
        pred = float(model.predict(X_last)[0])
        forecasts.append(pred)
        # Append predicted close on next business day to keep spacing consistent
        next_bd = pd.bdate_range(history.index[-1], periods=2)[-1]
        history = pd.concat([history, pd.Series([pred], index=[next_bd])])

    last_date = s.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps, freq="B")

    return {
        "historical_dates": [d.strftime("%Y-%m-%d") for d in s.index],
        "historical_values": s.tolist(),
        "forecast_dates": [d.strftime("%Y-%m-%d") for d in future_dates],
        "forecast_values": forecasts,
        "lags": lags,
        "windows": windows,
    }

