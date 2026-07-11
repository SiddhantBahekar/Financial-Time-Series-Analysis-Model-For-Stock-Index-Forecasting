from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    except Exception:
        return None


def _pct(a: float, b: float) -> Optional[float]:
    # Percent change from a to b
    if a is None or b is None:
        return None
    if a == 0:
        return None
    return (b - a) / a * 100.0


def compute_backtest_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_true.size == 0:
        return {"n": 0}

    mae = mean_absolute_error(y_true, y_pred)
    # scikit-learn >=1.7 removed `squared`; compute RMSE manually
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    # Avoid MAPE exploding around zeros; stock prices are typically > 0 but guard anyway.
    denom = np.where(np.abs(y_true) < 1e-12, np.nan, np.abs(y_true))
    mape = np.nanmean(np.abs((y_true - y_pred) / denom)) * 100.0
    # Symmetric MAPE
    smape = np.nanmean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-12)) * 100.0
    r2 = r2_score(y_true, y_pred) if y_true.size >= 2 else None

    # Directional accuracy: did we get the direction of change correct vs previous true close?
    if y_true.size >= 2:
        true_diff = np.diff(y_true)
        pred_diff = np.diff(y_pred)
        direction_acc = float(np.mean(np.sign(true_diff) == np.sign(pred_diff))) * 100.0
    else:
        direction_acc = None

    err = y_pred - y_true
    err_mean = float(np.mean(err))
    err_std = float(np.std(err, ddof=1)) if err.size >= 2 else 0.0

    return {
        "n": int(y_true.size),
        "mae": float(mae),
        "rmse": float(rmse),
        "mape_pct": _safe_float(mape),
        "smape_pct": _safe_float(smape),
        "r2": _safe_float(r2),
        "directional_accuracy_pct": _safe_float(direction_acc),
        "error_mean": _safe_float(err_mean),
        "error_std": _safe_float(err_std),
    }


def compute_series_stats(series: pd.Series) -> Dict[str, Any]:
    s = pd.Series(series).dropna()
    if s.empty:
        return {"n": 0}

    close = s.astype(float)
    returns = close.pct_change().dropna()

    n = int(close.shape[0])
    last = float(close.iloc[-1])
    first = float(close.iloc[0])
    min_v = float(close.min())
    max_v = float(close.max())

    # 252 trading days per year approximations
    if not returns.empty:
        mean_daily = float(returns.mean())
        vol_daily = float(returns.std(ddof=1)) if returns.size >= 2 else 0.0
        ann_return = (1.0 + mean_daily) ** 252 - 1.0
        ann_vol = vol_daily * np.sqrt(252.0)
        sharpe = (ann_return / ann_vol) if ann_vol > 0 else None
        skew = _safe_float(returns.skew())
        kurt = _safe_float(returns.kurtosis())
    else:
        mean_daily = vol_daily = ann_return = ann_vol = 0.0
        sharpe = skew = kurt = None

    # Max drawdown on close curve
    running_max = close.cummax()
    drawdown = (close / running_max) - 1.0
    max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0

    # 52-week high/low if available (252 trading days)
    last_252 = close.tail(252)
    hi_52w = float(last_252.max()) if not last_252.empty else max_v
    lo_52w = float(last_252.min()) if not last_252.empty else min_v

    return {
        "n": n,
        "first_close": first,
        "last_close": last,
        "min_close": min_v,
        "max_close": max_v,
        "change_pct": _safe_float(_pct(first, last)),
        "mean_daily_return_pct": _safe_float(mean_daily * 100.0),
        "daily_volatility_pct": _safe_float(vol_daily * 100.0),
        "annualized_return_pct": _safe_float(ann_return * 100.0),
        "annualized_volatility_pct": _safe_float(ann_vol * 100.0),
        "sharpe_ratio": _safe_float(sharpe),
        "max_drawdown_pct": _safe_float(max_drawdown * 100.0),
        "skew": skew,
        "kurtosis": kurt,
        "high_52w": hi_52w,
        "low_52w": lo_52w,
    }


def summarize_forecast(
    forecast_values: np.ndarray,
    residual_std: Optional[float] = None,
) -> Dict[str, Any]:
    fv = np.asarray(forecast_values, dtype=float)
    fv = fv[np.isfinite(fv)]
    if fv.size == 0:
        return {}

    out: Dict[str, Any] = {
        "forecast_mean": float(np.mean(fv)),
        "forecast_std": float(np.std(fv, ddof=1)) if fv.size >= 2 else 0.0,
        "forecast_min": float(np.min(fv)),
        "forecast_max": float(np.max(fv)),
    }

    if residual_std is not None and np.isfinite(residual_std) and residual_std > 0:
        # Simple, model-agnostic interval using residual std (approximate).
        z = 1.96
        out["interval_95"] = {
            "lower": (fv - z * residual_std).tolist(),
            "upper": (fv + z * residual_std).tolist(),
        }
        out["residual_std"] = float(residual_std)

    return out

