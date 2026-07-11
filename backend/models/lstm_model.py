"""
LSTM model for stock price prediction.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Lazy import TensorFlow to avoid loading if not used
def _get_keras():
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    return tf, Sequential, LSTM, Dense, Dropout, EarlyStopping


def create_sequences(data: np.ndarray, seq_length: int):
    """Create sequences for LSTM: X[i] = data[i:i+seq_length], y[i] = data[i+seq_length]."""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


def build_lstm_model(seq_length: int, units: int = 50):
    """Build LSTM model."""
    _, Sequential, LSTM, Dense, Dropout, _ = _get_keras()
    model = Sequential([
        LSTM(units, return_sequences=True, input_shape=(seq_length, 1)),
        Dropout(0.2),
        LSTM(units, return_sequences=False),
        Dropout(0.2),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def predict_lstm(
    series: pd.Series,
    steps: int = 30,
    seq_length: int = 60,
    epochs: int = 50,
) -> dict:
    """
    Train LSTM and return historical + forecast.
    """
    tf, Sequential, LSTM, Dense, Dropout, EarlyStopping = _get_keras()

    series_clean = series.dropna().values.reshape(-1, 1)
    if len(series_clean) < seq_length + 20:
        raise ValueError(f"Need at least {seq_length + 20} data points for LSTM")

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(series_clean)

    X, y = create_sequences(scaled, seq_length)

    model = build_lstm_model(seq_length)
    early = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
    )
    model.fit(
        X, y,
        epochs=epochs,
        validation_split=0.1,
        callbacks=[early],
        verbose=0,
    )

    # Forecast iteratively
    current = scaled[-seq_length:].copy()
    forecasts = []
    for _ in range(steps):
        pred = model.predict(current.reshape(1, seq_length, 1), verbose=0)
        forecasts.append(pred[0, 0])
        current = np.roll(current, -1, axis=0)
        current[-1] = pred[0, 0]

    forecast_values = scaler.inverse_transform(
        np.array(forecasts).reshape(-1, 1)
    ).flatten().tolist()

    last_date = series.dropna().index[-1]
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=steps,
        freq="B",
    )

    return {
        "historical_dates": [
            d.strftime("%Y-%m-%d")
            for d in series.dropna().index
        ],
        "historical_values": series.dropna().tolist(),
        "forecast_dates": [d.strftime("%Y-%m-%d") for d in future_dates],
        "forecast_values": forecast_values,
        "seq_length": seq_length,
    }
