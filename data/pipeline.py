"""
Data pipeline for the Stock Prediction LSTM project.

Handles:
- Downloading OHLCV data via yfinance
- Feature engineering (technical indicators)
- Target construction (log returns, NOT raw price)
- Chronological train/val/test split
- Scaling fit ONLY on train data (avoids leakage)
- Sliding-window sequence construction for LSTM input

Usage:
    from data.pipeline import load_dataset
    train_ds, val_ds, test_ds, scaler_x, scaler_y = load_dataset("AAPL")
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add SMA, EMA, RSI, MACD to an OHLCV dataframe. Expects columns:
    Open, High, Low, Close, Volume."""
    df = df.copy()

    df["SMA_10"] = df["Close"].rolling(window=10).mean()
    df["SMA_30"] = df["Close"].rolling(window=30).mean()
    df["EMA_10"] = df["Close"].ewm(span=10, adjust=False).mean()

    # RSI (14-day)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(window=14).mean()
    loss = (-delta.clip(upper=0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # MACD
    ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema_12 - ema_26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # Target: next-day log return (this is what the model actually predicts)
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["target_next_log_return"] = df["log_return"].shift(-1)

    df = df.dropna().reset_index(drop=True)
    return df


FEATURE_COLUMNS = [
    "Open", "High", "Low", "Close", "Volume",
    "SMA_10", "SMA_30", "EMA_10", "RSI_14", "MACD", "MACD_signal", "log_return",
]
TARGET_COLUMN = "target_next_log_return"


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #

def download_ohlcv(ticker: str, start: str = "2015-01-01", end: str | None = None) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'. Check the symbol.")
    df = df.reset_index()
    # yfinance sometimes returns MultiIndex columns for single tickers; flatten if so
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if c[0] != "" else c[1] for c in df.columns]
    # Ensure date column is named "Date"
    if "Date" not in df.columns:
        # yfinance uses "Datetime" for intraday, "Date" for daily
        for col in ["Datetime", "date", "datetime"]:
            if col in df.columns:
                df = df.rename(columns={col: "Date"})
                break
    return df


# --------------------------------------------------------------------------- #
# Windowing
# --------------------------------------------------------------------------- #

def make_sequences(features: np.ndarray, target: np.ndarray, window: int):
    """Turn a (T, F) feature array and (T,) target array into
    (N, window, F) sequences and (N,) targets, where each sequence
    predicts the target immediately following it."""
    X, y = [], []
    for i in range(len(features) - window):
        X.append(features[i:i + window])
        y.append(target[i + window])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Dataset container
# --------------------------------------------------------------------------- #

@dataclass
class SplitData:
    X: np.ndarray
    y: np.ndarray
    dates: pd.Series  # date of the day being predicted, for plotting later


def load_dataset(
    ticker: str,
    start: str = "2015-01-01",
    end: str | None = None,
    window: int = 60,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
):
    """Full pipeline: download -> engineer features -> scale (fit on train only) ->
    window into sequences -> chronological split on sequences.

    Returns: train, val, test (each a SplitData), and the fitted scalers.
    """
    raw = download_ohlcv(ticker, start=start, end=end)
    df = add_technical_indicators(raw)

    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    # Split raw dataframe chronologically FIRST (before windowing)
    # This ensures no data leakage between splits
    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train:n_train + n_val]
    test_df = df.iloc[n_train + n_val:]

    # Fit scalers on TRAIN ONLY -- critical to avoid leakage
    scaler_x = StandardScaler().fit(train_df[FEATURE_COLUMNS])
    scaler_y = StandardScaler().fit(train_df[[TARGET_COLUMN]])

    def _prep(part_df: pd.DataFrame) -> SplitData:
        x_scaled = scaler_x.transform(part_df[FEATURE_COLUMNS])
        y_scaled = scaler_y.transform(part_df[[TARGET_COLUMN]]).ravel()
        X, y = make_sequences(x_scaled, y_scaled, window)
        dates = part_df["Date"].iloc[window:].reset_index(drop=True)
        return SplitData(X=X, y=y, dates=dates)

    train_ds = _prep(train_df)
    val_ds = _prep(val_df)
    test_ds = _prep(test_df)

    return train_ds, val_ds, test_ds, scaler_x, scaler_y


if __name__ == "__main__":
    train_ds, val_ds, test_ds, sx, sy = load_dataset("AAPL", window=60)
    print("Train:", train_ds.X.shape, train_ds.y.shape)
    print("Val:  ", val_ds.X.shape, val_ds.y.shape)
    print("Test: ", test_ds.X.shape, test_ds.y.shape)
