"""Data pipeline: fetch, feature engineering, and train/test preparation."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from functools import lru_cache
from typing import Any

import pandas as pd
import yfinance as yf
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split

FEATURE_COLS: list[str] = [
    "return_5d",
    "return_10d",
    "return_20d",
    "rsi_14",
    "volume_ratio",
]


def fetch_data(ticker: str, lookback_years: int) -> pd.DataFrame:
    """Fetch daily OHLCV data for *ticker* covering the last *lookback_years* years."""
    end = datetime.now(tz=UTC)
    start = end - timedelta(days=lookback_years * 365)
    t = yf.Ticker(ticker)
    df: pd.DataFrame = t.history(
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
    )
    return df


def _rsi(series: pd.Series[float], period: int = 14) -> pd.Series[float]:
    """Compute Wilder RSI using exponentially weighted averages."""
    delta: pd.Series[float] = series.diff()
    gain: pd.Series[float] = delta.clip(lower=0.0)
    loss: pd.Series[float] = (-delta).clip(lower=0.0)
    avg_gain: pd.Series[float] = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss: pd.Series[float] = loss.ewm(com=period - 1, min_periods=period).mean()
    rs: pd.Series[float] = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling-return, RSI, volume-ratio features, and a binary next-day target."""
    out = df.copy()
    close: pd.Series[float] = out["Close"]
    out["return_5d"] = close.pct_change(5)
    out["return_10d"] = close.pct_change(10)
    out["return_20d"] = close.pct_change(20)
    out["rsi_14"] = _rsi(close)
    out["volume_ratio"] = out["Volume"] / out["Volume"].rolling(20).mean()
    out["target"] = (close.shift(-1) > close).astype(int)
    return out


@lru_cache(maxsize=4)
def prepare_dataset(ticker: str, lookback_years: int) -> dict[str, NDArray[Any]]:
    """Return a train/test split dict ready for model training.

    Keys: X_train, X_test, y_train, y_test (all numpy arrays, no NaN).
    """
    df = fetch_data(ticker, lookback_years)
    df = engineer_features(df)
    df = df.dropna(subset=FEATURE_COLS + ["target"])

    X: NDArray[Any] = df[FEATURE_COLS].to_numpy(dtype=float)
    y: NDArray[Any] = df["target"].to_numpy(dtype=float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }
