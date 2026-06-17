"""Data pipeline: fetch OHLCV data and prepare train/test splits."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from functools import lru_cache
from typing import Any

import pandas as pd
import yfinance as yf
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split

from wandb_demo.tickers import Ticker


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


@lru_cache(maxsize=4)
def prepare_dataset(ticker: Ticker, lookback_years: int) -> dict[str, NDArray[Any]]:
    """Return a train/test split dict ready for model training.

    Keys: X_train, X_test, y_train, y_test (all numpy arrays, no NaN).
    """
    df = fetch_data(ticker.symbol, lookback_years)
    df = ticker.features(df)
    df = df.dropna(subset=ticker.feature_cols + ["target"])

    X: NDArray[Any] = df[ticker.feature_cols].to_numpy(dtype=float)
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
