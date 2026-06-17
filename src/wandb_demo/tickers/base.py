"""Abstract Ticker base class and shared technical indicator helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf
from numpy.typing import NDArray


def _rsi(series: pd.Series[float], period: int = 14) -> pd.Series[float]:
    """Compute Wilder RSI using exponentially weighted averages."""
    delta: pd.Series[float] = series.diff()
    gain: pd.Series[float] = delta.clip(lower=0.0)
    loss: pd.Series[float] = (-delta).clip(lower=0.0)
    avg_gain: pd.Series[float] = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss: pd.Series[float] = loss.ewm(com=period - 1, min_periods=period).mean()
    rs: pd.Series[float] = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series[float]:
    """Compute Average True Range using exponential smoothing."""
    high: pd.Series[float] = df["High"]
    low: pd.Series[float] = df["Low"]
    prev_close: pd.Series[float] = df["Close"].shift(1)
    tr: NDArray[Any] = np.maximum(
        (high - low).to_numpy(),
        np.maximum(
            (high - prev_close).abs().to_numpy(),
            (low - prev_close).abs().to_numpy(),
        ),
    )
    tr_series: pd.Series[float] = pd.Series(tr, index=df.index)
    return tr_series.ewm(com=period - 1, min_periods=period).mean()


def _macd_hist(
    series: pd.Series[float], fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.Series[float]:
    """Compute MACD histogram (MACD line minus signal line).

    Positive values indicate bullish momentum; negative values indicate bearish.
    Normalised by price so values are comparable across different price levels.
    """
    ema_fast: pd.Series[float] = series.ewm(span=fast, min_periods=slow).mean()
    ema_slow: pd.Series[float] = series.ewm(span=slow, min_periods=slow).mean()
    macd_line: pd.Series[float] = ema_fast - ema_slow
    signal_line: pd.Series[float] = macd_line.ewm(span=signal, min_periods=signal).mean()
    return (macd_line - signal_line) / series


def _fetch_cross_asset(symbol: str, start: str, end: str) -> pd.Series[float]:
    """Fetch daily Close prices for a cross-asset symbol, normalised to tz-naive dates."""
    t = yf.Ticker(symbol)
    df: pd.DataFrame = t.history(start=start, end=end)
    s: pd.Series[float] = df["Close"]
    dti = pd.DatetimeIndex(s.index)
    if dti.tz is not None:
        dti = dti.tz_localize(None)
    s.index = dti.normalize()
    return s


def _sma50_ratio(series: pd.Series[float], period: int = 50) -> pd.Series[float]:
    """Return normalised distance of price from its 50-day simple moving average.

    Positive = price above SMA (uptrend); negative = price below (downtrend).
    """
    sma: pd.Series[float] = series.rolling(period).mean()
    return (series - sma) / sma


class Ticker(ABC):
    """Abstract ticker — defines the contract for ticker-specific feature engineering."""

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol

    @property
    @abstractmethod
    def feature_cols(self) -> list[str]:
        """Column names produced by features(), used as model inputs."""

    @abstractmethod
    def features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return df with engineered feature columns and a binary next-day target added."""

    def __hash__(self) -> int:
        return hash((type(self).__name__, self.symbol))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Ticker):
            return False
        return type(self) is type(other) and self.symbol == other.symbol

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.symbol!r})"
