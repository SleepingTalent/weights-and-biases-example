"""SPY ticker — S&P 500 ETF feature engineering."""

from __future__ import annotations

import pandas as pd

from wandb_demo.tickers.base import Ticker, _rsi


class SPYTicker(Ticker):
    """S&P 500 ETF (SPY) — price momentum and volume features.

    Volume is reliable for exchange-traded equities and provides useful
    context on conviction behind price moves.
    """

    @property
    def feature_cols(self) -> list[str]:
        return ["return_5d", "return_10d", "return_20d", "rsi_14", "volume_ratio"]

    def features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling returns, RSI, volume ratio, and binary next-day target."""
        out = df.copy()
        close: pd.Series[float] = out["Close"]
        out["return_5d"] = close.pct_change(5)
        out["return_10d"] = close.pct_change(10)
        out["return_20d"] = close.pct_change(20)
        out["rsi_14"] = _rsi(close)
        out["volume_ratio"] = out["Volume"] / out["Volume"].rolling(20).mean()
        out["target"] = (close.shift(-1) > close).astype(int)
        return out
