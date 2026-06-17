"""GBPUSD=X ticker — GBP/USD forex pair feature engineering."""

from __future__ import annotations

import pandas as pd

from wandb_demo.tickers.base import (
    Ticker,
    _atr,
    _fetch_cross_asset,
    _macd_hist,
    _rsi,
    _sma50_ratio,
)


class GBPUSDTicker(Ticker):
    """GBP/USD forex pair (GBPUSD=X) — volume-free features suited to OTC FX markets.

    DXY captures USD strength, VIX captures risk-off flows, and EURUSD returns
    capture the EUR/GBP cross — one of the most actively traded currency pairs,
    highly correlated with GBPUSD. GBP is ~11% of DXY (vs EUR ~57%), so the
    EUR/GBP relationship adds signal that DXY alone can't capture.
    """

    @property
    def feature_cols(self) -> list[str]:
        return [
            "return_5d", "return_10d", "return_20d",
            "rsi_14", "atr_14", "macd_hist", "sma50_ratio",
            "dxy_return_5d", "dxy_return_20d",
            "vix_return_5d", "vix_vs_sma20",
            "eurusd_return_5d", "eurusd_return_20d",
        ]

    def features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price, volatility, momentum, and cross-asset features plus next-day target."""
        out = df.copy()
        close: pd.Series[float] = out["Close"]

        out["return_5d"] = close.pct_change(5)
        out["return_10d"] = close.pct_change(10)
        out["return_20d"] = close.pct_change(20)
        out["rsi_14"] = _rsi(close)
        out["atr_14"] = _atr(out)
        out["macd_hist"] = _macd_hist(close)
        out["sma50_ratio"] = _sma50_ratio(close)

        # Normalise the main df index to tz-naive for cross-asset alignment
        idx = pd.DatetimeIndex(out.index)
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        idx = idx.normalize()
        start = idx[0].strftime("%Y-%m-%d")
        end = idx[-1].strftime("%Y-%m-%d")

        dxy = _fetch_cross_asset("DX-Y.NYB", start, end).reindex(idx, method="ffill")
        vix = _fetch_cross_asset("^VIX", start, end).reindex(idx, method="ffill")
        eurusd = _fetch_cross_asset("EURUSD=X", start, end).reindex(idx, method="ffill")

        # Assign by position to avoid index mismatch between tz-aware and tz-naive
        out["dxy"] = dxy.to_numpy()
        out["vix"] = vix.to_numpy()
        out["eurusd"] = eurusd.to_numpy()

        out["dxy_return_5d"] = out["dxy"].pct_change(5)
        out["dxy_return_20d"] = out["dxy"].pct_change(20)
        out["vix_return_5d"] = out["vix"].pct_change(5)
        vix_sma20: pd.Series[float] = out["vix"].rolling(20).mean()
        out["vix_vs_sma20"] = (out["vix"] - vix_sma20) / vix_sma20
        out["eurusd_return_5d"] = out["eurusd"].pct_change(5)
        out["eurusd_return_20d"] = out["eurusd"].pct_change(20)

        out = out.drop(columns=["dxy", "vix", "eurusd"])
        out["target"] = (close.shift(-1) > close).astype(int)
        return out
