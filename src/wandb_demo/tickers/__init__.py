"""Ticker subclasses — one per supported asset, each with hand-tuned features."""

from __future__ import annotations

from wandb_demo.tickers.base import Ticker
from wandb_demo.tickers.eurusd import EURUSDTicker
from wandb_demo.tickers.gbpusd import GBPUSDTicker
from wandb_demo.tickers.spy import SPYTicker

_TICKER_MAP: dict[str, type[Ticker]] = {
    "EURUSD=X": EURUSDTicker,
    "GBPUSD=X": GBPUSDTicker,
    "SPY": SPYTicker,
}


def make_ticker(symbol: str) -> Ticker:
    """Return the asset-specific Ticker subclass for the given symbol.

    Each subclass has features hand-tuned for that asset. Raises ValueError
    for unrecognised symbols — add a new subclass and register it in _TICKER_MAP.
    """
    cls = _TICKER_MAP.get(symbol)
    if cls is None:
        supported = ", ".join(sorted(_TICKER_MAP))
        raise ValueError(f"Unsupported ticker '{symbol}'. Supported: {supported}")
    return cls(symbol)


__all__ = [
    "Ticker",
    "EURUSDTicker",
    "GBPUSDTicker",
    "SPYTicker",
    "make_ticker",
    "_TICKER_MAP",
]
