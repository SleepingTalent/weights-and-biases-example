"""Tests for the data pipeline — all yfinance calls are mocked (offline)."""

import numpy as np
import pandas as pd
import pytest
from pytest_mock import MockerFixture

from wandb_demo.data import fetch_data, prepare_dataset
from wandb_demo.tickers import EURUSDTicker, GBPUSDTicker, SPYTicker, make_ticker


def _make_ohlcv(n: int = 200) -> pd.DataFrame:
    """Synthetic OHLCV DataFrame that mimics yfinance Ticker.history() output."""
    rng = np.random.default_rng(42)
    close = 400.0 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame(
        {
            "Open": close * 0.999,
            "High": close * 1.002,
            "Low": close * 0.998,
            "Close": close,
            "Volume": rng.integers(50_000_000, 100_000_000, n).astype(float),
            "Dividends": 0.0,
            "Stock Splits": 0.0,
        },
        index=pd.date_range("2020-01-01", periods=n, freq="B"),
    )


@pytest.fixture()
def mock_ticker(mocker: MockerFixture) -> None:
    """Mock yfinance in data.py and tickers/base.py (cross-asset fetches live there)."""
    ticker_mock = mocker.MagicMock()
    ticker_mock.history.return_value = _make_ohlcv()
    mocker.patch("wandb_demo.data.yf.Ticker", return_value=ticker_mock)
    mocker.patch("wandb_demo.tickers.base.yf.Ticker", return_value=ticker_mock)


@pytest.fixture()
def mock_forex_yf(mocker: MockerFixture) -> None:
    """Mock yfinance in tickers/base.py for cross-asset feature tests."""
    ticker_mock = mocker.MagicMock()
    ticker_mock.history.return_value = _make_ohlcv()
    mocker.patch("wandb_demo.tickers.base.yf.Ticker", return_value=ticker_mock)


# ---------------------------------------------------------------------------
# fetch_data
# ---------------------------------------------------------------------------


def test_fetch_data_returns_dataframe_with_ohlcv_columns(
    mocker: MockerFixture,
) -> None:
    ticker_mock = mocker.MagicMock()
    ticker_mock.history.return_value = _make_ohlcv()
    mocker.patch("wandb_demo.data.yf.Ticker", return_value=ticker_mock)

    df = fetch_data("SPY", lookback_years=1)

    assert isinstance(df, pd.DataFrame)
    assert {"Open", "High", "Low", "Close", "Volume"}.issubset(df.columns)
    ticker_mock.history.assert_called_once()


# ---------------------------------------------------------------------------
# make_ticker
# ---------------------------------------------------------------------------


def test_make_ticker_returns_equity_ticker_for_spy() -> None:
    assert isinstance(make_ticker("SPY"), SPYTicker)


def test_make_ticker_returns_forex_ticker_for_eurusd() -> None:
    assert isinstance(make_ticker("EURUSD=X"), EURUSDTicker)


def test_make_ticker_returns_gbpusd_ticker() -> None:
    assert isinstance(make_ticker("GBPUSD=X"), GBPUSDTicker)


def test_make_ticker_raises_for_unsupported_symbol() -> None:
    with pytest.raises(ValueError, match="Unsupported ticker"):
        make_ticker("UNKNOWN")


# ---------------------------------------------------------------------------
# SPYTicker.features
# ---------------------------------------------------------------------------


def test_equity_ticker_features_produces_expected_columns() -> None:
    ticker = SPYTicker("SPY")
    result = ticker.features(_make_ohlcv(200))

    expected = {"return_5d", "return_10d", "return_20d", "rsi_14", "volume_ratio", "target"}
    assert expected.issubset(result.columns)


def test_equity_ticker_features_target_is_binary() -> None:
    ticker = SPYTicker("SPY")
    result = ticker.features(_make_ohlcv(200))

    unique_values = set(result["target"].dropna().unique())
    assert unique_values.issubset({0, 1})


def test_equity_ticker_features_target_encodes_next_day_direction() -> None:
    """Target should be 1 when the next close is higher, 0 otherwise."""
    df = _make_ohlcv(50)
    ticker = SPYTicker("SPY")
    result = ticker.features(df)

    for i in range(10, 20):
        expected_target = int(df["Close"].iloc[i + 1] > df["Close"].iloc[i])
        assert result["target"].iloc[i] == expected_target


# ---------------------------------------------------------------------------
# EURUSDTicker.features
# ---------------------------------------------------------------------------


def test_forex_ticker_features_produces_expected_columns(mock_forex_yf: None) -> None:
    ticker = EURUSDTicker("EURUSD=X")
    result = ticker.features(_make_ohlcv(200))

    expected = {
        "return_5d", "return_10d", "return_20d",
        "rsi_14", "atr_14", "macd_hist", "sma50_ratio",
        "dxy_return_5d", "dxy_return_20d",
        "vix_return_5d", "vix_vs_sma20",
        "target",
    }
    assert expected.issubset(result.columns)


def test_forex_ticker_features_has_no_volume_ratio_or_bb_position(mock_forex_yf: None) -> None:
    ticker = EURUSDTicker("EURUSD=X")
    result = ticker.features(_make_ohlcv(200))

    assert "volume_ratio" not in result.columns
    assert "bb_position" not in result.columns


def test_forex_ticker_features_target_is_binary(mock_forex_yf: None) -> None:
    ticker = EURUSDTicker("EURUSD=X")
    result = ticker.features(_make_ohlcv(200))

    unique_values = set(result["target"].dropna().unique())
    assert unique_values.issubset({0, 1})


# ---------------------------------------------------------------------------
# prepare_dataset
# ---------------------------------------------------------------------------


def test_prepare_dataset_has_no_nan_values(mock_ticker: None) -> None:
    result = prepare_dataset(SPYTicker("SPY"), lookback_years=1)

    for key, arr in result.items():
        assert not np.any(np.isnan(arr)), f"NaN values found in {key}"


def test_prepare_dataset_train_test_split_shapes(mock_ticker: None) -> None:
    result = prepare_dataset(SPYTicker("SPY"), lookback_years=1)

    assert result["X_train"].shape[0] == result["y_train"].shape[0]
    assert result["X_test"].shape[0] == result["y_test"].shape[0]
    assert result["X_train"].shape[0] > result["X_test"].shape[0]
    assert result["X_train"].shape[1] == result["X_test"].shape[1]


def test_prepare_dataset_equity_returns_five_features(mock_ticker: None) -> None:
    result = prepare_dataset(SPYTicker("SPY"), lookback_years=1)

    assert result["X_train"].shape[1] == 5  # return_5d, 10d, 20d, rsi_14, volume_ratio


def test_prepare_dataset_forex_returns_eleven_features(mock_ticker: None) -> None:
    result = prepare_dataset(EURUSDTicker("EURUSD=X"), lookback_years=1)

    # return_5d, 10d, 20d, rsi_14, atr_14, macd_hist, sma50_ratio,
    # dxy_return_5d, dxy_return_20d, vix_return_5d, vix_vs_sma20
    assert result["X_train"].shape[1] == 11


def test_prepare_dataset_caches_result_for_same_arguments(
    mocker: MockerFixture,
) -> None:
    ticker_mock = mocker.MagicMock()
    ticker_mock.history.return_value = _make_ohlcv()
    mocker.patch("wandb_demo.data.yf.Ticker", return_value=ticker_mock)

    prepare_dataset.cache_clear()
    try:
        result1 = prepare_dataset(SPYTicker("SPY"), 1)
        result2 = prepare_dataset(SPYTicker("SPY"), 1)
    finally:
        prepare_dataset.cache_clear()

    assert ticker_mock.history.call_count == 1
    np.testing.assert_array_equal(result1["X_train"], result2["X_train"])
