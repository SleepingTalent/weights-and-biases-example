"""Tests for the data pipeline — all yfinance calls are mocked (offline)."""

import numpy as np
import pandas as pd
import pytest
from pytest_mock import MockerFixture

from wandb_demo.data import engineer_features, fetch_data, prepare_dataset


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
    ticker_mock = mocker.MagicMock()
    ticker_mock.history.return_value = _make_ohlcv()
    mocker.patch("wandb_demo.data.yf.Ticker", return_value=ticker_mock)


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


def test_engineer_features_produces_expected_columns() -> None:
    df = _make_ohlcv(200)
    result = engineer_features(df)

    expected = {"return_5d", "return_10d", "return_20d", "rsi_14", "volume_ratio", "target"}
    assert expected.issubset(result.columns)


def test_engineer_features_target_is_binary() -> None:
    df = _make_ohlcv(200)
    result = engineer_features(df)

    unique_values = set(result["target"].dropna().unique())
    assert unique_values.issubset({0, 1})


def test_engineer_features_target_encodes_next_day_direction() -> None:
    """Target should be 1 when the next close is higher, 0 otherwise."""
    df = _make_ohlcv(50)
    result = engineer_features(df)

    # Spot-check: row i target == 1 iff close[i+1] > close[i]
    for i in range(10, 20):
        expected_target = int(df["Close"].iloc[i + 1] > df["Close"].iloc[i])
        assert result["target"].iloc[i] == expected_target


def test_prepare_dataset_has_no_nan_values(mock_ticker: None) -> None:
    result = prepare_dataset("SPY", lookback_years=1)

    for key, arr in result.items():
        assert not np.any(np.isnan(arr)), f"NaN values found in {key}"


def test_prepare_dataset_train_test_split_shapes(mock_ticker: None) -> None:
    result = prepare_dataset("SPY", lookback_years=1)

    assert result["X_train"].shape[0] == result["y_train"].shape[0]
    assert result["X_test"].shape[0] == result["y_test"].shape[0]
    # 80/20 split — train set must be larger
    assert result["X_train"].shape[0] > result["X_test"].shape[0]
    # Both splits must have the same number of features
    assert result["X_train"].shape[1] == result["X_test"].shape[1]


def test_prepare_dataset_returns_five_features(mock_ticker: None) -> None:
    result = prepare_dataset("SPY", lookback_years=1)

    assert result["X_train"].shape[1] == 5  # return_5d, 10d, 20d, rsi_14, volume_ratio


def test_prepare_dataset_caches_result_for_same_arguments(
    mocker: MockerFixture,
) -> None:
    ticker_mock = mocker.MagicMock()
    ticker_mock.history.return_value = _make_ohlcv()
    mocker.patch("wandb_demo.data.yf.Ticker", return_value=ticker_mock)

    prepare_dataset.cache_clear()
    try:
        result1 = prepare_dataset("SPY", 1)
        result2 = prepare_dataset("SPY", 1)
    finally:
        prepare_dataset.cache_clear()

    assert ticker_mock.history.call_count == 1
    np.testing.assert_array_equal(result1["X_train"], result2["X_train"])
