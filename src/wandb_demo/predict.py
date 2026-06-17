"""Load the latest W&B model artifact and output today's directional signal."""

from __future__ import annotations

import os
import tempfile

import wandb
import xgboost as xgb
from dotenv import load_dotenv

from wandb_demo.data import fetch_data
from wandb_demo.tickers import make_ticker


def predict() -> None:
    """Fetch today's features and print the directional signal from the latest saved model."""
    ticker_symbol = os.getenv("TICKER", "EURUSD=X")
    project = os.getenv("WANDB_PROJECT", "wandb-demo")
    entity = os.getenv("WANDB_ENTITY", "ci-user")

    ticker = make_ticker(ticker_symbol)
    ticker_slug = ticker_symbol.lower().replace("=x", "").replace(".", "")
    artifact_name = f"xgboost-model-{ticker_slug}"

    api = wandb.Api()
    # Prefer the 'best' alias (set by task promote); fall back to 'latest'
    for alias in ("best", "latest"):
        try:
            artifact = api.artifact(f"{entity}/{project}/{artifact_name}:{alias}")
            print(f"Loading {artifact_name}:{alias} from {entity}/{project} …")
            break
        except Exception:
            continue
    else:
        print(
            f"No artifact '{artifact_name}' found. "
            "Run 'uv run task train' or 'uv run task sweep' first."
        )
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        artifact_dir = artifact.download(root=tmpdir)
        clf = xgb.XGBClassifier()
        clf.load_model(f"{artifact_dir}/model.json")

    print(f"Fetching latest {ticker_symbol} data …")
    df = fetch_data(ticker_symbol, lookback_years=1)
    df = ticker.features(df)
    df = df.dropna(subset=ticker.feature_cols)

    latest_row = df[ticker.feature_cols].iloc[[-1]]
    as_of = str(df.index[-1].date())

    prob_up: float = float(clf.predict_proba(latest_row)[0, 1])
    direction = "UP" if prob_up >= 0.5 else "DOWN"
    confidence = max(prob_up, 1.0 - prob_up)

    print()
    print(f"Ticker:      {ticker_symbol}")
    print(f"As of:       {as_of}")
    print(f"Signal:      {direction}")
    print(f"Prob(up):    {prob_up:.1%}")
    print(f"Confidence:  {confidence:.1%}")


if __name__ == "__main__":
    load_dotenv()
    predict()
