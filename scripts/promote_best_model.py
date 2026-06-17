"""Find the best model for a given ticker and promote it to the 'best' alias."""

from __future__ import annotations

import os
import sys

import wandb
from dotenv import load_dotenv

load_dotenv()


def _ticker_slug(symbol: str) -> str:
    return symbol.lower().replace("=x", "").replace(".", "")


def promote_best_model() -> None:
    entity = os.getenv("WANDB_ENTITY", "ci-user")
    project = os.getenv("WANDB_PROJECT", "wandb-demo")
    ticker_symbol = os.getenv("TICKER", "")

    if not ticker_symbol:
        print("ERROR: TICKER env var is required. Use task promote_eurusd / promote_gbpusd / promote_spy.")
        sys.exit(1)

    slug = _ticker_slug(ticker_symbol)
    artifact_name = f"xgboost-model-{slug}"

    api = wandb.Api()
    runs = list(api.runs(f"{entity}/{project}"))

    ticker_runs = [
        r for r in runs
        if r.config.get("ticker") == ticker_symbol
        and r.summary.get("test_accuracy") is not None
    ]

    if not ticker_runs:
        print(f"No runs found for ticker '{ticker_symbol}'.")
        sys.exit(1)

    best_run = max(ticker_runs, key=lambda r: r.summary["test_accuracy"])
    best_acc: float = best_run.summary["test_accuracy"]

    print(f"Ticker:    {ticker_symbol}")
    print(f"Best run:  {best_run.name} ({best_run.id})")
    print(f"Sweep:     {best_run.sweep_name or 'none (standalone run)'}")
    print(f"Accuracy:  {best_acc:.4f}")
    print(f"Config:    depth={best_run.config.get('max_depth')}  "
          f"lr={best_run.config.get('learning_rate'):.4f}  "
          f"trees={best_run.config.get('n_estimators')}")

    model_artifact = next(
        (a for a in best_run.logged_artifacts() if a.type == "model"),
        None,
    )

    if model_artifact is None:
        print(f"No model artifact found on run {best_run.name}.")
        sys.exit(1)

    existing = list(model_artifact.aliases)
    if "best" not in existing:
        model_artifact.aliases = existing + ["best"]
        model_artifact.save()
        print(f"\nPromoted {artifact_name} v{model_artifact.version} → alias 'best'")
    else:
        print(f"\n{artifact_name} v{model_artifact.version} is already tagged 'best'")


if __name__ == "__main__":
    promote_best_model()
