"""Hyperparameter sweep: 15-trial random search logged to the local W&B instance."""

from __future__ import annotations

import os
from typing import Any

import wandb
from dotenv import load_dotenv

from wandb_demo.data import prepare_dataset
from wandb_demo.tickers import make_ticker
from wandb_demo.train import fit_and_log

SWEEP_CONFIG: dict[str, Any] = {
    "method": "random",
    "metric": {"name": "val_accuracy", "goal": "maximize"},
    "parameters": {
        "max_depth": {"value": 5},
        "learning_rate": {"distribution": "uniform", "min": 0.07, "max": 0.10},
        "n_estimators": {"distribution": "int_uniform", "min": 180, "max": 250},
        "objective": {"value": "binary:logistic"},
        "random_state": {"value": 42},
    },
}


def sweep_train() -> None:
    """Single sweep trial — called repeatedly by wandb.agent().

    W&B pre-populates wandb.config with the sampled hyperparameters before
    each call, so wandb.init() here picks them up automatically.
    """
    ticker_symbol = os.getenv("TICKER", "SPY")
    lookback_years = int(os.getenv("LOOKBACK_YEARS", "5"))
    project = os.getenv("WANDB_PROJECT", "wandb-demo")
    ticker = make_ticker(ticker_symbol)

    dataset = prepare_dataset(ticker, lookback_years)

    with wandb.init(project=project, config={"ticker": ticker_symbol}) as run:
        fit_and_log(
            run,
            dataset["X_train"],
            dataset["X_test"],
            dataset["y_train"],
            dataset["y_test"],
        )


if __name__ == "__main__":
    load_dotenv()
    project = os.getenv("WANDB_PROJECT", "wandb-demo")
    print("Initialising sweep …")
    sweep_id = wandb.sweep(SWEEP_CONFIG, project=project)
    print(f"Sweep ID: {sweep_id} — launching 15 agents …")
    wandb.agent(sweep_id, function=sweep_train, count=20)
    print("Sweep complete.")
