"""Single XGBoost training run with full W&B experiment tracking."""

from __future__ import annotations

import os
import tempfile
from typing import Any

import xgboost as xgb
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, log_loss

import wandb
from wandb_demo.data import prepare_dataset

DEFAULT_CONFIG: dict[str, Any] = {
    "n_estimators": 200,
    "max_depth": 5,
    "learning_rate": 0.1,
    "objective": "binary:logistic",
    "random_state": 42,
}


class _WandbCallback(xgb.callback.TrainingCallback):
    """Log per-round eval metrics to the active W&B run after each boosting round."""

    def after_iteration(
        self,
        model: xgb.Booster,
        epoch: int,
        evals_log: xgb.callback.TrainingCallback.EvalsLog,
    ) -> bool:
        metrics: dict[str, float] = {}
        for dataset, metric_dict in evals_log.items():
            prefix = "train" if dataset == "validation_0" else "val"
            for metric_name, values in metric_dict.items():
                raw = values[-1]
                scalar: float = raw[0] if isinstance(raw, tuple) else raw
                if metric_name == "error":
                    metrics[f"{prefix}_accuracy"] = 1.0 - scalar
                else:
                    metrics[f"{prefix}_{metric_name}"] = scalar
        wandb.log(metrics, step=epoch)
        return False


def fit_and_log(
    run: wandb.sdk.wandb_run.Run,
    X_train: Any,
    X_test: Any,
    y_train: Any,
    y_test: Any,
) -> None:
    """Fit XGBClassifier and log all metrics and artifact to an active W&B run.

    Shared by both the standalone train() and the sweep agent's sweep_train().
    """
    cfg = run.config

    clf = xgb.XGBClassifier(
        n_estimators=int(cfg.n_estimators),
        max_depth=int(cfg.max_depth),
        learning_rate=float(cfg.learning_rate),
        objective=str(cfg.objective),
        eval_metric=["logloss", "error"],
        random_state=int(cfg.random_state),
        callbacks=[_WandbCallback()],
        verbosity=0,
    )

    print(f"Training XGBClassifier ({cfg.n_estimators} rounds) …")
    clf.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False,
    )

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    test_accuracy = float(accuracy_score(y_test, y_pred))
    test_logloss = float(log_loss(y_test, y_prob))

    run.log({"test_accuracy": test_accuracy, "test_logloss": test_logloss})
    print(f"Test accuracy: {test_accuracy:.4f}  |  Test logloss: {test_logloss:.4f}")

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.json")
        clf.save_model(model_path)
        artifact = wandb.Artifact("xgboost-model", type="model")
        artifact.add_file(model_path)
        run.log_artifact(artifact)
        print("Model artifact logged to W&B.")


def train(config: dict[str, Any]) -> str:
    """Run a single tracked XGBoost experiment and log everything to W&B.

    Returns the W&B run ID so callers (including BDD steps) can link to the run.
    """
    ticker = os.getenv("TICKER", "SPY")
    lookback_years = int(os.getenv("LOOKBACK_YEARS", "5"))
    project = os.getenv("WANDB_PROJECT", "wandb-demo")

    print(f"Fetching {lookback_years}y of {ticker} data …")
    dataset = prepare_dataset(ticker, lookback_years)

    run_id = ""
    with wandb.init(project=project, config=config) as run:
        run_id = run.id
        fit_and_log(
            run,
            dataset["X_train"],
            dataset["X_test"],
            dataset["y_train"],
            dataset["y_test"],
        )
    return run_id


if __name__ == "__main__":
    load_dotenv()
    train(DEFAULT_CONFIG)
