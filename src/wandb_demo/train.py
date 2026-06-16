"""Single XGBoost training run with full W&B experiment tracking."""

from __future__ import annotations

import os
import tempfile
from typing import Any

import wandb
import xgboost as xgb
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, log_loss

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
                if metric_name == "error":
                    # XGBoost "error" = 1 - accuracy; convert for readability
                    metrics[f"{prefix}_accuracy"] = 1.0 - values[-1]
                else:
                    metrics[f"{prefix}_{metric_name}"] = values[-1]
        wandb.log(metrics, step=epoch)
        return False  # returning True would stop training early


def train(config: dict[str, Any]) -> None:
    """Run a single tracked XGBoost experiment and log everything to W&B.

    Logs per-round train/val logloss and accuracy, final test metrics, and
    the saved model file as a W&B artifact.
    """
    ticker = os.getenv("TICKER", "SPY")
    lookback_years = int(os.getenv("LOOKBACK_YEARS", "5"))
    project = os.getenv("WANDB_PROJECT", "wandb-demo")

    print(f"Fetching {lookback_years}y of {ticker} data …")
    dataset = prepare_dataset(ticker, lookback_years)
    X_train = dataset["X_train"]
    X_test = dataset["X_test"]
    y_train = dataset["y_train"]
    y_test = dataset["y_test"]

    with wandb.init(project=project, config=config) as run:
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


if __name__ == "__main__":
    load_dotenv()
    train(DEFAULT_CONFIG)
