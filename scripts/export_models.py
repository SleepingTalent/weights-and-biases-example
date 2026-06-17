"""Download the best model artifacts from W&B and save them to models/<ticker>/model.json.

Run via: uv run task export_models
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import wandb
from dotenv import load_dotenv

load_dotenv()

TICKERS = ["EURUSD=X", "GBPUSD=X", "SPY"]


def _ticker_slug(symbol: str) -> str:
    return symbol.lower().replace("=x", "").replace(".", "")


def export_models() -> None:
    """Download the 'best' alias artifact for each ticker into models/<slug>/model.json."""
    entity = os.getenv("WANDB_ENTITY", "ci-user")
    project = os.getenv("WANDB_PROJECT", "wandb-demo")
    models_dir = Path(__file__).parent.parent / "models"

    api = wandb.Api()

    for ticker_symbol in TICKERS:
        slug = _ticker_slug(ticker_symbol)
        artifact_name = f"xgboost-model-{slug}"

        for alias in ("best", "latest"):
            try:
                artifact = api.artifact(f"{entity}/{project}/{artifact_name}:{alias}")
                break
            except Exception:
                continue
        else:
            print(f"[{ticker_symbol}] No artifact found — skipping.")
            continue

        out_dir = models_dir / slug
        out_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = artifact.download(root=tmpdir)
            src = Path(artifact_dir) / "model.json"
            dst = out_dir / "model.json"
            shutil.copy2(src, dst)

        print(f"[{ticker_symbol}] Saved {artifact_name}:{alias} → {dst.relative_to(Path.cwd())}")


if __name__ == "__main__":
    export_models()
