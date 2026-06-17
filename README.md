# W&B Local Tracking Demo

A self-contained demo that runs a self-hosted [Weights & Biases](https://wandb.ai) instance via Docker and trains XGBoost binary classifiers for directional prediction on FX and equity markets. No cloud account required.

Built as a portfolio artifact to demonstrate local MLOps experiment tracking, artifact logging, hyperparameter sweeps, model promotion, and inference using W&B's self-hosted server.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and Docker Compose
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (Python package manager)

## First-time setup

```bash
uv sync
cp .env.example .env
```

## Running the W&B Local Server

```bash
uv run task run-local-server
```

This will:
1. Tear down any existing instance and volume for a clean start
2. Pull the `wandb/local` Docker image and start it at [http://localhost:8080](http://localhost:8080)
3. Create the first user account with these credentials:

| Field | Value |
|---|---|
| Email | `ci@example.com` |
| Username | `ci-user` |
| Password | `CI_Password123!` |

4. Generate an API key and write it to `.env` — ready for training tasks immediately

To stop the server when you're done:

```bash
uv run task down
```

## Supported tickers

| Ticker | Class | Features |
|--------|-------|----------|
| `EURUSD=X` | `EURUSDTicker` | 11 — returns, RSI, ATR, MACD, SMA50 ratio, DXY, VIX |
| `GBPUSD=X` | `GBPUSDTicker` | 13 — same as EURUSD + EURUSD cross-asset returns |
| `SPY` | `SPYTicker` | 5 — returns, RSI, volume ratio |

Best observed test accuracy from narrowed sweeps: EURUSD=X **68.80%**, GBPUSD=X **67.60%**.

## Typical workflow

### 1. Start the server

```bash
uv run task run-local-server
```

### 2. Run a sweep

```bash
uv run task sweep_eurusd   # or sweep_gbpusd / sweep_spy
```

Runs 20 trials of random search over `learning_rate` (0.07–0.10) and `n_estimators` (180–250) with `max_depth=5` fixed. Open [http://localhost:8080](http://localhost:8080) to see the parallel coordinates plot.

### 3. Promote the best model

```bash
uv run task promote_eurusd   # or promote_gbpusd / promote_spy
```

Finds the highest `test_accuracy` run for the ticker across all completed sweep runs and tags that model artifact with the `best` alias.

### 4. Get today's signal

```bash
uv run task predict_eurusd   # or predict_gbpusd / predict_spy
```

Downloads the `best` artifact, fetches the latest market data, and prints a directional signal with probability.

### 5. Export models to disk

```bash
uv run task export_models
```

Downloads the `best` artifact for each ticker into `models/<ticker>/model.json` so models survive a server restart.

## Available tasks

| Command | Description |
|---|---|
| `uv run task run-local-server` | Fresh W&B instance with a provisioned user |
| `uv run task up` | Start W&B server (existing data preserved) |
| `uv run task down` | Stop W&B server (data preserved) |
| **Training** | |
| `uv run task train_eurusd` | Single training run on EURUSD=X |
| `uv run task train_gbpusd` | Single training run on GBPUSD=X |
| `uv run task train_spy` | Single training run on SPY |
| **Sweeps** | |
| `uv run task sweep_eurusd` | 20-trial sweep on EURUSD=X |
| `uv run task sweep_gbpusd` | 20-trial sweep on GBPUSD=X |
| `uv run task sweep_spy` | 20-trial sweep on SPY |
| **Promotion** | |
| `uv run task promote_eurusd` | Tag best EURUSD=X model artifact as `best` |
| `uv run task promote_gbpusd` | Tag best GBPUSD=X model artifact as `best` |
| `uv run task promote_spy` | Tag best SPY model artifact as `best` |
| **Prediction** | |
| `uv run task predict_eurusd` | Today's directional signal for EURUSD=X |
| `uv run task predict_gbpusd` | Today's directional signal for GBPUSD=X |
| `uv run task predict_spy` | Today's directional signal for SPY |
| **Models** | |
| `uv run task export_models` | Download best artifacts to `models/` |
| **Testing** | |
| `uv run task test` | Offline unit tests |
| `uv run task test-e2e` | E2E BDD tests (starts + stops W&B automatically) |
| `uv run task lint` | `ruff` + `mypy --strict` |

## Configuration

All config is via `.env` (copy from `.env.example`). `task run-local-server` writes the API key and entity automatically.

| Variable | Default | Description |
|---|---|---|
| `WANDB_BASE_URL` | `http://localhost:8080` | W&B server URL |
| `WANDB_API_KEY` | — | Written automatically by `run-local-server` |
| `WANDB_ENTITY` | `ci-user` | Written automatically by `run-local-server` |
| `WANDB_PROJECT` | `wandb-demo` | Project name in W&B |
| `LOOKBACK_YEARS` | `5` | Years of historical data to fetch |

The ticker is set per-task (e.g. `sweep_eurusd`) rather than as a `.env` variable.

## Project structure

```
weights-biases-example/
  src/
    wandb_demo/
      tickers/
        base.py       # Ticker ABC + shared helpers (RSI, ATR, MACD, cross-asset fetch)
        eurusd.py     # EURUSDTicker — 11 features incl. DXY + VIX
        gbpusd.py     # GBPUSDTicker — 13 features incl. DXY + VIX + EURUSD
        spy.py        # SPYTicker — 5 features
        __init__.py   # make_ticker() factory + _TICKER_MAP
      data.py         # Yahoo Finance fetch + train/test split
      train.py        # Single XGBoost run with W&B tracking
      sweep.py        # Hyperparameter sweep controller
      predict.py      # Load best artifact and output directional signal
  models/
    eurusd/
      model.json      # Best EURUSD=X model (exported from W&B)
    gbpusd/
      model.json      # Best GBPUSD=X model (exported from W&B)
  scripts/
    ci_setup_wandb.py       # Provisions a CI user on a fresh W&B instance
    promote_best_model.py   # Finds best run for TICKER and tags artifact as 'best'
    export_models.py        # Downloads best artifacts to models/
  tests/
    features/
      training_run.feature  # BDD scenario
    steps/
      test_training_run.py  # Playwright + pytest-bdd step definitions
    conftest.py
    test_data.py            # Offline unit tests for data pipeline + tickers
  docker-compose.yml
  pyproject.toml
  .env.example
  README.md
```

## Cleaning up

| What | How |
|---|---|
| Local SDK cache | `rm -rf wandb/` |
| All runs + artifacts (keep account) | Delete via W&B dashboard runs table |
| Full reset (wipes Docker volume) | `docker compose down -v` then `uv run task run-local-server` |

> **Warning:** `docker compose down -v` is irreversible — all experiment history and artifacts are lost. Run `uv run task export_models` first to preserve trained models locally.

## Stack

- **Python 3.12+** with `uv` and `pyproject.toml`
- **XGBoost** — gradient boosted classifier
- **yfinance** — Yahoo Finance market data (OHLCV + cross-asset)
- **wandb** — experiment tracking, artifact logging, sweep orchestration
- **Docker Compose** — self-hosted W&B local server
- **taskipy** — task runner
- **ruff + mypy** — linting and strict type checking
- **pytest + pytest-bdd + pytest-playwright** — unit and E2E BDD tests
