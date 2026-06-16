# W&B Local Tracking Demo

A self-contained demo that runs a self-hosted [Weights & Biases](https://wandb.ai) instance via Docker and logs a real XGBoost training run — plus a hyperparameter sweep — against it. No cloud account required.

Built as a portfolio artifact to demonstrate local MLOps experiment tracking, artifact logging, and hyperparameter optimisation using W&B's distinctive sweep visualisation.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and Docker Compose
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (Python package manager)

## First-time setup

Follow these steps once before running any experiments.

### 1. Install dependencies

```bash
uv sync
```

### 2. Create your environment file

```bash
cp .env.example .env
```

### 3. Start the W&B local server

```bash
uv run task up
```

This pulls the `wandb/local` Docker image and starts it at [http://localhost:8080](http://localhost:8080). The named volume `wandb-demo-data` means your experiments persist between restarts.

### 4. Create a local account

Visit [http://localhost:8080](http://localhost:8080) in your browser and click **Log in**. On the signup page, fill in your name, email, username, and a password, accept the terms, and click **Continue**.

> An existing account is already set up — see the [Local W&B account](#local-wb-account) section below for credentials.

### 5. Retrieve your API key

After logging in, navigate to:

```
http://localhost:8080/authorize
```

Click **Generate new API key**. A key will be displayed — **copy it now**, as it is only shown once in full.

### 6. Add the API key to `.env`

Open `.env` and paste the key:

```
WANDB_API_KEY=your-api-key-here
```

Your `.env` is gitignored and will never be committed. To rotate the key later, return to [http://localhost:8080/authorize](http://localhost:8080/authorize).

---

## Running experiments

### Run a training experiment

```bash
uv run task train
```

Fetches 5 years of SPY daily OHLCV data from Yahoo Finance, engineers features (rolling returns, RSI, volume ratio), trains an XGBoost binary classifier, and logs params, per-round metrics, and a model artifact to W&B.

### Run a hyperparameter sweep

```bash
uv run task sweep
```

Runs 15 trials of random search over `max_depth`, `learning_rate`, and `n_estimators`. Head to the W&B dashboard to see the **parallel coordinates plot** showing which parameter combinations drove the best accuracy.

### Stop the server

```bash
uv run task down
```

The named Docker volume (`wandb-demo-data`) persists your experiments between sessions.

## Dashboard highlights

After running both commands, the W&B dashboard at [http://localhost:8080](http://localhost:8080) will show:

- **Runs table** — each training run with logged hyperparameters and final metrics
- **Charts** — per-round logloss and accuracy curves for the single training run
- **Artifacts** — the saved XGBoost model file
- **Sweep** — parallel coordinates plot across 15 runs with hyperparameter importance

## Available tasks

| Command | Description |
|---|---|
| `uv run task up` | Start W&B local server on `localhost:8080` |
| `uv run task down` | Stop W&B local server |
| `uv run task train` | Run a single tracked training experiment |
| `uv run task sweep` | Run a 15-trial hyperparameter sweep |
| `uv run task test` | Run the test suite |
| `uv run task lint` | Lint and type-check (`ruff` + `mypy --strict`) |

## Project structure

```
weights-biases-example/
  src/
    wandb_demo/
      data.py       # Yahoo Finance fetch + feature engineering
      train.py      # Single XGBoost run with W&B tracking
      sweep.py      # Hyperparameter sweep controller
  tests/
    test_data.py    # Unit tests for data pipeline (offline, mocked)
  docker-compose.yml
  pyproject.toml
  .env.example
  README.md
```

## Local W&B account

The local instance has one admin account pre-configured:

| Field | Value |
|---|---|
| URL | [http://localhost:8080](http://localhost:8080) |
| Username | `jamesfairbairn` |
| Email | `jaybono30@googlemail.com` |
| Password | `WandbDemo1234!` |

The API key for this account is stored in `.env` (gitignored — never committed). To retrieve or rotate it, log in and visit [http://localhost:8080/authorize](http://localhost:8080/authorize).

## Configuration

All config is via `.env` (copy from `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `WANDB_BASE_URL` | `http://localhost:8080` | W&B server URL |
| `WANDB_API_KEY` | — | API key from W&B settings |
| `WANDB_PROJECT` | `wandb-demo` | Project name in W&B |
| `TICKER` | `SPY` | Yahoo Finance ticker symbol |
| `LOOKBACK_YEARS` | `5` | Years of historical data to fetch |

## Stack

- **Python 3.12+** with `uv` and `pyproject.toml`
- **XGBoost** — gradient boosted classifier
- **yfinance** — Yahoo Finance market data
- **wandb** — experiment tracking, artifact logging, sweep orchestration
- **Docker Compose** — self-hosted W&B local server
- **taskipy** — task runner
- **ruff + mypy** — linting and strict type checking
- **pytest** — testing
