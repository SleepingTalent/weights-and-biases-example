# W&B Local Tracking Demo

A self-contained demo that runs a self-hosted [Weights & Biases](https://wandb.ai) instance via Docker and logs a real XGBoost training run — plus a hyperparameter sweep — against it. No cloud account required.

Built as a portfolio artifact to demonstrate local MLOps experiment tracking, artifact logging, and hyperparameter optimisation using W&B's distinctive sweep visualisation.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and Docker Compose
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (Python package manager)

---

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

### 4. Create your account

Open [http://localhost:8080](http://localhost:8080) in your browser. On a fresh instance the server goes straight to the **Sign Up** page — there are no existing accounts.

Fill in the form:

| Field | What to enter |
|---|---|
| Full name | Your name |
| Email | Any email address |
| Username | Your preferred username |
| Password | Any password (min 8 characters) |

Check **I agree to the Terms** and click **Continue**. The server will log you in automatically.

> **Note:** The W&B local server assigns the username `local` to the very first account created on a fresh instance, regardless of what you type. You can see your actual username at [http://localhost:8080/settings](http://localhost:8080/settings).

### 5. Retrieve your API key

After logging in, navigate to:

```
http://localhost:8080/authorize
```

Click **Generate new API key**. The key will be displayed — **copy it now**, as it is only shown once in full.

### 6. Add your credentials to `.env`

Open `.env` and fill in the values from steps 4 and 5:

```
WANDB_API_KEY=your-api-key-here
WANDB_ENTITY=your-username-here
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

---

## Starting fresh

### Delete all runs from a project

To clear the run history without destroying your account or server:

1. Open the project in the dashboard: [http://localhost:8080](http://localhost:8080)
2. In the **Runs** table, tick the checkbox in the header row to select all runs
3. Click the **Delete** button (bin icon) that appears in the toolbar
4. Confirm the deletion

The project itself and your account are kept — only the run records and their logged metrics are removed.

### Full reset (wipe everything and start over)

This destroys all data — accounts, runs, artifacts, and the project history — and gives you a completely fresh W&B instance:

```bash
# Stop the server and remove the persistent volume
docker compose down -v
```

The `-v` flag removes the `wandb-demo-data` volume. The next `task up` will pull a clean container with no users or data, and you will need to go through the [first-time setup](#first-time-setup) again.

> **Warning:** This is irreversible. All experiment history will be lost.

---

## Dashboard highlights

After running both commands, the W&B dashboard at [http://localhost:8080](http://localhost:8080) will show:

- **Runs table** — each training run with logged hyperparameters and final metrics
- **Charts** — per-round logloss and accuracy curves for the single training run
- **Artifacts** — the saved XGBoost model file
- **Sweep** — parallel coordinates plot across 15 runs with hyperparameter importance

---

## Available tasks

| Command | Description |
|---|---|
| `uv run task up` | Start W&B local server on `localhost:8080` |
| `uv run task down` | Stop W&B local server |
| `uv run task train` | Run a single tracked training experiment |
| `uv run task sweep` | Run a 15-trial hyperparameter sweep |
| `uv run task test` | Run the offline unit tests |
| `uv run task test-e2e` | Start W&B, run E2E BDD tests, stop W&B |
| `uv run task test-e2e-watch` | Same as above but with a headed browser and 500 ms slowmo |
| `uv run task lint` | Lint and type-check (`ruff` + `mypy --strict`) |

---

## Project structure

```
weights-biases-example/
  src/
    wandb_demo/
      data.py       # Yahoo Finance fetch + feature engineering
      train.py      # Single XGBoost run with W&B tracking
      sweep.py      # Hyperparameter sweep controller
  tests/
    features/
      training_run.feature   # BDD scenario
    steps/
      test_training_run.py   # Playwright + pytest-bdd step definitions
    conftest.py
    test_data.py             # Offline unit tests for data pipeline
  scripts/
    ci_setup_wandb.py        # Provisions a CI user on a fresh W&B instance
  .github/
    workflows/
      ci.yml                 # lint → unit test pipeline
  docker-compose.yml
  pyproject.toml
  .env.example
  README.md
```

---

## Configuration

All config is via `.env` (copy from `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `WANDB_BASE_URL` | `http://localhost:8080` | W&B server URL |
| `WANDB_API_KEY` | — | API key from W&B settings |
| `WANDB_ENTITY` | — | Your W&B username (often `local` on a fresh instance) |
| `WANDB_PROJECT` | `wandb-demo` | Project name in W&B |
| `TICKER` | `SPY` | Yahoo Finance ticker symbol |
| `LOOKBACK_YEARS` | `5` | Years of historical data to fetch |

---

## Stack

- **Python 3.12+** with `uv` and `pyproject.toml`
- **XGBoost** — gradient boosted classifier
- **yfinance** — Yahoo Finance market data
- **wandb** — experiment tracking, artifact logging, sweep orchestration
- **Docker Compose** — self-hosted W&B local server
- **taskipy** — task runner
- **ruff + mypy** — linting and strict type checking
- **pytest + pytest-bdd + pytest-playwright** — unit and E2E BDD tests
