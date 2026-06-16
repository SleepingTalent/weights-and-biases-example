"""BDD step definitions for the training_run.feature scenario.

Runs exclusively via `task test-e2e` (requires W&B server and valid .env).
The @e2e tag on the feature file maps to pytest.mark.e2e, which excludes
these steps from the offline `task test` suite.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import pytest
import requests
from playwright.sync_api import Page, expect
from pytest_bdd import given, scenario, then, when

from wandb_demo.train import DEFAULT_CONFIG, train

FEATURE = str(Path(__file__).parent.parent / "features" / "training_run.feature")

scenario = partial(scenario, FEATURE)  # type: ignore[assignment]


@pytest.mark.e2e
@scenario("A training run appears in the W&B dashboard")
def test_training_run_in_dashboard() -> None:
    pass


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


@given("the W&B server is running", target_fixture="server_url")
def wb_server_running(wandb_base_url: str) -> str:
    """Assert the local W&B instance is reachable and return its base URL."""
    response = requests.get(wandb_base_url, timeout=10)
    assert response.status_code == 200, (
        f"W&B server not reachable at {wandb_base_url} — run `task up` first"
    )
    return wandb_base_url


@when("I run a training experiment with 10 estimators", target_fixture="run_id")
def run_training_experiment() -> str:
    """Run a fast (10-round) training experiment and return the W&B run ID."""
    config = {**DEFAULT_CONFIG, "n_estimators": 10}
    return train(config)


@then("a new run appears in the project dashboard")
def run_appears_in_dashboard(
    page: Page,
    run_id: str,
    server_url: str,
    wandb_entity: str,
    wandb_project: str,
) -> None:
    run_url = f"{server_url}/{wandb_entity}/{wandb_project}/runs/{run_id}"
    page.goto(run_url)
    page.wait_for_load_state("domcontentloaded")
    # W&B embeds the run ID in the page URL — confirm we landed on the right page
    expect(page).to_have_url(f"{run_url}")


@then("the run has train_accuracy metrics logged")
def run_has_train_accuracy_metrics(page: Page) -> None:
    """train_accuracy should appear in the run summary metrics panel."""
    expect(page.get_by_text("train_accuracy", exact=False).first).to_be_visible(
        timeout=15000
    )


@then("a model artifact is attached to the run")
def run_has_model_artifact(
    page: Page, server_url: str, wandb_entity: str, wandb_project: str, run_id: str
) -> None:
    """Navigate to the run's output artifacts section and verify the model is listed."""
    artifacts_url = f"{server_url}/{wandb_entity}/{wandb_project}/runs/{run_id}/artifacts"
    page.goto(artifacts_url)
    page.wait_for_load_state("domcontentloaded")
    expect(page.get_by_text("xgboost-model", exact=False).first).to_be_visible(
        timeout=15000
    )
