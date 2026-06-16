"""Shared pytest fixtures and configuration for all test suites."""

from __future__ import annotations

import os
from typing import Any

import pytest
from dotenv import load_dotenv

load_dotenv()


def pytest_bdd_apply_tag(tag: str, function: object) -> bool | None:
    """Map @e2e feature-file tags to the pytest.mark.e2e marker."""
    if tag == "e2e":
        pytest.mark.e2e(function)
        return True
    return None


@pytest.fixture(scope="session")
def wandb_base_url() -> str:
    return os.getenv("WANDB_BASE_URL", "http://localhost:8080")


@pytest.fixture(scope="session")
def wandb_entity() -> str:
    return os.getenv("WANDB_ENTITY", "jamesfairbairn")


@pytest.fixture(scope="session")
def wandb_project() -> str:
    return os.getenv("WANDB_PROJECT", "wandb-demo")


@pytest.fixture(scope="session")
def browser_context_args(browser_context_args: dict[str, Any]) -> dict[str, Any]:
    """Inject the W&B API key as a Bearer header on every browser request.

    The W&B local server's GraphQL endpoint accepts Bearer auth, which allows
    the React SPA to render authenticated run pages without a UI login flow.
    """
    api_key = os.getenv("WANDB_API_KEY", "")
    return {
        **browser_context_args,
        "extra_http_headers": {
            "Authorization": f"Bearer {api_key}",
        },
    }
