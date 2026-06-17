"""Shared pytest fixtures and configuration for all test suites."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
from dotenv import load_dotenv

load_dotenv()


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
    """Authenticate the Playwright browser for W&B local.

    In CI the setup script (ci_setup_wandb.py) saves the signed-in browser
    session to PLAYWRIGHT_AUTH_STATE after signup.  Loading that file gives
    the test browser real session cookies so protected pages don't redirect to
    /signup.

    For local dev (no auth-state file) we fall back to injecting the API key
    as a Bearer header, which works when the W&B instance already has users.
    """
    args: dict[str, Any] = {**browser_context_args}
    auth_state = os.getenv("PLAYWRIGHT_AUTH_STATE", "")
    if auth_state and Path(auth_state).exists():
        args["storage_state"] = auth_state
    else:
        api_key = os.getenv("WANDB_API_KEY", "")
        args["extra_http_headers"] = {"Authorization": f"Bearer {api_key}"}
    return args
