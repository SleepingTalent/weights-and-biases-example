"""Provision a W&B local user and write .env for CI e2e tests.

Run once after `docker compose up -d --wait` on a fresh W&B instance.
Uses Playwright to complete the first-user signup form, then generates
an API key via the authenticated GraphQL session.  A second browser
performs a clean /login so the saved storage state contains proper
session cookies (the signup form does not redirect on success, so its
cookies are incomplete).

Environment variables consumed (all optional with defaults):
  WANDB_BASE_URL       – W&B local server URL  (default: http://localhost:8080)
  WANDB_CI_FULLNAME    – signup full name       (default: CI User)
  WANDB_CI_EMAIL       – signup email           (default: ci@example.com)
  WANDB_CI_USERNAME    – signup username        (default: ci-user)
  WANDB_CI_PASSWORD    – signup password        (default: CI_Password123!)
  WANDB_PROJECT        – W&B project name       (default: wandb-demo)
  TICKER               – yfinance ticker symbol (default: SPY)
  LOOKBACK_YEARS       – years of history       (default: 5)
"""

from __future__ import annotations

import os
import sys

from playwright.sync_api import sync_playwright

WANDB_BASE_URL = os.getenv("WANDB_BASE_URL", "http://localhost:8080")
CI_FULLNAME = os.getenv("WANDB_CI_FULLNAME", "CI User")
CI_EMAIL = os.getenv("WANDB_CI_EMAIL", "ci@example.com")
CI_USERNAME = os.getenv("WANDB_CI_USERNAME", "ci-user")
CI_PASSWORD = os.getenv("WANDB_CI_PASSWORD", "CI_Password123!")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "wandb-demo")
TICKER = os.getenv("TICKER", "SPY")
LOOKBACK_YEARS = os.getenv("LOOKBACK_YEARS", "5")
AUTH_STATE_PATH = ".playwright-auth.json"


def _graphql(page: object, query: str) -> dict:  # type: ignore[type-arg]
    """Run a GraphQL query/mutation from the authenticated Playwright page."""
    result: dict = (page).evaluate(  # type: ignore[union-attr,assignment]
        "(q) => fetch('/graphql', {"
        "  method: 'POST',"
        "  headers: {'Content-Type': 'application/json'},"
        "  body: JSON.stringify({ query: q })"
        "}).then(r => r.json())",
        query,
    )
    return result


def signup_and_get_api_key() -> tuple[str, str]:
    """Complete the W&B first-user signup and return (api_key, actual_username).

    Two browser sessions are used:
    1. Signup browser — fills the first-user creation form and generates an API
       key via GraphQL.  The signup form does not redirect on success, so its
       cookies are not suitable for reuse.
    2. Login browser — performs a clean /login with the same credentials and
       saves the resulting session cookies to AUTH_STATE_PATH.  These cookies
       are what the Playwright test browser loads so protected pages render
       without redirecting to /signup.
    """
    with sync_playwright() as playwright:
        # --- Browser 1: create the account and get the API key ---
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()

        page.goto(f"{WANDB_BASE_URL}/signup")
        page.wait_for_load_state("networkidle")
        page.wait_for_selector('[data-test="name-input"]', timeout=60_000)

        page.locator('[data-test="name-input"]').fill(CI_FULLNAME)
        page.locator('[data-test="email-input"]').fill(CI_EMAIL)
        page.locator('[data-test="username-input"]').fill(CI_USERNAME)
        page.locator('[data-test="username-password"]').fill(CI_PASSWORD)
        page.locator('[data-test="terms-and-conditions-checkbox"]').get_by_role(
            "checkbox"
        ).check()
        page.get_by_text("Continue").click()

        # The signup form stays at /signup on success; wait for the viewer
        # query to confirm the account was created before proceeding.
        page.wait_for_timeout(3_000)

        viewer = _graphql(page, "{ viewer { username } }")
        actual_username: str = viewer["data"]["viewer"]["username"]

        key_result = _graphql(
            page,
            'mutation { generateApiKey(input: {description: "ci"}) { apiKey { name } } }',
        )
        api_key: str = key_result["data"]["generateApiKey"]["apiKey"]["name"]
        browser.close()

        # --- Browser 2: log in cleanly and save session cookies ---
        browser2 = playwright.chromium.launch(headless=True)
        context2 = browser2.new_context()
        page2 = context2.new_page()

        page2.goto(f"{WANDB_BASE_URL}/login")
        page2.wait_for_load_state("networkidle")
        page2.locator("input[name=email]").fill(CI_EMAIL)
        page2.locator("input[name=password]").fill(CI_PASSWORD)
        page2.locator("input[name=password]").press("Enter")
        page2.wait_for_url(f"{WANDB_BASE_URL}/home", timeout=20_000)
        page2.wait_for_load_state("networkidle")

        context2.storage_state(path=AUTH_STATE_PATH)
        browser2.close()

    return api_key, actual_username


def write_env(api_key: str, entity: str) -> None:
    """Write .env with all values needed for `task test-e2e`."""
    env_content = (
        f"WANDB_BASE_URL={WANDB_BASE_URL}\n"
        f"WANDB_API_KEY={api_key}\n"
        f"WANDB_ENTITY={entity}\n"
        f"WANDB_PROJECT={WANDB_PROJECT}\n"
        f"TICKER={TICKER}\n"
        f"LOOKBACK_YEARS={LOOKBACK_YEARS}\n"
        f"PLAYWRIGHT_AUTH_STATE={AUTH_STATE_PATH}\n"
    )
    with open(".env", "w") as fh:
        fh.write(env_content)


if __name__ == "__main__":
    print(f"Provisioning W&B CI user at {WANDB_BASE_URL} …")
    try:
        api_key, entity = signup_and_get_api_key()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    write_env(api_key, entity)
    print(f"Done — entity: {entity}, key: {api_key[:24]}…")
    print(".env written.")
