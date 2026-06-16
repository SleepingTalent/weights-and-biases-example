"""Provision a W&B local user and write .env for CI e2e tests.

Run once after `docker compose up -d --wait` on a fresh W&B instance.
Uses Playwright to complete the first-user signup form, then generates
an API key via the authenticated GraphQL session.

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

GRAPHQL_URL = f"{WANDB_BASE_URL}/graphql"
GENERATE_KEY_MUTATION = (
    'mutation { generateApiKey(input: {description: "ci"}) { apiKey { name } } }'
)


def signup_and_get_api_key() -> tuple[str, str]:
    """Complete the W&B first-user signup and return (api_key, username)."""
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()

        page.goto(f"{WANDB_BASE_URL}/signup")
        page.wait_for_load_state("domcontentloaded")

        page.locator('[data-test="name-input"]').fill(CI_FULLNAME)
        page.locator('[data-test="email-input"]').fill(CI_EMAIL)
        page.locator('[data-test="username-input"]').fill(CI_USERNAME)
        page.locator('[data-test="username-password"]').fill(CI_PASSWORD)
        page.locator('[data-test="terms-and-conditions-checkbox"]').get_by_role(
            "checkbox"
        ).check()
        page.get_by_text("Continue").click()

        page.wait_for_url(f"{WANDB_BASE_URL}/**", timeout=20_000)

        result: dict = page.evaluate(  # type: ignore[assignment]
            """(mutation) => fetch('/graphql', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ query: mutation })
            }).then(r => r.json())""",
            GENERATE_KEY_MUTATION,
        )

        api_key: str = result["data"]["generateApiKey"]["apiKey"]["name"]
        browser.close()

    return api_key, CI_USERNAME


def write_env(api_key: str, entity: str) -> None:
    """Write .env with all values needed for `task test-e2e`."""
    env_content = (
        f"WANDB_BASE_URL={WANDB_BASE_URL}\n"
        f"WANDB_API_KEY={api_key}\n"
        f"WANDB_ENTITY={entity}\n"
        f"WANDB_PROJECT={WANDB_PROJECT}\n"
        f"TICKER={TICKER}\n"
        f"LOOKBACK_YEARS={LOOKBACK_YEARS}\n"
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
