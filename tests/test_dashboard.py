"""Playwright tests for the model dashboard (pytest-playwright, running against the live service)."""

from __future__ import annotations

import pytest
import requests
from playwright.sync_api import Page, expect

pytestmark = pytest.mark.ui


@pytest.fixture
def dashboard(page: Page, live_server):
    page.goto(live_server)
    expect(page.get_by_test_id("subtitle")).to_contain_text("Champion:")
    return page


@pytest.fixture(scope="module")
def model_info(live_server):
    return requests.get(f"{live_server}/model", timeout=5).json()


def test_shows_the_champion_and_its_version(dashboard: Page, model_info):
    subtitle = dashboard.get_by_test_id("subtitle")
    expect(subtitle).to_contain_text(model_info["champion"])
    expect(subtitle).to_contain_text(model_info["model_version"])
    expect(dashboard.get_by_test_id("stat")).to_have_count(6)


def test_models_table_matches_the_api(dashboard: Page, model_info):
    rows = dashboard.get_by_test_id("model-row")
    expect(rows).to_have_count(len(model_info["models"]))
    for name, m in model_info["models"].items():
        row = dashboard.locator(f'[data-testid="model-row"][data-model="{name}"]')
        expect(row.get_by_test_id("bar-status")).to_have_text("PASS" if m["passes_quality_bar"] else "FAIL")
        expect(row).to_contain_text(f"{m['metrics']['pr_auc']:.3f}")


def test_only_the_champion_has_a_badge(dashboard: Page, model_info):
    badges = dashboard.get_by_test_id("champion-badge")
    expect(badges).to_have_count(1)
    expect(dashboard.locator(f'[data-model="{model_info["champion"]}"]').get_by_test_id("champion-badge")).to_be_visible()


def test_quality_bar_is_explained(dashboard: Page):
    expect(dashboard.get_by_test_id("quality-bar")).to_contain_text("precision ≥ 80.0%")


def test_typical_payment_is_approved(dashboard: Page):
    dashboard.get_by_role("button", name="Score payment").click()
    result = dashboard.get_by_test_id("score-result")
    expect(result).to_have_attribute("data-decision", "approve")
    expect(result).to_contain_text("Approve")


def test_account_takeover_payment_is_stopped(dashboard: Page):
    dashboard.get_by_label("Amount (EUR)").fill("1800")
    dashboard.get_by_label("Hour of day").fill("2")
    dashboard.get_by_label("Merchant category").select_option("electronics")
    dashboard.get_by_label("Country").select_option("NG")
    dashboard.get_by_label("Payments in last 24h").fill("6")
    dashboard.get_by_label("Amount vs. usual spend").fill("9")
    dashboard.get_by_label("New payee").check()
    dashboard.get_by_label("Trusted device").uncheck()

    with dashboard.expect_response("**/predict") as response_info:
        dashboard.get_by_role("button", name="Score payment").click()
    api_decision = response_info.value.json()["decision"]

    result = dashboard.get_by_test_id("score-result")
    expect(result).to_have_attribute("data-decision", api_decision)
    assert api_decision in {"review", "block"}


def test_invalid_input_shows_a_clear_error(dashboard: Page):
    dashboard.get_by_label("Amount (EUR)").fill("-20")
    dashboard.get_by_role("button", name="Score payment").click()
    result = dashboard.get_by_test_id("score-result")
    expect(result).to_contain_text("Invalid input: amount")
    expect(result).to_have_class("result error")
