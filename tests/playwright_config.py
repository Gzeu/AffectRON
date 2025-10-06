# Playwright configuration for AffectRON testing
import os
from playwright.sync_api import Playwright


def pytest_configure(config):
    """Configure pytest for Playwright."""
    # Set up environment variables for testing
    os.environ.setdefault("ENVIRONMENT", "testing")
    os.environ.setdefault("DEBUG", "true")


def pytest_playwright_args():
    """Configure Playwright browser arguments."""
    return [
        "--headed=false",  # Run headless by default
        "--browser=chromium",
        "--viewport-size=1280,720",
        "--disable-web-security",  # For testing CORS
        "--disable-features=VizDisplayCompositor",  # Reduce resource usage
    ]


def pytest_runtest_setup(item):
    """Set up test environment before each test."""
    # Ensure test environment is set
    if "playwright" in item.keywords:
        os.environ["PLAYWRIGHT_TEST"] = "true"


def browser_context_args(browser_context_args):
    """Configure browser context arguments."""
    return {
        **browser_context_args,
        "viewport": {"width": 1280, "height": 720},
        "ignore_https_errors": True,  # For testing with self-signed certificates
        "locale": "en-US",
        "timezone_id": "Europe/Bucharest",
    }
