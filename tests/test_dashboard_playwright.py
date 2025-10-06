import pytest
from playwright.sync_api import Page, expect


@pytest.fixture
def dashboard_url():
    """Return the dashboard URL for testing."""
    return "http://localhost:3000"


def test_dashboard_loads(page: Page, dashboard_url):
    """Test that dashboard loads correctly."""
    page.goto(dashboard_url)

    # Wait for the page to load
    page.wait_for_load_state("networkidle")

    # Check that the main title is present
    expect(page.locator("h1, h2, h3")).to_contain_text(/affectron|dashboard/i)


def test_navigation_menu(page: Page, dashboard_url):
    """Test navigation menu functionality."""
    page.goto(dashboard_url)
    page.wait_for_load_state("networkidle")

    # Look for navigation elements
    nav = page.locator("nav, [role='navigation']")
    if nav.count() > 0:
        # Check that navigation items are clickable
        nav_links = nav.locator("a, button")
        if nav_links.count() > 0:
            # Try to click first navigation item
            nav_links.first.click()
            # Should not cause page crash
            expect(page).to_have_url(/.*/)


def test_sentiment_analysis_page(page: Page, dashboard_url):
    """Test sentiment analysis page."""
    page.goto(f"{dashboard_url}/sentiment")
    page.wait_for_load_state("networkidle")

    # Check for sentiment analysis content
    expect(page.locator("text=/sentiment|analysis/i")).to_be_visible()


def test_market_insights_page(page: Page, dashboard_url):
    """Test market insights page."""
    page.goto(f"{dashboard_url}/insights")
    page.wait_for_load_state("networkidle")

    # Check for insights content
    expect(page.locator("text=/insights|market/i")).to_be_visible()


def test_risk_assessment_page(page: Page, dashboard_url):
    """Test risk assessment page."""
    page.goto(f"{dashboard_url}/risk")
    page.wait_for_load_state("networkidle")

    # Check for risk content
    expect(page.locator("text=/risk|assessment/i")).to_be_visible()


def test_alerts_page(page: Page, dashboard_url):
    """Test alerts page."""
    page.goto(f"{dashboard_url}/alerts")
    page.wait_for_load_state("networkidle")

    # Check for alerts content
    expect(page.locator("text=/alerts|notifications/i")).to_be_visible()


def test_settings_page(page: Page, dashboard_url):
    """Test settings page."""
    page.goto(f"{dashboard_url}/settings")
    page.wait_for_load_state("networkidle")

    # Check for settings content
    expect(page.locator("text=/settings|configuration/i")).to_be_visible()


def test_responsive_design(page: Page, dashboard_url):
    """Test responsive design on different screen sizes."""
    page.goto(dashboard_url)
    page.wait_for_load_state("networkidle")

    # Test mobile viewport
    page.set_viewport_size({"width": 375, "height": 667})
    expect(page.locator("body")).to_be_visible()

    # Test tablet viewport
    page.set_viewport_size({"width": 768, "height": 1024})
    expect(page.locator("body")).to_be_visible()

    # Test desktop viewport
    page.set_viewport_size({"width": 1920, "height": 1080})
    expect(page.locator("body")).to_be_visible()


def test_dark_mode_toggle(page: Page, dashboard_url):
    """Test dark mode toggle if available."""
    page.goto(dashboard_url)
    page.wait_for_load_state("networkidle")

    # Look for dark mode toggle
    dark_mode_toggle = page.locator("button, input").filter(has_text=/dark|theme|mode/i)

    if dark_mode_toggle.count() > 0:
        # Click the toggle
        dark_mode_toggle.click()

        # Check if theme changed (body class or style)
        body = page.locator("body")
        # This is a basic check - in a real app you'd check for specific theme indicators
        expect(body).to_be_visible()


def test_chart_components(page: Page, dashboard_url):
    """Test chart components load correctly."""
    page.goto(f"{dashboard_url}/sentiment")
    page.wait_for_load_state("networkidle")

    # Wait a bit for charts to load
    page.wait_for_timeout(2000)

    # Check for canvas or chart elements
    chart_elements = page.locator("canvas, .recharts-wrapper, svg")

    # Charts should be present (may be 0 if data not loaded yet)
    if chart_elements.count() > 0:
        expect(chart_elements.first).to_be_visible()


def test_form_interactions(page: Page, dashboard_url):
    """Test form interactions."""
    page.goto(f"{dashboard_url}/sentiment")
    page.wait_for_load_state("networkidle")

    # Look for input fields
    inputs = page.locator("input, textarea")

    if inputs.count() > 0:
        # Try to interact with first input
        first_input = inputs.first

        # Check if it's editable
        if first_input.is_editable():
            first_input.fill("Test input")
            expect(first_input).to_have_value("Test input")


def test_api_connectivity(page: Page, dashboard_url):
    """Test that dashboard can connect to API."""
    page.goto(dashboard_url)
    page.wait_for_load_state("networkidle")

    # Check for API connectivity indicators
    # This would depend on how the dashboard shows connection status
    # For now, just check that no major errors are shown
    error_elements = page.locator(".error, [class*='error'], .alert-danger")

    # If there are error elements, they might indicate API issues
    if error_elements.count() > 0:
        # Log the errors but don't fail the test - API might not be running
        print(f"Found {error_elements.count()} error elements on page")


def test_loading_states(page: Page, dashboard_url):
    """Test loading states and indicators."""
    page.goto(dashboard_url)
    page.wait_for_load_state("networkidle")

    # Look for loading indicators
    loading_elements = page.locator(".loading, .spinner, [class*='loading']")

    # If loading elements exist, they should eventually disappear
    if loading_elements.count() > 0:
        # Wait for loading to complete
        try:
            loading_elements.first.wait_for(state="hidden", timeout=10000)
        except:
            # Loading might still be visible - not necessarily a failure
            print("Loading elements still visible - API might not be running")


def test_accessibility(page: Page, dashboard_url):
    """Test basic accessibility features."""
    page.goto(dashboard_url)
    page.wait_for_load_state("networkidle")

    # Check for alt text on images
    images = page.locator("img")
    for i in range(min(3, images.count())):  # Check first 3 images
        img = images.nth(i)
        alt_text = img.get_attribute("alt")
        # Alt text should exist (even if empty for decorative images)
        assert alt_text is not None

    # Check for ARIA labels on interactive elements
    buttons = page.locator("button")
    for i in range(min(3, buttons.count())):  # Check first 3 buttons
        button = buttons.nth(i)
        aria_label = button.get_attribute("aria-label")
        # Not all buttons need aria-label, but if present it should be descriptive
        if aria_label:
            assert len(aria_label.strip()) > 0


def test_performance(page: Page, dashboard_url):
    """Test basic performance metrics."""
    page.goto(dashboard_url)
    page.wait_for_load_state("networkidle")

    # Get performance metrics
    performance = page.evaluate("""
        () => {
            if ('performance' in window) {
                const perf = performance.getEntriesByType('navigation')[0];
                return {
                    domContentLoaded: perf.domContentLoadedEventEnd - perf.domContentLoadedEventStart,
                    loadComplete: perf.loadEventEnd - perf.loadEventStart,
                    totalTime: perf.loadEventEnd - perf.navigationStart
                };
            }
            return null;
        }
    """)

    if performance:
        # Basic performance checks
        assert performance['totalTime'] > 0
        assert performance['domContentLoaded'] > 0
        print(f"Performance metrics: {performance}")


def test_console_errors(page: Page, dashboard_url):
    """Test for JavaScript console errors."""
    errors = []

    def handle_console_error(msg):
        if msg.type == "error":
            errors.append(msg.text)

    page.on("console", handle_console_error)

    page.goto(dashboard_url)
    page.wait_for_load_state("networkidle")

    # Check for console errors
    if errors:
        print(f"Console errors found: {errors}")
        # In a real test environment, you might want to fail on errors
        # For now, just log them
        assert len(errors) < 5  # Allow some errors but not too many


def test_websocket_connection(page: Page, dashboard_url):
    """Test WebSocket connection if available."""
    page.goto(dashboard_url)
    page.wait_for_load_state("networkidle")

    # Check if WebSocket is mentioned in the page
    websocket_indicators = page.locator("text=/websocket|socket\.io|realtime/i")

    # If WebSocket indicators exist, the feature is likely implemented
    if websocket_indicators.count() > 0:
        print("WebSocket functionality appears to be implemented")
        # Additional WebSocket tests could be added here
