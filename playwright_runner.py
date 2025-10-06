#!/usr/bin/env python3
"""
Playwright test runner for AffectRON.
Runs UI and API tests using Playwright.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, cwd=None, check=True):
    """Run a command and return the result."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=check
        )
        return result.stdout, result.stderr, 0
    except subprocess.CalledProcessError as e:
        return e.stdout, e.stderr, e.returncode


def check_playwright_installation():
    """Check if Playwright is properly installed."""
    print("🔍 Checking Playwright installation...")

    try:
        import playwright
        print("✅ Playwright Python package found")

        # Check if browsers are installed
        from playwright.sync_api import Playwright
        print("✅ Playwright browsers available")

        return True
    except ImportError:
        print("❌ Playwright not installed. Run: pip install playwright pytest-playwright")
        return False


def install_playwright_browsers():
    """Install Playwright browsers."""
    print("🌐 Installing Playwright browsers...")

    stdout, stderr, code = run_command("python -m playwright install chromium")

    if code == 0:
        print("✅ Playwright browsers installed")
        return True
    else:
        print(f"❌ Failed to install browsers: {stderr}")
        return False


def run_ui_tests(args):
    """Run UI tests with Playwright."""
    print("🖥️  Running UI tests...")

    cmd_parts = ["python", "-m", "pytest", "tests/test_dashboard_playwright.py"]

    if args.headed:
        cmd_parts.append("--headed")

    if args.slow_mo:
        cmd_parts.extend(["--slowmo", str(args.slow_mo)])

    if args.debug:
        cmd_parts.append("--pdb")

    if args.verbose:
        cmd_parts.append("-v")

    cmd = " ".join(cmd_parts)

    print(f"Executing: {cmd}")
    print("=" * 50)

    stdout, stderr, code = run_command(cmd)

    print(stdout)
    if stderr:
        print("STDERR:", stderr)

    print("=" * 50)

    if code == 0:
        print("✅ All UI tests passed!")
        return True
    else:
        print(f"❌ UI tests failed with exit code {code}")
        return False


def run_api_tests(args):
    """Run API tests with Playwright."""
    print("🔗 Running API tests...")

    cmd_parts = ["python", "-m", "pytest", "tests/test_api_playwright.py"]

    if args.verbose:
        cmd_parts.append("-v")

    if args.debug:
        cmd_parts.append("--pdb")

    cmd = " ".join(cmd_parts)

    print(f"Executing: {cmd}")
    print("=" * 50)

    stdout, stderr, code = run_command(cmd)

    print(stdout)
    if stderr:
        print("STDERR:", stderr)

    print("=" * 50)

    if code == 0:
        print("✅ All API tests passed!")
        return True
    else:
        print(f"❌ API tests failed with exit code {code}")
        return False


def run_all_playwright_tests(args):
    """Run all Playwright tests."""
    print("🚀 Running all Playwright tests...")

    success = True

    # Run API tests first (they don't require the dashboard to be running)
    if not run_api_tests(args):
        success = False

    # Run UI tests (require dashboard to be running)
    if not run_ui_tests(args):
        success = False

    return success


def check_services_status():
    """Check if required services are running."""
    print("🔍 Checking service status...")

    services = [
        ("API", "http://localhost:8000/health"),
        ("Dashboard", "http://localhost:3000")
    ]

    for service_name, url in services:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"✅ {service_name}: Running")
            else:
                print(f"⚠️  {service_name}: Responding ({response.status_code})")
        except requests.exceptions.RequestException:
            print(f"❌ {service_name}: Not running")
            if service_name == "Dashboard":
                print("   💡 Start dashboard: cd src/dashboard && npm start")
            elif service_name == "API":
                print("   💡 Start API: uvicorn src.api.main:app --reload")


def main():
    """Main Playwright test runner."""
    parser = argparse.ArgumentParser(description="AffectRON Playwright Test Runner")

    # Test execution options
    parser.add_argument("--ui-only", action="store_true", help="Run only UI tests")
    parser.add_argument("--api-only", action="store_true", help="Run only API tests")
    parser.add_argument("--headed", action="store_true", help="Run browser in headed mode")
    parser.add_argument("--slow-mo", type=int, help="Slow down actions by ms")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    # Utility options
    parser.add_argument("--check-services", action="store_true", help="Check service status")
    parser.add_argument("--install-browsers", action="store_true", help="Install Playwright browsers")

    args = parser.parse_args()

    # Check Playwright installation
    if not check_playwright_installation():
        print("❌ Playwright not properly installed")
        return 1

    # Install browsers if requested
    if args.install_browsers:
        if not install_playwright_browsers():
            return 1

    # Check services if requested
    if args.check_services:
        check_services_status()
        return 0

    # Run tests based on arguments
    if args.ui_only:
        success = run_ui_tests(args)
    elif args.api_only:
        success = run_api_tests(args)
    else:
        success = run_all_playwright_tests(args)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
