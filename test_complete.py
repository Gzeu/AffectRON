#!/usr/bin/env python3
"""
Comprehensive test suite runner for AffectRON.
Runs all tests including backend, frontend, and integration tests.
"""

import os
import sys
import time
import threading
import subprocess
import requests
from pathlib import Path


def run_command(cmd, cwd=None, check=True, timeout=300):
    """Run a command with timeout."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=check
        )
        return result.stdout, result.stderr, 0
    except subprocess.CalledProcessError as e:
        return e.stdout, e.stderr, e.returncode
    except subprocess.TimeoutExpired:
        return "", "Command timed out", -1


def check_service_health(url, service_name, timeout=10):
    """Check if a service is healthy."""
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            print(f"✅ {service_name}: Running (HTTP {response.status_code})")
            return True
        else:
            print(f"⚠️  {service_name}: Responding ({response.status_code})")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ {service_name}: Not running ({str(e)})")
        return False


def start_api_server():
    """Start the API server in background."""
    print("🚀 Starting API server...")

    # Set environment for testing
    env = os.environ.copy()
    env.update({
        'ENVIRONMENT': 'testing',
        'DEBUG': 'true',
        'MOCK_EXTERNAL_APIS': 'true'
    })

    try:
        process = subprocess.Popen(
            [sys.executable, '-m', 'uvicorn', 'src.api.main:app', '--host', '0.0.0.0', '--port', '8000', '--reload'],
            cwd=Path.cwd(),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Wait for server to start
        for _ in range(30):  # 30 seconds timeout
            if check_service_health("http://localhost:8000/health", "API"):
                print("✅ API server started successfully")
                return process
            time.sleep(1)

        print("❌ API server failed to start")
        process.terminate()
        return None

    except Exception as e:
        print(f"❌ Failed to start API server: {e}")
        return None


def start_dashboard():
    """Start the dashboard in background."""
    print("🖥️  Starting Dashboard...")

    dashboard_dir = Path.cwd() / "src" / "dashboard"

    if not (dashboard_dir / "node_modules").exists():
        print("📦 Installing dashboard dependencies...")
        stdout, stderr, code = run_command("npm install", cwd=dashboard_dir)
        if code != 0:
            print(f"❌ Failed to install dashboard dependencies: {stderr}")
            return None

    try:
        process = subprocess.Popen(
            ["npm", "start"],
            cwd=dashboard_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Wait for dashboard to start
        for _ in range(60):  # 60 seconds timeout
            if check_service_health("http://localhost:3000", "Dashboard"):
                print("✅ Dashboard started successfully")
                return process
            time.sleep(1)

        print("❌ Dashboard failed to start")
        process.terminate()
        return None

    except Exception as e:
        print(f"❌ Failed to start dashboard: {e}")
        return None


def run_backend_tests():
    """Run backend tests."""
    print("🧪 Running backend tests...")

    # Run Python tests
    stdout, stderr, code = run_command("python -m pytest tests/ -m 'not playwright' -v")

    print(stdout)
    if stderr:
        print("STDERR:", stderr)

    if code == 0:
        print("✅ Backend tests passed!")
        return True
    else:
        print(f"❌ Backend tests failed with exit code {code}")
        return False


def run_playwright_tests():
    """Run Playwright tests."""
    print("🎭 Running Playwright tests...")

    # Run Playwright tests
    stdout, stderr, code = run_command("python -m pytest tests/ -m 'playwright' -v")

    print(stdout)
    if stderr:
        print("STDERR:", stderr)

    if code == 0:
        print("✅ Playwright tests passed!")
        return True
    else:
        print(f"❌ Playwright tests failed with exit code {code}")
        return False


def run_integration_tests():
    """Run integration tests."""
    print("🔗 Running integration tests...")

    # Run integration tests with services running
    stdout, stderr, code = run_command("python -m pytest tests/ -m 'integration' -v")

    print(stdout)
    if stderr:
        print("STDERR:", stderr)

    if code == 0:
        print("✅ Integration tests passed!")
        return True
    else:
        print(f"❌ Integration tests failed with exit code {code}")
        return False


def run_all_tests():
    """Run complete test suite."""
    print("🚀 Running complete AffectRON test suite...")
    print("=" * 60)

    success = True

    # 1. Backend tests (don't require services)
    if not run_backend_tests():
        success = False

    # 2. Start services
    api_process = start_api_server()
    dashboard_process = start_dashboard()

    if not api_process or not dashboard_process:
        print("❌ Could not start required services")
        return False

    try:
        # Wait for services to fully initialize
        print("⏳ Waiting for services to initialize...")
        time.sleep(10)

        # 3. Playwright tests (require services)
        if not run_playwright_tests():
            success = False

        # 4. Integration tests (require services)
        if not run_integration_tests():
            success = False

    finally:
        # Clean up processes
        print("🧹 Cleaning up processes...")
        if api_process:
            api_process.terminate()
            api_process.wait()

        if dashboard_process:
            dashboard_process.terminate()
            dashboard_process.wait()

    print("=" * 60)
    if success:
        print("🎉 All tests completed successfully!")
    else:
        print("❌ Some tests failed")

    return success


def main():
    """Main test runner function."""
    if len(sys.argv) < 2:
        print("Usage: python test_complete.py {backend|frontend|integration|all}")
        print("")
        print("Commands:")
        print("  backend     - Run only backend tests")
        print("  playwright  - Run only Playwright tests")
        print("  integration - Run only integration tests")
        print("  all         - Run complete test suite")
        print("  check       - Check service status")
        sys.exit(1)

    command = sys.argv[1]

    if command == "backend":
        success = run_backend_tests()
    elif command == "playwright":
        success = run_playwright_tests()
    elif command == "integration":
        success = run_integration_tests()
    elif command == "all":
        success = run_all_tests()
    elif command == "check":
        check_service_health("http://localhost:8000/health", "API")
        check_service_health("http://localhost:3000", "Dashboard")
        success = True
    else:
        print(f"❌ Unknown command: {command}")
        success = False

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
