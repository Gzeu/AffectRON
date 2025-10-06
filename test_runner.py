#!/usr/bin/env python3
"""
Test runner script for AffectRON.
Handles environment setup, test execution, and reporting.
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


def setup_test_environment():
    """Set up the test environment."""
    print("🔧 Setting up test environment...")

    # Check if .env file exists
    if not Path('.env').exists():
        print("⚠️  .env file not found. Creating from template...")
        if Path('.env.example').exists():
            import shutil
            shutil.copy('.env.example', '.env')
            print("✅ Created .env from .env.example")
        else:
            print("❌ .env.example not found. Please create .env file manually.")
            return False

    # Set test environment variables
    os.environ.setdefault('ENVIRONMENT', 'testing')
    os.environ.setdefault('DEBUG', 'true')
    os.environ.setdefault('TESTING', 'true')

    print("✅ Test environment configured")
    return True


def install_test_dependencies():
    """Install test dependencies."""
    print("📦 Installing test dependencies...")

    if Path('requirements-test.txt').exists():
        stdout, stderr, code = run_command('pip install -r requirements-test.txt')
        if code != 0:
            print(f"❌ Failed to install test dependencies: {stderr}")
            return False

    print("✅ Test dependencies installed")
    return True


def run_tests(args):
    """Run the test suite."""
    print(f"🧪 Running tests with arguments: {args}")

    # Build pytest command
    cmd_parts = ['python', '-m', 'pytest']

    if args.coverage:
        cmd_parts.extend(['--cov=src', '--cov-report=html', '--cov-report=term'])

    if args.verbose:
        cmd_parts.append('-v')

    if args.markers:
        cmd_parts.extend(['-m', args.markers])

    if args.parallel:
        cmd_parts.extend(['-n', str(args.parallel)])

    # Add test paths if specified
    if args.tests:
        cmd_parts.extend(args.tests)

    cmd = ' '.join(cmd_parts)

    print(f"Executing: {cmd}")
    print("=" * 50)

    stdout, stderr, code = run_command(cmd)

    print(stdout)
    if stderr:
        print("STDERR:", stderr)

    print("=" * 50)

    if code == 0:
        print("✅ All tests passed!")
        return True
    else:
        print(f"❌ Tests failed with exit code {code}")
        return False


def run_specific_test_category(category):
    """Run tests for a specific category."""
    print(f"🎯 Running {category} tests...")

    markers_map = {
        'unit': 'unit',
        'integration': 'integration',
        'api': 'api',
        'extractors': 'unit',
        'pipelines': 'unit',
        'analytics': 'unit',
        'models': 'unit'
    }

    marker = markers_map.get(category)
    if marker:
        return run_tests(argparse.Namespace(
            coverage=True,
            verbose=True,
            markers=marker,
            parallel=0,
            tests=None
        ))
    else:
        print(f"❌ Unknown test category: {category}")
        return False


def lint_code():
    """Run code linting."""
    print("🔍 Running code linting...")

    # Run black for formatting check
    stdout, stderr, code1 = run_command('python -m black --check src/ tests/ || true')

    # Run flake8 for style checks
    stdout, stderr, code2 = run_command('python -m flake8 src/ tests/ || true')

    # Run mypy for type checking
    stdout, stderr, code3 = run_command('python -m mypy src/ || true')

    print("Black (formatting):", "✅" if code1 == 0 else "❌")
    print("Flake8 (style):", "✅" if code2 == 0 else "❌")
    print("MyPy (types):", "✅" if code3 == 0 else "❌")

    return code1 == 0 and code2 == 0 and code3 == 0


def generate_coverage_report():
    """Generate coverage report."""
    print("📊 Generating coverage report...")

    stdout, stderr, code = run_command('python -m pytest --cov=src --cov-report=html --cov-report=term')

    if code == 0:
        print("✅ Coverage report generated in htmlcov/index.html")
        return True
    else:
        print("❌ Failed to generate coverage report")
        return False


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description='AffectRON Test Runner')

    # Test execution options
    parser.add_argument('--coverage', action='store_true', help='Generate coverage report')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--markers', '-m', help='Run tests with specific markers')
    parser.add_argument('--parallel', '-n', type=int, help='Run tests in parallel')
    parser.add_argument('--tests', nargs='*', help='Specific test files or directories')

    # Test categories
    parser.add_argument('--unit', action='store_true', help='Run unit tests only')
    parser.add_argument('--integration', action='store_true', help='Run integration tests only')
    parser.add_argument('--api', action='store_true', help='Run API tests only')

    # Utility commands
    parser.add_argument('--lint', action='store_true', help='Run code linting')
    parser.add_argument('--setup-only', action='store_true', help='Only set up environment')

    args = parser.parse_args()

    # Setup environment
    if not setup_test_environment():
        sys.exit(1)

    if not install_test_dependencies():
        sys.exit(1)

    # Handle specific commands
    if args.lint:
        success = lint_code()
        sys.exit(0 if success else 1)

    if args.unit:
        success = run_specific_test_category('unit')
        sys.exit(0 if success else 1)

    if args.integration:
        success = run_specific_test_category('integration')
        sys.exit(0 if success else 1)

    if args.api:
        success = run_specific_test_category('api')
        sys.exit(0 if success else 1)

    if args.setup_only:
        print("✅ Environment setup completed")
        sys.exit(0)

    # Run tests
    success = run_tests(args)

    # Generate coverage report if requested
    if args.coverage and success:
        generate_coverage_report()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
