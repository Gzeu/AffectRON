"""
Dashboard package for AffectRON.
Contains React/TypeScript web interface for financial sentiment analysis.
"""

import os
import subprocess
from pathlib import Path


def build_dashboard():
    """Build the React dashboard for production."""
    dashboard_dir = Path(__file__).parent

    try:
        # Install dependencies
        subprocess.run(['npm', 'install'], cwd=dashboard_dir, check=True)

        # Build for production
        subprocess.run(['npm', 'run', 'build'], cwd=dashboard_dir, check=True)

        print("Dashboard built successfully")
        return True

    except subprocess.CalledProcessError as e:
        print(f"Error building dashboard: {e}")
        return False


def start_dashboard_dev():
    """Start the dashboard in development mode."""
    dashboard_dir = Path(__file__).parent

    try:
        # Install dependencies if needed
        if not (dashboard_dir / 'node_modules').exists():
            subprocess.run(['npm', 'install'], cwd=dashboard_dir, check=True)

        # Start development server
        subprocess.run(['npm', 'start'], cwd=dashboard_dir)

    except subprocess.CalledProcessError as e:
        print(f"Error starting dashboard development server: {e}")
        return False


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "build":
        build_dashboard()
    elif len(sys.argv) > 1 and sys.argv[1] == "dev":
        start_dashboard_dev()
    else:
        print("Usage: python -m dashboard [build|dev]")