#!/usr/bin/env python3
"""
Environment configuration script for AffectRON.
Helps set up the development and production environments.
"""

import os
import sys
import json
import secrets
import string
from pathlib import Path
from typing import Dict, Any


def generate_secret_key(length: int = 32) -> str:
    """Generate a secure secret key."""
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*(-_=+)"
    return ''.join(secrets.choice(alphabet) for _ in range(length))


def create_env_file(template_path: str, output_path: str, overrides: Dict[str, Any] = None):
    """Create .env file from template with overrides."""
    if not Path(template_path).exists():
        print(f"❌ Template file not found: {template_path}")
        return False

    with open(template_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Apply overrides
    if overrides:
        for key, value in overrides.items():
            # Replace or add environment variable
            pattern = f"{key}=.*"
            replacement = f"{key}={value}"
            if key in content:
                content = re.sub(pattern, replacement, content)
            else:
                content += f"\n{replacement}"

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"✅ Created {output_path}")
    return True


def setup_development_environment():
    """Set up development environment."""
    print("🚀 Setting up development environment...")

    overrides = {
        'ENVIRONMENT': 'development',
        'DEBUG': 'true',
        'SECRET_KEY': generate_secret_key(32),
        'DATABASE_URL': 'postgresql://affectron:affectron_password_2024@localhost:5432/affectron',
        'REDIS_URL': 'redis://localhost:6379/0',
        'API_PORT': '8000',
        'DASHBOARD_PORT': '3000',
        'MOCK_EXTERNAL_APIS': 'false'
    }

    success = create_env_file('.env.example', '.env', overrides)

    if success:
        print("✅ Development environment configured")
        print("📝 Edit .env file to customize your configuration")
        print("🚀 Run 'python test_runner.py' to start testing")

    return success


def setup_production_environment():
    """Set up production environment."""
    print("🏭 Setting up production environment...")

    overrides = {
        'ENVIRONMENT': 'production',
        'DEBUG': 'false',
        'SECRET_KEY': generate_secret_key(32),
        'API_PORT': '8000',
        'DASHBOARD_PORT': '3000',
        'NGINX_PORT': '80',
        'NGINX_SSL_PORT': '443',
        'POSTGRES_PASSWORD': generate_secret_key(16),
        'GRAFANA_PASSWORD': generate_secret_key(12),
        'MOCK_EXTERNAL_APIS': 'false'
    }

    success = create_env_file('.env.example', '.env', overrides)

    if success:
        print("✅ Production environment configured")
        print("🔐 Make sure to:")
        print("   1. Set up SSL certificates")
        print("   2. Configure external API keys")
        print("   3. Set up monitoring alerts")
        print("   4. Configure domain names")

    return success


def setup_testing_environment():
    """Set up testing environment."""
    print("🧪 Setting up testing environment...")

    overrides = {
        'ENVIRONMENT': 'testing',
        'DEBUG': 'true',
        'SECRET_KEY': 'test-secret-key-for-testing-only-32-chars-min',
        'TEST_DATABASE_URL': 'sqlite:///./test_affectron.db',
        'DATABASE_URL': 'sqlite:///./test_affectron.db',
        'REDIS_URL': 'redis://localhost:6379/15',
        'MOCK_EXTERNAL_APIS': 'true'
    }

    success = create_env_file('.env.example', '.env.test', overrides)

    if success:
        print("✅ Testing environment configured")
        print("🧪 Ready for running tests")

    return success


def validate_configuration(env_file: str = '.env'):
    """Validate environment configuration."""
    print(f"🔍 Validating configuration in {env_file}...")

    if not Path(env_file).exists():
        print(f"❌ Configuration file not found: {env_file}")
        return False

    required_vars = [
        'DATABASE_URL',
        'SECRET_KEY',
        'REDIS_URL'
    ]

    missing = []
    with open(env_file, 'r', encoding='utf-8') as f:
        content = f.read()
        for var in required_vars:
            if f"{var}=" not in content:
                missing.append(var)

    if missing:
        print(f"❌ Missing required environment variables: {', '.join(missing)}")
        return False

    print("✅ Configuration validation passed")
    return True


def show_environment_info():
    """Show current environment information."""
    print("📋 Current Environment Information:")
    print(f"   OS: {os.name}")
    print(f"   Python: {sys.version}")
    print(f"   Working Directory: {os.getcwd()}")

    env_files = ['.env', '.env.example', '.env.test']
    for env_file in env_files:
        if Path(env_file).exists():
            size = Path(env_file).stat().st_size
            print(f"   {env_file}: {size} bytes")
        else:
            print(f"   {env_file}: Not found")


def main():
    """Main configuration function."""
    print("⚙️  AffectRON Environment Configuration")
    print("=" * 50)

    if len(sys.argv) < 2:
        print("Usage: python configure.py {dev|prod|test|validate|info}")
        print("")
        print("Commands:")
        print("  dev      - Set up development environment")
        print("  prod     - Set up production environment")
        print("  test     - Set up testing environment")
        print("  validate - Validate current configuration")
        print("  info     - Show environment information")
        sys.exit(1)

    command = sys.argv[1]

    if command == "dev":
        success = setup_development_environment()
    elif command == "prod":
        success = setup_production_environment()
    elif command == "test":
        success = setup_testing_environment()
    elif command == "validate":
        success = validate_configuration()
    elif command == "info":
        show_environment_info()
        success = True
    else:
        print(f"❌ Unknown command: {command}")
        success = False

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
