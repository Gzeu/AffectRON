"""
Infrastructure deployment package for AffectRON.
Contains Docker, Kubernetes, and deployment configurations.
"""

import os
import subprocess
from pathlib import Path


def check_deployment_requirements():
    """Check if all deployment requirements are met."""
    requirements = {
        'docker': 'docker --version',
        'docker-compose': 'docker-compose --version',
        'git': 'git --version'
    }

    missing = []

    for name, command in requirements.items():
        try:
            subprocess.run(command.split(), capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            missing.append(name)

    if missing:
        print(f"Missing requirements: {', '.join(missing)}")
        return False

    print("All deployment requirements are met!")
    return True


def validate_environment():
    """Validate environment configuration."""
    required_env_vars = [
        'DATABASE_URL',
        'SECRET_KEY',
        'REDIS_URL'
    ]

    missing = []

    for var in required_env_vars:
        if not os.getenv(var):
            missing.append(var)

    if missing:
        print(f"Missing environment variables: {', '.join(missing)}")
        print("Please check your .env file")
        return False

    print("Environment configuration is valid!")
    return True


def deploy_to_environment(environment='development'):
    """Deploy AffectRON to specified environment."""
    print(f"Deploying AffectRON to {environment} environment...")

    # Check requirements
    if not check_deployment_requirements():
        return False

    # Validate environment
    if not validate_environment():
        return False

    # Set environment
    os.environ['ENVIRONMENT'] = environment

    try:
        # Build and deploy
        if environment == 'production':
            subprocess.run(['docker-compose', '--profile', 'production', 'up', '-d'], check=True)
        else:
            subprocess.run(['docker-compose', 'up', '-d'], check=True)

        print(f"Successfully deployed to {environment}!")
        return True

    except subprocess.CalledProcessError as e:
        print(f"Deployment failed: {e}")
        return False


def run_health_checks():
    """Run health checks on deployed services."""
    print("Running health checks...")

    health_checks = {
        'api': 'curl -f http://localhost:8000/health',
        'dashboard': 'curl -f http://localhost:3000',
        'postgres': 'docker-compose exec postgres pg_isready -U affectron',
        'redis': 'docker-compose exec redis redis-cli ping'
    }

    results = {}

    for service, command in health_checks.items():
        try:
            subprocess.run(command, shell=True, check=True, capture_output=True)
            results[service] = True
            print(f"✅ {service}: Healthy")
        except subprocess.CalledProcessError:
            results[service] = False
            print(f"❌ {service}: Unhealthy")

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "check":
            check_deployment_requirements()
        elif command == "validate":
            validate_environment()
        elif command == "deploy":
            env = sys.argv[2] if len(sys.argv) > 2 else 'development'
            deploy_to_environment(env)
        elif command == "health":
            run_health_checks()
        else:
            print("Usage: python -m infrastructure {check|validate|deploy|health}")
    else:
        print("AffectRON Infrastructure Module")
        print("Usage: python -m infrastructure {check|validate|deploy|health}")