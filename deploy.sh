#!/bin/bash

# AffectRON Deployment Script
# Handles deployment to different environments

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="affectron"
DOCKER_REGISTRY="${DOCKER_REGISTRY:-your-registry.com}"
ENVIRONMENT="${ENVIRONMENT:-development}"
COMPOSE_FILE="docker-compose.yml"

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_requirements() {
    log_info "Checking requirements..."

    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is required but not installed."
        exit 1
    fi

    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose is required but not installed."
        exit 1
    fi

    # Check .env file
    if [ ! -f .env ]; then
        log_warning ".env file not found. Copying from .env.example..."
        cp .env.example .env
        log_warning "Please edit .env file with your configuration before deploying!"
    fi

    log_success "Requirements check passed"
}

build_images() {
    log_info "Building Docker images..."

    # Build API image
    docker build -t ${PROJECT_NAME}/api:latest -f Dockerfile .

    # Build dashboard image
    if [ -d "src/dashboard" ]; then
        cd src/dashboard
        docker build -t ${PROJECT_NAME}/dashboard:latest -f Dockerfile .
        cd ../..
    fi

    log_success "Images built successfully"
}

deploy_local() {
    log_info "Deploying locally with Docker Compose..."

    # Stop existing containers
    docker-compose down || true

    # Start services
    if [ "$1" = "production" ]; then
        docker-compose --profile production up -d
    else
        docker-compose up -d
    fi

    log_success "Local deployment completed"
    log_info "API available at: http://localhost:8000"
    log_info "Dashboard available at: http://localhost:3000"
    log_info "API Documentation at: http://localhost:8000/docs"
}

deploy_kubernetes() {
    log_info "Deploying to Kubernetes..."

    # Apply configurations
    kubectl apply -f infrastructure/k8s-namespace.yaml || kubectl create namespace affectron
    kubectl apply -f infrastructure/k8s-secrets.yaml
    kubectl apply -f infrastructure/k8s-postgres.yaml
    kubectl apply -f infrastructure/k8s-redis.yaml
    kubectl apply -f infrastructure/k8s-api-deployment.yaml
    kubectl apply -f infrastructure/k8s-dashboard-deployment.yaml
    kubectl apply -f infrastructure/k8s-nginx-deployment.yaml
    kubectl apply -f infrastructure/k8s-ingress.yaml

    log_success "Kubernetes deployment completed"
}

run_tests() {
    log_info "Running tests..."

    # Run API tests
    docker-compose run --rm api python -m pytest tests/ -v

    # Run dashboard tests
    if [ -d "src/dashboard" ]; then
        cd src/dashboard
        npm test -- --watchAll=false --coverage
        cd ../..
    fi

    log_success "Tests completed"
}

cleanup() {
    log_info "Cleaning up..."

    # Remove unused images
    docker image prune -f

    # Remove unused volumes
    docker volume prune -f

    log_success "Cleanup completed"
}

show_status() {
    log_info "Current deployment status:"

    if command -v docker-compose &> /dev/null; then
        docker-compose ps
    elif docker compose version &> /dev/null; then
        docker compose ps
    fi

    # Show logs for main services
    log_info "Recent API logs:"
    docker-compose logs --tail=10 api || true

    log_info "Recent dashboard logs:"
    docker-compose logs --tail=10 dashboard || true
}

# Main deployment logic
main() {
    echo "🚀 AffectRON Deployment Script"
    echo "Environment: $ENVIRONMENT"

    case "${1:-local}" in
        "local")
            check_requirements
            build_images
            deploy_local
            ;;
        "production")
            check_requirements
            build_images
            deploy_local production
            ;;
        "k8s")
            check_requirements
            build_images
            deploy_kubernetes
            ;;
        "test")
            run_tests
            ;;
        "cleanup")
            cleanup
            ;;
        "status")
            show_status
            ;;
        *)
            echo "Usage: $0 {local|production|k8s|test|cleanup|status}"
            echo ""
            echo "Commands:"
            echo "  local      - Deploy locally with Docker Compose (default)"
            echo "  production - Deploy locally with production profile"
            echo "  k8s        - Deploy to Kubernetes cluster"
            echo "  test       - Run test suite"
            echo "  cleanup    - Clean up Docker resources"
            echo "  status     - Show deployment status"
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
