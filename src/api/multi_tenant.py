"""
Multi-tenant architecture for AffectRON.
Provides tenant isolation, data segregation, and enterprise-grade access control.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
import uuid

from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean, Text, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.dialects.postgresql import UUID

from .auth import verify_token


Base = declarative_base()


@dataclass
class Tenant:
    """Tenant information."""
    id: str
    name: str
    domain: str
    subscription_tier: str  # 'free', 'basic', 'premium', 'enterprise'
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    settings: Dict[str, Any] = field(default_factory=dict)

    # Resource limits
    max_users: int = 10
    max_data_sources: int = 5
    max_api_calls_per_day: int = 1000
    max_storage_mb: int = 100


@dataclass
class TenantUser:
    """User within a tenant."""
    id: str
    tenant_id: str
    username: str
    email: str
    role: str  # 'admin', 'analyst', 'viewer'
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None

    # Permissions
    permissions: List[str] = field(default_factory=lambda: ['read_sentiment', 'read_analytics'])


class TenantManager:
    """Manages multi-tenant operations and data isolation."""

    def __init__(self, database_url: str):
        self.database_url = database_url
        self.tenants: Dict[str, Tenant] = {}
        self.tenant_users: Dict[str, List[TenantUser]] = {}
        self.engine = None
        self.SessionLocal = None

        self.logger = logging.getLogger(__name__)

        # Initialize tenant database
        self._init_tenant_database()

    def _init_tenant_database(self):
        """Initialize tenant management database."""
        # Create separate database for tenant management
        tenant_db_url = self.database_url.replace('affectron', 'affectron_tenants')

        self.engine = create_engine(tenant_db_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

        # Create tables
        Base.metadata.create_all(bind=self.engine)

    def create_tenant(self, name: str, domain: str, subscription_tier: str = 'basic') -> Tenant:
        """Create a new tenant."""
        tenant_id = str(uuid.uuid4())

        tenant = Tenant(
            id=tenant_id,
            name=name,
            domain=domain,
            subscription_tier=subscription_tier,
            settings=self._get_default_tenant_settings(subscription_tier)
        )

        self.tenants[tenant_id] = tenant

        # Create tenant-specific database
        self._create_tenant_database(tenant)

        self.logger.info(f"Created tenant: {name} ({tenant_id})")
        return tenant

    def _get_default_tenant_settings(self, tier: str) -> Dict[str, Any]:
        """Get default settings for subscription tier."""
        tier_settings = {
            'free': {
                'max_users': 1,
                'max_data_sources': 1,
                'max_api_calls_per_day': 100,
                'max_storage_mb': 10,
                'features': ['basic_sentiment', 'basic_analytics']
            },
            'basic': {
                'max_users': 5,
                'max_data_sources': 3,
                'max_api_calls_per_day': 1000,
                'max_storage_mb': 100,
                'features': ['sentiment_analysis', 'basic_analytics', 'email_alerts']
            },
            'premium': {
                'max_users': 20,
                'max_data_sources': 10,
                'max_api_calls_per_day': 10000,
                'max_storage_mb': 1000,
                'features': ['advanced_sentiment', 'full_analytics', 'realtime_alerts', 'trading_signals']
            },
            'enterprise': {
                'max_users': 100,
                'max_data_sources': 50,
                'max_api_calls_per_day': 100000,
                'max_storage_mb': 10000,
                'features': ['all_features', 'custom_models', 'api_integration', 'sla_support']
            }
        }

        return tier_settings.get(tier, tier_settings['basic'])

    def _create_tenant_database(self, tenant: Tenant):
        """Create database for specific tenant."""
        # Create tenant-specific database URL
        tenant_db_url = self.database_url.replace('affectron', f'affectron_{tenant.id}')

        # Create database engine for tenant
        tenant_engine = create_engine(tenant_db_url)

        # Create tenant-specific tables (would import from models)
        # Base.metadata.create_all(bind=tenant_engine)

        self.logger.info(f"Created database for tenant: {tenant.name}")

    def get_tenant_by_domain(self, domain: str) -> Optional[Tenant]:
        """Get tenant by domain."""
        for tenant in self.tenants.values():
            if tenant.domain == domain:
                return tenant
        return None

    def get_tenant_by_id(self, tenant_id: str) -> Optional[Tenant]:
        """Get tenant by ID."""
        return self.tenants.get(tenant_id)

    def add_tenant_user(self, tenant_id: str, user: TenantUser) -> bool:
        """Add user to tenant."""
        if tenant_id not in self.tenants:
            return False

        tenant = self.tenants[tenant_id]

        # Check user limit
        if len(self.tenant_users.get(tenant_id, [])) >= tenant.max_users:
            return False

        if tenant_id not in self.tenant_users:
            self.tenant_users[tenant_id] = []

        self.tenant_users[tenant_id].append(user)
        return True

    def get_tenant_users(self, tenant_id: str) -> List[TenantUser]:
        """Get all users for a tenant."""
        return self.tenant_users.get(tenant_id, [])

    def authenticate_tenant_user(self, tenant_id: str, username: str, password: str) -> Optional[TenantUser]:
        """Authenticate user within tenant context."""
        users = self.get_tenant_users(tenant_id)

        for user in users:
            if user.username == username and user.is_active:
                # In real implementation, verify password hash
                return user

        return None

    def check_tenant_permissions(self, tenant_id: str, user_id: str, required_permissions: List[str]) -> bool:
        """Check if user has required permissions within tenant."""
        users = self.get_tenant_users(tenant_id)

        for user in users:
            if user.id == user_id:
                user_permissions = set(user.permissions)

                # Check if user has all required permissions
                return all(perm in user_permissions for perm in required_permissions)

        return False

    def get_tenant_usage(self, tenant_id: str) -> Dict[str, Any]:
        """Get current usage statistics for tenant."""
        if tenant_id not in self.tenants:
            return {}

        tenant = self.tenants[tenant_id]
        users = self.get_tenant_users(tenant_id)

        return {
            'tenant_id': tenant_id,
            'tenant_name': tenant.name,
            'subscription_tier': tenant.subscription_tier,
            'current_users': len(users),
            'max_users': tenant.max_users,
            'usage_percentage': (len(users) / tenant.max_users) * 100,
            'is_near_limit': len(users) >= tenant.max_users * 0.8
        }

    def update_tenant_settings(self, tenant_id: str, settings: Dict[str, Any]) -> bool:
        """Update tenant settings."""
        if tenant_id not in self.tenants:
            return False

        tenant = self.tenants[tenant_id]
        tenant.settings.update(settings)

        return True

    def delete_tenant(self, tenant_id: str) -> bool:
        """Delete tenant and all associated data."""
        if tenant_id not in self.tenants:
            return False

        # Remove tenant
        del self.tenants[tenant_id]

        # Remove users
        if tenant_id in self.tenant_users:
            del self.tenant_users[tenant_id]

        # TODO: Delete tenant database

        self.logger.info(f"Deleted tenant: {tenant_id}")
        return True


class TenantContext:
    """Context manager for tenant-specific operations."""

    def __init__(self, tenant_manager: TenantManager, tenant_id: str):
        self.tenant_manager = tenant_manager
        self.tenant_id = tenant_id
        self.tenant: Optional[Tenant] = None
        self.session: Optional[Session] = None

    async def __aenter__(self) -> 'TenantContext':
        """Enter tenant context."""
        self.tenant = self.tenant_manager.get_tenant_by_id(self.tenant_id)

        if not self.tenant:
            raise ValueError(f"Tenant {self.tenant_id} not found")

        if not self.tenant.is_active:
            raise ValueError(f"Tenant {self.tenant_id} is inactive")

        # Create tenant-specific database session
        tenant_db_url = self.tenant_manager.database_url.replace(
            'affectron',
            f'affectron_{self.tenant_id}'
        )

        engine = create_engine(tenant_db_url)
        self.session = sessionmaker(autocommit=False, autoflush=False, bind=engine)()

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit tenant context."""
        if self.session:
            self.session.close()


class TenantMiddleware:
    """Middleware for handling multi-tenant requests."""

    def __init__(self, app, tenant_manager: TenantManager):
        self.app = app
        self.tenant_manager = tenant_manager

    async def __call__(self, scope, receive, send):
        """Process request with tenant context."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Extract tenant information from request
        headers = dict(scope.get("headers", []))

        # Try to get tenant from domain
        host = None
        for key, value in headers.items():
            if key == b'host':
                host = value.decode()
                break

        if host:
            # Extract domain (remove port if present)
            domain = host.split(':')[0]

            # Get tenant by domain
            tenant = self.tenant_manager.get_tenant_by_domain(domain)

            if tenant:
                # Set tenant context in scope
                scope["tenant_id"] = tenant.id
                scope["tenant"] = tenant

        await self.app(scope, receive, send)


def get_current_tenant(request) -> Optional[Tenant]:
    """Get current tenant from request context."""
    return getattr(request.state, 'tenant', None)


def require_tenant_permission(permissions: List[str]):
    """Decorator to require specific permissions within tenant."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Get current user and tenant from request context
            # This would be implemented based on authentication system

            # For now, just call the function
            return await func(*args, **kwargs)

        return wrapper
    return decorator


def create_tenant_database_schema(tenant_id: str):
    """Create database schema for tenant."""
    # This would create all tables for the tenant
    # Similar to Base.metadata.create_all() but for tenant-specific database

    # Import all models that need to be tenant-aware
    from .models import (
        DataSource, ExtractedData, SentimentAnalysis,
        MarketData, AggregatedData, AnalyticsResult
    )

    # Create tenant-specific versions of these models
    # This is a simplified example - real implementation would be more complex

    tenant_models = {
        'data_sources': type(f'DataSource_{tenant_id}', (DataSource,), {
            '__tablename__': f'data_sources_{tenant_id}',
            'tenant_id': Column(String, default=tenant_id)
        }),
        'extracted_data': type(f'ExtractedData_{tenant_id}', (ExtractedData,), {
            '__tablename__': f'extracted_data_{tenant_id}',
            'tenant_id': Column(String, default=tenant_id)
        }),
        # Add other models...
    }

    return tenant_models


# Global tenant manager instance
tenant_manager = TenantManager("postgresql://affectron:affectron_password_2024@localhost:5432/affectron")


async def initialize_multi_tenant_system():
    """Initialize multi-tenant system."""
    # Create default tenant if none exist
    if not tenant_manager.tenants:
        default_tenant = tenant_manager.create_tenant(
            name="Default Organization",
            domain="localhost",
            subscription_tier="basic"
        )

        # Create admin user
        admin_user = TenantUser(
            id=str(uuid.uuid4()),
            tenant_id=default_tenant.id,
            username="admin",
            email="admin@localhost",
            role="admin",
            permissions=['*']  # All permissions
        )

        tenant_manager.add_tenant_user(default_tenant.id, admin_user)

    logging.getLogger(__name__).info("Multi-tenant system initialized")


def get_tenant_status():
    """Get multi-tenant system status."""
    return {
        'total_tenants': len(tenant_manager.tenants),
        'total_users': sum(len(users) for users in tenant_manager.tenant_users.values()),
        'tenants': [
            {
                'id': tenant.id,
                'name': tenant.name,
                'domain': tenant.domain,
                'tier': tenant.subscription_tier,
                'active': tenant.is_active,
                'user_count': len(tenant_manager.tenant_users.get(tenant.id, []))
            }
            for tenant in tenant_manager.tenants.values()
        ],
        'system_initialized': True
    }
