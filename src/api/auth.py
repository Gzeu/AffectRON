"""
Authentication module for AffectRON API.
Handles JWT token verification and user authentication.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

import jwt
from fastapi import HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from passlib.context import CryptContext

from .config import get_settings


logger = logging.getLogger(__name__)
settings = get_settings()

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security scheme for API key authentication
security = HTTPBearer()


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """
    Create JWT access token.

    Args:
        data: Data to encode in token
        expires_delta: Token expiration time

    Returns:
        Encoded JWT token
    """
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.access_token_expire_minutes)

    to_encode.update({"exp": expire, "type": "access"})

    encoded_jwt = jwt.encode(
        to_encode,
        settings.secret_key,
        algorithm=settings.algorithm
    )

    return encoded_jwt


def verify_token(credentials: HTTPAuthorizationCredentials) -> dict:
    """
    Verify JWT token from authorization header.

    Args:
        credentials: HTTP authorization credentials

    Returns:
        Decoded token data

    Raises:
        HTTPException: If token is invalid or expired
    """
    try:
        token = credentials.credentials

        payload = jwt.decode(
            token,
            settings.secret_key,
            algorithms=[settings.algorithm]
        )

        # Check if token type is access
        if payload.get("type") != "access":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type"
            )

        return payload

    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired"
        )
    except jwt.JWTError as e:
        logger.warning(f"JWT verification failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token"
        )


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Dependency to get current authenticated user.

    Args:
        credentials: HTTP authorization credentials

    Returns:
        User information from token

    Raises:
        HTTPException: If authentication fails
    """
    token_data = verify_token(credentials)

    # For now, we'll use the token data as user info
    # In a real application, you'd validate against a user database
    return {
        "user_id": token_data.get("user_id"),
        "username": token_data.get("username"),
        "scopes": token_data.get("scopes", []),
        "token_data": token_data
    }


def require_scopes(*required_scopes: str):
    """
    Dependency factory for requiring specific scopes.

    Args:
        *required_scopes: Required permission scopes

    Returns:
        Dependency function
    """
    def scope_checker(current_user: dict = Depends(get_current_user)):
        user_scopes = set(current_user.get("scopes", []))

        if not required_scopes:
            return current_user

        if not any(scope in user_scopes for scope in required_scopes):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required: {', '.join(required_scopes)}"
            )

        return current_user

    return scope_checker


# Predefined scope constants
READ_SENTIMENT = "read:sentiment"
WRITE_SENTIMENT = "write:sentiment"
READ_MARKET = "read:market"
READ_ANALYTICS = "read:analytics"
ADMIN = "admin"
