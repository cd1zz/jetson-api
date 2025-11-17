"""FastAPI dependencies for authentication and validation."""

from typing import Optional
from fastapi import Header, HTTPException, status

from .config import settings


async def verify_api_key(authorization: Optional[str] = Header(default=None)) -> None:
    """
    Verify the API key from the Authorization header.

    Args:
        authorization: Authorization header value (Bearer token)

    Raises:
        HTTPException: If the token is missing or invalid
    """
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Authorization header format. Expected: Bearer <token>",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = authorization.removeprefix("Bearer ").strip()

    if token != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key",
        )
