"""Authentication middleware for the Compileo API.

Note: API Key authentication has been removed in favor of JWT-based
multi-tenant authentication. JWT validation is handled at the route
level using FastAPI dependencies (get_current_active_user).

This middleware now only handles basic request logging and
allows all requests to pass through to the route handlers.
"""

from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from ...core.logging import get_logger

logger = get_logger(__name__)


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """JWT-based authentication middleware (legacy API key support removed)."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip authentication for health check and public endpoints
        # JWT authentication is handled at the route level via FastAPI dependencies
        public_paths = [
            "/health",
            "/",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/auth/login",
            "/auth/register",
        ]
        
        if request.url.path in public_paths:
            return await call_next(request)

        # All other endpoints rely on JWT authentication via route dependencies
        # The get_current_active_user dependency handles JWT validation
        return await call_next(request)
