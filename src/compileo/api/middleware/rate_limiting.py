"""Rate limiting middleware for the Compileo API."""

import time
from collections import defaultdict
from typing import Callable

from fastapi import HTTPException, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from ..core.config import settings


class RateLimitingMiddleware(BaseHTTPMiddleware):
    """Simple in-memory rate limiting middleware."""

    def __init__(self, app: Callable, requests_per_window: int = None, window_seconds: int = None):
        super().__init__(app)
        self.requests_per_window = requests_per_window or settings.rate_limit_requests
        self.window_seconds = window_seconds or settings.rate_limit_window
        self.requests = defaultdict(list)  # client_ip -> list of timestamps

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Get client identifier (IP address for now)
        client_ip = request.client.host if request.client else "unknown"

        # Clean old requests
        current_time = time.time()
        self.requests[client_ip] = [
            timestamp for timestamp in self.requests[client_ip]
            if current_time - timestamp < self.window_seconds
        ]

        # Check rate limit
        if len(self.requests[client_ip]) >= self.requests_per_window:
            raise HTTPException(
                status_code=429,
                detail="Too many requests. Please try again later.",
                headers={"Retry-After": str(self.window_seconds)}
            )

        # Add current request timestamp
        self.requests[client_ip].append(current_time)

        # Add rate limit headers to response
        response = await call_next(request)

        # Set rate limit headers
        remaining = max(0, self.requests_per_window - len(self.requests[client_ip]))
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_window)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(current_time + self.window_seconds))

        return response