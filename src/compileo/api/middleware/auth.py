"""Authentication middleware for the Compileo API."""

import hmac
import os
from typing import Callable

from fastapi import HTTPException, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from ..core.config import settings
from ...core.logging import get_logger

logger = get_logger(__name__)


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """API Key authentication middleware."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip authentication for health check and root endpoints
        if request.url.path in ["/health", "/", "/docs", "/redoc", "/openapi.json"]:
            return await call_next(request)

        # Force reload from all sources (Directly from source for reliability)
        from ...core.settings import BackendSettings
        
        all_keys = []
        
        # 1. Environment Variables
        k1 = os.environ.get("COMPILEO_API_KEY")
        if k1: 
            val = str(k1).strip()
            if val: all_keys.append(val)
        
        k2 = os.environ.get("COMPILEO_API_KEYS")
        if k2: 
            for k in str(k2).split(","):
                val = k.strip()
                if val: all_keys.append(val)
                
        # 2. CLI Argument
        if hasattr(settings, 'cli_api_key_override') and settings.cli_api_key_override:
            val = str(settings.cli_api_key_override).strip()
            if val: all_keys.append(val)
            
        # 3. Database
        try:
            db_k = BackendSettings.get_setting("api_key")
            if db_k: 
                val = str(db_k).strip()
                if val: all_keys.append(val)
        except: pass

        # If NO keys are configured ANYWHERE, operate in UNSECURED mode
        if len(all_keys) == 0:
            # logger.debug("No keys found. Allowing unsecured access.")
            return await call_next(request)

        # Get API key from header
        api_key = request.headers.get(settings.api_key_header)

        if not api_key:
            logger.warning(f"Missing API key for request to {request.url.path}")
            return JSONResponse(
                status_code=401,
                content={
                    "error": "Unauthorized",
                    "message": f"Missing {settings.api_key_header} header"
                }
            )

        # Validate API key against the consolidated list
        found = False
        for valid_key in all_keys:
            if hmac.compare_digest(str(api_key), str(valid_key)):
                found = True
                break
                
        if not found:
            logger.warning(f"Invalid API key provided for request to {request.url.path}")
            return JSONResponse(
                status_code=401,
                content={
                    "error": "Unauthorized",
                    "message": "Invalid API key"
                }
            )

        return await call_next(request)
