import logging
import httpx
import base64
from typing import Dict, Any, Optional
from .models import ActivationResponse, ValidationResponse

logger = logging.getLogger(__name__)

class LicenseConnector:
    """Connector for WooCommerce Digital License Manager API."""
    
    def __init__(self, base_url: str = "https://www.notin.ai"):
        self.base_url = base_url.rstrip("/") + "/wp-json/dlm/v1"
        self.ck = "ck_b72d1937752eaa086df70c52afedfb08b7957a44"
        self.cs = "cs_e369c5ac11d493d3b629f7ce89b44c2f21a63d22"
        
        # Prepare Basic Auth header
        auth_str = f"{self.ck}:{self.cs}"
        encoded_auth = base64.b64encode(auth_str.encode()).decode()
        self.headers = {
            "Authorization": f"Basic {encoded_auth}",
            "Accept": "application/json"
        }

    async def activate(self, license_key: str, label: str) -> ActivationResponse:
        """Activates a license key."""
        url = f"{self.base_url}/licenses/activate/{license_key}"
        params = {"label": label}
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.get(url, params=params, headers=self.headers)
                logger.info(f"License activation response: {response.text}")
                data = response.json()
                
                # Handle error responses that don't match the ActivationResponse schema
                if response.status_code != 200 or data.get('code'):
                    return ActivationResponse(
                        success=False,
                        code=data.get('code'),
                        message=data.get('message', 'Activation failed')
                    )
                
                return ActivationResponse(**data)
            except Exception as e:
                logger.error(f"License activation error: {e}")
                return ActivationResponse(success=False, message=str(e))

    async def validate(self, activation_token: str) -> ValidationResponse:
        """Validates an activation token."""
        url = f"{self.base_url}/licenses/validate/{activation_token}"
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.get(url, headers=self.headers)
                return ValidationResponse(**response.json())
            except Exception as e:
                return ValidationResponse(success=False, message=str(e))

    async def deactivate(self, activation_token: str) -> ActivationResponse:
        """Deactivates an activation token."""
        url = f"{self.base_url}/licenses/deactivate/{activation_token}"
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.get(url, headers=self.headers)
                return ActivationResponse(**response.json())
            except Exception as e:
                return ActivationResponse(success=False, message=str(e))

    async def get_license(self, license_key: str) -> Dict[str, Any]:
        """Gets detailed information about a license key."""
        url = f"{self.base_url}/licenses/{license_key}"
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.get(url, headers=self.headers)
                return response.json()
            except Exception as e:
                logger.error(f"Error fetching license info: {e}")
                return {"success": False, "message": str(e)}
