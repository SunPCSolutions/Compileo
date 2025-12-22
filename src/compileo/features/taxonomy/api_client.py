"""
API client for taxonomy generation using Grok.
"""

import time
from typing import Dict, Any, Optional
import requests


class GrokAPIClient:
    """
    Client for interacting with Grok API for taxonomy generation.
    """

    def __init__(self, api_key: str, model: Optional[str] = None):
        """
        Initialize the Grok API client.

        Args:
            api_key: xAI Grok API key
            model: Grok model to use (optional, defaults to "grok-4-fast-reasoning")
        """
        self.api_key = api_key
        self.api_url = "https://api.x.ai/v1/chat/completions"
        self.model = model if model is not None else "grok-4-fast-reasoning"

    def call_for_taxonomy_generation(self, prompt: str) -> str:
        """
        Call Grok API for taxonomy generation.

        Args:
            prompt: The prompt to send to Grok

        Returns:
            Raw Grok response text

        Raises:
            Exception: If API call fails after retries
        """
        return self._make_api_call(prompt, "taxonomy_generation")

    def call_for_taxonomy_extension(self, prompt: str) -> str:
        """
        Call Grok API for taxonomy extension.

        Args:
            prompt: The prompt to send to Grok

        Returns:
            Raw Grok response text

        Raises:
            Exception: If API call fails after retries
        """
        return self._make_api_call(prompt, "taxonomy_extension")

    def _make_api_call(self, prompt: str, operation_type: str) -> str:
        """
        Make API call to Grok with retry logic.

        Args:
            prompt: The prompt to send
            operation_type: Type of operation for error messages

        Returns:
            Raw response content

        Raises:
            Exception: If all retries fail
        """
        request_data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 4000,
            "temperature": 0.3  # Lower temperature for more consistent results
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    json=request_data,
                    headers=headers,
                    timeout=60
                )
                response.raise_for_status()

                response_json = response.json()
                content = response_json.get("choices", [{}])[0].get("message", {}).get("content", "")

                if content:
                    return content
                else:
                    raise ValueError("Empty response from Grok API")

            except (requests.exceptions.RequestException, ValueError) as e:
                if attempt == max_retries - 1:
                    raise Exception(f"Failed to {operation_type} after {max_retries} attempts: {e}")
                time.sleep(2 ** attempt)  # Exponential backoff

        raise Exception(f"Unexpected error in {operation_type}")