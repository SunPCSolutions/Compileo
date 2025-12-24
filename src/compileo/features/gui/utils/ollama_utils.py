"""
Utilities for AI model API integrations.
"""

import requests
from typing import List, Optional
from src.compileo.core.logging import get_logger

logger = get_logger(__name__)


def get_ollama_models(base_url: str = "http://localhost:11434") -> List[str]:
    """
    Get list of available Ollama models.

    Args:
        base_url: Ollama API base URL

    Returns:
        List of model names, or empty list if unable to fetch
    """
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        response.raise_for_status()

        data = response.json()
        models = data.get("models", [])

        # Extract model names
        model_names = []
        for model in models:
            if isinstance(model, dict) and "name" in model:
                model_names.append(model["name"])
            elif isinstance(model, str):
                model_names.append(model)

        return sorted(model_names)

    except (requests.exceptions.RequestException, ValueError, KeyError) as e:
        logger.error(f"Error fetching Ollama models: {e}")
        return []


def test_ollama_connection(base_url: str = "http://localhost:11434") -> bool:
    """
    Test connection to Ollama server.

    Args:
        base_url: Ollama API base URL

    Returns:
        True if connection successful, False otherwise
    """
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def get_gemini_models(api_key: str) -> List[str]:
    """
    Get list of available Gemini models.

    Args:
        api_key: Google Gemini API key

    Returns:
        List of model names, or default list if unable to fetch
    """
    if not api_key:
        return ["gemini-2.5-flash"]  # Default fallback

    try:
        url = "https://generativelanguage.googleapis.com/v1beta/models"
        params = {"key": api_key}
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()
        models = data.get("models", [])

        # Extract model names (remove 'models/' prefix if present)
        model_names = []
        for model in models:
            if isinstance(model, dict) and "name" in model:
                name = model["name"]
                if name.startswith("models/"):
                    name = name[7:]  # Remove 'models/' prefix
                model_names.append(name)

        return sorted(model_names) if model_names else ["gemini-2.5-flash"]

    except (requests.exceptions.RequestException, ValueError, KeyError) as e:
        logger.error(f"Error fetching Gemini models: {e}")
        return ["gemini-2.5-flash"]  # Default fallback


def get_openai_models(api_key: str) -> List[str]:
    """
    Get list of available OpenAI models.

    Args:
        api_key: OpenAI API key

    Returns:
        List of model names, or default list if unable to fetch
    """
    if not api_key:
        return ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]  # Default fallback

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        models = client.models.list()

        model_names = [
            model.id for model in models.data
            if model.id.startswith(("gpt-", "o1-", "o3-"))
        ]

        return sorted(model_names) if model_names else ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]

    except Exception as e:
        logger.error(f"Error fetching OpenAI models: {e}")
        return ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]  # Default fallback


def get_grok_models(api_key: str) -> List[str]:
    """
    Get list of available Grok models.

    Args:
        api_key: xAI Grok API key

    Returns:
        List of model names, or default list if unable to fetch
    """
    if not api_key:
        return ["grok-4-fast-reasoning"]  # Default fallback

    try:
        url = "https://api.x.ai/v1/models"
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        data = response.json()
        models = data.get("data", []) if isinstance(data, dict) else data

        # Extract model IDs
        model_names = []
        for model in models:
            if isinstance(model, dict) and "id" in model:
                model_names.append(model["id"])
            elif isinstance(model, str):
                model_names.append(model)

        return sorted(model_names) if model_names else ["grok-4-fast-reasoning"]

    except (requests.exceptions.RequestException, ValueError, KeyError) as e:
        logger.error(f"Error fetching Grok models: {e}")
        return ["grok-4-fast-reasoning"]  # Default fallback