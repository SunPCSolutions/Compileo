"""
Plugin caching service for the GUI.

Provides centralized caching for plugin-related API calls to avoid
redundant network requests when multiple components need the same data.
"""

import streamlit as st
from typing import List, Dict, Any, Optional


def get_plugins_cached() -> List[Dict[str, Any]]:
    """
    Get the list of installed plugins with caching.
    
    Uses Streamlit session state to cache plugin data, avoiding redundant
    API calls when multiple components need the same information.
    
    Returns:
        List of plugin dictionaries, or empty list if no plugins or error.
    """
    # Check cache first
    if 'gui_plugins_cache' not in st.session_state:
        # Import here to avoid circular imports
        from .api_client import api_client
        try:
            st.session_state.gui_plugins_cache = api_client.get("/api/v1/plugins/")
        except Exception as e:
            # Graceful fallback - return empty list if API fails
            # This ensures the app works even without plugins
            st.session_state.gui_plugins_cache = []
    
    return st.session_state.gui_plugins_cache or []


def get_dataset_formats_cached() -> List[str]:
    """
    Get available dataset formats (built-in + plugin) with caching.
    
    Returns:
        List of format strings. Always includes built-in formats
        as fallback even if plugin formats fail to load.
    """
    # Default built-in formats as fallback
    default_formats = ["jsonl", "parquet", "json"]
    
    # Check cache first
    if 'gui_dataset_formats_cache' not in st.session_state:
        from .api_client import api_client
        try:
            response = api_client.get("/api/v1/plugins/dataset-formats")
            if response:
                st.session_state.gui_dataset_formats_cache = response.get("formats", default_formats)
            else:
                st.session_state.gui_dataset_formats_cache = default_formats
        except Exception as e:
            # Graceful fallback - always return default formats
            st.session_state.gui_dataset_formats_cache = default_formats
    
    return st.session_state.gui_dataset_formats_cache


def check_scraper_plugin() -> bool:
    """
    Check if the scrapy-playwright plugin is installed.
    
    Uses cached plugin list for efficiency.
    
    Returns:
        True if scraper plugin is installed, False otherwise.
    """
    plugins = get_plugins_cached()
    return any(p.get('id') == 'compileo-scrapy-playwright-scraper' for p in plugins)


def invalidate_plugins_cache():
    """
    Invalidate the plugin caches.
    
    Call this after plugin install/uninstall operations to ensure
    fresh data is fetched on next access.
    """
    if 'gui_plugins_cache' in st.session_state:
        del st.session_state.gui_plugins_cache
    
    if 'gui_dataset_formats_cache' in st.session_state:
        del st.session_state.gui_dataset_formats_cache
