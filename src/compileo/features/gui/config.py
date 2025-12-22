"""
GUI configuration for the Compileo web frontend.
"""

import os
from typing import Optional

class GUIConfig:
    """Configuration class for the Streamlit GUI."""

    def __init__(self):
        # API Configuration
        self.api_base_url = os.getenv("API_BASE_URL", "http://localhost:8000")
        self.api_key = os.getenv("API_KEY", "")

        # GUI Configuration
        self.app_title = "Compileo Dataset Creator"
        self.app_icon = "📊"
        self.theme_primary_color = "#FF4B4B"
        self.theme_background_color = "#FFFFFF"
        self.theme_secondary_background_color = "#F0F2F6"
        self.theme_text_color = "#262730"

        # Page Configuration
        self.page_config = {
            "page_title": self.app_title,
            "page_icon": self.app_icon,
            "layout": "wide",
            "initial_sidebar_state": "expanded"
        }

        # File Upload Configuration
        self.max_file_size_mb = 200
        self.allowed_file_types = [
            "pdf", "docx", "doc", "txt", "md", "csv", "json", "xml"
        ]

        # Dataset Configuration
        self.default_page_size = 50
        self.max_concurrent_jobs = 3

        # Quality Thresholds
        self.default_quality_threshold = 0.7

        # Benchmarking Configuration
        self.default_benchmark_suite = "glue"

# Global config instance
config = GUIConfig()