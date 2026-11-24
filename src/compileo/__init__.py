"""
Compileo: A modular pipeline for dataset creation and curation.

This module provides compatibility for both development (src.compileo.*) and
production/compiled (compileo.*) import paths.
"""

__version__ = "0.1.0"

# Import main modules for easy access
# Use try/except to handle both import paths
try:
    # Try compiled path first (production)
    from compileo import core
    from compileo import features
    from compileo import storage
    from compileo import utils
except ImportError:
    # Fall back to source path (development)
    from src.compileo import core
    from src.compileo import features
    from src.compileo import storage
    from src.compileo import utils

__all__ = ["core", "features", "storage", "utils"]


def get_import_path():
    """
    Determine the current import path being used.

    Returns:
        str: 'compileo' for compiled/production, 'src.compileo' for development
    """
    try:
        import compileo
        return 'compileo'
    except ImportError:
        return 'src.compileo'
