"""Compileo: A modular pipeline for dataset creation and curation."""

__version__ = "0.1.0"

# Import main modules for easy access
from . import core
from . import features
from . import storage
from . import utils

__all__ = ["core", "features", "storage", "utils"]