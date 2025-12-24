"""
Performance evaluation metrics for benchmarking.

This package contains implementations of various performance metrics
including classification metrics, generation metrics, and custom metrics.
"""

from .base import BaseMetric
from .classification import Accuracy, F1Score, Precision, Recall
from .generation import BLEUScore, ROUGEScore, METEORScore

__all__ = [
    "BaseMetric",
    "Accuracy", "F1Score", "Precision", "Recall",
    "BLEUScore", "ROUGEScore", "METEORScore"
]