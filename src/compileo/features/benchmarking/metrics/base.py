"""
Base classes for performance metrics.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Union, Optional
from dataclasses import dataclass


@dataclass
class MetricResult:
    """Result of a metric calculation."""
    metric_name: str
    value: Union[float, Dict[str, float]]
    metadata: Optional[Dict[str, Any]] = None


class BaseMetric(ABC):
    """
    Abstract base class for performance metrics.

    Each metric should implement the calculate method
    to compute the metric from predictions and labels.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    @abstractmethod
    def calculate(self, predictions: List[Any], labels: List[Any], **kwargs) -> MetricResult:
        """
        Calculate the metric.

        Args:
            predictions: Model predictions
            labels: Ground truth labels
            **kwargs: Additional calculation parameters

        Returns:
            MetricResult object
        """
        pass

    @abstractmethod
    def name(self) -> str:
        """
        Get the name of this metric.

        Returns:
            Metric name
        """
        pass

    def is_available(self) -> bool:
        """
        Check if this metric is available (dependencies loaded).

        Returns:
            True if available, False otherwise
        """
        return True