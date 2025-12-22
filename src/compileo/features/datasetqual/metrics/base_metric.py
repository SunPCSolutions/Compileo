"""
Base metric class for dataset quality metrics.
All quality metrics should inherit from this class.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class MetricResult:
    """Result of a quality metric analysis."""
    name: str
    score: float
    threshold: Optional[float] = None
    passed: Optional[bool] = None
    details: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseMetric(ABC):
    """
    Abstract base class for all quality metrics.

    Each metric should implement the analyze method to compute
    quality scores for a given dataset.
    """

    def __init__(self, name: str, enabled: bool = True, threshold: Optional[float] = None):
        """
        Initialize the metric.

        Args:
            name: Human-readable name of the metric
            enabled: Whether this metric is enabled
            threshold: Optional threshold for pass/fail determination
        """
        self.name = name
        self.enabled = enabled
        self.threshold = threshold

    @abstractmethod
    def analyze(self, dataset: List[Dict[str, Any]]) -> MetricResult:
        """
        Analyze the dataset and return quality metrics.

        Args:
            dataset: List of dataset items (questions, answers, metadata)

        Returns:
            MetricResult with score and analysis details
        """
        pass

    def is_enabled(self) -> bool:
        """Check if this metric is enabled."""
        return self.enabled

    def set_threshold(self, threshold: float):
        """Set the threshold for pass/fail determination."""
        self.threshold = threshold

    def _calculate_pass_fail(self, score: float) -> Optional[bool]:
        """
        Calculate pass/fail based on score and threshold.

        Returns True if score meets or exceeds threshold (higher is better).
        Override for metrics where lower scores are better.
        """
        if self.threshold is None:
            return None
        return score >= self.threshold

    def _create_result(self, score: float, details: Optional[Dict[str, Any]] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> MetricResult:
        """Create a standardized MetricResult."""
        passed = self._calculate_pass_fail(score)
        return MetricResult(
            name=self.name,
            score=score,
            threshold=self.threshold,
            passed=passed,
            details=details or {},
            metadata=metadata or {}
        )