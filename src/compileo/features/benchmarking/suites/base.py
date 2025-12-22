"""
Base classes for benchmark suites.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class BenchmarkResult:
    """Result of a benchmark evaluation."""
    benchmark_name: str
    task_name: str
    metrics: Dict[str, float]
    predictions: Optional[List[Any]] = None
    labels: Optional[List[Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseBenchmark(ABC):
    """
    Abstract base class for benchmark suites.

    Each benchmark suite should implement the evaluate method
    to run the benchmark on a given model.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    @abstractmethod
    def evaluate(self, ai_config: Dict[str, Any], **kwargs) -> List[BenchmarkResult]:
        """
        Evaluate using AI provider configuration.

        Args:
            ai_config: AI provider configuration (gemini_api_key, grok_api_key, ollama_available, etc.)
            **kwargs: Additional evaluation parameters

        Returns:
            List of BenchmarkResult objects
        """
        pass

    @abstractmethod
    def get_tasks(self) -> List[str]:
        """
        Get the list of tasks in this benchmark suite.

        Returns:
            List of task names
        """
        pass

    @abstractmethod
    def load_dataset(self, task_name: str) -> Any:
        """
        Load the dataset for a specific task.

        Args:
            task_name: Name of the task

        Returns:
            Dataset object (format depends on implementation)
        """
        pass

    def is_available(self) -> bool:
        """
        Check if this benchmark is available (dependencies loaded).

        Returns:
            True if available, False otherwise
        """
        return True