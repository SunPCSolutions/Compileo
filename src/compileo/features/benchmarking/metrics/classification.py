"""
Classification performance metrics.
"""

from typing import Dict, List, Any, Optional
import logging
from .base import BaseMetric, MetricResult

logger = logging.getLogger(__name__)


class Accuracy(BaseMetric):
    """Accuracy metric for classification tasks."""

    def name(self) -> str:
        return "accuracy"

    def calculate(self, predictions: List[Any], labels: List[Any], **kwargs) -> MetricResult:
        """Calculate accuracy."""
        if len(predictions) != len(labels):
            raise ValueError("Predictions and labels must have the same length")

        correct = sum(1 for pred, label in zip(predictions, labels) if pred == label)
        accuracy = correct / len(labels) if labels else 0.0

        return MetricResult(
            metric_name=self.name(),
            value=accuracy,
            metadata={"correct": correct, "total": len(labels)}
        )


class F1Score(BaseMetric):
    """F1 Score metric for classification tasks."""

    def name(self) -> str:
        return "f1"

    def calculate(self, predictions: List[Any], labels: List[Any], **kwargs) -> MetricResult:
        """Calculate F1 score."""
        try:
            from sklearn.metrics import f1_score
        except ImportError:
            logger.warning("scikit-learn not available, using basic F1 calculation")
            return self._basic_f1(predictions, labels)

        # Handle different averaging strategies
        average = kwargs.get('average', 'macro')
        try:
            f1 = float(f1_score(labels, predictions, average=average))
        except Exception as e:
            logger.error(f"F1 calculation failed: {e}")
            f1 = 0.0

        return MetricResult(
            metric_name=self.name(),
            value=f1,
            metadata={"average": average}
        )

    def _basic_f1(self, predictions: List[Any], labels: List[Any]) -> MetricResult:
        """Basic F1 calculation for binary classification."""
        true_pos = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 1)
        false_pos = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 0)
        false_neg = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 1)

        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return MetricResult(
            metric_name=self.name(),
            value=f1,
            metadata={"precision": precision, "recall": recall}
        )


class Precision(BaseMetric):
    """Precision metric for classification tasks."""

    def name(self) -> str:
        return "precision"

    def calculate(self, predictions: List[Any], labels: List[Any], **kwargs) -> MetricResult:
        """Calculate precision."""
        try:
            from sklearn.metrics import precision_score
        except ImportError:
            logger.warning("scikit-learn not available, using basic precision calculation")
            return self._basic_precision(predictions, labels)

        average = kwargs.get('average', 'macro')
        try:
            precision = float(precision_score(labels, predictions, average=average))
        except Exception as e:
            logger.error(f"Precision calculation failed: {e}")
            precision = 0.0

        return MetricResult(
            metric_name=self.name(),
            value=precision,
            metadata={"average": average}
        )

    def _basic_precision(self, predictions: List[Any], labels: List[Any]) -> MetricResult:
        """Basic precision calculation for binary classification."""
        true_pos = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 1)
        false_pos = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 0)

        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0

        return MetricResult(
            metric_name=self.name(),
            value=precision
        )


class Recall(BaseMetric):
    """Recall metric for classification tasks."""

    def name(self) -> str:
        return "recall"

    def calculate(self, predictions: List[Any], labels: List[Any], **kwargs) -> MetricResult:
        """Calculate recall."""
        try:
            from sklearn.metrics import recall_score
        except ImportError:
            logger.warning("scikit-learn not available, using basic recall calculation")
            return self._basic_recall(predictions, labels)

        average = kwargs.get('average', 'macro')
        try:
            recall = float(recall_score(labels, predictions, average=average))
        except Exception as e:
            logger.error(f"Recall calculation failed: {e}")
            recall = 0.0

        return MetricResult(
            metric_name=self.name(),
            value=recall,
            metadata={"average": average}
        )

    def _basic_recall(self, predictions: List[Any], labels: List[Any]) -> MetricResult:
        """Basic recall calculation for binary classification."""
        true_pos = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 1)
        false_neg = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 1)

        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0

        return MetricResult(
            metric_name=self.name(),
            value=recall
        )