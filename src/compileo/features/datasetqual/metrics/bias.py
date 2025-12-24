"""
Bias detection metrics for dataset quality analysis.

Detects various types of bias in datasets:
- Demographic bias (gender, age, ethnicity)
- Content bias (topic representation)
- Representation bias (under/over-representation)
"""

import re
from collections import Counter
from typing import Any, Dict, List, Optional, Set
import logging

from .base_metric import BaseMetric, MetricResult

logger = logging.getLogger(__name__)


class BiasMetric(BaseMetric):
    """
    Analyzes dataset for various types of bias.

    Measures:
    - Demographic bias: Unequal representation of demographic groups
    - Content bias: Bias in topic or content coverage
    - Representation bias: Imbalanced representation of concepts
    """

    def __init__(self, enabled: bool = True, threshold: Optional[float] = None,
                 demographic_keywords: Optional[Dict[str, List[str]]] = None):
        """
        Initialize bias metric.

        Args:
            enabled: Whether this metric is enabled
            threshold: Maximum acceptable bias score (0-1, lower is better)
            demographic_keywords: Custom demographic keywords for detection
        """
        super().__init__("Bias Detection", enabled, threshold)

        # Default demographic keywords
        self.demographic_keywords = demographic_keywords or {
            'gender': ['man', 'woman', 'male', 'female', 'he', 'she', 'his', 'her'],
            'age': ['young', 'old', 'elderly', 'child', 'adult', 'teenage'],
            'ethnicity': ['race', 'ethnic', 'cultural', 'minority', 'majority']
        }

    def analyze(self, dataset: List[Dict[str, Any]]) -> MetricResult:
        """
        Analyze bias in the dataset.

        Args:
            dataset: List of dataset items

        Returns:
            MetricResult with bias analysis
        """
        if not dataset:
            return self._create_result(0.0, {"error": "Empty dataset"})

        # Extract text content
        all_text = []
        for item in dataset:
            question = item.get('question', '')
            answer = item.get('answer', '')
            all_text.append(f"{question} {answer}")

        # Calculate different bias types
        demographic_bias = self._calculate_demographic_bias(all_text)
        content_bias = self._calculate_content_bias(dataset)
        representation_bias = self._calculate_representation_bias(dataset)

        # Overall bias score (average of bias measures)
        # Lower scores indicate less bias
        overall_bias = (demographic_bias + content_bias + representation_bias) / 3.0

        details = {
            "demographic_bias": demographic_bias,
            "content_bias": content_bias,
            "representation_bias": representation_bias,
            "bias_interpretation": self._interpret_bias_score(overall_bias)
        }

        # For bias metrics, lower scores are better, so invert the pass/fail logic
        return self._create_result(overall_bias, details)

    def _calculate_demographic_bias(self, texts: List[str]) -> float:
        """
        Calculate demographic bias based on keyword representation.

        Returns bias score 0-1, where 0 is no bias, 1 is high bias.
        """
        if not texts:
            return 0.0

        # Count occurrences of demographic keywords
        keyword_counts = {}
        total_words = 0

        for text in texts:
            words = text.lower().split()
            total_words += len(words)

            for category, keywords in self.demographic_keywords.items():
                count = sum(1 for word in words if word in keywords)
                keyword_counts[category] = keyword_counts.get(category, 0) + count

        if total_words == 0:
            return 0.0

        # Calculate bias as coefficient of variation of keyword frequencies
        frequencies = [count / total_words for count in keyword_counts.values()]

        if len(frequencies) <= 1:
            return 0.0

        mean_freq = sum(frequencies) / len(frequencies)
        if mean_freq == 0:
            return 0.0

        variance = sum((f - mean_freq) ** 2 for f in frequencies) / len(frequencies)
        std_dev = variance ** 0.5
        cv = std_dev / mean_freq  # coefficient of variation

        # Normalize to 0-1 scale (cap at reasonable maximum)
        return min(cv / 2.0, 1.0)

    def _calculate_content_bias(self, dataset: List[Dict[str, Any]]) -> float:
        """
        Calculate content bias based on topic/category distribution.

        Returns bias score 0-1, where 0 is balanced, 1 is highly biased.
        """
        categories = []
        for item in dataset:
            category = item.get('category') or item.get('topic') or 'unknown'
            categories.append(category)

        if not categories:
            return 0.0

        category_counts = Counter(categories)
        total_items = len(categories)

        # Calculate Gini coefficient for category distribution
        # Gini = 0 is perfect equality, Gini = 1 is perfect inequality
        sorted_counts = sorted(category_counts.values())
        n = len(sorted_counts)

        if n <= 1:
            return 0.0

        gini = 0.0
        for i, count in enumerate(sorted_counts):
            gini += (2 * i - n + 1) * count

        gini = gini / (n * sum(sorted_counts))
        gini = 1 - gini  # Convert to inequality measure

        return gini

    def _calculate_representation_bias(self, dataset: List[Dict[str, Any]]) -> float:
        """
        Calculate representation bias based on answer lengths and complexity.

        Returns bias score 0-1, where 0 is balanced, 1 is highly biased.
        """
        if not dataset:
            return 0.0

        # Analyze answer characteristics
        answer_lengths = []
        answer_complexities = []

        for item in dataset:
            answer = item.get('answer', '')
            if answer:
                length = len(answer.split())
                answer_lengths.append(length)

                # Simple complexity measure: unique words / total words
                words = answer.split()
                if words:
                    unique_ratio = len(set(words)) / len(words)
                    answer_complexities.append(unique_ratio)

        if not answer_lengths:
            return 0.0

        # Calculate coefficient of variation for lengths
        mean_length = sum(answer_lengths) / len(answer_lengths)
        if mean_length == 0:
            length_cv = 0.0
        else:
            variance = sum((l - mean_length) ** 2 for l in answer_lengths) / len(answer_lengths)
            std_dev = variance ** 0.5
            length_cv = std_dev / mean_length

        # Calculate coefficient of variation for complexities
        if answer_complexities:
            mean_complexity = sum(answer_complexities) / len(answer_complexities)
            if mean_complexity == 0:
                complexity_cv = 0.0
            else:
                variance = sum((c - mean_complexity) ** 2 for c in answer_complexities) / len(answer_complexities)
                std_dev = variance ** 0.5
                complexity_cv = std_dev / mean_complexity
        else:
            complexity_cv = 0.0

        # Average bias from length and complexity variation
        representation_bias = (length_cv + complexity_cv) / 2.0

        # Normalize to 0-1
        return min(representation_bias / 2.0, 1.0)

    def _interpret_bias_score(self, score: float) -> str:
        """Provide human-readable interpretation of bias score."""
        if score < 0.2:
            return "Low bias - well balanced dataset"
        elif score < 0.4:
            return "Moderate bias - some imbalances present"
        elif score < 0.6:
            return "High bias - significant imbalances detected"
        else:
            return "Critical bias - major representation issues"

    def _calculate_pass_fail(self, score: float) -> Optional[bool]:
        """
        For bias metrics, lower scores are better.
        Pass if score is below threshold.
        """
        if self.threshold is None:
            return None
        return score <= self.threshold