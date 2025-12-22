"""
Diversity metrics for dataset quality analysis.

Measures lexical diversity, semantic diversity, and topic distribution
to ensure the dataset covers a wide range of content.
"""

import math
from collections import Counter
from typing import Any, Dict, List, Optional, Set
import logging

from .base_metric import BaseMetric, MetricResult

logger = logging.getLogger(__name__)


class DiversityMetric(BaseMetric):
    """
    Analyzes dataset diversity across multiple dimensions.

    Measures:
    - Lexical diversity: Variety of words used
    - Semantic diversity: Variety of concepts/topics
    - Topic distribution: Balance across different subject areas
    """

    def __init__(self, enabled: bool = True, threshold: Optional[float] = None,
                 min_lexical_diversity: float = 0.3, min_semantic_diversity: float = 0.4):
        """
        Initialize diversity metric.

        Args:
            enabled: Whether this metric is enabled
            threshold: Overall diversity threshold (0-1)
            min_lexical_diversity: Minimum acceptable lexical diversity
            min_semantic_diversity: Minimum acceptable semantic diversity
        """
        super().__init__("Diversity Analysis", enabled, threshold)
        self.min_lexical_diversity = min_lexical_diversity
        self.min_semantic_diversity = min_semantic_diversity

    def analyze(self, dataset: List[Dict[str, Any]]) -> MetricResult:
        """
        Analyze diversity of the dataset.

        Args:
            dataset: List of dataset items with questions/answers

        Returns:
            MetricResult with diversity scores
        """
        if not dataset:
            return self._create_result(0.0, {"error": "Empty dataset"})

        # Extract text content
        questions = [item.get('question', '') for item in dataset]
        answers = [item.get('answer', '') for item in dataset]
        all_text = questions + answers

        # Calculate lexical diversity
        lexical_score = self._calculate_lexical_diversity(all_text)

        # Calculate semantic diversity (simplified - could use embeddings)
        semantic_score = self._calculate_semantic_diversity(all_text)

        # Calculate topic distribution balance
        topic_balance = self._calculate_topic_balance(dataset)

        # Overall diversity score (weighted average)
        overall_score = (lexical_score * 0.4 + semantic_score * 0.4 + topic_balance * 0.2)

        details = {
            "lexical_diversity": lexical_score,
            "semantic_diversity": semantic_score,
            "topic_balance": topic_balance,
            "min_lexical_threshold": self.min_lexical_diversity,
            "min_semantic_threshold": self.min_semantic_diversity
        }

        # Check individual thresholds
        issues = []
        if lexical_score < self.min_lexical_diversity:
            issues.append(f"Low lexical diversity: {lexical_score:.3f} < {self.min_lexical_diversity}")
        if semantic_score < self.min_semantic_diversity:
            issues.append(f"Low semantic diversity: {semantic_score:.3f} < {self.min_semantic_diversity}")

        if issues:
            details["issues"] = issues

        return self._create_result(overall_score, details)

    def _calculate_lexical_diversity(self, texts: List[str]) -> float:
        """
        Calculate lexical diversity using Type-Token Ratio (TTR).

        TTR = unique words / total words
        Higher values indicate more diverse vocabulary.
        """
        if not texts:
            return 0.0

        # Tokenize and count
        all_words = []
        for text in texts:
            if text:
                words = text.lower().split()
                all_words.extend(words)

        if not all_words:
            return 0.0

        unique_words = len(set(all_words))
        total_words = len(all_words)

        # TTR can be inflated for small samples, so we use a corrected version
        if total_words > 100:
            # Use corrected TTR for larger samples
            ttr = unique_words / math.sqrt(total_words)
        else:
            ttr = unique_words / total_words

        # Normalize to 0-1 scale (rough approximation)
        return min(ttr, 1.0)

    def _calculate_semantic_diversity(self, texts: List[str]) -> float:
        """
        Calculate semantic diversity based on word categories and patterns.

        This is a simplified version. In production, this would use embeddings
        or topic modeling to measure conceptual diversity.
        """
        if not texts:
            return 0.0

        # Simple heuristic: diversity based on different word categories
        categories = {
            'questions': 0,
            'technical': 0,
            'medical': 0,
            'general': 0
        }

        question_words = {'what', 'how', 'why', 'when', 'where', 'who'}
        technical_words = {'algorithm', 'function', 'method', 'class', 'variable'}
        medical_words = {'patient', 'diagnosis', 'treatment', 'symptom', 'disease'}

        for text in texts:
            if not text:
                continue
            words = set(text.lower().split())

            if words & question_words:
                categories['questions'] += 1
            if words & technical_words:
                categories['technical'] += 1
            if words & medical_words:
                categories['medical'] += 1
            if len(words) > 0:
                categories['general'] += 1

        # Calculate entropy as diversity measure
        total = sum(categories.values())
        if total == 0:
            return 0.0

        entropy = 0.0
        for count in categories.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)

        # Normalize entropy to 0-1 (max entropy for 4 categories is ~2)
        return entropy / 2.0

    def _calculate_topic_balance(self, dataset: List[Dict[str, Any]]) -> float:
        """
        Calculate balance of topics/categories in the dataset.

        Assumes dataset items have 'category' or 'topic' field.
        """
        categories = []
        for item in dataset:
            category = item.get('category') or item.get('topic') or 'unknown'
            categories.append(category)

        if not categories:
            return 0.0

        # Count category distribution
        category_counts = Counter(categories)
        total_items = len(categories)

        # Calculate balance using normalized entropy
        entropy = 0.0
        for count in category_counts.values():
            p = count / total_items
            entropy -= p * math.log2(p)

        # Normalize by maximum possible entropy
        max_entropy = math.log2(len(category_counts)) if category_counts else 0
        balance = entropy / max_entropy if max_entropy > 0 else 0.0

        return balance