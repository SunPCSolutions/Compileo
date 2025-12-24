"""
Consistency validation metrics for dataset quality analysis.

Evaluates logical consistency and coherence in answers:
- Answer coherence: Logical consistency within answers
- Answer consistency: Consistency across similar questions
- Factual accuracy: Basic fact-checking capabilities
"""

from typing import Any, Dict, List, Optional, Set
import logging
from collections import defaultdict

from .base_metric import BaseMetric, MetricResult

logger = logging.getLogger(__name__)


class ConsistencyMetric(BaseMetric):
    """
    Analyzes consistency and coherence in dataset answers.

    Measures:
    - Answer coherence: Internal logical consistency
    - Cross-answer consistency: Consistency across related questions
    - Factual consistency: Basic contradiction detection
    """

    def __init__(self, enabled: bool = True, threshold: Optional[float] = None,
                 check_factual_consistency: bool = True):
        """
        Initialize consistency metric.

        Args:
            enabled: Whether this metric is enabled
            threshold: Minimum acceptable consistency score (0-1)
            check_factual_consistency: Whether to perform basic fact-checking
        """
        super().__init__("Consistency Validation", enabled, threshold)
        self.check_factual_consistency = check_factual_consistency

    def analyze(self, dataset: List[Dict[str, Any]]) -> MetricResult:
        """
        Analyze consistency of answers in the dataset.

        Args:
            dataset: List of dataset items

        Returns:
            MetricResult with consistency analysis
        """
        if not dataset:
            return self._create_result(0.0, {"error": "Empty dataset"})

        answers = [item.get('answer', '') for item in dataset if item.get('answer')]

        if not answers:
            return self._create_result(0.0, {"error": "No answers found"})

        # Calculate different consistency measures
        coherence_score = self._calculate_answer_coherence(answers)
        cross_consistency = self._calculate_cross_consistency(dataset)
        factual_consistency = self._calculate_factual_consistency(dataset) if self.check_factual_consistency else 1.0

        # Overall consistency score (harmonic mean for strict consistency)
        scores = [coherence_score, cross_consistency, factual_consistency]
        valid_scores = [s for s in scores if s > 0]

        if valid_scores:
            # Harmonic mean: penalizes low scores more heavily
            harmonic_mean = len(valid_scores) / sum(1/s for s in valid_scores)
        else:
            harmonic_mean = 0.0

        details = {
            "answer_coherence": coherence_score,
            "cross_consistency": cross_consistency,
            "factual_consistency": factual_consistency,
            "method": "harmonic_mean"
        }

        # Identify specific consistency issues
        issues = self._identify_consistency_issues(dataset, coherence_score, cross_consistency, factual_consistency)
        if issues:
            details["issues"] = issues

        return self._create_result(harmonic_mean, details)

    def _calculate_answer_coherence(self, answers: List[str]) -> float:
        """
        Calculate coherence within individual answers.

        Measures logical flow and consistency within each answer.
        Returns score 0-1, where 1 is perfectly coherent.
        """
        if not answers:
            return 0.0

        coherence_scores = []

        for answer in answers:
            if not answer.strip():
                continue

            score = self._analyze_single_answer_coherence(answer)
            coherence_scores.append(score)

        return sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0.0

    def _analyze_single_answer_coherence(self, answer: str) -> float:
        """
        Analyze coherence of a single answer.

        Simple heuristics:
        - Sentence connectivity
        - Logical flow indicators
        - Contradiction detection
        """
        sentences = [s.strip() for s in answer.split('.') if s.strip()]

        if len(sentences) <= 1:
            return 1.0  # Single sentence is coherent by definition

        # Check for logical connectors
        connectors = ['however', 'therefore', 'thus', 'because', 'although', 'since', 'while']
        connector_count = sum(1 for sentence in sentences
                            for connector in connectors
                            if connector in sentence.lower())

        # Check for contradictions (simple keyword-based)
        contradictions = ['but', 'however', 'although', 'despite', 'yet']
        contradiction_count = sum(1 for sentence in sentences
                                for contradiction in contradictions
                                if contradiction in sentence.lower())

        # Calculate coherence score
        connector_ratio = connector_count / (len(sentences) - 1)  # connectors per transition
        contradiction_penalty = min(contradiction_count * 0.1, 0.5)  # penalty for contradictions

        base_coherence = min(1.0, connector_ratio * 0.5 + 0.5)  # base score from connectivity
        coherence = base_coherence * (1 - contradiction_penalty)

        return max(0.0, min(1.0, coherence))

    def _calculate_cross_consistency(self, dataset: List[Dict[str, Any]]) -> float:
        """
        Calculate consistency across related answers.

        Groups answers by category/topic and checks for consistency within groups.
        Returns score 0-1, where 1 is perfectly consistent.
        """
        # Group by category
        category_answers = defaultdict(list)

        for item in dataset:
            category = item.get('category') or item.get('topic') or 'general'
            answer = item.get('answer', '')
            if answer.strip():
                category_answers[category].append(answer)

        if not category_answers:
            return 1.0

        # Calculate consistency within each category
        category_consistencies = []

        for answers in category_answers.values():
            if len(answers) > 1:
                consistency = self._calculate_category_consistency(answers)
                category_consistencies.append(consistency)
            else:
                category_consistencies.append(1.0)  # Single answer is consistent

        # Average consistency across categories
        return sum(category_consistencies) / len(category_consistencies)

    def _calculate_category_consistency(self, answers: List[str]) -> float:
        """
        Calculate consistency within a category of answers.

        Uses simple text similarity measures.
        """
        if len(answers) <= 1:
            return 1.0

        # Calculate pairwise similarities
        similarities = []

        for i in range(len(answers)):
            for j in range(i + 1, len(answers)):
                sim = self._calculate_text_similarity(answers[i], answers[j])
                similarities.append(sim)

        if not similarities:
            return 1.0

        # Average similarity as consistency measure
        avg_similarity = sum(similarities) / len(similarities)

        # For consistency, we want some similarity but not too much (avoid redundancy)
        # Optimal consistency: moderate similarity
        if avg_similarity < 0.3:
            consistency = avg_similarity / 0.3  # Scale up low similarity
        elif avg_similarity > 0.8:
            consistency = 1.0 - (avg_similarity - 0.8) / 0.2  # Penalize high similarity
        else:
            consistency = 1.0  # Optimal range

        return consistency

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate simple text similarity using Jaccard similarity of words.
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union)

    def _calculate_factual_consistency(self, dataset: List[Dict[str, Any]]) -> float:
        """
        Perform basic factual consistency checks.

        This is a simplified implementation. In production, this would
        integrate with fact-checking services or knowledge bases.
        """
        # Simple contradiction detection based on common patterns
        contradictions_found = 0
        total_checks = 0

        for item in dataset:
            answer = item.get('answer', '')
            if answer:
                contradictions_found += self._check_answer_contradictions(answer)
                total_checks += 1

        if total_checks == 0:
            return 1.0

        # Return consistency score (1.0 - contradiction ratio)
        contradiction_ratio = contradictions_found / total_checks
        return max(0.0, 1.0 - contradiction_ratio)

    def _check_answer_contradictions(self, answer: str) -> int:
        """
        Check for basic contradictions in an answer.

        Returns number of contradictions found.
        """
        contradictions = 0

        # Simple pattern-based contradiction detection
        lower_answer = answer.lower()

        # Check for yes/no contradictions
        if ('yes' in lower_answer and 'no' in lower_answer and
            'but' in lower_answer or 'however' in lower_answer):
            contradictions += 1

        # Check for impossible combinations
        impossible_pairs = [
            ('always', 'never'),
            ('all', 'none'),
            ('every', 'no'),
        ]

        for pair in impossible_pairs:
            if pair[0] in lower_answer and pair[1] in lower_answer:
                contradictions += 1

        return contradictions

    def _identify_consistency_issues(self, dataset: List[Dict[str, Any]],
                                   coherence: float, cross_consistency: float,
                                   factual: float) -> List[str]:
        """Identify specific consistency issues."""
        issues = []

        if coherence < 0.7:
            issues.append(f"Low answer coherence: {coherence:.3f}")
        if cross_consistency < 0.6:
            issues.append(f"Low cross-answer consistency: {cross_consistency:.3f}")
        if factual < 0.8:
            issues.append(f"Factual inconsistencies detected: {factual:.3f}")

        return issues