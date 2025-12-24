"""
Difficulty assessment metrics for dataset quality analysis.

Evaluates the complexity and difficulty of questions and answers:
- Complexity scoring based on linguistic features
- Readability metrics
- Cognitive load assessment
"""

import re
from typing import Any, Dict, List, Optional
import logging

from .base_metric import BaseMetric, MetricResult

logger = logging.getLogger(__name__)


class DifficultyMetric(BaseMetric):
    """
    Analyzes difficulty and complexity of dataset content.

    Measures:
    - Linguistic complexity: Sentence structure, vocabulary
    - Readability: Flesch-Kincaid, SMOG scores
    - Cognitive load: Information density and processing requirements
    """

    def __init__(self, enabled: bool = True, threshold: Optional[float] = None,
                 target_difficulty: str = "intermediate"):
        """
        Initialize difficulty metric.

        Args:
            enabled: Whether this metric is enabled
            threshold: Acceptable difficulty range threshold
            target_difficulty: Target difficulty level ("easy", "intermediate", "hard")
        """
        super().__init__("Difficulty Assessment", enabled, threshold)
        self.target_difficulty = target_difficulty

        # Difficulty ranges (Flesch Reading Ease scores)
        self.difficulty_ranges = {
            "easy": (60, 100),      # 60-100: Easy to read
            "intermediate": (30, 60), # 30-60: Intermediate
            "hard": (0, 30)         # 0-30: Hard to read
        }

    def analyze(self, dataset: List[Dict[str, Any]]) -> MetricResult:
        """
        Analyze difficulty of questions and answers in the dataset.

        Args:
            dataset: List of dataset items

        Returns:
            MetricResult with difficulty analysis
        """
        if not dataset:
            return self._create_result(0.0, {"error": "Empty dataset"})

        questions = [item.get('question', '') for item in dataset if item.get('question')]
        answers = [item.get('answer', '') for item in dataset if item.get('answer')]

        if not questions and not answers:
            return self._create_result(0.0, {"error": "No text content found"})

        # Calculate readability scores
        question_readability = self._calculate_readability(questions)
        answer_readability = self._calculate_readability(answers)

        # Calculate complexity scores
        question_complexity = self._calculate_complexity(questions)
        answer_complexity = self._calculate_complexity(answers)

        # Overall difficulty score (0-1, where 1 is most difficult)
        overall_difficulty = (
            question_readability * 0.3 +
            answer_readability * 0.4 +
            question_complexity * 0.15 +
            answer_complexity * 0.15
        )

        # Calculate alignment with target difficulty
        target_range = self.difficulty_ranges.get(self.target_difficulty, (30, 60))
        target_center = (target_range[0] + target_range[1]) / 2

        # Convert Flesch score to 0-1 difficulty scale (inverted)
        difficulty_score = 1.0 - (overall_difficulty / 100.0)

        # Calculate how well it matches target
        alignment_score = 1.0 - abs(difficulty_score - (target_center / 100.0))

        details = {
            "question_readability": question_readability,
            "answer_readability": answer_readability,
            "question_complexity": question_complexity,
            "answer_complexity": answer_complexity,
            "overall_difficulty": overall_difficulty,
            "target_difficulty": self.target_difficulty,
            "target_range": target_range,
            "alignment_score": alignment_score
        }

        return self._create_result(alignment_score, details)

    def _calculate_readability(self, texts: List[str]) -> float:
        """
        Calculate average Flesch Reading Ease score for texts.

        Returns score 0-100, where higher scores indicate easier text.
        """
        if not texts:
            return 50.0  # Default intermediate score

        scores = []
        for text in texts:
            if text.strip():
                score = self._flesch_reading_ease(text)
                scores.append(score)

        return sum(scores) / len(scores) if scores else 50.0

    def _flesch_reading_ease(self, text: str) -> float:
        """
        Calculate Flesch Reading Ease score for a single text.

        Formula: 206.835 - 1.015 * (words/sentences) - 84.6 * (syllables/words)
        """
        sentences = self._count_sentences(text)
        words = len(text.split())
        syllables = self._count_syllables(text)

        if sentences == 0 or words == 0:
            return 50.0

        # Flesch Reading Ease formula
        score = 206.835 - 1.015 * (words / sentences) - 84.6 * (syllables / words)

        # Clamp to reasonable range
        return max(0, min(100, score))

    def _count_sentences(self, text: str) -> int:
        """Count sentences in text."""
        # Simple sentence counting based on punctuation
        sentences = re.split(r'[.!?]+', text.strip())
        return len([s for s in sentences if s.strip()])

    def _count_syllables(self, text: str) -> int:
        """
        Count syllables in text using a simple heuristic.

        This is a simplified syllable counter. For production use,
        consider using a proper NLP library like NLTK or spaCy.
        """
        words = text.lower().split()
        syllable_count = 0

        for word in words:
            # Remove punctuation
            word = re.sub(r'[^\w]', '', word)
            if not word:
                continue

            # Count vowel groups
            vowels = 'aeiouy'
            count = 0
            prev_vowel = False

            for char in word:
                is_vowel = char in vowels
                if is_vowel and not prev_vowel:
                    count += 1
                prev_vowel = is_vowel

            # Adjust for silent 'e'
            if word.endswith('e') and count > 1:
                count -= 1

            # Ensure at least 1 syllable per word
            syllable_count += max(1, count)

        return syllable_count

    def _calculate_complexity(self, texts: List[str]) -> float:
        """
        Calculate linguistic complexity score.

        Returns score 0-1, where higher scores indicate more complex text.
        """
        if not texts:
            return 0.0

        complexity_scores = []

        for text in texts:
            if not text.strip():
                continue

            words = text.split()
            if not words:
                continue

            # Average word length
            avg_word_length = sum(len(word) for word in words) / len(words)

            # Vocabulary richness (unique words / total words)
            unique_ratio = len(set(words)) / len(words)

            # Sentence complexity (words per sentence)
            sentences = self._count_sentences(text)
            words_per_sentence = len(words) / max(1, sentences)

            # Combine metrics into complexity score
            complexity = (
                (avg_word_length - 4) / 6.0 * 0.4 +  # Normalize word length
                unique_ratio * 0.3 +                   # Vocabulary richness
                (words_per_sentence - 10) / 20.0 * 0.3  # Sentence complexity
            )

            complexity_scores.append(max(0, min(1, complexity)))

        return sum(complexity_scores) / len(complexity_scores) if complexity_scores else 0.0