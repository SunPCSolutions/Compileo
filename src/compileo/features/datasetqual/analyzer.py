"""
Main analyzer class for dataset quality assessment.

Orchestrates quality metrics and provides comprehensive analysis.
"""

from typing import Any, Dict, List, Optional
import logging
import json

from .config import QualityConfig, DEFAULT_CONFIG, MetricConfig
from .metrics import (
    BaseMetric, MetricResult,
    DiversityMetric, BiasMetric, DifficultyMetric, ConsistencyMetric
)

logger = logging.getLogger(__name__)


class QualityAnalyzer:
    """
    Main orchestrator for dataset quality analysis.

    Manages metric execution, result aggregation, and reporting.
    """

    def __init__(self, config: Optional[QualityConfig] = None):
        """
        Initialize the quality analyzer.

        Args:
            config: Quality configuration. Uses DEFAULT_CONFIG if None.
        """
        self.config = config or DEFAULT_CONFIG
        self._metrics = self._initialize_metrics()

        # Set up logging
        self._setup_logging()

    def _initialize_metrics(self) -> Dict[str, BaseMetric]:
        """Initialize all quality metrics based on configuration."""
        metrics = {}

        if self.config.is_metric_enabled('diversity'):
            metrics['diversity'] = DiversityMetric(
                enabled=True,
                threshold=self.config.diversity.threshold,
                min_lexical_diversity=self.config.diversity.min_lexical_diversity,
                min_semantic_diversity=self.config.diversity.min_semantic_diversity
            )

        if self.config.is_metric_enabled('bias'):
            metrics['bias'] = BiasMetric(
                enabled=True,
                threshold=self.config.bias.threshold,
                demographic_keywords=self.config.bias.demographic_keywords
            )

        if self.config.is_metric_enabled('difficulty'):
            metrics['difficulty'] = DifficultyMetric(
                enabled=True,
                threshold=self.config.difficulty.threshold,
                target_difficulty=self.config.difficulty.target_difficulty
            )

        if self.config.is_metric_enabled('consistency'):
            metrics['consistency'] = ConsistencyMetric(
                enabled=True,
                threshold=self.config.consistency.threshold,
                check_factual_consistency=self.config.consistency.check_factual_consistency
            )

        return metrics

    def _setup_logging(self):
        """Set up logging based on configuration."""
        level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        logger.setLevel(level)

        # Add handler if not already present
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

    def _normalize_dataset_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize dataset item to standard Q&A format for quality analysis."""
        # Handle extract mode datasets
        if item.get('source_type') == 'extract_adaptive':
            if isinstance(item.get('answer'), str):
                answer_str = item['answer']
                # Strip code block markers if present
                if answer_str.startswith('```json'):
                    answer_str = answer_str[7:].strip()
                if answer_str.endswith('```'):
                    answer_str = answer_str[:-3].strip()
                try:
                    qa_pairs = json.loads(answer_str)
                    if qa_pairs and isinstance(qa_pairs, list):
                        # Use the first Q&A pair for quality analysis
                        first_qa = qa_pairs[0]
                        return {
                            'question': first_qa.get('question', ''),
                            'answer': first_qa.get('answer', ''),
                            'reasoning': first_qa.get('reasoning', ''),
                            'confidence_score': first_qa.get('confidence_score', 0.5),
                            'reasoning_steps': first_qa.get('reasoning_steps', [])
                        }
                except (json.JSONDecodeError, KeyError, IndexError) as e:
                    logger.warning(f"Failed to parse extract answer: {str(e)}")
                    pass

        # Handle instruction following mode
        if 'instruction' in item:
            return {
                'question': item.get('instruction', ''),
                'answer': item.get('output', ''),
                'reasoning': item.get('reasoning', ''),
                'confidence_score': item.get('confidence_score', 0.5),
                'reasoning_steps': item.get('reasoning_steps', [])
            }

        # Default: assume standard Q&A format
        return item

    def analyze_dataset(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze the quality of a dataset.

        Args:
            dataset: List of dataset items (questions, answers, metadata)

        Returns:
            Comprehensive quality analysis report
        """
        if not self.config.enabled:
            logger.info("Quality analysis is disabled")
            return {"enabled": False, "message": "Quality analysis is disabled"}

        if not dataset:
            logger.warning("Empty dataset provided for analysis")
            return {"error": "Empty dataset", "enabled": True}

        logger.info(f"Starting quality analysis on dataset with {len(dataset)} items")
        logger.info(f"Enabled metrics: {list(self._metrics.keys())}")

        # Filter out non-dict items first to prevent normalization errors
        dataset = [item for item in dataset if isinstance(item, dict)]
        logger.info(f"Filtered dataset to {len(dataset)} dict items")

        # Normalize dataset to standard Q&A format
        normalized_dataset = [self._normalize_dataset_item(item) for item in dataset]
        logger.info(f"Normalized dataset to {len(normalized_dataset)} items")

        # Execute all enabled metrics
        results = {}
        for metric_name, metric in self._metrics.items():
            try:
                logger.debug(f"Running {metric_name} analysis")
                result = metric.analyze(normalized_dataset)
                results[metric_name] = result
                logger.info(f"{metric_name} score: {result.score:.3f}")
            except Exception as e:
                logger.error(f"Error running {metric_name} metric: {e}")
                results[metric_name] = MetricResult(
                    name=metric_name,
                    score=0.0,
                    details={"error": str(e)}
                )

        # Aggregate results
        summary = self._create_summary(results)

        report = {
            "enabled": True,
            "dataset_size": len(dataset),
            "metrics_run": list(results.keys()),
            "results": {k: self._result_to_dict(v) for k, v in results.items()},
            "summary": summary,
            "config": self._config_summary()
        }

        logger.info(f"Quality analysis complete. Overall score: {summary.get('overall_score', 'N/A')}")
        return report

    def _result_to_dict(self, result: MetricResult) -> Dict[str, Any]:
        """Convert MetricResult to dictionary for JSON serialization."""
        return {
            "name": result.name,
            "score": result.score,
            "threshold": result.threshold,
            "passed": result.passed,
            "details": result.details,
            "metadata": result.metadata
        }

    def _create_summary(self, results: Dict[str, MetricResult]) -> Dict[str, Any]:
        """Create summary statistics from all metric results."""
        if not results:
            return {"overall_score": 0.0, "passed": False}

        # Calculate weighted overall score
        total_weight = 0
        weighted_sum = 0

        passed_metrics = 0
        failed_metrics = 0
        issues = []

        for metric_name, result in results.items():
            weight = getattr(self.config, metric_name, MetricConfig()).weight
            weighted_sum += result.score * weight
            total_weight += weight

            if result.passed is True:
                passed_metrics += 1
            elif result.passed is False:
                failed_metrics += 1
                issues.append(f"{metric_name}: {result.score:.3f}")

            # Collect issues from individual metrics
            if "issues" in result.details:
                issues.extend(result.details["issues"])

        overall_score = weighted_sum / total_weight if total_weight > 0 else 0.0

        # Determine overall pass/fail
        if self.config.fail_on_any_failure:
            passed = failed_metrics == 0
        else:
            # Pass if overall score meets threshold
            passed = overall_score >= 0.5  # Default threshold

        summary = {
            "overall_score": round(overall_score, 3),
            "passed": passed,
            "passed_metrics": passed_metrics,
            "failed_metrics": failed_metrics,
            "total_metrics": len(results),
            "issues": issues[:10]  # Limit to first 10 issues
        }

        return summary

    def _config_summary(self) -> Dict[str, Any]:
        """Create summary of current configuration."""
        return {
            "enabled": self.config.enabled,
            "enabled_metrics": self.config.get_enabled_metrics(),
            "fail_on_any_failure": self.config.fail_on_any_failure,
            "output_format": self.config.output_format
        }

    def is_enabled(self) -> bool:
        """Check if quality analysis is enabled."""
        return self.config.enabled

    def get_enabled_metrics(self) -> List[str]:
        """Get list of enabled metric names."""
        return list(self._metrics.keys())