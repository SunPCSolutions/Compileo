"""
Main benchmark evaluator orchestrator.
"""

from typing import Dict, List, Any, Optional, Union
import logging
from datetime import datetime
from .suites import BaseBenchmark
from .metrics import BaseMetric
from .suites.base import BenchmarkResult

logger = logging.getLogger(__name__)


class BenchmarkEvaluator:
    """
    Main orchestrator for running AI model benchmarks.

    This class coordinates benchmark suites, metrics calculation,
    and result aggregation for comprehensive model evaluation.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.benchmarks: Dict[str, BaseBenchmark] = {}
        self.metrics: Dict[str, BaseMetric] = {}
        self.results: List[BenchmarkResult] = []

    def register_benchmark(self, name: str, benchmark: BaseBenchmark):
        """Register a benchmark suite."""
        self.benchmarks[name] = benchmark
        logger.info(f"Registered benchmark: {name}")

    def register_metric(self, name: str, metric: BaseMetric):
        """Register a performance metric."""
        self.metrics[name] = metric
        logger.info(f"Registered metric: {name}")

    def run_benchmarks(self, ai_config: Dict[str, Any], benchmark_names: Optional[List[str]] = None,
                      **kwargs) -> List[BenchmarkResult]:
        """
        Run benchmarks using AI provider configuration.

        Args:
            ai_config: AI provider configuration (gemini_api_key, grok_api_key, ollama_available, etc.)
            benchmark_names: List of benchmark names to run (None for all)
            **kwargs: Additional evaluation parameters

        Returns:
            List of benchmark results
        """
        if benchmark_names is None:
            benchmark_names = list(self.benchmarks.keys())

        results = []

        for benchmark_name in benchmark_names:
            if benchmark_name not in self.benchmarks:
                logger.warning(f"Benchmark not found: {benchmark_name}")
                continue

            benchmark = self.benchmarks[benchmark_name]
            if not benchmark.is_available():
                logger.warning(f"Benchmark not available: {benchmark_name}")
                continue

            try:
                logger.info(f"Running benchmark: {benchmark_name}")
                benchmark_results = benchmark.evaluate(ai_config, **kwargs)
                results.extend(benchmark_results)

                # Store results
                self.results.extend(benchmark_results)

            except Exception as e:
                logger.error(f"Failed to run benchmark {benchmark_name}: {e}")
                continue

        logger.info(f"Completed {len(results)} benchmark evaluations")
        return results

    def calculate_metrics(self, benchmark_results: List[BenchmarkResult],
                         metric_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Calculate performance metrics from benchmark results.

        Args:
            benchmark_results: Results from benchmark evaluations
            metric_names: List of metric names to calculate (None for all)

        Returns:
            Dictionary of metric results
        """
        if metric_names is None:
            metric_names = list(self.metrics.keys())

        aggregated_results = {}

        for result in benchmark_results:
            benchmark_key = f"{result.benchmark_name}_{result.task_name}"

            if benchmark_key not in aggregated_results:
                aggregated_results[benchmark_key] = {
                    'benchmark': result.benchmark_name,
                    'task': result.task_name,
                    'predictions': result.predictions or [],
                    'labels': result.labels or [],
                    'metrics': {}
                }

            # Aggregate predictions and labels
            if result.predictions:
                aggregated_results[benchmark_key]['predictions'].extend(result.predictions)
            if result.labels:
                aggregated_results[benchmark_key]['labels'].extend(result.labels)

            # Merge metrics
            aggregated_results[benchmark_key]['metrics'].update(result.metrics)

        # Calculate additional metrics
        for benchmark_key, data in aggregated_results.items():
            predictions = data['predictions']
            labels = data['labels']

            if predictions and labels and len(predictions) == len(labels):
                for metric_name in metric_names:
                    if metric_name not in self.metrics:
                        continue

                    metric = self.metrics[metric_name]
                    if not metric.is_available():
                        continue

                    try:
                        result = metric.calculate(predictions, labels)
                        data['metrics'][result.metric_name] = result.value
                    except Exception as e:
                        logger.error(f"Failed to calculate metric {metric_name}: {e}")

        return aggregated_results

    def get_summary_report(self, results: Optional[List[BenchmarkResult]] = None) -> Dict[str, Any]:
        """
        Generate a summary report of benchmark results.

        Args:
            results: Benchmark results to summarize (None uses stored results)

        Returns:
            Summary report dictionary
        """
        if results is None:
            results = self.results

        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_evaluations': len(results),
            'benchmarks_run': set(),
            'tasks_run': set(),
            'metrics_calculated': set(),
            'performance_summary': {}
        }

        for result in results:
            summary['benchmarks_run'].add(result.benchmark_name)
            summary['tasks_run'].add(result.task_name)
            summary['metrics_calculated'].update(result.metrics.keys())

            # Aggregate metrics by benchmark
            benchmark_key = result.benchmark_name
            if benchmark_key not in summary['performance_summary']:
                summary['performance_summary'][benchmark_key] = {}

            for metric_name, value in result.metrics.items():
                if metric_name not in summary['performance_summary'][benchmark_key]:
                    summary['performance_summary'][benchmark_key][metric_name] = []
                summary['performance_summary'][benchmark_key][metric_name].append(value)

        # Calculate averages
        for benchmark_key, metrics in summary['performance_summary'].items():
            for metric_name, values in metrics.items():
                if values:
                    metrics[metric_name] = {
                        'mean': sum(values) / len(values),
                        'min': min(values),
                        'max': max(values),
                        'count': len(values)
                    }

        # Convert sets to lists for JSON serialization
        summary['benchmarks_run'] = list(summary['benchmarks_run'])
        summary['tasks_run'] = list(summary['tasks_run'])
        summary['metrics_calculated'] = list(summary['metrics_calculated'])

        return summary

    def clear_results(self):
        """Clear stored results."""
        self.results.clear()

    def is_enabled(self) -> bool:
        """Check if benchmarking is enabled in configuration."""
        return self.config.get('enabled', False)