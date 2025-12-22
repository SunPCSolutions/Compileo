"""
Integration hooks for the benchmarking module.
"""

from typing import Dict, List, Any, Optional, Callable
import logging
from .config import BenchmarkingModuleConfig
from .evaluator import BenchmarkEvaluator
from .tracker import PerformanceTracker
from .correlation import CorrelationAnalyzer
from .suites.base import BenchmarkResult
from src.compileo.core.settings import backend_settings

logger = logging.getLogger(__name__)


class BenchmarkingHooks:
    """
    Hooks for integrating benchmarking with other modules.

    Provides optional integration points for dataset generation,
    model training, and evaluation workflows.
    """

    def __init__(self, config: BenchmarkingModuleConfig):
        self.config = config
        self.evaluator = BenchmarkEvaluator(config.dict() if config else {})
        self.tracker = PerformanceTracker(config.tracking.storage_path if config else None)
        self.correlation_analyzer = CorrelationAnalyzer(config.correlation.dict() if config else {})

        # Hook registry
        self.pre_generation_hooks: List[Callable] = []
        self.post_generation_hooks: List[Callable] = []
        self.pre_evaluation_hooks: List[Callable] = []
        self.post_evaluation_hooks: List[Callable] = []

    def register_pre_generation_hook(self, hook: Callable):
        """Register a hook to run before dataset generation."""
        self.pre_generation_hooks.append(hook)

    def register_post_generation_hook(self, hook: Callable):
        """Register a hook to run after dataset generation."""
        self.post_generation_hooks.append(hook)

    def register_pre_evaluation_hook(self, hook: Callable):
        """Register a hook to run before model evaluation."""
        self.pre_evaluation_hooks.append(hook)

    def register_post_evaluation_hook(self, hook: Callable):
        """Register a hook to run after model evaluation."""
        self.post_evaluation_hooks.append(hook)

    def on_dataset_generation_start(self, generation_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Hook called when dataset generation starts.

        Args:
            generation_config: Dataset generation configuration

        Returns:
            Modified configuration
        """
        if not self.config.enabled:
            return generation_config

        logger.info("Benchmarking hooks: Dataset generation started")

        # Run pre-generation hooks
        for hook in self.pre_generation_hooks:
            try:
                generation_config = hook(generation_config)
            except Exception as e:
                logger.error(f"Pre-generation hook failed: {e}")

        return generation_config

    def on_dataset_generation_complete(self, dataset_info: Dict[str, Any],
                                     quality_metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Hook called when dataset generation completes.

        Args:
            dataset_info: Information about the generated dataset
            quality_metrics: Quality metrics from datasetqual module

        Returns:
            Updated dataset info
        """
        if not self.config.enabled:
            return dataset_info

        logger.info("Benchmarking hooks: Dataset generation completed")

        # Store quality metrics for correlation analysis
        if quality_metrics and self.config.correlation.enabled:
            dataset_info['quality_metrics'] = quality_metrics

        # Run post-generation hooks
        for hook in self.post_generation_hooks:
            try:
                dataset_info = hook(dataset_info, quality_metrics)
            except Exception as e:
                logger.error(f"Post-generation hook failed: {e}")

        return dataset_info

    def on_model_evaluation_start(self, model_info: Dict[str, Any],
                                evaluation_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Hook called when model evaluation starts.

        Args:
            model_info: Information about the model being evaluated
            evaluation_config: Evaluation configuration

        Returns:
            Modified evaluation configuration
        """
        if not self.config.enabled or not self.config.run_benchmarks:
            return evaluation_config

        logger.info(f"Benchmarking hooks: Model evaluation started for {model_info.get('name', 'unknown')}")

        # Run pre-evaluation hooks
        for hook in self.pre_evaluation_hooks:
            try:
                evaluation_config = hook(model_info, evaluation_config)
            except Exception as e:
                logger.error(f"Pre-evaluation hook failed: {e}")

        return evaluation_config

    def on_model_evaluation_complete(self, model_info: Dict[str, Any],
                                   evaluation_results: Dict[str, Any],
                                   dataset_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Hook called when model evaluation completes.

        Args:
            model_info: Information about the evaluated model
            evaluation_results: Results from model evaluation
            dataset_info: Information about the dataset used

        Returns:
            Updated evaluation results
        """
        if not self.config.enabled or not self.config.run_benchmarks:
            return evaluation_results

        logger.info(f"Benchmarking hooks: Model evaluation completed for {model_info.get('name', 'unknown')}")

        try:
            # Run benchmarks if requested
            benchmark_results = self._run_benchmarks(model_info, evaluation_results)

            # Track performance
            if self.config.tracking.enabled:
                self._track_performance(model_info, benchmark_results)

            # Analyze correlations
            if self.config.correlation.enabled and dataset_info:
                self._analyze_correlations(dataset_info, benchmark_results)

            # Add benchmark results to evaluation results
            evaluation_results['benchmarking'] = {
                'results': benchmark_results,
                # 'summary': self.evaluator.get_summary_report(benchmark_results)  # TODO: Convert to BenchmarkResult objects
            }

        except Exception as e:
            logger.error(f"Benchmarking evaluation failed: {e}")
            evaluation_results['benchmarking'] = {'error': str(e)}

        # Run post-evaluation hooks
        for hook in self.post_evaluation_hooks:
            try:
                evaluation_results = hook(model_info, evaluation_results)
            except Exception as e:
                logger.error(f"Post-evaluation hook failed: {e}")

        return evaluation_results

    def _run_benchmarks(self, model_info: Dict[str, Any],
                       evaluation_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Run configured benchmarks."""
        # This is a placeholder - actual implementation would load the model
        # and run the benchmarks using the evaluator


        # Get AI configuration from environment variables (injected by RQ worker)
        import os
        ai_config = {
            'gemini_api_key': os.getenv('GOOGLE_API_KEY'),
            'grok_api_key': os.getenv('GROK_API_KEY'),
            'ollama_available': backend_settings.get_setting('ollama_available', True),
            'model_name': model_info.get('name', 'unknown')
        }

        logger.info(f"Running benchmarks for model: {ai_config['model_name']}")

        # Run benchmarks using the evaluator
        benchmark_results = self.evaluator.run_benchmarks(
            ai_config=ai_config,
            benchmark_names=self.config.benchmark.suites if hasattr(self.config, 'benchmark') and self.config.benchmark.suites else None
        )

        # Convert BenchmarkResult objects to dictionaries for compatibility
        results_dict = []
        for result in benchmark_results:
            results_dict.append({
                'benchmark_name': result.benchmark_name,
                'task_name': result.task_name,
                'metrics': result.metrics,
                'predictions': result.predictions,
                'labels': result.labels
            })

        return results_dict

    def _track_performance(self, model_info: Dict[str, Any], benchmark_results: List[Dict[str, Any]]):
        """Track model performance."""
        model_name = model_info.get('name', 'unknown_model')

        # Convert results to the format expected by tracker
        performance_data = {}
        for result in benchmark_results:
            key = f"{result['benchmark_name']}_{result['task_name']}"
            performance_data[key] = result

        self.tracker.add_record(model_name, performance_data, model_info)

    def _analyze_correlations(self, dataset_info: Dict[str, Any], benchmark_results: List[Dict[str, Any]]):
        """Analyze correlations between quality metrics and performance."""
        quality_metrics = dataset_info.get('quality_metrics', {})

        if quality_metrics:
            # Convert benchmark results to the format expected by correlation analyzer
            performance_results = {}
            for result in benchmark_results:
                key = f"{result['benchmark_name']}_{result['task_name']}"
                performance_results[key] = result

            correlation_analysis = self.correlation_analyzer.analyze_correlations(
                quality_metrics, performance_results
            )

            logger.info(f"Correlation analysis completed: {len(correlation_analysis.get('significant_factors', []))} significant factors found")

            # Store correlation results in dataset_info for reporting
            dataset_info['correlation_analysis'] = correlation_analysis


# Global hooks instance
_benchmarking_hooks: Optional[BenchmarkingHooks] = None


def get_benchmarking_hooks(config: Optional[BenchmarkingModuleConfig] = None) -> Optional[BenchmarkingHooks]:
    """Get the global benchmarking hooks instance."""
    global _benchmarking_hooks

    if config and config.enabled:
        if _benchmarking_hooks is None:
            _benchmarking_hooks = BenchmarkingHooks(config)
        return _benchmarking_hooks

    return None


def initialize_benchmarking_hooks(config: BenchmarkingModuleConfig):
    """Initialize the global benchmarking hooks."""
    global _benchmarking_hooks
    _benchmarking_hooks = BenchmarkingHooks(config)
    logger.info("Benchmarking hooks initialized")