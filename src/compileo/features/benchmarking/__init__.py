"""
Benchmarking module for AI model performance evaluation.

This optional module provides enterprise-grade performance benchmarking
capabilities for evaluating AI models on generated datasets.
"""

__version__ = "0.1.0"

from .evaluator import BenchmarkEvaluator
from .tracker import PerformanceTracker
from .correlation import CorrelationAnalyzer
from .hooks import BenchmarkingHooks, get_benchmarking_hooks
from .config import BenchmarkingModuleConfig, get_default_config

__all__ = [
    "BenchmarkEvaluator",
    "PerformanceTracker",
    "CorrelationAnalyzer",
    "BenchmarkingHooks",
    "get_benchmarking_hooks",
    "BenchmarkingModuleConfig",
    "get_default_config"
]