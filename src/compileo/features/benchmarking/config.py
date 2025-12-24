"""
Configuration models for the benchmarking module.
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field


class BenchmarkConfig(BaseModel):
    """Configuration for benchmark suites."""

    enabled: bool = Field(default=False, description="Enable benchmarking module")
    suites: List[str] = Field(default_factory=lambda: ["glue"], description="Benchmark suites to run")
    tasks: Optional[Dict[str, List[str]]] = Field(default=None, description="Specific tasks per suite")
    model_path: Optional[str] = Field(default=None, description="Path to model for evaluation")
    model_type: str = Field(default="auto", description="Model type (huggingface, local, etc.)")
    batch_size: int = Field(default=32, description="Batch size for evaluation")
    max_samples: Optional[int] = Field(default=None, description="Maximum samples per task")


class MetricsConfig(BaseModel):
    """Configuration for performance metrics."""

    enabled_metrics: List[str] = Field(
        default_factory=lambda: ["accuracy", "f1", "precision", "recall"],
        description="Metrics to calculate"
    )
    generation_metrics: List[str] = Field(
        default_factory=lambda: ["bleu", "rouge", "meteor"],
        description="Text generation metrics"
    )
    custom_metrics: List[str] = Field(default_factory=list, description="Custom metric configurations")


class TrackingConfig(BaseModel):
    """Configuration for performance tracking."""

    enabled: bool = Field(default=True, description="Enable performance tracking")
    storage_path: str = Field(default="benchmark_results", description="Path to store results")
    track_history: bool = Field(default=True, description="Track performance history")
    compare_models: bool = Field(default=True, description="Enable model comparison")
    degradation_threshold: float = Field(default=0.05, description="Performance degradation threshold")


class CorrelationConfig(BaseModel):
    """Configuration for quality-performance correlation analysis."""

    enabled: bool = Field(default=True, description="Enable correlation analysis")
    quality_sources: List[str] = Field(
        default_factory=lambda: ["datasetqual"],
        description="Sources of quality metrics"
    )
    significance_threshold: float = Field(default=0.05, description="Statistical significance threshold")
    min_correlation_strength: float = Field(default=0.3, description="Minimum correlation strength to report")


class BenchmarkingModuleConfig(BaseModel):
    """Main configuration for the benchmarking module."""

    enabled: bool = Field(default=False, description="Master switch for benchmarking module")

    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    correlation: CorrelationConfig = Field(default_factory=CorrelationConfig)

    # CLI override fields
    run_benchmarks: bool = Field(default=False, description="CLI flag to run benchmarks")
    benchmark_suite: Optional[str] = Field(default=None, description="CLI specified benchmark suite")
    output_path: Optional[str] = Field(default=None, description="CLI specified output path")

    class Config:
        """Pydantic configuration."""
        validate_assignment = True
        extra = "allow"


def get_default_config() -> BenchmarkingModuleConfig:
    """Get default benchmarking configuration."""
    return BenchmarkingModuleConfig()


def load_config_from_dict(config_dict: Dict[str, Any]) -> BenchmarkingModuleConfig:
    """Load configuration from dictionary."""
    return BenchmarkingModuleConfig(**config_dict)