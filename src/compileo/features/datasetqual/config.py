"""
Configuration models for dataset quality metrics.

Uses Pydantic for validation and type safety.
"""

from typing import Dict, Optional, List
from pydantic import BaseModel, Field, validator


class MetricConfig(BaseModel):
    """Configuration for a single quality metric."""
    enabled: bool = Field(default=True, description="Whether this metric is enabled")
    threshold: Optional[float] = Field(default=None, description="Pass/fail threshold for the metric")
    weight: float = Field(default=1.0, ge=0.0, le=2.0, description="Weight in overall quality score")


class DiversityConfig(MetricConfig):
    """Configuration for diversity metric."""
    min_lexical_diversity: float = Field(default=0.3, ge=0.0, le=1.0)
    min_semantic_diversity: float = Field(default=0.4, ge=0.0, le=1.0)


class BiasConfig(MetricConfig):
    """Configuration for bias metric."""
    demographic_keywords: Optional[Dict[str, List[str]]] = Field(
        default=None,
        description="Custom demographic keywords for bias detection"
    )


class DifficultyConfig(MetricConfig):
    """Configuration for difficulty metric."""
    target_difficulty: str = Field(
        default="intermediate",
        description="Target difficulty level"
    )

    @validator('target_difficulty')
    def validate_target_difficulty(cls, v):
        allowed = ['easy', 'intermediate', 'hard']
        if v not in allowed:
            raise ValueError(f'target_difficulty must be one of {allowed}')
        return v


class ConsistencyConfig(MetricConfig):
    """Configuration for consistency metric."""
    check_factual_consistency: bool = Field(
        default=True,
        description="Whether to perform factual consistency checks"
    )


class QualityConfig(BaseModel):
    """Main configuration for dataset quality analysis."""
    enabled: bool = Field(default=False, description="Whether quality analysis is enabled globally")

    # Individual metric configurations
    diversity: DiversityConfig = Field(default_factory=DiversityConfig)
    bias: BiasConfig = Field(default_factory=BiasConfig)
    difficulty: DifficultyConfig = Field(default_factory=DifficultyConfig)
    consistency: ConsistencyConfig = Field(default_factory=ConsistencyConfig)

    # Global settings
    fail_on_any_failure: bool = Field(
        default=False,
        description="Fail quality check if any individual metric fails"
    )
    output_format: str = Field(
        default="json",
        description="Format for quality reports"
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level for quality analysis"
    )

    @validator('output_format')
    def validate_output_format(cls, v):
        allowed = ['json', 'text', 'markdown']
        if v not in allowed:
            raise ValueError(f'output_format must be one of {allowed}')
        return v

    @validator('log_level')
    def validate_log_level(cls, v):
        allowed = ['DEBUG', 'INFO', 'WARNING', 'ERROR']
        if v not in allowed:
            raise ValueError(f'log_level must be one of {allowed}')
        return v

    @validator('enabled')
    def validate_module_enabled(cls, v):
        """Ensure module can be disabled."""
        return v

    def is_metric_enabled(self, metric_name: str) -> bool:
        """Check if a specific metric is enabled."""
        if not self.enabled:
            return False

        metric_config = getattr(self, metric_name, None)
        if metric_config and hasattr(metric_config, 'enabled'):
            return metric_config.enabled
        return False

    def get_metric_threshold(self, metric_name: str) -> Optional[float]:
        """Get threshold for a specific metric."""
        metric_config = getattr(self, metric_name, None)
        if metric_config and hasattr(metric_config, 'threshold'):
            return metric_config.threshold
        return None

    def get_enabled_metrics(self) -> List[str]:
        """Get list of enabled metric names."""
        if not self.enabled:
            return []

        enabled = []
        for metric_name in ['diversity', 'bias', 'difficulty', 'consistency']:
            if self.is_metric_enabled(metric_name):
                enabled.append(metric_name)
        return enabled


# Default configuration
DEFAULT_CONFIG = QualityConfig(
    enabled=False,  # Disabled by default for optional nature
    diversity=DiversityConfig(
        enabled=True,
        threshold=0.6,
        min_lexical_diversity=0.3,
        min_semantic_diversity=0.4
    ),
    bias=BiasConfig(
        enabled=True,
        threshold=0.3,  # Lower scores are better for bias
    ),
    difficulty=DifficultyConfig(
        enabled=True,
        threshold=0.7,
        target_difficulty="intermediate"
    ),
    consistency=ConsistencyConfig(
        enabled=True,
        threshold=0.8,
    ),
    fail_on_any_failure=False,
    output_format="json",
    log_level="INFO"
)