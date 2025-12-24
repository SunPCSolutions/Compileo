"""
Configuration management for the multi-stage classification pipeline.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import json
import os
from ...core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the multi-stage classification pipeline."""

    # Classifier settings
    classifiers: List[str] = field(default_factory=lambda: ['ollama', 'gemini', 'grok'])
    primary_classifier: str = 'ollama'

    # Confidence and agreement thresholds
    confidence_threshold: float = 0.7
    min_agreement: float = 0.6
    max_confidence_bonus: float = 0.2  # Bonus for refinement stage

    # Pipeline behavior
    enable_coarse_stage: bool = True
    enable_validation_stage: bool = True  # Renamed from enable_fine_stage
    enable_cross_validation: bool = True
    enable_ensemble_voting: bool = True

    # Performance settings
    max_workers: int = 3  # For parallel classification
    timeout_seconds: int = 30  # Timeout for individual classifications

    # Output settings
    include_stage_results: bool = True
    include_metadata: bool = True
    output_format: str = 'detailed'  # 'simple' or 'detailed'

    # API keys and credentials
    api_keys: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()

    def _validate_config(self):
        """Validate the configuration parameters."""
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")

        if not 0 <= self.min_agreement <= 1:
            raise ValueError("min_agreement must be between 0 and 1")

        if not 0 <= self.max_confidence_bonus <= 1:
            raise ValueError("max_confidence_bonus must be between 0 and 1")

        valid_classifiers = ['ollama', 'gemini', 'grok', 'openai']
        for classifier in self.classifiers:
            if classifier not in valid_classifiers:
                raise ValueError(f"Invalid classifier: {classifier}. Must be one of {valid_classifiers}")

        if self.primary_classifier not in self.classifiers:
            raise ValueError("primary_classifier must be in the classifiers list")

        if self.max_workers < 1:
            raise ValueError("max_workers must be at least 1")

        if self.timeout_seconds < 1:
            raise ValueError("timeout_seconds must be at least 1")

        valid_formats = ['simple', 'detailed']
        if self.output_format not in valid_formats:
            raise ValueError(f"output_format must be one of {valid_formats}")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PipelineConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)

    @classmethod
    def from_json_file(cls, filepath: str) -> 'PipelineConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_env(cls, prefix: str = 'CLASSIFY_') -> 'PipelineConfig':
        """Load configuration from environment variables."""
        config_dict = {}

        # Map environment variables to config fields
        env_mappings = {
            f'{prefix}CLASSIFIERS': ('classifiers', lambda x: x.split(',')),
            f'{prefix}PRIMARY_CLASSIFIER': ('primary_classifier', str),
            f'{prefix}CONFIDENCE_THRESHOLD': ('confidence_threshold', float),
            f'{prefix}MIN_AGREEMENT': ('min_agreement', float),
            f'{prefix}MAX_CONFIDENCE_BONUS': ('max_confidence_bonus', float),
            f'{prefix}ENABLE_COARSE_STAGE': ('enable_coarse_stage', lambda x: x.lower() == 'true'),
            f'{prefix}ENABLE_VALIDATION_STAGE': ('enable_validation_stage', lambda x: x.lower() == 'true'),  # Renamed from enable_fine_stage
            f'{prefix}ENABLE_CROSS_VALIDATION': ('enable_cross_validation', lambda x: x.lower() == 'true'),
            f'{prefix}ENABLE_ENSEMBLE_VOTING': ('enable_ensemble_voting', lambda x: x.lower() == 'true'),
            f'{prefix}MAX_WORKERS': ('max_workers', int),
            f'{prefix}TIMEOUT_SECONDS': ('timeout_seconds', int),
            f'{prefix}INCLUDE_STAGE_RESULTS': ('include_stage_results', lambda x: x.lower() == 'true'),
            f'{prefix}INCLUDE_METADATA': ('include_metadata', lambda x: x.lower() == 'true'),
            f'{prefix}OUTPUT_FORMAT': ('output_format', str),
        }

        for env_var, (field_name, converter) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    config_dict[field_name] = converter(value)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid value for {env_var}: {value} ({e})")

        # Handle API keys separately
        api_keys = {}
        for key in ['GEMINI_API_KEY', 'GROK_API_KEY', 'OLLAMA_API_KEY']:
            value = os.getenv(key)
            if value:
                api_keys[key.lower().replace('_api_key', '')] = value

        if api_keys:
            config_dict['api_keys'] = api_keys

        return cls(**config_dict) if config_dict else cls()

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'classifiers': self.classifiers,
            'primary_classifier': self.primary_classifier,
            'confidence_threshold': self.confidence_threshold,
            'min_agreement': self.min_agreement,
            'max_confidence_bonus': self.max_confidence_bonus,
            'enable_coarse_stage': self.enable_coarse_stage,
            'enable_validation_stage': self.enable_validation_stage,  # Renamed from enable_fine_stage
            'enable_cross_validation': self.enable_cross_validation,
            'enable_ensemble_voting': self.enable_ensemble_voting,
            'max_workers': self.max_workers,
            'timeout_seconds': self.timeout_seconds,
            'include_stage_results': self.include_stage_results,
            'include_metadata': self.include_metadata,
            'output_format': self.output_format,
            'api_keys': {k: '***' for k in self.api_keys.keys()}  # Mask API keys
        }

    def save_to_json(self, filepath: str):
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def get_api_key(self, classifier: str) -> Optional[str]:
        """Get API key for a specific classifier."""
        return self.api_keys.get(classifier)

    def set_api_key(self, classifier: str, api_key: str):
        """Set API key for a specific classifier."""
        self.api_keys[classifier] = api_key

    def is_stage_enabled(self, stage: str) -> bool:
        """Check if a pipeline stage is enabled."""
        stage_map = {
            'coarse': self.enable_coarse_stage,
            'validation': self.enable_validation_stage,  # Renamed from fine
            'cross_validation': self.enable_cross_validation,
            'ensemble': self.enable_ensemble_voting
        }
        return stage_map.get(stage, False)

    def get_effective_classifiers(self) -> List[str]:
        """Get the list of classifiers that should be used based on configuration."""
        if not self.enable_coarse_stage and not self.enable_validation_stage:
            return [self.primary_classifier]
        return self.classifiers


def create_config_from_selection(
    initial_classifier: str,
    enable_validation: bool,
    validation_classifier: Optional[str] = None,
    confidence_threshold: float = 0.7,
    api_keys: Dict[str, str] = {}
) -> PipelineConfig:
    """Create a PipelineConfig dynamically based on user selection."""
    
    classifiers = [initial_classifier]
    if enable_validation and validation_classifier:
        classifiers.append(validation_classifier)

    return PipelineConfig(
        classifiers=classifiers,
        primary_classifier=initial_classifier,
        enable_coarse_stage=True,
        enable_validation_stage=enable_validation,
        confidence_threshold=confidence_threshold,
        api_keys=api_keys
    )
def get_default_config(preset: str = 'balanced') -> PipelineConfig:
    """
    Get a default PipelineConfig based on preset configurations.

    Args:
        preset: Configuration preset ('fast', 'balanced', 'accurate', 'comprehensive')

    Returns:
        PipelineConfig with preset settings
    """
    presets = {
        'fast': {
            'classifiers': ['ollama'],
            'primary_classifier': 'ollama',
            'confidence_threshold': 0.5,
            'enable_coarse_stage': True,
            'enable_validation_stage': False,
            'enable_cross_validation': False,
            'enable_ensemble_voting': False,
            'max_workers': 1,
            'timeout_seconds': 15
        },
        'balanced': {
            'classifiers': ['ollama', 'gemini'],
            'primary_classifier': 'ollama',
            'confidence_threshold': 0.6,
            'enable_coarse_stage': True,
            'enable_validation_stage': True,
            'enable_cross_validation': True,
            'enable_ensemble_voting': True,
            'max_workers': 2,
            'timeout_seconds': 25
        },
        'accurate': {
            'classifiers': ['gemini', 'grok'],
            'primary_classifier': 'gemini',
            'confidence_threshold': 0.7,
            'enable_coarse_stage': True,
            'enable_validation_stage': True,
            'enable_cross_validation': True,
            'enable_ensemble_voting': True,
            'max_workers': 3,
            'timeout_seconds': 30
        },
        'comprehensive': {
            'classifiers': ['ollama', 'gemini', 'grok'],
            'primary_classifier': 'gemini',
            'confidence_threshold': 0.8,
            'enable_coarse_stage': True,
            'enable_validation_stage': True,
            'enable_cross_validation': True,
            'enable_ensemble_voting': True,
            'max_workers': 3,
            'timeout_seconds': 45
        }
    }

    if preset not in presets:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(presets.keys())}")

    return PipelineConfig(**presets[preset])