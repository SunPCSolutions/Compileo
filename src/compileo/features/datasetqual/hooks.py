"""
Integration hooks for dataset quality analysis.

Provides optional hooks that can be integrated into datasetgen workflow.
"""

from typing import Any, Dict, List, Optional, Callable
import logging

from .analyzer import QualityAnalyzer
from .config import QualityConfig

logger = logging.getLogger(__name__)


class QualityHooks:
    """
    Optional hooks for integrating quality analysis into dataset generation.

    These hooks can be registered with datasetgen to provide real-time
    quality monitoring without disrupting the core workflow.
    """

    def __init__(self, config: Optional[QualityConfig] = None):
        """
        Initialize quality hooks.

        Args:
            config: Quality configuration. If None, quality analysis is disabled.
        """
        self.config = config
        self.analyzer = QualityAnalyzer(config) if config and config.enabled else None
        self._callbacks = {
            'post_generation': [],
            'quality_check': []
        }

    def is_enabled(self) -> bool:
        """Check if quality hooks are enabled."""
        return self.analyzer is not None and self.config is not None and self.config.enabled

    def register_callback(self, hook_type: str, callback: Callable):
        """
        Register a callback for a specific hook type.

        Args:
            hook_type: Type of hook ('post_generation', 'quality_check')
            callback: Function to call when hook is triggered
        """
        if hook_type in self._callbacks:
            self._callbacks[hook_type].append(callback)
            logger.debug(f"Registered callback for {hook_type} hook")
        else:
            logger.warning(f"Unknown hook type: {hook_type}")

    def post_generation_hook(self, dataset: List[Dict[str, Any]],
                           metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Hook called after dataset generation is complete.

        Args:
            dataset: Generated dataset
            metadata: Additional metadata about generation

        Returns:
            Quality analysis results (empty dict if disabled)
        """
        if not self.is_enabled():
            return {}

        logger.info("Running post-generation quality analysis")

        try:
            results = self.analyzer.analyze_dataset(dataset)

            # Trigger callbacks
            for callback in self._callbacks['post_generation']:
                try:
                    callback(results, metadata)
                except Exception as e:
                    logger.error(f"Error in post_generation callback: {e}")

            # Trigger quality check callbacks
            if not results.get('summary', {}).get('passed', True):
                for callback in self._callbacks['quality_check']:
                    try:
                        callback(results, metadata)
                    except Exception as e:
                        logger.error(f"Error in quality_check callback: {e}")

            return results

        except Exception as e:
            logger.error(f"Error in post_generation_hook: {e}")
            return {"error": str(e)}

    def quality_check_hook(self, dataset: List[Dict[str, Any]],
                          threshold: Optional[float] = None) -> bool:
        """
        Quick quality check hook that returns pass/fail.

        Args:
            dataset: Dataset to check
            threshold: Optional custom threshold

        Returns:
            True if quality check passes, False otherwise
        """
        if not self.is_enabled():
            return True  # Pass by default when disabled

        try:
            results = self.analyzer.analyze_dataset(dataset)
            summary = results.get('summary', {})

            # Use custom threshold if provided
            check_threshold = threshold if threshold is not None else 0.5

            passed = summary.get('overall_score', 0.0) >= check_threshold

            if not passed:
                logger.warning(f"Quality check failed: score {summary.get('overall_score', 0.0)} < {check_threshold}")

            return passed

        except Exception as e:
            logger.error(f"Error in quality_check_hook: {e}")
            return False  # Fail on error

    def get_quality_report(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get detailed quality report for a dataset.

        Args:
            dataset: Dataset to analyze

        Returns:
            Detailed quality analysis report
        """
        if not self.is_enabled():
            return {"enabled": False, "message": "Quality analysis is disabled"}

        return self.analyzer.analyze_dataset(dataset)


# Convenience functions for easy integration

def create_quality_hooks(config: Optional[QualityConfig] = None) -> QualityHooks:
    """
    Create quality hooks instance.

    Args:
        config: Quality configuration

    Returns:
        Configured QualityHooks instance
    """
    return QualityHooks(config)


def quality_check_passed(dataset: List[Dict[str, Any]],
                        config: Optional[QualityConfig] = None,
                        threshold: Optional[float] = None) -> bool:
    """
    Convenience function for quick quality check.

    Args:
        dataset: Dataset to check
        config: Quality configuration
        threshold: Optional threshold

    Returns:
        True if quality check passes
    """
    hooks = QualityHooks(config)
    return hooks.quality_check_hook(dataset, threshold)