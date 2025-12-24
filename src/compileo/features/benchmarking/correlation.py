"""
Quality-performance correlation analysis.
"""

from typing import Dict, List, Any, Optional, Tuple
import logging
from scipy import stats
import numpy as np

logger = logging.getLogger(__name__)


class CorrelationAnalyzer:
    """
    Analyzes correlations between dataset quality metrics and model performance.

    Provides insights into which quality factors most impact model performance
    and recommendations for dataset improvements.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def analyze_correlations(self, quality_metrics: Dict[str, Any],
                           performance_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze correlations between quality metrics and performance.

        Args:
            quality_metrics: Dataset quality metrics
            performance_results: Model performance results

        Returns:
            Correlation analysis results
        """
        analysis = {
            'correlations': {},
            'significant_factors': [],
            'recommendations': [],
            'correlation_matrix': {}
        }

        # Extract quality factors and performance metrics
        quality_factors = self._extract_quality_factors(quality_metrics)
        performance_metrics = self._extract_performance_metrics(performance_results)

        if not quality_factors or not performance_metrics:
            logger.warning("Insufficient data for correlation analysis")
            return analysis

        # Calculate correlations
        for perf_metric_name, perf_values in performance_metrics.items():
            analysis['correlations'][perf_metric_name] = {}

            for quality_factor_name, quality_values in quality_factors.items():
                correlation = self._calculate_correlation(quality_values, perf_values)
                if correlation is not None:
                    analysis['correlations'][perf_metric_name][quality_factor_name] = correlation

                    # Check significance
                    if abs(correlation['coefficient']) > 0.5:  # Strong correlation
                        analysis['significant_factors'].append({
                            'performance_metric': perf_metric_name,
                            'quality_factor': quality_factor_name,
                            'correlation': correlation,
                            'strength': 'strong' if abs(correlation['coefficient']) > 0.7 else 'moderate'
                        })

        # Generate correlation matrix
        analysis['correlation_matrix'] = self._build_correlation_matrix(
            quality_factors, performance_metrics
        )

        # Generate recommendations
        analysis['recommendations'] = self._generate_recommendations(
            analysis['significant_factors']
        )

        return analysis

    def _extract_quality_factors(self, quality_metrics: Dict[str, Any]) -> Dict[str, List[float]]:
        """Extract quality factors as numerical values."""
        factors = {}

        # Handle different quality metric formats
        if isinstance(quality_metrics, dict):
            for key, value in quality_metrics.items():
                if isinstance(value, (int, float)):
                    factors[key] = [value]
                elif isinstance(value, dict):
                    # Extract numerical values from nested dict
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, (int, float)):
                            factor_key = f"{key}_{sub_key}"
                            factors[factor_key] = [sub_value]

        return factors

    def _extract_performance_metrics(self, performance_results: Dict[str, Any]) -> Dict[str, List[float]]:
        """Extract performance metrics as numerical values."""
        metrics = {}

        # Handle different performance result formats
        if isinstance(performance_results, dict):
            for benchmark_key, benchmark_data in performance_results.items():
                if isinstance(benchmark_data, dict) and 'metrics' in benchmark_data:
                    for metric_name, value in benchmark_data['metrics'].items():
                        if isinstance(value, (int, float)):
                            full_key = f"{benchmark_key}_{metric_name}"
                            metrics[full_key] = [value]
                        elif isinstance(value, dict) and 'mean' in value:
                            full_key = f"{benchmark_key}_{metric_name}"
                            metrics[full_key] = [value['mean']]

        return metrics

    def _calculate_correlation(self, x_values: List[float], y_values: List[float]) -> Optional[Dict[str, Any]]:
        """Calculate correlation between two sets of values."""
        try:
            if len(x_values) != len(y_values) or len(x_values) < 2:
                return None

            # Use Pearson correlation
            correlation_result = stats.pearsonr(x_values, y_values)
            coefficient = correlation_result[0]
            p_value = correlation_result[1]

            return {
                'coefficient': float(coefficient),
                'p_value': float(p_value),
                'significant': p_value < 0.05,
                'direction': 'positive' if coefficient > 0 else 'negative',
                'magnitude': abs(coefficient)
            }

        except Exception as e:
            logger.error(f"Correlation calculation failed: {e}")
            return None

    def _build_correlation_matrix(self, quality_factors: Dict[str, List[float]],
                                performance_metrics: Dict[str, List[float]]) -> Dict[str, Any]:
        """Build a correlation matrix between all factors and metrics."""
        matrix = {
            'quality_factors': list(quality_factors.keys()),
            'performance_metrics': list(performance_metrics.keys()),
            'correlations': {}
        }

        for q_factor in quality_factors.keys():
            matrix['correlations'][q_factor] = {}
            q_values = quality_factors[q_factor]

            for p_metric in performance_metrics.keys():
                p_values = performance_metrics[p_metric]
                correlation = self._calculate_correlation(q_values, p_values)

                matrix['correlations'][q_factor][p_metric] = correlation

        return matrix

    def _generate_recommendations(self, significant_factors: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on correlation analysis."""
        recommendations = []

        if not significant_factors:
            recommendations.append("No strong correlations found between quality metrics and performance.")
            return recommendations

        # Sort by correlation strength
        sorted_factors = sorted(
            significant_factors,
            key=lambda x: abs(x['correlation']['coefficient']),
            reverse=True
        )

        for factor in sorted_factors[:5]:  # Top 5 most significant
            perf_metric = factor['performance_metric']
            quality_factor = factor['quality_factor']
            correlation = factor['correlation']
            strength = factor['strength']

            if correlation['direction'] == 'positive':
                rec = f"Improve {quality_factor} to enhance {perf_metric} ({strength} {correlation['direction']} correlation: {correlation['coefficient']:.3f})"
            else:
                rec = f"Address {quality_factor} issues to improve {perf_metric} (strong negative correlation: {correlation['coefficient']:.3f})"

            recommendations.append(rec)

        return recommendations

    def analyze_performance_impact(self, quality_improvements: Dict[str, float],
                                 correlations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze expected performance impact from quality improvements.

        Args:
            quality_improvements: Dictionary of quality factor improvements (factor_name -> improvement_amount)
            correlations: Correlation analysis results

        Returns:
            Expected performance impacts
        """
        impacts = {
            'expected_improvements': {},
            'confidence_levels': {},
            'prioritized_actions': []
        }

        for factor_name, improvement in quality_improvements.items():
            impacts['expected_improvements'][factor_name] = {}
            impacts['confidence_levels'][factor_name] = {}

            # Find correlations for this factor
            for perf_metric, factor_correlations in correlations.get('correlations', {}).items():
                if factor_name in factor_correlations:
                    correlation = factor_correlations[factor_name]
                    if correlation and correlation['significant']:
                        # Estimate performance improvement
                        expected_change = improvement * correlation['coefficient']
                        confidence = 1 - correlation['p_value']  # Higher confidence for lower p-value

                        impacts['expected_improvements'][factor_name][perf_metric] = expected_change
                        impacts['confidence_levels'][factor_name][perf_metric] = confidence

                        impacts['prioritized_actions'].append({
                            'factor': factor_name,
                            'performance_metric': perf_metric,
                            'expected_improvement': expected_change,
                            'confidence': confidence,
                            'correlation_strength': abs(correlation['coefficient'])
                        })

        # Sort prioritized actions by expected improvement
        impacts['prioritized_actions'].sort(
            key=lambda x: x['expected_improvement'] * x['confidence'],
            reverse=True
        )

        return impacts