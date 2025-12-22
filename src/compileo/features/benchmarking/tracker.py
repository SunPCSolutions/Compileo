"""
Performance tracking and comparison system.
"""

from typing import Dict, List, Any, Optional
import json
import os
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class PerformanceRecord:
    """Record of a model's performance on benchmarks."""

    def __init__(self, model_name: str, benchmark_results: Dict[str, Any],
                 metadata: Optional[Dict[str, Any]] = None):
        self.model_name = model_name
        self.timestamp = datetime.now()
        self.benchmark_results = benchmark_results
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'model_name': self.model_name,
            'timestamp': self.timestamp.isoformat(),
            'benchmark_results': self.benchmark_results,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PerformanceRecord':
        """Create from dictionary."""
        record = cls(
            model_name=data['model_name'],
            benchmark_results=data['benchmark_results'],
            metadata=data.get('metadata', {})
        )
        record.timestamp = datetime.fromisoformat(data['timestamp'])
        return record


class PerformanceTracker:
    """
    Tracks and compares model performance across different versions and configurations.

    Provides historical analysis and performance trend monitoring.
    """

    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = Path(storage_path) if storage_path else Path("benchmark_results")
        self.storage_path.mkdir(exist_ok=True)
        self.records: List[PerformanceRecord] = []
        self._load_records()

    def _load_records(self):
        """Load existing performance records from storage."""
        records_file = self.storage_path / "performance_records.json"
        if records_file.exists():
            try:
                with open(records_file, 'r') as f:
                    data = json.load(f)
                    self.records = [PerformanceRecord.from_dict(record) for record in data]
                logger.info(f"Loaded {len(self.records)} performance records")
            except Exception as e:
                logger.error(f"Failed to load performance records: {e}")
                self.records = []

    def _save_records(self):
        """Save performance records to storage."""
        records_file = self.storage_path / "performance_records.json"
        try:
            data = [record.to_dict() for record in self.records]
            with open(records_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save performance records: {e}")

    def add_record(self, model_name: str, benchmark_results: Dict[str, Any],
                   metadata: Optional[Dict[str, Any]] = None):
        """
        Add a new performance record.

        Args:
            model_name: Name/identifier of the model
            benchmark_results: Results from benchmark evaluation
            metadata: Additional metadata about the evaluation
        """
        record = PerformanceRecord(model_name, benchmark_results, metadata)
        self.records.append(record)
        self._save_records()
        logger.info(f"Added performance record for model: {model_name}")

    def get_model_history(self, model_name: str) -> List[PerformanceRecord]:
        """
        Get performance history for a specific model.

        Args:
            model_name: Name of the model

        Returns:
            List of performance records for the model
        """
        return [record for record in self.records if record.model_name == model_name]

    def compare_models(self, model_names: List[str],
                      benchmark_filter: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Compare performance between multiple models.

        Args:
            model_names: List of model names to compare
            benchmark_filter: Optional list of benchmarks to include

        Returns:
            Comparison results dictionary
        """
        comparison = {
            'models': model_names,
            'benchmarks': {},
            'summary': {}
        }

        # Get latest records for each model
        latest_records = {}
        for model_name in model_names:
            model_records = self.get_model_history(model_name)
            if model_records:
                latest_records[model_name] = max(model_records, key=lambda r: r.timestamp)

        # Compare benchmarks
        all_benchmarks = set()
        for record in latest_records.values():
            all_benchmarks.update(record.benchmark_results.keys())

        if benchmark_filter:
            all_benchmarks = all_benchmarks.intersection(set(benchmark_filter))

        for benchmark in all_benchmarks:
            comparison['benchmarks'][benchmark] = {}

            for model_name, record in latest_records.items():
                if benchmark in record.benchmark_results:
                    results = record.benchmark_results[benchmark]
                    comparison['benchmarks'][benchmark][model_name] = results

                    # Calculate summary statistics
                    if 'metrics' in results:
                        for metric_name, value in results['metrics'].items():
                            if isinstance(value, dict) and 'mean' in value:
                                metric_key = f"{benchmark}_{metric_name}"
                                if metric_key not in comparison['summary']:
                                    comparison['summary'][metric_key] = {}
                                comparison['summary'][metric_key][model_name] = value['mean']
                            elif isinstance(value, (int, float)):
                                metric_key = f"{benchmark}_{metric_name}"
                                if metric_key not in comparison['summary']:
                                    comparison['summary'][metric_key] = {}
                                comparison['summary'][metric_key][model_name] = value

        return comparison

    def get_performance_trends(self, model_name: str,
                              metric_filter: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze performance trends for a model over time.

        Args:
            model_name: Name of the model
            metric_filter: Optional list of metrics to include

        Returns:
            Trend analysis dictionary
        """
        model_records = sorted(self.get_model_history(model_name), key=lambda r: r.timestamp)

        trends = {
            'model_name': model_name,
            'record_count': len(model_records),
            'trends': {}
        }

        if not model_records:
            return trends

        # Analyze trends for each benchmark and metric
        for record in model_records:
            for benchmark_name, benchmark_results in record.benchmark_results.items():
                if benchmark_name not in trends['trends']:
                    trends['trends'][benchmark_name] = {}

                if 'metrics' in benchmark_results:
                    for metric_name, value in benchmark_results['metrics'].items():
                        if metric_filter and metric_name not in metric_filter:
                            continue

                        if metric_name not in trends['trends'][benchmark_name]:
                            trends['trends'][benchmark_name][metric_name] = []

                        # Extract scalar value
                        if isinstance(value, dict) and 'mean' in value:
                            scalar_value = value['mean']
                        elif isinstance(value, (int, float)):
                            scalar_value = value
                        else:
                            continue

                        trends['trends'][benchmark_name][metric_name].append({
                            'timestamp': record.timestamp.isoformat(),
                            'value': scalar_value
                        })

        # Calculate trend statistics
        for benchmark_name, metrics in trends['trends'].items():
            for metric_name, values in metrics.items():
                if len(values) > 1:
                    sorted_values = sorted(values, key=lambda x: x['timestamp'])
                    first_value = sorted_values[0]['value']
                    last_value = sorted_values[-1]['value']
                    change = last_value - first_value
                    trend = "improving" if change > 0 else "declining" if change < 0 else "stable"

                    metrics[metric_name] = {
                        'values': sorted_values,
                        'change': change,
                        'trend': trend,
                        'first_value': first_value,
                        'last_value': last_value
                    }

        return trends

    def detect_performance_degradation(self, model_name: str,
                                      threshold: float = 0.05) -> List[Dict[str, Any]]:
        """
        Detect significant performance degradation.

        Args:
            model_name: Name of the model
            threshold: Minimum relative change to consider degradation

        Returns:
            List of degradation alerts
        """
        trends = self.get_performance_trends(model_name)
        alerts = []

        for benchmark_name, metrics in trends['trends'].items():
            for metric_name, trend_data in metrics.items():
                if isinstance(trend_data, dict) and 'change' in trend_data:
                    change = trend_data['change']
                    if change < -threshold:  # Significant decline
                        alerts.append({
                            'model_name': model_name,
                            'benchmark': benchmark_name,
                            'metric': metric_name,
                            'change': change,
                            'threshold': threshold,
                            'severity': 'high' if change < -0.1 else 'medium'
                        })

        return alerts