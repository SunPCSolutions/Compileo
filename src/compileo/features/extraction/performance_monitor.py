"""
Performance monitoring and metrics collection for extraction workflows.
Provides comprehensive monitoring of cache performance, job queues, and system health.
"""

import time
import threading
import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from collections import deque
import psutil
import os

logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """Container for various performance metrics."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.start_time = datetime.utcnow()

        # Cache metrics
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_evictions = 0
        self.cache_sets = 0

        # Job queue metrics
        self.jobs_submitted = 0
        self.jobs_completed = 0
        self.jobs_failed = 0
        self.jobs_cancelled = 0
        self.queue_size = 0

        # Search metrics
        self.searches_performed = 0
        self.search_response_times: deque = deque(maxlen=1000)
        self.lazy_loads_performed = 0

        # Cleanup metrics
        self.cleanup_runs = 0
        self.results_cleaned = 0
        self.cleanup_response_times: deque = deque(maxlen=100)

        # System metrics
        self.cpu_usage: deque = deque(maxlen=100)
        self.memory_usage: deque = deque(maxlen=100)
        self.disk_usage: deque = deque(maxlen=100)

        # Error tracking
        self.errors: deque = deque(maxlen=500)
        self.error_counts: Dict[str, int] = {}

    def record_cache_hit(self):
        """Record a cache hit."""
        self.cache_hits += 1

    def record_cache_miss(self):
        """Record a cache miss."""
        self.cache_misses += 1

    def record_cache_eviction(self):
        """Record a cache eviction."""
        self.cache_evictions += 1

    def record_cache_set(self):
        """Record a cache set operation."""
        self.cache_sets += 1

    def record_job_submitted(self):
        """Record a job submission."""
        self.jobs_submitted += 1

    def record_job_completed(self, duration: float):
        """Record a job completion."""
        self.jobs_completed += 1

    def record_job_failed(self):
        """Record a job failure."""
        self.jobs_failed += 1

    def record_job_cancelled(self):
        """Record a job cancellation."""
        self.jobs_cancelled += 1

    def update_queue_size(self, size: int):
        """Update current queue size."""
        self.queue_size = size

    def record_search(self, response_time: float):
        """Record a search operation."""
        self.searches_performed += 1
        self.search_response_times.append(response_time)

    def record_lazy_load(self):
        """Record a lazy load operation."""
        self.lazy_loads_performed += 1

    def record_cleanup_run(self, results_cleaned: int, response_time: float):
        """Record a cleanup run."""
        self.cleanup_runs += 1
        self.results_cleaned += results_cleaned
        self.cleanup_response_times.append(response_time)

    def record_error(self, error_type: str, error_message: str):
        """Record an error."""
        self.errors.append({
            'timestamp': datetime.utcnow(),
            'type': error_type,
            'message': error_message
        })
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1

    def update_system_metrics(self):
        """Update system resource usage metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.cpu_usage.append(cpu_percent)

            # Memory usage
            memory = psutil.virtual_memory()
            self.memory_usage.append(memory.percent)

            # Disk usage
            disk = psutil.disk_usage('/')
            self.disk_usage.append(disk.percent)

        except Exception as e:
            logger.warning(f"Failed to collect system metrics: {e}")

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all metrics."""
        uptime = (datetime.utcnow() - self.start_time).total_seconds()

        # Calculate rates
        cache_hit_rate = self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
        job_success_rate = self.jobs_completed / self.jobs_submitted if self.jobs_submitted > 0 else 0

        # Calculate averages
        avg_search_time = sum(self.search_response_times) / len(self.search_response_times) if self.search_response_times else 0
        avg_cleanup_time = sum(self.cleanup_response_times) / len(self.cleanup_response_times) if self.cleanup_response_times else 0

        # System averages
        avg_cpu = sum(self.cpu_usage) / len(self.cpu_usage) if self.cpu_usage else 0
        avg_memory = sum(self.memory_usage) / len(self.memory_usage) if self.memory_usage else 0
        avg_disk = sum(self.disk_usage) / len(self.disk_usage) if self.disk_usage else 0

        return {
            'uptime_seconds': uptime,
            'cache': {
                'hits': self.cache_hits,
                'misses': self.cache_misses,
                'evictions': self.cache_evictions,
                'sets': self.cache_sets,
                'hit_rate': cache_hit_rate,
                'total_requests': self.cache_hits + self.cache_misses
            },
            'jobs': {
                'submitted': self.jobs_submitted,
                'completed': self.jobs_completed,
                'failed': self.jobs_failed,
                'cancelled': self.jobs_cancelled,
                'success_rate': job_success_rate,
                'current_queue_size': self.queue_size
            },
            'search': {
                'total_searches': self.searches_performed,
                'average_response_time': avg_search_time,
                'lazy_loads': self.lazy_loads_performed
            },
            'cleanup': {
                'runs': self.cleanup_runs,
                'total_results_cleaned': self.results_cleaned,
                'average_response_time': avg_cleanup_time
            },
            'system': {
                'average_cpu_percent': avg_cpu,
                'average_memory_percent': avg_memory,
                'average_disk_percent': avg_disk,
                'current_cpu_percent': self.cpu_usage[-1] if self.cpu_usage else 0,
                'current_memory_percent': self.memory_usage[-1] if self.memory_usage else 0,
                'current_disk_percent': self.disk_usage[-1] if self.disk_usage else 0
            },
            'errors': {
                'total_errors': len(self.errors),
                'error_counts': self.error_counts,
                'recent_errors': list(self.errors)[-10:] if self.errors else []
            }
        }


class PerformanceMonitor:
    """Central performance monitoring system."""

    def __init__(self, collection_interval: int = 60):
        self.metrics = PerformanceMetrics()
        self.collection_interval = collection_interval
        self.monitoring_thread: Optional[threading.Thread] = None
        self.running = False

        # Alert thresholds
        self.alerts: Dict[str, Callable] = {}
        self.alert_thresholds = {
            'cache_hit_rate_low': 0.5,  # Below 50% hit rate
            'memory_usage_high': 85.0,  # Above 85% memory usage
            'error_rate_high': 0.1,     # Above 10% error rate
            'queue_size_high': 100      # Above 100 queued jobs
        }

    def start_monitoring(self):
        """Start the monitoring system."""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            logger.info("Monitoring already running")
            return

        self.running = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_worker,
            daemon=True,
            name="PerformanceMonitor"
        )
        self.monitoring_thread.start()
        logger.info("Started performance monitoring")

    def stop_monitoring(self):
        """Stop the monitoring system."""
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
            logger.info("Stopped performance monitoring")

    def _monitoring_worker(self):
        """Background monitoring worker."""
        while self.running:
            try:
                # Update system metrics
                self.metrics.update_system_metrics()

                # Check alert conditions
                self._check_alerts()

                # Sleep for collection interval
                time.sleep(self.collection_interval)

            except Exception as e:
                logger.error(f"Error in monitoring worker: {e}")
                self.metrics.record_error('monitoring_error', str(e))

    def _check_alerts(self):
        """Check for alert conditions."""
        summary = self.metrics.get_summary()

        # Cache hit rate alert
        cache_hit_rate = summary['cache']['hit_rate']
        if cache_hit_rate < self.alert_thresholds['cache_hit_rate_low']:
            self._trigger_alert('cache_hit_rate_low', f"Cache hit rate is low: {cache_hit_rate:.2%}")

        # Memory usage alert
        memory_usage = summary['system']['current_memory_percent']
        if memory_usage > self.alert_thresholds['memory_usage_high']:
            self._trigger_alert('memory_usage_high', f"Memory usage is high: {memory_usage:.1f}%")

        # Error rate alert
        total_operations = summary['cache']['total_requests'] + summary['jobs']['submitted']
        if total_operations > 0:
            error_rate = len(self.metrics.errors) / total_operations
            if error_rate > self.alert_thresholds['error_rate_high']:
                self._trigger_alert('error_rate_high', f"Error rate is high: {error_rate:.2%}")

        # Queue size alert
        queue_size = summary['jobs']['current_queue_size']
        if queue_size > self.alert_thresholds['queue_size_high']:
            self._trigger_alert('queue_size_high', f"Queue size is high: {queue_size}")

    def _trigger_alert(self, alert_type: str, message: str):
        """Trigger an alert."""
        logger.warning(f"ALERT [{alert_type}]: {message}")

        # Call alert handler if registered
        if alert_type in self.alerts:
            try:
                self.alerts[alert_type](alert_type, message)
            except Exception as e:
                logger.error(f"Error in alert handler for {alert_type}: {e}")

    def register_alert_handler(self, alert_type: str, handler: Callable):
        """Register an alert handler."""
        self.alerts[alert_type] = handler

    def set_alert_threshold(self, alert_type: str, threshold: float):
        """Set an alert threshold."""
        if alert_type in self.alert_thresholds:
            self.alert_thresholds[alert_type] = threshold

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get current metrics summary."""
        return self.metrics.get_summary()

    def get_detailed_metrics(self) -> Dict[str, Any]:
        """Get detailed metrics with time series data."""
        summary = self.metrics.get_summary()

        # Add time series data
        summary['time_series'] = {
            'search_response_times': list(self.metrics.search_response_times),
            'cleanup_response_times': list(self.metrics.cleanup_response_times),
            'cpu_usage': list(self.metrics.cpu_usage),
            'memory_usage': list(self.metrics.memory_usage),
            'disk_usage': list(self.metrics.disk_usage)
        }

        return summary

    def reset_metrics(self):
        """Reset all metrics."""
        self.metrics.reset()
        logger.info("Performance metrics reset")

    def export_metrics(self, format: str = 'json') -> str:
        """Export metrics in the specified format."""
        data = self.get_detailed_metrics()

        if format == 'json':
            import json
            return json.dumps(data, default=str, indent=2)
        elif format == 'csv':
            # Simple CSV export for basic metrics
            lines = ["metric,value"]
            summary = data
            for category, metrics in summary.items():
                if isinstance(metrics, dict):
                    for key, value in metrics.items():
                        if isinstance(value, (int, float)):
                            lines.append(f"{category}.{key},{value}")
            return "\n".join(lines)
        else:
            raise ValueError(f"Unsupported export format: {format}")


# Global performance monitor instance
performance_monitor = PerformanceMonitor()


def record_cache_operation(hit: bool):
    """Record a cache operation."""
    if hit:
        performance_monitor.metrics.record_cache_hit()
    else:
        performance_monitor.metrics.record_cache_miss()


def record_job_operation(operation: str, **kwargs):
    """Record a job operation."""
    if operation == 'submitted':
        performance_monitor.metrics.record_job_submitted()
    elif operation == 'completed':
        duration = kwargs.get('duration', 0)
        performance_monitor.metrics.record_job_completed(duration)
    elif operation == 'failed':
        performance_monitor.metrics.record_job_failed()
    elif operation == 'cancelled':
        performance_monitor.metrics.record_job_cancelled()


def record_search_operation(response_time: float):
    """Record a search operation."""
    performance_monitor.metrics.record_search(response_time)


def record_cleanup_operation(results_cleaned: int, response_time: float):
    """Record a cleanup operation."""
    performance_monitor.metrics.record_cleanup_run(results_cleaned, response_time)


def record_error(error_type: str, message: str):
    """Record an error."""
    performance_monitor.metrics.record_error(error_type, message)


# Decorator for automatic performance monitoring
def monitor_performance(operation_type: str):
    """Decorator to automatically monitor function performance."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                response_time = time.time() - start_time

                # Record based on operation type
                if operation_type == 'search':
                    record_search_operation(response_time)
                elif operation_type == 'cleanup':
                    # Assume cleanup returns results_cleaned
                    results_cleaned = getattr(result, 'results_cleaned', 0) if hasattr(result, 'results_cleaned') else 0
                    record_cleanup_operation(results_cleaned, response_time)

                return result
            except Exception as e:
                response_time = time.time() - start_time
                record_error(f"{operation_type}_error", str(e))
                raise
        return wrapper
    return decorator