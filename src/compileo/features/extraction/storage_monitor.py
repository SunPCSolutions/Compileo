"""
Storage monitoring and alerting system for filesystem storage.
Provides health checks, performance metrics, and automated alerting.
"""

import os
import time
import logging
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
import threading
import json

# Removed circular import - will be passed as parameters

logger = logging.getLogger(__name__)


class StorageMetrics:
    """Collects and manages storage performance metrics."""

    def __init__(self):
        self.metrics = {
            'operations_total': 0,
            'operations_by_type': {},
            'errors_total': 0,
            'errors_by_type': {},
            'performance': {
                'avg_response_time': 0.0,
                'max_response_time': 0.0,
                'min_response_time': float('inf'),
                'total_response_time': 0.0
            },
            'storage': {
                'total_files': 0,
                'total_size_bytes': 0,
                'jobs_count': 0,
                'corrupted_files': 0
            },
            'last_updated': datetime.utcnow().isoformat()
        }
        self._lock = threading.Lock()

    def record_operation(self, operation_type: str, duration: float, success: bool = True):
        """Record an operation with its duration and success status."""
        with self._lock:
            self.metrics['operations_total'] += 1

            if operation_type not in self.metrics['operations_by_type']:
                self.metrics['operations_by_type'][operation_type] = 0
            self.metrics['operations_by_type'][operation_type] += 1

            if not success:
                self.metrics['errors_total'] += 1
                if operation_type not in self.metrics['errors_by_type']:
                    self.metrics['errors_by_type'][operation_type] = 0
                self.metrics['errors_by_type'][operation_type] += 1

            # Update performance metrics
            perf = self.metrics['performance']
            perf['total_response_time'] += duration
            perf['max_response_time'] = max(perf['max_response_time'], duration)
            perf['min_response_time'] = min(perf['min_response_time'], duration)
            perf['avg_response_time'] = perf['total_response_time'] / self.metrics['operations_total']

            self.metrics['last_updated'] = datetime.utcnow().isoformat()

    def update_storage_stats(self, total_files: int, total_size: int, jobs_count: int, corrupted_files: int = 0):
        """Update storage statistics."""
        with self._lock:
            self.metrics['storage'].update({
                'total_files': total_files,
                'total_size_bytes': total_size,
                'jobs_count': jobs_count,
                'corrupted_files': corrupted_files
            })
            self.metrics['last_updated'] = datetime.utcnow().isoformat()

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics snapshot."""
        with self._lock:
            return self.metrics.copy()

    def reset(self):
        """Reset all metrics."""
        with self._lock:
            self.metrics = {
                'operations_total': 0,
                'operations_by_type': {},
                'errors_total': 0,
                'errors_by_type': {},
                'performance': {
                    'avg_response_time': 0.0,
                    'max_response_time': 0.0,
                    'min_response_time': float('inf'),
                    'total_response_time': 0.0
                },
                'storage': {
                    'total_files': 0,
                    'total_size_bytes': 0,
                    'jobs_count': 0,
                    'corrupted_files': 0
                },
                'last_updated': datetime.utcnow().isoformat()
            }


class AlertManager:
    """Manages alerts for storage issues and performance degradation."""

    def __init__(self):
        self.alerts = []
        self.alert_handlers = []
        self._lock = threading.Lock()

    def add_alert_handler(self, handler: Callable[[Dict[str, Any]], None]):
        """Add an alert handler function."""
        with self._lock:
            self.alert_handlers.append(handler)

    def remove_alert_handler(self, handler: Callable[[Dict[str, Any]], None]):
        """Remove an alert handler function."""
        with self._lock:
            if handler in self.alert_handlers:
                self.alert_handlers.remove(handler)

    def trigger_alert(self, alert_type: str, severity: str, message: str, details: Optional[Dict[str, Any]] = None):
        """Trigger an alert with the specified parameters."""
        alert = {
            'id': f"{int(time.time())}_{len(self.alerts)}",
            'type': alert_type,
            'severity': severity,  # 'info', 'warning', 'error', 'critical'
            'message': message,
            'details': details or {},
            'timestamp': datetime.utcnow().isoformat(),
            'acknowledged': False
        }

        with self._lock:
            self.alerts.append(alert)

            # Keep only last 1000 alerts
            if len(self.alerts) > 1000:
                self.alerts = self.alerts[-1000:]

        # Notify handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")

        logger.warning(f"Alert triggered: {alert_type} - {message}")

    def get_alerts(self, severity: Optional[str] = None, acknowledged: Optional[bool] = None,
                   limit: int = 100) -> List[Dict[str, Any]]:
        """Get alerts with optional filtering."""
        with self._lock:
            alerts = self.alerts.copy()

        # Apply filters
        if severity:
            alerts = [a for a in alerts if a['severity'] == severity]
        if acknowledged is not None:
            alerts = [a for a in alerts if a['acknowledged'] == acknowledged]

        return alerts[-limit:]

    def acknowledge_alert(self, alert_id: str):
        """Mark an alert as acknowledged."""
        with self._lock:
            for alert in self.alerts:
                if alert['id'] == alert_id:
                    alert['acknowledged'] = True
                    break

    def clear_old_alerts(self, days: int = 7):
        """Clear alerts older than specified days."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        with self._lock:
            self.alerts = [
                alert for alert in self.alerts
                if datetime.fromisoformat(alert['timestamp']) > cutoff
            ]


class StorageHealthMonitor:
    """Monitors storage health and performance."""

    def __init__(self, storage_manager, alert_manager: AlertManager):
        self.storage_manager = storage_manager
        self.alert_manager = alert_manager
        self.metrics = StorageMetrics()
        self.last_health_check = None
        self.health_check_interval = 300  # 5 minutes
        self._monitoring_thread = None
        self._stop_monitoring = False

    def start_monitoring(self):
        """Start the monitoring thread."""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            return

        self._stop_monitoring = False
        self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitoring_thread.start()
        logger.info("Storage monitoring started")

    def stop_monitoring(self):
        """Stop the monitoring thread."""
        self._stop_monitoring = True
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)
        logger.info("Storage monitoring stopped")

    def _monitoring_loop(self):
        """Main monitoring loop."""
        while not self._stop_monitoring:
            try:
                self._perform_health_check()
                self._check_performance_thresholds()
                self._check_storage_capacity()
            except Exception as e:
                logger.error(f"Monitoring error: {e}")

            time.sleep(self.health_check_interval)

    def _perform_health_check(self):
        """Perform comprehensive health check."""
        try:
            start_time = time.time()
            health_report = self.storage_manager.check_file_integrity()
            duration = time.time() - start_time

            self.metrics.record_operation('health_check', duration, True)

            # Update metrics
            self.metrics.update_storage_stats(
                total_files=health_report['total_files'],
                total_size=0,  # Would need to calculate
                jobs_count=0,  # Would need to count directories
                corrupted_files=health_report['corrupted_files']
            )

            # Check for issues
            if health_report['corrupted_files'] > 0:
                self.alert_manager.trigger_alert(
                    'integrity',
                    'error',
                    f"Found {health_report['corrupted_files']} corrupted files",
                    {'corrupted_files': health_report['corrupted_files_list']}
                )

            self.last_health_check = datetime.utcnow()

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            self.metrics.record_operation('health_check', 0, False)

    def _check_performance_thresholds(self):
        """Check performance against thresholds."""
        metrics = self.metrics.get_metrics()
        perf = metrics['performance']

        # Alert on slow operations
        if perf['avg_response_time'] > 10.0:  # 10 seconds
            self.alert_manager.trigger_alert(
                'performance',
                'warning',
                f"Average response time is high: {perf['avg_response_time']:.2f}s",
                {'avg_response_time': perf['avg_response_time']}
            )

        # Alert on high error rate
        total_ops = metrics['operations_total']
        if total_ops > 10:  # Only check after some operations
            error_rate = metrics['errors_total'] / total_ops
            if error_rate > 0.1:  # 10% error rate
                self.alert_manager.trigger_alert(
                    'errors',
                    'error',
                    f"High error rate: {error_rate:.1%}",
                    {'error_rate': error_rate, 'total_errors': metrics['errors_total']}
                )

    def _check_storage_capacity(self):
        """Check storage capacity and disk usage."""
        try:
            base_path = self.storage_manager.file_manager.base_path
            disk_usage = psutil.disk_usage(str(base_path))

            usage_percent = disk_usage.percent

            if usage_percent > 90:
                self.alert_manager.trigger_alert(
                    'capacity',
                    'critical',
                    f"Storage usage critical: {usage_percent:.1f}%",
                    {
                        'usage_percent': usage_percent,
                        'free_bytes': disk_usage.free,
                        'total_bytes': disk_usage.total
                    }
                )
            elif usage_percent > 80:
                self.alert_manager.trigger_alert(
                    'capacity',
                    'warning',
                    f"Storage usage high: {usage_percent:.1f}%",
                    {
                        'usage_percent': usage_percent,
                        'free_bytes': disk_usage.free
                    }
                )

        except Exception as e:
            logger.error(f"Capacity check failed: {e}")

    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        metrics = self.metrics.get_metrics()

        # Determine overall health
        health_status = 'healthy'
        issues = []

        if metrics['storage']['corrupted_files'] > 0:
            health_status = 'degraded'
            issues.append(f"{metrics['storage']['corrupted_files']} corrupted files")

        if metrics['performance']['avg_response_time'] > 5.0:
            health_status = 'degraded'
            issues.append(f"high response time: {metrics['performance']['avg_response_time']:.2f}s")

        error_rate = 0
        if metrics['operations_total'] > 0:
            error_rate = metrics['errors_total'] / metrics['operations_total']
            if error_rate > 0.05:
                health_status = 'unhealthy'
                issues.append(f"high error rate: {error_rate:.1%}")

        return {
            'status': health_status,
            'issues': issues,
            'metrics': metrics,
            'last_check': self.last_health_check.isoformat() if self.last_health_check else None
        }


class StorageLogger:
    """Enhanced logging for storage operations with structured logging."""

    def __init__(self, base_logger: Optional[logging.Logger] = None):
        self.logger = base_logger or logger

    def log_operation(self, operation: str, job_id: Optional[str] = None,
                     duration: Optional[float] = None, success: bool = True,
                     details: Optional[Dict[str, Any]] = None):
        """Log a storage operation with structured data."""
        log_data = {
            'operation': operation,
            'timestamp': datetime.utcnow().isoformat(),
            'success': success
        }

        if job_id is not None:
            log_data['job_id'] = job_id
        if duration is not None:
            log_data['duration_seconds'] = duration
        if details:
            log_data['details'] = details

        # Calculate file size if applicable
        if details and 'file_path' in details:
            try:
                file_path = Path(details['file_path'])
                if file_path.exists():
                    log_data['file_size_bytes'] = file_path.stat().st_size
            except:
                pass

        log_level = logging.INFO if success else logging.ERROR
        self.logger.log(log_level, f"Storage operation: {operation}", extra=log_data)

    def log_performance_metric(self, metric_name: str, value: float, unit: str = 'seconds'):
        """Log a performance metric."""
        self.logger.info(f"Performance metric: {metric_name} = {value} {unit}",
                        extra={
                            'metric_name': metric_name,
                            'metric_value': value,
                            'metric_unit': unit,
                            'timestamp': datetime.utcnow().isoformat()
                        })

    def log_error(self, operation: str, error: Exception, context: Optional[Dict[str, Any]] = None):
        """Log an error with full context."""
        error_data = {
            'operation': operation,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'timestamp': datetime.utcnow().isoformat()
        }

        if context:
            error_data['context'] = json.dumps(context)

        self.logger.error(f"Storage error in {operation}: {error}", extra=error_data, exc_info=True)


# Global instances for easy access
_metrics = StorageMetrics()
_alert_manager = AlertManager()
_monitor = None
_logger = StorageLogger()

def get_metrics() -> StorageMetrics:
    """Get the global metrics instance."""
    return _metrics

def get_alert_manager() -> AlertManager:
    """Get the global alert manager instance."""
    return _alert_manager

def get_monitor() -> Optional[StorageHealthMonitor]:
    """Get the global monitor instance."""
    return _monitor

def get_logger() -> StorageLogger:
    """Get the global logger instance."""
    return _logger

def initialize_monitoring(storage_manager) -> StorageHealthMonitor:
    """Initialize the global monitoring system."""
    global _monitor

    if _monitor is None:
        _monitor = StorageHealthMonitor(storage_manager, _alert_manager)
        _monitor.start_monitoring()

    return _monitor

def shutdown_monitoring():
    """Shutdown the monitoring system."""
    global _monitor
    if _monitor:
        _monitor.stop_monitoring()
        _monitor = None