"""
Enhanced cleanup service with scheduling, monitoring, and performance optimizations.
"""

from typing import List, Dict, Any, Optional, Callable
from datetime import datetime, timedelta
import threading
import time
import logging
from enum import Enum

from .models import ExtractionResult
from .storage import ResultRetentionPolicy

logger = logging.getLogger(__name__)


class CleanupPriority(str, Enum):
    """Cleanup priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class CleanupSchedule:
    """Represents a scheduled cleanup operation."""

    def __init__(
        self,
        name: str,
        interval_seconds: int,
        retention_policy: ResultRetentionPolicy,
        priority: CleanupPriority = CleanupPriority.NORMAL,
        enabled: bool = True,
        max_runtime: Optional[int] = None
    ):
        self.name = name
        self.interval_seconds = interval_seconds
        self.retention_policy = retention_policy
        self.priority = priority
        self.enabled = enabled
        self.max_runtime = max_runtime or (interval_seconds // 4)  # Default 25% of interval

        self.last_run: Optional[datetime] = None
        self.next_run = datetime.utcnow()
        self.run_count = 0
        self.total_cleaned = 0
        self.average_runtime = 0.0
        self.last_error: Optional[str] = None

    def should_run(self, current_time: datetime) -> bool:
        """Check if this cleanup should run now."""
        return self.enabled and current_time >= self.next_run

    def update_schedule(self, current_time: datetime):
        """Update the next run time."""
        self.next_run = current_time + timedelta(seconds=self.interval_seconds)

    def record_run(self, cleaned_count: int, runtime_seconds: float, error: Optional[str] = None):
        """Record a cleanup run."""
        self.last_run = datetime.utcnow()
        self.run_count += 1
        self.total_cleaned += cleaned_count

        if error:
            self.last_error = error
        else:
            # Update average runtime (exponential moving average)
            alpha = 0.1  # Smoothing factor
            self.average_runtime = (alpha * runtime_seconds) + ((1 - alpha) * self.average_runtime)


class EnhancedCleanupService:
    """Enhanced cleanup service with scheduling and monitoring."""

    def __init__(
        self,
        db_connection,
        default_retention_policy: ResultRetentionPolicy,
        cleanup_interval: int = 3600,  # 1 hour
        batch_size: int = 1000,
        enable_background: bool = True
    ):
        self.db_connection = db_connection
        self.default_retention_policy = default_retention_policy
        self.cleanup_interval = cleanup_interval
        self.batch_size = batch_size
        self.enable_background = enable_background

        # Background processing
        self.cleanup_thread: Optional[threading.Thread] = None
        self.running = False
        self.last_cleanup = datetime.utcnow()

        # Scheduling
        self.schedules: Dict[str, CleanupSchedule] = {}
        self._init_default_schedules()

        # Performance monitoring
        self.stats = {
            'total_cleaned': 0,
            'scheduled_runs': 0,
            'manual_runs': 0,
            'errors': 0,
            'average_runtime': 0.0,
            'uptime': 0.0
        }

        self.start_time = datetime.utcnow()

    def _init_default_schedules(self):
        """Initialize default cleanup schedules."""
        from .storage import TimeBasedRetentionPolicy, SizeBasedRetentionPolicy

        # Daily cleanup for old results
        self.add_schedule(
            "daily_cleanup",
            86400,  # 24 hours
            TimeBasedRetentionPolicy(retention_days=30),
            CleanupPriority.NORMAL
        )

        # Weekly aggressive cleanup
        self.add_schedule(
            "weekly_deep_cleanup",
            604800,  # 7 days
            TimeBasedRetentionPolicy(retention_days=90),
            CleanupPriority.LOW
        )

        # Size-based emergency cleanup
        self.add_schedule(
            "size_based_cleanup",
            3600,  # 1 hour
            SizeBasedRetentionPolicy(max_size_bytes=10 * 1024 * 1024 * 1024),  # 10GB
            CleanupPriority.HIGH
        )

    def add_schedule(
        self,
        name: str,
        interval_seconds: int,
        retention_policy: ResultRetentionPolicy,
        priority: CleanupPriority = CleanupPriority.NORMAL
    ) -> None:
        """Add a cleanup schedule."""
        if name in self.schedules:
            logger.warning(f"Schedule {name} already exists, updating")

        schedule = CleanupSchedule(name, interval_seconds, retention_policy, priority)
        self.schedules[name] = schedule
        logger.info(f"Added cleanup schedule: {name}")

    def remove_schedule(self, name: str) -> bool:
        """Remove a cleanup schedule."""
        if name in self.schedules:
            del self.schedules[name]
            logger.info(f"Removed cleanup schedule: {name}")
            return True
        return False

    def start_background_cleanup(self) -> None:
        """Start the background cleanup thread."""
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            logger.info("Cleanup thread already running")
            return

        if not self.enable_background:
            logger.info("Background cleanup disabled")
            return

        self.running = True
        self.cleanup_thread = threading.Thread(
            target=self._cleanup_worker,
            daemon=True,
            name="EnhancedCleanupWorker"
        )
        self.cleanup_thread.start()
        logger.info("Started enhanced background cleanup service")

    def stop_background_cleanup(self) -> None:
        """Stop the background cleanup thread."""
        self.running = False
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=30)
            logger.info("Stopped enhanced background cleanup service")

    def _cleanup_worker(self) -> None:
        """Background worker for periodic cleanup."""
        while self.running:
            try:
                start_time = time.time()
                self.run_scheduled_cleanups()
                runtime = time.time() - start_time

                # Update uptime stats
                self.stats['uptime'] = (datetime.utcnow() - self.start_time).total_seconds()

                # Update average runtime
                if self.stats['scheduled_runs'] > 0:
                    alpha = 0.1
                    self.stats['average_runtime'] = (alpha * runtime) + ((1 - alpha) * self.stats['average_runtime'])

            except Exception as e:
                logger.error(f"Error in cleanup worker: {e}")
                self.stats['errors'] += 1

            # Sleep for the cleanup interval
            time.sleep(self.cleanup_interval)

    def run_scheduled_cleanups(self) -> List[Dict[str, Any]]:
        """Run all due scheduled cleanups."""
        current_time = datetime.utcnow()
        results = []

        # Sort schedules by priority (urgent first)
        priority_order = {CleanupPriority.URGENT: 0, CleanupPriority.HIGH: 1,
                         CleanupPriority.NORMAL: 2, CleanupPriority.LOW: 3}

        sorted_schedules = sorted(
            self.schedules.values(),
            key=lambda s: priority_order[s.priority]
        )

        for schedule in sorted_schedules:
            if schedule.should_run(current_time):
                logger.info(f"Running scheduled cleanup: {schedule.name}")

                start_time = time.time()
                try:
                    # Temporarily switch retention policy
                    original_policy = self.default_retention_policy
                    self.retention_policy = schedule.retention_policy

                    cleanup_result = self.perform_cleanup(force=True, schedule_name=schedule.name)
                    runtime = time.time() - start_time

                    # Record the run
                    schedule.record_run(
                        cleanup_result.get('results_cleaned', 0),
                        runtime
                    )

                    # Update global stats
                    self.stats['total_cleaned'] += cleanup_result.get('results_cleaned', 0)
                    self.stats['scheduled_runs'] += 1

                    cleanup_result['schedule_name'] = schedule.name
                    cleanup_result['runtime_seconds'] = runtime
                    results.append(cleanup_result)

                    # Update schedule
                    schedule.update_schedule(current_time)

                except Exception as e:
                    error_msg = f"Scheduled cleanup {schedule.name} failed: {e}"
                    logger.error(error_msg)
                    schedule.record_run(0, time.time() - start_time, error_msg)
                    self.stats['errors'] += 1

                finally:
                    # Restore original policy
                    self.retention_policy = original_policy

        return results

    def perform_cleanup(
        self,
        force: bool = False,
        schedule_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform cleanup of old results.

        Args:
            force: If True, ignore the cleanup interval check
            schedule_name: Name of the schedule running this cleanup

        Returns:
            Dict with cleanup statistics
        """
        current_time = datetime.utcnow()

        # Check if cleanup is needed (unless forced)
        if not force and (current_time - self.last_cleanup) < timedelta(seconds=self.cleanup_interval):
            return {'status': 'skipped', 'reason': 'too_soon'}

        logger.info(f"Starting cleanup (schedule: {schedule_name or 'manual'})")

        # Get results that should be cleaned up
        cleanup_candidates = self._get_cleanup_candidates(current_time)

        if not cleanup_candidates:
            logger.info("No results to clean up")
            result = {
                'status': 'completed',
                'results_cleaned': 0,
                'chunks_cleaned': 0,
                'bytes_freed': 0,
                'schedule_name': schedule_name
            }
            if not schedule_name:
                self.stats['manual_runs'] += 1
            return result

        # Perform cleanup in batches
        total_cleaned = 0
        chunks_cleaned = set()
        bytes_freed = 0

        for i in range(0, len(cleanup_candidates), self.batch_size):
            batch = cleanup_candidates[i:i + self.batch_size]
            batch_stats = self._cleanup_batch(batch)
            total_cleaned += batch_stats['results_cleaned']
            chunks_cleaned.update(batch_stats['chunks_cleaned'])
            bytes_freed += batch_stats['bytes_freed']

        logger.info(f"Cleaned up {total_cleaned} results from {len(chunks_cleaned)} chunks")

        result = {
            'status': 'completed',
            'results_cleaned': total_cleaned,
            'chunks_cleaned': len(chunks_cleaned),
            'bytes_freed': bytes_freed,
            'schedule_name': schedule_name,
            'duration_seconds': (datetime.utcnow() - current_time).total_seconds()
        }

        self.last_cleanup = current_time
        if not schedule_name:
            self.stats['manual_runs'] += 1

        return result

    def _get_cleanup_candidates(self, current_time: datetime) -> List[ExtractionResult]:
        """Get all results that should be cleaned up."""
        cursor = self.db_connection.cursor()

        # Get all results with their metadata
        cursor.execute("""
            SELECT id, job_id, chunk_id, categories, confidence, extracted_data, created_at
            FROM extraction_results
            ORDER BY created_at ASC
        """)

        rows = cursor.fetchall()
        all_results = []

        for row in rows:
            result = ExtractionResult(
                id=row[0],
                job_id=row[1],
                chunk_id=row[2],
                categories=row[3] if isinstance(row[3], list) else [],
                confidence=row[4],
                extracted_data=row[5] if isinstance(row[5], dict) else {},
                created_at=row[6]
            )
            all_results.append(result)

        # Apply retention policy
        return self.default_retention_policy.get_cleanup_candidates(all_results, current_time)

    def _cleanup_batch(self, results: List[ExtractionResult]) -> Dict[str, Any]:
        """Clean up a batch of results."""
        cursor = self.db_connection.cursor()

        result_ids = [r.id for r in results]
        chunk_ids = list(set(r.chunk_id for r in results))

        # Delete results
        placeholders = ','.join('?' * len(result_ids))
        cursor.execute(
            f"DELETE FROM extraction_results WHERE id IN ({placeholders})",
            result_ids
        )

        results_cleaned = cursor.rowcount

        # Calculate bytes freed (rough estimate)
        bytes_freed = sum(
            len(str(r.extracted_data)) + len(r.chunk_id) + len(str(r.categories))
            for r in results
        )

        self.db_connection.commit()

        return {
            'results_cleaned': results_cleaned,
            'chunks_cleaned': chunk_ids,
            'bytes_freed': bytes_freed
        }

    def get_cleanup_stats(self) -> Dict[str, Any]:
        """Get comprehensive cleanup statistics."""
        current_time = datetime.utcnow()

        # Get retention policy info
        cleanup_candidates = self._get_cleanup_candidates(current_time)

        # Get total results count
        cursor = self.db_connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM extraction_results")
        total_results = cursor.fetchone()[0]

        # Schedule stats
        schedule_stats = {}
        for name, schedule in self.schedules.items():
            schedule_stats[name] = {
                'enabled': schedule.enabled,
                'priority': schedule.priority.value,
                'interval_seconds': schedule.interval_seconds,
                'last_run': schedule.last_run.isoformat() if schedule.last_run else None,
                'next_run': schedule.next_run.isoformat(),
                'run_count': schedule.run_count,
                'total_cleaned': schedule.total_cleaned,
                'average_runtime': schedule.average_runtime,
                'last_error': schedule.last_error,
                'is_due': schedule.should_run(current_time)
            }

        return {
            'total_results': total_results,
            'results_to_clean': len(cleanup_candidates),
            'cleanup_percentage': (len(cleanup_candidates) / total_results * 100) if total_results > 0 else 0,
            'last_cleanup': self.last_cleanup.isoformat(),
            'next_cleanup': (self.last_cleanup + timedelta(seconds=self.cleanup_interval)).isoformat(),
            'schedules': schedule_stats,
            'performance': self.stats,
            'uptime_seconds': (current_time - self.start_time).total_seconds()
        }

    def force_cleanup(self, schedule_name: Optional[str] = None) -> Dict[str, Any]:
        """Force an immediate cleanup."""
        return self.perform_cleanup(force=True, schedule_name=schedule_name)

    def optimize_cleanup_schedule(self) -> Dict[str, Any]:
        """Analyze and optimize cleanup schedules based on usage patterns."""
        # This would analyze cleanup patterns and adjust schedules
        # For now, return basic analysis
        stats = self.get_cleanup_stats()

        recommendations = []

        if stats['cleanup_percentage'] > 50:
            recommendations.append("Consider more aggressive cleanup schedules")
        elif stats['cleanup_percentage'] < 10:
            recommendations.append("Cleanup schedules may be too aggressive")

        avg_runtime = stats['performance']['average_runtime']
        if avg_runtime > 300:  # 5 minutes
            recommendations.append("Consider reducing batch sizes for faster cleanups")

        return {
            'current_stats': stats,
            'recommendations': recommendations,
            'suggested_changes': {}
        }

    @property
    def retention_policy(self):
        """Get current retention policy."""
        return self._current_policy

    @retention_policy.setter
    def retention_policy(self, policy: ResultRetentionPolicy):
        """Set current retention policy."""
        self._current_policy = policy