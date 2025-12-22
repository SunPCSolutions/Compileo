"""
Result cleanup and retention management for extraction results.
Handles automatic cleanup of old results based on retention policies.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import threading
import time
import logging

from .models import ExtractionResult
from .storage import ResultRetentionPolicy

logger = logging.getLogger(__name__)


class ResultCleanupService:
    """Service for managing result cleanup and retention."""

    def __init__(
        self,
        db_connection,
        retention_policy: ResultRetentionPolicy,
        cleanup_interval: int = 3600,  # 1 hour
        batch_size: int = 1000
    ):
        self.db_connection = db_connection
        self.retention_policy = retention_policy
        self.cleanup_interval = cleanup_interval
        self.batch_size = batch_size

        self.cleanup_thread: Optional[threading.Thread] = None
        self.running = False
        self.last_cleanup = datetime.utcnow()

    def start_background_cleanup(self) -> None:
        """Start the background cleanup thread."""
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            logger.info("Cleanup thread already running")
            return

        self.running = True
        self.cleanup_thread = threading.Thread(
            target=self._cleanup_worker,
            daemon=True,
            name="ResultCleanupWorker"
        )
        self.cleanup_thread.start()
        logger.info("Started background result cleanup service")

    def stop_background_cleanup(self) -> None:
        """Stop the background cleanup thread."""
        self.running = False
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=30)
            logger.info("Stopped background result cleanup service")

    def _cleanup_worker(self) -> None:
        """Background worker for periodic cleanup."""
        while self.running:
            try:
                self.perform_cleanup()
                self.last_cleanup = datetime.utcnow()
            except Exception as e:
                logger.error(f"Error during result cleanup: {e}")

            # Sleep for the cleanup interval
            time.sleep(self.cleanup_interval)

    def perform_cleanup(self, force: bool = False) -> Dict[str, Any]:
        """
        Perform cleanup of old results.

        Args:
            force: If True, ignore the cleanup interval check

        Returns:
            Dict with cleanup statistics
        """
        current_time = datetime.utcnow()

        # Check if cleanup is needed (unless forced)
        if not force and (current_time - self.last_cleanup) < timedelta(seconds=self.cleanup_interval):
            return {'status': 'skipped', 'reason': 'too_soon'}

        logger.info("Starting result cleanup")

        # Get results that should be cleaned up
        cleanup_candidates = self._get_cleanup_candidates(current_time)

        if not cleanup_candidates:
            logger.info("No results to clean up")
            return {
                'status': 'completed',
                'results_cleaned': 0,
                'chunks_cleaned': 0,
                'bytes_freed': 0
            }

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

        return {
            'status': 'completed',
            'results_cleaned': total_cleaned,
            'chunks_cleaned': len(chunks_cleaned),
            'bytes_freed': bytes_freed
        }

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
        return self.retention_policy.get_cleanup_candidates(all_results, current_time)

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
        """Get statistics about cleanup status."""
        current_time = datetime.utcnow()

        # Get retention policy info
        cleanup_candidates = self._get_cleanup_candidates(current_time)

        # Get total results count
        cursor = self.db_connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM extraction_results")
        total_results = cursor.fetchone()[0]

        return {
            'total_results': total_results,
            'results_to_clean': len(cleanup_candidates),
            'last_cleanup': self.last_cleanup.isoformat(),
            'next_cleanup': (self.last_cleanup + timedelta(seconds=self.cleanup_interval)).isoformat(),
            'cleanup_percentage': (len(cleanup_candidates) / total_results * 100) if total_results > 0 else 0
        }

    def force_cleanup(self) -> Dict[str, Any]:
        """Force an immediate cleanup."""
        return self.perform_cleanup(force=True)


class TimeBasedCleanupScheduler:
    """Scheduler for time-based cleanup operations."""

    def __init__(self, cleanup_service: ResultCleanupService):
        self.cleanup_service = cleanup_service
        self.schedules: List[Dict[str, Any]] = []

    def add_schedule(
        self,
        name: str,
        interval_seconds: int,
        retention_policy: ResultRetentionPolicy
    ) -> None:
        """Add a cleanup schedule."""
        schedule = {
            'name': name,
            'interval': interval_seconds,
            'policy': retention_policy,
            'last_run': None,
            'next_run': datetime.utcnow()
        }
        self.schedules.append(schedule)

    def run_scheduled_cleanups(self) -> List[Dict[str, Any]]:
        """Run all due scheduled cleanups."""
        current_time = datetime.utcnow()
        results = []

        for schedule in self.schedules:
            if current_time >= schedule['next_run']:
                # Switch to the scheduled policy temporarily
                original_policy = self.cleanup_service.retention_policy
                self.cleanup_service.retention_policy = schedule['policy']

                try:
                    cleanup_result = self.cleanup_service.force_cleanup()
                    cleanup_result['schedule_name'] = schedule['name']
                    results.append(cleanup_result)

                    # Update schedule
                    schedule['last_run'] = current_time
                    schedule['next_run'] = current_time + timedelta(seconds=schedule['interval'])

                finally:
                    # Restore original policy
                    self.cleanup_service.retention_policy = original_policy

        return results

    def get_schedule_status(self) -> List[Dict[str, Any]]:
        """Get status of all schedules."""
        current_time = datetime.utcnow()
        return [
            {
                'name': s['name'],
                'last_run': s['last_run'].isoformat() if s['last_run'] else None,
                'next_run': s['next_run'].isoformat(),
                'is_due': current_time >= s['next_run']
            }
            for s in self.schedules
        ]