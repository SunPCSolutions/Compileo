"""
Job history service for Compileo GUI.

Provides methods for retrieving and displaying job history from the database,
integrating with the job history retention system.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from ....storage.src.project.database_repositories import JobHistoryRepository

logger = logging.getLogger(__name__)


class JobHistoryService:
    """Service for managing job history in the GUI."""

    def __init__(self, owner_id: str):
        """
        Initialize the job history service.

        Args:
            owner_id: The user ID for tenant-specific database access
        """
        self.owner_id = owner_id
        self.history_repo = JobHistoryRepository(owner_id)

    def get_recent_jobs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get recent job history entries.

        Args:
            limit: Maximum number of entries to return

        Returns:
            List of job history entries
        """
        try:
            return self.history_repo.get_all_history(limit)
        except Exception as e:
            logger.error(f"Failed to get recent jobs: {e}")
            return []

    def get_project_jobs(self, project_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get job history for a specific project.

        Args:
            project_id: The project ID to filter by
            limit: Maximum number of entries to return

        Returns:
            List of job history entries for the project
        """
        try:
            return self.history_repo.get_history_by_project(project_id, limit)
        except Exception as e:
            logger.error(f"Failed to get project jobs: {e}")
            return []

    def get_jobs_by_type(self, job_type: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get job history filtered by job type.

        Args:
            job_type: Type of job (e.g., 'dataset_generation', 'extraction')
            limit: Maximum number of entries to return

        Returns:
            List of job history entries
        """
        try:
            return self.history_repo.get_history_by_job_type(job_type, limit)
        except Exception as e:
            logger.error(f"Failed to get jobs by type: {e}")
            return []

    def get_job_details(self, history_id: str) -> Optional[Dict[str, Any]]:
        """
        Get details for a specific job history entry.

        Args:
            history_id: The history entry ID

        Returns:
            Job history entry or None if not found
        """
        try:
            return self.history_repo.get_history_entry(history_id)
        except Exception as e:
            logger.error(f"Failed to get job details: {e}")
            return None

    def get_statistics(self, project_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get job statistics.

        Args:
            project_id: Optional project ID to filter by

        Returns:
            Dictionary with statistics:
            - total_jobs: Total number of jobs
            - completed_count: Number of completed jobs
            - failed_count: Number of failed jobs
            - avg_duration_seconds: Average job duration
        """
        try:
            return self.history_repo.get_job_statistics(project_id)
        except Exception as e:
            logger.error(f"Failed to get job statistics: {e}")
            return {
                'total_jobs': 0,
                'completed_count': 0,
                'failed_count': 0,
                'avg_duration_seconds': 0
            }

    def format_duration(self, seconds: Optional[int]) -> str:
        """
        Format duration in seconds to human-readable string.

        Args:
            seconds: Duration in seconds

        Returns:
            Formatted duration string (e.g., "2m 30s", "1h 15m")
        """
        if not seconds:
            return "N/A"

        if seconds < 60:
            return f"{seconds}s"
        elif seconds < 3600:
            minutes = seconds // 60
            secs = seconds % 60
            return f"{minutes}m {secs}s" if secs else f"{minutes}m"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours}h {minutes}m" if minutes else f"{hours}h"

    def format_job_for_display(self, job: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format a job history entry for GUI display.

        Args:
            job: Raw job history entry

        Returns:
            Formatted job entry with display-friendly fields
        """
        # Parse timestamps
        created_at = job.get('created_at')
        completed_at = job.get('completed_at')
        started_at = job.get('started_at')

        # Format status with emoji
        status = job.get('status', 'unknown')
        status_display = {
            'completed': '✅ Completed',
            'failed': '❌ Failed',
            'cancelled': '🚫 Cancelled'
        }.get(status, f'❓ {status.capitalize()}')

        # Format job type
        job_type = job.get('job_type', 'unknown')
        type_display = {
            'dataset_generation': '📊 Dataset Generation',
            'extraction': '🔍 Extraction',
            'benchmark': '📈 Benchmark',
            'quality': '✨ Quality Analysis'
        }.get(job_type, f'🔧 {job_type.replace("_", " ").title()}')

        # Parse parameters if present
        parameters = job.get('parameters', '{}')
        if isinstance(parameters, str):
            try:
                import json
                parameters = json.loads(parameters)
            except:
                parameters = {}

        # Get result summary
        result_summary = job.get('result_summary')
        if result_summary and isinstance(result_summary, str):
            try:
                import json
                result_summary = json.loads(result_summary)
            except:
                pass

        return {
            'id': job.get('id'),
            'job_type': type_display,
            'job_type_raw': job_type,
            'status': status_display,
            'status_raw': status,
            'project_id': job.get('project_id'),
            'document_id': job.get('document_id'),
            'progress': job.get('progress', 0),
            'duration_seconds': job.get('duration_seconds'),
            'duration_display': self.format_duration(job.get('duration_seconds')),
            'created_at': created_at,
            'completed_at': completed_at,
            'started_at': started_at,
            'parameters': parameters,
            'result_summary': result_summary,
            'error_summary': job.get('error_summary')
        }

    def get_formatted_jobs(self, project_id: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get formatted job history for display.

        Args:
            project_id: Optional project ID to filter by
            limit: Maximum number of entries

        Returns:
            List of formatted job history entries
        """
        if project_id:
            jobs = self.get_project_jobs(project_id, limit)
        else:
            jobs = self.get_recent_jobs(limit)

        return [self.format_job_for_display(job) for job in jobs]


# Singleton instance cache
_service_instances: Dict[str, JobHistoryService] = {}


def get_job_history_service(owner_id: str) -> JobHistoryService:
    """
    Get or create a job history service instance for the owner.

    Args:
        owner_id: The user ID

    Returns:
        JobHistoryService instance
    """
    if owner_id not in _service_instances:
        _service_instances[owner_id] = JobHistoryService(owner_id)
    return _service_instances[owner_id]
