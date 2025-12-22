"""
Job Handle Models - Data structures and enums for the job queue system.

This module defines all the core data models, enums, and dataclasses used
throughout the job handle system.
"""

from typing import Dict, Any, Optional, List, Set
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field


class JobStatus(str, Enum):
    """Job status enumeration."""
    PENDING = "pending"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class JobPriority(str, Enum):
    """Job priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class JobType(str, Enum):
    """Job type enumeration."""
    DOCUMENT_PROCESSING = "document_processing"
    EXTRACTION = "extraction"
    BATCH_EXTRACTION = "batch_extraction"
    TAXONOMY_PROCESSING = "taxonomy_processing"
    DATASET_GENERATION = "dataset_generation"
    BENCHMARKING = "benchmarking"
    CLEANUP = "cleanup"
    MAINTENANCE = "maintenance"


class ResourceType(str, Enum):
    """Resource type enumeration."""
    CPU = "cpu"
    MEMORY = "memory"
    API_RATE = "api_rate"
    DISK_IO = "disk_io"


@dataclass
class ResourceLimits:
    """
    Resource limits for job execution.

    Defines the maximum resource usage allowed for a job to prevent
    system overload and ensure fair resource allocation.
    """
    max_cpu_percent: float = 80.0
    max_memory_mb: int = 4096  # Increased from 1024MB to 4096MB (4GB) for development
    max_gpu_memory_mb: int = 8192  # 8GB GPU memory limit for ML models
    max_api_calls_per_minute: int = 60
    max_concurrent_jobs: int = 5


@dataclass
class JobDependency:
    """
    Job dependency specification.

    Defines relationships between jobs where one job must complete
    before another can start.
    """
    job_id: str
    dependency_type: str = "completion"  # completion, success, failure
    required: bool = True


@dataclass
class JobSchedule:
    """
    Job scheduling configuration.

    Controls when and how jobs are executed, including timing,
    retries, and timeouts.
    """
    scheduled_time: Optional[datetime] = None
    cron_expression: Optional[str] = None
    interval_seconds: Optional[int] = None
    max_retries: int = 3
    retry_delay_seconds: int = 60
    timeout_seconds: Optional[int] = 259200  # 72 hours default for long-running jobs


@dataclass
class JobMetrics:
    """
    Job execution metrics.

    Tracks performance and resource usage statistics during job execution.
    """
    cpu_usage_percent: float = 0.0
    memory_usage_mb: int = 0
    execution_time_seconds: float = 0.0
    api_calls_made: int = 0
    items_processed: int = 0
    errors_count: int = 0
    retry_count: int = 0


class EnhancedExtractionJob:
    """
    Enhanced extraction job with advanced features.

    Represents a background job with comprehensive tracking, scheduling,
    dependencies, and resource management capabilities.
    """

    def __init__(
        self,
        job_id: str,
        job_type: str,  # Union[str, JobType] for backward compatibility
        parameters: Dict[str, Any],
        priority: JobPriority = JobPriority.NORMAL,
        created_at: Optional[datetime] = None,
        schedule: Optional[JobSchedule] = None,
        dependencies: Optional[List[JobDependency]] = None,
        resource_limits: Optional[ResourceLimits] = None,
        user_id: Optional[str] = None
    ):
        """
        Initialize an enhanced extraction job.

        Args:
            job_id: Unique identifier for the job
            job_type: Type of job (extraction, batch_extraction, etc.)
            parameters: Job-specific parameters
            priority: Job execution priority
            created_at: Job creation timestamp
            schedule: Scheduling configuration
            dependencies: List of job dependencies
            resource_limits: Resource usage limits
            user_id: ID of the user who submitted the job
        """
        self.job_id = job_id
        self.job_type = job_type
        self.parameters = parameters
        self.priority = priority
        self.status = JobStatus.PENDING
        self.created_at = created_at or datetime.utcnow()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.result: Optional[Any] = None
        self.error: Optional[str] = None
        self.progress = 0.0
        self.metadata: Dict[str, Any] = {}
        self.user_id = user_id

        # Enhanced features
        self.schedule = schedule or JobSchedule()
        self.dependencies = dependencies or []
        self.resource_limits = resource_limits or ResourceLimits()
        self.metrics = JobMetrics()
        self.retry_count = 0
        self.worker_id: Optional[str] = None
        self.parent_job_id: Optional[str] = None
        self.child_jobs: List[str] = []
        self.tags: Set[str] = set()
        self.cache_key: Optional[str] = None
        self.last_heartbeat: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert job to dictionary representation.

        Returns:
            Dictionary containing all job data for serialization
        """
        return {
            'job_id': self.job_id,
            'job_type': self.job_type.value if isinstance(self.job_type, JobType) else self.job_type,
            'parameters': self.parameters,
            'priority': self.priority.value,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'result': self.result,
            'error': self.error,
            'progress': self.progress,
            'metadata': self.metadata,
            'user_id': self.user_id,
            'schedule': {
                'scheduled_time': self.schedule.scheduled_time.isoformat() if self.schedule.scheduled_time else None,
                'cron_expression': self.schedule.cron_expression,
                'interval_seconds': self.schedule.interval_seconds,
                'max_retries': self.schedule.max_retries,
                'retry_delay_seconds': self.schedule.retry_delay_seconds,
                'timeout_seconds': self.schedule.timeout_seconds
            } if self.schedule else None,
            'dependencies': [{'job_id': d.job_id, 'dependency_type': d.dependency_type, 'required': d.required} for d in self.dependencies],
            'resource_limits': {
                'max_cpu_percent': self.resource_limits.max_cpu_percent,
                'max_memory_mb': self.resource_limits.max_memory_mb,
                'max_gpu_memory_mb': self.resource_limits.max_gpu_memory_mb,
                'max_api_calls_per_minute': self.resource_limits.max_api_calls_per_minute,
                'max_concurrent_jobs': self.resource_limits.max_concurrent_jobs
            } if self.resource_limits else None,
            'metrics': {
                'cpu_usage_percent': self.metrics.cpu_usage_percent,
                'memory_usage_mb': self.metrics.memory_usage_mb,
                'execution_time_seconds': self.metrics.execution_time_seconds,
                'api_calls_made': self.metrics.api_calls_made,
                'items_processed': self.metrics.items_processed,
                'errors_count': self.metrics.errors_count,
                'retry_count': self.retry_count
            },
            'worker_id': self.worker_id,
            'parent_job_id': self.parent_job_id,
            'child_jobs': self.child_jobs,
            'tags': list(self.tags),
            'cache_key': self.cache_key,
            'last_heartbeat': self.last_heartbeat.isoformat() if self.last_heartbeat else None
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EnhancedExtractionJob':
        """
        Create job from dictionary representation.

        Args:
            data: Dictionary containing job data

        Returns:
            EnhancedExtractionJob instance
        """
        # Parse schedule
        schedule_data = data.get('schedule')
        schedule = None
        if schedule_data:
            schedule = JobSchedule()
            if schedule_data.get('scheduled_time'):
                schedule.scheduled_time = datetime.fromisoformat(schedule_data['scheduled_time'])
            schedule.cron_expression = schedule_data.get('cron_expression')
            schedule.interval_seconds = schedule_data.get('interval_seconds')
            schedule.max_retries = schedule_data.get('max_retries', 3)
            schedule.retry_delay_seconds = schedule_data.get('retry_delay_seconds', 60)
            schedule.timeout_seconds = schedule_data.get('timeout_seconds')

        # Parse dependencies
        dependencies_data = data.get('dependencies', [])
        dependencies = [JobDependency(**dep) for dep in dependencies_data]

        # Parse resource limits
        resource_limits_data = data.get('resource_limits')
        resource_limits = None
        if resource_limits_data:
            resource_limits = ResourceLimits(**resource_limits_data)

        job = cls(
            job_id=data['job_id'],
            job_type=JobType(data['job_type']),
            parameters=data['parameters'],
            priority=JobPriority(data['priority']),
            created_at=datetime.fromisoformat(data['created_at']),
            schedule=schedule,
            dependencies=dependencies,
            resource_limits=resource_limits,
            user_id=data.get('user_id')
        )

        job.status = JobStatus(data['status'])
        job.started_at = datetime.fromisoformat(data['started_at']) if data.get('started_at') else None
        job.completed_at = datetime.fromisoformat(data['completed_at']) if data.get('completed_at') else None
        job.result = data.get('result')
        job.error = data.get('error')
        job.progress = data.get('progress', 0.0)
        job.metadata = data.get('metadata', {})

        # Load metrics
        metrics_data = data.get('metrics', {})
        job.metrics = JobMetrics(**metrics_data)
        job.retry_count = metrics_data.get('retry_count', 0)

        job.worker_id = data.get('worker_id')
        job.parent_job_id = data.get('parent_job_id')
        job.child_jobs = data.get('child_jobs', [])
        job.tags = set(data.get('tags', []))
        job.cache_key = data.get('cache_key')
        job.last_heartbeat = datetime.fromisoformat(data['last_heartbeat']) if data.get('last_heartbeat') else None

        return job

    def can_start(self, completed_jobs: Set[str]) -> bool:
        """
        Check if job can start based on dependencies.

        Args:
            completed_jobs: Set of completed job IDs

        Returns:
            True if all dependencies are satisfied
        """
        for dep in self.dependencies:
            if dep.required:
                if dep.dependency_type == "completion":
                    if dep.job_id not in completed_jobs:
                        return False
                elif dep.dependency_type == "success":
                    # Would need job status lookup - simplified for now
                    pass
        return True

    def update_progress(self, progress: float, message: str = "") -> None:
        """
        Update job progress.

        Args:
            progress: Progress percentage (0-100)
            message: Optional progress message
        """
        self.progress = max(0.0, min(100.0, progress))
        if message:
            self.metadata['progress_message'] = message
        self.last_heartbeat = datetime.utcnow()

    def update_metrics(self, **kwargs) -> None:
        """
        Update job metrics.

        Args:
            **kwargs: Metric name-value pairs to update
        """
        for key, value in kwargs.items():
            if hasattr(self.metrics, key):
                setattr(self.metrics, key, value)

    def should_retry(self) -> bool:
        """
        Check if job should be retried.

        Returns:
            True if job should be retried based on configuration
        """
        return self.retry_count < self.schedule.max_retries and self.status == JobStatus.FAILED


# Export all classes and enums
__all__ = [
    'JobStatus',
    'JobPriority',
    'JobType',
    'ResourceType',
    'ResourceLimits',
    'JobDependency',
    'JobSchedule',
    'JobMetrics',
    'EnhancedExtractionJob'
]