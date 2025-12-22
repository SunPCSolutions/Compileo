"""
Job Handle Module - Phase 3: Background Processing System

This module provides comprehensive background job processing capabilities for Compileo,
including job queuing, scheduling, resource management, and monitoring.

Components:
- enhanced_job_queue: Core job queue system with Redis/RQ integration
- worker_manager: Worker process management and auto-scaling
- job_management: GUI interface for job monitoring and management

Key Features:
- Production-ready job queuing with Redis/RQ
- Resource management and load balancing
- Job scheduling and dependency management
- Real-time monitoring and analytics
- Comprehensive error handling and recovery
- Auto-scaling worker management
"""

from .models import (
    JobStatus, JobPriority, JobType, ResourceType,
    ResourceLimits, JobDependency, JobSchedule, JobMetrics,
    EnhancedExtractionJob
)

from .enhanced_job_queue import (
    enhanced_job_queue_manager,
    submit_extraction_job,
    submit_batch_extraction_job,
    submit_taxonomy_processing_job,
    submit_scheduled_job,
    perform_health_check,
    cleanup_old_jobs
)

from .worker_manager import (
    WorkerManager,
    WorkerPool,
    create_worker_manager,
    start_worker_process
)

__version__ = "1.0.0"
__all__ = [
    # Core job queue
    "enhanced_job_queue_manager",
    "EnhancedExtractionJob",
    "JobStatus",
    "JobPriority",
    "JobType",
    "ResourceLimits",
    "JobDependency",
    "JobSchedule",
    "JobMetrics",

    # Convenience functions
    "submit_extraction_job",
    "submit_batch_extraction_job",
    "submit_taxonomy_processing_job",
    "submit_scheduled_job",

    # Health and maintenance
    "perform_health_check",
    "cleanup_old_jobs",

    # Worker management
    "WorkerManager",
    "WorkerPool",
    "create_worker_manager",
    "start_worker_process",
]