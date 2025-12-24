# Job Handle Module - Phase 3: Background Processing System

## Overview

The Job Handle module provides a comprehensive, production-ready background job processing system for Compileo. It implements advanced job queuing with Redis/RQ integration, resource management, scheduling, dependency handling, and real-time monitoring.

## Architecture

### Core Components

#### 1. Models (`models.py`)
- **EnhancedExtractionJob**: Core job class with advanced features
- **JobStatus, JobPriority, JobType**: Enumeration classes for job metadata
- **ResourceLimits, JobSchedule, JobDependency**: Configuration dataclasses
- **JobMetrics**: Performance tracking and monitoring

#### 2. Queue System (`enhanced_job_queue.py`)
- **EnhancedRedisJobQueue**: Production Redis-based queue with RQ
- **EnhancedInMemoryJobQueue**: Development/testing in-memory queue
- **EnhancedJobQueueManager**: Central manager with auto-selection

#### 3. Worker Management (`worker_manager.py`)
- **WorkerManager**: Individual worker process management
- **WorkerPool**: Multi-worker coordination and scaling
- **Auto-scaling**: Dynamic worker adjustment based on load

#### 4. GUI Interface (`job_management.py`)
- **JobManagementView**: Streamlit-based monitoring dashboard
- **Real-time metrics**: Live job status and system health
- **Management controls**: Job cancellation, restart, and configuration

## Key Features

### Job Management
- **Priority System**: LOW, NORMAL, HIGH, URGENT priority levels
- **Job Types**: Extraction, batch extraction, taxonomy processing, cleanup, maintenance
- **Status Tracking**: Pending, scheduled, running, completed, failed, cancelled, retrying
- **Progress Monitoring**: Real-time progress updates with detailed metrics

### Scheduling & Dependencies
- **Time-based Scheduling**: Execute jobs at specific times or intervals
- **Cron Expressions**: Flexible recurring job scheduling
- **Job Dependencies**: Define prerequisite relationships between jobs
- **Dependency Types**: Completion, success, or failure-based triggers

### Resource Management
- **CPU Monitoring**: Prevent system overload with configurable limits
- **Memory Management**: Track and limit memory usage per job
- **API Rate Limiting**: Control external API call frequencies
- **Concurrent Job Limits**: Prevent resource contention

### Reliability & Recovery
- **Retry Mechanisms**: Exponential backoff for failed jobs
- **Error Handling**: Comprehensive error classification and recovery
- **Job Persistence**: Database-backed job state storage
- **Graceful Shutdown**: Clean worker termination and job recovery

### Monitoring & Analytics
- **Real-time Dashboard**: Live system health and job status
- **Performance Metrics**: CPU, memory, execution time tracking
- **Analytics Reports**: Success rates, duration trends, failure analysis
- **Health Checks**: Automated system health monitoring

## Usage Examples

### Basic Job Submission

```python
from src.compileo.features.jobhandle import (
    submit_extraction_job,
    JobPriority,
    JobSchedule
)

# Simple extraction job
job_id = submit_extraction_job(
    parameters={"document_id": 123, "taxonomy_id": 456},
    priority=JobPriority.HIGH
)

# Scheduled job with dependencies
schedule = JobSchedule(
    scheduled_time=datetime.utcnow() + timedelta(hours=2),
    max_retries=3
)

job_id = submit_extraction_job(
    parameters={"batch_size": 100},
    schedule=schedule,
    priority=JobPriority.NORMAL
)
```

### Worker Management

```python
from src.compileo.features.jobhandle import (
    create_worker_manager,
    WorkerPool
)

# Create worker manager
worker_mgr = create_worker_manager(
    worker_name_prefix="extraction_worker",
    scaling_config=ScalingConfig(min_workers=2, max_workers=10)
)

# Start auto-scaling
worker_mgr.start_auto_scaling()

# Or use worker pool for multiple queues
pool = WorkerPool()
pool.add_manager("extraction_queue", worker_mgr)
pool.start_all()
```

### GUI Integration

```python
from src.compileo.features.jobhandle.job_management import render_job_management

# In Streamlit app
def main():
    st.title("Compileo Job Management")
    render_job_management()
```

## Configuration

### Environment Variables

```bash
# Redis Configuration
REDIS_URL=redis://localhost:6379/0

# Worker Settings
JOB_QUEUE_WORKERS_MIN=2
JOB_QUEUE_WORKERS_MAX=10
JOB_QUEUE_CPU_THRESHOLD=80
JOB_QUEUE_MEMORY_THRESHOLD=85

# Job Settings
JOB_DEFAULT_TIMEOUT=259200
JOB_MAX_RETRIES=3
JOB_RETRY_DELAY=60

# Resource Limits
JOB_MAX_CPU_PERCENT=80
JOB_MAX_MEMORY_MB=1024
JOB_MAX_API_CALLS_PER_MINUTE=60
```

### Programmatic Configuration

```python
from src.compileo.features.jobhandle.models import ResourceLimits, JobSchedule

# Custom resource limits
limits = ResourceLimits(
    max_cpu_percent=70.0,
    max_memory_mb=2048,
    max_api_calls_per_minute=100
)

# Job scheduling
schedule = JobSchedule(
    cron_expression="0 */2 * * *",  # Every 2 hours
    max_retries=5,
    timeout_seconds=7200
)
```

## API Integration

### REST Endpoints

The module integrates with the existing FastAPI extraction endpoints:

- `POST /api/extraction/` - Submit extraction jobs
- `GET /api/extraction/{job_id}` - Get job status
- `DELETE /api/extraction/{job_id}` - Cancel jobs
- `POST /api/extraction/{job_id}/restart` - Restart failed jobs

### Enhanced Features

- **Background Processing**: Jobs run asynchronously with FastAPI BackgroundTasks
- **Progress Tracking**: Real-time progress updates via WebSocket or polling
- **Resource Monitoring**: API endpoints for system resource usage
- **Health Monitoring**: System health check endpoints

## Deployment

### Development Setup

```bash
# Install dependencies
pip install redis rq rq-scheduler psutil

# Start Redis (if using Docker)
docker run -d -p 6379:6379 redis:alpine

# Run development server
uvicorn src.compileo.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Production Deployment

```bash
# Start workers
python -m src.compileo.features.jobhandle.worker_manager start-worker \
    --name prod_worker_1 \
    --queue extraction_jobs \
    --redis redis://prod-redis:6379/0

# Start auto-scaling manager
python -c "
from src.compileo.features.jobhandle import create_worker_manager
mgr = create_worker_manager()
mgr.start_auto_scaling()
"

# Start GUI (optional)
streamlit run src/compileo/features/jobhandle/job_management.py
```

### Docker Compose

```yaml
version: '3.8'
services:
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"

  job-worker:
    build: .
    command: python -m src.compileo.features.jobhandle.worker_manager start-worker --name worker --queue extraction_jobs
    depends_on:
      - redis
    environment:
      - REDIS_URL=redis://redis:6379/0

  api:
    build: .
    ports:
      - "8000:8000"
    depends_on:
      - redis
    environment:
      - REDIS_URL=redis://redis:6379/0
```

## Monitoring & Maintenance

### Health Checks

```python
from src.compileo.features.jobhandle import perform_health_check

health = perform_health_check()
print(f"Status: {health['status']}")
print(f"Issues: {health['issues']}")
```

### Maintenance Tasks

```python
from src.compileo.features.jobhandle import cleanup_old_jobs

# Clean jobs older than 30 days
cleaned = cleanup_old_jobs(days_old=30)
print(f"Cleaned {cleaned} old jobs")
```

### Performance Profiling

```python
from src.compileo.features.jobhandle import enhanced_job_queue_manager

stats = enhanced_job_queue_manager.get_queue_stats()
print(f"Active jobs: {stats['running_jobs']}")
print(f"CPU usage: {stats['cpu_usage_percent']}%")
print(f"Cache size: {stats['cache_size']}")
```

## Testing

### Unit Tests

```python
import pytest
from src.compileo.features.jobhandle.models import EnhancedExtractionJob, JobStatus

def test_job_creation():
    job = EnhancedExtractionJob(
        job_id="test_123",
        job_type="extraction",
        parameters={"test": True}
    )
    assert job.status == JobStatus.PENDING
    assert job.progress == 0.0
```

### Integration Tests

```python
def test_job_queue_integration():
    # Submit job
    job_id = submit_extraction_job({"test": True})

    # Check status
    job = enhanced_job_queue_manager.get_job(job_id)
    assert job is not None
    assert job.status in [JobStatus.PENDING, JobStatus.RUNNING]
```

## Troubleshooting

### Common Issues

1. **Redis Connection Failed**
   - Check Redis server is running
   - Verify REDIS_URL environment variable
   - Ensure network connectivity

2. **Jobs Stuck in Pending**
   - Check worker processes are running
   - Verify queue name matches
   - Check resource limits

3. **High Memory Usage**
   - Reduce concurrent jobs
   - Increase resource limits
   - Check for memory leaks in job handlers

4. **Job Failures**
   - Check job error messages
   - Verify input parameters
   - Review dependency requirements

### Debug Commands

```bash
# Check Redis queue status
redis-cli -h localhost -p 6379 LLEN extraction_jobs

# View worker processes
ps aux | grep "worker_manager"

# Check job logs
tail -f /var/log/compileo/job_queue.log
```

## Contributing

### Code Standards

- Follow PEP 8 style guidelines
- Add type hints for all functions
- Write comprehensive docstrings
- Include unit tests for new features
- Update documentation for API changes

### Development Workflow

1. Create feature branch
2. Implement changes with tests
3. Run full test suite
4. Update documentation
5. Submit pull request

## License

This module is part of the Compileo project and follows the same license terms.