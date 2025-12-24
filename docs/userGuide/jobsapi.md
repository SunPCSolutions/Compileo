# Jobs Module API Usage Guide

The Compileo Jobs API provides comprehensive REST endpoints for real-time job monitoring, status tracking, and job management. This API enables clients to monitor long-running background jobs such as document parsing, chunking, taxonomy generation, and dataset creation.

## Base URL: `/api/v1/jobs`

---

## Job Status Monitoring

### GET `/status/{job_id}`

Get the current status of a specific job.

**Request:**
```bash
curl "http://localhost:8000/api/v1/jobs/status/123e4567-e89b-12d3-a456-426614174000"
```

**Response:**
```json
{
  "job_id": "123e4567-e89b-12d3-a456-426614174000",
  "status": "running",
  "progress": 65.5,
  "current_step": "Processing document 3 of 5",
  "result": null,
  "error": null,
  "created_at": "2024-01-21T10:30:00Z",
  "started_at": "2024-01-21T10:30:05Z",
  "completed_at": null,
  "updated_at": "2024-01-21T10:35:22Z"
}
```

**Status Values:**
- `pending`: Job is queued and waiting to start
- `running`: Job is currently executing
- `completed`: Job finished successfully
- `failed`: Job failed with an error
- `cancelled`: Job was cancelled by user

---

## Long Polling for Status Changes

### GET `/status/{job_id}/poll`

Wait for job status changes with long polling. Returns immediately when status changes or after timeout.

**Request:**
```bash
curl "http://localhost:8000/api/v1/jobs/status/123e4567-e89b-12d3-a456-426614174000/poll?timeout=30&current_status=running"
```

**Query Parameters:**
- `timeout`: Maximum wait time in seconds (default: 30, range: 1-300)
- `current_status`: Current known status - returns immediately if status has changed

**Response:**
Returns the same format as `/status/{job_id}` when status changes or timeout occurs.

---

## Real-Time Status Streaming

### GET `/status/{job_id}/stream`

Server-sent events endpoint for real-time job status updates.

**Request:**
```bash
curl -N "http://localhost:8000/api/v1/jobs/status/123e4567-e89b-12d3-a456-426614174000/stream"
```

**Response:**
Server-sent events stream with JSON data:
```
data: {"job_id": "123e4567-e89b-12d3-a456-426614174000", "status": "running", "progress": 10.0, "current_step": "Initializing", "timestamp": "2024-01-21T10:30:05Z"}

data: {"job_id": "123e4567-e89b-12d3-a456-426614174000", "status": "running", "progress": 25.0, "current_step": "Processing document 1", "timestamp": "2024-01-21T10:31:15Z"}

data: {"job_id": "123e4567-e89b-12d3-a456-426614174000", "status": "completed", "progress": 100.0, "current_step": "Job completed successfully", "timestamp": "2024-01-21T10:35:22Z"}
```

---

## Queue Statistics

### GET `/queue/stats`

Get comprehensive statistics about the job queue system.

**Request:**
```bash
curl "http://localhost:8000/api/v1/jobs/queue/stats"
```

**Response:**
```json
{
  "pending_jobs": 3,
  "running_jobs": 2,
  "scheduled_jobs": 1,
  "completed_jobs": 45,
  "failed_jobs": 2,
  "cancelled_jobs": 1,
  "retrying_jobs": 0,
  "total_jobs": 54,
  "queue_type": "redis",
  "cache_size": 12,
  "cpu_usage_percent": 45.2,
  "memory_usage_mb": 234.8,
  "active_workers": 3,
  "queue_health": {
    "status": "healthy",
    "resource_utilization": "optimal",
    "limit_enforcement": "active"
  }
}
```

---

## Job Management

### POST `/cancel/{job_id}`

Cancel a running or pending job.

**Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/jobs/cancel/123e4567-e89b-12d3-a456-426614174000"
```

**Response:**
```json
{
  "message": "Job 123e4567-e89b-12d3-a456-426614174000 cancelled successfully"
}
```

### POST `/restart/{job_id}`

Restart a failed or cancelled job.

**Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/jobs/restart/123e4567-e89b-12d3-a456-426614174000"
```

**Response:**
```json
{
  "message": "Job 123e4567-e89b-12d3-a456-426614174000 restarted successfully"
}
```

---

## Best Practices

### 1. Job Status Monitoring

**Polling Strategy:**
```python
import requests
import time

def monitor_job_completion(job_id, max_attempts=60, poll_interval=5):
    """Monitor job until completion with exponential backoff."""
    attempt = 0

    while attempt < max_attempts:
        try:
            response = requests.get(f'http://localhost:8000/api/v1/jobs/status/{job_id}')
            response.raise_for_status()

            job_status = response.json()

            print(f"Job {job_id}: {job_status['status']} - {job_status['progress']:.1f}% - {job_status['current_step']}")

            if job_status['status'] in ['completed', 'failed', 'cancelled']:
                return job_status

            # Exponential backoff with jitter
            sleep_time = min(poll_interval * (2 ** attempt), 30)  # Max 30 seconds
            time.sleep(sleep_time)
            attempt += 1

        except requests.exceptions.RequestException as e:
            print(f"Error polling job status: {e}")
            time.sleep(poll_interval)
            attempt += 1

    raise TimeoutError(f"Job {job_id} did not complete within {max_attempts * poll_interval} seconds")
```

**Long Polling Implementation:**
```python
def wait_for_job_change(job_id, current_status=None, timeout=30):
    """Wait for job status change using long polling."""
    params = {'timeout': timeout}
    if current_status:
        params['current_status'] = current_status

    response = requests.get(
        f'http://localhost:8000/api/v1/jobs/status/{job_id}/poll',
        params=params
    )
    response.raise_for_status()

    return response.json()
```

**Real-Time Streaming:**
```python
import json
import requests

def stream_job_updates(job_id):
    """Stream real-time job updates using server-sent events."""
    response = requests.get(
        f'http://localhost:8000/api/v1/jobs/status/{job_id}/stream',
        stream=True
    )

    for line in response.iter_lines():
        if line.startswith(b'data: '):
            data = line[6:].decode('utf-8')  # Remove 'data: ' prefix
            try:
                update = json.loads(data)
                print(f"Job Update: {update}")

                # Handle different update types
                if 'error' in update:
                    print(f"Error: {update['error']}")
                    break
                elif update.get('status') in ['completed', 'failed', 'cancelled']:
                    print("Job finished")
                    break

            except json.JSONDecodeError:
                continue
```

### 2. Error Handling and Recovery

**Robust Job Monitoring:**
```python
def monitor_job_with_recovery(job_id):
    """Monitor job with automatic error recovery."""
    try:
        # First, try to get initial status
        status_response = requests.get(f'http://localhost:8000/api/v1/jobs/status/{job_id}')
        status_response.raise_for_status()
        job_status = status_response.json()

        # If job failed, try to restart
        if job_status['status'] == 'failed':
            print(f"Job {job_id} failed: {job_status.get('error', 'Unknown error')}")
            restart_response = requests.post(f'http://localhost:8000/api/v1/jobs/restart/{job_id}')
            if restart_response.status_code == 200:
                print("Job restarted successfully")
                return monitor_job_completion(job_id)
            else:
                print("Failed to restart job")

        # If job is cancelled, ask user if they want to restart
        elif job_status['status'] == 'cancelled':
            user_input = input(f"Job {job_id} was cancelled. Restart? (y/n): ")
            if user_input.lower() == 'y':
                restart_response = requests.post(f'http://localhost:8000/api/v1/jobs/restart/{job_id}')
                if restart_response.status_code == 200:
                    return monitor_job_completion(job_id)

        # For running or pending jobs, monitor normally
        elif job_status['status'] in ['running', 'pending']:
            return monitor_job_completion(job_id)

        # For completed jobs, return status
        elif job_status['status'] == 'completed':
            return job_status

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            print(f"Job {job_id} not found")
        else:
            print(f"HTTP error: {e}")
    except requests.exceptions.ConnectionError:
        print("Connection error - check if API server is running")
    except Exception as e:
        print(f"Unexpected error: {e}")

    return None
```

### 3. Batch Job Management

**Monitor Multiple Jobs:**
```python
def monitor_multiple_jobs(job_ids):
    """Monitor multiple jobs concurrently."""
    import threading
    import queue

    results = {}
    result_queue = queue.Queue()

    def monitor_single_job(job_id):
        try:
            result = monitor_job_completion(job_id)
            result_queue.put((job_id, result))
        except Exception as e:
            result_queue.put((job_id, {'error': str(e)}))

    # Start monitoring threads
    threads = []
    for job_id in job_ids:
        thread = threading.Thread(target=monitor_single_job, args=(job_id,))
        thread.start()
        threads.append(thread)

    # Collect results
    for _ in range(len(job_ids)):
        job_id, result = result_queue.get()
        results[job_id] = result

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    return results
```

**Queue Health Monitoring:**
```python
def check_queue_health():
    """Check overall queue health and performance."""
    try:
        response = requests.get('http://localhost:8000/api/v1/jobs/queue/stats')
        response.raise_for_status()
        stats = response.json()

        # Check for concerning metrics
        alerts = []

        if stats['failed_jobs'] > stats['total_jobs'] * 0.1:  # >10% failure rate
            alerts.append(f"High failure rate: {stats['failed_jobs']}/{stats['total_jobs']} jobs failed")

        if stats['cpu_usage_percent'] > 90:
            alerts.append(f"High CPU usage: {stats['cpu_usage_percent']}%")

        if stats['memory_usage_mb'] > 1000:  # >1GB
            alerts.append(f"High memory usage: {stats['memory_usage_mb']}MB")

        if stats['pending_jobs'] > 10:
            alerts.append(f"Large pending queue: {stats['pending_jobs']} jobs waiting")

        if not alerts:
            print("Queue health: GOOD")
            print(f"Active jobs: {stats['running_jobs']}")
            print(f"Workers: {stats.get('active_workers', 'unknown')}")
        else:
            print("Queue health: WARNING")
            for alert in alerts:
                print(f"  - {alert}")

        return stats

    except Exception as e:
        print(f"Failed to check queue health: {e}")
        return None
```

### 4. Job Lifecycle Management

**Complete Job Workflow:**
```python
def submit_and_monitor_job(job_type, parameters):
    """Complete workflow: submit job, monitor progress, handle results."""
    # This would typically be called after submitting a job via another API
    # For example, after calling document processing API

    # Assume job_id is obtained from job submission
    job_id = "example-job-id-from-submission"

    print(f"Monitoring job {job_id}...")

    # Monitor job completion
    final_status = monitor_job_completion(job_id)

    if final_status['status'] == 'completed':
        print("Job completed successfully!")
        if final_status.get('result'):
            print(f"Result: {final_status['result']}")

        # Process results based on job type
        if job_type == 'document_processing':
            print("Documents processed successfully")
        elif job_type == 'taxonomy_generation':
            print("Taxonomy generated successfully")
        elif job_type == 'dataset_generation':
            print("Dataset created successfully")

    elif final_status['status'] == 'failed':
        print(f"Job failed: {final_status.get('error', 'Unknown error')}")
        # Handle failure - retry, notify user, etc.

    elif final_status['status'] == 'cancelled':
        print("Job was cancelled")
        # Handle cancellation

    return final_status
```

### 5. Performance Optimization

**Efficient Polling Strategies:**
```python
def adaptive_polling(job_id, initial_interval=1, max_interval=30):
    """Adapt polling frequency based on job progress."""
    import time

    last_progress = 0
    interval = initial_interval

    while True:
        response = requests.get(f'http://localhost:8000/api/v1/jobs/status/{job_id}')
        job_status = response.json()

        current_progress = job_status['progress']

        # If progress changed significantly, reset to faster polling
        if current_progress - last_progress > 5:  # 5% progress change
            interval = max(initial_interval, interval / 2)
        else:
            # Gradually slow down polling
            interval = min(interval * 1.2, max_interval)

        print(f"Progress: {current_progress:.1f}% - Next check in {interval:.1f}s")

        if job_status['status'] in ['completed', 'failed', 'cancelled']:
            return job_status

        time.sleep(interval)
        last_progress = current_progress
```

**Connection Pooling for High-Frequency Monitoring:**
```python
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def create_resilient_session():
    """Create HTTP session with connection pooling and retries."""
    session = requests.Session()

    # Configure retries
    retry_strategy = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504]
    )

    # Configure adapter with connection pooling
    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=10,
        pool_maxsize=20
    )

    session.mount("http://", adapter)
    session.mount("https://", adapter)

    return session

# Use resilient session for monitoring
session = create_resilient_session()
response = session.get(f'http://localhost:8000/api/v1/jobs/status/{job_id}')
```

This API provides comprehensive job monitoring and management capabilities essential for tracking long-running background operations in the Compileo system.