# Job Handling Module API Usage Guide

The Compileo Job Handling API provides comprehensive endpoints for monitoring, managing, and controlling background jobs using an enhanced Redis-based queue system with RQ (Redis Queue).

## Base URL: `/api/v1/jobs`

## Overview

The job handling system uses Redis Queue (RQ) for reliable background job processing with the following enhancements:

- **Datetime Compatibility**: Proper handling of timezone-aware timestamps
- **Worker Health Monitoring**: Automatic cleanup of stale worker processes
- **Comprehensive Job Cleanup**: Multi-level cleanup of jobs, registries, and locks
- **Real-time Statistics**: Accurate job counts and system monitoring
- **Duplicate Prevention**: Atomic status updates prevent job re-execution
- **Resource Limit Enforcement**: Strict checking of global and per-user concurrency limits (jobs queue instead of failing)

---

## Get Job Status

### GET `/status/{job_id}`

Retrieve the current status and details of a specific job.

**Request:**
```bash
curl "http://localhost:8000/api/v1/jobs/status/550e8400-e29b-41d4-a716-446655440000"
```

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "job_type": "extraction",
  "status": "running",
  "progress": 0.75,
  "created_at": "2025-11-12T15:30:00Z",
  "started_at": "2025-11-12T15:30:05Z",
  "completed_at": null,
  "error": null,
  "user_id": "default_user",
  "metrics": {
    "execution_time_seconds": 45.2,
    "items_processed": 75,
    "total_items": 100
  }
}
```

**Status Values:**
- `pending`: Job queued, waiting for worker
- `running`: Job currently being processed
- `completed`: Job finished successfully
- `failed`: Job failed with error
- `cancelled`: Job manually cancelled
- `scheduled`: Job scheduled for future execution

---

## Poll Job Status for Changes

### GET `/status/{job_id}/poll`

Long-polling endpoint that waits for status changes or timeout.

**Parameters:**
- `current_status` (optional): Current known status - returns immediately if different
- `timeout` (optional, default: 30): Maximum seconds to wait

**Request:**
```bash
curl "http://localhost:8000/api/v1/jobs/status/550e8400-e29b-41d4-a716-446655440000/poll?timeout=60"
```

---

## Stream Job Status (Server-Sent Events)

### GET `/status/{job_id}/stream`

Real-time job status updates via Server-Sent Events.

**Response (text/event-stream):**
```
event: job_update
data: {"job_id": "550e8400-e29b-41d4-a716-446655440000", "status": "running", "progress": 0.1}

event: job_update
data: {"job_id": "550e8400-e29b-41d4-a716-446655440000", "status": "running", "progress": 0.5}

event: job_complete
data: {"job_id": "550e8400-e29b-41d4-a716-446655440000", "status": "completed", "result": {...}}
```

---

## Get Queue Statistics

### GET `/queue/stats`

Comprehensive queue statistics and system health metrics.

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
  "completed_jobs": 150,
  "failed_jobs": 2,
  "cancelled_jobs": 1,
  "total_jobs": 159,
  "queue_type": "redis",
  "cache_size": 100,
  "active_workers": 2,
  "cpu_usage_percent": 45.2,
  "memory_usage_mb": 850,
  "global_max_concurrent_jobs": 10,
  "per_user_max_concurrent_jobs": 3,
  "worker_health_status": "healthy",
  "last_cleanup": "2025-11-12T15:25:00Z"
}
```

---

## Cancel Job

### POST `/cancel/{job_id}`

Cancel a pending or running job.

**Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/jobs/cancel/550e8400-e29b-41d4-a716-446655440000"
```

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "cancelled",
  "message": "Job cancelled successfully"
}
```

---

## Restart Failed Job

### POST `/restart/{job_id}`

Restart a failed or cancelled job with fresh parameters.

**Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/jobs/restart/550e8400-e29b-41d4-a716-446655440000"
```

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "message": "Job restarted and re-queued"
}
```

---

## RQ System Features

### Enhanced Reliability Features

**1. Datetime Compatibility**
- Automatic handling of timezone-aware vs naive timestamps
- Prevents "can't subtract offset-naive and offset-aware datetimes" errors
- Compatible with RQ's internal timestamp handling

**2. Worker Health Monitoring**
- Continuous monitoring of worker processes
- Automatic cleanup of stale workers (>5 minutes old)
- Prevents accumulation of dead worker registrations

**3. Comprehensive Job Cleanup**
- Multi-level cleanup every 10 minutes:
  - RQ failed jobs registry (immediate cleanup)
  - RQ finished jobs registry (24-hour retention)
  - Custom job storage (2-hour retention)
  - Orphaned processing locks (10-minute cleanup)
- Startup cleanup removes jobs older than 1 hour

**4. Duplicate Execution Prevention**
- Atomic status updates in Redis prevent race conditions
- Early status validation before job execution
- UUID-based job IDs ensure uniqueness
- Processing locks prevent concurrent execution

### Job Types Supported

- **extraction**: Taxonomy-based content extraction
- **document_processing**: Parse and chunk documents
- **taxonomy_processing**: Generate taxonomies from chunks
- **dataset_generation**: Create datasets from extractions

### Error Handling

**Common Error Responses:**
```json
{
  "error": "JobNotFoundError",
  "message": "Job 550e8400-e29b-41d4-a716-446655440000 not found",
  "job_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

```json
{
  "error": "JobAlreadyCompletedError",
  "message": "Cannot cancel job that is already completed",
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "current_status": "completed"
}
```

### Performance Characteristics

- **Job Submission**: < 100ms average latency
- **Status Queries**: < 50ms average response time
- **Queue Throughput**: 100+ jobs/minute sustained
- **Memory Usage**: < 2GB per worker process
- **Cleanup Frequency**: Every 10 minutes (comprehensive)
- **Worker Monitoring**: Every 5 minutes (health checks)

### Best Practices

1. **Use Streaming for Real-time Updates**: Prefer `/stream` endpoint for live job monitoring
2. **Implement Proper Error Handling**: Always check response status and handle errors gracefully
3. **Monitor Queue Statistics**: Use `/queue/stats` for system health monitoring
4. **Cancel Unnecessary Jobs**: Clean up jobs that are no longer needed
5. **Handle Timeouts**: Implement client-side timeouts for long-running operations

### Troubleshooting

**Job Stuck in Pending:**
- Check worker health: `GET /queue/stats` - verify active workers > 0
- Check Redis connectivity
- Review worker logs for errors

**Job Shows Incorrect Status:**
- Wait for cache refresh (30 seconds) or force refresh
- Check if cleanup recently ran (may have updated statistics)
- Verify job ID format (should be UUID)

**High Memory Usage:**
- Check for worker accumulation: stale workers not cleaned up
- Monitor job cleanup frequency
- Review large job result storage