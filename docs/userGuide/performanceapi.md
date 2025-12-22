# Performance Optimization API in Compileo

## Overview

The Compileo Performance API provides comprehensive optimization capabilities for extraction workflows, including caching, job queue management, lazy loading, cleanup operations, and performance monitoring.

## Base URL: `/api/v1/performance`

---

## 1. Cache Management

### Get Cache Statistics

**Endpoint:** `GET /cache/stats`

**Description:** Retrieves comprehensive statistics for all cache systems.

**Success Response (200 OK):**
```json
{
  "result_cache": {
    "total_entries": 150,
    "total_size_bytes": 2048576,
    "hit_rate": 0.85,
    "miss_rate": 0.15,
    "avg_access_time_ms": 2.3
  },
  "search_cache": {
    "total_entries": 75,
    "total_size_bytes": 1024000,
    "hit_rate": 0.92,
    "miss_rate": 0.08,
    "avg_access_time_ms": 1.8
  }
}
```

### Clear All Caches

**Endpoint:** `POST /cache/clear`

**Description:** Clears all cache systems to free memory and ensure fresh data.

**Success Response (200 OK):**
```json
{
  "message": "All caches cleared successfully"
}
```

---

## 2. Job Queue Management

### Submit Job

**Endpoint:** `POST /jobs/submit`

**Description:** Submits a new job to the enhanced job queue with priority support.

**Request Body:**
```json
{
  "job_type": "extraction",
  "parameters": {
    "taxonomy_id": 123,
    "selected_categories": ["category1", "category2"]
  },
  "priority": "high"
}
```

**Parameters:**
- `job_type` (string): Type of job (extraction, analysis, etc.)
- `parameters` (object): Job-specific parameters
- `priority` (string, optional): Job priority (low, normal, high, urgent)

**Success Response (200 OK):**
```json
{
  "job_id": "job_12345",
  "status": "submitted",
  "message": "Job job_12345 submitted successfully"
}
```

### Get Job Status

**Endpoint:** `GET /jobs/{job_id}`

**Description:** Retrieves current status and details of a specific job.

**Path Parameters:**
- `job_id` (string, required): Job identifier

**Success Response (200 OK):**
```json
{
  "job_id": "job_12345",
  "job_type": "extraction",
  "status": "running",
  "priority": "normal",
  "progress": 0.65,
  "created_at": "2024-01-15T10:30:00Z",
  "started_at": "2024-01-15T10:31:00Z",
  "parameters": {...}
}
```

### Cancel Job

**Endpoint:** `DELETE /jobs/{job_id}`

**Description:** Cancels a pending or running job.

**Path Parameters:**
- `job_id` (string, required): Job identifier

**Success Response (200 OK):**
```json
{
  "message": "Job job_12345 cancelled successfully"
}
```

### Get Queue Statistics

**Endpoint:** `GET /jobs/queue/stats`

**Description:** Retrieves comprehensive job queue statistics and performance metrics.

**Success Response (200 OK):**
```json
{
  "pending_jobs": 5,
  "running_jobs": 3,
  "completed_jobs": 150,
  "failed_jobs": 2,
  "total_jobs": 160,
  "avg_processing_time_seconds": 45.2,
  "queue_depth": 8,
  "worker_utilization": 0.75
}
```

---

## 3. Enhanced Search

### Paginated Search

**Endpoint:** `GET /search/paginated`

**Description:** Performs advanced search with caching and pagination support.

**Query Parameters:**
- `query` (string, optional): Search query string
- `categories` (array, optional): Filter by taxonomy categories
- `min_confidence` (float, optional): Minimum confidence threshold
- `date_from` (datetime, optional): Start date filter
- `date_to` (datetime, optional): End date filter
- `page` (integer, optional, default: 0): Page number
- `per_page` (integer, optional, default: 50): Results per page

**Success Response (200 OK):**
```json
{
  "results": [
    {
      "chunk_id": "chunk_123",
      "chunk_text": "...",
      "categories_matched": ["category1"],
      "confidence_score": 0.89
    }
  ],
  "total_count": 250,
  "metadata": {
    "cache_hit": true,
    "search_time_ms": 45,
    "filtered_categories": ["category1"]
  }
}
```

### Get Search Count

**Endpoint:** `GET /search/lazy/count`

**Description:** Gets total count of search results without loading all data (lazy loading).

**Query Parameters:** Same as paginated search

**Success Response (200 OK):**
```json
{
  "total_count": 1250
}
```

### Get Search Suggestions

**Endpoint:** `GET /search/suggestions`

**Description:** Provides search suggestions based on partial query input.

**Query Parameters:**
- `query` (string, required): Partial search query
- `limit` (integer, optional, default: 10): Maximum suggestions to return

**Success Response (200 OK):**
```json
{
  "suggestions": [
    "machine learning",
    "artificial intelligence",
    "data processing",
    "neural networks"
  ]
}
```

---

## 4. Cleanup Management

### Run Cleanup

**Endpoint:** `POST /cleanup/run`

**Description:** Manually triggers cleanup operations or forces scheduled cleanup.

**Query Parameters:**
- `force` (boolean, optional): Force immediate cleanup
- `schedule_name` (string, optional): Specific schedule to run

**Success Response (200 OK):**
```json
{
  "cleanup_id": "cleanup_123",
  "files_removed": 25,
  "space_freed_bytes": 10485760,
  "duration_seconds": 2.5,
  "status": "completed"
}
```

### Get Cleanup Statistics

**Endpoint:** `GET /cleanup/stats`

**Description:** Retrieves comprehensive cleanup operation statistics.

**Success Response (200 OK):**
```json
{
  "total_cleanups": 45,
  "files_removed": 1250,
  "space_freed_bytes": 524288000,
  "avg_cleanup_time_seconds": 3.2,
  "last_cleanup": "2024-01-15T14:30:00Z",
  "cleanup_percentage": 85.5
}
```

### Add Cleanup Schedule

**Endpoint:** `POST /cleanup/schedules`

**Description:** Creates a new automated cleanup schedule.

**Request Body:**
```json
{
  "name": "daily_cleanup",
  "interval_seconds": 86400,
  "retention_days": 30,
  "priority": "normal"
}
```

**Success Response (200 OK):**
```json
{
  "message": "Cleanup schedule 'daily_cleanup' added successfully"
}
```

### Remove Cleanup Schedule

**Endpoint:** `DELETE /cleanup/schedules/{schedule_name}`

**Description:** Removes an existing cleanup schedule.

**Path Parameters:**
- `schedule_name` (string, required): Name of schedule to remove

**Success Response (200 OK):**
```json
{
  "message": "Schedule 'daily_cleanup' removed successfully"
}
```

### Optimize Cleanup Schedule

**Endpoint:** `GET /cleanup/optimize`

**Description:** Provides optimization recommendations for cleanup schedules.

**Success Response (200 OK):**
```json
{
  "recommendations": [
    "Increase retention period for high-value data",
    "Reduce cleanup frequency for low-activity periods",
    "Consolidate overlapping schedules"
  ],
  "estimated_savings": {
    "storage_mb": 500,
    "processing_time_seconds": 120
  }
}
```

---

## 5. Performance Monitoring

### Get Performance Metrics

**Endpoint:** `GET /performance/metrics`

**Description:** Retrieves comprehensive system performance metrics.

**Success Response (200 OK):**
```json
{
  "cache": {
    "result_cache": {
      "total_entries": 150,
      "hit_rate": 0.85,
      "avg_access_time_ms": 2.3
    }
  },
  "jobs": {
    "pending_jobs": 5,
    "running_jobs": 3,
    "avg_processing_time_seconds": 45.2
  },
  "cleanup": {
    "files_removed": 1250,
    "space_freed_bytes": 524288000,
    "cleanup_percentage": 85.5
  },
  "system": {
    "total_cache_size_bytes": 3072000,
    "active_jobs": 3,
    "pending_jobs": 5,
    "cleanup_percentage": 85.5
  }
}
```

### Reset Performance Metrics

**Endpoint:** `POST /performance/reset`

**Description:** Resets all performance metrics and clears caches for fresh monitoring.

**Success Response (200 OK):**
```json
{
  "message": "Performance metrics reset successfully"
}
```

---

## Performance Optimization Features

### Caching Strategy
- **Multi-level Caching**: Result cache, search cache, metadata cache
- **Intelligent Invalidation**: Automatic cache cleanup based on data changes
- **Performance Monitoring**: Hit rates, access times, and cache utilization

### Job Queue Optimization
- **Priority Queues**: Support for urgent, high, normal, and low priority jobs
- **Load Balancing**: Automatic distribution across available workers
- **Queue Monitoring**: Real-time statistics and performance metrics

### Lazy Loading
- **On-demand Loading**: Load data only when needed
- **Memory Efficient**: Reduce memory usage for large datasets
- **Fast Counting**: Get result counts without loading full datasets

### Automated Cleanup
- **Scheduled Cleanup**: Configurable retention policies
- **Space Optimization**: Automatic removal of old/unused data
- **Performance Maintenance**: Prevent storage bloat and performance degradation

---

## Best Practices

### Cache Management
- Monitor cache hit rates and adjust cache sizes accordingly
- Clear caches during maintenance windows
- Use appropriate cache TTL settings for different data types

### Job Queue Optimization
- Use appropriate job priorities for different workloads
- Monitor queue depth and processing times
- Scale worker processes based on load

### Search Performance
- Use lazy counting for large result sets
- Implement appropriate pagination limits
- Cache frequent search queries

### Cleanup Operations
- Schedule cleanups during low-usage periods
- Set appropriate retention policies based on data value
- Monitor cleanup effectiveness and adjust schedules

### Performance Monitoring
- Regularly review performance metrics
- Set up alerts for performance degradation
- Use metrics to guide optimization efforts

This performance API provides comprehensive optimization capabilities to ensure Compileo operates efficiently at scale.