# Job Handling Module GUI Usage Guide

The Compileo Job Handling GUI provides a comprehensive web interface for monitoring, managing, and controlling background jobs using an enhanced Redis-based queue system.

## Accessing the Job Dashboard

Navigate to the **"📊 Job Dashboard"** page from the main menu to access the job monitoring interface.

<div align="center">
  <a href="../img/jobhandle.png"><img src="../img/jobhandle.png" width="400" alt="Asynchronous Job Management"></a>
  <p><i>Asynchronous Job Management Interface</i></p>
</div>

---

## Job Queue Overview

The dashboard displays real-time job queue statistics and system health:

### Queue Statistics Panel
- **Pending Jobs**: Jobs waiting in queue
- **Running Jobs**: Currently executing jobs
- **Scheduled Jobs**: Jobs scheduled for future execution
- **Total Jobs**: All jobs in system (with automatic cleanup)
- **Active Workers**: Number of healthy worker processes
- **System Health**: CPU and memory usage metrics

### Enhanced Features
- **Real-time Updates**: Statistics refresh automatically every 30 seconds
- **Worker Health Monitoring**: Automatic detection and cleanup of stale workers
- **Job Cleanup Status**: Last cleanup time and items removed
- **System Alerts**: Warnings for high resource usage or failed jobs

---

## Job Monitoring

### Job List View
The main job table provides comprehensive job information:

1.  **Job Information:**
    *   **Job ID**: Unique UUID identifier
    *   **Job Type**: extraction, document_processing, taxonomy_processing, dataset_generation
    *   **Status**: pending, running, completed, failed, cancelled, scheduled
    *   **Progress**: Percentage complete (0-100%)
    *   **User**: User who submitted the job

2.  **Timestamps:**
    *   **Created**: When job was submitted
    *   **Started**: When processing began
    *   **Completed**: When job finished (or failed/cancelled)

3.  **Performance Metrics:**
    *   **Execution Time**: Total processing duration
    *   **Items Processed**: Number of items handled
    *   **Error Messages**: Failure details (if applicable)

### Filtering and Sorting
- **Status Filter**: View jobs by status (All, Pending, Running, Completed, Failed, Cancelled)
- **Type Filter**: Filter by job type
- **Time Range**: Filter jobs by creation date
- **Search**: Find jobs by ID or user
- **Sorting**: Sort by creation time, status, or progress

---

## Job Management

### View Job Details

1.  **Select Job**: Click on any job row in the table
2.  **Detailed View**: Expand to see complete job information including:
    - Full parameters and configuration
    - Progress history and milestones
    - Result summaries (for completed jobs)
    - Error details (for failed jobs)
    - Performance metrics and timing

### Cancel a Job

1.  **Locate Target Job**: Find the running or pending job in the list
2.  **Cancel Action**: Click the "❌ Cancel" button in the Actions column
3.  **Confirmation**: Confirm cancellation in the dialog prompt
4.  **Status Update**: Job status changes to "cancelled" immediately

### Restart Failed Jobs

1.  **Identify Failed Job**: Find job with "failed" status
2.  **Restart Action**: Click the "🔄 Restart" button
3.  **Confirmation**: Confirm restart with same parameters
4.  **Re-queue**: Job returns to "pending" status with fresh execution

---

## Real-time Monitoring

### In-View Synchronous Monitoring
Compileo uses a standardized **Synchronous In-View Monitoring** pattern for all long-running jobs (parsing, chunking, extraction, etc.) started from the GUI:

- **UI Stability**: Status updates occur within `st.empty()` placeholders, preventing disruptive whole-page reruns and preserving your scroll position.
- **Accurate Status**: Displays stage-based progress messages (e.g., "Generating Splits", "Analyzing Chunks") fetched directly from backend metadata.
- **Blocked Execution**: The submit button handler remains active until the job reaches a terminal state (Completed, Failed, or Cancelled), ensuring you see the final result immediately.

### Live Progress Updates
- **Progress Bars**: Visual progress indicators for running jobs
- **Status Changes**: Automatic status updates without page refresh
- **Performance Metrics**: Live execution throughput updates
- **Worker Status**: Real-time worker health and activity monitoring

### Server-Sent Events (SSE)
The GUI uses Server-Sent Events for real-time updates:
- **Instant Notifications**: Immediate status change alerts
- **Progress Streaming**: Live progress updates during execution
- **Error Alerts**: Real-time failure notifications
- **Completion Alerts**: Success notifications with result summaries

---

## RQ System Features

### Enhanced Reliability Features

**1. Datetime Compatibility**
- Automatic handling of timezone-aware timestamps from RQ
- Prevents datetime comparison errors during worker monitoring
- Compatible with Redis timestamp storage

**2. Worker Health Management**
- Continuous monitoring of RQ worker processes
- Automatic cleanup of workers unresponsive for >5 minutes
- Prevention of dead worker accumulation in Redis

**3. Comprehensive Job Cleanup**
- **Multi-level Cleanup**: Runs every 10 minutes
  - RQ failed jobs: Immediate cleanup
  - RQ finished jobs: 24-hour retention
  - Custom jobs: 2-hour retention
  - Processing locks: 10-minute cleanup
- **Startup Cleanup**: Removes jobs >1 hour old on system start
- **Registry Management**: Proper RQ registry maintenance

**4. Duplicate Execution Prevention**
- Atomic Redis operations prevent race conditions
- Early status validation before job execution
- UUID-based job identification
- Processing locks for concurrent execution safety

---

## Job Types and Operations

### Document Processing Jobs
- **Parse Documents**: Convert PDFs/text to markdown chunks
- **Chunk Documents**: Apply chunking strategies (token, character, semantic, schema)
- **Progress Tracking**: Real-time parsing and chunking progress

### Taxonomy Processing Jobs
- **Generate Taxonomy**: AI-powered taxonomy creation from document chunks
- **Category Analysis**: Hierarchical category structure generation
- **Validation**: Taxonomy completeness and consistency checks

### Extraction Jobs
- **Selective Extraction**: Taxonomy-based content extraction
- **Multi-stage Classification**: Primary + validation classifiers
- **Confidence Scoring**: Result confidence and category matching

### Dataset Generation Jobs
- **Automated Dataset Creation**: Convert extractions to training datasets
- **Format Conversion**: JSON, CSV, and other dataset formats
- **Quality Validation**: Dataset completeness and consistency checks

---

## Troubleshooting

### Common Issues

**Jobs Stuck in Pending:**
- Check worker health in queue statistics
- Verify Redis connectivity
- Review worker logs for startup errors
- Check resource limits (CPU/memory)

**Incorrect Job Counts:**
- Wait for automatic refresh (30 seconds)
- Check last cleanup time in statistics
- Verify job status accuracy

**High Memory Usage:**
- Monitor worker accumulation
- Check for failed job cleanup
- Review large result storage

**Datetime Errors:**
- System automatically handles timezone issues
- Check server logs for RQ compatibility
- Verify system clock synchronization

---

## Performance Optimization

### Best Practices
1. **Monitor Queue Health**: Use statistics panel for system monitoring
2. **Cancel Unnecessary Jobs**: Clean up jobs no longer needed
3. **Use Appropriate Timeouts**: Set reasonable job timeouts
4. **Monitor Resource Usage**: Watch CPU/memory trends
5. **Regular Cleanup**: System performs automatic maintenance

### System Limits
- **Global Max Jobs**: 10 concurrent jobs (configurable)
- **Per-User Max Jobs**: 3 concurrent jobs (configurable)
- **Job Timeout**: 72 hours default (configurable)
- **Result TTL**: 500 seconds default (configurable)
- **Cleanup Frequency**: Every 10 minutes

---

## Advanced Features

### Job Dependencies (Future)
- Chain jobs with prerequisite relationships
- Automatic dependency resolution
- Failure cascade handling

### Scheduled Jobs
- Time-based job execution
- Cron-style scheduling
- Recurring job patterns

### Bulk Operations
- Multiple job cancellation
- Batch status updates
- Bulk cleanup operations

This enhanced job handling system provides enterprise-grade reliability with comprehensive monitoring, automatic maintenance, and real-time user feedback for all background processing operations.
