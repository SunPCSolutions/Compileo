# Job Handling Module CLI Usage Guide

The Compileo Job Handling CLI provides tools for monitoring and managing background jobs.

## Command Overview

---

## Get Job Status

### `compileo jobs status <job_id>`

Retrieve the current status and details of a specific job.

**Usage:**
```bash
compileo jobs status b3013a51-4b10-4fca-ac09-239a2b886b7e
```

**Example Output:**
```
📊 Job Status:
  ID: b3013a51-4b10-4fca-ac09-239a2b886b7e
  Status: completed
  Progress: 100%
  Created: 2024-01-21 12:00:00
  Started: 2024-01-21 12:00:05
  Completed: 2024-01-21 12:01:00
```

---

## Cancel a Job

### `compileo jobs cancel <job_id>`

Cancel a running or pending job.

**Usage:**
```bash
compileo jobs cancel b3013a51-4b10-4fca-ac09-239a2b886b7e
```

**Example Output:**
```
✅ Job b3013a51-4b10-4fca-ac09-239a2b886b7e cancelled successfully.
```

---

## Duplicate Job Execution Prevention

The job handling system includes a critical fix to prevent the same job from being executed multiple times. This is achieved by:
1.  **Early Status Check:** The system immediately checks if a job's status is already `COMPLETED`, `FAILED`, or `CANCELLED` before execution. If so, the worker exits immediately.
2.  **Atomic Status Updates:** Job statuses are updated atomically in Redis to prevent race conditions where a job might be picked up by another worker before its status is updated.

This ensures that each job is processed exactly once, preventing redundant operations and inconsistent outputs.