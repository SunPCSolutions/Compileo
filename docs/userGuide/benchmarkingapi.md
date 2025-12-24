# Benchmarking API in Compileo

## Overview

The Compileo Benchmarking API provides comprehensive evaluation capabilities for AI models across multiple benchmark suites. It supports automated testing, performance tracking, and comparative analysis with full integration into Compileo's asynchronous job processing system.

## Base URL: `/api/v1/benchmarking`

---

## 1. Run Benchmarks

**Endpoint:** `POST /run`

**Description:** Initiates a new benchmarking job for an AI model against specified evaluation suites using Compileo's job queue system.

**Request Body:**
```json
{
  "project_id": 1,
  "suite": "glue",
  "config": {
    "provider": "ollama",
    "model": "mistral:latest",
    "ollama_params": {
      "temperature": 0.1,
      "top_p": 0.9
    }
  }
}
```

**Parameters:**
- `project_id` (integer, required): Project ID for the benchmark job
- `suite` (string): Benchmark suite (`glue`, `superglue`, `mmlu`, `medical`)
- `config` (object, required): AI model configuration
  - `provider` (string): AI provider (`ollama`, `gemini`, `grok`)
  - `model` (string): Model identifier
  - `ollama_params` (object, optional): Ollama-specific parameters
    - `temperature` (float): Sampling temperature
    - `top_p` (float): Top-p sampling
    - `top_k` (integer): Top-k sampling
    - `num_predict` (integer): Maximum tokens to generate
    - `num_ctx` (integer): Context window size
    - `seed` (integer): Random seed

**Success Response (200 OK):**
```json
{
  "job_id": "a967f363-ee96-4bac-9f52-4d169bbc4851",
  "message": "Benchmarking started for suite: glue",
  "estimated_duration": "10-30 minutes"
}
```

---

## 2. Get Benchmark Results

**Endpoint:** `GET /results/{job_id}`

**Description:** Retrieves the current status and results of a benchmarking job.

**Path Parameters:**
- `job_id` (string, required): Benchmarking job identifier

**Success Response (200 OK):**
```json
{
  "job_id": "a967f363-ee96-4bac-9f52-4d169bbc4851",
  "status": "completed",
  "summary": {
    "total_evaluations": 8,
    "benchmarks_run": ["glue"],
    "models_evaluated": 1,
    "total_time_seconds": 847.23
  },
  "performance_data": {
    "glue": {
      "cola": {"accuracy": {"mean": 0.823, "std": 0.012}},
      "sst2": {"accuracy": {"mean": 0.945, "std": 0.008}},
      "mrpc": {"f1": {"mean": 0.876, "std": 0.015}},
      "qqp": {"f1": {"mean": 0.892, "std": 0.011}},
      "mnli": {"accuracy": {"mean": 0.834, "std": 0.009}},
      "qnli": {"accuracy": {"mean": 0.901, "std": 0.007}},
      "rte": {"accuracy": {"mean": 0.678, "std": 0.023}},
      "wnli": {"accuracy": {"mean": 0.512, "std": 0.031}}
    }
  },
  "completed_at": "2025-12-07T20:55:17Z"
}
```

---

## 3. Cancel Benchmark Job

**Endpoint:** `POST /cancel/{job_id}`

**Description:** Cancels a running or pending benchmark job.

**Path Parameters:**
- `job_id` (string, required): Benchmarking job identifier

**Success Response (200 OK):**
```json
{
  "message": "Job a967f363-ee96-4bac-9f52-4d169bbc4851 cancelled successfully"
}
```

---

## 4. List Benchmark Results

**Endpoint:** `GET /results`

**Description:** Retrieves a list of benchmark jobs with optional filtering.

**Query Parameters:**
- `model_name` (string, optional): Filter by model name
- `suite` (string, optional): Filter by benchmark suite
- `status` (string, optional): Filter by job status (`pending`, `running`, `completed`, `failed`)
- `limit` (integer, optional, default: 20): Maximum number of results

**Success Response (200 OK):**
```json
{
  "results": [
    {
      "job_id": "a967f363-ee96-4bac-9f52-4d169bbc4851",
      "status": "completed",
      "model_name": "mistral:latest",
      "benchmark_suite": "glue",
      "created_at": "2025-12-07T20:55:17Z",
      "completed_at": "2025-12-07T20:56:44Z"
    }
  ],
  "total": 5
}
```

---

## 5. Compare Models

**Endpoint:** `POST /compare`

**Description:** Compares multiple models across specified metrics and benchmark suites.

**Request Body:**
```json
{
  "model_ids": ["gpt-4", "claude-3", "gemini-pro"],
  "benchmark_suite": "glue",
  "metrics": ["accuracy", "f1"]
}
```

**Success Response (200 OK):**
```json
{
  "comparison": {
    "models_compared": ["gpt-4", "claude-3", "gemini-pro"],
    "best_performing": "gpt-4",
    "performance_gap": 0.023,
    "statistical_significance": "p < 0.05",
    "recommendations": [
      "GPT-4 shows superior performance across all metrics",
      "Consider GPT-4 for production use"
    ]
  }
}
```

**Note:** This endpoint provides mock comparison data. Full implementation planned for future release.

---

## 6. Get Benchmark History

**Endpoint:** `GET /history`

**Description:** Retrieves historical benchmarking data with optional filtering.

**Query Parameters:**
- `model_name` (string, optional): Filter by model name
- `days` (integer, optional, default: 30): Number of days to look back

**Success Response (200 OK):**
```json
{
  "history": [
    {
      "job_id": "a967f363-ee96-4bac-9f52-4d169bbc4851",
      "status": "completed",
      "model_name": "mistral:latest",
      "benchmark_suite": "glue",
      "created_at": "2025-12-07T20:55:17Z",
      "completed_at": "2025-12-07T20:56:44Z"
    }
  ],
  "total_runs": 3,
  "date_range": "Last 30 days"
}
```

---

## 7. Get Leaderboard

**Endpoint:** `GET /leaderboard`

**Description:** Retrieves a ranked leaderboard of models for specified criteria.

**Query Parameters:**
- `suite` (string, default: "glue"): Benchmark suite
- `metric` (string, default: "accuracy"): Ranking metric
- `limit` (integer, optional, default: 10): Number of top models to return

**Success Response (200 OK):**
```json
{
  "leaderboard": [
    {
      "rank": 1,
      "model": "gpt-4",
      "score": 0.892,
      "provider": "OpenAI",
      "benchmark_count": 5
    }
  ],
  "total_models": 3,
  "last_updated": "2025-12-07T20:56:44Z"
}
```

**Note:** This endpoint provides mock leaderboard data. Full implementation planned for future release.

---

## Error Handling

### Common Error Responses

**400 Bad Request:**
```json
{
  "detail": "Invalid benchmark suite. Supported: glue, superglue, mmlu, medical"
}
```

**404 Not Found:**
```json
{
  "detail": "Benchmark job a967f363-ee96-4bac-9f52-4d169bbc4851 not found"
}
```

**429 Too Many Requests:**
```json
{
  "detail": "Queue is full. Please try again later.",
  "retry_after": 300
}
```

**500 Internal Server Error:**
```json
{
  "detail": "Benchmark execution failed: job_id and project_id are required"
}
```

### Job-Specific Errors

**Dataset Loading Errors:**
```json
{
  "detail": "Failed to load GLUE dataset: Invalid pattern: '**' can only be an entire path component"
}
```

**API Provider Errors:**
```json
{
  "detail": "AI provider error: API key not configured for Gemini"
}
```

**Resource Limit Errors:**
```json
{
  "detail": "Job execution failed: Resource limits exceeded"
}
```

---

## Rate Limiting & Queue Management

- **Concurrent Jobs:** Maximum 3 concurrent benchmarking jobs system-wide
- **Queue Size:** Maximum 10 queued jobs per user
- **Job Timeout:** 3 hours maximum execution time
- **API Rate Limits:** 100 requests per minute per user

### Queue Priorities
- **High Priority:** Interactive jobs (GUI/API initiated)
- **Normal Priority:** Background jobs
- **Low Priority:** Scheduled maintenance jobs

---

## Best Practices

### Job Management
- Monitor job progress using real-time status updates
- Use appropriate AI models for your use case (Ollama for local, Gemini/Grok for API)
- Cancel unnecessary jobs to free up queue resources
- Check job status before starting new evaluations

### Model Selection
- **Ollama**: Best for local, private model evaluation
- **Gemini**: Good for Google's latest models with custom configuration
- **Grok**: Ideal for xAI models with advanced reasoning

### Performance Optimization
- GLUE benchmarks typically take 10-30 minutes per model
- Schedule large benchmarking runs during off-peak hours
- Monitor system resources (CPU/memory) during execution
- Use appropriate Ollama parameters for your model size

### Result Analysis
- Focus on accuracy as the primary metric for classification tasks
- Compare models using the same benchmark suite for fair evaluation
- Consider both mean performance and standard deviation
- Use historical data to track model performance trends

### Troubleshooting
- Check RQ worker logs for detailed error information
- Verify API keys are properly configured in environment variables
- Ensure sufficient system resources for benchmark execution
- Use smaller test runs before full benchmark suites