# Datasets Module API Usage Guide

The Compileo Datasets API provides comprehensive REST endpoints for dataset generation, management, and quality assessment. This API enables users to create training datasets from processed documents using various AI models and quality control mechanisms. All operations are strictly multi-tenant and require a valid JWT token.

## Base URL: `/api/v1/datasets`

---

## Dataset Generation

### POST `/generate`

Generate a dataset from processed document chunks using AI models. This operation is user-scoped and requires a valid JWT token.

**Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/datasets/generate" \
  -H "Authorization: Bearer <your_jwt_token>" \
  -H "Content-Type: application/json" \
  -d '{
    "project_id": "f33e9780-04af-4675-a348-58eacd49ca54",
    "prompt_name": "qa_pairs",
    "custom_prompt": null,
    "generation_mode": "default",
    "format_type": "jsonl",
    "concurrency": 1,
    "include_evaluation_sets": false,
    "taxonomy_project": null,
    "taxonomy_name": null,
    "output_dir": ".",
    "analyze_quality": true,
    "quality_threshold": 0.7,
    "enable_versioning": false,
    "dataset_name": "medical_qa_dataset",
    "run_benchmarks": false,
    "benchmark_suite": "glue",
    "parsing_model": "gemini",
    "chunking_model": "gemini",
    "classification_model": "gemini",
    "datasets_per_chunk": 3
  }'
```

**Request Body:**
- `project_id`: UUID of the project containing processed documents (required)
- `prompt_name`: Name of prompt template to use (default: "qa_pairs")
- `custom_prompt`: Custom prompt content (optional). If provided, `prompt_used` in results will be `"custom"`.
- `generation_mode`: Generation mode - "default", "question", "answer", "summarization" (default: "default")
- `format_type`: Output format - "jsonl" or "parquet" (default: "jsonl")
- `concurrency`: Number of concurrent processing threads (default: 1)
- `include_evaluation_sets`: Generate train/validation/test splits (default: false)
- `taxonomy_project`: Project containing taxonomy for categorization (optional)
- `taxonomy_name`: Name of taxonomy to use for categorization (optional)
- `output_dir`: Output directory path (default: ".")
- `analyze_quality`: Enable quality analysis of generated data (default: true)
- `quality_threshold`: Minimum quality score threshold (default: 0.7)
- `enable_versioning`: Enable dataset versioning (default: false)
- `dataset_name`: Custom name for the dataset (optional)
- `run_benchmarks`: Run benchmark tests on generated dataset (default: false)
- `benchmark_suite`: Benchmark suite to use (default: "glue")
- `parsing_model`: AI model for document parsing (default: "gemini")
- `chunking_model`: AI model for text chunking (default: "gemini")
- `classification_model`: AI model for content classification (default: "gemini")
- `datasets_per_chunk`: Number of dataset entries to generate per text chunk (default: 3, use 0 for "Let AI Decide" mode)

**"Let AI Decide" Mode:**

Set `datasets_per_chunk` to `0` to enable AI-determined entry count. In this mode:
- The AI analyzes content richness and generates an appropriate number of entries
- **Sparse content** → 1-3 entries
- **Moderate content** → 3-7 entries
- **Rich content** → 7-15+ entries

This is ideal when you want the AI to optimize entry count based on content density rather than using a fixed number.

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "message": "Dataset generation started"
}
```

---

## Evaluation Dataset Generation

### POST `/generate-evaluation`

Generate comprehensive evaluation datasets with train/validation/test splits.

**Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/datasets/generate-evaluation" \
  -H "Authorization: Bearer <your_jwt_token>" \
  -H "Content-Type: application/json" \
  -d '{
    "project_id": "f33e9780-04af-4675-a348-58eacd49ca54",
    "prompt_name": "evaluation_qa",
    "format_type": "jsonl",
    "concurrency": 2,
    "include_evaluation_sets": true,
    "analyze_quality": true,
    "quality_threshold": 0.8,
    "classification_model": "gemini",
    "datasets_per_chunk": 5
  }'
```

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440001",
  "message": "Evaluation dataset generation started"
}
```

---

## Job Status Monitoring

### GET `/generate/{job_id}/status`

Get the status of a dataset generation job.

**Request:**
```bash
curl -H "Authorization: Bearer <your_jwt_token>" \
  "http://localhost:8000/api/v1/datasets/generate/550e8400-e29b-41d4-a716-446655440000/status"
```

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "running",
  "progress": 65,
  "current_step": "Processing chunk 45 of 100",
  "estimated_completion": "2024-01-21T11:45:00Z",
  "result": null,
  "error": null
}
```

**Status Values:**
- `pending`: Job queued and waiting to start
- `running`: Job currently executing
- `completed`: Job finished successfully
- `failed`: Job failed with an error

---

## Dataset Retrieval

### GET `/{dataset_id}/details`

Get detailed information about a specific dataset. The `project_id` query parameter is highly recommended for efficient retrieval, though the system will attempt to resolve it from the database if omitted.

**Request:**
```bash
curl -H "Authorization: Bearer <your_jwt_token>" \
  "http://localhost:8000/api/v1/datasets/dataset_550e8400-e29b-41d4-a716-446655440000/details?project_id=f33e9780-04af-4675-a348-58eacd49ca54"
```

**Response:**
```json
{
  "id": "dataset_550e8400-e29b-41d4-a716-446655440000",
  "name": "medical_qa_dataset",
  "entries": [
    {
      "id": "entry_1",
      "question": "What are the common symptoms of diabetes?",
      "answer": "Common symptoms of diabetes include frequent urination, excessive thirst, unexplained weight loss, increased hunger, fatigue, slow-healing sores, frequent infections, blurred vision, and tingling or numbness in hands or feet.",
      "category": "Endocrinology",
      "quality_score": 0.92,
      "difficulty": "intermediate",
      "source_chunk": "chunk_001.md",
      "metadata": {
        "model": "gemini",
        "prompt_used": "qa_pairs",
        "generation_time": "2024-01-21T10:35:22Z",
        "chunk_length": 1250
      }
    }
  ],
  "total_entries": 150,
  "created_at": "2024-01-21T10:30:00Z",
  "format_type": "jsonl",
  "quality_summary": {
    "average_score": 0.87,
    "distribution": {
      "high": 120,
      "medium": 25,
      "low": 5
    },
    "categories": {
      "Endocrinology": 45,
      "Cardiology": 38,
      "Neurology": 32,
      "Other": 35
    }
  }
}
```

---

## Dataset Entries Management

### GET `/{dataset_id}/entries`

Get dataset entries with pagination and filtering.

**Request:**
```bash
curl -H "Authorization: Bearer <your_jwt_token>" \
  "http://localhost:8000/api/v1/datasets/dataset_550e8400-e29b-41d4-a716-446655440000/entries?page=1&per_page=50&filter_quality=0.8&sort_by=quality"
```

**Query Parameters:**
- `page`: Page number (default: 1)
- `per_page`: Entries per page (default: 50)
- `filter_quality`: Minimum quality score filter (optional)
- `sort_by`: Sort field - "quality", "category", "difficulty" (default: "quality")

**Response:**
```json
{
  "entries": [
    {
      "id": "entry_1",
      "question": "What are the common symptoms of diabetes?",
      "answer": "Common symptoms of diabetes include frequent urination...",
      "category": "Endocrinology",
      "quality_score": 0.92,
      "difficulty": "intermediate",
      "source_chunk": "chunk_001.md",
      "metadata": {
        "model": "gemini",
        "prompt_used": "qa_pairs",
        "generation_time": "2024-01-21T10:35:22Z"
      }
    }
  ],
  "total": 150,
  "page": 1,
  "per_page": 50,
  "quality_summary": {
    "average_score": 0.87,
    "distribution": {
      "high": 120,
      "medium": 25,
      "low": 5
    }
  }
}
```

---

## Dataset Deletion

### DELETE `/{dataset_id}/delete`

Delete a dataset file and its associated data. The `project_id` query parameter is highly recommended for efficient retrieval, though the system will attempt to resolve it from the database if omitted.

**Request:**
```bash
curl -X DELETE -H "Authorization: Bearer <your_jwt_token>" \
  "http://localhost:8000/api/v1/datasets/dataset_550e8400-e29b-41d4-a716-446655440000/delete?project_id=f33e9780-04af-4675-a348-58eacd49ca54"
```

**Response:**
```json
{
  "message": "Successfully deleted 1 dataset file(s)"
}
```

---

## Best Practices

1.  **Strict Isolation**: All requests must include the user's JWT token. Operations are automatically scoped to the authenticated user's data.
2.  **Robust Formatting**: The system uses a safe formatting strategy that allows document content to contain special characters without breaking prompt templates.
3.  **Prompt Tracking**: Use the `prompt_used` metadata field to verify whether your generation run used a standard template or a custom prompt.
4.  **UUID Usage**: Modern projects use UUID strings for `project_id` and other identifiers instead of integers.
