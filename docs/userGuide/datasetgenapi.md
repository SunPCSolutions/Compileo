# Dataset Generation in Compileo API

## Overview

The Compileo API provides comprehensive endpoints for generating high-quality datasets from processed document chunks. The dataset generation process supports multiple AI models, taxonomy integration, and various output formats with optional quality analysis and benchmarking.

## Base URL: `/api/v1`

## Key Features

- **Multi-Model Support**: Choose from Gemini, Grok, or Ollama for parsing, chunking, and classification
- **Taxonomy Integration**: Generate datasets aligned with existing taxonomies
- **Flexible Output**: JSONL and Parquet formats supported
- **Quality Assurance**: Optional quality analysis with configurable thresholds
- **Performance Benchmarking**: Optional AI model benchmarking after generation
- **Asynchronous Processing**: All generation runs as background jobs with real-time monitoring

## API Endpoints

### 1. Generate Dataset

Submits a dataset generation job with comprehensive configuration options.

- **Endpoint:** `POST /datasets/generate`
- **Description:** Initiates dataset generation from processed chunks with full configuration control
- **Request Body:**
  ```json
  {
    "project_id": 1,
    "data_source": "Chunks Only",
    "prompt_name": "default",
    "custom_prompt": "Optional custom prompt content",
    "selected_categories": ["category1", "category2"],
    "generation_mode": "default",
    "format_type": "jsonl",
    "concurrency": 1,
    "batch_size": 50,
    "include_evaluation_sets": false,
    "taxonomy_project": null,
    "taxonomy_name": null,
    "output_dir": ".",
    "analyze_quality": true,
    "quality_threshold": 0.7,
    "enable_versioning": false,
    "dataset_name": null,
    "run_benchmarks": false,
    "benchmark_suite": "glue",
    "parsing_model": "gemini",
    "chunking_model": "gemini",
    "classification_model": "gemini",
    "datasets_per_chunk": 3,
    "only_validated": false
  }
  ```

- **Parameters:**
  - `project_id` (integer, required): Project containing processed chunks
  - `data_source` (string): Data source mode - "Chunks Only", "Taxonomy", "Extract" (default: "Chunks Only")
  - `extraction_file_id` (string, optional): Specific extraction job ID (UUID) when data_source is "Extract". Use `/api/v1/datasets/extraction-files/{project_id}` to get a list of available extraction jobs.
  - `selected_categories` (array of strings, optional): List of category names to filter by when data_source is "Extract"
  - `prompt_name` (string): Name of prompt template to use (default: "default")
  - `custom_prompt` (string, optional): Custom prompt content for generation
  - `generation_mode` (string): Generation mode - "instruction following", "question and answer", "question", "answer", "summarization"
  - `format_type` (string): Output format - "jsonl", "parquet", or plugin formats (e.g., "anki")
  - `concurrency` (integer): Number of parallel processing threads
  - `batch_size` (integer): Number of chunks to process per batch (0 = all at once)
  - `include_evaluation_sets` (boolean): Generate train/validation/test splits
  - `taxonomy_project` (string, optional): Project name containing taxonomy
  - `taxonomy_name` (string, optional): Name of taxonomy to align with
  - `output_dir` (string): Output directory path
  - `analyze_quality` (boolean): Enable quality analysis
  - `quality_threshold` (float): Quality threshold for pass/fail (0-1)
  - `enable_versioning` (boolean): Enable dataset versioning
  - `dataset_name` (string, optional): Name for versioned datasets
  - `run_benchmarks` (boolean): Run AI model benchmarks after generation
  - `benchmark_suite` (string): Benchmark suite - "glue", "superglue", "mmlu", "medical"
  - `parsing_model` (string): AI model for document parsing
  - `chunking_model` (string): AI model for text chunking
  - `classification_model` (string): AI model for content classification
  - `datasets_per_chunk` (integer): Maximum datasets per text chunk
  - `only_validated` (boolean): Filter extraction results to only include data that has passed validation

  **Data Source Modes:**

  - **"Chunks Only"**: Uses raw text chunks directly from processed documents. No taxonomy or extraction filtering required. Best for basic dataset generation from any content.
  - **"Taxonomy"**: Applies taxonomy definitions to enhance generation prompts. Works with all chunks in the project (no extraction dependency). Adds domain-specific context and terminology. This mode strictly bypasses extraction data loading for efficiency.
  - **"Extract"**: Uses extracted entities as the primary content source. Generates datasets focused on specific concepts/entities. Creates educational content about extracted terms.

  **Batch Processing:**
  - `batch_size` controls memory usage by processing chunks in smaller groups
  - Set to 0 to process all chunks at once (legacy behavior)
  - Smaller batches reduce memory usage but may take longer
  - Each batch creates a separate output file: `dataset_[job_id]_batch_[N].[format]`. Parquet exports are saved as binary `.parquet` files, while JSON/JSONL are saved as UTF-8 text files.

  **Generation Mode Options:**
  
  - **"default"**: Generates content based on the provided prompt name or custom prompt.
  - **"instruction following"** (Recommended): Generates modern instruction-response pairs following Alpaca/Dolly standards for advanced AI fine-tuning
  - **"question and answer"**: Creates traditional Q&A pairs from document content
  - **"question"**: Generates questions without corresponding answers
  - **"answer"**: Generates answers without explicit questions
  - **"summarization"**: Creates concise summaries of document content
  
  - **Success Response (200 OK):**
  ```json
  {
    "job_id": "dataset-gen-uuid-123",
    "message": "Dataset generation started successfully",
    "status": "submitted",
    "estimated_duration": "Processing in background"
  }
  ```

### 2. Generate Evaluation Dataset

Creates comprehensive evaluation datasets with train/validation/test splits.

- **Endpoint:** `POST /datasets/generate-evaluation`
- **Description:** Generates evaluation-ready datasets with multiple splits and quality analysis
- **Request Body:** Same as `/datasets/generate` but optimized for evaluation
- **Success Response:** Same format as dataset generation

### 3. Get Dataset Generation Status

Monitors the progress of dataset generation jobs.

- **Endpoint:** `GET /datasets/generate/{job_id}/status`
- **Description:** Returns current status and progress of dataset generation. This endpoint implements a persistent database fallback, ensuring status is available even after server restarts or worker instance changes.
- **Path Parameters:**
  - `job_id` (string, required): Dataset generation job ID
- **Success Response (200 OK):**
  ```json
  {
    "job_id": "dataset-gen-uuid-123",
    "status": "running",
    "progress": 65,
    "current_step": "Processing batch 2/3 (25 chunks)",
    "estimated_completion": "2024-01-21T14:30:00Z",
    "result": {
      "batch_files": [
        {
          "batch_index": 0,
          "file_path": "/storage/datasets/1/dataset_dataset-gen-uuid-123_batch_0.json",
          "entries_count": 15,
          "chunks_processed": 5
        },
        {
          "batch_index": 1,
          "file_path": "/storage/datasets/1/dataset_dataset-gen-uuid-123_batch_1.json",
          "entries_count": 18,
          "chunks_processed": 5
        }
      ],
      "completed_batches": 2,
      "total_batches": 3,
      "total_entries_so_far": 33
    }
  }
  ```

### 4. Get Dataset

Retrieves generated dataset details and entries.

- **Endpoint:** `GET /datasets/{dataset_id}`
- **Description:** Returns dataset metadata and summary information
- **Path Parameters:**
  - `dataset_id` (string, required): Dataset identifier
- **Success Response (200 OK):**
  ```json
  {
    "id": "dataset_123",
    "name": "Medical Diagnosis Dataset",
    "entries": [],
    "total_entries": 1500,
    "created_at": "2024-01-21T12:00:00Z",
    "format_type": "jsonl",
    "batch_files": [
      {
        "batch_index": 0,
        "file_path": "/storage/datasets/1/dataset_123_batch_0.json",
        "entries_count": 500,
        "chunks_processed": 50
      },
      {
        "batch_index": 1,
        "file_path": "/storage/datasets/1/dataset_123_batch_1.json",
        "entries_count": 500,
        "chunks_processed": 50
      },
      {
        "batch_index": 2,
        "file_path": "/storage/datasets/1/dataset_123_batch_2.json",
        "entries_count": 500,
        "chunks_processed": 50
      }
    ],
    "total_batches": 3,
    "quality_summary": {
      "overall_score": 0.85,
      "diversity_score": 0.82,
      "bias_score": 0.88
    }
  }
  ```

### 5. Get Dataset Entries

Retrieves dataset entries with pagination and filtering.

- **Endpoint:** `GET /datasets/{dataset_id}/entries`
- **Description:** Returns paginated dataset entries with optional filtering
- **Path Parameters:**
  - `dataset_id` (string, required): Dataset identifier
- **Query Parameters:**
  - `page` (integer): Page number (default: 1)
  - `per_page` (integer): Entries per page (default: 50)
  - `filter_quality` (float, optional): Minimum quality score filter
  - `sort_by` (string): Sort field - "quality", "category", "difficulty"
- **Success Response (200 OK):**
  ```json
  {
    "entries": [
      {
        "id": "entry_1",
        "question": "What are the symptoms of myocardial infarction?",
        "answer": "Chest pain, shortness of breath, diaphoresis...",
        "category": "cardiology",
        "quality_score": 0.92,
        "difficulty": "intermediate",
        "source_chunk": "chunk_45",
        "metadata": {
          "model": "gemini",
          "taxonomy_level": 2
        }
      }
    ],
    "total": 1500,
    "page": 1,
    "per_page": 50,
    "quality_summary": {
      "average_score": 0.85,
      "distribution": {"high": 1200, "medium": 250, "low": 50}
    }
  }
  ```

### 6. Update Dataset Entry

Updates individual dataset entries for refinement.

- **Endpoint:** `PUT /datasets/{dataset_id}/entries/{entry_id}`
- **Description:** Modifies question, answer, or metadata for dataset entries
- **Path Parameters:**
  - `dataset_id` (string, required): Dataset identifier
  - `entry_id` (string, required): Entry identifier
- **Request Body:**
  ```json
  {
    "question": "Updated question text",
    "answer": "Updated answer text",
    "category": "updated_category",
    "feedback": "User feedback on this entry"
  }
  ```

### 7. Submit Dataset Feedback

Collects user feedback on dataset quality and relevance.

- **Endpoint:** `POST /datasets/{dataset_id}/feedback`
- **Description:** Submits bulk feedback for multiple dataset entries
- **Request Body:**
  ```json
  {
    "entry_ids": ["entry_1", "entry_2"],
    "feedback_type": "bulk_edit",
    "comments": "General feedback on these entries",
    "rating": 4
  }
  ```

### 8. Get Extraction Files

Retrieves available extraction jobs for dataset generation.

- **Endpoint:** `GET /datasets/extraction-files/{project_id}`
- **Description:** Returns list of completed extraction jobs for a project, used to populate extraction file selection dropdown.
- **Path Parameters:**
  - `project_id` (string, required): Project ID (UUID) to get extraction files for. Accepts both integer and string formats for compatibility.
- **Success Response (200 OK):**
  ```json
  {
    "extraction_files": [
      {
        "id": 123,
        "job_id": 123,
        "status": "completed",
        "created_at": "2024-01-15T10:30:00Z",
        "extraction_type": "ner",
        "entity_count": 150,
        "display_name": "Job 123 - ner (150 entities)"
      },
      {
        "id": 124,
        "job_id": 124,
        "status": "completed",
        "created_at": "2024-01-16T14:20:00Z",
        "extraction_type": "whole_text",
        "entity_count": 89,
        "display_name": "Job 124 - whole_text (89 entities)"
      }
    ]
  }
  ```

### 9. Regenerate Dataset Entries

Triggers regeneration of specific dataset entries.

- **Endpoint:** `POST /datasets/{dataset_id}/regenerate`
- **Description:** Regenerates entries with updated parameters or models
- **Request Body:**
  ```json
  {
    "entry_ids": ["entry_1", "entry_2"],
    "regeneration_config": {
      "model": "grok",
      "temperature": 0.7,
      "max_tokens": 500
    }
  }
  ```

### 10. Download Dataset

Downloads generated datasets in the specified format.

- **Endpoint:** `GET /datasets/{dataset_id}/download`
- **Description:** Downloads dataset file (JSONL/Parquet)
- **Path Parameters:**
  - `dataset_id` (string, required): Dataset identifier
- **Success Response:** File download with appropriate content-type

## Integration with Job Management

All dataset generation operations return a `job_id` and run asynchronously. Dataset generation jobs are now stored in the database and persist across API server restarts. Use the dedicated dataset generation status endpoint to monitor progress:

```http
POST /api/v1/datasets/generate
Content-Type: application/json

{
  "project_id": 1,
  "classification_model": "gemini",
  "analyze_quality": true
}

Response:
{
  "job_id": "dataset-gen-uuid-789",
  "message": "Dataset generation started"
}

# Monitor progress (dataset jobs persist across server restarts)
GET /api/v1/datasets/generate/dataset-gen-uuid-789/status
```

## Model Selection Guidelines

### Parsing Models
- **Gemini**: Best for complex document understanding and multi-format support
- **Grok**: Good for technical and structured content
- **Ollama**: Local processing, no API keys required

### Classification Models
- **Gemini**: Superior taxonomy alignment and category classification
- **Grok**: Better for nuanced medical and technical categorization
- **Ollama**: Cost-effective for high-volume processing

### Chunking Models
- **Gemini**: Intelligent semantic chunking with context awareness
- **Grok**: Good for technical document segmentation
- **Ollama**: Fast processing for simple chunking strategies

## Error Handling

### Common Error Responses

- **400 Bad Request**: Invalid parameters or missing required fields
- **404 Not Found**: Project, taxonomy, or dataset not found
- **503 Service Unavailable**: Job queue manager not initialized (Redis unavailable)

### Job-Specific Errors

Dataset generation jobs may fail with specific error messages:
- `"No chunks found for project"`: Project has no processed chunks
- `"API key not configured"`: Selected model requires API key
- `"Taxonomy validation failed"`: Specified taxonomy not found or invalid
- `"Quality threshold not met"`: Generated dataset failed quality checks

## Best Practices

1. **Model Selection**: Choose appropriate models based on content type and requirements
2. **Chunk Preparation**: Ensure high-quality chunking before generation
3. **Quality Analysis**: Enable quality analysis for production datasets
4. **Versioning**: Use versioning for iterative dataset improvement
5. **Monitoring**: Always monitor job progress using the jobs API
6. **Resource Planning**: Consider concurrency limits and processing time for large datasets