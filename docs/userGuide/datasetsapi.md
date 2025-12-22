# Datasets Module API Usage Guide

The Compileo Datasets API provides comprehensive REST endpoints for dataset generation, management, and quality assessment. This API enables users to create training datasets from processed documents using various AI models and quality control mechanisms.

## Base URL: `/api/v1/datasets`

---

## Dataset Generation

### POST `/generate`

Generate a dataset from processed document chunks using AI models.

**Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/datasets/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "project_id": 1,
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
- `project_id`: Project containing processed documents (required)
- `prompt_name`: Name of prompt template to use (default: "qa_pairs")
- `custom_prompt`: Custom prompt content (optional)
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
- `datasets_per_chunk`: Maximum datasets to generate per text chunk (default: 3)

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
  -H "Content-Type: application/json" \
  -d '{
    "project_id": 1,
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
curl "http://localhost:8000/api/v1/datasets/generate/550e8400-e29b-41d4-a716-446655440000/status"
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

### GET `/{dataset_id}`

Get detailed information about a specific dataset.

**Request:**
```bash
curl "http://localhost:8000/api/v1/datasets/dataset_550e8400-e29b-41d4-a716-446655440000"
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
curl "http://localhost:8000/api/v1/datasets/dataset_550e8400-e29b-41d4-a716-446655440000/entries?page=1&per_page=50&filter_quality=0.8&sort_by=quality"
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

### PUT `/{dataset_id}/entries/{entry_id}`

Update a specific dataset entry.

**Request:**
```bash
curl -X PUT "http://localhost:8000/api/v1/datasets/dataset_550e8400-e29b-41d4-a716-446655440000/entries/entry_1" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the most common symptoms of diabetes mellitus?",
    "answer": "The most common symptoms include polydipsia (excessive thirst), polyuria (frequent urination), unexplained weight loss, and fatigue.",
    "category": "Endocrinology",
    "feedback": "Improved medical terminology"
  }'
```

**Request Body:**
- `question`: Updated question text (optional)
- `answer`: Updated answer text (optional)
- `category`: Updated category (optional)
- `feedback`: User feedback/comments (optional)

**Response:**
```json
{
  "message": "Entry entry_1 updated successfully"
}
```

---

## Dataset Enhancement

### POST `/{dataset_id}/feedback`

Submit feedback for multiple dataset entries.

**Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/datasets/dataset_550e8400-e29b-41d4-a716-446655440000/feedback" \
  -H "Content-Type: application/json" \
  -d '{
    "entry_ids": ["entry_1", "entry_2", "entry_3"],
    "feedback_type": "bulk_edit",
    "comments": "Improved medical accuracy and terminology",
    "rating": 4
  }'
```

**Request Body:**
- `entry_ids`: Array of entry IDs to provide feedback for (required)
- `feedback_type`: Type of feedback - "bulk_edit", "quality_issue", "content_error" (default: "bulk_edit")
- `comments`: Feedback comments (optional)
- `rating`: Quality rating 1-5 (optional)

**Response:**
```json
{
  "message": "Feedback submitted for 3 entries"
}
```

### POST `/{dataset_id}/regenerate`

Regenerate specific dataset entries with improved configuration.

**Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/datasets/dataset_550e8400-e29b-41d4-a716-446655440000/regenerate" \
  -H "Content-Type: application/json" \
  -d '{
    "entry_ids": ["entry_1", "entry_5", "entry_12"],
    "regeneration_config": {
      "model": "gemini-2.5-flash",
      "temperature": 0.7,
      "max_tokens": 500,
      "improve_quality": true
    }
  }'
```

**Request Body:**
- `entry_ids`: Array of entry IDs to regenerate (required)
- `regeneration_config`: Configuration for regeneration (required)
  - `model`: AI model to use
  - `temperature`: Generation temperature
  - `max_tokens`: Maximum token length
  - `improve_quality`: Enable quality improvements

**Response:**
```json
{
  "job_id": "regenerate_550e8400-e29b-41d4-a716-446655440002",
  "message": "Regeneration started for 3 entries"
}
```

---

## Dataset Export

### GET `/{dataset_id}/download`

Download the complete dataset file. The `dataset_id` should be the unique database ID (UUID) returned in the generation result.

**Features:**
- **Database Mediation**: Resolves physical file paths using the database for high integrity.
- **Automatic Zipping**: If the dataset consists of multiple batches, they are automatically archived into a single ZIP file.
- **Extension Detection**: Intelligently applies the correct file extension based on metadata.

**Request:**
```bash
curl -O "http://localhost:8000/api/v1/datasets/f0efd610-ffdb-4d38-8506-835320f7270d/download"
```

**Response:**
Returns the dataset file (or ZIP archive) with appropriate headers for browser download.

---

## Configuration Endpoints

### GET `/config/high-level-prompts`

Retrieve default options for high-level prompt configuration.

**Response:**
```json
{
  "audience_defaults": ["healthcare professionals", "students", ...],
  "purpose_defaults": ["patient education", "research", ...],
  "complexity_options": ["beginner", "intermediate", "advanced", "expert"],
  "domain_defaults": ["general", ...]
}
```

### GET `/config/default-prompts`

Retrieve default prompt templates for each generation mode.

**Response:**
```json
{
  "prompts": {
    "instruction following": "...",
    "question and answer": "...",
    "summarization": "..."
  }
}
```

---

## Dataset Parameters Management

### POST `/parameters`

Save dataset generation parameters for a project.

**Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/datasets/parameters" \
  -H "Content-Type: application/json" \
  -d '{
    "project_id": 1,
    "purpose": "medical_qa_training",
    "audience": "medical_students",
    "extraction_rules": "clinical_guidelines",
    "dataset_format": "jsonl",
    "question_style": "factual",
    "answer_style": "comprehensive",
    "negativity_ratio": 0.1,
    "data_augmentation": "synonym_replacement",
    "custom_audience": "3rd_year_medical_students",
    "custom_purpose": "board_exam_preparation",
    "complexity_level": "intermediate",
    "domain": "medicine"
  }'
```

**Request Body:**
- `project_id`: Project ID (required)
- `purpose`: Dataset purpose (required)
- `audience`: Target audience (required)
- `extraction_rules`: Rules for data extraction (default: "default")
- `dataset_format`: Output format (required)
- `question_style`: Question style - "factual", "analytical", "case_based" (default: "factual")
- `answer_style`: Answer style - "concise", "comprehensive", "step_by_step" (default: "comprehensive")
- `negativity_ratio`: Ratio of negative examples (default: 0.1)
- `data_augmentation`: Augmentation techniques (default: "none")
- `custom_audience`: Custom audience description (optional)
- `custom_purpose`: Custom purpose description (optional)
- `complexity_level`: Complexity level - "basic", "intermediate", "advanced" (default: "intermediate")
- `domain`: Content domain (default: "general")

**Response:**
```json
{
  "message": "Dataset parameters saved successfully",
  "parameter_id": 101,
  "project_id": 1
}
```

### GET `/parameters/{project_id}`

Get the latest dataset generation parameters for a project.

**Request:**
```bash
curl "http://localhost:8000/api/v1/datasets/parameters/b357e573-89a5-4b40-8e1b-4c075a1835a6"
```

**Response:**
```json
{
  "parameters": {
    "id": "uuid-1",
    "project_id": "b357e573-89a5-4b40-8e1b-4c075a1835a6",
    "purpose": "fine-tuning",
    "audience": "researchers",
    "dataset_format": "jsonl",
    "complexity_level": "advanced",
    "domain": "medicine",
    "created_at": "2024-01-21T12:00:00Z"
  }
}
```

---

## Best Practices

### 1. Dataset Generation Strategy

**Model Selection:**
```python
# Choose appropriate models based on use case
generation_configs = {
    "high_quality": {
        "classification_model": "gemini-2.5-flash",
        "concurrency": 1,
        "datasets_per_chunk": 2
    },
    "high_volume": {
        "classification_model": "ollama",
        "concurrency": 4,
        "datasets_per_chunk": 5
    },
    "balanced": {
        "classification_model": "gemini",
        "concurrency": 2,
        "datasets_per_chunk": 3
    }
}
```

**Quality Thresholds:**
```python
# Set appropriate quality thresholds
quality_configs = {
    "research": {"quality_threshold": 0.9, "analyze_quality": True},
    "training": {"quality_threshold": 0.8, "analyze_quality": True},
    "prototyping": {"quality_threshold": 0.6, "analyze_quality": False}
}
```

### 2. Job Monitoring and Management

**Progress Tracking:**
```python
import requests
import time

def monitor_dataset_generation(job_id):
    """Monitor dataset generation with progress updates."""
    while True:
        response = requests.get(f'http://localhost:8000/api/v1/datasets/generate/{job_id}/status')
        status = response.json()

        print(f"Progress: {status['progress']}% - {status['current_step']}")

        if status['status'] == 'completed':
            print("Dataset generation completed!")
            return status['result']
        elif status['status'] == 'failed':
            print(f"Generation failed: {status['error']}")
            return None

        time.sleep(10)  # Check every 10 seconds
```

**Batch Processing:**
```python
def generate_multiple_datasets(project_ids, config):
    """Generate datasets for multiple projects."""
    jobs = {}

    # Start all jobs
    for project_id in project_ids:
        job_config = config.copy()
        job_config['project_id'] = project_id

        response = requests.post('http://localhost:8000/api/v1/datasets/generate', json=job_config)
        jobs[project_id] = response.json()['job_id']

    # Monitor all jobs
    results = {}
    for project_id, job_id in jobs.items():
        result = monitor_dataset_generation(job_id)
        results[project_id] = result

    return results
```

### 3. Quality Assessment and Improvement

**Quality Analysis:**
```python
def analyze_dataset_quality(dataset_id):
    """Analyze the quality distribution of a dataset."""
    response = requests.get(f'http://localhost:8000/api/v1/datasets/{dataset_id}')
    dataset = response.json()

    quality_summary = dataset['quality_summary']

    print(f"Average Quality Score: {quality_summary['average_score']:.2f}")
    print("Quality Distribution:")
    for level, count in quality_summary['distribution'].items():
        print(f"  {level.capitalize()}: {count}")

    print("Category Distribution:")
    for category, count in quality_summary['categories'].items():
        print(f"  {category}: {count}")

    return quality_summary
```

**Targeted Improvements:**
```python
def improve_low_quality_entries(dataset_id, min_quality=0.7):
    """Identify and regenerate low-quality entries."""
    # Get all entries
    response = requests.get(f'http://localhost:8000/api/v1/datasets/{dataset_id}/entries?per_page=1000')
    entries = response.json()['entries']

    # Find low-quality entries
    low_quality = [e for e in entries if e['quality_score'] < min_quality]
    low_quality_ids = [e['id'] for e in low_quality]

    if low_quality_ids:
        print(f"Found {len(low_quality_ids)} low-quality entries to regenerate")

        # Regenerate low-quality entries
        regenerate_response = requests.post(
            f'http://localhost:8000/api/v1/datasets/{dataset_id}/regenerate',
            json={
                "entry_ids": low_quality_ids,
                "regeneration_config": {
                    "model": "gemini-2.5-flash",
                    "temperature": 0.3,  # Lower temperature for higher quality
                    "improve_quality": True
                }
            }
        )

        return regenerate_response.json()
    else:
        print("No low-quality entries found")
        return None
```

### 4. Taxonomy Integration

**Taxonomy-Guided Generation:**
```python
def generate_taxonomy_aware_dataset(project_id, taxonomy_name):
    """Generate dataset with taxonomy-based categorization."""

    # First, ensure taxonomy exists
    taxonomy_response = requests.get('http://localhost:8000/api/v1/taxonomy/')
    taxonomies = taxonomy_response.json()['taxonomies']
    taxonomy = next((t for t in taxonomies if t['name'] == taxonomy_name), None)

    if not taxonomy:
        raise ValueError(f"Taxonomy '{taxonomy_name}' not found")

    # Generate dataset with taxonomy
    dataset_config = {
        "project_id": project_id,
        "taxonomy_project": taxonomy['project_id'],
        "taxonomy_name": taxonomy_name,
        "generation_mode": "question",
        "analyze_quality": True,
        "datasets_per_chunk": 3
    }

    response = requests.post('http://localhost:8000/api/v1/datasets/generate', json=dataset_config)
    return response.json()
```

### 5. Version Management

**Versioned Datasets:**
```python
def create_versioned_dataset(project_id, base_name, version="1.0.0"):
    """Create a versioned dataset with proper naming."""

    dataset_config = {
        "project_id": project_id,
        "dataset_name": f"{base_name}_v{version}",
        "enable_versioning": True,
        "generation_mode": "default",
        "format_type": "jsonl",
        "analyze_quality": True
    }

    response = requests.post('http://localhost:8000/api/v1/datasets/generate', json=dataset_config)
    job_id = response.json()['job_id']

    # Wait for completion
    result = monitor_dataset_generation(job_id)

    return result
```

### 6. Performance Optimization

**Concurrent Processing:**
```python
import concurrent.futures
import requests

def batch_generate_datasets(projects, config_template):
    """Generate datasets for multiple projects concurrently."""

    def generate_single_dataset(project):
        config = config_template.copy()
        config['project_id'] = project['id']
        config['dataset_name'] = f"{project['name']}_dataset"

        response = requests.post('http://localhost:8000/api/v1/datasets/generate', json=config)
        return {
            'project': project['name'],
            'job_id': response.json()['job_id']
        }

    # Process up to 3 projects concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(generate_single_dataset, project) for project in projects]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]

    return results
```

### 7. Error Handling and Recovery

**Robust Dataset Operations:**
```python
def safe_dataset_generation(project_id, config, max_retries=3):
    """Generate dataset with error handling and retries."""

    for attempt in range(max_retries):
        try:
            # Start generation
            response = requests.post('http://localhost:8000/api/v1/datasets/generate', json=config)
            response.raise_for_status()
            job_id = response.json()['job_id']

            # Monitor completion
            result = monitor_dataset_generation(job_id)

            if result and result.get('error'):
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1} failed: {result['error']}. Retrying...")
                    time.sleep(5 * (attempt + 1))  # Exponential backoff
                    continue
                else:
                    raise Exception(f"Dataset generation failed after {max_retries} attempts: {result['error']}")

            return result

        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                print(f"Network error on attempt {attempt + 1}: {e}. Retrying...")
                time.sleep(5 * (attempt + 1))
                continue
            else:
                raise Exception(f"Dataset generation failed after {max_retries} attempts due to network errors")

    return None
```

This API provides comprehensive dataset generation and management capabilities with support for quality assessment, taxonomy integration, versioning, and performance optimization suitable for both development and production workflows.