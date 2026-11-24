# Compileo API End-to-End Example

This guide walks through a complete pipeline using the Compileo API, from authentication to dataset generation.

## Prerequisites

- Compileo API running at `http://localhost:8000`
- API documentation available at `http://localhost:8000/docs`

---

## Step 1: Authentication

### Register a New User

```bash
curl -X POST "http://localhost:8000/api/v1/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "your_password",
    "full_name": "John Doe"
  }'
```

**Response:**
```json
{
  "id": "900057c0-fc2d-4c67-98f3-9c7ecffc5ce5",
  "email": "user@example.com",
  "full_name": "John Doe",
  "is_active": true,
  "created_at": "2026-02-25T12:00:00Z"
}
```

### Login to Get Access Token

```bash
curl -X POST "http://localhost:8000/api/v1/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=user@example.com&password=your_password"
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

### Get Current User Info

```bash
curl -X GET "http://localhost:8000/api/v1/auth/me" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

---

## Step 2: Project Management

### Create a Project

```bash
curl -X POST "http://localhost:8000/api/v1/projects/" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Medical Dataset Project",
    "description": "Creating a dataset for healthcare AI training"
  }'
```

**Response:**
```json
{
  "id": "24d5f486-78be-48e6-bf5c-f1df048b1d78",
  "name": "Medical Dataset Project",
  "description": "Creating a dataset for healthcare AI training",
  "created_at": "2026-02-25T12:05:00Z",
  "updated_at": null,
  "document_count": 0,
  "dataset_count": 0,
  "status": "active"
}
```

### List Projects

```bash
curl -X GET "http://localhost:8000/api/v1/projects/" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

---

## Step 3: Document Upload

### Upload a Document

```bash
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -F "files=@medical_article.pdf" \
  -F "project_id=24d5f486-78be-48e6-bf5c-f1df048b1d78"
```

**Response:**
```json
{
  "job_id": "doc_upload_abc123",
  "message": "Documents uploaded successfully",
  "files_count": 1,
  "uploaded_files": [
    {
      "id": 101,
      "project_id": "24d5f486-78be-48e6-bf5c-f1df048b1d78",
      "file_name": "medical_article.pdf",
      "source_file_path": "storage/uploads/24d5f486-78be-48e6-bf5c-f1df048b1d78/medical_article.pdf",
      "parsed_file_path": null,
      "created_at": "2026-02-25T12:10:00Z",
      "status": "uploaded"
    }
  ]
}
```

---

## Step 4: Document Parsing

### Parse Uploaded Documents

```bash
curl -X POST "http://localhost:8000/api/v1/documents/parse" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "project_id": "24d5f486-78be-48e6-bf5c-f1df048b1d78",
    "document_ids": [101],
    "parser": "gemini"
  }'
```

**Response:**
```json
{
  "job_id": "parse-job-uuid-123",
  "message": "Parsing job submitted successfully",
  "parsed_documents": 1
}
```

### Check Parsing Job Status

```bash
curl -X GET "http://localhost:8000/api/v1/jobs/status/parse-job-uuid-123" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

**Response:**
```json
{
  "job_id": "parse-job-uuid-123",
  "job_type": "parsing",
  "status": "completed",
  "progress": 1.0,
  "created_at": "2026-02-25T12:10:00Z",
  "started_at": "2026-02-25T12:10:05Z",
  "completed_at": "2026-02-25T12:12:00Z",
  "error": null
}
```

---

## Step 5: Document Chunking

### Chunk Parsed Documents

```bash
curl -X POST "http://localhost:8000/api/v1/documents/chunk" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "project_id": "24d5f486-78be-48e6-bf5c-f1df048b1d78",
    "document_ids": [101],
    "chunker": "gemini",
    "chunk_strategy": "schema",
    "chunk_size": 512,
    "overlap": 50
  }'
```

**Response:**
```json
{
  "job_id": "chunk-job-uuid-456",
  "message": "Chunking job submitted successfully",
  "processed_documents": 1,
  "total_chunks": 15
}
```

### Get Document Chunks

```bash
curl -X GET "http://localhost:8000/api/v1/documents/chunks/document/101" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

**Response:**
```json
{
  "document_id": 101,
  "chunks": [
    {
      "id": "chunk_001",
      "chunk_index": 1,
      "token_count": 278,
      "file_path": "storage/chunks/24d5f486-78be-48e6-bf5c-f1df048b1d78/101/chunk_1.md",
      "content_preview": "# PROGNOSIS\nMost patients respond well...",
      "chunk_strategy": "schema"
    }
  ],
  "total": 15
}
```

---

## Step 6: Taxonomy Generation

### Generate Taxonomy from Chunks

```bash
curl -X POST "http://localhost:8000/api/v1/taxonomy/generate" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "project_id": "24d5f486-78be-48e6-bf5c-f1df048b1d78",
    "name": "Medical Categories",
    "documents": [101],
    "depth": 3,
    "generator": "gemini",
    "domain": "medicine",
    "batch_size": 10,
    "processing_mode": "fast"
  }'
```

**Response:**
```json
{
  "job_id": "taxonomy-job-uuid-789",
  "status": "pending",
  "message": "Taxonomy generation job created"
}
```

### Get Taxonomy Details

```bash
curl -X GET "http://localhost:8000/api/v1/taxonomy/1" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

**Response:**
```json
{
  "id": 1,
  "project_id": "24d5f486-78be-48e6-bf5c-f1df048b1d78",
  "name": "Medical Categories",
  "taxonomy_data": {
    "name": "Medical Categories",
    "categories": [
      {
        "id": "10000",
        "name": "Diseases and Conditions",
        "children": [
          {"id": "11000", "name": "Immune Dysregulation"},
          {"id": "12000", "name": "Cardiovascular Disorders"}
        ]
      }
    ]
  },
  "created_at": "2026-02-25T12:15:00Z"
}
```

---

## Step 7: Content Extraction

### Create Extraction Job

```bash
curl -X POST "http://localhost:8000/api/v1/extraction/" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "taxonomy_id": 1,
    "selected_categories": ["11000", "12000"],
    "parameters": {
      "confidence_threshold": 0.7,
      "max_chunks": 100
    },
    "initial_classifier": "gemini",
    "enable_validation_stage": true,
    "validation_classifier": "gemini",
    "extraction_type": "ner",
    "extraction_mode": "contextual"
  }'
```

**Response:**
```json
{
  "job_id": "extraction-job-uuid-101",
  "status": "pending",
  "message": "Extraction job created and queued for processing"
}
```

### Get Extraction Results

```bash
curl -X GET "http://localhost:8000/api/v1/extraction/extraction-job-uuid-101/results" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

**Response:**
```json
{
  "job_id": "extraction-job-uuid-101",
  "results": [
    {
      "chunk_id": "chunk_001",
      "classifications": {
        "custom": {
          "Diseases and Conditions": "Immune Dysregulation"
        }
      },
      "confidence_score": 0.9,
      "categories_matched": ["11000"],
      "metadata": {
        "extracted_data": {
          "11000": {
            "category_name": "Immune Dysregulation",
            "chunks": [
              {
                "chunk_id": "chunk_001",
                "text": "Patient presented with fever...",
                "confidence": 0.9,
                "classified_into": "Diseases and Conditions"
              }
            ]
          }
        }
      }
    }
  ],
  "total_results": 25,
  "page": 1,
  "page_size": 50,
  "has_more": false
}
```

---

## Step 8: Dataset Generation

### Generate Dataset from Extraction Results

```bash
curl -X POST "http://localhost:8000/api/v1/datasets/generate" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "project_id": "24d5f486-78be-48e6-bf5c-f1df048b1d78",
    "classification_model": "gemini",
    "generation_mode": "question_and_answer",
    "format_type": "jsonl",
    "datasets_per_chunk": 5,
    "quality_threshold": 0.70,
    "data_source": "Extract",
    "extraction_file_id": "extraction-job-uuid-101",
    "audience": "healthcare professionals",
    "purpose": "creating evidence-based question answer pairs for fine tuning large language models",
    "domain": "medicine"
  }'
```

**Response:**
```json
{
  "job_id": "dataset-gen-job-uuid-202",
  "message": "Dataset generation started"
}
```

### Check Dataset Generation Status

```bash
curl -X GET "http://localhost:8000/api/v1/jobs/status/dataset-gen-job-uuid-202" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

### Get Generated Dataset

```bash
curl -X GET "http://localhost:8000/api/v1/datasets/1/download" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

**Response:**
```json
{
  "dataset_id": 1,
  "format": "jsonl",
  "download_url": "/api/v1/datasets/storage/24d5f486-78be-48e6-bf5c-f1df048b1d78/datasets/dataset_1.jsonl",
  "record_count": 150,
  "file_size_bytes": 45678
}
```

---

## Complete Pipeline Summary

| Step | Endpoint | Description |
|------|----------|-------------|
| 1 | `POST /api/v1/auth/register` | Register new user |
| 2 | `POST /api/v1/auth/login` | Get access token |
| 3 | `POST /api/v1/projects/` | Create project |
| 4 | `POST /api/v1/documents/upload` | Upload document |
| 5 | `POST /api/v1/documents/parse` | Parse document to markdown |
| 6 | `POST /api/v1/documents/chunk` | Chunk parsed content |
| 7 | `POST /api/v1/taxonomy/generate` | Generate taxonomy |
| 8 | `POST /api/v1/extraction/` | Extract content by taxonomy |
| 9 | `POST /api/v1/datasets/generate` | Generate Q&A dataset |
| 10 | `GET /api/v1/datasets/{id}/download` | Download dataset |
| 11 | `GET /api/v1/datasets/{id}/details` | Get dataset details |
| 12 | `DELETE /api/v1/datasets/{id}/delete` | Delete dataset |

---

## Environment Variables

Make sure your environment has the required API keys configured:

```bash
# In your .env file or environment
GEMINI_API_KEY=your_gemini_api_key
GROK_API_KEY=your_grok_api_key
OLLAMA_HOST=localhost:11434
REDIS_URL=redis://localhost:6379
```

---

## Error Handling

All endpoints return standard HTTP status codes:
- `200` - Success
- `400` - Bad Request (invalid parameters)
- `401` - Unauthorized (invalid or missing token)
- `404` - Not Found (resource doesn't exist)
- `500` - Internal Server Error

Error response format:
```json
{
  "detail": "Error message description"
}