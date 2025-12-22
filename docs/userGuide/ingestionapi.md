# Ingestion Module API Usage Guide

The Compileo Ingestion API provides programmatic access to document parsing and processing capabilities. This guide covers all ingestion-related API endpoints with examples and best practices.

## Base URL: `/api/v1`

All ingestion operations are performed through the main API endpoints. This guide focuses on the ingestion-specific functionality.

---

## Document Parsing API

### Parse Documents

**Endpoint:** `POST /api/v1/documents/parse`

Initiates asynchronous document parsing with automatic PDF splitting support.

**Request Body:**
```json
{
  "project_id": 1,
  "document_ids": [1, 2, 3],
  "parser": "gemini",
  "pages_per_split": 5,
  "overlap_pages": 0
}
```

**Parameters:**
- `project_id` (integer, required): Target project ID
- `document_ids` (array, required): List of document IDs to parse
- `parser` (string, optional): AI parser (`gemini`, `grok`, `ollama`, `pypdf`, `unstructured`, `huggingface`, `novlm`)
- `pages_per_split` (integer, optional): Pages per PDF chunk (default: 5)
- `overlap_pages` (integer, optional): Overlap pages between chunks (default: 0)

**Response:**
```json
{
  "job_id": "parse-job-uuid-123",
  "message": "Parsing job submitted successfully",
  "document_count": 3
}
```

**PDF Splitting & Analysis Architecture:**
- PDFs are automatically split when `total_pages > pages_per_split`.
- **Two-Pass VLM Strategy**: For VLM parsers (`grok`, `gemini`, `openai`, `ollama`, `huggingface`), a structure analysis is performed *before* parsing content.
    - **Structure Skim**: The system analyzes the middle chunk (or full file if not split) to detect the visual hierarchy (fonts, sizes for H1, H2, etc.) at 300 DPI.
    - **Style Guide**: A resulting "Style Guide" is generated and injected into the prompt for *every* page parsed.
- Each chunk is parsed individually with the selected AI model, using the global Style Guide for consistent heading hierarchy.
- Results are stored as separate JSON files per chunk.
- Manifest file tracks split relationships and metadata.

---

## Document Upload API

### Upload Documents

**Endpoint:** `POST /api/v1/documents/upload`

Uploads documents to a project with automatic ingestion processing.

**Request Body (multipart/form-data):**
- `files`: Document files (PDF, DOCX, TXT, etc.)
- `project_id`: Target project ID
- `auto_parse`: Automatically start parsing after upload (default: false)
- `parser`: Parser to use if auto_parse is enabled
- `pages_per_split`: PDF splitting configuration

**Response:**
```json
{
  "job_id": "upload-job-uuid-456",
  "message": "Documents uploaded successfully",
  "uploaded_count": 2,
  "document_ids": [4, 5]
}
```

---

## Parser Configuration

### Available Parsers

| Parser | Description | API Key Required | GPU Support |
|--------|-------------|------------------|-------------|
| `gemini` | Google Gemini with file upload | Yes | Yes |
| `grok` | xAI Grok with preprocessing | Yes | No |
| `ollama` | Local Ollama models | No | No |
| `pypdf` | Direct PDF text extraction | No | No |
| `unstructured` | Office document parsing | No | No |
| `huggingface` | GPU-accelerated OCR | Yes | Yes |
| `novlm` | Intelligent parser selection | No | No |

### Parser-Specific Settings

**Ollama Parser:**
```json
{
  "parser": "ollama",
  "model": "llama2:7b",
  "temperature": 0.1,
  "num_predict": 512
}
```

**Gemini Parser:**
```json
{
  "parser": "gemini",
  "model": "gemini-2.5-flash"
}
```

---

## PDF Splitting Details

### Automatic Splitting Logic

PDFs are automatically split when:
- `total_pages > pages_per_split` (default: 5)
- Document exceeds AI model token limits
- Improves parsing quality for large documents

### Split File Structure

```
storage/uploads/{project_id}/
├── {doc_id}_{filename}_manifest.json    # Split metadata
├── {doc_id}_{filename}_chunk_001.pdf   # Pages 1-5
├── {doc_id}_{filename}_chunk_002.pdf   # Pages 6-10
└── {doc_id}_{filename}_chunk_003.pdf   # Pages 11-15

storage/parsed/{project_id}/
├── {doc_id}_1.json    # Parsed chunk 1
├── {doc_id}_2.json    # Parsed chunk 2
└── {doc_id}_3.json    # Parsed chunk 3
```

### Manifest File Format

```json
{
  "original_file": "document.pdf",
  "total_pages": 15,
  "pages_per_split": 5,
  "overlap_pages": 0,
  "splits": [
    {
      "chunk_id": 1,
      "start_page": 1,
      "end_page": 5,
      "filename": "chunk_001.pdf"
    },
    {
      "chunk_id": 2,
      "start_page": 6,
      "end_page": 10,
      "filename": "chunk_002.pdf"
    },
    {
      "chunk_id": 3,
      "start_page": 11,
      "end_page": 15,
      "filename": "chunk_003.pdf"
    }
  ]
}
```

---

## Job Monitoring

### Check Parse Job Status

**Endpoint:** `GET /api/v1/jobs/status/{job_id}`

**Response:**
```json
{
  "job_id": "parse-job-uuid-123",
  "status": "completed",
  "progress": 1.0,
  "created_at": "2025-11-11T10:25:53Z",
  "completed_at": "2025-11-11T10:27:23Z",
  "result": {
    "processed_documents": 3,
    "total_chunks_created": 9,
    "parser": "gemini",
    "pages_per_split": 5
  }
}
```

---

## Best Practices

### 1. Parser Selection

**For Speed:**
- Use `pypdf` for simple text extraction
- Use `unstructured` for Office documents

**For Quality:**
- Use `gemini` or `grok` for complex layouts
- Use `ollama` for local processing

### 2. PDF Splitting

**Small Documents (< 50 pages):**
- Set `pages_per_split: 10` to avoid unnecessary splitting

**Large Documents (> 200 pages):**
- Set `pages_per_split: 5` for optimal parsing
- Use `overlap_pages: 1` for context continuity

### 3. Batch Processing

**Optimal Batch Sizes:**
- 5-10 documents per job for AI parsers
- 20-50 documents per job for fast parsers
- Monitor job queue status to avoid overload

### 4. Error Handling

**Common Issues:**
- API key validation before job submission
- File format verification
- Storage space monitoring
- **HuggingFace Network Timeouts**: The `huggingface` parser downloads a large (~6GB) model. In Docker, this can hang due to network/SSL issues. Pre-populating the `compileo_hf_models` volume with model weights is recommended if parsing jobs remain stuck in the "running" state without progress.

**Retry Logic:**
- Failed jobs can be restarted
- Different parsers can be tried
- Chunk size adjustment for parsing failures

---

## API Examples

### Python Client Example

```python
import requests

# Parse documents with PDF splitting
response = requests.post(
    "http://localhost:8000/api/v1/documents/parse",
    json={
        "project_id": 1,
        "document_ids": [1, 2, 3],
        "parser": "gemini",
        "pages_per_split": 5,
        "overlap_pages": 0
    }
)

job_id = response.json()["job_id"]

# Monitor job progress
while True:
    status = requests.get(f"http://localhost:8000/api/v1/jobs/status/{job_id}")
    if status.json()["status"] == "completed":
        break
    time.sleep(5)
```

### cURL Example

```bash
# Parse documents
curl -X POST http://localhost:8000/api/v1/documents/parse \
  -H "Content-Type: application/json" \
  -d '{
    "project_id": 1,
    "document_ids": [1, 2],
    "parser": "gemini",
    "pages_per_split": 5
  }'
```

This API provides complete programmatic access to Compileo's document ingestion and parsing capabilities, with automatic PDF splitting for optimal AI model performance.