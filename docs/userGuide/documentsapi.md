# Documents Module API Usage Guide

The Compileo Documents API provides comprehensive REST endpoints for document management, including upload, parsing, processing, and content retrieval. This guide covers all available API endpoints with examples and best practices.

## Base URL: `/api/v1/documents`

---

## Document Upload

### POST `/upload`

Upload one or more documents to a project.

**Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -F "project_id=1" \
  -F "files=@document1.pdf" \
  -F "files=@document2.docx"
```

**Request Body (multipart/form-data):**
- `files`: Document files (multiple allowed)
- `project_id`: Target project ID

**Response:**
```json
{
  "job_id": "doc_upload_12345",
  "message": "Documents uploaded successfully",
  "files_count": 2,
  "uploaded_files": [
    {
      "id": 101,
      "project_id": 1,
      "file_name": "document1.pdf",
      "source_file_path": "storage/uploads/1/document1.pdf",
      "parsed_file_path": null,
      "created_at": "2024-01-21T12:00:00Z",
      "status": "uploaded"
    }
  ]
}
```

---

## Document Parsing

### POST `/parse`

Parse documents to paginated markdown files without chunking. The consolidated `.md` file is no longer created to avoid storage duplication.

**Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/documents/parse" \
  -H "Content-Type: application/json" \
  -d '{
    "project_id": 1,
    "document_ids": [101, 102],
    "parser": "gemini"
  }'
```

**Request Body:**
- `project_id`: Project containing documents
- `document_ids`: Array of document IDs to parse
- `parser`: Parser type (`gemini`, `grok`, `ollama`, `pypdf`, `unstructured`, `huggingface`, `novlm`)

**Response:**
```json
{
  "job_id": "parse_job_12345",
  "message": "Successfully parsed 2 documents to markdown",
  "parsed_documents": 2,
  "status": "completed"
}
```

---

## Document Processing

### POST `/process`

Process documents with parsing and chunking in a single operation.

**Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/documents/process" \
  -H "Content-Type: application/json" \
  -d '{
    "project_id": 1,
    "document_ids": [101, 102],
    "parser": "gemini",
    "chunk_strategy": "character",
    "character_chunk_size": 500,
    "character_overlap": 50
  }'
```

**Request Body:**
- `project_id`: Project containing documents
- `document_ids`: Array of document IDs to process
- `parser`: Parser type
- `chunk_strategy`: Chunking strategy (`token`, `character`, `semantic`, `delimiter`, `schema`)
- `chunk_size`: Chunk size (tokens/characters)
- `overlap`: Overlap between chunks
- `skip_parsing`: Skip parsing if documents are already parsed

**Response:**
```json
{
  "job_id": "process_job_12345",
  "message": "Successfully processed 2 documents, created 15 chunks",
  "processed_documents": 2,
  "total_chunks": 15,
  "status": "completed"
}
```

---

## Document Content Retrieval

### GET `/{document_id}/content`

Retrieve paginated parsed document content with pagination support. The consolidated `.md` file is no longer created.

**Request:**
```bash
curl "http://localhost:8000/api/v1/documents/101/content?parsed_file=101_1.md"
```

**Query Parameters:**
- `parsed_file`: Name of specific paginated parsed file to retrieve full content (e.g., "document_1.md"). If omitted, returns a list of available paginated parsed files.
- `page`: Page number for pagination (only used when `parsed_file` is not specified).
- `page_size`: Number of characters per page (only used when `parsed_file` is not specified).

**Response (when `parsed_file` is specified):**
```json
{
  "document_id": 101,
  "file_name": "document1.pdf",
  "content": "Full content of the parsed file...",
  "total_length": 10891,
  "word_count": 2156,
  "line_count": 234,
  "current_file": "101_1.md",
  "parsed_file": "101_1.md"
}
```

**Response (when `parsed_file` is omitted):**
```json
{
  "document_id": 101,
  "file_name": "document1.pdf",
  "parsed_files": [
    "101_1.md",
    "101_2.md",
    "101_3.md"
  ],
  "total_files": 3,
  "manifest_path": "storage/parsed/101/manifest.json"
}
```

---

## Document Management

### GET `/`

List documents with optional project filtering.

**Request:**
```bash
curl "http://localhost:8000/api/v1/documents/?project_id=1"
```

**Query Parameters:**
- `project_id`: Filter by project ID (optional)

**Response:**
```json
{
  "documents": [
    {
      "id": 101,
      "project_id": 1,
      "file_name": "document1.pdf",
      "source_file_path": "storage/uploads/1/document1.pdf",
      "parsed_file_path": null,
      "created_at": "2024-01-21T12:00:00Z",
      "status": "parsed"
    }
  ],
  "total": 1
}
```

### DELETE `/{document_id}`

Delete a document and all associated files.

**Request:**
```bash
curl -X DELETE "http://localhost:8000/api/v1/documents/101"
```

**Response:**
```json
{
  "message": "Document 'document1.pdf' and all associated files deleted successfully"
}
```

---

## Job Status Monitoring

### GET `/upload/{job_id}/status`

Check upload job status.

**Request:**
```bash
curl "http://localhost:8000/api/v1/documents/upload/doc_upload_12345/status"
```

### GET `/process/{job_id}/status`

Check processing job status.

**Request:**
```bash
curl "http://localhost:8000/api/v1/documents/process/process_job_12345/status"
```

**Response:**
```json
{
  "job_id": "process_job_12345",
  "status": "completed",
  "progress": 100,
  "current_step": "Processing complete",
  "estimated_completion": "2024-01-21T12:05:00Z",
  "processed_files": []
}
```

---

## PDF Preprocessing

### POST `/split-pdf`

Split large PDFs into smaller chunks for processing.

**Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/documents/split-pdf" \
  -H "Content-Type: application/json" \
  -d '{
    "pdf_path": "large_document.pdf",
    "pages_per_split": 200,
    "overlap_pages": 1
  }'
```

**Request Body:**
- `pdf_path`: Path to PDF file
- `pages_per_split`: Pages per split (default: 200)
- `overlap_pages`: Overlapping pages (default: 1)

**Response:**
```json
{
  "split_files": ["split_1.pdf", "split_2.pdf"],
  "message": "Successfully split PDF into 2 files",
  "total_splits": 2,
  "manifest_path": "manifest.json"
}
```

---

## Best Practices

### 1. Batch Processing
```python
import requests

# Upload multiple files
files = [
    ('files', open('doc1.pdf', 'rb')),
    ('files', open('doc2.pdf', 'rb'))
]
response = requests.post(
    'http://localhost:8000/api/v1/documents/upload',
    files=files,
    data={'project_id': 1}
)

# Process all uploaded documents
doc_ids = [doc['id'] for doc in response.json()['uploaded_files']]
process_response = requests.post(
    'http://localhost:8000/api/v1/documents/process',
    json={
        'project_id': 1,
        'document_ids': doc_ids,
        'parser': 'gemini',
        'chunk_strategy': 'character',
        'character_chunk_size': 500
    }
)
```

### 2. Progress Monitoring
```python
import time

job_id = process_response.json()['job_id']

while True:
    status_response = requests.get(
        f'http://localhost:8000/api/v1/documents/process/{job_id}/status'
    )
    status = status_response.json()

    if status['status'] == 'completed':
        print("Processing completed!")
        break
    elif status['status'] == 'failed':
        print("Processing failed!")
        break

    print(f"Progress: {status['progress']}%")
    time.sleep(5)
```

### 3. Content Pagination
```python
# Get document content (list paginated files)
content_response = requests.get(
    'http://localhost:8000/api/v1/documents/101/content'
)
data = content_response.json()
print(f"Available parsed files: {data['parsed_files']}")

# Get content of a specific paginated file
file_content_response = requests.get(
    f"http://localhost:8000/api/v1/documents/101/content?parsed_file={data['parsed_files'][0]}"
)
file_data = file_content_response.json()
print(f"Content of {file_data['parsed_file']}: {file_data['content'][:200]}...")
```

### 4. Error Handling
```python
try:
    response = requests.post('http://localhost:8000/api/v1/documents/upload', ...)
    response.raise_for_status()
    # Process successful response
except requests.exceptions.HTTPError as e:
    if e.response.status_code == 404:
        print("Project not found")
    elif e.response.status_code == 400:
        print("Invalid file type")
    else:
        print(f"API error: {e}")
except requests.exceptions.RequestException as e:
    print(f"Network error: {e}")
```

This API provides comprehensive document management capabilities with support for multiple file formats, various parsing options, and flexible chunking strategies suitable for both development and production workflows.