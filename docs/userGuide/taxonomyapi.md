# Taxonomy Module API Usage Guide

The Compileo Taxonomy API provides comprehensive REST endpoints for taxonomy management, including creation, generation, extension, and retrieval. This guide covers all available API endpoints with examples and best practices.

## Base URL: `/api/v1/taxonomy`

---

## Taxonomy Creation

### POST `/`

Create a new manual taxonomy from JSON structure.

**Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/taxonomy/" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Medical Conditions",
    "description": "Classification of medical conditions",
    "project_id": "b357e573-89a5-4b40-8e1b-4c075a1835a6",
    "taxonomy": {
      "name": "Medical Conditions",
      "description": "Hierarchical classification of medical conditions",
      "children": [
        {
          "name": "Cardiovascular",
          "description": "Heart and blood vessel conditions",
          "confidence_threshold": 0.8,
          "children": []
        },
        {
          "name": "Respiratory",
          "description": "Lung and breathing conditions",
          "confidence_threshold": 0.8,
          "children": []
        }
      ]
    }
  }'
```

**Request Body:**
- `name`: Taxonomy name (required)
- `description`: Taxonomy description (optional)
- `project_id`: Associated project ID (required)
- `taxonomy`: JSON taxonomy structure (required)

**Response:**
```json
{
  "id": "069671d5-6c2c-4327-88dc-6abd12b5671c",
  "name": "Medical Conditions",
  "description": "Classification of medical conditions",
  "project_id": "b357e573-89a5-4b40-8e1b-4c075a1835a6",
  "categories_count": 3,
  "confidence_score": 0.8,
  "created_at": "2024-01-21T12:00:00Z",
  "file_path": "storage/taxonomy/b357e573-89a5-4b40-8e1b-4c075a1835a6/manual_taxonomy_uuid.json"
}
```

---

## AI Taxonomy Generation

### POST `/generate`

Generate a new taxonomy using AI from document chunks.

**Important:** Documents must be parsed and chunked before taxonomy generation. The system follows a multi-step workflow: Upload → Parse → Chunk → Generate Taxonomy.

**Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/taxonomy/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "project_id": "b357e573-89a5-4b40-8e1b-4c075a1835a6",
    "name": "AI Generated Taxonomy",
    "documents": ["b0b69234-8f99-41d2-a0e7-b7cd5cfcb593", "a1c2d3e4-f5g6-h7i8-j9k0-l1m2n3o4p5q6"],
    "depth": 3,
    "generator": "gemini",
    "domain": "medical",
    "batch_size": 10,
    "category_limits": [5, 10, 15],
    "specificity_level": 1,
    "processing_mode": "complete"
  }'
```

**Request Body:**
- `project_id`: Project containing documents (required)
- `name`: Taxonomy name (optional)
- `documents`: Array of document IDs (required) - documents must be parsed and chunked
- `depth`: Taxonomy hierarchy depth (default: 3)
- `generator`: AI model (`gemini`, `grok`, `ollama`, `openai`) (default: `gemini`)
- `domain`: Content domain (default: `general`)
- `batch_size`: Number of complete chunks to process per batch (default: 10)
- `category_limits`: Max categories per level (optional)
- `specificity_level`: Specificity level 1-5 (default: 1)
- `processing_mode`: Generation mode (`"fast"` or `"complete"`) (default: `"fast"`)
  - `"fast"`: Samples a single batch of chunks (quickest, but may miss content).
  - `"complete"`: Iteratively processes 100% of chunks in batches (comprehensive, but slower).

**Prerequisites:**
- Documents must be uploaded and parsed
- Documents must be chunked using the chunking API
- Chunks are stored as individual `.md` files

**Response:**
```json
{
  "id": "069671d5-6c2c-4327-88dc-6abd12b5671c",
  "name": "AI Generated Taxonomy",
  "description": "AI-generated taxonomy using gemini",
  "project_id": "b357e573-89a5-4b40-8e1b-4c075a1835a6",
  "categories_count": 45,
  "confidence_score": 0.85,
  "created_at": "2024-01-21T12:05:00Z",
  "file_path": "storage/taxonomy/b357e573-89a5-4b40-8e1b-4c075a1835a6/ai_taxonomy_uuid.json"
}
```

---

## Taxonomy Extension

### POST `/extend`

Extend an existing taxonomy with additional hierarchy levels.

**Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/taxonomy/extend" \
  -H "Content-Type: application/json" \
  -d '{
    "taxonomy_id": 102,
    "additional_depth": 2,
    "generator": "gemini",
    "domain": "medical",
    "batch_size": 10,
    "category_limits": [8, 12],
    "specificity_level": 2,
    "processing_mode": "complete"
  }'
```

**Request Body:**
- `taxonomy_id`: ID of taxonomy to extend (required)
- `additional_depth`: Number of levels to add (default: 2)
- `generator`: AI model for extension (`gemini`, `grok`, `ollama`, `openai`) (default: `gemini`)
- `domain`: Content domain (default: `general`)
- `batch_size`: Number of complete chunks to process per batch (optional)
- `category_limits`: Max categories per new level (optional)
- `specificity_level`: Specificity level 1-5 (default: 1)
- `documents`: Array of document IDs to analyze for extension context (optional)
- `processing_mode`: Generation mode (`"fast"` or `"complete"`) (default: `"fast"`)

**Alternative: Extend from taxonomy data**
```json
{
  "taxonomy_data": {
    "name": "Cardiovascular",
    "description": "Heart conditions",
    "children": []
  },
  "project_id": 1,
  "additional_depth": 2,
  "generator": "gemini"
}
```

**Response:**
```json
{
  "id": 102,
  "name": "AI Generated Taxonomy",
  "description": "AI-extended taxonomy with 2 additional levels using gemini",
  "project_id": 1,
  "categories_count": 78,
  "confidence_score": 0.82,
  "created_at": "2024-01-21T12:10:00Z",
  "file_path": "storage/taxonomy/1/ai_taxonomy_uuid.json"
}
```

---

## Taxonomy Retrieval

### GET `/`

List all available taxonomies with optional project filtering.

**Request:**
```bash
curl "http://localhost:8000/api/v1/taxonomy/?project_id=b357e573-89a5-4b40-8e1b-4c075a1835a6"
```

**Query Parameters:**
- `project_id`: Filter by project ID (optional)

**Response:**
```json
{
  "taxonomies": [
    {
      "id": "069671d5-6c2c-4327-88dc-6abd12b5671c",
      "name": "Medical Conditions",
      "description": "Classification of medical conditions",
      "project_id": "b357e573-89a5-4b40-8e1b-4c075a1835a6",
      "categories_count": 45,
      "confidence_score": 0.85,
      "created_at": "2024-01-21T12:00:00Z",
      "file_path": "storage/taxonomy/b357e573-89a5-4b40-8e1b-4c075a1835a6/manual_taxonomy_uuid.json"
    },
    {
      "id": "1a2b3c4d-5e6f-7g8h-9i0j-1k2l3m4n5o6p",
      "name": "AI Generated Taxonomy",
      "description": "AI-generated taxonomy using gemini",
      "project_id": "b357e573-89a5-4b40-8e1b-4c075a1835a6",
      "categories_count": 78,
      "confidence_score": 0.82,
      "created_at": "2024-01-21T12:05:00Z",
      "file_path": "storage/taxonomy/b357e573-89a5-4b40-8e1b-4c075a1835a6/ai_taxonomy_uuid.json"
    }
  ],
  "total": 2
}
```

### GET `/{taxonomy_id}`

Get detailed taxonomy information including structure and analytics.

**Request:**
```bash
curl "http://localhost:8000/api/v1/taxonomy/069671d5-6c2c-4327-88dc-6abd12b5671c"
```

**Response:**
```json
{
  "taxonomy": {
    "name": "Medical Conditions",
    "description": "Classification of medical conditions",
    "children": [
      {
        "name": "Cardiovascular",
        "description": "Heart and blood vessel conditions",
        "confidence_threshold": 0.8,
        "children": [
          {
            "name": "Coronary Artery Disease",
            "description": "Blockage of coronary arteries",
            "confidence_threshold": 0.85,
            "children": []
          }
        ]
      }
    ]
  },
  "metadata": {
    "type": "manual",
    "confidence_score": 0.8,
    "created_manually": true
  },
  "analytics": {
    "depth_analysis": {
      "total_categories": 3,
      "max_depth": 2
    }
  }
}
```

---

## Taxonomy Management

### PUT `/{taxonomy_id}`

Update taxonomy information.

**Request:**
```bash
curl -X PUT "http://localhost:8000/api/v1/taxonomy/069671d5-6c2c-4327-88dc-6abd12b5671c" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Updated Medical Conditions"
  }'
```

**Request Body:**
- `name`: New taxonomy name (required)

**Response:**
```json
{
  "id": 101,
  "name": "Updated Medical Conditions",
  "description": "Classification of medical conditions",
  "project_id": 1,
  "categories_count": 45,
  "confidence_score": 0.85,
  "created_at": "2024-01-21T12:00:00Z",
  "file_path": "storage/taxonomy/1/manual_taxonomy_uuid.json"
}
```

### DELETE `/{taxonomy_id}`

Delete a taxonomy. This operation performs a **complete cleanup**, removing the taxonomy's file from the filesystem and deleting all associated extraction jobs and their results (both database entries and filesystem files).

**Request:**
```bash
curl -X DELETE "http://localhost:8000/api/v1/taxonomy/069671d5-6c2c-4327-88dc-6abd12b5671c"
```

**Response:**
```json
{
  "message": "Taxonomy 101 deleted successfully. Associated files and extraction results have been cleaned up."
}
```

### DELETE `/`

Bulk delete multiple taxonomies. This operation performs a **complete cleanup** for all specified taxonomies, removing their files from the filesystem and deleting all associated extraction jobs and their results (both database entries and filesystem files).

**Request:**
```bash
curl -X DELETE "http://localhost:8000/api/v1/taxonomy/" \
  -H "Content-Type: application/json" \
  -d '{
    "taxonomy_ids": ["069671d5-6c2c-4327-88dc-6abd12b5671c", "1a2b3c4d-5e6f-7g8h-9i0j-1k2l3m4n5o6p", "2b3c4d5-6e7f-8g9h-0i1j-2k3l4m5n6o7p"]
  }'
```

**Request Body:**
- `taxonomy_ids`: Array of taxonomy IDs to delete (required)

**Response:**
```json
{
  "message": "Successfully deleted 3 taxonomies",
  "deleted": [101, 102, 103]
}
```

---

## Best Practices

### 1. Taxonomy Generation

**Document Selection:**
```python
import requests

# Get available documents
docs_response = requests.get('http://localhost:8000/api/v1/documents/?project_id=1')
documents = docs_response.json()['documents']

# Select documents with chunks
doc_ids = [doc['id'] for doc in documents if doc.get('status') == 'parsed']

# Generate taxonomy
taxonomy_response = requests.post('http://localhost:8000/api/v1/taxonomy/generate', json={
    'project_id': 1,
    'name': 'Medical Taxonomy',
    'documents': doc_ids,
    'depth': 3,
    'generator': 'gemini',
    'domain': 'medical',
    'batch_size': 10
})
```

**Category Limits:**
```python
# Balanced taxonomy structure
category_limits = [5, 10, 15]  # Level 1: 5, Level 2: 10, Level 3: 15

requests.post('http://localhost:8000/api/v1/taxonomy/generate', json={
    'project_id': 1,
    'documents': doc_ids,
    'category_limits': category_limits,
    'specificity_level': 2  # More specific categories
})
```

### 2. Taxonomy Extension

**Incremental Growth:**
```python
# Extend existing taxonomy with specific documents
requests.post('http://localhost:8000/api/v1/taxonomy/extend', json={
    'taxonomy_id': 101,
    'additional_depth': 1,
    'generator': 'gemini',
    'domain': 'medical',
    'documents': [101, 102]
})
```

**Category-Specific Extension:**
```python
# Get taxonomy details
tax_response = requests.get('http://localhost:8000/api/v1/taxonomy/101')
taxonomy = tax_response.json()['taxonomy']

# Find specific category
cardiovascular = None
for category in taxonomy.get('children', []):
    if category['name'] == 'Cardiovascular':
        cardiovascular = category
        break

# Extend specific category
if cardiovascular:
    requests.post('http://localhost:8000/api/v1/taxonomy/extend', json={
        'taxonomy_data': cardiovascular,
        'project_id': 1,
        'additional_depth': 2,
        'generator': 'gemini'
    })
```

### 3. Taxonomy Management

**Bulk Operations:**
```python
# List all taxonomies
taxonomies = requests.get('http://localhost:8000/api/v1/taxonomy/').json()

# Filter by confidence score
high_confidence = [t for t in taxonomies['taxonomies'] if t['confidence_score'] > 0.8]

# Bulk delete low-confidence taxonomies
low_confidence_ids = [t['id'] for t in taxonomies['taxonomies'] if t['confidence_score'] < 0.6]
if low_confidence_ids:
    requests.delete('http://localhost:8000/api/v1/taxonomy/', json={
        'taxonomy_ids': low_confidence_ids
    })
```

### 4. Error Handling

**API Error Responses:**
```python
try:
    response = requests.post('http://localhost:8000/api/v1/taxonomy/generate', json={...})
    response.raise_for_status()

    taxonomy = response.json()
    print(f"Created taxonomy: {taxonomy['name']}")

except requests.exceptions.HTTPError as e:
    if e.response.status_code == 404:
        print("Project not found")
    elif e.response.status_code == 400:
        error_detail = e.response.json().get('detail', 'Bad request')
        print(f"Validation error: {error_detail}")
    else:
        print(f"API error: {e}")

except requests.exceptions.RequestException as e:
    print(f"Network error: {e}")
```

### 5. Performance Optimization

**Batch Processing:**
```python
# Process multiple taxonomy generations
taxonomies_to_create = [
    {'name': 'Medical', 'domain': 'medical'},
    {'name': 'Legal', 'domain': 'legal'},
    {'name': 'Technical', 'domain': 'technical'}
]

for tax_config in taxonomies_to_create:
    response = requests.post('http://localhost:8000/api/v1/taxonomy/generate', json={
        'project_id': 1,
        'documents': doc_ids,
        'depth': 3,
        'generator': 'gemini',
        **tax_config
    })

    if response.status_code == 200:
        print(f"Created taxonomy: {response.json()['name']}")
    else:
        print(f"Failed to create {tax_config['name']}: {response.text}")
```

### 6. Taxonomy Analytics

**Structure Analysis:**
```python
# Get detailed taxonomy information
tax_response = requests.get('http://localhost:8000/api/v1/taxonomy/101')
tax_data = tax_response.json()

# Analyze structure
taxonomy = tax_data['taxonomy']
analytics = tax_data['analytics']

print(f"Total categories: {analytics['depth_analysis']['total_categories']}")
print(f"Maximum depth: {analytics['depth_analysis']['max_depth']}")
print(f"Confidence score: {tax_data['metadata']['confidence_score']}")

# Traverse taxonomy tree
def analyze_category(category, level=0):
    indent = "  " * level
    print(f"{indent}{category['name']} (confidence: {category.get('confidence_threshold', 0)})")

    for child in category.get('children', []):
        analyze_category(child, level + 1)

analyze_category(taxonomy)
```

This API provides comprehensive taxonomy management capabilities with support for manual creation, AI generation, extension, and full CRUD operations suitable for both development and production workflows.