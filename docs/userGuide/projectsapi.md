# Projects Module API Usage Guide

The Compileo Projects API provides comprehensive REST endpoints for project management, including creation, retrieval, updating, and deletion of projects with associated document and dataset tracking.

## Base URL: `/api/v1/projects`

---

## Project Listing

### GET `/`

List all projects with pagination support.

**Request:**
```bash
curl "http://localhost:8000/api/v1/projects?page=1&per_page=20"
```

**Query Parameters:**
- `page`: Page number (default: 1, minimum: 1)
- `per_page`: Items per page (default: 20, range: 1-1000)

**Response:**
```json
{
  "projects": [
    {
      "id": 1,
      "name": "Medical Research Project",
      "description": "Analysis of medical documents",
      "created_at": "2024-01-15T10:30:00Z",
      "updated_at": null,
      "document_count": 15,
      "dataset_count": 3,
      "status": "active"
    },
    {
      "id": 2,
      "name": "Legal Document Analysis",
      "description": "Contract analysis and classification",
      "created_at": "2024-01-20T14:45:00Z",
      "updated_at": null,
      "document_count": 8,
      "dataset_count": 1,
      "status": "active"
    }
  ],
  "total": 2,
  "page": 1,
  "per_page": 20
}
```

---

## Project Creation

### POST `/`

Create a new project.

**Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/projects/" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "New Research Project",
    "description": "A project for analyzing research documents"
  }'
```

**Request Body:**
- `name`: Project name (required, unique)
- `description`: Project description (optional)

**Response:**
```json
{
  "id": 3,
  "name": "New Research Project",
  "description": "A project for analyzing research documents",
  "created_at": "2024-01-25T09:15:00Z",
  "updated_at": null,
  "document_count": 0,
  "dataset_count": 0,
  "status": "active"
}
```

---

## Project Retrieval

### GET `/{project_id}`

Get detailed information about a specific project.

**Request:**
```bash
curl "http://localhost:8000/api/v1/projects/1"
```

**Response:**
```json
{
  "id": 1,
  "name": "Medical Research Project",
  "description": "Analysis of medical documents",
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": null,
  "document_count": 15,
  "dataset_count": 3,
  "status": "active"
}
```

### GET `/{project_id}/documents`

Get all documents associated with a project.

**Request:**
```bash
curl "http://localhost:8000/api/v1/projects/1/documents"
```

**Response:**
```json
{
  "documents": [
    {
      "id": 101,
      "project_id": 1,
      "file_name": "medical_report.pdf",
      "source_file_path": "storage/uploads/1/medical_report.pdf",
      "parsed_file_path": "storage/parsed/1/medical_report.md",
      "parsed_files_manifest": null,
      "created_at": "2024-01-15T11:00:00Z",
      "status": "parsed"
    }
  ]
}
```

### GET `/{project_id}/datasets`

Get all datasets associated with a project.

**Request:**
```bash
curl "http://localhost:8000/api/v1/projects/1/datasets"
```

**Response:**
```json
{
  "datasets": [
    {
      "id": 201,
      "project_id": 1,
      "output_type": "dataset",
      "output_file_path": "storage/datasets/1/dataset_v1.0.0.json",
      "created_at": "2024-01-16T14:30:00Z"
    }
  ]
}
```

---

## Project Updates

### PUT `/{project_id}`

Update project information.

**Request:**
```bash
curl -X PUT "http://localhost:8000/api/v1/projects/1" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Updated Medical Research Project",
    "description": "Updated analysis of medical documents with new focus"
  }'
```

**Request Body:**
- `name`: New project name (optional, must be unique if changed)
- `description`: New project description (optional)

**Response:**
```json
{
  "id": 1,
  "name": "Updated Medical Research Project",
  "description": "Updated analysis of medical documents with new focus",
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-25T10:15:00Z",
  "document_count": 15,
  "dataset_count": 3,
  "status": "active"
}
```

---

## Project Deletion

### DELETE `/{project_id}`

Delete a project and all associated data with comprehensive cascading cleanup.

**Warning:** This operation permanently deletes the project and performs complete cascading deletion across the entire data pipeline. All associated records and files are removed in the correct order to maintain data integrity.

**What Gets Deleted:**
- **Database Records** (in reverse pipeline order):
  - Dataset records (versions, parameters, jobs, lineage, changes)
  - Extraction records (results, jobs)
  - Taxonomy records and associated files
  - Document chunks
  - Parsed document records and files
  - Document records
  - Project record
- **Filesystem Files**:
  - All uploaded document files
  - Parsed markdown files and manifests
  - Chunk files and split PDFs
  - Taxonomy JSON files
  - Extraction result files
  - Dataset files
- **Project Directories**:
  - `storage/uploads/{project_id}/`
  - `storage/parsed/{project_id}/`
  - `storage/chunks/{project_id}/`
  - `storage/taxonomy/{project_id}/`
  - `storage/extracted/{project_id}/`
  - `storage/datasets/{project_id}/`

**Request:**
```bash
curl -X DELETE "http://localhost:8000/api/v1/projects/1"
```

**Response:**
```json
{
  "message": "Project 1 and all associated data deleted successfully"
}
```

**Error Handling:**
- If any deletion step fails, the operation continues with remaining items
- Warning logs are generated for individual failures
- The operation completes successfully as long as the project record is deleted
- Database transactions ensure consistency where possible

### DELETE `/`

Bulk delete multiple projects.

**Request:**
```bash
curl -X DELETE "http://localhost:8000/api/v1/projects/" \
  -H "Content-Type: application/json" \
  -d '{
    "project_ids": [1, 2, 3]
  }'
```

**Request Body:**
- `project_ids`: Array of project IDs to delete (required)

**Response:**
```json
{
  "message": "Successfully deleted 3 projects and all associated data",
  "deleted": [1, 2, 3]
}
```

---

## Best Practices

### 1. Project Organization

**Naming Conventions:**
```python
import requests

# Use descriptive, unique project names
projects_to_create = [
    {"name": "Q4_2024_Medical_Research", "description": "Medical document analysis for Q4"},
    {"name": "Legal_Contract_Analysis", "description": "Automated contract review system"},
    {"name": "Technical_Documentation_AI", "description": "AI-powered technical documentation processing"}
]

for project in projects_to_create:
    response = requests.post('http://localhost:8000/api/v1/projects/', json=project)
    if response.status_code == 200:
        print(f"Created project: {response.json()['name']}")
```

**Project Structure Planning:**
```python
# Plan project structure based on use case
def create_project_structure(domain, year, quarter):
    base_name = f"{domain}_{year}_Q{quarter}"
    projects = [
        {"name": f"{base_name}_Raw_Data", "description": "Initial data ingestion and parsing"},
        {"name": f"{base_name}_Processed", "description": "Cleaned and processed datasets"},
        {"name": f"{base_name}_Analysis", "description": "Final analysis and reporting"}
    ]
    return projects
```

### 2. Project Lifecycle Management

**Regular Cleanup:**
```python
# Archive old projects instead of deleting
def archive_old_projects(days_old=365):
    # Get all projects
    response = requests.get('http://localhost:8000/api/v1/projects/')
    projects = response.json()['projects']

    # Find projects older than threshold
    import datetime
    cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days_old)

    old_projects = []
    for project in projects:
        created_date = datetime.datetime.fromisoformat(project['created_at'].replace('Z', '+00:00'))
        if created_date < cutoff_date:
            old_projects.append(project)

    print(f"Found {len(old_projects)} projects older than {days_old} days")
    return old_projects

# Use with caution - deletion is permanent
# archive_projects = archive_old_projects()
```

**Project Metrics Tracking:**
```python
def get_project_metrics(project_id):
    # Get project details
    project_response = requests.get(f'http://localhost:8000/api/v1/projects/{project_id}')
    project = project_response.json()

    # Get associated data
    docs_response = requests.get(f'http://localhost:8000/api/v1/projects/{project_id}/documents')
    datasets_response = requests.get(f'http://localhost:8000/api/v1/projects/{project_id}/datasets')

    metrics = {
        "project_name": project["name"],
        "document_count": project["document_count"],
        "dataset_count": project["dataset_count"],
        "documents": docs_response.json()["documents"],
        "datasets": datasets_response.json()["datasets"]
    }

    return metrics
```

### 3. Bulk Operations

**Batch Project Creation:**
```python
def create_projects_batch(project_list):
    created_projects = []
    failed_projects = []

    for project_data in project_list:
        try:
            response = requests.post('http://localhost:8000/api/v1/projects/', json=project_data)
            response.raise_for_status()
            created_projects.append(response.json())
        except requests.exceptions.HTTPError as e:
            failed_projects.append({
                "project": project_data,
                "error": e.response.json().get('detail', str(e))
            })
        except Exception as e:
            failed_projects.append({
                "project": project_data,
                "error": str(e)
            })

    return {
        "created": created_projects,
        "failed": failed_projects,
        "success_rate": len(created_projects) / len(project_list) if project_list else 0
    }
```

**Bulk Project Deletion:**
```python
def cleanup_test_projects(name_pattern="test_"):
    # Get all projects
    response = requests.get('http://localhost:8000/api/v1/projects/')
    all_projects = response.json()['projects']

    # Filter test projects
    test_projects = [p for p in all_projects if name_pattern in p['name'].lower()]
    test_project_ids = [p['id'] for p in test_projects]

    if not test_project_ids:
        print("No test projects found")
        return

    print(f"Found {len(test_project_ids)} test projects to delete")

    # Bulk delete
    delete_response = requests.delete('http://localhost:8000/api/v1/projects/', json={
        "project_ids": test_project_ids
    })

    result = delete_response.json()
    print(f"Deleted: {len(result.get('deleted', []))}")
    if result.get('failed'):
        print(f"Failed: {len(result['failed'])}")
```

### 4. Error Handling

**Robust API Usage:**
```python
def safe_project_operations():
    try:
        # Create project
        create_response = requests.post('http://localhost:8000/api/v1/projects/', json={
            "name": "Test Project",
            "description": "Testing project operations"
        })

        if create_response.status_code == 400:
            error_detail = create_response.json().get('detail', '')
            if 'already exists' in error_detail:
                print("Project name already taken, choosing alternative...")
                # Handle duplicate name
            else:
                print(f"Validation error: {error_detail}")
        elif create_response.status_code == 200:
            project = create_response.json()
            project_id = project['id']
            print(f"Created project: {project['name']}")

            # Get project details
            get_response = requests.get(f'http://localhost:8000/api/v1/projects/{project_id}')
            if get_response.status_code == 200:
                details = get_response.json()
                print(f"Project has {details['document_count']} documents")

            # Clean up - delete project
            delete_response = requests.delete(f'http://localhost:8000/api/v1/projects/{project_id}')
            if delete_response.status_code == 200:
                print("Project deleted successfully")
        else:
            print(f"Unexpected status code: {create_response.status_code}")

    except requests.exceptions.ConnectionError:
        print("Cannot connect to API server")
    except requests.exceptions.Timeout:
        print("Request timed out")
    except Exception as e:
        print(f"Unexpected error: {e}")
```

### 5. Project Templates

**Project Template System:**
```python
PROJECT_TEMPLATES = {
    "medical_research": {
        "description": "Medical document analysis and research project",
        "default_documents": ["medical_reports", "clinical_trials", "research_papers"],
        "suggested_taxonomies": ["medical_conditions", "treatments", "diagnostics"]
    },
    "legal_analysis": {
        "description": "Legal document review and contract analysis",
        "default_documents": ["contracts", "agreements", "legal_opinions"],
        "suggested_taxonomies": ["contract_types", "legal_entities", "obligations"]
    },
    "technical_docs": {
        "description": "Technical documentation processing and analysis",
        "default_documents": ["api_docs", "user_manuals", "specifications"],
        "suggested_taxonomies": ["components", "features", "functionality"]
    }
}

def create_project_from_template(template_name, project_name):
    if template_name not in PROJECT_TEMPLATES:
        raise ValueError(f"Unknown template: {template_name}")

    template = PROJECT_TEMPLATES[template_name]

    project_data = {
        "name": project_name,
        "description": template["description"]
    }

    response = requests.post('http://localhost:8000/api/v1/projects/', json=project_data)

    if response.status_code == 200:
        project = response.json()
        print(f"Created project '{project_name}' from {template_name} template")
        print(f"Suggested document types: {', '.join(template['default_documents'])}")
        print(f"Suggested taxonomies: {', '.join(template['suggested_taxonomies'])}")
        return project
    else:
        raise Exception(f"Failed to create project: {response.text}")
```

This API provides comprehensive project management capabilities essential for organizing document processing workflows, tracking progress, and maintaining data organization within the Compileo system.