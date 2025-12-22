# Dataset Versioning Module API Usage Guide

The Compileo Dataset Versioning API provides comprehensive version control for datasets, enabling tracking of dataset changes, comparisons between versions, and safe rollbacks when needed.

## Base URL: `/api/v1/datasets/versions`

---

## List Dataset Versions

### GET `/`

Get all versions of a specific dataset.

**Request:**
```bash
curl "http://localhost:8000/api/v1/datasets/versions/?project_id=1&dataset_name=my_dataset"
```

**Query Parameters:**
- `project_id`: Project ID (required)
- `dataset_name`: Dataset name (required)
- `active_only`: Return only active versions (default: true)

**Response:**
```json
{
  "versions": [
    {
      "version": "1.0.0",
      "total_entries": 1000,
      "is_active": true,
      "created_at": "2024-01-15T10:30:00Z",
      "description": "Initial dataset version"
    },
    {
      "version": "1.0.1",
      "total_entries": 1050,
      "is_active": true,
      "created_at": "2024-01-16T14:20:00Z",
      "description": "Added quality improvements"
    }
  ],
  "total": 2
}
```

---

## Compare Dataset Versions

### POST `/compare`

Compare two versions of a dataset to see differences.

**Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/datasets/versions/compare" \
  -H "Content-Type: application/json" \
  -d '{
    "project_id": 1,
    "dataset_name": "my_dataset",
    "version1": "1.0.0",
    "version2": "1.0.1"
  }'
```

**Request Body:**
```json
{
  "project_id": 1,
  "dataset_name": "my_dataset",
  "version1": "1.0.0",
  "version2": "1.0.1"
}
```

**Response:**
```json
{
  "comparison": {
    "entries_added": 50,
    "entries_removed": 0,
    "entries_modified": 25,
    "total_entries_v1": 1000,
    "total_entries_v2": 1050,
    "quality_score_change": 0.05,
    "categories_added": ["cardiology"],
    "categories_removed": [],
    "metadata_changes": {
      "generation_parameters": "updated",
      "model_version": "improved"
    }
  },
  "summary": "Version 1.0.1 added 50 entries and improved quality score by 5%"
}
```

---

## Rollback Dataset Version

### POST `/rollback`

Rollback a dataset to a previous version.

**Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/datasets/versions/rollback" \
  -H "Content-Type: application/json" \
  -d '{
    "project_id": 1,
    "dataset_name": "my_dataset",
    "target_version": "1.0.0"
  }'
```

**Request Body:**
```json
{
  "project_id": 1,
  "dataset_name": "my_dataset",
  "target_version": "1.0.0"
}
```

**Response:**
```json
{
  "message": "Successfully rolled back to version 1.0.0"
}
```

---

## Increment Dataset Version

### POST `/increment`

Increment the version number of a dataset (semantic versioning).

**Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/datasets/versions/increment" \
  -H "Content-Type: application/json" \
  -d '{
    "project_id": 1,
    "dataset_name": "my_dataset",
    "version_type": "patch",
    "description": "Fixed minor issues"
  }'
```

**Request Body:**
```json
{
  "project_id": 1,
  "dataset_name": "my_dataset",
  "version_type": "patch",
  "description": "Fixed minor issues"
}
```

**Version Types:**
- `major`: Breaking changes (1.0.0 → 2.0.0)
- `minor`: New features (1.0.0 → 1.1.0)
- `patch`: Bug fixes (1.0.0 → 1.0.1)

**Response:**
```json
{
  "new_version": "1.0.1",
  "version_id": 123,
  "message": "Version incremented to 1.0.1"
}
```

---

## Get Latest Version

### GET `/latest`

Get the latest version of a dataset.

**Request:**
```bash
curl "http://localhost:8000/api/v1/datasets/versions/latest?project_id=1&dataset_name=my_dataset"
```

**Query Parameters:**
- `project_id`: Project ID (required)
- `dataset_name`: Dataset name (required)

**Response:**
```json
{
  "latest_version": "1.0.1"
}
```

---

## Best Practices

### 1. Version Management Strategy

**Semantic Versioning:**
```python
# Use semantic versioning for clear change tracking
# major.minor.patch
# - major: breaking changes
# - minor: new features
# - patch: bug fixes

version_types = {
    'major': '1.0.0 → 2.0.0',  # Breaking changes
    'minor': '1.0.0 → 1.1.0',  # New features
    'patch': '1.0.0 → 1.0.1'   # Bug fixes
}
```

**Version Increment Timing:**
```python
def should_increment_version(change_type):
    """Determine when to increment versions."""
    if change_type == 'breaking_change':
        return 'major'
    elif change_type == 'new_feature':
        return 'minor'
    elif change_type in ['bug_fix', 'improvement']:
        return 'patch'
    else:
        return 'patch'  # Default
```

### 2. Rollback Safety

**Safe Rollback Process:**
```python
def safe_rollback(project_id, dataset_name, target_version):
    """Safely rollback with validation."""
    # 1. Create backup of current version
    backup_version = create_backup(project_id, dataset_name)

    # 2. Validate target version exists and is healthy
    if not validate_version(project_id, dataset_name, target_version):
        raise ValueError(f"Target version {target_version} is invalid")

    # 3. Perform rollback
    rollback_result = rollback_to_version(project_id, dataset_name, target_version)

    # 4. Validate rollback success
    if not validate_rollback(project_id, dataset_name, target_version):
        # Restore from backup if rollback failed
        restore_backup(project_id, dataset_name, backup_version)
        raise RuntimeError("Rollback failed, restored from backup")

    return rollback_result
```

### 3. Version Comparison and Analysis

**Automated Version Analysis:**
```python
def analyze_version_changes(project_id, dataset_name, version1, version2):
    """Analyze changes between versions."""
    comparison = compare_versions(project_id, dataset_name, version1, version2)

    insights = {
        'growth_rate': calculate_growth_rate(comparison),
        'quality_trend': analyze_quality_trend(comparison),
        'category_evolution': track_category_changes(comparison),
        'recommendations': generate_recommendations(comparison)
    }

    return insights
```

### 4. Integration with Dataset Generation

**Version-Aware Dataset Generation:**
```python
def generate_dataset_with_versioning(project_id, generation_params):
    """Generate dataset with automatic versioning."""
    # Generate new dataset
    dataset = generate_dataset(project_id, generation_params)

    # Analyze changes from previous version
    if has_previous_version(project_id, generation_params['dataset_name']):
        comparison = compare_with_previous_version(project_id, generation_params['dataset_name'])

        # Determine version increment type
        version_type = determine_version_increment(comparison)

        # Increment version
        new_version = increment_version(project_id, generation_params['dataset_name'], version_type)

        # Associate dataset with version
        associate_dataset_with_version(dataset, new_version)

    return dataset
```

### 5. Monitoring and Alerts

**Version Health Monitoring:**
```python
def monitor_version_health():
    """Monitor version health and send alerts."""
    issues = []

    # Check for datasets without versions
    unversioned = find_unversioned_datasets()
    if unversioned:
        issues.append(f"Unversioned datasets: {unversioned}")

    # Check for rapid version changes (potential issues)
    rapid_changes = detect_rapid_version_changes()
    if rapid_changes:
        issues.append(f"Rapid version changes detected: {rapid_changes}")

    # Check for large version gaps
    gaps = detect_version_gaps()
    if gaps:
        issues.append(f"Version gaps detected: {gaps}")

    if issues:
        send_alert("Version Health Issues", issues)

    return issues
```

### 6. Backup and Recovery

**Version-Based Backup Strategy:**
```python
def create_version_backup(project_id, dataset_name, version):
    """Create backup of specific version."""
    # Export version data
    version_data = export_version(project_id, dataset_name, version)

    # Create backup record
    backup_id = create_backup_record(version_data, version)

    # Store in backup location
    store_backup(backup_id, version_data)

    return backup_id
```

## Error Handling

### Common Error Responses

**Version Not Found:**
```json
{
  "detail": "Version 1.0.2 not found for dataset 'my_dataset'"
}
```

**Invalid Version Format:**
```json
{
  "detail": "Invalid version format. Expected semantic version (major.minor.patch)"
}
```

**Rollback Failed:**
```json
{
  "detail": "Cannot rollback: target version is corrupted"
}
```

## Integration Examples

### Python Client

```python
import requests

class DatasetVersioningClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url

    def list_versions(self, project_id, dataset_name):
        response = requests.get(
            f"{self.base_url}/api/v1/datasets/versions/",
            params={"project_id": project_id, "dataset_name": dataset_name}
        )
        return response.json()

    def compare_versions(self, project_id, dataset_name, v1, v2):
        response = requests.post(
            f"{self.base_url}/api/v1/datasets/versions/compare",
            json={
                "project_id": project_id,
                "dataset_name": dataset_name,
                "version1": v1,
                "version2": v2
            }
        )
        return response.json()

    def rollback(self, project_id, dataset_name, target_version):
        response = requests.post(
            f"{self.base_url}/api/v1/datasets/versions/rollback",
            json={
                "project_id": project_id,
                "dataset_name": dataset_name,
                "target_version": target_version
            }
        )
        return response.json()
```

### CLI Integration

```bash
# List versions
compileo dataset-version list-versions --project-id 1 --dataset-name my_dataset

# Compare versions
compileo dataset-version compare-versions --project-id 1 --dataset-name my_dataset --version1 1.0.0 --version2 1.0.1

# Rollback
compileo dataset-version rollback --project-id 1 --dataset-name my_dataset --target-version 1.0.0

# Increment version
compileo dataset-version increment-version --project-id 1 --dataset-name my_dataset --version-type patch --description "Bug fixes"
```

This API provides essential version control capabilities for dataset management, ensuring data integrity and enabling safe experimentation with dataset improvements.