# Dataset Versioning Module CLI Usage Guide

The Compileo Dataset Versioning CLI provides comprehensive command-line tools for managing dataset versions, enabling tracking of dataset changes, comparisons between versions, and safe rollbacks when needed.

## Commands Overview

```bash
compileo dataset-version [COMMAND] [OPTIONS]
```

Available commands:
- `list-versions` - List all versions of a dataset
- `compare-versions` - Compare two dataset versions
- `rollback` - Rollback dataset to a previous version
- `increment-version` - Increment the version number of a dataset

---

## List Dataset Versions

### `compileo dataset-version list-versions`

List all versions of a specific dataset.

**Usage:**
```bash
compileo dataset-version list-versions --project-id PROJECT_ID --dataset-name DATASET_NAME [--format FORMAT]
```

**Options:**
- `--project-id INTEGER` - Project ID (required)
- `--dataset-name TEXT` - Dataset name (required)
- `--format [table|json]` - Output format (default: table)

**Examples:**

**List versions in table format:**
```bash
compileo dataset-version list-versions --project-id 1 --dataset-name my_dataset
```

**Output:**
```
Versions for dataset 'my_dataset':
--------------------------------------------------------------------------------
Version      Entries  Active   Created              Description
--------------------------------------------------------------------------------
1.0.0        1000     ✓        2024-01-15 10:30    Initial dataset version
1.0.1        1050     ✓        2024-01-16 14:20    Added quality improvements
1.0.2        1100     ✓        2024-01-17 09:15    Fixed minor issues
```

**List versions in JSON format:**
```bash
compileo dataset-version list-versions --project-id 1 --dataset-name my_dataset --format json
```

---

## Compare Dataset Versions

### `compileo dataset-version compare-versions`

Compare two versions of a dataset to see differences.

**Usage:**
```bash
compileo dataset-version compare-versions --project-id PROJECT_ID --dataset-name DATASET_NAME --version1 VERSION1 --version2 VERSION2
```

**Options:**
- `--project-id INTEGER` - Project ID (required)
- `--dataset-name TEXT` - Dataset name (required)
- `--version1 TEXT` - First version to compare (required)
- `--version2 TEXT` - Second version to compare (required)

**Examples:**

**Compare two versions:**
```bash
compileo dataset-version compare-versions --project-id 1 --dataset-name my_dataset --version1 1.0.0 --version2 1.0.1
```

**Output:**
```
Comparison between 1.0.0 and 1.0.1:
============================================================

Version 1.0.0:
  Entries: 1000
  Created: 2024-01-15 10:30:00
  Changes: 0

Version 1.0.1:
  Entries: 1050
  Created: 2024-01-16 14:20:00
  Changes: 50

Comparison:
  Entries difference: +50
  Version relationship: 1.0.1 is newer than 1.0.0
```

---

## Rollback Dataset Version

### `compileo dataset-version rollback`

Rollback a dataset to a previous version.

**Usage:**
```bash
compileo dataset-version rollback --project-id PROJECT_ID --dataset-name DATASET_NAME --target-version VERSION [--confirm]
```

**Options:**
- `--project-id INTEGER` - Project ID (required)
- `--dataset-name TEXT` - Dataset name (required)
- `--target-version TEXT` - Version to rollback to (required)
- `--confirm` - Skip confirmation prompt

**Examples:**

**Rollback with confirmation:**
```bash
compileo dataset-version rollback --project-id 1 --dataset-name my_dataset --target-version 1.0.0
```

**Output:**
```
Are you sure you want to rollback dataset 'my_dataset' to version 1.0.0? This will deactivate all newer versions. [y/N]: y
Successfully rolled back dataset 'my_dataset' to version 1.0.0
```

**Rollback without confirmation:**
```bash
compileo dataset-version rollback --project-id 1 --dataset-name my_dataset --target-version 1.0.0 --confirm
```

---

## Increment Dataset Version

### `compileo dataset-version increment-version`

Increment the version number of a dataset (semantic versioning).

**Usage:**
```bash
compileo dataset-version increment-version --project-id PROJECT_ID --dataset-name DATASET_NAME --version-type TYPE [--description DESCRIPTION]
```

**Options:**
- `--project-id INTEGER` - Project ID (required)
- `--dataset-name TEXT` - Dataset name (required)
- `--version-type [major|minor|patch]` - Type of version increment (default: patch)
- `--description TEXT` - Description of the version change

**Version Types:**
- `major`: Breaking changes (1.0.0 → 2.0.0)
- `minor`: New features (1.0.0 → 1.1.0)
- `patch`: Bug fixes (1.0.0 → 1.0.1)

**Examples:**

**Increment patch version:**
```bash
compileo dataset-version increment-version --project-id 1 --dataset-name my_dataset --version-type patch --description "Fixed minor issues"
```

**Output:**
```
Successfully incremented dataset 'my_dataset' to version 1.0.1
```

**Increment minor version:**
```bash
compileo dataset-version increment-version --project-id 1 --dataset-name my_dataset --version-type minor --description "Added new features"
```

**Increment major version:**
```bash
compileo dataset-version increment-version --project-id 1 --dataset-name my_dataset --version-type major --description "Breaking changes"
```

---

## Best Practices

### 1. Version Management Strategy

**Semantic Versioning:**
```bash
# Use semantic versioning for clear change tracking
# major.minor.patch
# - major: breaking changes
# - minor: new features
# - patch: bug fixes

# Examples:
compileo dataset-version increment-version --project-id 1 --dataset-name my_dataset --version-type major --description "API breaking changes"
compileo dataset-version increment-version --project-id 1 --dataset-name my_dataset --version-type minor --description "Added new classification features"
compileo dataset-version increment-version --project-id 1 --dataset-name my_dataset --version-type patch --description "Fixed null pointer exception"
```

**Version Increment Timing:**
```bash
# Increment versions when:
# - Adding new features or capabilities
# - Fixing bugs that affect output quality
# - Making breaking changes to data format
# - Improving model performance significantly

# Don't increment for:
# - Internal code refactoring
# - Documentation updates
# - Minor performance optimizations
```

### 2. Rollback Safety

**Safe Rollback Process:**
```bash
# 1. Always backup current state before rollback
cp -r datasets/ datasets_backup/

# 2. Check what will be affected
compileo dataset-version compare-versions --project-id 1 --dataset-name my_dataset --version1 current --version2 target_version

# 3. Perform rollback
compileo dataset-version rollback --project-id 1 --dataset-name my_dataset --target-version 1.0.0

# 4. Verify rollback success
compileo dataset-version list-versions --project-id 1 --dataset-name my_dataset
```

**Rollback Recovery:**
```bash
# If rollback fails, restore from backup
cp -r datasets_backup/ datasets/

# Then investigate the failure
# Check logs for error details
tail -f logs/compileo.log
```

### 3. Version Comparison and Analysis

**Regular Version Audits:**
```bash
# Compare recent versions to track progress
compileo dataset-version compare-versions --project-id 1 --dataset-name my_dataset --version1 1.0.0 --version2 1.1.0

# Monitor version growth over time
compileo dataset-version list-versions --project-id 1 --dataset-name my_dataset --format json | jq '.[] | select(.is_active) | .version + ": " + (.total_entries | tostring) + " entries"'
```

**Quality Tracking Across Versions:**
```bash
# Track quality improvements
# Run quality analysis on different versions
# Compare quality scores between versions
# Identify versions with significant quality changes
```

### 4. Integration with Dataset Generation

**Version-Aware Dataset Generation:**
```bash
# Generate dataset with automatic versioning
compileo generate-dataset --project-id 1 --enable-versioning --dataset-name my_dataset

# After generation, check new version
compileo dataset-version list-versions --project-id 1 --dataset-name my_dataset

# Compare with previous version
compileo dataset-version compare-versions --project-id 1 --dataset-name my_dataset --version1 1.0.0 --version2 1.0.1
```

### 5. Backup and Recovery

**Version-Based Backup Strategy:**
```bash
# Create backups before major changes
mkdir -p backups/$(date +%Y%m%d)
cp -r datasets/ backups/$(date +%Y%m%d)/

# Tag backup with version
echo "1.0.0" > backups/$(date +%Y%m%d)/version.txt

# Restore from version-specific backup
VERSION="1.0.0"
BACKUP_DIR=$(find backups/ -name "version.txt" -exec grep -l "$VERSION" {} \; | head -1 | xargs dirname)
cp -r $BACKUP_DIR/datasets/* datasets/
```

### 6. Monitoring and Alerts

**Version Health Monitoring:**
```bash
#!/bin/bash
# version_health_check.sh

DATASET_NAME="my_dataset"
PROJECT_ID="1"

# Check for datasets without recent versions
LAST_VERSION=$(compileo dataset-version list-versions --project-id $PROJECT_ID --dataset-name $DATASET_NAME --format json | jq -r '.[0].version')
DAYS_SINCE_LAST=$(echo $(( ($(date +%s) - $(date -d "$(compileo dataset-version list-versions --project-id $PROJECT_ID --dataset-name $DATASET_NAME --format json | jq -r '.[0].created_at')" +%s)) / 86400 )))

if [ "$DAYS_SINCE_LAST" -gt 30 ]; then
    echo "WARNING: No new versions created in $DAYS_SINCE_LAST days"
fi

# Check version growth
ENTRIES=$(compileo dataset-version list-versions --project-id $PROJECT_ID --dataset-name $DATASET_NAME --format json | jq -r '.[0].total_entries')
if [ "$ENTRIES" -lt 100 ]; then
    echo "WARNING: Latest version has only $ENTRIES entries"
fi
```

## Error Handling

### Common Error Messages

**Dataset Not Found:**
```
Error: No versions found for dataset 'nonexistent_dataset' in project 1
```

**Version Not Found:**
```
Error: Version 1.0.3 not found for dataset 'my_dataset'
```

**Invalid Version Format:**
```
Error: Invalid version format. Expected semantic version (major.minor.patch)
```

**Rollback Failed:**
```
Error: Cannot rollback: target version is corrupted or rollback failed
```

## Integration Examples

### Automated Version Management Script

```bash
#!/bin/bash
# auto_version.sh - Automatically manage dataset versions

PROJECT_ID="$1"
DATASET_NAME="$2"
CHANGE_TYPE="$3"  # major, minor, patch
DESCRIPTION="$4"

if [ -z "$PROJECT_ID" ] || [ -z "$DATASET_NAME" ] || [ -z "$CHANGE_TYPE" ]; then
    echo "Usage: $0 <project_id> <dataset_name> <change_type> [description]"
    exit 1
fi

# Generate dataset (your generation command here)
echo "Generating dataset..."
# compileo generate-dataset --project-id $PROJECT_ID --dataset-name $DATASET_NAME

# Run quality analysis
echo "Running quality analysis..."
QUALITY_SCORE=$(compileo analyze-quality generated_dataset.jsonl --format json | jq -r '.summary.overall_score')

# Determine version increment based on quality
if (( $(echo "$QUALITY_SCORE > 0.9" | bc -l) )); then
    VERSION_TYPE="minor"
    DESC="High quality dataset generated (score: $QUALITY_SCORE)"
elif (( $(echo "$QUALITY_SCORE > 0.7" | bc -l) )); then
    VERSION_TYPE="patch"
    DESC="Dataset generated with acceptable quality (score: $QUALITY_SCORE)"
else
    echo "Quality score too low ($QUALITY_SCORE), skipping version increment"
    exit 1
fi

# Increment version
compileo dataset-version increment-version \
    --project-id $PROJECT_ID \
    --dataset-name $DATASET_NAME \
    --version-type $VERSION_TYPE \
    --description "${DESC}"

echo "Dataset version incremented successfully"
```

### Version Comparison Report

```bash
#!/bin/bash
# version_report.sh - Generate version comparison reports

PROJECT_ID="$1"
DATASET_NAME="$2"

echo "=== Dataset Version Report ==="
echo "Dataset: $DATASET_NAME (Project: $PROJECT_ID)"
echo "Generated: $(date)"
echo

# List all versions
echo "All Versions:"
compileo dataset-version list-versions --project-id $PROJECT_ID --dataset-name $DATASET_NAME
echo

# Compare last two versions
LATEST=$(compileo dataset-version list-versions --project-id $PROJECT_ID --dataset-name $DATASET_NAME --format json | jq -r '.[0].version')
PREVIOUS=$(compileo dataset-version list-versions --project-id $PROJECT_ID --dataset-name $DATASET_NAME --format json | jq -r '.[1].version')

if [ "$PREVIOUS" != "null" ]; then
    echo "Comparison ($LATEST vs $PREVIOUS):"
    compileo dataset-version compare-versions --project-id $PROJECT_ID --dataset-name $DATASET_NAME --version1 $PREVIOUS --version2 $LATEST
fi
```

This CLI provides essential version control capabilities for dataset management, ensuring data integrity and enabling safe experimentation with dataset improvements.