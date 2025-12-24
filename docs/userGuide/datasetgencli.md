# Dataset Generation in Compileo CLI

## Overview

The Compileo Command Line Interface provides comprehensive dataset generation capabilities with full configuration control. Generate high-quality datasets from processed document chunks using various AI models, taxonomy integration, and quality assurance features.

## Key Features

- **Multi-Model Support**: Choose from Gemini, Grok, or Ollama models
- **Taxonomy Integration**: Align datasets with existing project taxonomies
- **Quality Assurance**: Built-in quality analysis with configurable thresholds
- **Performance Benchmarking**: Optional AI model benchmarking after generation
- **Flexible Output**: JSONL and Parquet format support
- **Versioning Support**: Dataset versioning for iterative improvement
- **Asynchronous Processing**: Background job execution with progress monitoring

## CLI Command

### Generate Dataset

```bash
compileo generate-dataset [OPTIONS]
```

**Core Parameters:**
- `--project-id INTEGER` (required): Project ID containing processed chunks
- `--data-source [Chunks Only|Taxonomy|Extract]`: Data source for generation (default: "Chunks Only")
- `--prompt-name TEXT`: Prompt template name (default: "default")
- `--format-type [jsonl|parquet]`: Output format (default: "jsonl")
- `--concurrency INTEGER`: Parallel processing threads (default: 1)
- `--batch-size INTEGER`: Number of chunks to process per batch (default: 50, 0 = all at once)
- `--output-dir PATH`: Output directory (default: ".")

**Model Selection:**
- `--parsing-model [gemini|grok|ollama]`: AI model for document parsing (default: "gemini")
- `--chunking-model [gemini|grok|ollama]`: AI model for text chunking (default: "gemini")
- `--classification-model [gemini|grok|ollama]`: AI model for content classification (default: "gemini")

**Taxonomy Integration:**
- `--taxonomy-project TEXT`: Project name containing taxonomy
- `--taxonomy-name TEXT`: Taxonomy name to align with
- `--extraction-file-id TEXT`: Specific extraction job ID (UUID) when using "Extract" data source
- `--selected-categories TEXT`: Comma-separated list of category names to filter by (only for Extract mode)

**Quality Assurance:**
- `--analyze-quality`: Enable quality analysis
- `--quality-threshold FLOAT`: Quality threshold (0-1, default: 0.7)
- `--quality-config PATH`: Path to quality configuration JSON

**Benchmarking:**
- `--run-benchmarks`: Run AI model benchmarks after generation
- `--benchmark-suite TEXT`: Benchmark suite (default: "glue")
- `--benchmark-config PATH`: Path to benchmarking configuration JSON

**Advanced Options:**
- `--include-evaluation-sets`: Generate train/validation/test splits
- `--enable-versioning`: Enable dataset versioning
- `--dataset-name TEXT`: Name for versioned datasets
- `--category-limits TEXT`: Comma-separated category limits per taxonomy level
- `--specificity-level INTEGER`: Taxonomy specificity level (1-5, default: 1)
- `--custom-audience TEXT`: Target audience description
- `--custom-purpose TEXT`: Dataset purpose description
- `--complexity-level [auto|basic|intermediate|advanced]`: Content complexity (default: "intermediate")
- `--domain TEXT`: Knowledge domain (default: "general")
- `--datasets-per-chunk INTEGER`: Maximum datasets per chunk (default: 3)

## Data Source Options

### Chunks Only Mode
```bash
# Basic generation from raw chunks
compileo generate-dataset --project-id 1 --data-source "Chunks Only"
```

### Taxonomy Mode
```bash
# Taxonomy-enhanced generation (no extraction required)
compileo generate-dataset --project-id 1 --data-source "Taxonomy" --taxonomy-project medical --taxonomy-name icd_10
```

### Extract Mode
```bash
# Entity-focused generation from extracted concepts
compileo generate-dataset --project-id 1 --data-source "Extract"

# Use specific extraction job for targeted dataset generation
compileo generate-dataset --project-id 1 --data-source "Extract" --extraction-file-id 123

# Filter by specific categories
compileo generate-dataset --project-id 1 --data-source "Extract" --selected-categories "Medical Conditions,Treatments"
```

## Usage Examples

### Basic Dataset Generation

```bash
# Generate dataset from project 1
compileo generate-dataset --project-id 1

# Specify output directory and format
compileo generate-dataset --project-id 1 --output-dir ./datasets --format-type jsonl

# Use batch processing for memory efficiency
compileo generate-dataset --project-id 1 --batch-size 25 --concurrency 2
```

### Advanced Configuration

```bash
# Full configuration with quality analysis
compileo generate-dataset \
  --project-id 1 \
  --parsing-model gemini \
  --chunking-model grok \
  --classification-model gemini \
  --taxonomy-project medical_taxonomy \
  --taxonomy-name icd_10 \
  --analyze-quality \
  --quality-threshold 0.8 \
  --run-benchmarks \
  --benchmark-suite glue \
  --concurrency 3 \
  --output-dir ./medical_datasets
```

### Evaluation Dataset Creation

```bash
# Generate evaluation-ready datasets with splits
compileo generate-dataset \
  --project-id 1 \
  --include-evaluation-sets \
  --enable-versioning \
  --dataset-name medical_qa_v1 \
  --analyze-quality \
  --quality-threshold 0.85
```

### Custom Taxonomy Integration

```bash
# Use custom taxonomy with specific parameters
compileo generate-dataset \
  --project-id 1 \
  --taxonomy-project cardiology \
  --taxonomy-name heart_conditions \
  --specificity-level 3 \
  --category-limits 10,15,20 \
  --custom-audience "medical residents" \
  --custom-purpose "board exam preparation"
```

## Model Selection Guidelines

### When to Use Each Model

**Gemini:**
- Best for complex document understanding
- Superior taxonomy alignment
- Good for multi-format document processing
- Recommended for production datasets

**Grok:**
- Excellent for technical and medical content
- Better nuance detection in specialized domains
- Good for complex categorization tasks
- Cost-effective alternative to Gemini

**Ollama:**
- Local processing (no API keys required)
- Best for development and testing
- Limited by local hardware capabilities
- Good for high-volume processing on local infrastructure

### Model Combinations

```bash
# Production setup - Gemini for all tasks
compileo generate-dataset --project-id 1 \
  --parsing-model gemini \
  --chunking-model gemini \
  --classification-model gemini

# Balanced approach - Mix models for cost optimization
compileo generate-dataset --project-id 1 \
  --parsing-model gemini \
  --chunking-model grok \
  --classification-model gemini

# Local development - Ollama for all tasks
compileo generate-dataset --project-id 1 \
  --parsing-model ollama \
  --chunking-model ollama \
  --classification-model ollama
```

## Quality Assurance Integration

### Automatic Quality Analysis

```bash
# Enable quality analysis with default settings
compileo generate-dataset --project-id 1 --analyze-quality

# Custom quality threshold
compileo generate-dataset --project-id 1 \
  --analyze-quality \
  --quality-threshold 0.9

# Use custom quality configuration
compileo generate-dataset --project-id 1 \
  --analyze-quality \
  --quality-config ./quality_config.json
```

### Quality Configuration File

Create a `quality_config.json`:

```json
{
  "enabled": true,
  "diversity": {
    "enabled": true,
    "threshold": 0.6,
    "min_lexical_diversity": 0.3
  },
  "bias": {
    "enabled": true,
    "threshold": 0.3
  },
  "difficulty": {
    "enabled": true,
    "threshold": 0.7,
    "target_difficulty": "intermediate"
  },
  "consistency": {
    "enabled": true,
    "threshold": 0.8
  },
  "output_format": "json",
  "fail_on_any_failure": false
}
```

## Benchmarking Integration

### Running Benchmarks

```bash
# Run GLUE benchmarks after generation
compileo generate-dataset --project-id 1 --run-benchmarks

# Specify benchmark suite
compileo generate-dataset --project-id 1 \
  --run-benchmarks \
  --benchmark-suite superglue

# Use custom benchmark configuration
compileo generate-dataset --project-id 1 \
  --run-benchmarks \
  --benchmark-config ./benchmark_config.json
```

### Benchmark Configuration File

Create a `benchmark_config.json`:

```json
{
  "enabled": true,
  "benchmark": {
    "suites": ["glue", "mmlu"],
    "model_path": "/path/to/model",
    "batch_size": 32
  },
  "metrics": {
    "enabled_metrics": ["accuracy", "f1", "bleu"],
    "custom_metrics": []
  },
  "tracking": {
    "enabled": true,
    "storage_path": "benchmark_results"
  }
}
```

## Dataset Versioning

### Version Control for Datasets

```bash
# Enable versioning with custom name
compileo generate-dataset --project-id 1 \
  --enable-versioning \
  --dataset-name medical_qa_dataset

# Version types: major, minor, patch
compileo dataset-version increment-version \
  --project-id 1 \
  --dataset-name medical_qa_dataset \
  --version-type minor \
  --description "Added cardiology questions"
```

### Version Management

```bash
# List dataset versions
compileo dataset-version list-versions \
  --project-id 1 \
  --dataset-name medical_qa_dataset

# Compare versions
compileo dataset-version compare-versions \
  --project-id 1 \
  --dataset-name medical_qa_dataset \
  --version1 1.0.0 \
  --version2 1.1.0

# Rollback to previous version
compileo dataset-version rollback \
  --project-id 1 \
  --dataset-name medical_qa_dataset \
  --target-version 1.0.0
```

## Job Monitoring and Management

### Monitoring Dataset Generation

Dataset generation jobs are now stored in the database and persist across CLI sessions and server restarts. Use the dedicated dataset status commands:

```bash
# Start dataset generation
compileo generate-dataset --project-id 1 --analyze-quality
# Output: Job submitted with ID: dataset-gen-uuid-123

# Monitor job status (jobs persist across sessions)
compileo generate-dataset --status dataset-gen-uuid-123

# Poll for completion with timeout
compileo generate-dataset --status dataset-gen-uuid-123 --poll --timeout 300

# Get general job statistics
compileo jobs stats
```

### Managing Jobs

```bash
# Cancel a running job
compileo jobs cancel dataset-gen-uuid-123 --confirm

# Restart a failed job
compileo jobs restart dataset-gen-uuid-123 --confirm

# Start worker process (required for job processing)
compileo jobs worker --redis-url redis://localhost:6379/0
```

**Note:** Dataset generation jobs use database persistence instead of the traditional job queue system, ensuring reliability across server restarts.

## Output and Results

### Generated Files

Dataset generation creates per-batch output files:

```
output_dir/
├── dataset_[job_id]_batch_0.jsonl     # First batch results
├── dataset_[job_id]_batch_1.jsonl     # Second batch results
├── dataset_[job_id]_batch_N.jsonl     # Nth batch results
├── dataset_[job_id]_quality.json      # Quality analysis report (if enabled)
├── benchmark_results_[job_id]/        # Benchmarking results (if enabled)
│   ├── glue_results.json
│   └── performance_metrics.json
└── dataset_[job_id]_metadata.json     # Generation metadata
```

**Batch File Naming:**
- Files are created as each batch completes
- Format: `dataset_[job_id]_batch_[batch_index].[format_type]`
- Number of files equals number of batches (controlled by `--batch-size`)
- Each file contains only the results from that specific batch

### Dataset Format

**JSONL Format:**
```json
{"question": "What are the symptoms of myocardial infarction?", "answer": "Chest pain, shortness of breath, diaphoresis...", "category": "cardiology", "quality_score": 0.92, "difficulty": "intermediate"}
{"question": "How is hypertension diagnosed?", "answer": "Blood pressure measurement above 130/80 mmHg...", "category": "cardiology", "quality_score": 0.88, "difficulty": "basic"}
```

### Quality Report

```json
{
  "summary": {
    "overall_score": 0.85,
    "total_entries": 1500,
    "passed_threshold": 1275,
    "failed_entries": 225
  },
  "diversity": {
    "lexical_diversity": 0.78,
    "semantic_diversity": 0.82,
    "topic_balance": 0.79
  },
  "bias": {
    "demographic_bias": 0.15,
    "content_bias": 0.12
  },
  "difficulty": {
    "average_readability": 0.72,
    "complexity_distribution": {
      "basic": 450,
      "intermediate": 900,
      "advanced": 150
    }
  }
}
```

## Error Handling

### Common Issues and Solutions

**"No chunks found for project"**
- Ensure documents have been uploaded and processed
- Check that chunking has completed successfully

**"API key not configured"**
- Set required API keys in environment variables
- Use Ollama for local processing without API keys

**"Quality threshold not met"**
- Lower quality threshold or improve source content
- Review quality configuration settings

**"Redis connection failed"**
- Ensure Redis server is running
- Check REDIS_URL environment variable

## Best Practices

### Performance Optimization

1. **Concurrency**: Use appropriate concurrency based on available resources
2. **Model Selection**: Balance cost and quality requirements
3. **Chunk Preparation**: Ensure high-quality chunking before generation

### Quality Assurance

1. **Enable Quality Analysis**: Always use `--analyze-quality` for production datasets
2. **Set Appropriate Thresholds**: Balance quality with generation volume
3. **Review Quality Reports**: Use quality metrics to guide dataset improvement

### Resource Management

1. **Monitor Job Queue**: Use `compileo jobs stats` to track system load
2. **Worker Management**: Ensure sufficient worker processes are running
3. **Version Control**: Use versioning for iterative dataset improvement

### Integration Patterns

```bash
# Complete pipeline: upload → process → generate → analyze
compileo documents upload --project-id 1 --file-paths document1.pdf document2.pdf
compileo documents process --project-id 1 --document-ids 1,2 --parser gemini
compileo generate-dataset --project-id 1 --analyze-quality --run-benchmarks
compileo analyze-quality dataset_*.jsonl --format markdown --output quality_report.md
```

This CLI provides comprehensive control over dataset generation with enterprise-grade features for quality assurance, benchmarking, and version management.