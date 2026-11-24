# Compileo

A modular and scalable pipeline for dataset creation and curation. This system processes documents in various formats, extracts and structures information, and generates high-quality datasets for tasks like clinical diagnosis matching.

## Features

### Core Modules

- **Document Ingestion**: Support for multiple document formats (PDF, DOCX, TXT, MD, etc.)
- **Text Chunking**: Advanced text splitting with semantic, token, and schema-based methods
- **Information Extraction**: Category-based extraction with Named Entity Recognition
- **Dataset Generation**: Configurable question-answer pair generation
- **Vector Storage**: Integration with Qdrant for embedding storage and indexing

### Optional Quality Assurance Module

The `datasetqual` module provides enterprise-grade quality assurance for dataset generation:

- **Automated Dataset Quality Metrics**: Comprehensive scoring across multiple dimensions
- **Bias Detection and Diversity Analysis**: Advanced algorithms for detecting and mitigating dataset biases
- **Difficulty Assessment**: Readability and complexity evaluation
- **Consistency Validation**: Answer coherence and logical consistency checking
- **Integration with Dataset Generation Pipeline**: Seamless quality checks during dataset creation

### Optional Performance Benchmarking Module

The `benchmarking` module provides enterprise-grade AI model performance evaluation:

- **Benchmark Suite Integration**: GLUE, SuperGLUE, MMLU, and medical domain benchmarks
- **Performance Metrics**: Accuracy, F1, BLEU, ROUGE, and custom evaluation metrics
- **Model Performance Tracking**: Historical performance tracking and comparison across model versions
- **Quality-Performance Correlation**: Analysis of relationships between dataset quality and model performance
- **Integration with Dataset Generation Pipeline**: Optional benchmarking after dataset creation

### Web GUI Interface

The `gui` module provides a complete Streamlit-based web interface for the Compileo pipeline:

- **Enhanced Dataset Creation Wizard**: 6-step guided workflow with AI model selection and auto-save
  1. Project Selection (existing or new project creation)
  2. Smart Document Upload (multi-format support with existing document detection)
  3. AI Model Configuration (parsing, chunking, classification model selection)
  4. Taxonomy Selection (from available project taxonomies with existing document support)
  5. Generation Parameters (purpose, audience, output format, concurrency=5 default)
  6. Real Dataset Generation (actual API integration with progress monitoring)

- **Multi-page Application**: Dedicated pages for different functionalities
  - **Home/Dashboard**: Overview and quick actions
  - **Projects**: Project management and document organization
  - **Wizard**: Dataset creation workflow
  - **Taxonomy**: Taxonomy browsing and hierarchical visualization
  - **Quality**: Quality metrics dashboard with interactive charts
  - **Benchmarking**: Performance benchmarking visualization
  - **Settings**: Configuration and API key management with Ollama parameter tuning

- **Ollama Parameter Configuration**: Fine-tune AI model behavior across different processing roles
  - **Role-Specific Settings**: Separate parameters for parsing, chunking, taxonomy, classification, and generation
  - **7 Configurable Parameters**: temperature, repeat_penalty, top_p, top_k, num_predict, seed, num_ctx
  - **Compact Grid Layout**: 4-column by 2-row parameter arrangement for space efficiency
  - **Real-time Application**: Parameters automatically applied to subsequent processing jobs

- **API Integration**: Full REST API integration with error handling and real-time status updates
- **Responsive Design**: Mobile-friendly interface with adaptive layouts
- **State Management**: Persistent session state and wizard progress tracking

## Prerequisites

Before installing Compileo, ensure you have the following dependencies:

- **Python 3.8+**
- **Redis Server** (mandatory for job queuing)
- **SQLite** (for local data storage)
- **Qdrant** (for vector storage, optional but recommended)

### GPU Support

**HuggingFace model processing and GPU resource monitoring are supported only for NVIDIA GPUs** in Docker environments. The system includes PyTorch with CUDA support for optimal performance with compatible hardware.

For CPU-only deployments or systems without NVIDIA GPUs, the system will automatically fall back to CPU processing, though with reduced performance for GPU-accelerated tasks.

### Installing Redis

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install redis-server
sudo systemctl start redis-server
sudo systemctl enable redis-server
```

**macOS (with Homebrew):**
```bash
brew install redis
brew services start redis
```

**Docker:**
```bash
docker run -d -p 6379:6379 --name redis redis:7-alpine
```

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

### Redis Configuration

Compileo requires Redis for all job processing operations. Configure your Redis connection:

```bash
# Set Redis URL (required)
export REDIS_URL=redis://localhost:6379/0

# For Docker deployments
export REDIS_URL=redis://redis:6379/0
```

### Environment Variables

Create a `.env` file with your configuration:

```bash
# Redis (required)
REDIS_URL=redis://localhost:6379/0

# JWT authentication is used for multi-tenant isolation.

# Database
DATABASE_URL=sqlite:///database.db

# Vector Storage
QDRANT_URL=http://localhost:6333

# AI Model API Keys
GEMINI_API_KEY=your-gemini-api-key
GROK_API_KEY=your-grok-api-key
```

## Quick Start

### Basic Dataset Generation

```bash
# Generate a dataset
compileo generate-dataset --project-id 1 --output-dir ./output

# Include quality analysis
compileo generate-dataset --project-id 1 --analyze-quality --quality-threshold 0.8

# Include AI model benchmarking
compileo generate-dataset --project-id 1 --run-benchmarks --benchmark-suite glue
```

### Quality Analysis

```bash
# Analyze existing dataset
compileo analyze-quality dataset.jsonl --format markdown --output report.md

# Custom quality configuration
compileo analyze-quality dataset.jsonl --config quality_config.json
```

### GUI Interface

The web GUI provides an intuitive interface for dataset creation and management:

#### Starting the GUI
```bash
# Start the web interface (requires API server running)
streamlit run src/compileo/features/gui/main.py

# Or using Docker Compose (recommended for development)
docker-compose -f docker-compose.yml up gui
```

#### GUI Features
- **Dataset Creation Wizard**: Step-by-step guided dataset generation
- **Project Management**: Create and manage projects through the web interface
- **Document Upload**: Drag-and-drop file upload with progress tracking
- **Taxonomy Browser**: Interactive visualization of hierarchical taxonomies
- **Quality Dashboard**: Real-time quality metrics and analysis
- **Benchmarking Visualization**: Performance charts and model comparisons

#### GUI Configuration
Create a `.streamlit/config.toml` file:
```toml
[server]
headless = true
port = 8501

[browser]
gatherUsageStats = false

[theme]
base = "light"
primaryColor = "#FF4B4B"
```

## Project Structure

```
src/compileo/
├── features/
│   ├── ingestion/          # Document processing
│   ├── chunk/             # Text chunking
│   ├── extraction/        # Information extraction
│   ├── datasetgen/        # Dataset generation
│   ├── datasetqual/       # Quality assurance (optional)
│   ├── benchmarking/      # Performance benchmarking (optional)
│   └── gui/               # Web GUI interface
├── storage/               # Data persistence
├── core/                  # Shared utilities
└── utils/                 # Helper functions
```

## Configuration

### Quality Module Configuration

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

### Benchmarking Module Configuration

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
  },
  "correlation": {
    "enabled": true,
    "significance_threshold": 0.05
  }
}
```

## CLI Commands

### Dataset Generation
```bash
compileo generate-dataset [OPTIONS]

Options:
  --project-id INTEGER          Project ID for dataset generation [required]
  --analyze-quality             Enable quality analysis
  --quality-threshold FLOAT     Quality threshold (0-1) [default: 0.7]
  --quality-config PATH         Path to quality configuration JSON
  --run-benchmarks              Run AI model benchmarks after generation
  --benchmark-suite TEXT        Benchmark suite (glue, superglue, mmlu, medical) [default: glue]
  --benchmark-config PATH       Path to benchmarking configuration JSON
  --output-dir PATH             Output directory [default: .]
```

### Quality Analysis
```bash
compileo analyze-quality [OPTIONS] DATASET_FILE

Arguments:
  DATASET_FILE  Path to dataset file (JSONL/JSON)

Options:
  --config PATH         Quality configuration JSON
  --threshold FLOAT     Quality threshold (0-1) [default: 0.7]
  --format [json|text|markdown]  Output format [default: text]
  --output PATH         Save report to file
```

## Quality Metrics

### Diversity Analysis
- **Lexical Diversity**: Measures vocabulary richness using Type-Token Ratio
- **Semantic Diversity**: Analyzes conceptual variety in content
- **Topic Balance**: Evaluates distribution across categories

### Bias Detection
- **Demographic Bias**: Detects unequal representation of demographic groups
- **Content Bias**: Identifies skewed topic coverage
- **Representation Bias**: Measures imbalanced representation patterns

### Difficulty Assessment
- **Readability**: Flesch Reading Ease scores
- **Complexity**: Linguistic complexity metrics
- **Target Alignment**: Matches content to intended audience level

### Consistency Validation
- **Answer Coherence**: Internal logical consistency
- **Cross-Consistency**: Consistency across related answers
- **Factual Accuracy**: Basic contradiction detection

## Performance Benchmarking

### Benchmark Suites
- **GLUE**: General Language Understanding Evaluation (CoLA, SST-2, MRPC, QQP, MNLI, QNLI, RTE, WNLI)
- **SuperGLUE**: Advanced GLUE tasks (BoolQ, CB, COPA, MultiRC, ReCoRD, RTE, WiC, WSC)
- **MMLU**: Massive Multitask Language Understanding (57 academic subjects)
- **Medical Benchmarks**: PubMedQA, MedMCQA, MedQA, Medical Board Exams

### Performance Metrics
- **Classification Metrics**: Accuracy, Precision, Recall, F1-Score
- **Generation Metrics**: BLEU, ROUGE, METEOR scores
- **Custom Metrics**: Domain-specific evaluation metrics

### Model Performance Tracking
- **Historical Tracking**: Performance trends over time
- **Model Comparison**: Side-by-side performance analysis
- **Performance Degradation Detection**: Automatic alerts for performance drops

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run quality module tests
pytest tests/test_datasetqual.py

# Run benchmarking module tests
pytest tests/test_benchmarking.py

# Run with coverage
pytest --cov=src.compileo.features.datasetqual
```

### Adding Quality Metrics

1. Create new metric class inheriting from `BaseMetric`
2. Implement `analyze()` method
3. Add to `QualityConfig` and `QualityAnalyzer`
4. Write comprehensive tests

## API Usage

### Programmatic Quality Analysis

```python
from compileo.features.datasetqual import QualityAnalyzer, QualityConfig

# Configure quality analysis
config = QualityConfig(enabled=True)
analyzer = QualityAnalyzer(config)

# Analyze dataset
dataset = [
    {"question": "What is AI?", "answer": "Artificial Intelligence", "category": "tech"}
]
results = analyzer.analyze_dataset(dataset)

print(f"Quality Score: {results['summary']['overall_score']}")
```

### Integration Hooks

```python
from compileo.features.datasetqual import QualityHooks

# Add quality checks to dataset generation
hooks = QualityHooks(config)
hooks.register_callback('post_generation', my_quality_callback)
```

## Contributing

1. Follow the established modular architecture
2. Add comprehensive tests for new features
3. Update documentation for API changes
4. Ensure quality metrics are optional and configurable

## License

[Add license information]