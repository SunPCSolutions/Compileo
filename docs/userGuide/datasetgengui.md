# Dataset Generation Wizard in Compileo GUI

## Overview

The Compileo Dataset Creation Wizard provides a comprehensive, step-by-step interface for generating high-quality datasets from documents. The wizard features flexible navigation, automatic file upload, complete AI model selection, and intelligent data source handling for optimal dataset creation.

## Key Features

- **6-Step Guided Wizard**: Intuitive workflow with flexible step navigation
- **Automatic File Upload**: Drag-and-drop with immediate processing
- **Complete AI Model Selection**: 4-model configuration (parsing, chunking, classification, generation)
- **Full Chunking Strategy Parity**: All Document Processing tab strategies available
- **Smart Data Source Selection**: Automatic taxonomy/chunks fallback
- **Document Management**: Upload and delete capabilities with error correction
- **Real-time Progress Monitoring**: Live job tracking with detailed status updates
- **Multi-format Support**: JSONL, Parquet, and extensible plugin formats

## GUI Workflow

### Step 1: Project Selection

**Location:** Dataset Creation Wizard → Step 1
**Purpose:** Choose or create a project for dataset generation

**Features:**
- **Existing Project Selection**: Browse and select from available projects
- **New Project Creation**: Create project with name and description
- **Project Statistics**: View document count, chunk count, and processing status
- **Flexible Navigation**: Click any step tab to navigate non-linearly

**Navigation:**
- Use dropdown to select existing project
- Click "Create New Project" for new projects
- Click "Next" or use step tabs to navigate

### Step 2: Document Upload

**Location:** Dataset Creation Wizard → Step 2
**Purpose:** Upload documents for processing with automatic upload and management

**Features:**
- **Drag-and-Drop Upload**: Support for PDF, DOCX, TXT, MD, CSV, JSON, XML files
- **Automatic Upload**: Files upload immediately upon selection (no manual button)
- **Document Management**: Delete uploaded documents if wrong files were selected
- **Existing Document Detection**: Automatically shows previously uploaded project documents
- **Progress Feedback**: Real-time upload status with success/error notifications

**File Support:**
- **PDF**: Automatic splitting for large documents (200+ pages)
- **Office Documents**: DOCX, DOC with full formatting preservation
- **Text Files**: TXT, MD, CSV, JSON, XML
- **Batch Upload**: Multiple files simultaneously

**Document Management:**
- **View Uploaded Files**: See all successfully uploaded documents
- **Delete Files**: Remove incorrect uploads with confirmation
- **File Validation**: Automatic format checking and error reporting

### Step 3: Processing Configuration

**Location:** Dataset Creation Wizard → Step 3
**Purpose:** Configure AI models, chunking strategies, and document selection

**AI Model Selection (4 Models Side-by-Side):**
- **Document Parsing**: grok, gemini, ollama, huggingface, unstructured, pypdf, novlm
- **Text Chunking**: grok, gemini, ollama
- **Document Classification**: grok, gemini, ollama
- **Dataset Generation**: grok, gemini, ollama

**Complete Chunking Strategies (Full Parity with Document Processing Tab):**
- **Character**: Chunk size (100-4000), Overlap (0-500)
- **Token**: Chunk size (100-2000), Overlap (0-200)
- **Semantic**: Similarity threshold (0.1-0.9), Min chunk size (50-500), Custom prompt (text area)
- **Delimiter**: Quick select common delimiters + custom delimiter input
- **Schema**: JSON schema definition (text area)

**Document Selection:**
- **Multiselect Interface**: Choose which documents to include in processing
- **Document Status**: View processing status for each document
- **Batch Selection**: Select all, none, or individual documents

**Prerequisites:**
- Project must be selected (Step 1)
- Documents must be uploaded/available (Step 2)

### Step 4: Taxonomy Selection

**Location:** Dataset Creation Wizard → Step 4
**Purpose:** Choose or generate taxonomy for enhanced dataset generation

**Features:**
- **Browse Taxonomies**: View existing project taxonomies with confidence scores
- **Auto-Generation**: Generate new taxonomy from processed documents
- **Taxonomy Preview**: View hierarchical structure and category counts
- **Optional Step**: Can proceed without taxonomy (automatic chunks-only fallback)

**Taxonomy Generation:**
- **AI-Powered**: Uses selected classification model for intelligent categorization
- **Document Analysis**: Analyzes processed documents to extract categories
- **Depth Control**: Configurable hierarchy levels (default: 3)
- **Domain Adaptation**: General or domain-specific taxonomy generation

**Selection Options:**
- Choose from existing taxonomies
- Generate new taxonomy automatically
- Skip taxonomy (uses chunks-only data source)

**Prerequisites:**
- Project selected (Step 1)
- Documents processed (Step 3)

### Step 5: Generation Parameters

**Location:** Dataset Creation Wizard → Step 5
**Purpose:** Configure dataset generation mode, format, and quality settings

**Dataset Generation Mode:**
- **Instruction Following**: Modern instruction-response pairs (Alpaca/Dolly format)
- **Question and Answer**: Traditional Q&A pairs from content
- **Questions Only**: Generate questions without answers
- **Answers Only**: Generate answers without questions
- **Summarization**: Create content summaries

**Output Configuration:**
- **Format**: JSONL, Parquet, or plugin-provided formats (e.g., "anki" for Anki flashcard export)
- **Plugin Formats**: Dynamically loaded from installed plugins - formats appear/disappear automatically with plugin installation/uninstallation
- **Entries per Chunk**: Number of dataset entries per document chunk (0-50). Set to `0` for "Let AI Decide" mode where the AI determines the appropriate number based on content richness.
- **Quality Threshold**: Minimum quality score (0.0-1.0)
- **Validated Results Only**: Filter extraction results to only include data that has passed the strict validation stage.

**"Let AI Decide" Mode (Entries per Chunk = 0):**

When set to `0`, the AI analyzes content richness and dynamically determines the optimal number of entries:
- **Sparse/simple content** → 1-3 entries
- **Moderate content** → 3-7 entries
- **Rich/comprehensive content** → 7-15+ entries

This mode is recommended when you have varied content density across chunks and want the AI to optimize entry count rather than using a fixed number for all chunks.

**Advanced Options:**
- **Custom Prompt**: Optional custom prompt template with {chunk} placeholder
- **Evaluation Sets**: Generate train/validation/test splits
- **Versioning**: Enable dataset versioning
- **Benchmarking**: Run AI model performance tests

**Prerequisites:**
- Project selected (Step 1)
- Documents processed (Step 3)

### Step 6: Review & Generate

**Location:** Dataset Creation Wizard → Step 6
**Purpose:** Review configuration and execute dataset generation

**Configuration Summary:**
- **Project & Documents**: Selected project and document count
- **AI Models**: All 4 selected models (parsing, chunking, classification, generation)
- **Processing Config**: Chunking strategy and parameters
- **Taxonomy**: Selected taxonomy or "Chunks Only" fallback
- **Generation Settings**: Mode, format, quality parameters

**Smart Data Source Selection:**
- **Taxonomy**: Used when taxonomy is available (Step 4)
- **Chunks Only**: Automatic fallback when no taxonomy exists
- **Display**: Shows which data source will be used

**Dataset Generation:**
- **Background Processing**: Real-time job execution with granular progress monitoring (especially during long category-aggregation phases).
- **Status Updates**: Live progress bar and step-by-step status updates.
- **Job Management**: View job details and handle errors
- **Completion Notification**: Success/failure alerts with results summary
- **Instant Download**: Upon success, a "Prepare Download" button is shown to immediately download the generated dataset locally.

**Prerequisites:**
- Project selected (Step 1)
- Documents processed (Step 3)
- Generation parameters configured (Step 5)

## Interactive Dataset Refinement

### Post-Generation Editing

**Location:** Dataset Management → Refinement Interface
**Purpose:** Manually improve generated datasets through interactive editing

**Features:**
- **Entry Browser**: Paginated view of all dataset entries
- **Inline Editing**: Direct editing of questions, answers, and metadata
- **Bulk Operations**: Apply changes to multiple entries simultaneously
- **Quality Filtering**: Filter entries by quality score or category
- **Feedback Submission**: Provide feedback for AI learning

**Editing Capabilities:**
- **Question Refinement**: Improve clarity and specificity
- **Answer Enhancement**: Correct or expand answer content
- **Category Assignment**: Update or correct categorization
- **Metadata Updates**: Modify difficulty levels and tags

### Quality Dashboard

**Location:** Quality Dashboard
**Purpose:** Visualize and analyze dataset quality metrics

**Dashboard Components:**
- **Overall Quality Score**: Aggregate quality metric with trend
- **Quality Distribution**: Charts showing high/medium/low quality entries
- **Diversity Metrics**: Lexical and semantic diversity visualizations
- **Bias Detection**: Demographic and content bias indicators
- **Difficulty Analysis**: Readability and complexity distributions

**Interactive Features:**
- **Drill-down**: Click charts to see specific entries
- **Filtering**: Filter dashboard by category, quality score, etc.
- **Export**: Download quality reports in various formats
- **Alerts**: Notifications for quality threshold violations

## Benchmarking Visualization

### Performance Analytics

**Location:** Benchmarking Dashboard
**Purpose:** Analyze AI model performance on generated datasets

**Visualization Types:**
- **Performance Charts**: Accuracy, F1, BLEU scores by model
- **Comparison Tables**: Side-by-side model performance
- **Trend Analysis**: Performance changes over time
- **Benchmark Suite Results**: GLUE, SuperGLUE, MMLU, Medical results

**Interactive Features:**
- **Model Filtering**: Compare specific model combinations
- **Metric Selection**: Choose which metrics to display
- **Historical Tracking**: View performance trends over dataset versions
- **Export Capabilities**: Download performance reports

## Settings and Configuration

### API Key Management

**Location:** Settings → API Keys
**Purpose:** Configure AI model API keys for dataset generation

**Key Management:**
- **Gemini API Key**: Google AI Studio key configuration
- **Grok API Key**: xAI API key setup
- **HuggingFace Token**: For HuggingFace model access
- **Status Indicators**: Real-time validation of key functionality

**Security Features:**
- **Encrypted Storage**: Secure key storage and transmission
- **Validation Testing**: Automatic key validation on save
- **Access Logging**: Audit trail for key usage

### Generation Preferences

**Location:** Settings → Dataset Generation
**Purpose:** Set default preferences for dataset generation

**Default Settings:**
- **Preferred Models**: Set default AI models for each task
- **Quality Thresholds**: Default quality requirements
- **Output Preferences**: Default formats and directories
- **Concurrency Limits**: Default parallel processing limits

## Job Monitoring and Management

### Real-time Job Tracking

**Location:** Sidebar → Job Queue
**Purpose:** Monitor active and completed dataset generation jobs

**Job Queue Features:**
- **Active Jobs**: Currently running generation jobs
- **Job Progress**: Real-time progress bars and status
- **Queue Statistics**: Pending, running, completed job counts
- **Job Controls**: Cancel or restart jobs directly from sidebar
- **Persistent Jobs**: Dataset generation jobs are stored in database and survive GUI/browser refreshes and server restarts

### Job History and Details

**Location:** Job Management Page
**Purpose:** Comprehensive job history and detailed status information

**Job Management:**
- **Job History**: Complete list of past generation jobs (persisted across sessions)
- **Detailed Status**: Step-by-step job execution details
- **Performance Metrics**: CPU, memory, and timing statistics
- **Error Logs**: Detailed error information for failed jobs
- **Retry Options**: Restart failed jobs with modified parameters
- **Database Reliability**: Job status and progress are maintained even if the server restarts during processing

## Best Practices for GUI Usage

### Project Preparation

1. **Document Upload**: Ensure all documents are uploaded and processed
2. **Chunking Completion**: Verify chunking has completed successfully
3. **Taxonomy Setup**: Create or select appropriate taxonomy
4. **API Key Configuration**: Verify all required API keys are configured

### Model Selection Strategy

1. **Content Analysis**: Choose models based on document complexity
2. **Cost Consideration**: Balance quality needs with API costs
3. **Local vs Cloud**: Use Ollama for development, cloud models for production
4. **Testing**: Start with small datasets to validate model performance

### Plugin Format Usage

1. **Format Discovery**: Plugin-provided formats appear automatically in the format dropdown after plugin installation
2. **Dual Output**: Plugin formats generate both standard JSON and custom format files for reliability
3. **Format Compatibility**: Verify plugin formats meet your target system's requirements (e.g., Anki import specifications)
4. **Plugin Management**: Uninstall unused plugins to clean up format options in the GUI

### Quality Assurance Workflow

1. **Enable Quality Analysis**: Always enable for production datasets
2. **Set Appropriate Thresholds**: Balance quality with generation volume
3. **Review Quality Reports**: Use dashboards to identify improvement areas
4. **Iterative Refinement**: Use refinement interface for continuous improvement

### Performance Optimization

1. **Concurrency Tuning**: Adjust based on available resources and API limits
2. **Chunk Size Optimization**: Balance chunk size with generation quality
3. **Monitoring**: Use job monitoring to identify bottlenecks
4. **Resource Planning**: Monitor API usage and costs

## Troubleshooting

### Common Issues

**"No chunks found for project"**
- Solution: Ensure documents have been uploaded and processed through the document pipeline
- Check: Document status in project view should show "processed"

**"API key not configured"**
- Solution: Configure required API keys in Settings → API Keys
- Check: Status indicators should show green for configured keys

**"Quality threshold not met"**
- Solution: Lower quality threshold or improve source document quality
- Alternative: Disable quality analysis for initial dataset generation

**"Job stuck in pending"**
- Solution: Dataset generation jobs are now database-backed and persist across server restarts. If a job appears stuck, it may be processing or waiting for resources.
- Check: Use Job Management page to view detailed status and error logs

### Performance Issues

**Slow Generation**
- Increase concurrency setting (if API limits allow)
- Use more efficient models (Ollama for local processing)
- Reduce datasets per chunk

**Memory Issues**
- Reduce concurrency setting
- Process smaller batches
- Monitor system resources in Job Management

**API Rate Limits**
- Reduce concurrency to stay within API limits
- Implement delays between requests
- Monitor API usage in settings

## Integration with CLI

The GUI provides seamless integration with CLI tools for advanced workflows. For instance, if you generated a dataset in the GUI, you can instantly download or list it via the CLI:

```bash
# List datasets
compileo datasets list --project-id 1

# Download dataset
compileo datasets download my_dataset.jsonl --project-id 1

# View dataset details
compileo datasets details my_dataset.jsonl --project-id 1

# Delete dataset
compileo datasets delete my_dataset.jsonl --project-id 1
```

Other integration examples:

```bash
# Generate dataset via GUI, then analyze quality via CLI
compileo analyze-quality dataset_from_gui.jsonl --format markdown --output quality_report.md

# Use CLI for bulk operations after GUI refinement
compileo dataset-version increment-version --project-id 1 --dataset-name refined_dataset --version-type minor
```

The GUI offers an intuitive, visual approach to dataset generation while maintaining full compatibility with CLI tools for advanced users and automation scenarios.