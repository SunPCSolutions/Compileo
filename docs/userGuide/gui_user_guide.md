# Compileo GUI User Guide

## Overview

The Compileo GUI provides a user-friendly web interface for document processing, taxonomy management, and dataset generation. This guide covers the main features and workflows available through the GUI.

## Getting Started

### Prerequisites
- Running Compileo API server (`uvicorn src.compileo.api.main:app --host 0.0.0.0 --port 8000`)
- API keys configured for AI services (Gemini, Grok, etc.)
- At least one project created

### Launching the GUI
```bash
streamlit run src/compileo/features/gui/main.py
```

The GUI will be available at `http://localhost:8501` by default.

## Main Interface

### Navigation Layout
The GUI features a modern header and grouped sidebar navigation:

**Header:**
- **🔬 Compileo Dataset Creator**: Application branding

**Sidebar Navigation (Grouped):**
- 🏠 Home
- ⚙️ Settings
- ──────────
- 🧙 **Wizard**
  - Dataset Generation Wizard
- ──────────
- ⚙️ **Workflow**
  - Projects
  - Document Processing
  - Taxonomy
  - Extraction
  - Extraction Results
  - Dataset Generation
- ──────────
- 📊 **Analysis**
  - Quality Metrics
  - Benchmarking
- ──────────
- ⚙️ **Job Management**
  - Job Queue (Real-time monitoring of active and pending jobs)
  - Job History (View all past jobs with filters)

## 📄 Document Processing Tabs

### Parse Documents Tab

#### Purpose
Upload new documents and parse existing ones into a clean markdown format.

#### Workflow
1. **Select Project**: Choose the target project.
2. **Upload Files**: Upload new documents (PDF, DOCX, TXT, etc.).
3. **Select Parser**: Choose the parsing engine (`gemini`, `grok`, `ollama`, `pypdf`, `unstructured`, `huggingface`, `novlm`).
4. **Select Documents**: Check the boxes next to the documents you want to parse.
5. **Parse**: Click the "Parse Documents" button. A job will be submitted to the background queue, and you can monitor its progress in the Job Queue sidebar or the dedicated Job Management page.

### Pre-Parsing PDF Splitter

#### Purpose
For very large PDF documents (e.g., thousands of pages), direct parsing by AI models can lead to token limit issues or summarization. The Pre-Parsing PDF Splitter automatically divides these large PDFs into smaller, manageable chunks (individual PDF files) before they are sent to any parsing model. This ensures that each segment of the document can be processed completely and accurately.

#### Automatic Splitting
- If a PDF document has more than **200 pages**, it will be automatically split into multiple smaller PDF files.
- Each split file will contain approximately **200 pages**.
- An **overlap of 1 page** is included between consecutive split files. This overlap helps maintain content continuity, allowing downstream chunking and parsing processes to handle information that spans across the split boundaries effectively.
- The split files are named sequentially (e.g., `original_document_name_1.pdf`, `original_document_name_2.pdf`).

#### How it Works
When you upload a large PDF or initiate a parsing job for one, the system first checks its page count. If it exceeds the 200-page threshold, the splitter automatically creates the smaller PDF files. These smaller files are then processed sequentially by the chosen parsing engine. From the user's perspective, this process is largely transparent, ensuring reliable parsing of even the largest documents.

### Configure & Chunk Documents Tab

#### Purpose
Configure chunking strategies and apply them to your parsed documents using either manual configuration or AI-assisted recommendations.

#### Configuration Modes

**Manual Configuration (Default):**
Direct parameter setting for experienced users.

**AI-Assisted Configuration:**
Intelligent recommendations based on your document structure and goals.

#### Manual Configuration Workflow
1. **Select Strategy**: Choose chunking method (`character`, `token`, `semantic`, `delimiter`, `schema`)
2. **Configure Parameters**: Set strategy-specific parameters manually
3. **Select Documents**: Choose parsed documents to process
4. **Process**: Apply chunking with your chosen settings. A job will be submitted to the background queue, and you can monitor its progress in the Job Queue sidebar or the dedicated Job Management page.

#### AI-Assisted Configuration Workflow
1. **Describe Goal**: Provide chunking objective (required field)
2. **Select Document**: Choose representative document for analysis
3. **Preview Content**: Browse document with pagination (10K chars per page)
4. **Extract Examples**: Select text directly in the preview area to gather examples
5. **Get Recommendations**: AI analyzes goal, content, and examples
6. **Apply Settings**: Use AI-recommended parameters or make adjustments
7. **Process Documents**: Apply configuration to selected documents

#### AI-Assisted Features
- **Goal Description**: Required field describing chunking objectives
- **Document Preview**: Paginated content viewing with header highlighting
- **Text Selection**: Click and drag to select any portion of document content
- **Real-time Feedback**: Selected text appears in a dedicated field
- **Flexible Examples**: Any text portion can be added to the AI example pool
- **AI Recommendations**: Intelligent strategy and parameter suggestions
- **JSON Schema Auto-Fix**: Automatic correction of backslash escaping issues when copying AI recommendations

#### AI-Assisted Features
- **Goal Description**: Required field describing chunking objectives
- **Document Preview**: Paginated content viewing with header highlighting
- **Text Selection**: Click and drag to select any portion of document content
- **Real-time Feedback**: Selected text appears in a dedicated field
- **Flexible Examples**: Any text portion can be added to the AI example pool
- **AI Recommendations**: Intelligent strategy and parameter suggestions

### Best Practices
- Use `gemini` parser for complex document layouts
- Set chunk size to 1000 for balanced processing
- Use 10-20% overlap for context continuity
- Process multiple related documents together
- For semantic: Use the placeholder example as a template for custom prompts
- For schema: Start with simple regex patterns and build to complex rules
- **Schema Include Pattern**: Use `include_pattern: true` when you want chunks to START with matched patterns (e.g., disease names), `false` when patterns should be excluded from chunks

## 🏷️ Taxonomy Tab

### Three Main Sub-tabs

#### 1. 🏗️ Build Taxonomy
Create new taxonomies using AI generation or manual construction with hybrid capabilities.

**AI Generation Mode**:
- Select project and enter taxonomy name
- Choose AI generator (`gemini`, `grok`, `ollama`)
- Set domain and specificity level
- Select documents to analyze
- Configure category limits per hierarchy level
- Generate taxonomy automatically

**Hybrid Mode**:
- Manually define basic category structure
- Use AI to extend and refine the taxonomy
- Add subcategories automatically
- Load existing taxonomies as starting points

**Manual Structure Building**:
- Add top-level categories with descriptions
- Build hierarchical subcategories
- Set confidence thresholds per category
- AI enhancement for existing manual structures

#### 2. 🔍 Classification & Extraction
Apply taxonomies to content for selective categorization and information extraction.

**Selective Category Selection**:
- Interactive taxonomy tree selector
- Check/uncheck specific categories for extraction
- Preview selection with statistics
- Hierarchical category navigation

**Extraction Parameters**:
- **Extraction Depth**: Maximum taxonomy hierarchy levels to traverse (1-5)
- **Confidence Threshold**: Minimum confidence score for results (0.0-1.0)
- **Skip Fine Classification**: Enable for faster processing (coarse only)
- **Advanced Settings**: Max chunks, batch size, processing controls

**Extraction Workflow**:
1. Select project and taxonomy
2. Choose specific categories using tree selector
3. Configure extraction parameters
4. Run selective extraction job
5. Monitor progress and view results
6. Export results as JSON or CSV

**Results Viewer**:
- Organized by selected categories
- Confidence score filtering
- Paginated results with metadata
- Export options for analysis

#### 3. 📋 Browse & Manage Taxonomies
- Search and filter existing taxonomies
- View taxonomies in tree or list format
- Edit taxonomy metadata
- Export taxonomies as JSON or CSV
- Bulk operations and management
- Delete taxonomies (with confirmation)

### Best Practices
- Start with AI generation for initial taxonomy creation
- Use hybrid mode for iterative refinement
- Choose domain-specific settings for better categorization
- Select specific categories for focused extraction
- Regularly update taxonomies as content evolves
- Use extraction results to improve taxonomy accuracy

## 🔍 Extraction Tab

### Purpose
Advanced entity extraction, relationship inference, and Q&A dataset generation from unstructured text documents using AI-powered analysis.

### Three-Tab Unified Interface

#### 🏃 **Run Extraction Tab**

**Extraction Type & Mode Selection:**
- **Extraction Type**: Select between `Named Entity Recognition (NER)` (extracts specific entities) or `Whole Text Extraction` (extracts complete text portions).
- **Extraction Mode**: Choose `Contextual Extraction` (filters by parent context for precision) or `Document-Wide Extraction` (processes all chunks for maximum coverage).

**AI Model Selection:**
- Choose from Grok, Gemini, or Ollama AI models as the `Primary Classifier`.
- Optionally enable a `Validation Stage` with a different AI model for quality assurance.

**Taxonomy Integration:**
- Select project and taxonomy for entity categorization.
- Interactive taxonomy tree for category selection.
- Search and filter categories for precise targeting.

**Extraction Parameters:**
- **Depth Control**: Maximum taxonomy hierarchy levels (1-5).
- **Confidence Threshold**: Minimum quality score (0.0-1.0).
- **Batch Processing**: Chunk size and processing limits.
- **Advanced Controls**: Performance tuning options.

**Workflow:**
1. Select project and taxonomy.
2. Choose `Extraction Type` and `Extraction Mode`.
3. Choose AI model(s) for extraction.
4. Configure extraction parameters.
5. Select specific categories using tree interface.
6. Start extraction job and monitor progress.

#### 📊 **Monitor Jobs Tab**

**Real-time Job Tracking:**
- Live progress updates for all extraction jobs
- Status indicators (Pending, Running, Completed, Failed)
- Detailed job parameters and timing information
- Action buttons for job management (restart, cancel, view results)

**Job Management:**
- Filter jobs by project and status
- View comprehensive job metadata
- Monitor resource usage and performance
- Handle failed jobs with restart capabilities

#### 📋 **View Results Tab**

**Entity Results Display:**
- Extracted entities organized by taxonomy categories
- Confidence scores and source chunk references
- Frequency analysis across document collections
- Advanced filtering and search capabilities

**Relationship Analysis:**
- Automatic discovery of entity relationships
- Relationship type distribution and quality metrics
- Interactive relationship visualization
- Confidence-weighted association analysis

**Q&A Dataset Generation:**
- One-click generation of question-answer pairs
- Template-based customization for different domains
- Multiple export formats (JSONL, JSON, CSV)
- Quality preview and statistics

### Advanced Features

#### Multi-Model Validation
- Cross-model agreement checking
- Enhanced confidence through AI consensus
- Error detection and quality assurance

#### Scalable Processing
- Large document collection handling
- Batch processing optimization
- Memory-efficient streaming operations
- Parallel AI model utilization

### Best Practices
- **AI Model Selection**: Use Grok for accuracy, Gemini for speed
- **Category Targeting**: Select specific categories for focused extraction
- **Confidence Tuning**: Adjust thresholds based on domain requirements
- **Quality Validation**: Enable multi-model validation for critical applications
- **Resource Monitoring**: Track performance for large-scale operations

## 📊 Extraction Results Tab

### Purpose
View, analyze, and export results from completed extraction jobs.

### Features
- **Job Management**: View all extraction jobs with status
- **Results Organization**: Results organized by selected categories
- **Filtering & Search**: Filter by confidence, category, or content
- **Export Options**: Export as JSON or CSV for analysis
- **Pagination**: Navigate through large result sets
- **Metadata Display**: View extraction metadata and statistics

### Workflow
1. **Select Job**: Choose from completed extraction jobs
2. **Browse Results**: Navigate through categorized results
3. **Apply Filters**: Filter by confidence score or categories
4. **Export Data**: Download results for further analysis
5. **Review Statistics**: Analyze extraction performance metrics

### Best Practices
- Review high-confidence results first
- Use category filtering for focused analysis
- Export results regularly for backup
- Monitor extraction quality metrics

## 🔧 Dataset Generation Tab

### Purpose
Generate high-quality datasets from processed document chunks and extraction results using advanced controls. **Note:** Dataset generation now follows an extraction-first approach - perform taxonomy-based extraction before generating datasets to ensure structured, categorized content is used as input.

### Configuration Sections

#### Basic Settings
- **Project Selection**: Choose source project
- **Generation Mode**: `default`, `question`, `answer`, `summarization`
- **Output Format**: `jsonl` or `parquet`
- **Concurrent Workers**: Number of parallel processing threads (1-10)

#### Quality Control
- **Analyze Quality**: Enable/disable quality analysis
- **Quality Threshold**: Minimum acceptable quality score (0.0-1.0)

#### Advanced Options
- **Include Evaluation Sets**: Generate train/validation/test splits
- **Enable Versioning**: Create versioned dataset snapshots
- **Data Source**: Choose data source for generation (Chunks Only, Taxonomy, Extract)
- **Taxonomy Selection**: Choose taxonomy for content filtering (when using Taxonomy mode)

#### High-Level Prompts
Define the target audience and purpose for more relevant content:
- **Custom Audience**: "medical residents", "data scientists", etc.
- **Custom Purpose**: Specific use case description
- **Complexity Level**: `beginner` to `expert`
- **Domain**: Knowledge area (e.g., "cardiology", "machine learning")

#### Dataset Size Control
- **Datasets per Chunk**: Number of entries to generate per document chunk (1-10)

#### Data Source Modes

**Chunks Only**
- Uses raw text chunks directly from processed documents
- No taxonomy or extraction filtering required
- Best for basic dataset generation from any content

**Taxonomy**
- Applies taxonomy definitions to enhance generation prompts
- Works with all chunks in the project (no extraction dependency)
- Adds domain-specific context and terminology

**Extract**
- Uses extracted entities as the primary content source
- Generates datasets focused on specific concepts/entities
- Creates educational content about extracted terms

#### Model Selection
- **Parsing Model**: Document parsing AI
- **Chunking Model**: Text chunking AI
- **Classification Model**: Content classification AI

### Workflow
1. Configure all parameters according to your needs
2. Click **🚀 Generate Dataset**. A job will be submitted to the background queue, and you can monitor its progress in the Job Queue sidebar or the dedicated Job Management page.
3. Monitor progress in the status section in real-time.
4. Review results and download generated datasets.

### Best Practices
- Start with small datasets (2-3 per chunk) for testing
- Use high-level prompts for domain-specific content
- Enable quality analysis for production datasets
- Use taxonomy filtering for focused content generation

## 🧙 Dataset Creation Wizard

### Purpose
Comprehensive guided workflow for dataset generation with flexible navigation, automatic processing, and complete AI model selection.

### Key Features
- **5-Step Guided Process**: From project selection to review & generate
- **Flexible Navigation**: Click any step tab to navigate non-linearly. Most steps are "resume-ready" and retrieve state directly from the database.
- **Automatic File Upload**: Drag-and-drop with immediate processing
- **Complete AI Model Selection**: 4-model configuration (parsing, chunking, classification, generation)
- **Full Chunking Strategy Parity**: All Document Processing tab strategies available
- **Database-Mediated Workflow**: UI state is synchronized with the database, allowing progress to survive session resets.
- **Smart Data Source Selection**: Automatic taxonomy/chunks fallback
- **Document Management**: Upload and delete capabilities with error correction
- **Real-time Progress Monitoring**: Live job tracking with detailed status updates

### Steps
1. **Project Selection**: Choose or create project with statistics display
2. **Parse & Chunk & Taxonomy**: Automated end-to-end processing. Upload documents, select models and chunking strategy, then initiate the full pipeline from parsing to automatic taxonomy generation.
3. **Edit Taxonomy**: Reactive simplified editor for picking and refining taxonomy structures. Supports renaming and real-time category management.
4. **Generation Parameters**: Configure generation mode, output format, quality settings, and high-level prompt parameters (Audience, Purpose, Complexity, Domain).
5. **Review & Generate**: Comprehensive configuration summary and background job execution with progress monitoring.

### Navigation Features
- **Clickable Step Tabs**: Navigate to any completed step or future steps
- **Prerequisite Validation**: Clear error messages when required steps are missing
- **Progress Tracking**: Visual progress indicators and completion status
- **State Persistence**: Configuration saved across navigation

### Benefits
- **Beginner-Friendly**: Step-by-step guidance with clear instructions
- **Expert Control**: Full access to advanced configuration options
- **Error Prevention**: Validation prevents invalid configurations
- **Workflow Flexibility**: Non-linear navigation for iterative refinement
- **Quality Assurance**: Built-in validation and progress monitoring

## 📊 Quality Metrics Tab

### Features
- Analyze existing datasets for quality issues
- View detailed quality reports
- Compare dataset versions
- Identify areas for improvement

### Quality Metrics
- **Diversity**: Content variety and coverage
- **Consistency**: Internal coherence
- **Difficulty**: Appropriate complexity levels
- **Bias Detection**: Identify potential biases
- **Relevance**: Alignment with intended purpose

## 📈 Benchmarking Tab

### Purpose
Evaluate AI models on generated datasets.

### Supported Benchmarks
- **GLUE**: General Language Understanding
- **SuperGLUE**: Advanced language tasks
- **MMLU**: Massive Multitask Language Understanding
- **Medical Benchmarks**: Domain-specific evaluation

### Workflow
1. Select dataset and benchmark suite
2. Configure evaluation parameters
3. Run benchmark tests
4. Review performance results
5. Compare model performance

## ⚙️ Settings Tab

### Job Handling Configuration
Configure global and per-user limits for concurrent jobs. These settings help manage system resources and ensure fair usage.

- **Max Concurrent Jobs (Global)**: The maximum number of jobs that can run simultaneously across all users.
- **Max Concurrent Jobs Per User**: The maximum number of jobs a single user can run concurrently.

### API Key Configuration

### API Key Configuration
Configure API keys for AI services:
- **Gemini API Key**: Google AI services
- **Grok API Key**: xAI services
- **HuggingFace API Key**: HuggingFace model access
- **Ollama**: Local AI models (no key required)

### System Settings
- **Default Models**: Set preferred AI models
- **Quality Thresholds**: Default quality settings
- **Output Directories**: Configure storage locations

### Plugin Management
- **Plugins Tab**: Manage extensions to Compileo's functionality (upload, list, uninstall).

## Common Workflows

### Complete Entity Extraction & Dataset Generation Pipeline

1. **Create Project** (Projects tab)
2. **Process Documents** (Document Processing tab)
   - Upload medical PDFs
   - Use Gemini parser for document processing
   - Configure chunking with appropriate size and overlap

3. **Generate Taxonomy** (Taxonomy → Build Taxonomy tab)
   - AI generation mode for medical domain
   - Analyze processed documents for category discovery
   - Create hierarchical taxonomy structure

4. **Run Advanced Entity Extraction** (Extraction → Run Extraction tab)
   - Select Grok AI model for high accuracy
   - Choose taxonomy with medical categories
   - Configure extraction parameters (depth, confidence, batch size)
   - Select specific categories (symptoms, diagnoses, medications)
   - Start extraction job and monitor real-time progress

5. **Monitor Extraction Jobs** (Extraction → Monitor Jobs tab)
   - Track job status and progress updates
   - View detailed job parameters and timing
   - Handle any failed jobs with restart functionality

6. **Analyze Extraction Results** (Extraction → View Results tab)
   - Review extracted entities by category
   - Examine relationship discoveries between entities
   - Filter results by confidence scores
   - Analyze entity frequency and distribution

7. **Generate Q&A Dataset** (Extraction → View Results tab)
   - Use one-click Q&A generation from relationships
   - Preview generated question-answer pairs
   - Customize templates for medical education
   - Export in JSONL format for ML training

8. **Quality Assurance** (Quality Metrics tab)
   - Analyze generated Q&A dataset quality
   - Review diversity, consistency, and relevance metrics
   - Validate medical accuracy of generated content

9. **Advanced Analysis** (Benchmarking tab)
   - Test AI models on generated medical datasets
   - Compare performance across different benchmarks
   - Validate dataset effectiveness for training

### Quick Dataset Generation

For users with existing processed content:

1. Select project with processed documents
2. Go to Core Dataset Generation
3. Set basic parameters (mode, format, workers)
4. Configure high-level prompts
5. Generate dataset
6. Review results

## Troubleshooting

### Common Issues

#### "GUI is frozen during processing"
- This issue has been resolved with the **Synchronous In-View Monitoring** system. While the interface "waits" for your specific job to complete to show you the result, it does so using non-blocking placeholders. This prevents the entire page from flickering or resetting your scroll position, providing a smooth and stable experience.

#### "Job stuck in pending/running"
- Check the Job Queue sidebar or Job Management page for detailed status.
- Verify that worker processes are running and connected to Redis.
- Check server logs for errors related to job execution or resource limits.

#### "Job failed unexpectedly"
- Review the job details in the Job Management page for error messages.
- Check server logs for detailed traceback information.
- Ensure all required API keys are configured in the Settings tab.
- Restart the job if it's a transient error.

#### "Too many concurrent jobs"
- Adjust the "Max Concurrent Jobs (Global)" or "Max Concurrent Jobs Per User" settings in the Settings tab.
- Consider scaling up your worker processes if you have available resources.

### Performance Tips

#### "No projects available"
- Create a project first in the Projects tab
- Check API server is running

#### "API key not configured"
- Go to Settings tab
- Add required API keys
- Restart GUI if necessary

#### "No chunks found"
- Process documents first in Document Processing tab
- Check document formats are supported
- Verify processing completed successfully

#### "Taxonomy generation failed"
- Check document content quality
- Try different domain settings
- Reduce sample size if needed

#### "Dataset generation timeout"
- Reduce concurrent workers
- Decrease datasets per chunk
- Process in smaller batches

#### "Invalid \escape error when using schema chunking"
- This occurs when copying AI-recommended JSON schemas into the GUI text area
- The system automatically detects and fixes this issue - look for the "🔧 Auto-fixed JSON schema backslash escaping issues" message
- If the error persists, try re-pasting the JSON from the AI recommendations dialog
- The GUI includes automatic validation and correction for common JSON formatting issues

### Performance Tips

- **Memory Usage**: Reduce concurrent workers on low-memory systems
- **Processing Speed**: Use appropriate chunk sizes (smaller = faster processing)
- **Quality vs Speed**: Disable quality analysis for faster generation
- **Batch Processing**: Process multiple documents together when possible

## API Integration

The GUI uses REST API endpoints for all operations. You can also use these endpoints directly:

```python
import requests

# Example: Generate dataset
data = {
    "project_id": 123,
    "generation_mode": "default",
    "custom_audience": "medical residents",
    "datasets_per_chunk": 3
}

response = requests.post("http://localhost:8000/api/v1/datasets/generate", json=data)
```

## Support and Resources

- **Documentation**: Check `docs/` folder for detailed guides
- **Logging System**: See [Logging System Guide](logging.md) for details on log levels and configuration.
- **CLI Reference**: See `docs/parametersTree.md` for command-line options
- **API Documentation**: Available at `http://localhost:8000/docs` when API server is running
- **Logs**: Check terminal output for detailed error messages

## Best Practices Summary

1. **AI Model Selection**: Choose Grok for accuracy, Gemini for speed, based on your quality vs. performance needs
2. **Taxonomy Design**: Create domain-specific taxonomies that reflect real-world entity relationships
3. **Category Targeting**: Select specific categories rather than extracting everything for better quality
4. **Confidence Tuning**: Adjust confidence thresholds based on domain requirements and use case sensitivity
5. **Multi-Model Validation**: Enable validation with different AI models for critical applications
6. **Relationship Analysis**: Review discovered relationships to improve taxonomy and extraction quality
7. **Q&A Customization**: Use domain-specific templates and customize prompts for your target audience
8. **Quality Assurance**: Always validate extraction results and generated datasets before production use
9. **Synchronous Monitoring**: Trust the in-view status messages for newly started jobs; they are more accurate than simple percentage bars.
10. **Scalable Processing**: Monitor resource usage in the Job Management dashboard and adjust batch sizes for optimal performance.