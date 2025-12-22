# Documents Module GUI Usage Guide

The Compileo Documents GUI provides an intuitive web interface for document management, including upload, parsing, chunking, and content viewing. This guide covers all GUI features with step-by-step instructions.

## Accessing Document Processing

Navigate to the **"📄 Document Processing"** page from the main menu. The interface is organized into two main tabs:

1. **📄 Parse Documents** - Upload and parse documents
2. **✂️ Configure & Chunk Documents** - Configure chunking and process parsed documents

---

## Project Selection

Before working with documents, select a project from the dropdown at the top of the page:

- **Project Selection**: Choose from available projects
- **Auto-save**: Your selection is remembered across sessions
- **Validation**: System prevents operations without a valid project

---

## Tab 1: Parse Documents

### Document Upload

1. **File Upload Area**:
   - Click "Browse files" or drag-and-drop files
   - **Supported formats**: PDF, DOCX, TXT, MD, CSV, JSON, XML
   - **Multiple files**: Upload several documents at once

2. **PDF Splitting Configuration** (appears after file selection):
    - **Pages per Split**: Number of pages per PDF chunk (default: 5, recommended: 5-10)
    - **Overlap Pages**: Overlapping pages between chunks for context continuity (default: 0)
    - **Purpose**: Automatically splits large PDFs to optimize AI parsing performance
    - **Automatic**: PDFs are split when `total_pages > pages_per_split`

### Parser Selection

Choose from available AI parsers:
- **gemini**: Google's Gemini models
- **grok**: xAI's Grok models
- **ollama**: Local Ollama models (configurable parameters)
- **pypdf**: Python PDF parser (fast, no API required)
- **unstructured**: Unstructured.io parser
- **huggingface**: Hugging Face models
- **novlm**: No-VLM models

**Ollama Parser Configuration**: When using Ollama parsers, you can fine-tune AI behavior by configuring parameters in Settings → AI Model Configuration. Available parameters include temperature, repeat penalty, top-p, top-k, and num_predict for optimal parsing results.

### Document Selection

**Existing Documents**:
- View all documents in the selected project
- **Status indicators**:
  - ✅ **Parsed**: Document successfully converted to markdown
  - 📄 **Uploaded**: Document uploaded but not yet parsed
- **Selection**: Check boxes to select documents for parsing
- **Delete**: Click 🗑️ to remove documents

**Upload + Parse**:
- Upload new files AND select existing documents simultaneously
- System processes both in a single operation

### Parsing Action

1. **Parse Button**: Shows count of documents to be processed
2. **Progress**: Real-time status updates during parsing
3. **Results**: Success/failure notifications with job IDs
4. **Background Processing**: Long operations run asynchronously

---

## Tab 2: Configure & Chunk Documents

### Chunking Strategy Selection

Choose from five chunking strategies:

#### 1. Character-Based Chunking
- **Best for**: Speed and predictability
- **Parameters**:
  - **Chunk Size**: Characters per chunk (100-4000)
  - **Overlap**: Overlapping characters (0-500)
- **Use cases**: Large document sets, batch processing

#### 2. Token-Based Chunking
- **Best for**: LLM compatibility
- **Parameters**:
  - **Chunk Size**: Tokens per chunk (100-2000)
  - **Overlap**: Overlapping tokens (0-200)
- **Use cases**: Preparing data for language models

#### 3. Semantic Chunking
- **Best for**: Meaningful content boundaries
- **Parameters**:
  - **Similarity Threshold**: Boundary detection sensitivity (0.1-0.9)
  - **Min Chunk Size**: Minimum characters per chunk (50-500)
  - **Custom Prompt**: AI instructions for boundary detection
- **Use cases**: Research papers, technical documentation

#### 4. Delimiter-Based Chunking
- **Best for**: Structured documents with known separators
- **Parameters**:
  - **Delimiters**: Select from predefined patterns
  - Custom delimiters: `\n\n`, `\n`, `. `, `! `, `? `
- **Use cases**: Markdown files, structured text

#### 5. Schema-Based Chunking
- **Best for**: Complex splitting rules
- **Parameters**:
  - **Schema JSON**: Custom rules combining patterns and delimiters
- **Use cases**: Proprietary formats, complex document structures

### AI Model Selection

Choose the AI model for intelligent chunking:
- **gemini**: Google's Gemini (recommended)
- **grok**: xAI's Grok
- **ollama**: Local models

### Configuration Mode

#### Manual Configuration
- Set all parameters manually
- Full control over chunking behavior
- Suitable for experienced users

#### AI-Assisted Configuration
- **Smart Recommendations**: AI analyzes your documents and suggests optimal settings
- **User Instructions**: Describe your chunking goals
- **Examples**: Provide sample text from your documents
- **Automatic Optimization**: System recommends strategy, size, and overlap

### Document Selection for Chunking

**Available Documents**: Only parsed documents (status: ✅) can be chunked

**Multi-file Selection**:
- Check boxes to select multiple documents
- Grid layout for efficient selection
- Real-time count of selected documents

### Advanced Features: Multi-part Documents

**Split Document Handling**:
- System detects documents split into multiple parts
- **File Selection Dropdown**: Choose which part to chunk
- **Metadata Display**: Page ranges, overlap information
- **Content Preview**: View content before chunking

**Manifest Support**:
- Automatic detection of document manifests
- Page range information for split files
- Overlap visualization between chunks

### Content Preview

**Before Chunking**:
- Preview parsed content for selected files
- **File-Specific Viewing**: Select individual parsed file chunks from multi-part documents
- **Full Content Access**: View complete content of selected parsed files (up to 10,000+ characters)
- **Pagination Support**: Navigate through large content with 10,000-character pages
- Copy-to-clipboard functionality for examples
- Verify content integrity before processing

### Chunking Execution

1. **Chunk Button**: Initiates chunking process
2. **Progress Monitoring**: Real-time status updates
3. **Background Processing**: Asynchronous execution for large jobs
4. **Results Display**: Success metrics and chunk counts

---

## Job Monitoring

### Processing Status Display

**Active Jobs**:
- Real-time progress bars
- Current operation status
- Estimated completion time
- Error notifications

**Job History**:
- Previous processing jobs
- Success/failure status
- Performance metrics
- Detailed logs

### Status Indicators

- **⏳ Pending**: Job queued for processing
- **🔄 Running**: Currently being processed
- **✅ Completed**: Successfully finished
- **❌ Failed**: Processing errors occurred

---

## Document Management

### Viewing Document List

**Project Documents**:
- All documents in selected project
- Status indicators for each document
- File sizes and upload dates
- Quick actions (delete, view content)

### Document Content Viewer

**Parsed Content Access**:
- Click document name to view parsed markdown
- Pagination support for large documents
- Search functionality within content
- Export options (copy, download)

### Document Deletion

**Safe Deletion**:
- Confirmation prompts prevent accidents
- Associated chunks automatically removed
- File system cleanup included
- Database record removal

---

## Best Practices

### 1. Document Preparation

**File Organization**:
- Use consistent naming conventions
- Group related documents in same project
- Check file sizes before upload (large PDFs may need splitting)

**Format Selection**:
- PDFs: Use for scanned documents or complex layouts
- DOCX/TXT: Use for text-heavy content
- MD: Use for already structured content

### 2. Parsing Strategy

**Parser Selection**:
- **pypdf**: Fast, no API costs, good for simple PDFs
- **gemini/grok**: Best quality, handles complex layouts
- **ollama**: Local processing, privacy-focused

**Batch Processing**:
- Parse multiple documents together for efficiency
- Monitor job status for large batches
- Use pagination settings for oversized documents

### 3. Chunking Optimization

**Strategy Selection**:
- **Character**: Fast processing, predictable results
- **Semantic**: Quality chunks, slower processing
- **Schema**: Precise control, requires expertise

**Parameter Tuning**:
- Start with defaults, adjust based on results
- Use AI-assisted configuration for optimization
- Test on sample documents before full processing

### 4. Quality Assurance

**Content Verification**:
- Always preview parsed content before chunking
- Check for parsing errors or missing content
- Validate chunk boundaries make sense

**Performance Monitoring**:
- Track processing times for different strategies
- Monitor API usage and costs
- Log successful configurations for reuse

---

## Troubleshooting

### Common Issues

**Upload Failures**:
- Check file size limits
- Verify supported formats
- Ensure write permissions on storage directory

**Parsing Errors**:
- Verify API keys are configured
- Check document isn't corrupted
- Try different parser if one fails

**Chunking Problems**:
- Ensure documents are parsed first
- Check chunking parameters are valid
- Verify AI model is available

**Performance Issues**:
- Use smaller batch sizes for large documents
- Switch to faster parsers (pypdf) for bulk processing
- Monitor system resources during processing

---

## Advanced Features

### Split PDF Management

**Large Document Handling**:
- Automatic splitting for oversized PDFs
- Configurable page ranges and overlaps
- Manifest tracking for split relationships
- Selective chunking of individual splits

### Custom Schema Chunking

**Rule Definition**:
```json
{
  "rules": [
    {"type": "pattern", "value": "^## "},
    {"type": "delimiter", "value": "\n\n"}
  ],
  "combine": "any"
}
```

**Advanced Patterns**:
- Regex patterns for headers and sections
- Multiple delimiter combinations
- Hierarchical rule processing

### API Integration

**Programmatic Access**:
- All GUI features available via REST API
- Batch processing scripts
- Integration with external workflows
- Automated document pipelines

This GUI provides a complete document processing workflow from upload to chunking, with both manual control and AI-assisted optimization for maximum efficiency and quality.