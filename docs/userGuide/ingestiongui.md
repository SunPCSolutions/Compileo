# Ingestion Module GUI Usage Guide

The Compileo Ingestion GUI provides an intuitive web interface for document parsing and processing. This guide covers the document upload, parsing, and PDF splitting features available in the GUI.

<div align="center">
  <a href="../img/DocumentProcessing.png"><img src="../img/DocumentProcessing.png" width="400" alt="Intelligent Document Processing Workflow"></a>
  <p><i>Intelligent Document Processing Workflow</i></p>
</div>

## Accessing Document Processing

Navigate to the **"📄 Document Processing"** tab from the main menu. The interface provides comprehensive document management capabilities.

---

## Project Selection

Before working with documents, select a project from the dropdown at the top of the page:

- **Project Selection**: Choose from available projects in your workspace
- **Auto-save**: Your selection is remembered across browser sessions
- **Validation**: System prevents operations without a valid project selection

---

## Document Upload & Parsing

### File Upload Area

1. **Upload Interface**:
   - Click "Browse files" or drag-and-drop documents onto the upload area
   - **Supported Formats**: PDF, DOCX, DOC, TXT, MD, CSV, JSON, XML, Images (PNG, JPEG, WEBP)
   - **Multiple Files**: Upload several documents simultaneously
   - **File Size Limits**: Maximum 200MB per file

2. **Website Scraping** (if plugin installed):
   - If the Scrapy-Playwright plugin is installed, a toggle will appear to select between "Upload" and "Website" mode.
   - **URL Input**: Enter the full URL (e.g., `https://example.com`) to scrape.
   - **Depth**: Select crawling depth (default 1 for single page).
   - The system will scrape the website content and process it as a document.

3. **PDF Splitting Configuration** (appears after file selection):
   - **Pages per Split**: Number of pages per PDF chunk (default: 5, recommended: 5-10)
   - **Overlap Pages**: Overlapping pages between chunks for context continuity (default: 0)
   - **Purpose**: Automatically splits large PDFs to prevent AI token limit issues

### Parser Selection

The GUI features **Reactive Parser Filtering**. When you upload files, the system automatically detects their types and restricts the available parsers to only those capable of processing your specific documents.

| Parser | Best For | Supported Formats | API Key |
|--------|----------|-------------------|---------|
| **gemini** | Complex layouts, Vision | PDF, TXT, MD, CSV, Images | Yes |
| **grok** | Technical PDFs | PDF | Yes |
| **ollama** | Local processing | PDF, TXT, MD | No |
| **pypdf** | Simple extraction | PDF | No |
| **unstructured**| Office, MD, Data | **ALL** (DOCX, CSV, XML, etc.) | No |
| **huggingface** | Advanced OCR | PDF | Yes |
| **novlm** | Smart routing | **ALL** (Auto-selects best engine) | No |

**Ollama Configuration**: When using Ollama parsers, fine-tune AI behavior in Settings → AI Model Configuration with parameters like temperature, repeat penalty, and token limits.

### Document Selection & Management

**Existing Documents**:
- View all documents in the selected project
- **Status Indicators**:
  - ✅ **Parsed**: Successfully converted to structured markdown
  - 📄 **Uploaded**: Document uploaded but not yet parsed
  - 🔄 **Processing**: Currently being parsed
  - ❌ **Failed**: Parsing encountered errors
- **Bulk Selection**: Check boxes to select multiple documents
- **Delete Function**: Click 🗑️ to remove documents with confirmation

**Combined Operations**:
- Upload new files AND select existing documents simultaneously
- System processes both uploaded and selected documents in a single operation

---

## PDF Splitting Integration

### Automatic PDF Processing

When you upload or parse PDFs, the system automatically:

1. **Page Count Analysis**: Checks total pages in each PDF
2. **Splitting Decision**: Splits PDFs when `total_pages > pages_per_split`
3. **Chunk Creation**: Creates individual PDF files for each page range
4. **Manifest Generation**: Creates metadata file tracking all splits
5. **Structure Analysis (VLM)**: For VLM parsers, the system analyzes the middle chunk (or full file) to generate a "Style Guide" based on the document's visual hierarchy (fonts, sizes).
6. **Context-Aware Parsing**: Each chunk is parsed separately with your chosen AI model, using the generated Style Guide to ensure consistent headings (#, ##, ###) and clean output (no icons).

### Split File Organization

**Upload Directory Structure**:
```
storage/uploads/{project_id}/
├── {document_id}_{original_name}_manifest.json
├── {document_id}_{original_name}_chunk_001.pdf  # Pages 1-5
├── {document_id}_{original_name}_chunk_002.pdf  # Pages 6-10
└── {document_id}_{original_name}_chunk_003.pdf  # Pages 11-15
```

**Parsed Results**:
```
storage/parsed/{project_id}/
├── {document_id}_1.json    # Parsed content from chunk 1
├── {document_id}_2.json    # Parsed content from chunk 2
└── {document_id}_3.json    # Parsed content from chunk 3
```

### Manifest File Details

The manifest file contains complete splitting metadata:

```json
{
  "original_file": "medical_textbook.pdf",
  "total_pages": 150,
  "pages_per_split": 5,
  "overlap_pages": 0,
  "splits": [
    {
      "chunk_id": 1,
      "start_page": 1,
      "end_page": 5,
      "filename": "chunk_001.pdf"
    },
    {
      "chunk_id": 2,
      "start_page": 6,
      "end_page": 10,
      "filename": "chunk_002.pdf"
    }
  ]
}
```

---

## Parsing Execution

### Start Parsing Process

1. **Parse Button**: Shows count of documents to be processed
2. **Progress Monitoring**: Real-time status updates with progress bars
3. **Background Processing**: Large parsing jobs run asynchronously
4. **Job Tracking**: Each parsing operation gets a unique job ID

### Job Status Monitoring

**Active Jobs Display**:
- Real-time progress bars for current operations
- Estimated completion times
- Current processing phase (uploading, splitting, parsing)
- Error notifications with retry options

**Job History**:
- Previous parsing jobs with success/failure status
- Performance metrics (processing time, pages parsed)
- Detailed error logs for troubleshooting

---

## Content Preview & Validation

### Parsed Content Viewer

**Access Parsed Results**:
- Click document names to view parsed markdown content
- **Pagination Support**: Navigate through large documents (10,000+ characters per page)
- **Search Functionality**: Find specific content within parsed documents
- **Export Options**: Copy content or download as files

### Multi-Part Document Handling

**Split Document Management**:
- System automatically detects documents split into multiple parts
- **File Selection**: Choose which chunk to view from dropdown
- **Metadata Display**: Shows page ranges and overlap information
- **Content Preview**: Verify parsing quality before further processing

---

## Best Practices

### 1. Document Preparation

**File Organization**:
- Use consistent naming conventions for related documents
- Group documents by project or content type
- Check file sizes before upload (large PDFs will be automatically split)

**Format Selection**:
- **PDFs**: Best for scanned documents, complex layouts, or images
- **DOCX/TXT**: Ideal for text-heavy content and structured documents
- **MD**: Use for already structured or pre-processed content

### 2. Parser Selection Strategy

**For Speed & Cost Efficiency**:
- Use `pypdf` for simple text extraction (no API costs)
- Use `unstructured` for Office documents and structured content
- Use `ollama` for local processing without internet dependency

**For Quality & Complex Content (VLM Parsers)**:
These parsers use a **Two-Pass Strategy** (Skim + Parse) for superior structure detection:
- Use `gemini` for documents with images, tables, or complex layouts.
- Use `grok` for technical documentation and research papers.
- Use `huggingface` for scanned documents requiring OCR.
- Use `openai` or `ollama` for versatile, vision-based parsing.

### 3. PDF Splitting Optimization

**Small Documents (< 50 pages)**:
- Default `pages_per_split: 5` works well
- Consider increasing to 10 if parsing speed is priority

**Large Documents (> 200 pages)**:
- Keep `pages_per_split: 5` for optimal AI model performance
- Consider `overlap_pages: 1` for documents where context spans page boundaries

**Special Cases**:
- **Image-heavy PDFs**: Smaller chunks (3-5 pages) for better OCR accuracy
- **Text-dense PDFs**: Larger chunks (8-10 pages) for better context preservation

### 4. Batch Processing Guidelines

**Optimal Batch Sizes**:
- **AI Parsers** (gemini, grok): 3-5 documents per job
- **Fast Parsers** (pypdf, unstructured): 10-20 documents per job
- **Mixed Batches**: Group similar document types together

**Monitoring & Scaling**:
- Monitor job queue status to avoid system overload
- Use job history to track processing times and success rates
- Scale batch sizes based on observed performance

### 5. Quality Assurance

**Content Verification**:
- Always preview parsed content before chunking or dataset generation
- Check for parsing artifacts, missing content, or formatting issues
- Validate that PDF splitting preserved document structure

**Error Handling**:
- Failed jobs can be restarted with different parser settings
- Check API key validity for cloud-based parsers
- Monitor system resources during large batch operations

---

## Troubleshooting

### Common Upload Issues

**File Size Limits**:
- Maximum 200MB per file (configurable via GUI Settings)
- Large PDFs are automatically split, but consider pre-splitting very large files
- Check available disk space before large uploads

**Unsupported Formats**:
- Verify file extensions match supported formats
- Some file types may require specific parser selection

### Parsing Problems

**API Key Issues**:
- Verify API keys are configured in Settings for cloud parsers
- Check API key validity and rate limits
- Consider switching to local parsers (ollama, pypdf) if API issues persist

**Document Corruption**:
- Ensure PDF files are not corrupted or password-protected
- Try different parsers for problematic documents
- Check file encoding for text-based documents

**HuggingFace Model Download Issues**:
- The Nanonets-OCR2-3B model is large (~6GB) and can hang during download in certain Docker network environments.
- **Symptoms**: Parsing jobs "hang" at the model loading stage without errors, or SSL connection errors appear in logs.
- **Solution**: Pre-populate the `compileo_hf_models` Docker volume by manually copying model weights if download issues persist.
- **GPU Driver Mismatch**: If HuggingFace defaults to CPU, ensure your host NVIDIA driver supports CUDA 13.0.

**Memory Issues**:
- Large documents may require more system memory
- Consider splitting very large PDFs into smaller chunks
- Monitor system resources during processing

### PDF Splitting Issues

**Unexpected Splitting**:
- PDFs are automatically split when `total_pages > pages_per_split`
- Adjust `pages_per_split` setting if splitting is too aggressive
- Some documents may benefit from manual pre-splitting

**Split Quality Problems**:
- Page boundaries may split content inappropriately
- Use `overlap_pages: 1` to maintain context across splits
- Review manifest files to verify split quality

---

## Advanced Configuration

### Custom Parser Settings

**Ollama Advanced Configuration**:
```json
{
  "temperature": 0.1,
  "repeat_penalty": 1.1,
  "top_p": 0.9,
  "top_k": 40,
  "num_predict": 512,
  "seed": 42
}
```

**HuggingFace Optimization**:
- Automatic GPU detection and utilization
- Model caching for improved performance
- Batch processing for multiple images

### API Integration

**Programmatic Access**:
- All GUI features available via REST API
- Batch processing scripts using API endpoints
- Integration with external document processing workflows
- Automated document pipelines

This GUI provides a complete document ingestion workflow with intelligent PDF splitting, multiple parser options, and comprehensive monitoring for efficient and high-quality document processing.
