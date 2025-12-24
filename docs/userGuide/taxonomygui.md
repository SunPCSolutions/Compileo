# Taxonomy Module GUI Usage Guide

The Compileo Taxonomy GUI provides an intuitive web interface for taxonomy management, including creation, generation, extension, and content extraction. This guide covers all GUI features with step-by-step instructions.

## Accessing Taxonomy Builder

Navigate to the **"🏷️ Taxonomy Builder"** page from the main menu. The interface is organized into three main tabs:

1. **🏗️ Build Taxonomy** - Create and edit taxonomies
2. **📤 Extraction** - Extract content using taxonomies
3. **📋 Browse & Manage Taxonomies** - View and manage existing taxonomies

---

## Tab 1: Build Taxonomy

### Unified Taxonomy Builder

The main taxonomy building interface combines manual editing with AI assistance:

#### Manual Taxonomy Creation

1. **Start New Taxonomy**:
   - Click "Create New Taxonomy"
   - Enter taxonomy name and description
   - Select project association

2. **Add Root Categories**:
   - Click "Add Category" to create top-level categories
   - Enter category name and description
   - Set confidence threshold (0.0-1.0)

3. **Build Hierarchy**:
   - Click on any category to expand
   - Add subcategories with "Add Subcategory"
   - Drag and drop to reorganize structure
   - Delete categories with confirmation

4. **Import/Export**:
   - Import taxonomy from JSON file
   - Export current taxonomy structure
   - Validate taxonomy structure before saving

#### AI-Assisted Generation

1. **AI Generation Setup**:
    - Select "Generate with AI" mode
    - Choose AI model (Gemini, Grok, Ollama)
    - Select source documents (must be parsed)

**Ollama Generator Configuration**: When using Ollama for taxonomy generation, you can fine-tune AI behavior by configuring parameters in Settings → AI Model Configuration. Available parameters include temperature, repeat penalty, top-p, top-k, and num_predict for optimal taxonomy generation results.

2. **Generation Parameters**:
   - **Domain**: Content domain (medical, legal, technical, general)
   - **Processing Mode**:
     - **Fast (Sampled)**: Quickly generates taxonomy from a sample of up to 10 chunks.
     - **Complete (All Content)**: Iteratively processes every chunk in the document for comprehensive coverage.
   - **Depth**: Hierarchy levels (1-5)
   - **Chunk Batch Size**: Number of complete chunks to process per batch (1-50)
   - **Category Limits**: Max categories per level
   - **Specificity Level**: Detail level (1-5)

3. **Generation Process**:
   - Click "Generate Taxonomy"
   - Monitor progress in real-time
   - Review generated structure
   - Edit manually if needed
   - Save final taxonomy

#### Taxonomy Extension

1. **Extend Existing Taxonomy**:
   - Select taxonomy to extend
   - Choose extension method:
     - **Add Levels**: Add depth to entire taxonomy
     - **Expand Category**: Extend specific category
     - **Refine Existing**: Improve existing categories

2. **Extension Parameters**:
   - Additional depth levels
   - AI model selection
   - Domain specification
   - Sample size adjustment

3. **Review and Apply**:
   - Preview extension results
   - Accept or modify changes
   - Save extended taxonomy

---

## Tab 2: Extraction

### Content Classification Setup

1. **Select Taxonomy**:
   - Choose taxonomy for extraction
   - View taxonomy structure preview
   - Select specific categories or use entire taxonomy

2. **Document Selection**:
   - Choose project containing documents
   - Select individual documents or all documents
   - Filter by document status (parsed, chunked)

3. **Extraction Parameters**:
   - **Confidence Threshold**: Minimum classification confidence (0.0-1.0)
   - **Max Chunks**: Limit processing volume
   - **Validation Stage**: Enable two-stage classification
   - **Primary Classifier**: Main AI model
   - **Validation Classifier**: Secondary model for validation

### Extraction Process

1. **Start Extraction**:
   - Click "Start Extraction Job"
   - Monitor progress with real-time updates
   - View processing statistics

2. **Results Review**:
   - Browse extracted content by category
   - Filter results by confidence score
   - Export results to various formats
   - Generate summary reports

3. **Quality Assessment**:
   - View classification accuracy metrics
   - Identify low-confidence classifications
   - Re-run extraction with adjusted parameters

### Advanced Features

**Batch Extraction**:
- Process multiple documents simultaneously
- Queue extraction jobs for background processing
- Monitor multiple jobs in job dashboard

**Incremental Extraction**:
- Extract from new documents only
- Update existing extractions
- Merge results across multiple runs

---

## Tab 3: Browse & Manage Taxonomies

### Taxonomy Browser

1. **Taxonomy List**:
   - View all taxonomies in selected project
   - Sort by name, creation date, confidence score
   - Filter by taxonomy type (manual, AI-generated)

2. **Taxonomy Details**:
   - Click taxonomy name to view full structure
   - Expand/collapse hierarchy levels
   - View category statistics and confidence scores
   - Export taxonomy structure

3. **Analytics Dashboard**:
   - **Depth Analysis**: Hierarchy depth and distribution
   - **Category Count**: Total categories and distribution
   - **Confidence Metrics**: Average and distribution of confidence scores
   - **Usage Statistics**: Extraction jobs and results

### Taxonomy Management

#### Edit Taxonomy

1. **Modify Structure**:
   - Add, remove, or rename categories
   - Reorganize hierarchy with drag-and-drop
   - Update category descriptions and confidence thresholds

2. **Bulk Operations**:
   - Import categories from CSV/JSON
   - Export selected branches
   - Clone taxonomy structure

#### Delete Taxonomy

1. **Safe Deletion**:
    - Confirmation prompts prevent accidents
    - **Complete Cleanup**: Deleting a taxonomy automatically removes its file from the filesystem and cleans up all associated extraction jobs and their results (both database entries and filesystem files).
    - Check for dependent extraction jobs
    - Option to archive instead of delete

2. **Bulk Management**:
   - Select multiple taxonomies for deletion
   - Filter by criteria (old, low-confidence, unused)
   - Batch operation confirmation

### Taxonomy Comparison

1. **Side-by-Side View**:
   - Compare two taxonomies visually
   - Highlight differences in structure
   - Merge compatible branches

2. **Metrics Comparison**:
   - Compare depth, category count, confidence scores
   - View overlap analysis
   - Generate comparison reports

---

## Advanced GUI Features

### State Management

**Session Persistence**:
- Current selections remembered across page refreshes
- Unsaved changes protected with confirmation prompts
- Progress tracking for long-running operations

**Navigation States**:
- Seamless transitions between views
- Breadcrumb navigation for deep taxonomy editing
- Back/forward navigation support

### Real-time Updates

**Live Progress**:
- Real-time progress bars for generation and extraction
- Live statistics updates during processing
- Instant feedback on parameter changes

**Collaborative Features**:
- Lock mechanism for concurrent editing
- Change notifications for shared taxonomies
- Version history tracking

### Keyboard Shortcuts

- **Ctrl+S**: Save current taxonomy
- **Ctrl+Z**: Undo last change
- **Ctrl+Y**: Redo last change
- **Delete**: Remove selected category
- **Enter**: Add subcategory to selected item

---

## Best Practices

### Taxonomy Design

**Structure Guidelines**:
- Start with 3-4 levels maximum for usability
- Use clear, descriptive category names
- Maintain consistent naming conventions
- Set appropriate confidence thresholds

**Quality Assurance**:
- Regularly review and update taxonomy structure
- Test extraction accuracy on sample documents
- Maintain version history for important taxonomies
- Document taxonomy purpose and scope

### Performance Optimization

**Large Taxonomies**:
- Use category limits to control growth
- Implement pagination for deep hierarchies
- Consider splitting very large taxonomies

**Processing Efficiency**:
- Select appropriate sample sizes for generation
- Use incremental extraction for updates
- Monitor and optimize confidence thresholds

### Maintenance Workflows

**Regular Maintenance**:
```python
# Monthly taxonomy review checklist
- [ ] Review extraction accuracy metrics
- [ ] Update category descriptions
- [ ] Remove unused categories
- [ ] Test on new document types
- [ ] Archive outdated taxonomies
```

**Version Control**:
- Create backups before major changes
- Use descriptive names for taxonomy versions
- Document changes and rationale
- Maintain changelog for important taxonomies

---

## Troubleshooting

### Common Issues

**Generation Failures**:
- Check document parsing status
- Verify API key configuration
- Ensure sufficient document content
- Try different AI models

**Extraction Problems**:
- Validate taxonomy structure
- Check confidence threshold settings
- Review document chunking quality
- Monitor API rate limits

**Performance Issues**:
- Reduce sample sizes for large document sets
- Use category limits to control taxonomy size
- Implement pagination for large result sets

**UI Responsiveness**:
- Clear browser cache and cookies
- Check internet connection stability
- Close unused browser tabs
- Update browser to latest version

---

## Integration Examples

### Workflow Automation

**Document Processing Pipeline**:
1. Upload and parse documents
2. Generate or select taxonomy
3. Configure extraction parameters
4. Run batch extraction jobs
5. Review and export results

**Quality Assurance Process**:
1. Create test taxonomy with known categories
2. Run extraction on labeled test documents
3. Compare results with expected classifications
4. Adjust parameters and re-run tests
5. Validate on production documents

### API Integration

**Programmatic Taxonomy Management**:
```javascript
// Create taxonomy via API
const taxonomy = await fetch('/api/v1/taxonomy/', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    name: 'Medical Conditions',
    project_id: 1,
    taxonomy: taxonomyStructure
  })
});

// Generate taxonomy
const generation = await fetch('/api/v1/taxonomy/generate', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    project_id: 1,
    documents: [101, 102, 103],
    generator: 'gemini',
    domain: 'medical'
  })
});
```

**Real-time Updates**:
```javascript
// WebSocket connection for live updates
const ws = new WebSocket('ws://localhost:8000/ws/taxonomy');

ws.onmessage = (event) => {
  const update = JSON.parse(event.data);
  if (update.type === 'extraction_progress') {
    updateProgressBar(update.progress);
  }
};
```

This GUI provides a comprehensive taxonomy management system with intuitive interfaces for creation, editing, generation, and extraction, suitable for both novice users and advanced taxonomy designers.