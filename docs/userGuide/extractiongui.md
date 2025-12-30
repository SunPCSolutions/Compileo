# Advanced Entity Extraction & Dataset Generation in Compileo GUI

## Overview

The Compileo GUI provides a sophisticated, unified interface for advanced entity extraction, relationship inference, and automated Q&A dataset generation. The extraction system transforms unstructured text into structured datasets through AI-powered entity recognition and intelligent relationship discovery.

<div align="center">
  <a href="../img/Extraction_main.png"><img src="../img/Extraction_main.png" width="400" alt="Multi-Stage Entity Extraction"></a>
  <p><i>Multi-Stage Entity Extraction Interface</i></p>
</div>

## Key Capabilities

### Entity Extraction
- **Dual Extraction Modes**: Supports both Named Entity Recognition (NER) and Whole Text Extraction.
  - **NER**: Extracts specific entities (names, terms, concepts) from text chunks.
  - **Whole Text**: Extracts complete relevant text portions classified into taxonomy categories.
- **High-Precision Validation**: Strict subtractive validation stage that programmatically filters out hallucinations and discovery errors.
- **Snippet Deduplication**: Programmatic deduplication of extracted segments to ensure unique and clean results.
- **Flexible Extraction Modes**: Choose between Contextual and Document-Wide processing.
  - **Contextual Extraction**: Only extracts from child categories when parent context is present in the text, preventing false positives.
  - **Document-Wide Extraction**: Processes all chunks for selected categories regardless of contextual relevance, maximizing coverage.
- **AI-Powered Recognition**: Extract specific entities from text using Grok, Gemini, or Ollama models
- **Taxonomy Integration**: Work with hierarchical taxonomies for precise categorization
- **Multi-Category Support**: Extract entities across multiple taxonomy categories simultaneously
- **Confidence Scoring**: Quality assessment for all extracted entities

### Relationship Inference
- **Automatic Discovery**: Identify relationships between co-occurring entities
- **Domain Agnostic**: Works across medical, business, legal, and technical domains
- **Confidence Weighting**: Quality scoring for relationship strength

### Q&A Dataset Generation
- **Template-Based Creation**: Generate question-answer pairs from entity relationships
- **Multiple Formats**: Export in JSONL, JSON, and CSV formats
- **Context Preservation**: Maintain source relationships and metadata

## Accessing Extraction

1. **Navigate to the Application**: Open Compileo in your web browser
2. **Select Extraction**: Click on "🔍 Extraction" in the sidebar
3. **Choose Operation**: Use the three-tab interface for different workflows

## Interface Components

### Three-Tab Structure

The extraction interface is organized into three main tabs:

#### 🏃 **Run Extraction Tab**

##### Project & Taxonomy Selection
- **Project Selector**: Choose from available projects in "Project Name (ID: 123)" format
- **Taxonomy Selector**: Pick the taxonomy to use for entity categorization

##### AI Model Configuration
- **Extraction Type**: Select the type of extraction: `Named Entity Recognition (NER)` or `Whole Text Extraction`.
- **Extraction Mode**: Choose the extraction mode: `Contextual Extraction` (default) or `Document-Wide Extraction`.
- **Primary Classifier**: Select the AI model (Grok, Gemini, Ollama) for the initial classification stage.
- **Validation Stage**: Optionally enable a second AI model for result validation. This stage is **subtractive**, meaning it verifies existing findings and removes errors but is strictly forbidden from discovering new categories.
- **Validation Classifier**: Choose a different AI model for the validation stage if enabled.

##### Taxonomy Tree Selection
- **Interactive Tree**: Expandable taxonomy hierarchy with checkboxes
- **Search Functionality**: Filter categories by name or description
- **Batch Controls**: Expand/collapse all, select/deselect all visible categories
- **Selection Summary**: Shows count of selected categories for extraction

##### Extraction Parameters
- **Depth Control**: How deep in taxonomy hierarchy to extract (1-5 levels)
- **Confidence Threshold**: Minimum confidence score for results (0.0-1.0)
- **Batch Size**: Number of chunks to process per batch (1-100)
- **Max Chunks**: Maximum chunks to process (1-10,000)

##### Job Initiation
- **Start Extraction**: Launch extraction job with configured parameters. The interface will provide in-view synchronous monitoring, allowing you to follow the job's progress through various stages without leaving the page.
- **Progress Feedback**: Real-time stage-based updates (e.g., "Initializing", "Analyzing Chunks", "Storing Results") ensure you know exactly what the system is doing.

#### 📊 **Monitor Jobs Tab**

##### Project Selection
- **Project Filter**: View jobs for specific projects

##### Job Dashboard
- **Status Overview**: Summary metrics for total, running, completed, and failed jobs
- **Job Cards**: Individual cards for each extraction job with:
  - **Status Indicators**: Color-coded status badges and progress bars
  - **Timing Information**: Created, started, completed timestamps
  - **Expandable Details**: Parameters, error messages, and timing breakdowns

##### Job Management Actions
- **Restart Failed Jobs**: Green "Restart" button for failed/cancelled jobs
- **Cancel Running Jobs**: Red "Cancel" button for pending/running jobs
- **View Completed Results**: Blue "View Results" button for completed jobs

#### 📋 **Browse & Manage Extractions Tab**

##### Enhanced Management Interface
- **Search & Filter Controls**: Find specific extraction jobs by name, status, or date range
- **Bulk Selection**: Select multiple jobs for batch operations
- **Advanced Filtering**: Filter by project, taxonomy, AI model, or extraction type
- **Sorting Options**: Sort by creation date, completion time, or status

##### Job Management Actions
- **View Results**: Access detailed extraction results for completed jobs
- **Delete Jobs**: Permanently remove extraction jobs and associated data
- **Restart Failed Jobs**: Retry failed extractions with the same parameters
- **Cancel Running Jobs**: Stop active extraction processes

##### Results Organization
- **Optimized Data Structure**: Results are deduplicated and optimized for downstream processing, focusing on unique text snippets and entities.
- **Category-Based Display**: Results grouped by taxonomy categories
- **Expandable Sections**: Each category shows extracted entities with:
  - **Entity Frequency**: How many chunks contain each entity
  - **Confidence Scores**: Quality indicators for extractions
  - **Source References**: Links to original text chunks

##### Relationship Analysis
- **Relationship Summary**: Overview of discovered entity relationships
- **Type Distribution**: Breakdown by relationship types (associative, causal, etc.)
- **Confidence Metrics**: Quality assessment of relationship inferences

##### Q&A Dataset Generation
- **One-Click Generation**: Automatic Q&A pair creation from relationships
- **Sample Preview**: View generated questions and answers before export
- **Export Options**: Download in JSONL, JSON, or CSV formats
- **Statistics Display**: Generation metrics and quality indicators

## Job Status Types

### Pending
- **Appearance**: Status shows "Pending", progress at 0%
- **Actions Available**: Cancel
- **Description**: Job is queued and waiting for processing resources

### Running
- **Appearance**: Status shows "Running", progress bar updates in real-time
- **Actions Available**: Cancel
- **Description**: Job is actively being processed with live progress updates

### Completed
- **Appearance**: Status shows "Completed", progress at 100%
- **Actions Available**: View Results
- **Description**: Job finished successfully with results available

### Failed
- **Appearance**: Status shows "Failed", progress bar shows last completed percentage
- **Actions Available**: Restart
- **Description**: Job encountered an error and stopped

### Cancelled
- **Appearance**: Status shows "Cancelled", progress shows completion at cancellation time
- **Actions Available**: Restart
- **Description**: Job was manually stopped by user

## Error Handling

The GUI provides intelligent error handling with user-friendly messages:

### Common Error Types

- **🔌 Connection Error**: Network issues preventing communication with the server
- **⏱️ Timeout Error**: Operations taking longer than expected
- **🔐 Authentication Error**: Session expired, requiring re-login
- **❌ Not Found**: Requested job or resource doesn't exist
- **🐌 Rate Limit Exceeded**: Too many requests, need to wait

### Specific Extraction Errors

- **📂 Taxonomy not found**: Selected taxonomy doesn't exist
- **🤖 AI model unavailable**: Selected AI service issues
- **🗄️ Database connection failed**: Backend database problems
- **🐌 API rate limit exceeded**: External AI service limits reached
- **🔍 Entity extraction failed**: AI parsing or response issues

## Workflow Examples

### Complete Entity Extraction Workflow

1. **Setup Project & Taxonomy**:
   - Select project containing your documents
   - Choose appropriate taxonomy for entity categorization

2. **Configure AI Models**:
   - Select primary AI model (Grok recommended for accuracy)
   - Optionally enable validation with different AI model
   - Adjust confidence threshold based on use case

3. **Select Categories**:
   - Use taxonomy tree to select relevant categories
   - Search and filter categories as needed
   - Review selection summary before proceeding

4. **Run Extraction**:
   - Set extraction parameters (depth, batch size, max chunks)
   - Click "Start Extraction" to launch job
   - Monitor progress in Monitor Jobs tab

5. **Review Results**:
   - Switch to Browse & Manage Extractions tab
   - Explore extracted entities by category
   - Review relationship discoveries
   - Generate Q&A datasets if needed

### Advanced Dataset Generation

1. **Complete Entity Extraction**: Ensure extraction job finishes successfully
2. **Review Relationships**: Check discovered entity associations
3. **Generate Q&A Pairs**: Use one-click generation from relationships
4. **Preview & Export**: Review samples and download in preferred format
5. **Integration**: Use exported datasets in ML training pipelines

## Best Practices

### Entity Extraction Optimization

- **AI Model Selection**: Use Grok for highest accuracy, Gemini for speed
- **Confidence Thresholds**: Start with 0.5, adjust based on domain requirements
- **Category Selection**: Be specific - fewer, well-chosen categories yield better results
- **Batch Size Tuning**: Larger batches for homogeneous content, smaller for diverse content

### Relationship Inference

- **Domain Understanding**: Ensure taxonomy reflects real-world relationships
- **Quality Validation**: Review relationship confidence scores
- **Iterative Refinement**: Adjust taxonomy based on extraction results

### Q&A Dataset Generation

- **Template Customization**: Review generated Q&A pairs for quality
- **Format Selection**: Choose JSONL for most ML frameworks
- **Context Preservation**: Include relationship metadata when possible
- **Quality Assurance**: Manually review samples before large-scale generation

## Integration with Other Features

### Document Processing Pipeline

1. **Upload Documents**: Use Documents section to ingest and chunk files
2. **Create Taxonomy**: Build domain-specific taxonomies for categorization
3. **Run Extraction**: Use unified extraction interface for entity discovery
4. **Generate Datasets**: Create training data from extracted entities and relationships
5. **Quality Analysis**: Review extraction metrics and relationship quality

### Advanced Analytics

1. **Monitor Performance**: Track extraction accuracy across different domains
2. **Relationship Mining**: Discover patterns in entity associations
3. **Dataset Quality**: Assess generated Q&A pair quality and diversity
4. **Model Improvement**: Use extraction results to refine AI prompts and taxonomies

## Troubleshooting

### No Entities Found

- **Check AI Model**: Ensure selected AI model has valid API keys
- **Verify Taxonomy**: Confirm taxonomy categories are relevant to content
- **Adjust Confidence**: Lower confidence threshold if too restrictive
- **Review Content**: Ensure documents contain extractable entities

### Poor Relationship Quality

- **Taxonomy Refinement**: Improve category definitions and relationships
- **Content Quality**: Ensure documents have clear entity associations
- **AI Model Selection**: Try different models for better relationship inference

### Q&A Generation Issues

- **Relationship Quality**: Ensure high-confidence relationships exist
- **Template Relevance**: Customize templates for your domain
- **Content Coverage**: Verify sufficient entity pairs for generation

## Advanced Features

### Multi-Model Validation

- **Cross-Model Agreement**: Use different AI models for validation
- **Confidence Boosting**: Higher confidence for agreed-upon extractions
- **Error Detection**: Identify inconsistent AI responses

### Batch Processing

- **Large-Scale Operations**: Process thousands of documents efficiently
- **Progress Monitoring**: Real-time updates for long-running jobs
- **Resource Management**: Automatic load balancing across available resources

### Custom Templates

- **Domain-Specific Q&A**: Create templates for specialized domains
- **Question Variety**: Generate multiple question types from same relationships
- **Context Enhancement**: Include domain-specific context in generated pairs

This comprehensive GUI transforms complex entity extraction and relationship inference into an accessible, powerful tool for creating high-quality structured datasets from unstructured text. The unified interface eliminates the need for technical expertise while providing advanced capabilities for expert users.

## Job Status Types

### Pending
- **Appearance**: Status shows "Pending", progress at 0%
- **Actions Available**: Cancel
- **Description**: Job is queued and waiting for processing resources

### Running
- **Appearance**: Status shows "Running", progress bar updates in real-time
- **Actions Available**: Cancel
- **Description**: Job is actively being processed with live progress updates

### Completed
- **Appearance**: Status shows "Completed", progress at 100%
- **Actions Available**: View Results
- **Description**: Job finished successfully with results available

### Failed
- **Appearance**: Status shows "Failed", progress bar shows last completed percentage
- **Actions Available**: Restart
- **Description**: Job encountered an error and stopped

### Cancelled
- **Appearance**: Status shows "Cancelled", progress shows completion at cancellation time
- **Actions Available**: Restart
- **Description**: Job was manually stopped by user

## Error Handling

The GUI provides intelligent error handling with user-friendly messages:

### Common Error Types

- **🔌 Connection Error**: Network issues preventing communication with the server
- **⏱️ Timeout Error**: Operations taking longer than expected
- **🔐 Authentication Error**: Session expired, requiring re-login
- **❌ Not Found**: Requested job or resource doesn't exist
- **🐌 Rate Limit Exceeded**: Too many requests, need to wait

### Specific Extraction Errors

- **📂 Taxonomy not found**: Selected taxonomy doesn't exist
- **🤖 Classification service unavailable**: AI service issues
- **🗄️ Database connection failed**: Backend database problems
- **🐌 API rate limit exceeded**: External service limits reached

## Workflow Examples

### Monitoring Active Extractions

1. **Select Project**: Choose the project with active extraction jobs
2. **View Dashboard**: See all jobs with their current status and progress
3. **Monitor Progress**: Watch progress bars update in real-time for running jobs
4. **Check Details**: Expand job cards to see parameters and timing information

### Handling Failed Jobs

1. **Identify Failed Jobs**: Look for jobs with "Failed" status
2. **Review Error Details**: Expand the job card to see specific error messages
3. **Assess Feasibility**: Determine if the error is recoverable
4. **Restart if Appropriate**: Click "Restart" button for transient failures
5. **Monitor Restart**: Watch the restarted job progress through the dashboard

### Managing Long-Running Jobs

1. **Identify Long Jobs**: Look for jobs running for extended periods
2. **Check Progress**: Monitor if progress is still advancing
3. **Cancel if Stuck**: Use "Cancel" button for jobs that appear stuck
4. **Restart After Review**: Restart cancelled jobs after investigating issues

## Best Practices

### Job Monitoring

- **Regular Checks**: Periodically refresh the page to see latest job status
- **Progress Tracking**: Use progress bars to estimate completion time
- **Error Review**: Always check error details before restarting failed jobs
- **Resource Management**: Cancel unnecessary jobs to free up system resources

### Error Recovery

- **Understand Errors**: Read error messages carefully to identify root causes
- **Retry Strategically**: Only restart jobs where the error appears transient
- **Contact Support**: For persistent errors, gather error details before contacting support
- **Prevent Recurrence**: Note patterns in failures to avoid similar issues

### Performance Optimization

- **Batch Monitoring**: Use the dashboard to monitor multiple jobs simultaneously
- **Priority Management**: Cancel lower-priority jobs if high-priority work is blocked
- **Timing Awareness**: Note typical completion times for different job types
- **Load Balancing**: Distribute large extraction workloads across multiple projects

## Integration with Other Features

### Document Processing Workflow

1. **Upload Documents**: Use the Documents section to upload and process files
2. **Create Taxonomy**: Build or generate taxonomies for content categorization
3. **Start Extraction**: Initiate extraction jobs (typically through API or other interfaces)
4. **Monitor Progress**: Use the Extraction GUI to track job completion
5. **View Results**: Access extraction results through the completed job interface

### Dataset Generation Follow-up

1. **Complete Extractions**: Ensure extraction jobs finish successfully
2. **Generate Datasets**: Use the Dataset section to create training data
3. **Quality Analysis**: Review extraction quality metrics
4. **Iterate**: Restart failed extractions or adjust parameters as needed

## Troubleshooting

### Jobs Not Appearing

- **Check Project Selection**: Ensure correct project is selected
- **Refresh Page**: Use browser refresh to reload job list
- **Verify Permissions**: Ensure you have access to view jobs in the selected project

### Progress Not Updating

- **Check Connection**: Verify internet connectivity
- **Refresh Browser**: Hard refresh (Ctrl+F5) to clear cache
- **Contact Support**: If progress consistently doesn't update

### Action Buttons Not Working

- **Check Job Status**: Buttons only appear for valid state transitions
- **Verify Permissions**: Ensure you have permissions to modify jobs
- **Network Issues**: Check connection and retry

## Advanced Features

### Real-time Updates

The Job Management dashboard provides comprehensive visibility into all active and completed jobs across the platform.

- **Manual Refresh**: Use the "Refresh Dashboard" button for a clean, non-flickering update of all job metrics and lists.
- **In-Place Monitoring**: Jobs started from the extraction tabs are monitored synchronously, keeping you informed of their specific status in real-time.
- **Accurate Progress**: Monitoring focuses on stage-based status messages from the backend, providing a more reliable indicator of work completed than simple percentage bars.

### Bulk Operations

While individual job control is available, consider:

- **Project Organization**: Group related jobs in dedicated projects
- **Batch Processing**: Use API or CLI for bulk job operations
- **Automation**: Set up monitoring scripts for large-scale operations

### Performance Metrics

Track job performance over time:

- **Completion Times**: Note typical duration for different job types
- **Success Rates**: Monitor failure frequency and types
- **Resource Usage**: Observe system impact during peak processing

This GUI provides a user-friendly way to manage the complex asynchronous nature of extraction jobs while maintaining full visibility into the processing pipeline.
