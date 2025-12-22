# Extraction CLI Commands

This guide covers the command-line interface for creating and managing extraction jobs in Compileo.

## Overview

The extraction CLI provides complete parity with the extraction API, allowing users to create extraction jobs through command-line interface with identical functionality and parameters.

## Commands

### `extraction create`

Creates a new extraction job using a taxonomy to extract content from processed documents.

#### Syntax
```bash
compileo extraction create --project-id <project_id> --taxonomy-id <taxonomy_id> --selected-categories "category1,category2" --effective-classifier <model> [options]
```

#### Required Parameters

- `--project-id` (integer): The ID of the project containing the data
- `--taxonomy-id` (integer): The ID of the taxonomy to use for extraction
- `--selected-categories` (string): Comma-separated list of category names to extract
- `--initial-classifier` (string): The AI model for the initial classification stage (e.g., `grok`, `gemini`, `ollama`, `openai`).
- `--extraction-type` (string, default: "ner"): Type of extraction: `'ner'` for named entities, `'whole_text'` for complete text portions.

#### Optional Parameters

- `--extraction-mode` (string, default: "contextual"): Extraction mode: `'contextual'` or `'document-wide'`.
- `--enable-validation-stage`: Enable validation stage for improved accuracy
- `--validation-classifier` (string): Separate classifier for validation phase
- `--confidence-threshold` (float, default: 0.5): Minimum confidence score (0.0-1.0)
- `--max-chunks` (integer): Maximum number of chunks to process

#### Available AI Models

- **Gemini**: `gemini/gemini-1.5-flash`, `gemini/gemini-1.5-pro`
- **Ollama**: `ollama/llama3.1`, `ollama/llama3.2`, `ollama/mistral`
- **Grok**: `grok/grok-1`, `grok/grok-beta`
- **OpenAI**: `openai/gpt-4o`, `openai/gpt-4-turbo`, `openai/gpt-3.5-turbo`

#### Examples

**Basic extraction with Gemini:**
```bash
compileo extraction create \
  --project-id 1 \
  --taxonomy-id 5 \
  --selected-categories "diagnosis,treatment" \
  --effective-classifier "gemini/gemini-1.5-flash"
```

**Advanced extraction with validation:**
```bash
compileo extraction create \
  --project-id 1 \
  --taxonomy-id 5 \
  --selected-categories "symptoms,causes,effects" \
  --effective-classifier "ollama/llama3.1" \
  --enable-validation-stage \
  --validation-classifier "gemini/gemini-1.5-pro" \
  --confidence-threshold 0.7 \
  --max-chunks 1000
```

**Limited processing for testing:**
```bash
compileo extraction create \
  --project-id 1 \
  --taxonomy-id 5 \
  --selected-categories "all" \
  --effective-classifier "gemini/gemini-1.5-flash" \
  --max-chunks 100
```

**Document-wide extraction for maximum coverage:**
```bash
compileo extraction create \
  --project-id 1 \
  --taxonomy-id 5 \
  --selected-categories "diagnosis,treatment" \
  --effective-classifier "ollama/llama3.1" \
  --extraction-mode document-wide \
  --confidence-threshold 0.3
```

#### Output

**Success Response:**
```
🚀 Creating extraction job for project 1
📋 Taxonomy ID: 5
🏷️ Categories: diagnosis, treatment
🤖 Classifier: gemini/gemini-1.5-flash

✅ Extraction job created successfully!
📋 Job ID: 12345
🔍 Monitor progress: compileo jobs poll 12345
📊 Check status: compileo jobs status 12345
```

## Job Monitoring

After creating an extraction job, you can monitor its progress using the job management commands:

```bash
# Check job status
compileo jobs status 12345

# Poll for completion (long-running)
compileo jobs poll 12345

# Stream real-time updates
compileo jobs stream 12345
```

## Error Handling

The CLI provides clear error messages for common issues:

- **Invalid project ID**: "Project with ID X not found"
- **Invalid taxonomy ID**: "Taxonomy with ID Y not found"
- **Invalid categories**: "Category 'invalid' not found in taxonomy"
- **Model unavailable**: "AI model 'invalid/model' is not available"
- **Connection errors**: "Failed to connect to API server"

## Integration with Other Commands

### Complete Workflow Example

```bash
# 1. Create project
compileo projects create --name "Medical Study" --description "Clinical trial data"

# 2. Upload documents
compileo documents upload --project-id 1 --files "study1.pdf,study2.pdf"

# 3. Parse documents
compileo documents parse --project-id 1 --document-ids "1,2"

# 4. Chunk documents
compileo documents chunk --project-id 1 --document-ids "1,2" --chunker "gemini"

# 5. Generate taxonomy
compileo taxonomy generate --project-id 1 --name "Medical Conditions" --documents "1,2"

# 6. Create extraction job
compileo extraction create \
  --project-id 1 \
  --taxonomy-id 1 \
  --selected-categories "diagnosis,treatment,outcome" \
  --effective-classifier "gemini/gemini-1.5-flash"

# 7. Monitor and retrieve results
compileo jobs status <job_id>
compileo extraction results <job_id>
```

## Configuration

The CLI uses the same configuration as the API and GUI:

- **API Keys**: Retrieved from GUI settings database
- **Model Selection**: Based on available AI providers
- **Validation**: Same parameter validation as API endpoints

## Troubleshooting

### Common Issues

**"Model not available"**
- Check that the AI model is properly configured in settings
- Verify API keys are set for the selected provider
- Try a different model from the same provider

**"Taxonomy not found"**
- Verify the taxonomy ID exists for the project
- Check that taxonomy generation completed successfully
- Use `compileo taxonomy list --project-id X` to see available taxonomies

**"No chunks found"**
- Ensure documents have been parsed and chunked
- Check that chunking completed successfully
- Verify project contains processed documents

### Getting Help

```bash
# Show command help
compileo extraction create --help

# List all extraction commands
compileo extraction --help

# General CLI help
compileo --help
```

## API Parity

The CLI provides complete feature parity with the `/api/v1/extraction/` endpoint:

- **Same parameters**: All API parameters are supported
- **Same validation**: Identical input validation rules
- **Same processing**: Uses the same backend extraction pipeline
- **Same results**: Produces identical extraction results

This ensures consistent behavior whether using the API, GUI, or CLI interfaces.