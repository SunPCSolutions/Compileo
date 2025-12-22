
# Extraction API in Compileo

## Overview

The Compileo Extraction API provides endpoints for performing content extraction jobs on processed documents using taxonomy-based classification. This API enables asynchronous extraction of structured information from documents based on predefined taxonomies with support for both **Named Entity Recognition (NER)** and **Whole Text Extraction** modes.

## Base URL: `/api/v1`

## Key Features

- **Dual Extraction Modes**: 
  - **NER**: Extract specific entities (names, terms, concepts) from text chunks
  - **Whole Text**: Extract complete relevant text portions classified into taxonomy categories
- **Selective Taxonomy-Based Extraction**: Extract content only from user-selected taxonomy categories
- **Multi-Model AI Support**: Choose between Grok, Gemini, Ollama, and OpenAI AI models for extraction
- **Contextual Extraction**: Only extracts from child categories when parent context is present in the text
- **Document-Wide Extraction**: Processes all chunks for selected categories without contextual filtering
- **Relationship Inference**: Automatically discover relationships between co-occurring entities
- **High-Precision Validation**: Strict subtractive validation stage that programmatically filters out hallucinations.
- **Snippet Deduplication**: Programmatic deduplication of extracted segments to ensure unique results.
- **Progress Tracking**: Real-time progress monitoring with detailed step updates
- **Result Organization**: Results organized by chunk with entity/text details and confidence scores. Optimized JSON schema for downstream processing.
- **Job Management**: Full lifecycle management including cancellation and restart

## Contextual Extraction Behavior

The extraction system implements intelligent contextual filtering to ensure accuracy and prevent false positives:

### How Contextual Extraction Works

1. **Parent-Child Relationship Analysis**: The system analyzes taxonomy hierarchies to understand parent-child relationships between categories.

2. **Context Relevance Check**: For each text chunk, the system determines if the parent categories of selected child categories are relevant to the content.

3. **Selective Extraction**: Child categories are only processed for extraction if their parent context is present in the text (explicitly or implied).

4. **Empty Results for Irrelevant Content**: If a parent category is not relevant to a chunk, all its child categories return empty results rather than extracting unrelated content.

### Example Behavior

**Selected Categories**: "Associated Conditions and Prevention", "Diagnosis and Pathophysiology"

**Text Chunk**: "The patient presented with chest pain and shortness of breath. ECG showed ST elevation."

- **Analysis**: The text discusses cardiac symptoms but does not mention "Metabolic Syndrome" (parent of "Associated Conditions") or "Mitral Regurgitation" (parent of "Diagnosis")
- **Result**: Both selected categories return empty results, ensuring no false extractions of cardiac content into metabolic syndrome categories

**Text Chunk**: "Metabolic syndrome patients often develop associated conditions like hypertension and diabetes."

- **Analysis**: The text explicitly discusses "Metabolic Syndrome" and its associated conditions
- **Result**: "Associated Conditions and Prevention" extracts relevant content; "Diagnosis and Pathophysiology" returns empty (no diagnosis content present)

## Extraction Modes

The API supports two extraction modes that control how content is processed:

### 1. Contextual Extraction (Default)
- **Behavior**: Only extracts from child categories when parent context is present in the text
- **Purpose**: Prevents false positives by ensuring child categories are only extracted when parent context is relevant
- **Use Case**: Recommended for most scenarios requiring high precision
- **Trade-off**: May miss valid extractions in edge cases where relevant content doesn't explicitly mention parent topics

### 2. Document-Wide Extraction
- **Behavior**: Processes ALL chunks in the document regardless of content relevance to parent categories
- **Purpose**: Maximizes extraction coverage by attempting extraction on every chunk
- **Use Case**: When you want maximum extraction coverage and are willing to review more results
- **Trade-off**: Higher risk of false positives but potentially more comprehensive results

### Choosing an Extraction Mode

- **Use Contextual Mode** when:
  - You need high-precision results with minimal false positives
  - You're working with well-structured taxonomies
  - You want to avoid reviewing irrelevant extractions

- **Use Document-Wide Mode** when:
  - You want maximum extraction coverage
  - You're willing to manually review and filter results
  - The taxonomy structure may not perfectly match content organization
  - You're doing exploratory extraction to discover all possible content

## API Endpoints

### 1. Create Extraction Job

Submits a new selective extraction job for processing.

- **Endpoint:** `POST /extraction/`
- **Description:** Creates and queues a new extraction job with specified taxonomy and categories.

- **Request Body:**
   ```json
   {
     "taxonomy_id": 123,
     "selected_categories": ["category_id_1", "category_id_2"],
     "parameters": {
       "extraction_depth": 3,
       "confidence_threshold": 0.5,
       "batch_size": 10,
       "max_chunks": 1000
     },
     "initial_classifier": "grok",
     "enable_validation_stage": false,
     "validation_classifier": null,
     "only_validated": false,
     "extraction_type": "ner",
     "extraction_mode": "contextual"
   }
   ```

### 2. Delete Extraction Job

Permanently deletes an extraction job and all associated results from the filesystem and database.

- **Endpoint:** `DELETE /extraction/{job_id}/delete`
- **Description:** Removes the specified extraction job, its results, and cleans up all associated files and database entries.

- **Parameters:**
  - `job_id` (path): The unique identifier of the extraction job to delete

- **Response:**
  - **200 OK**: Job successfully deleted
  - **404 Not Found**: Job not found
  - **500 Internal Server Error**: Deletion failed

- **Example:**
  ```bash
  curl -X DELETE "http://localhost:8000/api/v1/extraction/123/delete"
  ```