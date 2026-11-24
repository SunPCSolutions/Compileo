# Plugin Management API

The Plugin API allows programmatic management of Compileo extensions.

## Endpoints

### List Plugins

Retrieves a list of all installed plugins.

*   **URL**: `/api/v1/plugins/`
*   **Method**: `GET`
*   **Response**: `200 OK`
    ```json
    [
      {
        "id": "compileo-anki-plugin",
        "name": "Anki Dataset Exporter",
        "version": "1.0.0",
        "author": "Compileo Team",
        "description": "Exports datasets in Anki-compatible text format.",
        "entry_point": "src.anki_formatter",
        "extensions": {
          "compileo.datasetgen.formatter": {
            "anki": "AnkiOutputFormatter"
          }
        }
      }
    ]
    ```

### Upload Plugin

Uploads and installs a plugin from a `.zip` file.

*   **URL**: `/api/v1/plugins/upload`
*   **Method**: `POST`
*   **Content-Type**: `multipart/form-data`
*   **Parameters**:
    *   `file`: The plugin `.zip` file.
*   **Response**: `200 OK`
    ```json
    {
      "status": "success",
      "message": "Plugin compileo-anki-plugin installed successfully",
      "plugin_id": "compileo-anki-plugin"
    }
    ```

### Get Dataset Formats

Retrieves all available dataset output formats, including built-in formats and plugin-provided formats.

*   **URL**: `/api/v1/plugins/dataset-formats`
*   **Method**: `GET`
*   **Response**: `200 OK`
    ```json
    {
      "formats": [
        {
          "id": "jsonl",
          "name": "JSON Lines",
          "description": "JSON Lines format for dataset entries",
          "file_extension": "jsonl",
          "category": "built-in"
        },
        {
          "id": "parquet",
          "name": "Apache Parquet",
          "description": "Columnar storage format for datasets",
          "file_extension": "parquet",
          "category": "built-in"
        },
        {
          "id": "anki",
          "name": "Anki Flashcards",
          "description": "Anki-compatible semicolon-separated text format",
          "file_extension": "txt",
          "category": "plugin",
          "plugin_id": "compileo-anki-plugin"
        }
      ]
    }
    ```

### Uninstall Plugin

Uninstalls a plugin by ID.

*   **URL**: `/api/v1/plugins/{plugin_id}`
*   **Method**: `DELETE`
*   **Parameters**:
    *   `plugin_id`: The ID of the plugin to uninstall.
*   **Response**: `200 OK`
    ```json
    {
      "status": "success",
      "message": "Plugin compileo-anki-plugin uninstalled"
    }

## Plugin-Specific APIs

Plugins can also register their own API routers, which become available under `/api/v1/plugins/{plugin-id}/`.

### Example: Qdrant Embeddings Plugin

Base path: `/api/v1/plugins/embeddings`

Endpoints:

- `GET /api/v1/plugins/embeddings/settings`
- `POST /api/v1/plugins/embeddings/settings`
- `GET /api/v1/plugins/embeddings/scope?project_id=<project_id>`
- `POST /api/v1/plugins/embeddings/index/preview`
- `POST /api/v1/plugins/embeddings/index`

#### 1) Get settings

`GET /api/v1/plugins/embeddings/settings`

Returns current plugin settings (Qdrant/Ollama/taxonomy/segmentation).

#### 2) Save settings

`POST /api/v1/plugins/embeddings/settings`

Request body fields:

- `qdrant_url: string`
- `collection: string`
- `ollama_url: string`
- `ollama_model: string`
- `distance: string` (`Cosine`, `Dot`, `Euclid`)
- `batch_size: integer`
- `include_taxonomy: boolean`
- `taxonomy_mode: string` (`chunk`, `legacy`)
- `preprocess_policy: string`
- `taxonomy_generator: string` (`gemini`, `grok`, `ollama`, `openai`)
- `taxonomy_processing_mode: string` (`fast`, `complete`)
- `taxonomy_anchor_main_topic: boolean`
- `taxonomy_chunk_batch_size: integer` (1..256, default 10)
- `segment_token_size: integer`
- `segment_overlap_tokens: integer`
- `skip_segmentation: boolean`

#### 3) Scope discovery

`GET /api/v1/plugins/embeddings/scope?project_id=<project_id>`

Returns project list, project documents/chunk counts, taxonomy availability, and detected Ollama models.

#### 4) Preview indexing payload

`POST /api/v1/plugins/embeddings/index/preview`

Use this endpoint to validate payloads without submitting an async job.

Supported request fields:

- `project_id`
- `document_id`
- `document_ids[]`
- `chunk_ids[]`
- `include_taxonomy`
- `taxonomy_mode`
- `auto_generate_taxonomy`
- `taxonomy_generator`
- `taxonomy_processing_mode`
- `taxonomy_anchor_main_topic`
- `taxonomy_chunk_batch_size`
- `collection`
- `model`
- `segment_token_size`
- `segment_overlap_tokens`
- `skip_segmentation`
- `dry_run`

Response includes:

- `preview` (chunk-level)
- `preview_segments` (micro-segment-level)
- `preview_count`
- `preview_segment_count`
- `segmented_parent_count`
- `segment_token_size`
- `segment_overlap_tokens`
- `skip_segmentation`

#### 5) Submit indexing job

`POST /api/v1/plugins/embeddings/index`

Submits an async embedding/indexing task (`executor_key = qdrant_embed_index`).

`project_id` is required.

Comprehensive request example:

```json
{
  "project_id": "09694174-b265-4900-936b-debabf85ef82",
  "document_ids": ["doc-a", "doc-b"],
  "include_taxonomy": true,
  "taxonomy_mode": "chunk",
  "auto_generate_taxonomy": true,
  "taxonomy_generator": "gemini",
  "taxonomy_processing_mode": "fast",
  "taxonomy_anchor_main_topic": true,
  "taxonomy_chunk_batch_size": 10,
  "collection": "compileo_chunks",
  "model": "nomic-embed-text",
  "segment_token_size": 512,
  "segment_overlap_tokens": 80,
  "skip_segmentation": false,
  "dry_run": true
}
```

Notes:
- `taxonomy_mode = chunk` uses embedding-specific chunk taxonomy cache and can materialize a project taxonomy record for UI visibility.
- Payload taxonomy contract is compact (`tree` + `leaf_ids`).
- Micro-segmentation improves retrieval granularity for long chunk content.
- `skip_segmentation=true` bypasses additional splitting and embeds each selected chunk as one segment.
- `taxonomy_chunk_batch_size` controls both chunk grouping and in-batch concurrency for chunk-mode taxonomy generation (e.g., `50` means a full batch can run up to 50 concurrent chunk taxonomy calls), while taxonomy output remains chunk-specific for each chunk.

### Example: Scrapy-Playwright Scraper

*   **Scrape Website**: `POST /api/v1/plugins/scrapy-playwright/scrape`
    *   **Body**:
        ```json
        {
          "url": "https://example.com",
          "depth": 1,
          "project_id": "1",
          "document_id": 123,  // Optional: Attach to existing document
          "wait_for": "networkidle", // Optional: load, domcontentloaded, networkidle
          "scroll_to_bottom": true   // Optional: Scroll to load dynamic content
        }
        ```
*   **Get Configuration**: `GET /api/v1/plugins/scrapy-playwright/config`
