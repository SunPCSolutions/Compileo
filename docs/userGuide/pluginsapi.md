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