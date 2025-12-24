# Plugin Development Guide

This guide provides step-by-step instructions for developers who want to extend Compileo's functionality by creating plugins.

## Overview

Compileo's plugin system allows you to extend the platform's capabilities without modifying the core codebase. Plugins are self-contained `.zip` packages that include a manifest file and Python source code.

The system is designed around **Extension Points**. Core modules define these points (standard interfaces), and plugins implement them. This allows plugins to provide new functionality—such as custom dataset formats, new data sources, or additional processing steps—depending on which extension points they target.

## 1. Plugin Structure

A valid plugin package is a `.zip` file with the following structure:

```
my-plugin.zip
├── plugin.yaml        # Manifest file (Required)
├── README.md          # Documentation (Recommended)
└── src/               # Source code directory
    ├── __init__.py
    └── my_module.py   # Implementation code
```

## 2. The Manifest File (`plugin.yaml`)

The `plugin.yaml` file is the heart of your plugin. It defines metadata and tells Compileo how to load your code.

### Fields

*   **`id`**: Unique identifier for your plugin (e.g., `my-custom-plugin`). Use lowercase, hyphens allowed.
*   **`name`**: Human-readable name (e.g., "My Custom Plugin").
*   **`version`**: Semantic version string (e.g., "1.0.0").
*   **`author`**: Name of the creator or team.
*   **`description`**: Brief description of what the plugin does.
*   **`entry_point`**: The Python module path to your code (e.g., `src.my_module`).
*   **`extensions`**: A dictionary where keys are **Extension Point IDs** (defined by core modules) and values are mappings of names to your implementation classes.
*   **`format_metadata`**: (Optional) Metadata specific to certain extension types (like formatters).
*   **`install_script`**: (Optional) A shell command string to run after the plugin is installed (e.g., to download external binaries). Runs in the project's virtual environment.
*   **`uninstall_script`**: (Optional) A shell command string to run before the plugin is uninstalled.

### Example `plugin.yaml`

```yaml
id: "compileo-anki-plugin"
name: "Anki Dataset Exporter"
version: "1.0.0"
author: "Compileo Team"
description: "Exports datasets in Anki-compatible text format."
entry_point: "src.anki_formatter"
extensions:
  # Hooking into the dataset generation formatter extension point
  compileo.datasetgen.formatter:
    anki: "AnkiOutputFormatter"
format_metadata:
  anki:
    file_extension: "txt"
    description: "Anki-compatible semicolon-separated text format"
# Example of an install script (e.g., for a scraper plugin)
# install_script: "python -m playwright install"
```

## 3. Implementing Extensions

To extend Compileo, you need to implement the interface defined by a specific Extension Point.

### Available Extension Points

#### 1. Dataset Formatter
*   **ID**: `compileo.datasetgen.formatter`
*   **Purpose**: Define custom output formats for generated datasets.
*   **Interface**: `format(self, dataset_content: Union[str, List[Dict]]) -> str`
*   **Generation Mode Awareness**: Formatters should check the `generation_mode` field in dataset items to adapt output format accordingly (e.g., instruction-response pairs vs Q&A vs summarization).
*   **Data Structure**: Dataset items include fields like `instruction`, `input`, `output`, `question`, `answer`, `summary`, `key_points`, and `generation_mode`.

#### 2. Ingestion Handler
*   **ID**: `compileo.ingestion.handler`
*   **Purpose**: Handle custom input sources (e.g., URLs) for ingestion.
*   **Interface**:
    *   `can_handle(self, input_path: str) -> bool`: Return True if plugin can handle this input.
    *   `process(self, path: str, **kwargs) -> Optional[str]`: Process the input and return extracted text.

#### 3. API Router
*   **ID**: `compileo.api.router`
*   **Purpose**: Expose new API endpoints.
*   **Interface**: The registered object should be a FastAPI `APIRouter` instance or a class with a `router` attribute.

#### 4. CLI Command
*   **ID**: `compileo.cli.command`
*   **Purpose**: Add new commands to the Compileo CLI.
*   **Interface**: The registered object should be a `click.Command` or `click.Group`.

### Example: Dataset Formatter Extension

```python
from typing import List, Dict, Any, Union
import json

class MyCustomFormatter:
    """
    Example implementation for the dataset formatter extension point.
    Supports generation mode awareness for proper field mapping.
    """

    def format(self, dataset_content: Union[str, List[Dict[str, Any]]]) -> str:
        # Handle JSON string input (standard Compileo output)
        if isinstance(dataset_content, str):
            dataset_content = json.loads(dataset_content)

        # Ensure we have a list of dictionaries
        if not isinstance(dataset_content, list):
            raise ValueError("Formatter requires a list of items")

        lines = []

        for item in dataset_content:
            if isinstance(item, dict):
                # Determine generation mode and extract appropriate fields
                generation_mode = item.get("generation_mode", "question_answer")

                if generation_mode == "instruction_following" or item.get("instruction"):
                    # Instruction following mode
                    instruction = item.get("instruction", "")
                    input_text = item.get("input", "")
                    output_text = item.get("output", "")

                    # Format for your specific output requirements
                    front = instruction
                    if input_text:
                        front = f"{instruction}\n\nInput: {input_text}"
                    back = output_text

                elif generation_mode == "summarization" or item.get("summary"):
                    # Summarization mode
                    front = "Summary:"
                    back = item.get("summary", "")

                else:
                    # Default Q&A mode
                    front = item.get("question", "")
                    back = item.get("answer", "")

                # Format fields for your output format
                # ... processing logic ...
                lines.append(f"{front}|{back}")

        return "\n".join(lines)
```

## 4. Packaging Your Plugin

1.  **Organize your files** according to the structure defined in Section 1.
2.  **Dependencies**:
    *   **Option A (Preferred)**: Include a `requirements.txt` file at the root of your plugin zip. The system will attempt to install these dependencies when the plugin is installed.
    *   **Option B (Legacy)**: Vendor dependencies by including their source code inside your `src/` directory.
3.  **Post-Install Scripts**:
    *   If your plugin requires external tools (like browsers for Playwright), define an `install_script` in `plugin.yaml`.
    *   The command will be executed in the project's virtual environment. Use `python -m module ...` to ensure the correct Python interpreter is used.
4.  **Create a zip archive** of your plugin directory. Ensure `plugin.yaml` is at the root of the archive.

**Important**: Do not include a top-level folder in the zip file. When unzipped, `plugin.yaml` should be immediate, not inside `my-plugin-folder/plugin.yaml`.

## 5. Documentation

It is highly recommended to include a `README.md` file in your plugin package.

*   **Description**: Detailed explanation of features.
*   **Usage**: Instructions on how to use the plugin.
*   **Configuration**: If your plugin accepts specific parameters (via naming conventions or other means), document them clearly.
*   **Endpoints**: If your plugin introduces new behaviors that affect API endpoints, describe them here.

## 6. Installation

1.  Navigate to the **Plugins** section in the Compileo GUI.
2.  Click **Upload Plugin**.
3.  Select your `.zip` file.
4.  The system will validate and install the plugin.

Alternatively, you can use the API:
`POST /api/v1/plugins/install` with the file upload.

## 7. Best Practices

*   **Security**: Plugins run with "Trusted Execution" privileges. Validate all inputs carefully. Do not use `eval()` or execute arbitrary system commands.
*   **Robustness**: Your plugin logic might run in a separate process (e.g., RQ worker). Avoid relying on global state that isn't pickle-able.
*   **Subprocesses**: If your plugin spawns subprocesses (e.g., to run a scraper script), ensure you set up the environment (`PYTHONPATH`) correctly so it can import project modules. Be mindful of relative path calculations (e.g., use `../../../` to reach project root from `plugins/my-plugin/src/`).
*   **Metadata**: Use `format_metadata` or similar fields in `plugin.yaml` to provide UI hints if supported by the extension point.
*   **Error Handling**: Raise clear `ValueError` or `TypeError` exceptions if input data is invalid. These will be caught and logged by the core system.