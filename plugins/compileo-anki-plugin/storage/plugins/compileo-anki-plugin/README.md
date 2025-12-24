# Anki Dataset Exporter Plugin

This plugin extends Compileo to export generated datasets into an Anki-compatible text format.

## Features

*   **Custom Output Format**: Adds "anki" as an output format option in the Dataset Generation settings.
*   **Automatic Formatting**: Converts Question/Answer pairs into semicolon-separated values (`Front;Back;Tags`).
*   **HTML Support**: Preserves line breaks by converting them to `<br>` tags.
*   **Tagging**: Automatically includes dataset tags in the export.

## How to use it

### GUI Usage

1.  **Install the Plugin**: Upload the `compileo-anki-plugin.zip` file via the Compileo Settings > Plugins tab.
2.  **Generate a Dataset**:
    *   Go to the **Dataset Generation** page.
    *   In the **Output Format** dropdown, select **anki** (or whatever label the UI displays based on the metadata).
    *   Run the generation job.
3.  **Download**: The generated file will have a `.txt` extension and can be imported directly into Anki.

### API Usage

**Endpoint:** `POST /api/v1/datasets/generate`

**Request Body:** You must include `format_type: "anki"` in your JSON payload.

**Example Request:**

```json
{
  "project_id": 123,
  "format_type": "anki",
  "generation_mode": "question",
  "prompt_name": "default",
  "batch_size": 10
}
```

## Configuration

This plugin does not currently require any additional configuration parameters.

## Import into Anki

1.  Open Anki.
2.  Click **Import File**.
3.  Select the generated `.txt` file.
4.  Ensure the "Field separator" is set to **Semicolon**.
5.  Map the fields:
    *   Field 1 -> Front
    *   Field 2 -> Back
    *   Field 3 -> Tags
6.  Click **Import**.