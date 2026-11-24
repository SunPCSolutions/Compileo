# Plugin Management CLI

The Compileo CLI provides a comprehensive set of commands for managing plugins from the terminal.

## Command Group: `plugins`

All plugin-related commands are grouped under `compileo plugins`.

**Note:** Plugins may also extend the CLI with their own top-level commands. For example, the `compileo-scrapy-playwright-scraper` plugin adds a `compileo scrape-website` command. These commands become available automatically after installing the plugin.

## Example Plugin CLI: Embeddings

The Qdrant embedding plugin adds an `embeddings` command group.

### Configure embedding settings

Command:

```bash
compileo embeddings settings [OPTIONS]
```

Options:

- `--owner-id` (required)
- `--qdrant-url`
- `--collection`
- `--ollama-url`
- `--ollama-model`
- `--distance`
- `--batch-size`
- `--include-taxonomy / --no-include-taxonomy`
- `--taxonomy-mode`
- `--taxonomy-generator`
- `--taxonomy-processing-mode`
- `--taxonomy-anchor-main-topic / --no-taxonomy-anchor-main-topic`
- `--taxonomy-chunk-batch-size`
- `--segment-token-size`
- `--segment-overlap-tokens`
- `--skip-segmentation / --no-skip-segmentation`
- `--preprocess-policy`

Example:

```bash
compileo embeddings settings \
  --owner-id <owner_id> \
  --qdrant-url http://localhost:6333 \
  --collection compileo_chunks \
  --ollama-url http://localhost:11434 \
  --ollama-model nomic-embed-text \
  --distance Cosine \
  --batch-size 128 \
  --include-taxonomy \
  --taxonomy-mode chunk \
  --taxonomy-generator gemini \
  --taxonomy-processing-mode fast \
  --taxonomy-anchor-main-topic \
  --taxonomy-chunk-batch-size 10 \
  --segment-token-size 512 \
  --segment-overlap-tokens 80 \
  --no-skip-segmentation \
  --preprocess-policy markdown_preserve_headings
```

### Submit embedding index helper

Command:

```bash
compileo embeddings index [OPTIONS]
```

Options:

- `--owner-id` (required)
- `--project-id` (required)
- `--document-id` (repeatable)
- `--taxonomy-anchor-main-topic / --no-taxonomy-anchor-main-topic`
- `--taxonomy-chunk-batch-size`
- `--segment-token-size`
- `--segment-overlap-tokens`
- `--skip-segmentation / --no-skip-segmentation`
- `--dry-run`

Example:

```bash
compileo embeddings index \
  --owner-id <owner_id> \
  --project-id <project_id> \
  --document-id <doc_id_1> \
  --document-id <doc_id_2> \
  --taxonomy-anchor-main-topic \
  --taxonomy-chunk-batch-size 10 \
  --segment-token-size 512 \
  --segment-overlap-tokens 80 \
  --skip-segmentation \
  --dry-run
```

Notes:

- `--segment-token-size` and `--segment-overlap-tokens` control embedding micro-segmentation for long chunks.
- `--skip-segmentation` bypasses extra splitting and embeds each selected chunk as-is.
- Recommended starting values for textbook QA RAG: `512` and `80`.
- `--taxonomy-chunk-batch-size` controls both taxonomy batch grouping and in-batch concurrency for chunk taxonomy generation (for full batches), while taxonomy output remains chunk-specific (not batch-wide).

For complete behavior and RAG tuning guidance, see [Embeddings Plugin User Guide](embeddings.md).

### List Plugins

List all currently installed plugins.

```bash
compileo plugins list
```

**Options:**
*   `--format [table|json]`: Output format (default: table).

**Example Output:**
```text
Installed Plugins (1):
--------------------------------------------------------------------------------
ID                        Name                      Version    Author
--------------------------------------------------------------------------------
compileo-anki-plugin      Anki Dataset Exporter     1.0.0      Compileo Team
```

### Install Plugin

Install a new plugin from a zip file.

```bash
compileo plugins install <path_to_plugin_zip>
```

**Arguments:**
*   `plugin_file`: Path to the plugin `.zip` file.

**Example:**
```bash
compileo plugins install ./compileo-anki-plugin.zip
```

### Uninstall Plugin

Remove an installed plugin.

```bash
compileo plugins uninstall <plugin_id>
```

**Arguments:**
*   `plugin_id`: The unique ID of the plugin to uninstall.

**Options:**
*   `--confirm`: Skip confirmation prompt.

**Example:**
```bash
compileo plugins uninstall compileo-anki-plugin
