# Plugin Management User Guide

Compileo's Plugin System allows you to easily extend the platform's capabilities by installing add-ons. This guide explains how to manage plugins via the Graphical User Interface (GUI).

## Accessing the Plugin Manager

1.  Navigate to the **Settings** page from the sidebar.
2.  Click on the **Plugins** tab.

## Installing a Plugin

To install a new plugin:

1.  Locate the **Install Plugin** section in the Plugins tab.
2.  Click **Browse files** or drag and drop your plugin `.zip` file into the upload area.
3.  Click the **Install** button.
4.  The system will upload, verify, and install the plugin. A success message will appear upon completion.

**Note:** Some plugins may perform automated setup tasks during installation (e.g., downloading necessary tools or drivers). This may take a few moments. Ensure you only install plugins from trusted sources.

## Managing Installed Plugins

The **Installed Plugins** section lists all currently active plugins on your system. For each plugin, you can see:

*   **Name & Version**: The display name and version number.
*   **Description**: A brief summary of what the plugin does.
*   **Author**: The creator of the plugin.
*   **Details**: Technical details like the plugin ID and entry point.

## Uninstalling a Plugin

To remove a plugin:

1.  Find the plugin you wish to remove in the **Installed Plugins** list.
2.  Expand the plugin details if necessary.
3.  Click the **🗑️ Uninstall** button.
4.  Confirm the action if prompted. The plugin will be immediately removed from the system.

## Example: Anki Dataset Generator

A reference plugin, "Anki Dataset Exporter", is available to add Anki flashcard export capabilities to the Dataset Generation module.

1.  Install the `compileo-anki-plugin.zip`.
2.  Go to the **Dataset Generation** page.
3.  In the "Output Format" dropdown, you will now see an option for **anki**.
4.  Select your desired **Generation Mode** (Instruction Following, Question and Answer, Summarization, etc.).
5.  Generating a dataset with this format will produce a semicolon-separated text file (`.txt` extension) compatible with Anki import.

**Generation Mode Support:**
- **Instruction Following**: Creates flashcards with instructions on the front and responses on the back
- **Question and Answer**: Traditional Q&A format with questions on the front and answers on the back
- **Summarization**: Content summaries on the front with key points on the back
- **Other Modes**: Automatically adapts to any generation mode supported by Compileo

**File Format:**
- Output: `dataset_[job_id]_extract.txt`
- Format: `question;answer` (semicolon-separated)
- Compatible with Anki's built-in import feature
- HTML formatting supported for rich text display

## Example: Scrapy-Playwright Website Scraper

The "Scrapy-Playwright Website Scraper" plugin enables extracting content from dynamic websites that require JavaScript rendering.

1.  **Install**: Upload and install the `compileo-scrapy-playwright-scraper.zip`. The plugin will automatically install browser dependencies (`playwright install`) during setup.
2.  **Usage (CLI)**: Use the new command provided by the plugin:
    ```bash
    compileo scrape-website --url "https://example.com" --project-id 1 --depth 2
    ```
3.  **Usage (API)**: Send a POST request to `/api/v1/plugins/scrapy-playwright/scrape` with the URL and configuration.
4.  **Usage (GUI)**: In the **Document Processing** tab, you can use the "Scrape Website" feature to ingest content directly from URLs.