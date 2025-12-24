# Scrapy-Playwright Website Scraper Plugin

This plugin enables Compileo to scrape websites using Scrapy and Playwright, supporting JavaScript-rendered content.

## Features

- Scrapes single or multiple URLs.
- Supports JavaScript rendering via Playwright (headless Chromium).
- Configurable waiting strategies (networkidle, selectors, timeouts).
- Captures full-page screenshots (optional).
- Extracts main content and title.
- Integration with Compileo ingestion pipeline.

## Dependencies

- scrapy
- scrapy-playwright
- crochet
- twisted

These dependencies are automatically installed when the plugin is installed.

## Usage

### GUI
1. Go to **Document Processing** -> **Parse Documents**.
2. Select **Scrape Website** option.
3. Enter URL and select depth (currently depth is single page per URL).
4. Click **Process**.

### API
Use the `/api/v1/plugins/scrapy-playwright/scrape` endpoint.

**Parameters:**
- `url` (string, required): The URL to scrape.
- `depth` (integer, optional): Depth of crawling (default 1).
- `config` (object, optional): Configuration dictionary (e.g., `{"wait_for_load": "networkidle"}`).

### CLI
Use the `compileo scrape-website` command.

**Options:**
- `--url`: The URL to scrape (required).
- `--depth`: Depth of crawling (default 1).
- `--wait-for`: Wait strategy (load, domcontentloaded, networkidle).
- `--project-id`: Project ID to associate the scraped document with.