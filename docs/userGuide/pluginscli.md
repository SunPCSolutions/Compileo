# Plugin Management CLI

The Compileo CLI provides a comprehensive set of commands for managing plugins from the terminal.

## Command Group: `plugins`

All plugin-related commands are grouped under `compileo plugins`.

**Note:** Plugins may also extend the CLI with their own top-level commands. For example, the `compileo-scrapy-playwright-scraper` plugin adds a `compileo scrape-website` command. These commands become available automatically after installing the plugin.

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