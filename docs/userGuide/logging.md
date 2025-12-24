# Logging System in Compileo

Compileo provides a project-wide, persistent logging system that allows you to control the verbosity of log output across the CLI, API, and GUI. This is particularly useful for developers troubleshooting complex AI processing workflows or for production environments where minimal output is desired.

## Log Levels

The system supports three basic log levels:

1.  **none**: Disables all logging output. No logs will be printed to the console or captured by the middleware.
2.  **error**: Only logs critical errors and exceptions. This is the recommended setting for standard production usage.
3.  **debug**: Enables extensive log reporting for developers. This includes detailed internal process information, request/response headers, and JSON-structured debug data for AI model interactions.

## Controlling the Log Level

You can control the global log level through three primary interfaces:

### 1. Web GUI (Settings Page)

The easiest way to change the log level is through the Compileo GUI:
1.  Open the Compileo GUI in your browser.
2.  Navigate to the **Settings** tab.
3.  In the **General** settings section, locate the **Log Level** dropdown.
4.  Select your desired level (**none**, **error**, or **debug**).
5.  Click **Save Settings**. The change will be persisted in the database and applied immediately to the API server and all background workers.

### 2. REST API

You can programmatically manage global settings, including the log level, via the settings API endpoint:

*   **GET `/api/v1/settings/`**: Retrieves the current global settings, including the current log level.
*   **POST `/api/v1/settings/`**: Updates global settings.
    *   **Payload Example**:
        ```json
        {
          "log_level": "debug"
        }
        ```

### 3. CLI Parameters

The Compileo CLI supports a global `--log-level` option that allows you to override or set the persistent log level for the duration of the command:

```bash
# Set log level to debug for a specific command
compileo --log-level debug projects list

# The CLI also updates the persistent database setting when this flag is used
```

## Internal Implementation

The logging system is built on top of the standard Python `logging` module but is centralized through a unified utility in `src/compileo/core/logging.py`. 

### Synchronization
Because Compileo uses multiple processes (API server, RQ workers, CLI), the log level is stored in a shared SQLite database (`gui_settings` table). All components call `setup_logging()` during their initialization lifecycle to ensure they inherit the current global setting.

### Structured Logging
For developers using the `debug` level, many internal components emit structured JSON debug messages. These are captured as `DEBUG` level log entries and contain valuable context such as timestamps, component names, and raw AI model inputs/outputs.
