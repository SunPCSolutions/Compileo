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

You can programmatically manage the log level via the dedicated API endpoint:

*   **GET `/api/v1/settings/log-level`**: Retrieves the current log level.
    *   **Response Example**:
        ```json
        {
          "log_level": "debug"
        }
        ```
*   **POST `/api/v1/settings/log-level`**: Updates the log level.
    *   **Payload Example**:
        ```json
        {
          "level": "debug"
        }
        ```
    *   **Response Example**:
        ```json
        {
          "message": "Log level updated to debug"
        }
        ```

### 3. CLI Commands

The Compileo CLI provides a dedicated `config` command group for managing settings:

```bash
# Show current log level
compileo config log-level --show

# Set log level to debug
compileo config log-level --level debug

# Set log level to error
compileo config log-level --level error

# Set log level to none (minimal logging)
compileo config log-level --level none
```

**Note:** Changes made via the CLI are persisted to the database and will affect all running components (API server and workers).

## Internal Implementation

The logging system is built on top of the standard Python `logging` module but is centralized through a unified utility in `src/compileo/core/logging.py`.

### Synchronization
Because Compileo uses multiple processes (API server, RQ workers, CLI), the log level is stored in a shared SQLite database (`gui_settings` table). All components call `setup_logging()` during their initialization lifecycle to ensure they inherit the current global setting.

### Docker Support
In Docker deployments, the centralized logging system ensures that all containers (API and workers) respect the same log level setting. When you change the log level via the GUI, API, or CLI, it is persisted to the database and affects:
- The API server container
- All RQ worker containers
- Future CLI commands

### Structured Logging
For developers using the `debug` level, many internal components emit structured JSON debug messages. These are captured as `DEBUG` level log entries and contain valuable context such as timestamps, component names, and raw AI model inputs/outputs.
