"""
Structured error logging utilities for the extraction system.

This module provides utilities for consistent, structured error logging
across all extraction components with context information.
"""

import logging
import sys
import traceback
from typing import Any, Dict, Optional
from datetime import datetime

from .exceptions import ExtractionError
from ...core.logging import get_logger


def _sanitize_message(text: str) -> str:
    """
    Sanitize message text to prevent logging conflicts.

    Replaces 'message' with 'msg' to avoid LogRecord attribute conflicts.
    Also escapes quotes for JSON safety.

    Args:
        text: Text to sanitize

    Returns:
        Sanitized text
    """
    if not isinstance(text, str):
        text = str(text)
    # Replace 'message' with 'msg' to avoid conflicts
    text = text.replace('message', 'msg')
    # Escape quotes for JSON safety
    text = text.replace('"', '\\"').replace("'", "\\'")
    return text

def _sanitize_context(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize context dictionary to prevent LogRecord attribute conflicts.

    Replaces reserved logging attribute names in keys with safe alternatives.

    Args:
        context: Context dictionary to sanitize

    Returns:
        Sanitized context dictionary
    """
    if not context:
        return {}

    sanitized = {}
    reserved_keys = {
        'message': 'context_message',
        'msg': 'context_msg',
        'name': 'component_name',
        'level': 'log_level',
        'levelname': 'level_name',
        'levelno': 'level_number',
        'pathname': 'file_path',
        'filename': 'file_name',
        'module': 'module_name',
        'exc_info': 'exception_info',
        'exc_text': 'exception_text',
        'stack_info': 'stack_information',
        'lineno': 'line_number',
        'funcName': 'function_name',
        'created': 'timestamp_created',
        'msecs': 'milliseconds',
        'relativeCreated': 'relative_created',
        'thread': 'thread_id',
        'threadName': 'thread_name',
        'processName': 'process_name',
        'process': 'process_id'
    }

    for key, value in context.items():
        # Replace reserved keys
        safe_key = reserved_keys.get(key, key)
        # Sanitize the value as well
        sanitized[safe_key] = _sanitize_message(value)

    return sanitized



class ErrorLogger:
    """
    Structured error logger for extraction operations.

    Provides consistent logging format with context information,
    error categorization, and performance metrics.
    """

    def __init__(self, name: str, logger: Optional[logging.Logger] = None):
        """
        Initialize error logger.

        Args:
            name: Component name for logging
            logger: Optional custom logger instance
        """
        self.name = name
        self.logger = logger or get_logger(f"extraction.{name}")

    def _setup_structured_logging(self):
        """Set up structured logging with JSON format."""
        # Check if handler already exists to prevent duplicate logs
        if self.logger.handlers:
            return

        handler = logging.StreamHandler(sys.stdout)
        # Use a formatter that ensures the message is properly escaped JSON
        class JsonFormatter(logging.Formatter):
            def format(self, record):
                # Sanitize message to ensure it's safe for JSON string inclusion
                # The _sanitize_message function handles this, but here we double check
                msg = record.msg
                if isinstance(msg, str):
                    msg = msg.replace('"', '\\"').replace('\n', '\\n')
                
                # Format extra fields
                extra_fields = {}
                for key, value in record.__dict__.items():
                    if key not in ['args', 'asctime', 'created', 'exc_info', 'exc_text', 'filename',
                                  'funcName', 'levelname', 'levelno', 'lineno', 'module',
                                  'msecs', 'message', 'msg', 'name', 'pathname', 'process',
                                  'processName', 'relativeCreated', 'stack_info', 'thread', 'threadName']:
                        extra_fields[key] = value
                
                extra_json = ""
                if extra_fields:
                    import json
                    try:
                        extra_str = json.dumps(extra_fields)[1:-1] # Remove braces
                        extra_json = f", {extra_str}"
                    except:
                        pass

                return f'{{"timestamp": "{self.formatTime(record)}", "level": "{record.levelname}", "component": "{record.name}", "message": "{msg}"{extra_json}}}'

        handler.setFormatter(JsonFormatter())
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        # Prevent propagation to avoid double logging if root logger is configured
        self.logger.propagate = False

    def log_error(
        self,
        error: Exception,
        operation: str,
        context: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        project_id: Optional[int] = None,
        job_id: Optional[int] = None
    ):
        """
        Log an error with structured context.

        Args:
            error: The exception that occurred
            operation: Description of the operation that failed
            context: Additional context information
            user_id: User ID if applicable
            project_id: Project ID if applicable
            job_id: Job ID if applicable
        """
        error_context = {
            "operation": operation,
            "error_type": type(error).__name__,
            "error_message": _sanitize_message(str(error)),
            "traceback": _sanitize_message(traceback.format_exc())
        }

        if context:
            error_context.update(_sanitize_context(context))

        if user_id:
            error_context["user_id"] = str(user_id)
        if project_id:
            error_context["project_id"] = str(project_id)
        if job_id:
            error_context["job_id"] = str(job_id)

        # Add error details if it's an ExtractionError
        if isinstance(error, ExtractionError):
            error_context["error_details"] = _sanitize_message(str(error.details))

        self.logger.error(
            f"Error in {operation}: {error}",
            extra=error_context
        )

    def log_warning(
        self,
        message: str,
        operation: str,
        context: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        project_id: Optional[int] = None,
        job_id: Optional[int] = None
    ):
        """
        Log a warning with structured context.

        Args:
            message: Warning message
            operation: Description of the operation
            context: Additional context information
            user_id: User ID if applicable
            project_id: Project ID if applicable
            job_id: Job ID if applicable
        """
        warning_context = {
            "operation": operation,
            "warning_message": _sanitize_message(message)
        }

        if context:
            warning_context.update(_sanitize_context(context))

        if user_id:
            warning_context["user_id"] = str(user_id)
        if project_id:
            warning_context["project_id"] = str(project_id)
        if job_id:
            warning_context["job_id"] = str(job_id)

        self.logger.warning(
            f"Warning in {operation}: {message}",
            extra=warning_context
        )

    def log_performance_issue(
        self,
        operation: str,
        duration: float,
        threshold: float,
        context: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        project_id: Optional[int] = None,
        job_id: Optional[int] = None
    ):
        """
        Log a performance issue.

        Args:
            operation: Description of the slow operation
            duration: Actual duration in seconds
            threshold: Expected maximum duration in seconds
            context: Additional context information
            user_id: User ID if applicable
            project_id: Project ID if applicable
            job_id: Job ID if applicable
        """
        perf_context = {
            "operation": operation,
            "duration_seconds": duration,
            "threshold_seconds": threshold,
            "slowdown_ratio": duration / threshold if threshold > 0 else float('inf')
        }

        if context:
            perf_context.update(_sanitize_context(context))

        if user_id:
            perf_context["user_id"] = user_id
        if project_id:
            perf_context["project_id"] = project_id
        if job_id:
            perf_context["job_id"] = job_id

        self.logger.warning(
            f"Performance issue in {operation}: {duration:.2f}s (threshold: {threshold:.2f}s)",
            extra=perf_context
        )

    def log_retry_attempt(
        self,
        attempt: int,
        max_attempts: int,
        error: Exception,
        operation: str,
        delay: float,
        context: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        project_id: Optional[int] = None,
        job_id: Optional[int] = None
    ):
        """
        Log a retry attempt.

        Args:
            attempt: Current attempt number
            max_attempts: Maximum number of attempts
            error: The error that caused the retry
            operation: Description of the operation
            delay: Delay before next attempt in seconds
            context: Additional context information
            user_id: User ID if applicable
            project_id: Project ID if applicable
            job_id: Job ID if applicable
        """
        retry_context = {
            "operation": operation,
            "attempt": attempt,
            "max_attempts": max_attempts,
            "error_type": type(error).__name__,
            "error_message": _sanitize_message(str(error)),
            "delay_seconds": delay
        }

        if context:
            retry_context.update(_sanitize_context(context))

        if user_id:
            retry_context["user_id"] = user_id
        if project_id:
            retry_context["project_id"] = project_id
        if job_id:
            retry_context["job_id"] = job_id

        self.logger.info(
            f"Retrying {operation} (attempt {attempt}/{max_attempts}) after {delay:.2f}s: {error}",
            extra=retry_context
        )

    def log_operation_start(
        self,
        operation: str,
        context: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        project_id: Optional[int] = None,
        job_id: Optional[int] = None
    ):
        """
        Log the start of an operation.

        Args:
            operation: Description of the operation
            context: Additional context information
            user_id: User ID if applicable
            project_id: Project ID if applicable
            job_id: Job ID if applicable
        """
        start_context = {
            "operation": operation,
            "event": "start",
            "timestamp": datetime.now().isoformat()
        }

        if context:
            start_context.update(_sanitize_context(context))

        if user_id:
            start_context["user_id"] = str(user_id)
        if project_id:
            start_context["project_id"] = str(project_id)
        if job_id:
            start_context["job_id"] = str(job_id)

        # Map operation starts to DEBUG level for "extensive log reporting"
        self.logger.debug(
            f"Starting {operation}",
            extra=start_context
        )

    def log_operation_complete(
        self,
        operation: str,
        duration: float,
        result_count: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        project_id: Optional[int] = None,
        job_id: Optional[int] = None
    ):
        """
        Log the completion of an operation.

        Args:
            operation: Description of the operation
            duration: Duration in seconds
            result_count: Number of results if applicable
            context: Additional context information
            user_id: User ID if applicable
            project_id: Project ID if applicable
            job_id: Job ID if applicable
        """
        complete_context = {
            "operation": operation,
            "event": "complete",
            "duration_seconds": duration,
            "timestamp": datetime.now().isoformat()
        }

        if result_count is not None:
            complete_context["result_count"] = result_count

        if context:
            complete_context.update(_sanitize_context(context))

        if user_id:
            complete_context["user_id"] = user_id
        if project_id:
            complete_context["project_id"] = project_id
        if job_id:
            complete_context["job_id"] = job_id

        # Map operation completion to DEBUG level for "extensive log reporting"
        self.logger.debug(
            f"Completed {operation} in {duration:.2f}s" +
            (f" ({result_count} results)" if result_count is not None else ""),
            extra=complete_context
        )


# Global error logger instances for different components
taxonomy_logger = ErrorLogger("taxonomy")
extraction_logger = ErrorLogger("extraction")
api_logger = ErrorLogger("api")
gui_logger = ErrorLogger("gui")
storage_logger = ErrorLogger("storage")


def get_error_logger(component: str) -> ErrorLogger:
    """
    Get an error logger for a specific component.

    Args:
        component: Component name

    Returns:
        ErrorLogger instance
    """
    return ErrorLogger(component)