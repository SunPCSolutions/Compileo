"""
Custom exceptions for the extraction system.

This module defines domain-specific exceptions for various error conditions
that can occur during content extraction, taxonomy processing, and related operations.
"""

from typing import Any, Dict, Optional


class ExtractionError(Exception):
    """Base exception for all extraction-related errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} (Details: {self.details})"
        return self.message


class TaxonomyError(ExtractionError):
    """Errors related to taxonomy loading, validation, or processing."""
    pass


class TaxonomyNotFoundError(TaxonomyError):
    """Raised when a requested taxonomy cannot be found."""

    def __init__(self, taxonomy_project: str, taxonomy_name: Optional[str] = None):
        message = f"Taxonomy not found: {taxonomy_project}"
        if taxonomy_name:
            message += f"/{taxonomy_name}"
        details = {
            "taxonomy_project": taxonomy_project,
            "taxonomy_name": taxonomy_name
        }
        super().__init__(message, details)


class TaxonomyValidationError(TaxonomyError):
    """Raised when taxonomy structure is invalid."""

    def __init__(self, message: str, taxonomy_path: Optional[str] = None):
        details = {}
        if taxonomy_path:
            details["taxonomy_path"] = taxonomy_path
        super().__init__(message, details)


class TaxonomyCategoryError(TaxonomyError):
    """Raised when category resolution or matching fails."""

    def __init__(self, message: str, category_path: Optional[str] = None):
        details = {}
        if category_path:
            details["category_path"] = category_path
        super().__init__(message, details)


class ClassifierError(ExtractionError):
    """Errors related to classification operations."""
    pass


class ClassifierConfigurationError(ClassifierError):
    """Raised when classifier configuration is invalid."""

    def __init__(self, message: str, classifier_name: Optional[str] = None):
        details = {}
        if classifier_name:
            details["classifier_name"] = classifier_name
        super().__init__(message, details)


class ClassifierUnavailableError(ClassifierError):
    """Raised when no suitable classifier is available."""

    def __init__(self, message: str = "No classifier available for extraction"):
        details = {"available_classifiers": []}
        super().__init__(message, details)


class ClassificationFailureError(ClassifierError):
    """Raised when classification operation fails."""

    def __init__(self, message: str, chunk_id: Optional[str] = None, classifier_name: Optional[str] = None):
        details = {}
        if chunk_id:
            details["chunk_id"] = chunk_id
        if classifier_name:
            details["classifier_name"] = classifier_name
        super().__init__(message, details)


class APIError(ExtractionError):
    """Errors related to external API calls."""
    pass


class APIConnectionError(APIError):
    """Raised when API connection fails."""

    def __init__(self, service_name: str, endpoint: Optional[str] = None):
        message = f"Failed to connect to {service_name} API"
        details = {"service_name": service_name}
        if endpoint:
            details["endpoint"] = endpoint
        super().__init__(message, details)


class APIAuthenticationError(APIError):
    """Raised when API authentication fails."""

    def __init__(self, service_name: str):
        message = f"Authentication failed for {service_name} API"
        details = {"service_name": service_name}
        super().__init__(message, details)


class APIRateLimitError(APIError):
    """Raised when API rate limit is exceeded."""

    def __init__(self, service_name: str, retry_after: Optional[int] = None):
        message = f"Rate limit exceeded for {service_name} API"
        details = {"service_name": service_name}
        if retry_after:
            details["retry_after_seconds"] = str(retry_after)
        super().__init__(message, details)


class APIResponseError(APIError):
    """Raised when API returns an error response."""

    def __init__(self, service_name: str, status_code: int, response_body: Optional[str] = None):
        message = f"{service_name} API returned error (status: {status_code})"
        details = {
            "service_name": service_name,
            "status_code": status_code
        }
        if response_body:
            details["response_body"] = response_body
        super().__init__(message, details)


class ProcessingError(ExtractionError):
    """Errors related to data processing operations."""
    pass


class ChunkProcessingError(ProcessingError):
    """Raised when chunk processing fails."""

    def __init__(self, message: str, chunk_id: Optional[str] = None, chunk_path: Optional[str] = None):
        details = {}
        if chunk_id:
            details["chunk_id"] = chunk_id
        if chunk_path:
            details["chunk_path"] = chunk_path
        super().__init__(message, details)


class DataValidationError(ProcessingError):
    """Raised when data validation fails."""

    def __init__(self, message: str, field: Optional[str] = None, value: Optional[Any] = None):
        details = {}
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = str(value)
        super().__init__(message, details)


class StorageError(ExtractionError):
    """Errors related to data storage operations."""
    pass


class DatabaseConnectionError(StorageError):
    """Raised when database connection fails."""

    def __init__(self, database_name: Optional[str] = None):
        message = "Database connection failed"
        details = {}
        if database_name:
            details["database_name"] = database_name
        super().__init__(message, details)


class FileStorageError(StorageError):
    """Raised when file storage operations fail."""

    def __init__(self, message: str, file_path: Optional[str] = None, operation: Optional[str] = None):
        details = {}
        if file_path:
            details["file_path"] = file_path
        if operation:
            details["operation"] = operation
        super().__init__(message, details)


class DiskSpaceError(StorageError):
    """Raised when insufficient disk space is available."""

    def __init__(self, required_space: Optional[int] = None, available_space: Optional[int] = None):
        message = "Insufficient disk space for operation"
        details = {}
        if required_space:
            details["required_space_bytes"] = required_space
        if available_space:
            details["available_space_bytes"] = available_space
        super().__init__(message, details)


class JobError(ExtractionError):
    """Errors related to job processing and management."""
    pass


class JobNotFoundError(JobError):
    """Raised when a requested job cannot be found."""

    def __init__(self, job_id):
        message = f"Extraction job with ID {job_id} not found"
        details = {"job_id": job_id}
        super().__init__(message, details)


class JobStateError(JobError):
    """Raised when job is in an invalid state for the requested operation."""

    def __init__(self, job_id: int, current_status: str, required_status: Optional[str] = None):
        message = f"Job {job_id} is in invalid state '{current_status}'"
        if required_status:
            message += f" (required: {required_status})"
        details = {
            "job_id": job_id,
            "current_status": current_status,
            "required_status": required_status
        }
        super().__init__(message, details)


class ConfigurationError(ExtractionError):
    """Errors related to system configuration."""
    pass


class MissingConfigurationError(ConfigurationError):
    """Raised when required configuration is missing."""

    def __init__(self, config_key: str):
        message = f"Required configuration missing: {config_key}"
        details = {"config_key": config_key}
        super().__init__(message, details)


class InvalidConfigurationError(ConfigurationError):
    """Raised when configuration values are invalid."""

    def __init__(self, config_key: str, value: Any, expected_type: Optional[str] = None):
        message = f"Invalid configuration for {config_key}: {value}"
        details = {
            "config_key": config_key,
            "value": value
        }
        if expected_type:
            details["expected_type"] = expected_type
        super().__init__(message, details)