"""API client for GUI to interact with Compileo extraction services."""

import requests
import json
from typing import Dict, Any, List, Optional
from datetime import datetime


class ExtractionAPIClient:
    """Client for interacting with the Compileo extraction API."""

    def __init__(self, base_url: str = "http://localhost:8000/api/v1"):
        """
        Initialize the API client.

        Args:
            base_url: Base URL for the API
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()

    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """
        Make an HTTP request to the API.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            **kwargs: Additional request parameters

        Returns:
            Response data as dictionary

        Raises:
            requests.RequestException: For HTTP errors
        """
        url = f"{self.base_url}{endpoint}"

        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            # Re-raise with more context
            raise requests.RequestException(f"API request failed: {e}")

    def create_extraction_job(
        self,
        taxonomy_id: int,
        selected_categories: List[str],
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a new selective extraction job.

        Args:
            taxonomy_id: ID of the taxonomy to use
            selected_categories: List of category paths to extract
            parameters: Additional extraction parameters

        Returns:
            Job creation response
        """
        data = {
            "taxonomy_id": taxonomy_id,
            "selected_categories": selected_categories,
            "parameters": parameters or {}
        }

        return self._make_request("POST", "/extraction/", json=data)

    def get_job_status(self, job_id: int) -> Dict[str, Any]:
        """
        Get the status of an extraction job.

        Args:
            job_id: Job ID to check

        Returns:
            Job status information
        """
        return self._make_request("GET", f"/extraction/{job_id}")

    def get_extraction_results(
        self,
        job_id: int,
        page: int = 1,
        page_size: int = 50,
        min_confidence: Optional[float] = None,
        category_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get paginated results for an extraction job.

        Args:
            job_id: Job ID to get results for
            page: Page number (1-based)
            page_size: Number of results per page
            min_confidence: Minimum confidence threshold
            category_filter: Category filter string

        Returns:
            Paginated results
        """
        params = {
            "page": page,
            "page_size": page_size
        }

        if min_confidence is not None:
            params["min_confidence"] = min_confidence
        if category_filter:
            params["category_filter"] = category_filter

        return self._make_request("GET", f"/extraction/{job_id}/results", params=params)

    def cancel_job(self, job_id: int) -> Dict[str, Any]:
        """
        Cancel a running extraction job.

        Args:
            job_id: Job ID to cancel

        Returns:
            Cancellation response
        """
        return self._make_request("DELETE", f"/extraction/{job_id}")

    def restart_job(self, job_id: int) -> Dict[str, Any]:
        """
        Restart a failed or cancelled extraction job.

        Args:
            job_id: Job ID to restart

        Returns:
            Restart response
        """
        return self._make_request("POST", f"/extraction/{job_id}/restart")

    def get_project_jobs(self, project_id: int) -> List[Dict[str, Any]]:
        """
        Get all extraction jobs for a project.

        Args:
            project_id: Project ID

        Returns:
            List of job status information
        """
        return self._make_request("GET", f"/extraction/projects/{project_id}/jobs")

    def get_taxonomies(self, project_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get available taxonomies.

        Args:
            project_id: Optional project ID to filter taxonomies

        Returns:
            List of available taxonomies
        """
        params = {}
        if project_id:
            params["project_id"] = project_id

        return self._make_request("GET", "/taxonomy/", params=params)

    def get_taxonomy_details(self, taxonomy_id: int) -> Dict[str, Any]:
        """
        Get detailed information about a taxonomy.

        Args:
            taxonomy_id: Taxonomy ID

        Returns:
            Taxonomy details
        """
        return self._make_request("GET", f"/taxonomy/{taxonomy_id}")

    def get_projects(self) -> List[Dict[str, Any]]:
        """
        Get list of available projects.

        Returns:
            List of projects
        """
        return self._make_request("GET", "/projects/")

    def create_project(self, name: str, description: str = "") -> Dict[str, Any]:
        """
        Create a new project.

        Args:
            name: Project name
            description: Project description

        Returns:
            Created project information
        """
        data = {
            "name": name,
            "description": description
        }

        return self._make_request("POST", "/projects/", json=data)

    def upload_documents(self, project_id: int, files: List[tuple]) -> Dict[str, Any]:
        """
        Upload documents to a project.

        Args:
            project_id: Project ID
            files: List of (filename, file_content, content_type) tuples

        Returns:
            Upload response
        """
        files_data = [("files", file_info) for file_info in files]
        data = {"project_id": str(project_id)}

        # For file uploads, we need to use data parameter differently
        response = self.session.post(
            f"{self.base_url}/documents/upload",
            files=files_data,
            data=data
        )
        response.raise_for_status()
        return response.json()

    def get_project_documents(self, project_id: int) -> List[Dict[str, Any]]:
        """
        Get documents for a project.

        Args:
            project_id: Project ID

        Returns:
            List of project documents
        """
        return self._make_request("GET", f"/documents/project/{project_id}")

    def generate_dataset(
        self,
        project_id: int,
        prompt_name: str,
        format_type: str = "jsonl",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a dataset from project documents.

        Args:
            project_id: Project ID
            prompt_name: Name of the prompt to use
            format_type: Output format (jsonl, json, etc.)
            **kwargs: Additional generation parameters

        Returns:
            Dataset generation response
        """
        data = {
            "project_id": project_id,
            "prompt_name": prompt_name,
            "format_type": format_type,
            **kwargs
        }

        return self._make_request("POST", "/datasets/generate", json=data)

    def get_dataset_status(self, job_id: int) -> Dict[str, Any]:
        """
        Get dataset generation status.

        Args:
            job_id: Dataset generation job ID

        Returns:
            Job status
        """
        return self._make_request("GET", f"/datasets/status/{job_id}")

    def download_dataset(self, dataset_id: int) -> bytes:
        """
        Download a generated dataset.

        Args:
            dataset_id: Dataset ID

        Returns:
            Dataset content as bytes
        """
        response = self.session.get(f"{self.base_url}/datasets/download/{dataset_id}")
        response.raise_for_status()
        return response.content

    def get_quality_metrics(self, dataset_file: str, **kwargs) -> Dict[str, Any]:
        """
        Analyze dataset quality.

        Args:
            dataset_file: Path to dataset file
            **kwargs: Quality analysis parameters

        Returns:
            Quality analysis results
        """
        data = {
            "dataset_file": dataset_file,
            **kwargs
        }

        return self._make_request("POST", "/quality/analyze", json=data)

    def run_benchmark(self, model_info: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Run model benchmarking.

        Args:
            model_info: Model information
            **kwargs: Benchmark parameters

        Returns:
            Benchmark results
        """
        data = {
            "model_info": model_info,
            **kwargs
        }

        return self._make_request("POST", "/benchmarking/run", json=data)

    def post(self, endpoint: str, **kwargs) -> Dict[str, Any]:
        """
        Make a POST request to the API.

        Args:
            endpoint: API endpoint
            **kwargs: Additional request parameters (json, data, files, etc.)

        Returns:
            Response data as dictionary
        """
        return self._make_request("POST", endpoint, **kwargs)

    def health_check(self) -> Dict[str, Any]:
        """
        Check API health status.

        Returns:
            Health status information
        """
        try:
            return self._make_request("GET", "/health")
        except requests.RequestException:
            return {"status": "unhealthy", "timestamp": datetime.utcnow().isoformat()}


class APIClientError(Exception):
    """Base exception for API client errors."""
    pass


class APIConnectionError(APIClientError):
    """Exception raised when API connection fails."""
    pass


class APIAuthenticationError(APIClientError):
    """Exception raised when API authentication fails."""
    pass


class APIRateLimitError(APIClientError):
    """Exception raised when API rate limit is exceeded."""
    pass


class APIValidationError(APIClientError):
    """Exception raised when API validation fails."""
    pass