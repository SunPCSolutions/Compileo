import requests
import streamlit as st
from typing import List, Dict, Any, Optional
import os

class APIError(Exception):
    """Custom exception for API errors."""
    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code

class APIClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.api_key = None

    def set_api_key(self, api_key: str):
        """Set the API key for authentication."""
        self.api_key = api_key

    def update_settings(self, base_url: str, api_key: str):
        """Update both the base URL and API key."""
        self.base_url = base_url
        self.api_key = api_key

    def _handle_response(self, response: requests.Response) -> Optional[Dict[str, Any]]:
        """Centralized response handling."""
        try:
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                raise APIError("Too many requests. Please wait a moment and try again.", status_code=429)
            elif e.response.status_code >= 500:
                raise APIError(f"Server Error: {e.response.reason}", status_code=e.response.status_code)
            raise APIError(f"HTTP Error: {e}", status_code=e.response.status_code)
        except requests.exceptions.RequestException as e:
            raise APIError(f"API request failed: {e}")

    def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        try:
            response = requests.get(f"{self.base_url}{endpoint}", params=params)
            return self._handle_response(response)
        except APIError as e:
            st.error(str(e))
            return None

    def post(self, endpoint: str, data: Optional[Dict[str, Any]] = None, files: Optional[List] = None) -> Optional[Dict[str, Any]]:
        try:
            if files:
                response = requests.post(f"{self.base_url}{endpoint}", data=data, files=files)
            else:
                response = requests.post(f"{self.base_url}{endpoint}", json=data)
            return self._handle_response(response)
        except APIError as e:
            st.error(str(e))
            return None

    def put(self, endpoint: str, data: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        try:
            response = requests.put(f"{self.base_url}{endpoint}", json=data)
            return self._handle_response(response)
        except APIError as e:
            st.error(str(e))
            return None

    def delete(self, endpoint: str, data: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        try:
            response = requests.delete(f"{self.base_url}{endpoint}", json=data)
            return self._handle_response(response)
        except APIError as e:
            st.error(str(e))
            return None

    def upload_documents(self, project_id: int, files: List) -> Optional[Dict[str, Any]]:
        file_list = [("files", (file.name, file, file.type)) for file in files]
        return self.post("/api/v1/documents/upload", data={"project_id": project_id}, files=file_list)

    def get_upload_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self.get(f"/api/v1/documents/upload/{job_id}/status")

    def get_processing_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self.get(f"/api/v1/documents/process/{job_id}/status")

    def parse_documents(self, parse_request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse documents to markdown without chunking."""
        # Use the /process endpoint with parsing enabled and chunking disabled
        process_request = dict(parse_request)
        process_request["skip_parsing"] = False
        return self.post("/api/v1/documents/process", data=process_request)

    def process_documents(self, process_request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process documents (parse and/or chunk)."""
        return self.post("/api/v1/documents/process", data=process_request)

    def analyze_quality(self, dataset_file: str, config: Optional[Dict[str, Any]] = None,
                       threshold: float = 0.7, output_format: str = "json", quality_model: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Start quality analysis for a dataset."""
        data = {
            "dataset_file": dataset_file,
            "config": config,
            "threshold": threshold,
            "output_format": output_format
        }
        if quality_model:
            data["quality_model"] = quality_model
        return self.post("/api/v1/quality/analyze", data=data)

    def get_quality_results(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get quality analysis results for a job."""
        return self.get(f"/api/v1/quality/{job_id}/results")

    def get_quality_history(self, limit: int = 20) -> Optional[Dict[str, Any]]:
        """Get quality analysis history."""
        return self.get("/api/v1/quality/history", params={"limit": limit})

    def list_datasets(self, project_id: int, page: int = 1, page_size: int = 20) -> Optional[Dict[str, Any]]:
        """List datasets for a project."""
        return self.get(f"/api/v1/datasets/list/{project_id}", params={"page": page, "page_size": page_size})

# Initialize API clients
api_base_url = os.getenv("API_BASE_URL", "http://localhost:8000")
api_client = APIClient(base_url=api_base_url)
benchmarking_api_client = APIClient(base_url=api_base_url)