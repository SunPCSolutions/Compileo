"""
File manager for handling document file operations.
"""

import os
import shutil
from pathlib import Path
from typing import Optional, Dict, Any


class FileManager:
    """Manager for file operations related to documents."""

    def __init__(self, storage_type: Optional[str] = None):
        """
        Initialize FileManager.
        
        Args:
            storage_type: Optional storage type (e.g. 'datasets', 'uploads', 'extract').
                          If provided, used to set a specific base path.
        """
        self.storage_type = storage_type

    def store_file(self, project_id: str, document_id: Optional[str], file_name: str, file_content: bytes) -> str:
        """
        Store a file content to the storage.
        
        Args:
            project_id: The project ID.
            document_id: The document ID (optional).
            file_name: The name of the file.
            file_content: The content of the file.
            
        Returns:
            The path to the stored file.
        """
        if self.storage_type == 'datasets':
            storage_path = self.get_storage_base_path() / "datasets" / str(project_id)
        elif self.storage_type == 'extract':
            storage_path = self.get_storage_base_path() / "extract" / str(project_id)
        else:
            storage_path = self.get_project_storage_path(project_id)
            
        storage_path.mkdir(parents=True, exist_ok=True)
        file_path = storage_path / file_name
        
        with open(file_path, 'wb') as f:
            f.write(file_content)
            
        return str(file_path)

    @staticmethod
    def get_storage_base_path() -> Path:
        """Get the base storage path."""
        # Fix path resolution to point to project root/storage
        # Current file: src/compileo/storage/src/project/file_manager.py
        # Target: storage/
        # Need to go up 6 levels to reach project root from this file location
        # or use absolute path based on workspace root
        
        # Try to find workspace root by looking for pyproject.toml or .git
        current = Path(__file__).resolve()
        while current.parent != current:
            if (current / "pyproject.toml").exists() or (current / ".git").exists():
                return current / "storage"
            current = current.parent
            
        # Fallback to relative path calculation if root not found
        # src/compileo/storage/src/project/file_manager.py -> ../../../../../storage
        return Path(__file__).resolve().parent.parent.parent.parent.parent.parent / "storage"

    @staticmethod
    def get_project_storage_path(project_id: str) -> Path:
        """Get the storage path for a specific project."""
        return FileManager.get_storage_base_path() / "uploads" / project_id

    @staticmethod
    def ensure_project_directory(project_id: str) -> Path:
        """Ensure the project directory exists and return its path."""
        project_path = FileManager.get_project_storage_path(project_id)
        project_path.mkdir(parents=True, exist_ok=True)
        return project_path

    @staticmethod
    def save_uploaded_file(project_id: str, filename: str, file_content) -> Optional[str]:
        """Save an uploaded file to the project directory.

        Args:
            project_id: The project ID
            filename: The filename
            file_content: The file content (UploadFile or bytes)

        Returns:
            The relative path to the saved file, or None if failed
        """
        try:
            project_path = FileManager.ensure_project_directory(project_id)
            file_path = project_path / filename

            if hasattr(file_content, 'read'):  # UploadFile
                with open(file_path, 'wb') as f:
                    shutil.copyfileobj(file_content.file, f)
            else:  # bytes
                with open(file_path, 'wb') as f:
                    f.write(file_content)

            # Return relative path from storage
            return str(file_path.relative_to(FileManager.get_storage_base_path()))
        except Exception:
            return None

    @staticmethod
    def get_file_info(project_id: str, filename: str) -> Optional[Dict[str, Any]]:
        """Get information about a file."""
        try:
            file_path = FileManager.get_project_storage_path(project_id) / filename
            if file_path.exists():
                stat = file_path.stat()
                return {
                    'filename': filename,
                    'path': str(file_path),
                    'size': stat.st_size,
                    'modified': stat.st_mtime,
                    'exists': True
                }
            return None
        except Exception:
            return None

    @staticmethod
    def delete_file(project_id: str, filename: str) -> bool:
        """Delete a file from the project directory."""
        try:
            file_path = FileManager.get_project_storage_path(project_id) / filename
            if file_path.exists():
                file_path.unlink()
                return True
            return False
        except Exception:
            return False

    @staticmethod
    def list_project_files(project_id: str) -> list:
        """List all files in a project directory."""
        try:
            project_path = FileManager.get_project_storage_path(project_id)
            if project_path.exists():
                return [f.name for f in project_path.iterdir() if f.is_file()]
            return []
        except Exception:
            return []