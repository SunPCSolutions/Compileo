from abc import ABC, abstractmethod
from typing import BinaryIO, Optional
import os
import shutil
from pathlib import Path

class StorageProvider(ABC):
    """Abstract base class for storage providers."""
    
    @abstractmethod
    def upload_file(self, user_id: str, project_id: str, filename: str, file_obj: BinaryIO) -> str:
        """Upload a file to storage and return its identifier/path."""
        pass
        
    @abstractmethod
    def get_file_path(self, user_id: str, project_id: str, filename: str) -> str:
        """Get the absolute path or URL for a stored file."""
        pass
        
    @abstractmethod
    def delete_file(self, user_id: str, project_id: str, filename: str) -> bool:
        """Delete a file from storage."""
        pass
        
    @abstractmethod
    def list_files(self, user_id: str, project_id: str) -> list[str]:
        """List all files for a project."""
        pass

class LocalStorageProvider(StorageProvider):
    """Local filesystem implementation of StorageProvider."""
    
    def __init__(self, base_path: str = "storage"):
        self.base_path = Path(base_path)
        
    def _get_user_project_dir(self, user_id: str, project_id: str) -> Path:
        """Construct the directory path for user/project files."""
        # Clean path construction: storage/{user_id}/uploads/{project_id}/
        return self.base_path / user_id / "uploads" / project_id
        
    def upload_file(self, user_id: str, project_id: str, filename: str, file_obj: BinaryIO) -> str:
        target_dir = self._get_user_project_dir(user_id, project_id)
        target_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = target_dir / filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file_obj, buffer)
            
        return str(file_path)
        
    def get_file_path(self, user_id: str, project_id: str, filename: str) -> str:
        return str(self._get_user_project_dir(user_id, project_id) / filename)
        
    def delete_file(self, user_id: str, project_id: str, filename: str) -> bool:
        file_path = self._get_user_project_dir(user_id, project_id) / filename
        if file_path.exists():
            file_path.unlink()
            return True
        return False
        
    def list_files(self, user_id: str, project_id: str) -> list[str]:
        target_dir = self._get_user_project_dir(user_id, project_id)
        if not target_dir.exists():
            return []
        return [f.name for f in target_dir.iterdir() if f.is_file()]
