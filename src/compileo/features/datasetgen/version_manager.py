import json
import hashlib
import os
import uuid
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from src.compileo.storage.src.project.database_repositories import (
    DatasetVersionRepository, DatasetChangeRepository, DatasetLineageRepository
)

class DatasetVersionManager:
    """
    Manages dataset versioning, change tracking, and lineage with multi-tenant support.
    """

    def __init__(self, version_repo: DatasetVersionRepository, 
                 change_repo: Optional[DatasetChangeRepository] = None,
                 lineage_repo: Optional[DatasetLineageRepository] = None):
        self.version_repo = version_repo
        self.change_repo = change_repo or DatasetChangeRepository(version_repo.db)
        self.lineage_repo = lineage_repo or DatasetLineageRepository(version_repo.db)

    def create_new_version(self, project_id: str, owner_id: str, dataset_name: str, entries: List[Dict],
                          description: Optional[str] = None, metadata: Optional[Dict] = None,
                          created_by: str = "system") -> str:
        # Calculate next version - implementation omitted for brevity, assuming standard semver
        versions = self.version_repo.get_versions_by_project(project_id, owner_id)
        if not versions:
            new_version = "1.0.0"
        else:
            last_version = versions[0]['version']
            v_parts = list(map(int, last_version.split('.')))
            v_parts[2] += 1
            new_version = ".".join(map(str, v_parts))

        # Calculate file hash
        entries_json = json.dumps(entries, sort_keys=True)
        file_hash = hashlib.sha256(entries_json.encode()).hexdigest()

        version_data = {
            "id": str(uuid.uuid4()),
            "owner_id": owner_id,
            "project_id": project_id,
            "version": new_version,
            "major_version": int(new_version.split('.')[0]),
            "minor_version": int(new_version.split('.')[1]),
            "patch_version": int(new_version.split('.')[2]),
            "dataset_name": dataset_name,
            "description": description or f"Auto-generated version {new_version}",
            "total_entries": len(entries),
            "file_path": "",
            "file_hash": file_hash,
            "metadata": metadata or {},
            "is_active": True,
            "created_at": datetime.utcnow(),
            "created_by": created_by
        }
        self.version_repo.create_version(version_data)

        # Record initial change
        self.change_repo.create_change({
            "id": str(uuid.uuid4()),
            "owner_id": owner_id,
            "dataset_version_id": version_data['id'],
            "change_type": "create",
            "change_description": f"Initial creation of dataset version {new_version}",
            "entries_affected": len(entries),
            "changed_by": created_by,
            "created_at": datetime.utcnow()
        })

        return new_version

    def record_lineage(self, dataset_version_id: str, owner_id: str, processing_parameters: Optional[Dict] = None):
        self.lineage_repo.create_lineage({
            "id": str(uuid.uuid4()),
            "owner_id": owner_id,
            "dataset_version_id": dataset_version_id,
            "source_type": "processing",
            "processing_parameters": processing_parameters or {},
            "created_at": datetime.utcnow()
        })
