"""
Dataset Versioning and Management System.

This module provides comprehensive dataset versioning capabilities including:
- Semantic versioning for datasets
- Change tracking and audit trails
- Dataset lineage tracking
- Incremental updates
- Rollback and comparison functionality
"""

import json
import hashlib
import os
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from ...core.logging import get_logger

logger = get_logger(__name__)

from ...storage.src.project.database_repositories import DatasetVersionRepository


class DatasetVersionManager:
    """
    Manages dataset versioning, change tracking, and lineage.
    """

    def __init__(self, version_repo: DatasetVersionRepository):
        """
        Initialize the version manager.

        Args:
            version_repo: Dataset version repository instance.
        """
        self.version_repo = version_repo

    def create_new_version(self, project_id: int, dataset_name: str, entries: List[Dict],
                          description: Optional[str] = None, metadata: Optional[Dict] = None,
                          created_by: str = "system") -> str:
        """
        Creates a new dataset version.

        Args:
            project_id: Project ID
            dataset_name: Name of the dataset
            entries: List of dataset entries
            description: Optional description
            metadata: Optional metadata dictionary
            created_by: Who created this version

        Returns:
            str: The new version string
        """
        # Calculate next version
        new_version = self.version_repo.increment_version(project_id, dataset_name, "patch")

        # Parse version components
        major, minor, patch = map(int, new_version.split('.'))

        # Calculate file hash
        entries_json = json.dumps(entries, sort_keys=True)
        file_hash = hashlib.sha256(entries_json.encode()).hexdigest()

        # Convert metadata to JSON string
        metadata_json = json.dumps(metadata) if metadata else None

        # Create version record
        version_id = self.version_repo.create_version(
            project_id=project_id,
            version=new_version,
            major_version=major,
            minor_version=minor,
            patch_version=patch,
            dataset_name=dataset_name,
            description=description,
            total_entries=len(entries),
            file_hash=file_hash,
            metadata=metadata_json,
            created_by=created_by
        )

        # Record initial creation change
        self.version_repo.record_change(
            dataset_version_id=version_id,
            change_type="create",
            change_description=f"Initial creation of dataset version {new_version}",
            entries_affected=len(entries),
            changed_by=created_by
        )

        return new_version

    def increment_version(self, project_id: int, dataset_name: str, version_type: str = "patch",
                         description: Optional[str] = None, changed_by: str = "system") -> str:
        """
        Increments the version of an existing dataset.

        Args:
            project_id: Project ID
            dataset_name: Name of the dataset
            version_type: Type of version increment ('major', 'minor', 'patch')
            description: Description of the change
            changed_by: Who made the change

        Returns:
            str: The new version string
        """
        new_version = self.version_repo.increment_version(project_id, dataset_name, version_type)

        # Parse version components
        major, minor, patch = map(int, new_version.split('.'))

        # Get the latest version record to copy metadata
        latest_version_record = None
        versions = self.version_repo.get_versions_by_project(project_id, dataset_name, active_only=True)
        if versions:
            latest_version_record = versions[0]  # Already ordered by version desc

        # Create new version record
        version_id = self.version_repo.create_version(
            project_id=project_id,
            version=new_version,
            major_version=major,
            minor_version=minor,
            patch_version=patch,
            dataset_name=dataset_name,
            description=description or f"Version {version_type} increment",
            total_entries=latest_version_record[7] if latest_version_record else 0,  # total_entries
            file_hash=None,  # Will be set when file is saved
            metadata=latest_version_record[10] if latest_version_record else None,  # metadata
            created_by=changed_by
        )

        # Record version increment change
        self.version_repo.record_change(
            dataset_version_id=version_id,
            change_type="version_increment",
            change_description=f"Version incremented to {new_version} ({version_type})",
            changed_by=changed_by
        )

        return new_version

    def record_incremental_update(self, project_id: int, dataset_name: str, new_entries: List[Dict],
                                 description: Optional[str] = None, changed_by: str = "system") -> str:
        """
        Records an incremental update to a dataset.

        Args:
            project_id: Project ID
            dataset_name: Name of the dataset
            new_entries: New entries to add
            description: Description of the update
            changed_by: Who made the change

        Returns:
            str: The new version string
        """
        # Get latest version
        latest_version_record = None
        versions = self.version_repo.get_versions_by_project(project_id, dataset_name, active_only=True)
        if versions:
            latest_version_record = versions[0]

        if not latest_version_record:
            raise ValueError(f"No existing version found for dataset {dataset_name}")

        # Create new patch version
        new_version = self.increment_version(
            project_id, dataset_name, "patch",
            description or f"Incremental update: added {len(new_entries)} entries",
            changed_by
        )

        # Get the new version record
        new_version_record = None
        versions = self.version_repo.get_versions_by_project(project_id, dataset_name, active_only=True)
        if versions:
            new_version_record = versions[0]

        if new_version_record:
            # Record the incremental change
            self.version_repo.record_change(
                dataset_version_id=new_version_record[0],  # id
                change_type="incremental",
                change_description=f"Added {len(new_entries)} new entries",
                entries_affected=len(new_entries),
                changed_by=changed_by
            )

        return new_version

    def rollback_to_version(self, project_id: int, dataset_name: str, target_version: str,
                           changed_by: str = "system") -> bool:
        """
        Rolls back to a specific version by deactivating newer versions.

        Args:
            project_id: Project ID
            dataset_name: Name of the dataset
            target_version: Version to rollback to
            changed_by: Who performed the rollback

        Returns:
            bool: Success status
        """
        try:
            # Get all versions for the dataset
            versions = self.version_repo.get_versions_by_project(project_id, dataset_name, active_only=False)

            target_found = False
            for version in versions:
                version_str = version[1]  # version field
                version_id = version[0]   # id field

                if version_str == target_version:
                    target_found = True
                    # Activate target version
                    self.version_repo.update_version_status(version_id, True)
                    # Record rollback change
                    self.version_repo.record_change(
                        dataset_version_id=version_id,
                        change_type="rollback",
                        change_description=f"Rolled back to version {target_version}",
                        changed_by=changed_by
                    )
                elif target_found:
                    # Deactivate newer versions
                    self.version_repo.update_version_status(version_id, False)

            return target_found

        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False

    def compare_versions(self, project_id: int, dataset_name: str, version1: str, version2: str) -> Dict:
        """
        Compares two dataset versions.

        Args:
            project_id: Project ID
            dataset_name: Name of the dataset
            version1: First version to compare
            version2: Second version to compare

        Returns:
            Dict: Comparison results
        """
        # Get version records
        v1_record = None
        v2_record = None

        versions = self.version_repo.get_versions_by_project(project_id, dataset_name, active_only=False)
        for version in versions:
            if version[1] == version1:  # version field
                v1_record = version
            elif version[1] == version2:
                v2_record = version

        if not v1_record or not v2_record:
            raise ValueError("One or both versions not found")

        # Get changes for each version
        v1_changes = self.version_repo.get_changes_for_version(v1_record[0])
        v2_changes = self.version_repo.get_changes_for_version(v2_record[0])

        return {
            "version1": {
                "version": version1,
                "total_entries": v1_record[7],  # total_entries
                "created_at": v1_record[12],   # created_at
                "changes_count": len(v1_changes)
            },
            "version2": {
                "version": version2,
                "total_entries": v2_record[7],
                "created_at": v2_record[12],
                "changes_count": len(v2_changes)
            },
            "comparison": {
                "entries_difference": v2_record[7] - v1_record[7],
                "version_difference": self._compare_version_strings(version1, version2)
            }
        }

    def get_version_history(self, project_id: int, dataset_name: str) -> List[Dict]:
        """
        Gets the complete version history for a dataset.

        Args:
            project_id: Project ID
            dataset_name: Name of the dataset

        Returns:
            List[Dict]: Version history
        """
        versions = self.version_repo.get_versions_by_project(project_id, dataset_name, active_only=False)

        history = []
        for version in versions:
            changes = self.version_repo.get_changes_for_version(version[0])
            lineage = self.version_repo.get_lineage_for_version(version[0])

            history.append({
                "id": version[0],
                "version": version[1],
                "major": version[2],
                "minor": version[3],
                "patch": version[4],
                "dataset_name": version[5],
                "description": version[6],
                "total_entries": version[7],
                "file_path": version[8],
                "file_hash": version[9],
                "metadata": version[10],
                "is_active": version[11],
                "created_at": version[12],
                "created_by": version[13],
                "changes": changes,
                "lineage": lineage
            })

        return history

    def record_lineage(self, dataset_version_id: int, source_documents: Optional[List[Dict]] = None,
                      processing_parameters: Optional[Dict] = None, prompt_info: Optional[Dict] = None,
                      taxonomy_info: Optional[Dict] = None):
        """
        Records lineage information for a dataset version.

        Args:
            dataset_version_id: Dataset version ID
            source_documents: List of source document info
            processing_parameters: Processing parameters used
            prompt_info: Prompt information
            taxonomy_info: Taxonomy information
        """
        # Record source documents
        if source_documents:
            for doc in source_documents:
                self.version_repo.record_lineage(
                    dataset_version_id=dataset_version_id,
                    source_type="document",
                    source_id=doc.get("id"),
                    source_name=doc.get("name", doc.get("file_name")),
                    source_hash=doc.get("hash"),
                    processing_parameters=json.dumps({"document_info": doc})
                )

        # Record processing parameters
        if processing_parameters:
            self.version_repo.record_lineage(
                dataset_version_id=dataset_version_id,
                source_type="parameter",
                source_name="processing_parameters",
                processing_parameters=json.dumps(processing_parameters)
            )

        # Record prompt information
        if prompt_info:
            self.version_repo.record_lineage(
                dataset_version_id=dataset_version_id,
                source_type="prompt",
                source_id=prompt_info.get("id"),
                source_name=prompt_info.get("name"),
                source_hash=prompt_info.get("content_hash"),
                processing_parameters=json.dumps({"prompt_info": prompt_info})
            )

        # Record taxonomy information
        if taxonomy_info:
            self.version_repo.record_lineage(
                dataset_version_id=dataset_version_id,
                source_type="taxonomy",
                source_name=taxonomy_info.get("name"),
                source_hash=taxonomy_info.get("hash"),
                processing_parameters=json.dumps({"taxonomy_info": taxonomy_info})
            )

    def _compare_version_strings(self, v1: str, v2: str) -> str:
        """
        Compares two version strings.

        Args:
            v1: First version string
            v2: Second version string

        Returns:
            str: Comparison result
        """
        v1_parts = list(map(int, v1.split('.')))
        v2_parts = list(map(int, v2.split('.')))

        if v1_parts > v2_parts:
            return f"{v1} is newer than {v2}"
        elif v1_parts < v2_parts:
            return f"{v2} is newer than {v1}"
        else:
            return f"{v1} and {v2} are identical"

    def validate_version_string(self, version: str) -> bool:
        """
        Validates a semantic version string.

        Args:
            version: Version string to validate

        Returns:
            bool: True if valid
        """
        try:
            parts = version.split('.')
            if len(parts) != 3:
                return False
            return all(part.isdigit() and int(part) >= 0 for part in parts)
        except:
            return False