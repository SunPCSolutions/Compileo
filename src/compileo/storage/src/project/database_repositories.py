"""
Database repositories for project-related entities.
"""

from typing import List, Optional, Dict, Any, Union, Tuple
import json
import uuid
from datetime import datetime
from ....core.logging import get_logger

logger = get_logger(__name__)

class BaseRepository:
    """Base repository class handling database connection."""
    
    def __init__(self, db_connection):
        self.db = db_connection
        self.db_connection = db_connection  # Alias for compatibility

    def cursor(self):
        return self.db.cursor()

    def commit(self):
        self.db.commit()

class ProjectRepository(BaseRepository):
    """Repository for project operations."""

    def get_all_projects(self) -> List[Dict[str, Any]]:
        cursor = self.cursor()
        cursor.execute('SELECT * FROM projects ORDER BY created_at DESC')
        rows = cursor.fetchall()
        return [dict(row) for row in rows] if rows else []

    def get_project_by_id(self, project_id: str) -> Optional[Dict[str, Any]]:
        cursor = self.cursor()
        cursor.execute('SELECT * FROM projects WHERE id = ?', (project_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_project_by_name(self, name: str) -> Optional[str]:
        cursor = self.cursor()
        cursor.execute('SELECT id FROM projects WHERE name = ?', (name,))
        row = cursor.fetchone()
        return row[0] if row else None

    def create_project(self, name: str, description: str = "") -> str:
        """Create a new project."""
        # Check if argument is a dict (legacy/compatibility)
        if isinstance(name, dict):
            project_data = name
            name = project_data.get('name')
            description = project_data.get('description', '')
            project_id = project_data.get('id', str(uuid.uuid4()))
            created_at = project_data.get('created_at', datetime.utcnow())
            updated_at = project_data.get('updated_at', created_at)
        else:
            project_id = str(uuid.uuid4())
            created_at = datetime.utcnow()
            updated_at = created_at
        
        cursor = self.cursor()
        cursor.execute('''
            INSERT INTO projects (id, name, description, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (project_id, name, description, created_at, updated_at))
        self.commit()
        return project_id

class DocumentRepository(BaseRepository):
    """Repository for document operations."""

    def get_documents_by_project_id(self, project_id: Union[str, int]) -> List[Dict[str, Any]]:
        """Get all documents for a project."""
        cursor = self.cursor()
        cursor.execute('SELECT * FROM documents WHERE project_id = ? ORDER BY created_at DESC', (project_id,))
        rows = cursor.fetchall()
        return [dict(row) for row in rows] if rows else []
        
    # Alias for compatibility
    get_documents_by_project = get_documents_by_project_id

    def get_document_by_id(self, document_id: Union[str, int]) -> Optional[Dict[str, Any]]:
        """Get document by ID."""
        cursor = self.cursor()
        cursor.execute('SELECT * FROM documents WHERE id = ?', (document_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def create_document(self, project_id: Union[str, int], filename: str, file_path: str = "") -> str:
        """Create a new document."""
        # Handle if first arg is a dict
        if isinstance(project_id, dict):
            doc_data = project_id
            project_id = doc_data['project_id']
            filename = doc_data['file_name']
            file_path = doc_data.get('source_file_path', '')
            # Ignoring other fields for now to match method signature
        
        document_id = str(uuid.uuid4())
        
        cursor = self.cursor()
        cursor.execute('''
            INSERT INTO documents (id, project_id, file_name, source_file_path, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (document_id, project_id, filename, file_path, datetime.utcnow(), datetime.utcnow()))
        
        self.commit()
        return document_id

    def update_document_path(self, document_id: Union[str, int], file_path: str):
        """Update document file path."""
        cursor = self.cursor()
        cursor.execute('UPDATE documents SET source_file_path = ? WHERE id = ?', (file_path, document_id))
        self.commit()

    def delete_document(self, document_id: Union[str, int]):
        """Delete a document."""
        cursor = self.cursor()
        cursor.execute('DELETE FROM documents WHERE id = ?', (document_id,))
        self.commit()

class DatasetJobRepository(BaseRepository):
    """Repository for dataset job operations."""

    def get_jobs_by_project(self, project_id: str) -> List[Dict[str, Any]]:
        cursor = self.cursor()
        cursor.execute('SELECT * FROM dataset_jobs WHERE project_id = ? ORDER BY created_at DESC', (project_id,))
        rows = cursor.fetchall()
        return [dict(row) for row in rows] if rows else []

    def get_job_by_id(self, job_id: str) -> Optional[Dict[str, Any]]:
        cursor = self.cursor()
        cursor.execute('SELECT * FROM dataset_jobs WHERE id = ?', (job_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def create_job(self, job_data: Dict[str, Any]) -> str:
        cursor = self.cursor()
        cursor.execute('''
            INSERT INTO dataset_jobs (id, project_id, job_type, status, parameters, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            job_data['id'],
            job_data['project_id'],
            job_data['job_type'],
            job_data['status'],
            json.dumps(job_data.get('parameters', {})),
            job_data['created_at'],
            job_data['updated_at']
        ))
        self.commit()
        return job_data['id']

    def update_job_status(self, job_id: str, status: str, error: Optional[str] = None):
        """Update job status and error message."""
        cursor = self.cursor()
        cursor.execute('''
            UPDATE dataset_jobs
            SET status = ?, updated_at = ?
            WHERE id = ?
        ''', (status, datetime.utcnow(), job_id))
        self.commit()

    def update_job_progress(self, job_id: str, progress: float):
        """Update job progress (0.0 to 1.0)."""
        cursor = self.cursor()
        # Add progress column dynamically if it doesn't exist (safety for migrations)
        try:
            cursor.execute("ALTER TABLE dataset_jobs ADD COLUMN progress REAL DEFAULT 0.0")
        except:
            pass
            
        cursor.execute('''
            UPDATE dataset_jobs
            SET progress = ?, updated_at = ?
            WHERE id = ?
        ''', (progress, datetime.utcnow(), job_id))
        self.commit()

    def update_job_result(self, job_id: str, result: Dict[str, Any]):
        """Update job result."""
        cursor = self.cursor()
        # Add result column dynamically if it doesn't exist
        try:
            cursor.execute("ALTER TABLE dataset_jobs ADD COLUMN result TEXT")
        except:
            pass

        cursor.execute('''
            UPDATE dataset_jobs
            SET result = ?, updated_at = ?
            WHERE id = ?
        ''', (json.dumps(result), datetime.utcnow(), job_id))
        self.commit()

    def cleanup_document_data(self, document_id: Union[str, int]) -> Dict[str, int]:
        """Clean up all data related to a document."""
        counts = {}
        cursor = self.cursor()
        
        # Delete extraction results
        cursor.execute('''
            DELETE FROM extraction_results 
            WHERE chunk_id IN (SELECT id FROM chunks WHERE document_id = ?)
        ''', (document_id,))
        counts['extraction_results'] = cursor.rowcount
        
        # Delete chunks
        cursor.execute('DELETE FROM chunks WHERE document_id = ?', (document_id,))
        counts['chunks'] = cursor.rowcount
        
        # Delete parsed files
        cursor.execute('''
            DELETE FROM parsed_files 
            WHERE parsed_document_id IN (SELECT id FROM parsed_documents WHERE document_id = ?)
        ''', (document_id,))
        counts['parsed_files'] = cursor.rowcount
        
        # Delete parsed documents
        cursor.execute('DELETE FROM parsed_documents WHERE document_id = ?', (document_id,))
        counts['parsed_documents'] = cursor.rowcount
        
        self.commit()
        return counts

class DatasetParameterRepository(BaseRepository):
    """Repository for dataset parameter operations."""

    def get_parameters_by_project(self, project_id: str) -> List[Dict[str, Any]]:
        cursor = self.cursor()
        cursor.execute('SELECT * FROM dataset_parameters WHERE project_id = ? ORDER BY created_at DESC', (project_id,))
        rows = cursor.fetchall()
        return [dict(row) for row in rows] if rows else []

    def get_parameter_by_id(self, parameter_id: str) -> Optional[Dict[str, Any]]:
        cursor = self.cursor()
        cursor.execute('SELECT * FROM dataset_parameters WHERE id = ?', (parameter_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def create(self, parameter_data: Dict[str, Any]) -> str:
        """Alias for create_parameter to match other repositories."""
        return self.create_parameter(parameter_data)

    def create_parameter(self, parameter_data: Dict[str, Any]) -> str:
        cursor = self.cursor()
        
        # Check if argument is a dict with individual fields or older name/value format
        if 'name' in parameter_data and 'value' in parameter_data and len(parameter_data) <= 6:
            # Handle legacy format if needed, but for now assuming new wide format
            # Or map 'value' dict to columns if name is 'parameters'
            pass

        # Use parameter_data directly assuming it matches the wide schema
        cursor.execute('''
            INSERT INTO dataset_parameters (
                id, project_id, purpose, audience, extraction_rules,
                dataset_format, question_style, answer_style, negativity_ratio,
                data_augmentation, custom_audience, custom_purpose,
                complexity_level, domain, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            parameter_data['id'],
            str(parameter_data['project_id']),
            parameter_data.get('purpose'),
            parameter_data.get('audience'),
            parameter_data.get('extraction_rules'),
            parameter_data.get('dataset_format'),
            parameter_data.get('question_style'),
            parameter_data.get('answer_style'),
            parameter_data.get('negativity_ratio'),
            parameter_data.get('data_augmentation'),
            parameter_data.get('custom_audience'),
            parameter_data.get('custom_purpose'),
            parameter_data.get('complexity_level'),
            parameter_data.get('domain'),
            parameter_data['created_at'],
            parameter_data['updated_at']
        ))
        self.commit()
        return parameter_data['id']
    
    def get_by_project_id(self, project_id: str) -> List[Dict[str, Any]]:
        """Alias for get_parameters_by_project."""
        return self.get_parameters_by_project(project_id)

class PromptRepository(BaseRepository):
    """Repository for prompt operations."""

    def get_prompts_by_project(self, project_id: str) -> List[Dict[str, Any]]:
        cursor = self.cursor()
        cursor.execute('SELECT * FROM prompts WHERE project_id = ? ORDER BY created_at DESC', (project_id,))
        rows = cursor.fetchall()
        return [dict(row) for row in rows] if rows else []

    def get_prompt_by_id(self, prompt_id: str) -> Optional[Dict[str, Any]]:
        cursor = self.cursor()
        cursor.execute('SELECT * FROM prompts WHERE id = ?', (prompt_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_by_name(self, name: str, project_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get prompt by name, optionally filtered by project."""
        cursor = self.cursor()
        if project_id:
            cursor.execute('SELECT * FROM prompts WHERE name = ? AND project_id = ?', (name, project_id))
        else:
            cursor.execute('SELECT * FROM prompts WHERE name = ?', (name,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def create(self, prompt_data: Dict[str, Any]) -> str:
        """Alias for create_prompt to match other repositories."""
        return self.create_prompt(prompt_data)

    def create_prompt(self, prompt_data: Dict[str, Any]) -> str:
        cursor = self.cursor()
        cursor.execute('''
            INSERT INTO prompts (id, project_id, name, content, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            prompt_data['id'],
            str(prompt_data['project_id']),
            prompt_data['name'],
            prompt_data['content'],
            prompt_data['created_at'],
            prompt_data['updated_at']
        ))
        self.commit()
        return prompt_data['id']

class ChunkRepository(BaseRepository):
    """Repository for chunk operations."""

    def get_chunks_by_document(self, document_id: Union[str, int]) -> List[Dict[str, Any]]:
        cursor = self.cursor()
        cursor.execute('SELECT * FROM chunks WHERE document_id = ? ORDER BY chunk_index', (document_id,))
        rows = cursor.fetchall()
        return [dict(row) for row in rows] if rows else []

    def get_project_chunks(self, project_id: Union[str, int]) -> List[Tuple[str]]:
        """Get chunk file paths for all documents in a project."""
        cursor = self.cursor()
        cursor.execute('''
            SELECT c.file_path
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE d.project_id = ?
            ORDER BY d.created_at DESC, c.chunk_index ASC
        ''', (project_id,))
        rows = cursor.fetchall()
        return rows if rows else []

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        cursor = self.cursor()
        cursor.execute('SELECT * FROM chunks WHERE id = ?', (chunk_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def create_chunk(self, chunk_data: Dict[str, Any]) -> str:
        metadata = chunk_data.get('metadata', {})
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except:
                metadata = {}

        file_path = chunk_data.get('file_path') or metadata.get('file_path')
        chunk_strategy = chunk_data.get('chunk_strategy') or metadata.get('strategy', 'unknown')
        content_preview = chunk_data['content'][:200] if chunk_data.get('content') else ""

        cursor = self.cursor()
        cursor.execute('''
            INSERT INTO chunks (id, document_id, chunk_index, content_preview, metadata, file_path, chunk_strategy, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            chunk_data['id'],
            chunk_data['document_id'],
            chunk_data['chunk_index'],
            content_preview,  # Store preview instead of full content
            json.dumps(metadata),
            file_path,
            chunk_strategy,
            chunk_data.get('status', 'active'),
            chunk_data['created_at']
        ))
        self.commit()
        return chunk_data['id']

    def delete_chunk(self, chunk_id: str) -> bool:
        """Delete a chunk by ID."""
        cursor = self.cursor()
        cursor.execute('DELETE FROM chunks WHERE id = ?', (chunk_id,))
        self.commit()
        return cursor.rowcount > 0

class TaxonomyRepository(BaseRepository):
    """Repository for taxonomy operations."""

    def get_taxonomies_by_project(self, project_id: str) -> List[Dict[str, Any]]:
        cursor = self.cursor()
        cursor.execute('SELECT * FROM taxonomies WHERE project_id = ? ORDER BY created_at DESC', (project_id,))
        rows = cursor.fetchall()
        return [dict(row) for row in rows] if rows else []

    def get_taxonomy_by_id(self, taxonomy_id: str) -> Optional[Dict[str, Any]]:
        cursor = self.cursor()
        cursor.execute('SELECT * FROM taxonomies WHERE id = ?', (taxonomy_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def create_taxonomy(self, taxonomy_data: Dict[str, Any]) -> str:
        cursor = self.cursor()
        cursor.execute('''
            INSERT INTO taxonomies (id, project_id, name, structure, file_path, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            taxonomy_data['id'],
            taxonomy_data['project_id'],
            taxonomy_data['name'],
            json.dumps(taxonomy_data.get('structure', {})),
            taxonomy_data.get('file_path'),
            taxonomy_data['created_at'],
            taxonomy_data['updated_at']
        ))
        self.commit()
        return taxonomy_data['id']

    def update_taxonomy(self, taxonomy_id: str, taxonomy_data: Dict[str, Any]) -> bool:
        """Update a taxonomy record."""
        cursor = self.cursor()
        
        # Prepare updates
        updates = []
        params = []
        
        if 'name' in taxonomy_data:
            updates.append("name = ?")
            params.append(taxonomy_data['name'])
            
        if 'structure' in taxonomy_data:
            updates.append("structure = ?")
            params.append(json.dumps(taxonomy_data['structure']))
            
        if 'file_path' in taxonomy_data:
            updates.append("file_path = ?")
            params.append(taxonomy_data['file_path'])
            
        if not updates:
            return False
            
        updates.append("updated_at = ?")
        params.append(datetime.utcnow())
        
        params.append(taxonomy_id)
        
        query = f"UPDATE taxonomies SET {', '.join(updates)} WHERE id = ?"
        cursor.execute(query, params)
        self.commit()
        return cursor.rowcount > 0

    def delete_taxonomy(self, taxonomy_id: str) -> bool:
        """Delete a taxonomy by ID."""
        # First get the taxonomy data to find the file path
        taxonomy_data = self.get_taxonomy_by_id(taxonomy_id)
        if not taxonomy_data:
            return False

        # Delete the taxonomy file from filesystem
        taxonomy_file_path = taxonomy_data.get('file_path')
        if taxonomy_file_path:
            from pathlib import Path
            try:
                taxonomy_file_path_obj = Path(taxonomy_file_path)
                taxonomy_file_path_obj.unlink()
            except FileNotFoundError:
                pass
            except Exception as e:
                # Log but don't fail the operation
                logger.warning(f"Failed to delete taxonomy file {taxonomy_file_path}: {e}")

        # Delete the taxonomy record from database
        cursor = self.cursor()
        cursor.execute('DELETE FROM taxonomies WHERE id = ?', (taxonomy_id,))
        self.commit()
        return cursor.rowcount > 0

class ExtractionResultRepository(BaseRepository):
    """Repository for extraction result operations."""

    def get_results_by_chunk(self, chunk_id: str) -> List[Dict[str, Any]]:
        cursor = self.cursor()
        cursor.execute('SELECT * FROM extraction_results WHERE chunk_id = ? ORDER BY created_at DESC', (chunk_id,))
        rows = cursor.fetchall()
        return [dict(row) for row in rows] if rows else []

    def get_result_by_id(self, result_id: str) -> Optional[Dict[str, Any]]:
        cursor = self.cursor()
        cursor.execute('SELECT * FROM extraction_results WHERE id = ?', (result_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def create_result(self, job_id: str, chunk_id: str, categories: List[str], confidence: float, extracted_data: Dict[str, Any], project_id: str = None, id: str = None, created_at: datetime = None, result_type: str = "extraction", file_path: str = None) -> str:
        """Create a new extraction result."""
        result_id = id if id else str(uuid.uuid4())
        created_at = created_at or datetime.utcnow()

        cursor = self.cursor()
        cursor.execute('''
            INSERT INTO extraction_results (id, job_id, project_id, chunk_id, categories, confidence, extracted_data, result_type, file_path, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            result_id,
            job_id,
            project_id,
            chunk_id,
            json.dumps(categories),
            confidence,
            json.dumps(extracted_data),
            result_type,
            file_path,
            created_at,
            created_at
        ))
        # self.commit()  # Removed commit here to allow batch inserts in transaction
        return result_id

    def get_result_metadata(self, job_id: str) -> Dict[str, Any]:
        """Get metadata about results for a job."""
        cursor = self.cursor()

        # Get total count
        cursor.execute('SELECT COUNT(*) FROM extraction_results WHERE job_id = ?', (job_id,))
        total_results = cursor.fetchone()[0]

        if total_results == 0:
            return {
                'total_results': 0,
                'avg_confidence': 0.0,
                'categories': []
            }

        # Get average confidence
        cursor.execute('SELECT AVG(confidence) FROM extraction_results WHERE job_id = ?', (job_id,))
        avg_confidence = cursor.fetchone()[0] or 0.0

        # Get unique categories
        cursor.execute('SELECT categories FROM extraction_results WHERE job_id = ?', (job_id,))
        categories = set()
        for row in cursor.fetchall():
            if row[0]:
                try:
                    cats = json.loads(row[0]) if isinstance(row[0], str) else row[0]
                    if isinstance(cats, list):
                        categories.update(cats)
                except (json.JSONDecodeError, TypeError):
                    continue

        return {
            'total_results': total_results,
            'avg_confidence': float(avg_confidence),
            'categories': list(categories)
        }

    def delete_results_by_job(self, job_id: str) -> int:
        """Delete all results for a job."""
        cursor = self.cursor()
        cursor.execute('DELETE FROM extraction_results WHERE job_id = ?', (job_id,))
        self.commit()
        return cursor.rowcount

class ExtractionJobRepository(BaseRepository):
    """Repository for extraction job operations."""

    def get_jobs_by_project(self, project_id: str) -> List[Dict[str, Any]]:
        cursor = self.cursor()
        cursor.execute('SELECT * FROM extraction_jobs WHERE project_id = ? ORDER BY created_at DESC', (project_id,))
        rows = cursor.fetchall()
        return [dict(row) for row in rows] if rows else []

    def get_job_by_id(self, job_id: str) -> Optional[Dict[str, Any]]:
        cursor = self.cursor()
        cursor.execute('SELECT * FROM extraction_jobs WHERE id = ?', (job_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def create_job(self, job_data: Dict[str, Any]) -> str:
        cursor = self.cursor()
        cursor.execute('''
            INSERT INTO extraction_jobs (id, project_id, document_id, status, parameters, progress, created_at, updated_at, started_at, completed_at, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            job_data['id'],
            job_data['project_id'],
            job_data['document_id'],
            job_data['status'],
            json.dumps(job_data.get('parameters', {})),
            job_data.get('progress', 0.0),
            job_data['created_at'],
            job_data['updated_at'],
            job_data.get('started_at'),
            job_data.get('completed_at'),
            job_data.get('error_message')
        ))
        self.commit()
        return job_data['id']

    def update_job_status(self, job_id: str, status: str, error_message: str = None):
        """Update job status."""
        cursor = self.cursor()
        now = datetime.utcnow()

        if status == 'running' and not self._get_job_field(job_id, 'started_at'):
            # Set started_at when job starts running
            cursor.execute('''
                UPDATE extraction_jobs
                SET status = ?, updated_at = ?, started_at = ?
                WHERE id = ?
            ''', (status, now, now, job_id))
        elif status in ['completed', 'failed']:
            # Set completed_at when job finishes
            cursor.execute('''
                UPDATE extraction_jobs
                SET status = ?, updated_at = ?, completed_at = ?, error_message = ?
                WHERE id = ?
            ''', (status, now, now, error_message, job_id))
        else:
            # Regular status update
            cursor.execute('''
                UPDATE extraction_jobs
                SET status = ?, updated_at = ?, error_message = ?
                WHERE id = ?
            ''', (status, now, error_message, job_id))

        self.commit()

    def _get_job_field(self, job_id: str, field: str):
        """Get a specific field from a job record."""
        cursor = self.cursor()
        cursor.execute(f'SELECT {field} FROM extraction_jobs WHERE id = ?', (job_id,))
        row = cursor.fetchone()
        return row[0] if row else None

class DatasetVersionRepository(BaseRepository):
    """Repository for dataset version operations."""

    def create_version(self, version_data: Dict[str, Any]) -> str:
        """Create a new dataset version."""
        version_id = version_data.get('id', str(uuid.uuid4()))
        created_at = version_data.get('created_at', datetime.utcnow())

        cursor = self.cursor()
        cursor.execute('''
            INSERT INTO dataset_versions (
                id, project_id, version, major_version, minor_version, patch_version,
                dataset_name, description, total_entries, file_path, file_hash,
                metadata, is_active, created_at, created_by
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            version_id,
            version_data['project_id'],
            version_data['version'],
            version_data.get('major_version'),
            version_data.get('minor_version'),
            version_data.get('patch_version'),
            version_data.get('dataset_name'),
            version_data.get('description'),
            version_data.get('total_entries'),
            version_data.get('file_path'),
            version_data.get('file_hash'),
            json.dumps(version_data.get('metadata', {})),
            version_data.get('is_active', True),
            created_at,
            version_data.get('created_by')
        ))

        self.commit()
        return version_id

    def get_versions_by_project(self, project_id: Union[str, int], dataset_name: Optional[str] = None) -> List[Tuple]:
        cursor = self.cursor()
        if dataset_name:
            cursor.execute('''
                SELECT * FROM dataset_versions 
                WHERE project_id = ? AND dataset_name = ? 
                ORDER BY version DESC
            ''', (project_id, dataset_name))
        else:
            cursor.execute('''
                SELECT * FROM dataset_versions 
                WHERE project_id = ? 
                ORDER BY version DESC
            ''', (project_id,))
        return cursor.fetchall()

    def get_version_by_id(self, version_id: str) -> Optional[Dict[str, Any]]:
        cursor = self.cursor()
        cursor.execute('SELECT * FROM dataset_versions WHERE id = ?', (version_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def increment_version(self, project_id: int, dataset_name: str, version_type: str) -> str:
        """Increment version number."""
        versions = self.get_versions_by_project(project_id, dataset_name)
        if not versions:
            return "1.0.0"

        # Version string is in column index 1 (after id)
        latest_version = versions[0][1]
        
        try:
            major, minor, patch = map(int, latest_version.split('.'))
        except ValueError:
            return "1.0.0" # Fallback if version string is weird
        
        if version_type == "major":
            major += 1
            minor = 0
            patch = 0
        elif version_type == "minor":
            minor += 1
            patch = 0
        else: # patch
            patch += 1
            
        return f"{major}.{minor}.{patch}"

    def record_change(self, change_data: Dict[str, Any]) -> str:
        """Record a change to a dataset."""
        change_id = change_data.get('id', str(uuid.uuid4()))

        cursor = self.cursor()
        cursor.execute('''
            INSERT INTO dataset_changes (
                id, dataset_version_id, change_type, change_description,
                entries_affected, changed_by, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            change_id,
            change_data['dataset_version_id'],
            change_data['change_type'],
            change_data.get('change_description'),
            change_data.get('entries_affected'),
            change_data.get('changed_by'),
            change_data.get('created_at', datetime.utcnow())
        ))
        self.commit()
        return change_id

    def record_lineage(self, lineage_data: Dict[str, Any]) -> str:
        """Record lineage."""
        lineage_id = lineage_data.get('id', str(uuid.uuid4()))

        cursor = self.cursor()
        cursor.execute('''
            INSERT INTO dataset_lineage (
                id, dataset_version_id, source_type, source_id, source_name,
                source_hash, processing_parameters, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            lineage_id,
            lineage_data['dataset_version_id'],
            lineage_data['source_type'],
            lineage_data.get('source_id'),
            lineage_data.get('source_name'),
            lineage_data.get('source_hash'),
            json.dumps(lineage_data.get('processing_parameters', {})),
            lineage_data.get('created_at', datetime.utcnow())
        ))
        self.commit()
        return lineage_id
        
    def update_version_status(self, version_id: str, is_active: bool):
        """Update version status."""
        cursor = self.cursor()
        cursor.execute('UPDATE dataset_versions SET is_active = ? WHERE id = ?', (is_active, version_id))
        self.commit()

    def get_changes_for_version(self, version_id: str) -> List[Dict[str, Any]]:
        """Get all changes for a specific version."""
        cursor = self.cursor()
        cursor.execute('SELECT * FROM dataset_changes WHERE dataset_version_id = ? ORDER BY created_at DESC', (version_id,))
        rows = cursor.fetchall()
        return [dict(row) for row in rows] if rows else []

    def get_lineage_for_version(self, version_id: str) -> List[Dict[str, Any]]:
        """Get lineage information for a specific version."""
        cursor = self.cursor()
        cursor.execute('SELECT * FROM dataset_lineage WHERE dataset_version_id = ? ORDER BY created_at DESC', (version_id,))
        rows = cursor.fetchall()
        return [dict(row) for row in rows] if rows else []

class ProcessedOutputRepository(BaseRepository):
    """Repository for processed output operations."""

    def get_outputs_by_project(self, project_id: str) -> List[Dict[str, Any]]:
        cursor = self.cursor()
        cursor.execute('SELECT * FROM processed_outputs WHERE project_id = ? ORDER BY created_at DESC', (project_id,))
        rows = cursor.fetchall()
        return [dict(row) for row in rows] if rows else []

    def get_output_by_id(self, output_id: str) -> Optional[Dict[str, Any]]:
        cursor = self.cursor()
        cursor.execute('SELECT * FROM processed_outputs WHERE id = ?', (output_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def create_output(self, output_data: Dict[str, Any]) -> str:
        cursor = self.cursor()
        cursor.execute('''
            INSERT INTO processed_outputs (id, project_id, output_type, content, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            output_data['id'],
            output_data['project_id'],
            output_data['output_type'],
            json.dumps(output_data.get('content', {})),
            json.dumps(output_data.get('metadata', {})),
            output_data['created_at']
        ))
        self.commit()
        return output_data['id']

class BenchmarkJobRepository(BaseRepository):
    """Repository for benchmark job operations."""

    def get_all_jobs(self) -> List[Dict[str, Any]]:
        """Get all benchmark jobs."""
        cursor = self.cursor()
        cursor.execute('SELECT * FROM benchmark_jobs ORDER BY created_at DESC')
        rows = cursor.fetchall()
        return [dict(row) for row in rows] if rows else []

    def get_job_by_id(self, job_id: str) -> Optional[Dict[str, Any]]:
        cursor = self.cursor()
        cursor.execute('SELECT * FROM benchmark_jobs WHERE id = ?', (job_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def create_job(self, job_data: Dict[str, Any]) -> str:
        cursor = self.cursor()
        cursor.execute('''
            INSERT INTO benchmark_jobs (id, model_name, status, parameters, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            job_data['id'],
            job_data['model_name'],
            job_data['status'],
            json.dumps(job_data.get('parameters', {})),
            job_data['created_at'],
            job_data['updated_at']
        ))
        self.commit()
        return job_data['id']

    def update_job_status(self, job_id: str, status: str, error_message: str = None):
        """Update job status."""
        cursor = self.cursor()
        now = datetime.utcnow()

        if status == 'running' and not self._get_job_field(job_id, 'started_at'):
            # Set started_at when job starts running
            cursor.execute('''
                UPDATE benchmark_jobs
                SET status = ?, updated_at = ?, started_at = ?
                WHERE id = ?
            ''', (status, now, now, job_id))
        elif status in ['completed', 'failed']:
            # Set completed_at when job finishes
            cursor.execute('''
                UPDATE benchmark_jobs
                SET status = ?, updated_at = ?, completed_at = ?, error_message = ?
                WHERE id = ?
            ''', (status, now, now, error_message, job_id))
        else:
            # Regular status update
            cursor.execute('''
                UPDATE benchmark_jobs
                SET status = ?, updated_at = ?, error_message = ?
                WHERE id = ?
            ''', (status, now, error_message, job_id))

        self.commit()

    def _get_job_field(self, job_id: str, field: str):
        """Get a specific field from a job record."""
        cursor = self.cursor()
        cursor.execute(f'SELECT {field} FROM benchmark_jobs WHERE id = ?', (job_id,))
        row = cursor.fetchone()
        return row[0] if row else None

class BenchmarkResultRepository(BaseRepository):
    """Repository for benchmark result operations."""

    def get_results_by_job(self, job_id: str) -> List[Dict[str, Any]]:
        cursor = self.cursor()
        cursor.execute('SELECT * FROM benchmark_results WHERE job_id = ? ORDER BY created_at DESC', (job_id,))
        rows = cursor.fetchall()
        return [dict(row) for row in rows] if rows else []

    def get_result_by_id(self, result_id: str) -> Optional[Dict[str, Any]]:
        cursor = self.cursor()
        cursor.execute('SELECT * FROM benchmark_results WHERE id = ?', (result_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def create_result(self, result_data: Dict[str, Any]) -> str:
        cursor = self.cursor()
        cursor.execute('''
            INSERT INTO benchmark_results (id, job_id, metrics, metadata, created_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            result_data['id'],
            result_data['job_id'],
            json.dumps(result_data.get('metrics', {})),
            json.dumps(result_data.get('metadata', {})),
            result_data['created_at']
        ))
        self.commit()
        return result_data['id']

class ParsedDocumentRepository(BaseRepository):
    """Repository for parsed document operations."""

    def get_documents_by_project(self, project_id: str) -> List[Dict[str, Any]]:
        cursor = self.cursor()
        cursor.execute('SELECT * FROM parsed_documents WHERE project_id = ? ORDER BY created_at DESC', (project_id,))
        rows = cursor.fetchall()
        return [dict(row) for row in rows] if rows else []

    def get_document_by_id(self, document_id: str) -> Optional[Dict[str, Any]]:
        cursor = self.cursor()
        cursor.execute('SELECT * FROM parsed_documents WHERE id = ?', (document_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def create_document(self, document_data: Dict[str, Any]) -> str:
        cursor = self.cursor()
        cursor.execute('''
            INSERT INTO parsed_documents (id, project_id, document_id, filename, content, metadata, parser_used, parse_config, total_pages, parsing_time, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            document_data['id'],
            document_data['project_id'],
            document_data['document_id'],
            document_data['filename'],
            document_data['content'],
            json.dumps(document_data.get('metadata', {})),
            document_data.get('parser_used'),
            document_data.get('parse_config'),
            document_data.get('total_pages'),
            document_data.get('parsing_time'),
            document_data['created_at']
        ))
        self.commit()
        return document_data['id']

class ParsedFileRepository(BaseRepository):
    """Repository for parsed file operations."""

    def get_files_by_document(self, document_id: str) -> List[Dict[str, Any]]:
        cursor = self.cursor()
        cursor.execute('SELECT * FROM parsed_files WHERE document_id = ? ORDER BY created_at DESC', (document_id,))
        rows = cursor.fetchall()
        return [dict(row) for row in rows] if rows else []

    def get_file_by_id(self, file_id: str) -> Optional[Dict[str, Any]]:
        cursor = self.cursor()
        cursor.execute('SELECT * FROM parsed_files WHERE id = ?', (file_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def create_file(self, file_data: Dict[str, Any]) -> str:
        cursor = self.cursor()
        cursor.execute('''
            INSERT INTO parsed_files (id, parsed_document_id, filename, file_path, file_type, page_number, content_length, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            file_data['id'],
            file_data['parsed_document_id'],
            file_data['filename'],
            file_data['file_path'],
            file_data['file_type'],
            file_data.get('page_number'),
            file_data.get('content_length'),
            file_data['created_at']
        ))
        self.commit()
        return file_data['id']
