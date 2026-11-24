"""
Database repositories for project-related entities with multi-tenant support.

This module provides repository classes with thread-safe database operations
for use in multi-threaded and compiled environments. Each tenant has their own
SQLite database file for true data isolation.
"""

from typing import List, Optional, Dict, Any, Union, Tuple
import json
import uuid
import threading
from datetime import datetime
from contextlib import contextmanager

# Thread-local storage for database connections
_connection_local = threading.local()


class BaseRepository:
    """Base repository class handling tenant-specific database connections with thread safety."""

    def __init__(self, owner_id: str):
        """
        Initialize repository with tenant isolation.

        Args:
            owner_id: The user ID for tenant-specific database access
        """
        self.owner_id = owner_id
        # Import here to avoid circular dependency
        from ..database import get_db_lock
        self._db_lock = get_db_lock()

    @contextmanager
    def cursor(self):
        """
        Context manager for database cursor operations.
        Ensures proper connection handling and closing.
        """
        from ..database import get_tenant_db_connection
        # Use context manager to get connection, which handles closing/committing
        with get_tenant_db_connection(self.owner_id) as conn:
            cursor = conn.cursor()
            yield cursor
            # Commit the transaction before closing
            conn.commit()

    def commit(self):
        """
        Commit changes to the database.
        
        NOTE: When using the cursor() context manager, commits are handled automatically
        by the underlying get_tenant_db_connection context manager.
        This method is kept for compatibility but operations should prefer using
        the cursor context manager which handles transaction lifecycle.
        """
        pass

    @contextmanager
    def _locked_cursor(self):
        """Context manager for thread-safe cursor operations."""
        self._db_lock.acquire()
        try:
            with self.cursor() as cursor:
                yield cursor
        finally:
            self._db_lock.release()

class ProjectRepository(BaseRepository):
    """Repository for project operations."""

    def get_all_projects(self) -> List[Dict[str, Any]]:
        with self.cursor() as cursor:
            cursor.execute('SELECT * FROM projects WHERE owner_id = ? ORDER BY created_at DESC', (self.owner_id,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows] if rows else []

    def get_project_by_id(self, project_id: str) -> Optional[Dict[str, Any]]:
        with self.cursor() as cursor:
            cursor.execute('SELECT * FROM projects WHERE id = ? AND owner_id = ?', (project_id, self.owner_id))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_project_by_name(self, name: str) -> Optional[str]:
        with self.cursor() as cursor:
            cursor.execute('SELECT id FROM projects WHERE name = ? AND owner_id = ?', (name, self.owner_id))
            row = cursor.fetchone()
            return row[0] if row else None

    def create_project(self, name: Union[str, Dict[str, Any]], description: str = "") -> str:
        """Create a new project."""
        # Check if argument is a dict (legacy/compatibility)
        if isinstance(name, dict):
            project_data = name
            name = str(project_data.get('name', ''))
            description = project_data.get('description', '')
            project_id = project_data.get('id', str(uuid.uuid4()))
            created_at = project_data.get('created_at', datetime.utcnow())
            updated_at = project_data.get('updated_at', created_at)
        else:
            project_id = str(uuid.uuid4())
            created_at = datetime.utcnow()
            updated_at = created_at
        
        with self.cursor() as cursor:
            cursor.execute('''
                INSERT INTO projects (id, owner_id, name, description, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (project_id, self.owner_id, name, description, created_at, updated_at))
        return project_id

class DocumentRepository(BaseRepository):
    """Repository for document operations."""

    def get_documents_by_project_id(self, project_id: Union[str, int]) -> List[Dict[str, Any]]:
        """Get all documents for a project."""
        with self.cursor() as cursor:
            cursor.execute('SELECT * FROM documents WHERE project_id = ? AND owner_id = ? ORDER BY created_at DESC', (project_id, self.owner_id))
            rows = cursor.fetchall()
            return [dict(row) for row in rows] if rows else []
        
    def get_document_by_id(self, document_id: Union[str, int]) -> Optional[Dict[str, Any]]:
        """Get document by ID."""
        with self.cursor() as cursor:
            cursor.execute('SELECT * FROM documents WHERE id = ? AND owner_id = ?', (document_id, self.owner_id))
            row = cursor.fetchone()
            return dict(row) if row else None

    def create_document(self, project_id: Union[str, int, Dict[str, Any]], filename: str, file_path: str = "") -> str:
        """Create a new document."""
        # Handle if first arg is a dict
        if isinstance(project_id, dict):
            doc_data = project_id
            project_id = doc_data.get('project_id', '')
            filename = doc_data.get('file_name', '')
            file_path = doc_data.get('source_file_path', '')
        
        document_id = str(uuid.uuid4())
        
        with self.cursor() as cursor:
            cursor.execute('''
                INSERT INTO documents (id, owner_id, project_id, file_name, source_file_path, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (document_id, self.owner_id, project_id, filename, file_path, datetime.utcnow(), datetime.utcnow()))
        
        return document_id

    def update_document_path(self, document_id: Union[str, int], file_path: str):
        """Update document file path."""
        with self.cursor() as cursor:
            cursor.execute('UPDATE documents SET source_file_path = ? WHERE id = ?', (file_path, document_id))

    def delete_document(self, document_id: Union[str, int]):
        """Delete a document."""
        with self.cursor() as cursor:
            cursor.execute('DELETE FROM documents WHERE id = ? AND owner_id = ?', (document_id, self.owner_id))

    def update_page_selection(self, document_id: Union[str, int], page_selection: Optional[str]):
        """Update the page selection for a document."""
        with self.cursor() as cursor:
            cursor.execute(
                'UPDATE documents SET page_selection = ? WHERE id = ? AND owner_id = ?',
                (page_selection, document_id, self.owner_id)
            )

    def get_page_selection(self, document_id: Union[str, int]) -> Optional[str]:
        """Get the page selection for a document."""
        with self.cursor() as cursor:
            cursor.execute(
                'SELECT page_selection FROM documents WHERE id = ? AND owner_id = ?',
                (document_id, self.owner_id)
            )
            row = cursor.fetchone()
            return row[0] if row else None

class DatasetJobRepository(BaseRepository):
    """Repository for dataset job operations."""

    def get_jobs_by_project(self, project_id: str) -> List[Dict[str, Any]]:
        with self.cursor() as cursor:
            cursor.execute('SELECT * FROM dataset_jobs WHERE project_id = ? AND owner_id = ? ORDER BY created_at DESC', (project_id, self.owner_id))
            rows = cursor.fetchall()
            return [dict(row) for row in rows] if rows else []

    def get_job_by_id(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self.cursor() as cursor:
            cursor.execute('SELECT * FROM dataset_jobs WHERE id = ? AND owner_id = ?', (job_id, self.owner_id))
            row = cursor.fetchone()
            return dict(row) if row else None

    def create_job(self, job_data: Dict[str, Any]) -> str:
        with self.cursor() as cursor:
            cursor.execute('''
                INSERT INTO dataset_jobs (id, owner_id, project_id, job_type, status, parameters, progress, created_at, updated_at, started_at, completed_at, error_message, document_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                job_data['id'],
                job_data['owner_id'],
                job_data['project_id'],
                job_data['job_type'],
                job_data['status'],
                json.dumps(job_data.get('parameters', {})),
                job_data.get('progress', 0.0),
                job_data['created_at'],
                job_data['updated_at'],
                job_data.get('started_at'),
                job_data.get('completed_at'),
                job_data.get('error_message'),
                job_data.get('document_id')
            ))
        return job_data['id']

    def update_job_status(self, job_id: str, status: str, error_message: Optional[str] = None):
        """Update job status."""
        now = datetime.utcnow()
        with self.cursor() as cursor:
            if status == 'running':
                cursor.execute('''
                    UPDATE dataset_jobs
                    SET status = ?, updated_at = ?, started_at = ?
                    WHERE id = ? AND started_at IS NULL
                ''', (status, now, now, job_id))
            elif status in ['completed', 'failed']:
                cursor.execute('''
                    UPDATE dataset_jobs
                    SET status = ?, updated_at = ?, completed_at = ?, error_message = ?
                    WHERE id = ?
                ''', (status, now, now, error_message, job_id))
            else:
                cursor.execute('''
                    UPDATE dataset_jobs
                    SET status = ?, updated_at = ?, error_message = ?
                    WHERE id = ?
                ''', (status, now, error_message, job_id))

    def update_job_progress(self, job_id: str, progress: float):
        """Update job progress percentage."""
        with self.cursor() as cursor:
            cursor.execute('''
                UPDATE dataset_jobs
                SET progress = ?, updated_at = ?
                WHERE id = ?
            ''', (progress, datetime.utcnow(), job_id))

    def update_job_result(self, job_id: str, result: Dict[str, Any]):
        """Update job result."""
        with self.cursor() as cursor:
            cursor.execute('''
                UPDATE dataset_jobs
                SET result = ?, updated_at = ?
                WHERE id = ?
            ''', (json.dumps(result), datetime.utcnow(), job_id))

    def archive_completed_job(self, job_id: str, history_repo: 'JobHistoryRepository') -> Optional[str]:
        """
        Archive a completed/failed job to history and remove from active jobs.
        Returns the history entry ID if archived.
        """
        job = self.get_job_by_id(job_id)
        if not job or job['status'] not in ('completed', 'failed'):
            return None
        
        # Create history entry
        history_id = history_repo.create_history_entry(job)
        
        # Delete from active jobs (but keep in history)
        with self.cursor() as cursor:
            cursor.execute('DELETE FROM dataset_jobs WHERE id = ?', (job_id,))
        
        return history_id

    def get_all_jobs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all jobs for the owner, ordered by most recent."""
        with self.cursor() as cursor:
            cursor.execute('''
                SELECT * FROM dataset_jobs 
                WHERE owner_id = ? 
                ORDER BY created_at DESC 
                LIMIT ?
            ''', (self.owner_id, limit))
            rows = cursor.fetchall()
            return [dict(row) for row in rows] if rows else []

class DatasetParameterRepository(BaseRepository):
    """Repository for dataset parameter operations."""

    def get_parameters_by_project(self, project_id: str, **kwargs) -> List[Dict[str, Any]]:
        with self.cursor() as cursor:
            cursor.execute('SELECT * FROM dataset_parameters WHERE project_id = ? AND owner_id = ? ORDER BY created_at DESC', (project_id, self.owner_id))
            rows = cursor.fetchall()
            return [dict(row) for row in rows] if rows else []

    def create(self, parameter_data: Dict[str, Any]) -> str:
        with self.cursor() as cursor:
            cursor.execute('''
                INSERT INTO dataset_parameters (
                    id, owner_id, project_id, purpose, audience, extraction_rules,
                    dataset_format, question_style, answer_style, negativity_ratio,
                    data_augmentation, custom_audience, custom_purpose,
                    complexity_level, domain, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                parameter_data['id'],
                parameter_data['owner_id'],
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
        return parameter_data['id']

class PromptRepository(BaseRepository):
    """Repository for prompt operations."""

    def get_prompts_by_project(self, project_id: str) -> List[Dict[str, Any]]:
        with self.cursor() as cursor:
            cursor.execute('SELECT * FROM prompts WHERE project_id = ? AND owner_id = ? ORDER BY created_at DESC', (project_id, self.owner_id))
            rows = cursor.fetchall()
            return [dict(row) for row in rows] if rows else []

    def get_by_name(self, name: str, project_id: Optional[str] = None, owner_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        # Use locked cursor for thread safety during concurrent dataset generation
        with self._locked_cursor() as cursor:
            if project_id and owner_id:
                # Try to find specific override first
                cursor.execute('SELECT * FROM prompts WHERE name = ? AND project_id = ? AND owner_id = ?', (name, project_id, self.owner_id))
                row = cursor.fetchone()
                if row:
                    return dict(row)
                
                # Fallback to system default
                cursor.execute('SELECT * FROM prompts WHERE name = ? AND owner_id = ?', (name, 'system'))
                row = cursor.fetchone()
                if row:
                    return dict(row)
                    
            # For backward compatibility or if no context provided, just search by name
            cursor.execute('SELECT * FROM prompts WHERE name = ?', (name,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def create(self, prompt_data: Dict[str, Any]) -> str:
        with self.cursor() as cursor:
            cursor.execute('''
                INSERT INTO prompts (id, owner_id, project_id, name, content, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                prompt_data['id'],
                prompt_data['owner_id'],
                str(prompt_data['project_id']),
                prompt_data['name'],
                prompt_data['content'],
                prompt_data['created_at'],
                prompt_data['updated_at']
            ))
        return prompt_data['id']

class ChunkRepository(BaseRepository):
    """Repository for chunk operations."""

    def get_chunks_by_document(self, document_id: str) -> List[Dict[str, Any]]:
        with self.cursor() as cursor:
            cursor.execute('''
                SELECT * FROM chunks
                WHERE document_id = ? AND owner_id = ?
                ORDER BY chunk_index
            ''', (document_id, self.owner_id))
            rows = cursor.fetchall()
            return [dict(row) for row in rows] if rows else []

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get a chunk by ID."""
        with self.cursor() as cursor:
            cursor.execute('''
                SELECT * FROM chunks
                WHERE id = ? AND owner_id = ?
            ''', (chunk_id, self.owner_id))
            row = cursor.fetchone()
            return dict(row) if row else None

    def delete_chunk(self, chunk_id: str) -> bool:
        """Delete a chunk by ID."""
        with self.cursor() as cursor:
            cursor.execute('''
                DELETE FROM chunks
                WHERE id = ? AND owner_id = ?
            ''', (chunk_id, self.owner_id))
            return cursor.rowcount > 0

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

        with self.cursor() as cursor:
            cursor.execute('''
                INSERT INTO chunks (id, owner_id, document_id, chunk_index, content_preview, metadata, file_path, chunk_strategy, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                chunk_data['id'],
                chunk_data['owner_id'],
                chunk_data['document_id'],
                chunk_data['chunk_index'],
                content_preview,
                json.dumps(metadata),
                file_path,
                chunk_strategy,
                chunk_data.get('status', 'active'),
                chunk_data['created_at']
            ))
        return chunk_data['id']

class TaxonomyRepository(BaseRepository):
    """Repository for taxonomy operations."""

    def get_taxonomies_by_project(self, project_id: str) -> List[Dict[str, Any]]:
        with self.cursor() as cursor:
            cursor.execute('SELECT * FROM taxonomies WHERE project_id = ? AND owner_id = ? ORDER BY created_at DESC', (project_id, self.owner_id))
            rows = cursor.fetchall()
            return [dict(row) for row in rows] if rows else []

    def get_taxonomy_by_id(self, taxonomy_id: str) -> Optional[Dict[str, Any]]:
        with self.cursor() as cursor:
            cursor.execute('SELECT * FROM taxonomies WHERE id = ? AND owner_id = ?', (taxonomy_id, self.owner_id))
            row = cursor.fetchone()
            return dict(row) if row else None

    def create_taxonomy(self, taxonomy_data: Dict[str, Any]) -> str:
        with self.cursor() as cursor:
            cursor.execute('''
                INSERT INTO taxonomies (id, owner_id, project_id, name, structure, file_path, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                taxonomy_data['id'],
                taxonomy_data['owner_id'],
                taxonomy_data['project_id'],
                taxonomy_data['name'],
                json.dumps(taxonomy_data.get('structure', {})),
                taxonomy_data.get('file_path'),
                taxonomy_data['created_at'],
                taxonomy_data['updated_at']
            ))
        return taxonomy_data['id']

    def delete_taxonomy(self, taxonomy_id: str) -> bool:
        """Delete a taxonomy by ID for the current owner."""
        with self.cursor() as cursor:
            cursor.execute(
                'DELETE FROM taxonomies WHERE id = ? AND owner_id = ?',
                (taxonomy_id, self.owner_id)
            )
            return cursor.rowcount > 0

    def update_taxonomy_structure(self, taxonomy_id: str, structure: Dict[str, Any]) -> bool:
        """Update taxonomy structure JSON for a taxonomy owned by current user."""
        with self.cursor() as cursor:
            cursor.execute(
                'UPDATE taxonomies SET structure = ?, updated_at = ? WHERE id = ? AND owner_id = ?',
                (json.dumps(structure), datetime.utcnow(), taxonomy_id, self.owner_id)
            )
            return cursor.rowcount > 0


class EmbedRepository(BaseRepository):
    """Repository for embedding plugin chunk-focused taxonomy cache."""

    def get_embed_by_chunk_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        with self.cursor() as cursor:
            cursor.execute(
                'SELECT * FROM embed WHERE owner_id = ? AND chunk_id = ? LIMIT 1',
                (self.owner_id, chunk_id)
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_embed_by_chunk_ids(self, chunk_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        if not chunk_ids:
            return {}
        placeholders = ','.join(['?'] * len(chunk_ids))
        with self.cursor() as cursor:
            cursor.execute(
                f'SELECT * FROM embed WHERE owner_id = ? AND chunk_id IN ({placeholders})',
                [self.owner_id] + chunk_ids
            )
            rows = cursor.fetchall() or []
        return {str(r['chunk_id']): dict(r) for r in rows}

    def upsert_embed_record(self, record: Dict[str, Any]) -> str:
        record_id = str(record.get('id') or uuid.uuid4())
        taxonomy_tree = record.get('taxonomy_tree')
        if isinstance(taxonomy_tree, str):
            try:
                taxonomy_tree = json.loads(taxonomy_tree)
            except Exception:
                taxonomy_tree = []

        leaf_ids = record.get('leaf_ids') or []
        if isinstance(leaf_ids, str):
            try:
                leaf_ids = json.loads(leaf_ids)
            except Exception:
                leaf_ids = []

        with self.cursor() as cursor:
            cursor.execute(
                '''
                INSERT INTO embed (
                    id, owner_id, project_id, document_id, chunk_id, chunk_index,
                    chunk_hash, taxonomy_tree, leaf_ids, taxonomy_provider, taxonomy_model,
                    generation_mode, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(owner_id, chunk_id) DO UPDATE SET
                    project_id = excluded.project_id,
                    document_id = excluded.document_id,
                    chunk_index = excluded.chunk_index,
                    chunk_hash = excluded.chunk_hash,
                    taxonomy_tree = excluded.taxonomy_tree,
                    leaf_ids = excluded.leaf_ids,
                    taxonomy_provider = excluded.taxonomy_provider,
                    taxonomy_model = excluded.taxonomy_model,
                    generation_mode = excluded.generation_mode,
                    updated_at = excluded.updated_at
                ''',
                (
                    record_id,
                    self.owner_id,
                    record.get('project_id'),
                    record.get('document_id'),
                    record.get('chunk_id'),
                    record.get('chunk_index'),
                    record.get('chunk_hash'),
                    json.dumps(taxonomy_tree or []),
                    json.dumps(leaf_ids or []),
                    record.get('taxonomy_provider'),
                    record.get('taxonomy_model'),
                    record.get('generation_mode', 'chunk'),
                    record.get('created_at', datetime.utcnow()),
                    record.get('updated_at', datetime.utcnow()),
                )
            )
        return record_id

class ExtractionResultRepository(BaseRepository):
    """Repository for extraction result operations."""

    def get_results_by_chunk(self, chunk_id: str) -> List[Dict[str, Any]]:
        with self.cursor() as cursor:
            cursor.execute('SELECT * FROM extraction_results WHERE chunk_id = ? AND owner_id = ? ORDER BY created_at DESC', (chunk_id, self.owner_id))
            rows = cursor.fetchall()
            return [dict(row) for row in rows] if rows else []

    def get_results_by_project(self, project_id: str, owner_id: str, min_confidence: float = 0.0, limit: Optional[int] = None, offset: Optional[int] = 0) -> List[Dict[str, Any]]:
        """Get all extraction results for a project."""
        with self.cursor() as cursor:
            query = 'SELECT * FROM extraction_results WHERE project_id = ? AND owner_id = ? AND confidence >= ? ORDER BY created_at DESC'
            params = [project_id, owner_id, min_confidence]
            
            if limit is not None:
                query += ' LIMIT ?'
                params.append(limit)
            if offset:
                query += ' OFFSET ?'
                params.append(offset)
                
            cursor.execute(query, tuple(params))
            rows = cursor.fetchall()
            return [dict(row) for row in rows] if rows else []

    def get_results_by_job(self, job_id: str, owner_id: str, page: int = 1, page_size: int = 100, min_confidence: float = 0.0) -> Tuple[List[Dict[str, Any]], int]:
        """Get extraction results for a specific job with pagination."""
        with self.cursor() as cursor:
            
            # Get total count
            cursor.execute('SELECT COUNT(*) FROM extraction_results WHERE job_id = ? AND owner_id = ? AND confidence >= ?', (job_id, self.owner_id, min_confidence))
            total = cursor.fetchone()[0]
            
            # Get results
            offset = (page - 1) * page_size
            cursor.execute('SELECT * FROM extraction_results WHERE job_id = ? AND owner_id = ? AND confidence >= ? ORDER BY created_at DESC LIMIT ? OFFSET ?', 
                           (job_id, self.owner_id, min_confidence, page_size, offset))
            rows = cursor.fetchall()
            
            return [dict(row) for row in rows] if rows else [], total

    def get_result_metadata(self, job_id: str) -> Dict[str, Any]:
        """Get summary metadata for extraction results of a job."""
        with self.cursor() as cursor:
            
            # Get total count
            cursor.execute('SELECT COUNT(*) FROM extraction_results WHERE job_id = ?', (job_id,))
            total_results = cursor.fetchone()[0]
            
            # Get avg confidence
            cursor.execute('SELECT AVG(confidence) FROM extraction_results WHERE job_id = ?', (job_id,))
            avg_confidence = cursor.fetchone()[0] or 0.0
            
            # Get categories found (unique list from JSON arrays)
            cursor.execute('SELECT categories FROM extraction_results WHERE job_id = ?', (job_id,))
            rows = cursor.fetchall()
            
            unique_categories = set()
            for row in rows:
                try:
                    cats = json.loads(row[0])
                    if isinstance(cats, list):
                        unique_categories.update(cats)
                except:
                    pass
                    
            return {
                "total_results": total_results,
                "avg_confidence": round(avg_confidence, 3),
                "categories_found": list(unique_categories)
            }

    def create_result(self, result_data: Dict[str, Any]) -> str:
        with self.cursor() as cursor:
            cursor.execute('''
                INSERT INTO extraction_results (id, owner_id, project_id, job_id, chunk_id, categories, confidence, extracted_data, file_path, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result_data['id'],
                result_data['owner_id'],
                result_data['project_id'],
                result_data.get('job_id'),
                result_data['chunk_id'],
                json.dumps(result_data.get('categories', [])),
                result_data.get('confidence', 0.0),
                json.dumps(result_data.get('extracted_data', {})),
                result_data.get('file_path'),  # Store path to filesystem data
                result_data.get('created_at', datetime.utcnow())
            ))
        return result_data['id']

class DatasetChangeRepository(BaseRepository):
    """Repository for dataset change operations."""

    def create_change(self, change_data: Dict[str, Any]) -> str:
        with self.cursor() as cursor:
            cursor.execute('''
                INSERT INTO dataset_changes (id, owner_id, dataset_version_id, change_type, change_description, entries_affected, changed_by, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                change_data['id'],
                change_data['owner_id'],
                change_data['dataset_version_id'],
                change_data['change_type'],
                change_data['change_description'],
                change_data.get('entries_affected'),
                change_data.get('changed_by'),
                change_data.get('created_at', datetime.utcnow())
            ))
        return change_data['id']

class DatasetLineageRepository(BaseRepository):
    """Repository for dataset lineage operations."""

    def create_lineage(self, lineage_data: Dict[str, Any]) -> str:
        with self.cursor() as cursor:
            cursor.execute('''
                INSERT INTO dataset_lineage (id, owner_id, dataset_version_id, source_type, source_id, source_name, source_hash, processing_parameters, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                lineage_data['id'],
                lineage_data['owner_id'],
                lineage_data['dataset_version_id'],
                lineage_data['source_type'],
                lineage_data.get('source_id'),
                lineage_data.get('source_name'),
                lineage_data.get('source_hash'),
                json.dumps(lineage_data.get('processing_parameters', {})),
                lineage_data.get('created_at', datetime.utcnow())
            ))
        return lineage_data['id']

class ExtractionJobRepository(BaseRepository):
    """Repository for extraction job operations."""

    def get_jobs_by_project(self, project_id: str, owner_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get extraction jobs by project, optionally filtering by owner_id."""
        # Use provided owner_id or fall back to repository's owner_id
        filter_owner_id = owner_id if owner_id is not None else self.owner_id
        with self.cursor() as cursor:
            cursor.execute('SELECT * FROM extraction_jobs WHERE project_id = ? AND owner_id = ? ORDER BY created_at DESC', (project_id, filter_owner_id))
            rows = cursor.fetchall()
            return [dict(row) for row in rows] if rows else []

    def get_job_by_id(self, job_id: str, owner_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get extraction job by ID, optionally filtering by owner_id."""
        # Use provided owner_id or fall back to repository's owner_id
        filter_owner_id = owner_id if owner_id is not None else self.owner_id
        with self.cursor() as cursor:
            cursor.execute('SELECT * FROM extraction_jobs WHERE id = ? AND owner_id = ?', (job_id, filter_owner_id))
            row = cursor.fetchone()
            return dict(row) if row else None

    def create_job(self, job_data: Dict[str, Any]) -> str:
        try:
            with self.cursor() as cursor:
                cursor.execute('''
                    INSERT INTO extraction_jobs (id, owner_id, project_id, document_id, status, parameters, progress, created_at, updated_at, started_at, completed_at, error_message)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    job_data['id'],
                    job_data['owner_id'],
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
            return job_data['id']
        except Exception as e:
            raise

    def update_job_status(self, job_id: str, status: str, error_message: Optional[str] = None):
        now = datetime.utcnow()
        with self.cursor() as cursor:
            if status == 'running':
                cursor.execute('UPDATE extraction_jobs SET status = ?, updated_at = ?, started_at = ? WHERE id = ?', (status, now, now, job_id))
            elif status in ['completed', 'failed']:
                cursor.execute('UPDATE extraction_jobs SET status = ?, updated_at = ?, completed_at = ?, error_message = ? WHERE id = ?', (status, now, now, error_message, job_id))
            else:
                cursor.execute('UPDATE extraction_jobs SET status = ?, updated_at = ? WHERE id = ?', (status, now, job_id))

    def archive_completed_job(self, job_id: str, history_repo: 'JobHistoryRepository') -> Optional[str]:
        """
        Archive a completed/failed extraction job to history and remove from active jobs.
        Returns the history entry ID if archived.
        """
        job = self.get_job_by_id(job_id)
        if not job or job['status'] not in ('completed', 'failed'):
            return None
        
        # Create history entry
        history_id = history_repo.create_history_entry(job)
        
        # Delete from active jobs (but keep in history)
        with self.cursor() as cursor:
            cursor.execute('DELETE FROM extraction_jobs WHERE id = ?', (job_id,))
        
        return history_id

class DatasetVersionRepository(BaseRepository):
    """Repository for dataset version operations."""

    def create_version(self, version_data: Dict[str, Any]) -> str:
        with self.cursor() as cursor:
            cursor.execute('''
                INSERT INTO dataset_versions (
                    id, owner_id, project_id, version, major_version, minor_version, patch_version,
                    dataset_name, description, total_entries, file_path, file_hash,
                    metadata, is_active, created_at, created_by
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                version_data['id'],
                version_data['owner_id'],
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
                version_data['created_at'],
                version_data.get('created_by')
            ))
        return version_data['id']

    def get_versions_by_project(self, project_id: str) -> List[Dict[str, Any]]:
        with self.cursor() as cursor:
            cursor.execute('SELECT * FROM dataset_versions WHERE project_id = ? AND owner_id = ? ORDER BY version DESC', (project_id, self.owner_id))
            rows = cursor.fetchall()
            return [dict(row) for row in rows] if rows else []

class ProcessedOutputRepository(BaseRepository):
    """Repository for processed output operations."""

    def get_outputs_by_project(self, project_id: str) -> List[Dict[str, Any]]:
        with self.cursor() as cursor:
            cursor.execute('SELECT * FROM processed_outputs WHERE project_id = ? AND owner_id = ? ORDER BY created_at DESC', (project_id, self.owner_id))
            rows = cursor.fetchall()
            return [dict(row) for row in rows] if rows else []

    def create_output(self, output_data: Dict[str, Any]) -> str:
        with self.cursor() as cursor:
            cursor.execute('''
                INSERT INTO processed_outputs (id, owner_id, project_id, output_type, content, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                output_data['id'],
                output_data['owner_id'],
                output_data['project_id'],
                output_data['output_type'],
                json.dumps(output_data.get('content', {})),
                json.dumps(output_data.get('metadata', {})),
                output_data['created_at']
            ))
        return output_data['id']

class ParsedDocumentRepository(BaseRepository):
    """Repository for parsed document operations."""

    def get_documents_by_project(self, project_id: str) -> List[Dict[str, Any]]:
        with self.cursor() as cursor:
            cursor.execute('SELECT * FROM parsed_documents WHERE project_id = ? AND owner_id = ? ORDER BY created_at DESC', (project_id, self.owner_id))
            rows = cursor.fetchall()
            return [dict(row) for row in rows] if rows else []

    def create_document(self, document_data: Dict[str, Any]) -> str:
        with self.cursor() as cursor:
            cursor.execute('''
                INSERT INTO parsed_documents (id, owner_id, project_id, document_id, filename, content, metadata, parser_used, parse_config, total_pages, parsing_time, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                document_data['id'],
                document_data['owner_id'],
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
        return document_data['id']

class ParsedFileRepository(BaseRepository):
    """Repository for parsed file operations."""

    def get_files_by_document(self, parsed_document_id: str) -> List[Dict[str, Any]]:
        with self.cursor() as cursor:
            cursor.execute('SELECT * FROM parsed_files WHERE parsed_document_id = ? AND owner_id = ? ORDER BY page_number', (parsed_document_id, self.owner_id))
            rows = cursor.fetchall()
            return [dict(row) for row in rows] if rows else []

    def create_file(self, file_data: Dict[str, Any]) -> str:
        with self.cursor() as cursor:
            cursor.execute('''
                INSERT INTO parsed_files (id, owner_id, parsed_document_id, filename, file_path, file_type, page_number, content_length, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                file_data['id'],
                file_data['owner_id'],
                file_data['parsed_document_id'],
                file_data['filename'],
                file_data['file_path'],
                file_data['file_type'],
                file_data.get('page_number'),
                file_data.get('content_length'),
                file_data['created_at']
            ))
        return file_data['id']

class BenchmarkJobRepository(BaseRepository):
    """Repository for benchmark job operations."""

    def get_jobs_by_owner(self) -> List[Dict[str, Any]]:
        with self.cursor() as cursor:
            cursor.execute('SELECT * FROM benchmark_jobs WHERE owner_id = ? ORDER BY created_at DESC', (self.owner_id,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows] if rows else []

    def get_job_by_id(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self.cursor() as cursor:
            cursor.execute('SELECT * FROM benchmark_jobs WHERE id = ? AND owner_id = ?', (job_id, self.owner_id))
            row = cursor.fetchone()
            return dict(row) if row else None

    def create_job(self, job_data: Dict[str, Any]) -> str:
        with self.cursor() as cursor:
            cursor.execute('''
                INSERT INTO benchmark_jobs (id, owner_id, model_name, status, parameters, progress, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                job_data['id'],
                job_data['owner_id'],
                job_data['model_name'],
                job_data['status'],
                json.dumps(job_data.get('parameters', {})),
                job_data.get('progress', 0.0),
                job_data.get('created_at', datetime.utcnow()),
                job_data.get('updated_at', datetime.utcnow())
            ))
        return job_data['id']

    def update_job_status(self, job_id: str, status: str, error_message: Optional[str] = None):
        now = datetime.utcnow()
        with self.cursor() as cursor:
            if status == 'running':
                cursor.execute('UPDATE benchmark_jobs SET status = ?, updated_at = ?, started_at = ? WHERE id = ?', (status, now, now, job_id))
            elif status in ['completed', 'failed']:
                cursor.execute('UPDATE benchmark_jobs SET status = ?, updated_at = ?, completed_at = ?, error_message = ? WHERE id = ?', (status, now, now, error_message, job_id))
            else:
                cursor.execute('UPDATE benchmark_jobs SET status = ?, updated_at = ? WHERE id = ?', (status, now, job_id))

class BenchmarkResultRepository(BaseRepository):
    """Repository for benchmark result operations."""

    def create_result(self, result_data: Dict[str, Any]) -> str:
        with self.cursor() as cursor:
            cursor.execute('''
                INSERT INTO benchmark_results (id, owner_id, job_id, metrics, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                result_data['id'],
                result_data['owner_id'],
                result_data['job_id'],
                json.dumps(result_data.get('metrics', {})),
                json.dumps(result_data.get('metadata', {})),
                result_data.get('created_at', datetime.utcnow())
            ))
        return result_data['id']

class PluginRepository(BaseRepository):
    """Repository for plugin operations."""

    def get_all_plugins(self) -> List[Dict[str, Any]]:
        with self.cursor() as cursor:
            cursor.execute('SELECT * FROM plugins WHERE owner_id = ? ORDER BY installed_at DESC', (self.owner_id,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows] if rows else []

    def get_plugin_by_id(self, plugin_id: str) -> Optional[Dict[str, Any]]:
        with self.cursor() as cursor:
            cursor.execute('SELECT * FROM plugins WHERE id = ? AND owner_id = ?', (plugin_id, self.owner_id))
            row = cursor.fetchone()
            return dict(row) if row else None

    def create_plugin(self, plugin_data: Dict[str, Any]) -> str:
        with self.cursor() as cursor:
            cursor.execute('''
                INSERT INTO plugins (id, owner_id, name, version, author, description, entry_point, installed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                plugin_data['id'],
                plugin_data['owner_id'],
                plugin_data['name'],
                plugin_data['version'],
                plugin_data.get('author'),
                plugin_data.get('description'),
                plugin_data['entry_point'],
                plugin_data.get('installed_at', datetime.utcnow())
            ))
        return plugin_data['id']

    def delete_plugin(self, plugin_id: str):
        with self.cursor() as cursor:
            cursor.execute('DELETE FROM plugins WHERE id = ? AND owner_id = ?', (plugin_id, self.owner_id))


class JobHistoryRepository(BaseRepository):
    """Repository for job history operations - retains metadata of completed jobs."""

    def create_history_entry(self, job_data: Dict[str, Any]) -> str:
        """
        Create a job history entry from job data.
        Stores only metadata, not full job output data.
        """
        # Calculate duration if both timestamps exist
        duration_seconds = None
        started_at = job_data.get('started_at')
        completed_at = job_data.get('completed_at')
        if started_at and completed_at:
            try:
                if isinstance(started_at, str):
                    started_at = datetime.fromisoformat(started_at.replace('Z', '+00:00'))
                if isinstance(completed_at, str):
                    completed_at = datetime.fromisoformat(completed_at.replace('Z', '+00:00'))
                duration_seconds = int((completed_at - started_at).total_seconds())
            except:
                pass

        # Extract result summary - store only key metrics, not full data
        result_summary = None
        result = job_data.get('result')
        if result:
            try:
                if isinstance(result, str):
                    result = json.loads(result)
                # Store only summary metadata
                if isinstance(result, dict):
                    result_summary = json.dumps({
                        'entries_count': result.get('entries_count'),
                        'total_batches': result.get('total_batches'),
                        'dataset_id': result.get('dataset_id')
                    })
            except:
                pass

        # Extract error summary - truncate if too long
        error_summary = None
        error_message = job_data.get('error_message')
        if error_message:
            error_summary = error_message[:500] if len(error_message) > 500 else error_message

        history_id = str(uuid.uuid4())
        with self.cursor() as cursor:
            cursor.execute('''
                INSERT INTO job_history (
                    id, owner_id, project_id, document_id, job_type, status, parameters,
                    progress, result_summary, error_summary, started_at, completed_at,
                    duration_seconds, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                history_id,
                job_data['owner_id'],
                job_data.get('project_id'),
                job_data.get('document_id'),
                job_data.get('job_type', 'unknown'),
                job_data['status'],
                json.dumps(job_data.get('parameters', {})),
                job_data.get('progress', 0.0),
                result_summary,
                error_summary,
                job_data.get('started_at'),
                job_data.get('completed_at'),
                duration_seconds,
                datetime.utcnow()
            ))
        return history_id

    def get_history_by_project(self, project_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get job history for a project, ordered by most recent."""
        with self.cursor() as cursor:
            cursor.execute('''
                SELECT * FROM job_history 
                WHERE project_id = ? AND owner_id = ? 
                ORDER BY created_at DESC 
                LIMIT ?
            ''', (project_id, self.owner_id, limit))
            rows = cursor.fetchall()
            return [dict(row) for row in rows] if rows else []

    def get_history_by_job_type(self, job_type: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get job history by type, ordered by most recent."""
        with self.cursor() as cursor:
            cursor.execute('''
                SELECT * FROM job_history 
                WHERE job_type = ? AND owner_id = ? 
                ORDER BY created_at DESC 
                LIMIT ?
            ''', (job_type, self.owner_id, limit))
            rows = cursor.fetchall()
            return [dict(row) for row in rows] if rows else []

    def get_history_entry(self, history_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific history entry by ID."""
        with self.cursor() as cursor:
            cursor.execute('''
                SELECT * FROM job_history 
                WHERE id = ? AND owner_id = ?
            ''', (history_id, self.owner_id))
            row = cursor.fetchone()
            return dict(row) if row else None

    def cleanup_old_history(self, days_to_keep: int = 90) -> int:
        """
        Clean up job history entries older than specified days.
        Returns number of entries deleted.
        """
        from datetime import timedelta
        cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
        
        with self.cursor() as cursor:
            cursor.execute('''
                DELETE FROM job_history 
                WHERE owner_id = ? AND created_at < ?
            ''', (self.owner_id, cutoff_date))
            return cursor.rowcount

    def archive_job(self, job_id: str, job_repo) -> Optional[str]:
        """
        Archive a job from the active jobs table to history.
        Returns the history entry ID if successful.
        """
        # Get the job data from the active repository
        job_data = job_repo.get_job_by_id(job_id)
        if not job_data:
            return None
        
        # Only archive completed or failed jobs
        if job_data['status'] not in ('completed', 'failed'):
            return None
        
        # Create history entry
        history_id = self.create_history_entry(job_data)
        return history_id

    def get_all_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all job history for the owner, ordered by most recent."""
        with self.cursor() as cursor:
            cursor.execute('''
                SELECT * FROM job_history 
                WHERE owner_id = ? 
                ORDER BY created_at DESC 
                LIMIT ?
            ''', (self.owner_id, limit))
            rows = cursor.fetchall()
            return [dict(row) for row in rows] if rows else []

    def get_job_statistics(self, project_id: Optional[str] = None) -> Dict[str, Any]:
        """Get job statistics for the owner, optionally filtered by project."""
        with self.cursor() as cursor:
            if project_id:
                cursor.execute('''
                    SELECT 
                        COUNT(*) as total_jobs,
                        AVG(duration_seconds) as avg_duration,
                        SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed_count,
                        SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed_count
                    FROM job_history 
                    WHERE owner_id = ? AND project_id = ?
                ''', (self.owner_id, project_id))
            else:
                cursor.execute('''
                    SELECT 
                        COUNT(*) as total_jobs,
                        AVG(duration_seconds) as avg_duration,
                        SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed_count,
                        SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed_count
                    FROM job_history 
                    WHERE owner_id = ?
                ''', (self.owner_id,))
            
            row = cursor.fetchone()
            if row:
                return {
                    'total_jobs': row[0] or 0,
                    'avg_duration_seconds': round(row[1], 2) if row[1] else 0,
                    'completed_count': row[2] or 0,
                    'failed_count': row[3] or 0
                }
            return {'total_jobs': 0, 'avg_duration_seconds': 0, 'completed_count': 0, 'failed_count': 0}
