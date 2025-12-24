"""
Enhanced search capabilities for extraction results.
Provides full-text search, filtering, and advanced query operations.
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json

from .models import ExtractionResult


class ResultSearchEngine:
    """Advanced search engine for extraction results."""

    def __init__(self, db_connection):
        self.db_connection = db_connection

    def full_text_search(
        self,
        query: str,
        categories: Optional[List[str]] = None,
        min_confidence: Optional[float] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        limit: int = 50,
        offset: int = 0
    ) -> Tuple[List[ExtractionResult], int]:
        """
        Perform full-text search across extraction results.

        Args:
            query: Search query string
            categories: Filter by categories
            min_confidence: Minimum confidence threshold
            date_from: Start date filter
            date_to: End date filter
            limit: Maximum results to return
            offset: Pagination offset

        Returns:
            Tuple of (results, total_count)
        """
        cursor = self.db_connection.cursor()

        # Build the WHERE clause
        where_conditions = []
        params = []

        # Full-text search on extracted_data (JSON)
        if query:
            # Simple text search in JSON fields
            # Note: extracted_data might be empty in database, so this search relies on metadata or categories
            where_conditions.append("(extracted_data::text ILIKE ? OR categories::text ILIKE ?)")
            search_pattern = f"%{query}%"
            params.extend([search_pattern, search_pattern])

        # Category filter
        if categories:
            category_conditions = []
            for category in categories:
                category_conditions.append("categories::text ILIKE ?")
                params.append(f"%{category}%")
            where_conditions.append(f"({' OR '.join(category_conditions)})")

        # Confidence filter
        if min_confidence is not None:
            where_conditions.append("confidence >= ?")
            params.append(min_confidence)

        # Date filters
        if date_from:
            where_conditions.append("created_at >= ?")
            params.append(date_from)
        if date_to:
            where_conditions.append("created_at <= ?")
            params.append(date_to)

        where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"

        # Get total count
        count_query = f"SELECT COUNT(*) FROM extraction_results WHERE {where_clause}"
        cursor.execute(count_query, params)
        total_count = cursor.fetchone()[0]

        # Get results with pagination
        select_query = f"""
            SELECT id, job_id, chunk_id, categories, confidence, extracted_data, created_at
            FROM extraction_results
            WHERE {where_clause}
            ORDER BY confidence DESC, created_at DESC
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])
        cursor.execute(select_query, params)
        rows = cursor.fetchall()

        # Convert to ExtractionResult objects
        results = []
        for row in rows:
            result = ExtractionResult(
                id=row[0],
                job_id=row[1],
                chunk_id=row[2],
                categories=json.loads(row[3]) if row[3] else [],
                confidence=row[4],
                extracted_data=json.loads(row[5]) if row[5] else {},
                created_at=row[6]
            )
            results.append(result)

        return results, total_count

    def search_by_categories(
        self,
        categories: List[str],
        match_all: bool = False,
        limit: int = 50,
        offset: int = 0
    ) -> Tuple[List[ExtractionResult], int]:
        """
        Search results by categories.

        Args:
            categories: List of categories to search for
            match_all: If True, result must have all categories; if False, any category
            limit: Maximum results to return
            offset: Pagination offset

        Returns:
            Tuple of (results, total_count)
        """
        cursor = self.db_connection.cursor()

        if match_all:
            # Result must contain all specified categories
            where_conditions = []
            params = []
            for category in categories:
                where_conditions.append("categories::text ILIKE ?")
                params.append(f"%{category}%")
            where_clause = " AND ".join(where_conditions)
        else:
            # Result can contain any of the specified categories
            where_conditions = []
            params = []
            for category in categories:
                where_conditions.append("categories::text ILIKE ?")
                params.append(f"%{category}%")
            where_clause = f"({' OR '.join(where_conditions)})"

        # Get total count
        count_query = f"SELECT COUNT(*) FROM extraction_results WHERE {where_clause}"
        cursor.execute(count_query, params)
        total_count = cursor.fetchone()[0]

        # Get results
        select_query = f"""
            SELECT id, job_id, chunk_id, categories, confidence, extracted_data, created_at
            FROM extraction_results
            WHERE {where_clause}
            ORDER BY confidence DESC, created_at DESC
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])
        cursor.execute(select_query, params)
        rows = cursor.fetchall()

        results = []
        for row in rows:
            result = ExtractionResult(
                id=row[0],
                job_id=row[1],
                chunk_id=row[2],
                categories=json.loads(row[3]) if row[3] else [],
                confidence=row[4],
                extracted_data=json.loads(row[5]) if row[5] else {},
                created_at=row[6]
            )
            results.append(result)

        return results, total_count

    def search_by_chunk(
        self,
        chunk_id: str,
        min_confidence: Optional[float] = None,
        limit: int = 50,
        offset: int = 0
    ) -> Tuple[List[ExtractionResult], int]:
        """Search results within a specific chunk."""
        cursor = self.db_connection.cursor()

        where_conditions = ["chunk_id = ?"]
        params = [chunk_id]

        if min_confidence is not None:
            where_conditions.append("confidence >= ?")
            params.append(min_confidence)

        where_clause = " AND ".join(where_conditions)

        # Get total count
        count_query = f"SELECT COUNT(*) FROM extraction_results WHERE {where_clause}"
        cursor.execute(count_query, params)
        total_count = cursor.fetchone()[0]

        # Get results
        select_query = f"""
            SELECT id, job_id, chunk_id, categories, confidence, extracted_data, created_at
            FROM extraction_results
            WHERE {where_clause}
            ORDER BY confidence DESC, created_at DESC
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])
        cursor.execute(select_query, params)
        rows = cursor.fetchall()

        results = []
        for row in rows:
            result = ExtractionResult(
                id=row[0],
                job_id=row[1],
                chunk_id=row[2],
                categories=json.loads(row[3]) if row[3] else [],
                confidence=row[4],
                extracted_data=json.loads(row[5]) if row[5] else {},
                created_at=row[6]
            )
            results.append(result)

        return results, total_count

    def get_category_stats(self, project_id: Optional[int] = None) -> Dict[str, Any]:
        """Get statistics about categories in extraction results."""
        cursor = self.db_connection.cursor()

        query = """
            SELECT
                jsonb_array_elements_text(COALESCE(categories, '[]'::jsonb)) as category,
                COUNT(*) as count,
                AVG(confidence) as avg_confidence
            FROM extraction_results er
        """

        params = []
        if project_id:
            query += " JOIN extraction_jobs ej ON er.job_id = ej.id WHERE ej.project_id = ?"
            params.append(project_id)

        query += " GROUP BY category ORDER BY count DESC"

        cursor.execute(query, params)
        rows = cursor.fetchall()

        stats = {
            'categories': {},
            'total_results': 0,
            'total_categories': len(rows)
        }

        for row in rows:
            category = row[0]
            count = row[1]
            avg_confidence = row[2]

            stats['categories'][category] = {
                'count': count,
                'avg_confidence': avg_confidence
            }
            stats['total_results'] += count

        return stats

    def get_chunk_stats(self, project_id: Optional[int] = None) -> Dict[str, Any]:
        """Get statistics about chunks."""
        cursor = self.db_connection.cursor()

        query = """
            SELECT
                chunk_id,
                COUNT(*) as result_count,
                AVG(confidence) as avg_confidence,
                MIN(created_at) as oldest_result,
                MAX(created_at) as newest_result
            FROM extraction_results er
        """

        params = []
        if project_id:
            query += " JOIN extraction_jobs ej ON er.job_id = ej.id WHERE ej.project_id = ?"
            params.append(project_id)

        query += " GROUP BY chunk_id ORDER BY oldest_result DESC"

        cursor.execute(query, params)
        rows = cursor.fetchall()

        stats = {
            'chunks': {},
            'total_chunks': len(rows),
            'total_results': sum(row[1] for row in rows)
        }

        for row in rows:
            chunk_id = row[0]
            stats['chunks'][chunk_id] = {
                'result_count': row[1],
                'avg_confidence': row[2],
                'oldest_result': row[3],
                'newest_result': row[4]
            }

        return stats