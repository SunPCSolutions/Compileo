"""
Metadata tracking and management for extraction results.
Provides comprehensive statistics and analytics.
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import json

from .models import ExtractionResult


class ResultMetadataTracker:
    """Tracks and analyzes metadata for extraction results."""

    def __init__(self, db_connection):
        self.db_connection = db_connection

    def get_comprehensive_stats(
        self,
        project_id: Optional[int] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get comprehensive statistics about extraction results."""
        cursor = self.db_connection.cursor()

        # Build base query
        base_query = """
            SELECT
                COUNT(*) as total_results,
                COUNT(DISTINCT job_id) as total_jobs,
                COUNT(DISTINCT chunk_id) as total_chunks,
                AVG(confidence) as avg_confidence,
                MIN(confidence) as min_confidence,
                MAX(confidence) as max_confidence,
                MIN(created_at) as oldest_result,
                MAX(created_at) as newest_result
            FROM extraction_results er
        """

        params = []
        where_conditions = []

        if project_id:
            where_conditions.append("EXISTS (SELECT 1 FROM extraction_jobs ej WHERE ej.id = er.job_id AND ej.project_id = ?)")
            params.append(project_id)

        if date_from:
            where_conditions.append("er.created_at >= ?")
            params.append(date_from)

        if date_to:
            where_conditions.append("er.created_at <= ?")
            params.append(date_to)

        if where_conditions:
            base_query += " WHERE " + " AND ".join(where_conditions)

        cursor.execute(base_query, params)
        row = cursor.fetchone()

        if not row:
            return self._empty_stats()

        stats = {
            'total_results': row[0],
            'total_jobs': row[1],
            'total_chunks': row[2],
            'confidence_stats': {
                'average': row[3],
                'minimum': row[4],
                'maximum': row[5]
            },
            'time_range': {
                'oldest': row[6],
                'newest': row[7]
            }
        }

        # Get category distribution
        stats['category_distribution'] = self._get_category_distribution(project_id, date_from, date_to)

        # Get chunk statistics
        stats['chunk_stats'] = self._get_chunk_statistics(project_id, date_from, date_to)

        # Get temporal distribution
        stats['temporal_distribution'] = self._get_temporal_distribution(project_id, date_from, date_to)

        # Get quality metrics
        stats['quality_metrics'] = self._get_quality_metrics(project_id, date_from, date_to)

        return stats

    def _get_category_distribution(
        self,
        project_id: Optional[int] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get distribution of categories."""
        cursor = self.db_connection.cursor()

        query = """
            SELECT
                category,
                COUNT(*) as count,
                AVG(confidence) as avg_confidence
            FROM (
                SELECT
                    jsonb_array_elements_text(COALESCE(categories, '[]'::jsonb)) as category,
                    confidence
                FROM extraction_results er
                WHERE 1=1
        """

        params = []
        if project_id:
            query += " AND EXISTS (SELECT 1 FROM extraction_jobs ej WHERE ej.id = er.job_id AND ej.project_id = ?)"
            params.append(project_id)

        if date_from:
            query += " AND er.created_at >= ?"
            params.append(date_from)

        if date_to:
            query += " AND er.created_at <= ?"
            params.append(date_to)

        query += """
            ) categories
            GROUP BY category
            ORDER BY count DESC
        """

        cursor.execute(query, params)
        rows = cursor.fetchall()

        distribution = {}
        for row in rows:
            category = row[0]
            distribution[category] = {
                'count': row[1],
                'avg_confidence': row[2]
            }

        return distribution

    def _get_chunk_statistics(
        self,
        project_id: Optional[int] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None
    ) -> Dict[str, Any]:
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
            WHERE 1=1
        """

        params = []
        if project_id:
            query += " AND EXISTS (SELECT 1 FROM extraction_jobs ej WHERE ej.id = er.job_id AND ej.project_id = ?)"
            params.append(project_id)

        if date_from:
            query += " AND er.created_at >= ?"
            params.append(date_from)

        if date_to:
            query += " AND er.created_at <= ?"
            params.append(date_to)

        query += " GROUP BY chunk_id ORDER BY oldest_result DESC"

        cursor.execute(query, params)
        rows = cursor.fetchall()

        chunk_stats = {
            'total_chunks': len(rows),
            'chunks': {}
        }

        for row in rows:
            chunk_id = row[0]
            chunk_stats['chunks'][chunk_id] = {
                'result_count': row[1],
                'avg_confidence': row[2],
                'oldest_result': row[3],
                'newest_result': row[4]
            }

        return chunk_stats

    def _get_temporal_distribution(
        self,
        project_id: Optional[int] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        granularity: str = 'hour'
    ) -> Dict[str, Any]:
        """Get temporal distribution of results."""
        cursor = self.db_connection.cursor()

        if granularity == 'hour':
            time_format = "strftime('%Y-%m-%d %H:00:00', created_at)"
        elif granularity == 'day':
            time_format = "strftime('%Y-%m-%d', created_at)"
        elif granularity == 'month':
            time_format = "strftime('%Y-%m', created_at)"
        else:
            time_format = "strftime('%Y-%m-%d %H:00:00', created_at)"

        query = f"""
            SELECT
                {time_format} as time_bucket,
                COUNT(*) as result_count,
                AVG(confidence) as avg_confidence
            FROM extraction_results er
            WHERE 1=1
        """

        params = []
        if project_id:
            query += " AND EXISTS (SELECT 1 FROM extraction_jobs ej WHERE ej.id = er.job_id AND ej.project_id = ?)"
            params.append(project_id)

        if date_from:
            query += " AND er.created_at >= ?"
            params.append(date_from)

        if date_to:
            query += " AND er.created_at <= ?"
            params.append(date_to)

        query += f" GROUP BY {time_format} ORDER BY time_bucket"

        cursor.execute(query, params)
        rows = cursor.fetchall()

        distribution = {}
        for row in rows:
            time_bucket = row[0]
            distribution[time_bucket] = {
                'result_count': row[1],
                'avg_confidence': row[2]
            }

        return distribution

    def _get_quality_metrics(
        self,
        project_id: Optional[int] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get quality metrics for results."""
        cursor = self.db_connection.cursor()

        # Confidence distribution
        query = """
            SELECT
                CASE
                    WHEN confidence >= 0.9 THEN 'high'
                    WHEN confidence >= 0.7 THEN 'medium'
                    WHEN confidence >= 0.5 THEN 'low'
                    ELSE 'very_low'
                END as confidence_level,
                COUNT(*) as count
            FROM extraction_results er
            WHERE 1=1
        """

        params = []
        if project_id:
            query += " AND EXISTS (SELECT 1 FROM extraction_jobs ej WHERE ej.id = er.job_id AND ej.project_id = ?)"
            params.append(project_id)

        if date_from:
            query += " AND er.created_at >= ?"
            params.append(date_from)

        if date_to:
            query += " AND er.created_at <= ?"
            params.append(date_to)

        query += " GROUP BY confidence_level"

        cursor.execute(query, params)
        rows = cursor.fetchall()

        confidence_distribution = {}
        for row in rows:
            confidence_distribution[row[0]] = row[1]

        # Category consistency (results with multiple categories)
        cursor.execute("""
            SELECT
                AVG(jsonb_array_length(COALESCE(categories, '[]'::jsonb))) as avg_categories_per_result,
                MAX(jsonb_array_length(COALESCE(categories, '[]'::jsonb))) as max_categories_per_result
            FROM extraction_results er
            WHERE 1=1
        """ + (" AND EXISTS (SELECT 1 FROM extraction_jobs ej WHERE ej.id = er.job_id AND ej.project_id = ?)" if project_id else "") +
              (" AND er.created_at >= ?" if date_from else "") +
              (" AND er.created_at <= ?" if date_to else ""),
              [p for p in [project_id, date_from, date_to] if p is not None])

        row = cursor.fetchone()
        category_stats = {
            'avg_categories_per_result': row[0] if row[0] else 0,
            'max_categories_per_result': row[1] if row[1] else 0
        }

        return {
            'confidence_distribution': confidence_distribution,
            'category_stats': category_stats
        }

    def _empty_stats(self) -> Dict[str, Any]:
        """Return empty statistics structure."""
        return {
            'total_results': 0,
            'total_jobs': 0,
            'total_chunks': 0,
            'confidence_stats': {
                'average': 0.0,
                'minimum': 0.0,
                'maximum': 0.0
            },
            'time_range': {
                'oldest': None,
                'newest': None
            },
            'category_distribution': {},
            'chunk_stats': {'total_chunks': 0, 'chunks': {}},
            'temporal_distribution': {},
            'quality_metrics': {
                'confidence_distribution': {},
                'category_stats': {
                    'avg_categories_per_result': 0,
                    'max_categories_per_result': 0
                }
            }
        }

    def get_result_size_estimate(
        self,
        project_id: Optional[int] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Estimate the storage size of results."""
        cursor = self.db_connection.cursor()

        query = """
            SELECT
                COUNT(*) as result_count,
                SUM(LENGTH(extracted_data::text)) as data_size,
                SUM(LENGTH(categories::text)) as category_size,
                SUM(LENGTH(chunk_id)) as chunk_id_size
            FROM extraction_results er
            WHERE 1=1
        """

        params = []
        if project_id:
            query += " AND EXISTS (SELECT 1 FROM extraction_jobs ej WHERE ej.id = er.job_id AND ej.project_id = ?)"
            params.append(project_id)

        if date_from:
            query += " AND er.created_at >= ?"
            params.append(date_from)

        if date_to:
            query += " AND er.created_at <= ?"
            params.append(date_to)

        cursor.execute(query, params)
        row = cursor.fetchone()

        if not row:
            return {'total_size_bytes': 0, 'result_count': 0}

        # Rough estimate: each result has some overhead
        overhead_per_result = 100  # bytes for metadata
        total_size = (row[1] or 0) + (row[2] or 0) + (row[3] or 0) + (row[0] * overhead_per_result)

        return {
            'result_count': row[0],
            'data_size_bytes': row[1] or 0,
            'category_size_bytes': row[2] or 0,
            'chunk_id_size_bytes': row[3] or 0,
            'overhead_bytes': row[0] * overhead_per_result,
            'total_size_bytes': total_size
        }