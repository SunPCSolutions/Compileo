"""Quality routes for the Compileo API."""

import uuid
import json
import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel
from datetime import datetime

from ...storage.src.database import get_db_connection
from ...features.datasetqual import QualityAnalyzer

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Pydantic models
class QualityAnalysisRequest(BaseModel):
    dataset_file: str
    config: Optional[dict] = None
    threshold: float = 0.7
    output_format: str = "json"
    quality_model: Optional[str] = None

class QualityResult(BaseModel):
    job_id: str
    status: str
    summary: dict
    metrics: dict
    details: dict
    completed_at: Optional[datetime] = None

class QualityHistoryResponse(BaseModel):
    analyses: list
    total: int

# In-memory storage for quality jobs (replace with proper storage in production)
quality_jobs = {}

# Repository class for quality jobs
class QualityJobRepository:
    def __init__(self, db_connection):
        self.db = db_connection

    def create_job(self, job_id: str, project_id: str, dataset_file: str, parameters: dict = None):
        """Create a new quality job."""
        cursor = self.db.cursor()
        cursor.execute("""
            INSERT INTO quality_jobs (id, project_id, dataset_file, status, parameters)
            VALUES (?, ?, ?, 'pending', ?)
        """, (job_id, project_id, dataset_file, json.dumps(parameters or {})))
        self.db.commit()

    def update_job_status(self, job_id: str, status: str, results: dict = None, report_files: dict = None, error_message: str = None):
        """Update job status and results."""
        cursor = self.db.cursor()
        update_fields = ["status = ?", "updated_at = CURRENT_TIMESTAMP"]
        params = [status]

        if results:
            update_fields.append("results = ?")
            params.append(json.dumps(results))

        if report_files:
            update_fields.append("report_files = ?")
            params.append(json.dumps(report_files))

        if error_message:
            update_fields.append("error_message = ?")
            params.append(error_message)

        if status == 'completed':
            update_fields.append("completed_at = CURRENT_TIMESTAMP")
        elif status == 'running':
            update_fields.append("started_at = CURRENT_TIMESTAMP")

        query = f"UPDATE quality_jobs SET {', '.join(update_fields)} WHERE id = ?"
        params.append(job_id)

        cursor.execute(query, params)
        self.db.commit()

    def get_job(self, job_id: str):
        """Get job by ID."""
        cursor = self.db.cursor()
        cursor.execute("""
            SELECT id, project_id, dataset_file, status, parameters, results, report_files,
                   created_at, updated_at, completed_at, error_message, started_at, progress
            FROM quality_jobs WHERE id = ?
        """, (job_id,))

        row = cursor.fetchone()
        if not row:
            return None

        return {
            'job_id': row[0],
            'project_id': row[1],
            'dataset_file': row[2],
            'status': row[3],
            'parameters': json.loads(row[4]) if row[4] else {},
            'results': json.loads(row[5]) if row[5] else {},
            'report_files': json.loads(row[6]) if row[6] else {},
            'created_at': row[7],
            'updated_at': row[8],
            'completed_at': row[9],
            'error_message': row[10],
            'started_at': row[11],
            'progress': row[12] or 0.0
        }

    def get_jobs_history(self, limit: int = 20):
        """Get jobs history."""
        cursor = self.db.cursor()
        cursor.execute("""
            SELECT id, status, results, completed_at
            FROM quality_jobs
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,))

        jobs = []
        for row in cursor.fetchall():
            jobs.append({
                'job_id': row[0],
                'status': row[1],
                'summary': json.loads(row[2]).get('summary', {}) if row[2] else {},
                'completed_at': row[3]
            })
        return jobs

# Dependency to get database connection
def get_db():
    return get_db_connection()

@router.post("/analyze")
async def analyze_quality(
    request: QualityAnalysisRequest,
    background_tasks: BackgroundTasks,
    db=Depends(get_db)
):
    """Analyze quality of a dataset."""
    try:
        import os
        if not os.path.exists(request.dataset_file):
            raise HTTPException(status_code=404, detail=f"Dataset file not found: {request.dataset_file}")

        # Extract project_id from dataset_file path
        project_id = None
        if request.dataset_file.startswith("storage/datasets/"):
            parts = request.dataset_file.split("/")
            if len(parts) >= 3:
                project_id = parts[2]  # project_id is a string UUID, not int

        if not project_id:
            raise HTTPException(status_code=400, detail="Could not determine project ID from dataset file path")

        job_id = str(uuid.uuid4())

        # Create job in database
        repo = QualityJobRepository(db)
        repo.create_job(job_id, project_id, request.dataset_file, {
            "threshold": request.threshold,
            "output_format": request.output_format,
            "config": request.config,
            "quality_model": request.quality_model
        })

        # Start background analysis
        background_tasks.add_task(
            process_quality_analysis,
            job_id,
            request,
            db
        )

        return {
            "job_id": job_id,
            "message": "Quality analysis started",
            "estimated_duration": "2-5 minutes"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start quality analysis: {str(e)}")

@router.get("/{job_id}/results", response_model=QualityResult)
async def get_quality_results(job_id: str, db=Depends(get_db)):
    """Get quality analysis results."""
    repo = QualityJobRepository(db)
    job = repo.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Quality analysis job {job_id} not found")

    return QualityResult(
        job_id=job['job_id'],
        status=job['status'],
        summary=job['results'].get('summary', {}),
        metrics={k: v['score'] for k, v in job['results'].get('results', {}).items()},
        details={
            "results": job['results'].get('results', {}),
            "config": job['results'].get('config', {}),
            "dataset_size": job['results'].get('dataset_size', 0),
            "report_files": job['report_files']
        },
        completed_at=job['completed_at']
    )

@router.get("/history", response_model=QualityHistoryResponse)
async def get_quality_history(
    dataset_id: Optional[str] = None,
    limit: int = 20,
    db=Depends(get_db)
):
    """Get quality analysis history."""
    try:
        repo = QualityJobRepository(db)
        jobs = repo.get_jobs_history(limit)
        return QualityHistoryResponse(
            analyses=jobs,
            total=len(jobs)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get quality history: {str(e)}")

@router.post("/threshold-check")
async def check_quality_threshold(
    dataset_file: str,
    threshold: float = 0.7,
    db=Depends(get_db)
):
    """Check if dataset meets quality threshold."""
    try:
        import os
        if not os.path.exists(dataset_file):
            raise HTTPException(status_code=404, detail=f"Dataset file not found: {dataset_file}")

        # TODO: Implement actual threshold checking
        # For now, return mock result
        return {
            "dataset_file": dataset_file,
            "threshold": threshold,
            "overall_score": 0.85,
            "passed": True,
            "message": "Dataset meets quality threshold"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to check quality threshold: {str(e)}")

# Background task function
def process_quality_analysis(job_id: str, request: QualityAnalysisRequest, db):
    """Background task for quality analysis."""
    try:
        import time
        import json
        from pathlib import Path

        repo = QualityJobRepository(db)

        # Update status to running
        repo.update_job_status(job_id, "running")

        # Load dataset
        try:
            with open(request.dataset_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()

            # First try to load as standard JSON
            try:
                dataset = json.loads(content)
                if not isinstance(dataset, list):
                    dataset = [dataset]
                
                # Handle double-serialization (list of strings)
                if dataset and isinstance(dataset[0], str):
                    parsed_dataset = []
                    for item in dataset:
                        try:
                            if isinstance(item, str):
                                # Try to parse the string as JSON
                                try:
                                    parsed_item = json.loads(item)
                                    parsed_dataset.append(parsed_item)
                                except json.JSONDecodeError:
                                    # If it fails, it might be multiple JSON objects in one string (newline separated)
                                    # This happens in some extract formats
                                    sub_items = []
                                    for line in item.split('\n'):
                                        if line.strip():
                                            try:
                                                sub_items.append(json.loads(line))
                                            except json.JSONDecodeError:
                                                pass
                                    if sub_items:
                                        parsed_dataset.extend(sub_items)
                                    else:
                                        # If still fails, maybe it's just a string?
                                        pass
                            else:
                                parsed_dataset.append(item)
                        except Exception as e:
                            logger.warning(f"Failed to parse dataset item: {item[:50]}... Error: {e}")
                    dataset = parsed_dataset

            except json.JSONDecodeError:
                # Fallback to line-based parsing for malformed JSON
                dataset = []
                lines = content.split('\n')
                for line in lines:
                    line = line.strip()
                    if line:
                        try:
                            item = json.loads(line.rstrip(','))
                            dataset.append(item)
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse line: {line[:50]}...")
        except Exception as e:
            repo.update_job_status(job_id, "failed", error_message=f"Failed to load dataset: {str(e)}")
            return
        except Exception as e:
            repo.update_job_status(job_id, "failed", error_message=f"Failed to load dataset: {str(e)}")
            return

        # Initialize quality analyzer with config if provided
        config = None
        if request.config:
            try:
                from ...features.datasetqual.config import QualityConfig
                config = QualityConfig(**request.config)
            except Exception as e:
                logger.warning(f"Invalid config provided: {e}, using default config")
                config = None

        analyzer = QualityAnalyzer(config)

        # Get quality model from job parameters
        job_data = repo.get_job(job_id)
        quality_model = job_data['parameters'].get('quality_model') if job_data and job_data['parameters'] else None

        # Run quality analysis
        try:
            results = analyzer.analyze_dataset(dataset)

            # Generate report files based on output_format
            report_files = {}
            output_format = request.output_format or "json"

            # Extract project_id from dataset_file path (storage/datasets/{project_id}/{filename})
            project_id = None
            if request.dataset_file.startswith("storage/datasets/"):
                parts = request.dataset_file.split("/")
                if len(parts) >= 3:
                    try:
                        project_id = int(parts[2])
                    except ValueError:
                        pass

            # Create quality reports directory following storage convention
            if project_id is not None:
                reports_dir = Path("storage/dataqual") / str(project_id)
            else:
                reports_dir = Path("storage/dataqual")
            reports_dir.mkdir(parents=True, exist_ok=True)

            base_filename = f"{job_id}_datasetqual"

            if output_format == "json":
                json_path = reports_dir / f"{base_filename}.json"
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, default=str)
                report_files["json"] = str(json_path)

            elif output_format == "html":
                html_path = reports_dir / f"{base_filename}.html"
                # Generate simple HTML report
                html_content = f"""
                <html>
                <head><title>Quality Analysis Report</title></head>
                <body>
                <h1>Dataset Quality Analysis Report</h1>
                <h2>Summary</h2>
                <p>Overall Score: {results.get('summary', {}).get('overall_score', 'N/A')}</p>
                <p>Dataset Size: {results.get('dataset_size', 0)}</p>
                <h2>Detailed Results</h2>
                <pre>{json.dumps(results, indent=2, default=str)}</pre>
                </body>
                </html>
                """
                with open(html_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                report_files["html"] = str(html_path)

            elif output_format == "pdf":
                # For PDF, we'd need a library like reportlab or fpdf
                # For now, create a text file as placeholder
                pdf_path = reports_dir / f"{base_filename}.txt"
                with open(pdf_path, 'w', encoding='utf-8') as f:
                    f.write(f"Quality Analysis Report\n\n{json.dumps(results, indent=2, default=str)}")
                report_files["pdf"] = str(pdf_path)  # Note: This is actually a .txt file

            # Add quality model to results
            if quality_model:
                results['quality_model'] = quality_model

            # Update job with results
            repo.update_job_status(job_id, "completed", results=results, report_files=report_files)

        except Exception as e:
            repo.update_job_status(job_id, "failed", error_message=f"Quality analysis failed: {str(e)}")

    except Exception as e:
        repo = QualityJobRepository(db)
        repo.update_job_status(job_id, "failed", error_message=str(e))
