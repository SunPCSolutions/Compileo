from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field


class ExtractionJobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ExtractionJob(BaseModel):
    """Model for extraction job metadata."""
    id: Optional[Union[str, int]] = None
    project_id: Union[str, int]
    document_id: Optional[Union[str, int]] = None
    status: ExtractionJobStatus = ExtractionJobStatus.PENDING
    parameters: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None

    class Config:
        from_attributes = True


class ExtractionResult(BaseModel):
    """Model for individual extraction result."""
    id: Optional[Union[str, int]] = None
    job_id: Union[str, int]
    project_id: Optional[Union[str, int]] = None
    chunk_id: str
    categories: List[str] = Field(default_factory=list)
    confidence: float = 0.0
    extracted_data: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        from_attributes = True


class ExtractionResultChunk(BaseModel):
    """Model for chunked extraction results."""
    chunk_id: str
    results: List[ExtractionResult]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ExtractionResultMetadata(BaseModel):
    """Metadata for extraction results."""
    total_results: int
    categories: List[str]
    size_bytes: int
    created_at: datetime
    expires_at: Optional[datetime] = None


# Serialization/Deserialization utilities
def serialize_extraction_result(result: ExtractionResult) -> Dict[str, Any]:
    """Serialize an ExtractionResult to a dictionary."""
    return result.model_dump()


def deserialize_extraction_result(data: Dict[str, Any]) -> ExtractionResult:
    """Deserialize a dictionary to an ExtractionResult."""
    return ExtractionResult(**data)


def serialize_extraction_job(job: ExtractionJob) -> Dict[str, Any]:
    """Serialize an ExtractionJob to a dictionary."""
    return job.model_dump()


def deserialize_extraction_job(data: Dict[str, Any]) -> ExtractionJob:
    """Deserialize a dictionary to an ExtractionJob."""
    return ExtractionJob(**data)


def serialize_result_chunk(chunk: ExtractionResultChunk) -> Dict[str, Any]:
    """Serialize an ExtractionResultChunk to a dictionary."""
    return {
        'chunk_id': chunk.chunk_id,
        'results': [serialize_extraction_result(r) for r in chunk.results],
        'metadata': chunk.metadata
    }


def deserialize_result_chunk(data: Dict[str, Any]) -> ExtractionResultChunk:
    """Deserialize a dictionary to an ExtractionResultChunk."""
    return ExtractionResultChunk(
        chunk_id=data['chunk_id'],
        results=[deserialize_extraction_result(r) for r in data['results']],
        metadata=data.get('metadata', {})
    )


class ExtractionSummary(BaseModel):
    """Summary of extraction operation results."""
    total_chunks: int
    processed_chunks: int
    filtered_chunks: int
    categories_used: List[str]
    confidence_threshold: float
    extraction_time: float
    results_file: Optional[str] = None

    class Config:
        from_attributes = True