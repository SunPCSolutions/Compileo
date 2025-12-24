from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime


class SelectiveExtractionRequest(BaseModel):
    """API request model for selective extraction."""
    taxonomy_id: str
    selected_categories: List[str] = Field(default_factory=list)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    initial_classifier: str = Field(description="Classifier for the initial classification stage")
    enable_validation_stage: bool = Field(default=False, description="Enable the validation stage")
    validation_classifier: Optional[str] = Field(default=None, description="Classifier for the validation stage")
    extraction_type: str = Field(default="ner", description="Type of extraction: 'ner' for named entities, 'whole_text' for complete text portions")
    extraction_mode: str = Field(default="contextual", description="Extraction mode: 'contextual' (default) or 'document_wide'")


class ExtractionJobResponse(BaseModel):
    """API response model for extraction job creation."""
    job_id: str
    status: str = "pending"
    message: str = "Extraction job created successfully"


class ExtractionJobStatus(BaseModel):
    """API response model for extraction job status."""
    job_id: str
    taxonomy_id: Optional[str] = None
    status: str
    progress_percentage: float = 0.0
    progress: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None


class ExtractionResultItem(BaseModel):
    """Individual extraction result item."""
    chunk_id: str
    chunk_text: Optional[str] = None
    classifications: Dict[str, Any]
    confidence_score: float
    categories_matched: List[str]
    metadata: Dict[str, Any]


class ExtractionResultsResponse(BaseModel):
    """API response model for extraction results."""
    job_id: str
    results: List[ExtractionResultItem]
    total_results: int
    page: int = 1
    page_size: int = 50
    has_more: bool = False
    filters: Optional[Dict[str, Any]] = None


class ExtractionJobCancellation(BaseModel):
    """API response model for job cancellation."""
    job_id: str
    status: str
    message: str


class ExtractionJobRestart(BaseModel):
    """API response model for job restart."""
    job_id: str
    status: str
    message: str