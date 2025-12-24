"""Core configuration for Compileo API."""

from ..storage.src.database import get_db_connection

# Database configuration
def get_db():
    """Get database connection."""
    return get_db_connection()

# API Configuration
API_V1_PREFIX = "/api/v1"

# CORS settings
cors_origins = [
    "http://localhost:3000",  # React dev server
    "http://localhost:8501",  # Streamlit dev server
    "http://localhost:8080",  # Alternative dev port
]

allowed_hosts = ["*"]  # In production, specify actual domains

# Rate limiting (requests per minute)
RATE_LIMIT_REQUESTS = 100
RATE_LIMIT_WINDOW = 60

# File upload settings
MAX_UPLOAD_SIZE = 100 * 1024 * 1024  # 100MB
ALLOWED_EXTENSIONS = {'.pdf', '.docx', '.txt', '.md', '.csv', '.json', '.xml'}

# Job processing settings
JOB_TIMEOUT = 3600  # 1 hour
MAX_CONCURRENT_JOBS = 5

# Pagination defaults
DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100