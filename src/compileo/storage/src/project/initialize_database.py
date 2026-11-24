"""
Database initialization utilities with multi-tenant support.

This module handles initialization of:
- System database: Global users, sessions, quotas, licenses
- Tenant databases: Per-user projects, documents, jobs, settings
"""

import sqlite3
from typing import Optional


def setup_database():
    """
    Initialize the system database with multi-user support.
    This creates the global system tables (users, sessions, quotas, licenses).
    """
    # Import here to avoid circular imports
    from ..database import get_system_db_connection
    
    conn = None
    try:
        with get_system_db_connection() as conn:
            cursor = conn.cursor()
            _setup_system_schema(cursor)
            conn.commit()
        print("System database initialized successfully")
    except Exception as e:
        print(f"Error initializing system database: {e}")
        if conn:
            conn.rollback()
        raise


def _setup_system_schema(cursor: sqlite3.Cursor) -> None:
    """Set up the system-level schema (users, sessions, quotas, licenses)."""
    
    # 1. Create Users Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            username TEXT UNIQUE,
            hashed_password TEXT NOT NULL,
            is_active BOOLEAN DEFAULT 1,
            is_admin BOOLEAN DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # 2. Create User Sessions Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_sessions (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            token TEXT UNIQUE NOT NULL,
            expires_at TIMESTAMP NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')

    # 3. Create User Quotas Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_quotas (
            user_id TEXT PRIMARY KEY,
            max_projects INTEGER DEFAULT 10,
            max_documents_per_project INTEGER DEFAULT 100,
            max_storage_mb INTEGER DEFAULT 1024,
            max_concurrent_jobs INTEGER DEFAULT 3,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')

    # 4. Create Licenses Table (encrypted sensitive data)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS licenses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            owner_id TEXT UNIQUE,
            license_key_encrypted TEXT,
            activation_token_encrypted TEXT,
            status TEXT DEFAULT 'inactive',
            expiry TIMESTAMP,
            last_check TIMESTAMP,
            instance_id TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (owner_id) REFERENCES users (id)
        )
    ''')

    # 5. Create Plugins Table (for ownership tracking across multi-tenant setup)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS plugins (
            id TEXT PRIMARY KEY,
            owner_id TEXT NOT NULL,
            name TEXT NOT NULL,
            version TEXT NOT NULL,
            author TEXT,
            description TEXT,
            entry_point TEXT,
            installed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (owner_id) REFERENCES users (id)
        )
    ''')


def setup_tenant_schema(conn: sqlite3.Connection, owner_id: str) -> None:
    """
    Set up the tenant-specific schema for a user.
    
    Args:
        conn: SQLite connection to the tenant database
        owner_id: The user ID this database belongs to
    """
    cursor = conn.cursor()
    
    # Projects
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS projects (
            id TEXT PRIMARY KEY,
            owner_id TEXT NOT NULL,
            name TEXT NOT NULL,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Documents
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            owner_id TEXT NOT NULL,
            project_id TEXT NOT NULL,
            file_name TEXT NOT NULL,
            source_file_path TEXT NOT NULL,
            file_size INTEGER,
            mime_type TEXT,
            status TEXT DEFAULT 'uploaded',
            page_selection TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (project_id) REFERENCES projects (id)
        )
    ''')

    # Chunks
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY,
            owner_id TEXT NOT NULL,
            document_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            content_preview TEXT,
            metadata TEXT,
            file_path TEXT,
            chunk_strategy TEXT,
            status TEXT DEFAULT 'active',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (document_id) REFERENCES documents (id)
        )
    ''')

    # Dataset Jobs
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS dataset_jobs (
            id TEXT PRIMARY KEY,
            owner_id TEXT NOT NULL,
            project_id TEXT NOT NULL,
            document_id TEXT,
            job_type TEXT,
            status TEXT NOT NULL,
            parameters TEXT,
            progress REAL DEFAULT 0.0,
            result TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            started_at TIMESTAMP,
            completed_at TIMESTAMP,
            error_message TEXT,
            FOREIGN KEY (project_id) REFERENCES projects (id),
            FOREIGN KEY (document_id) REFERENCES documents (id)
        )
    ''')

    # Extraction Jobs
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS extraction_jobs (
            id TEXT PRIMARY KEY,
            owner_id TEXT NOT NULL,
            project_id TEXT NOT NULL,
            document_id TEXT,
            status TEXT NOT NULL,
            parameters TEXT,
            progress REAL DEFAULT 0.0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            started_at TIMESTAMP,
            completed_at TIMESTAMP,
            error_message TEXT,
            FOREIGN KEY (project_id) REFERENCES projects (id),
            FOREIGN KEY (document_id) REFERENCES documents (id)
        )
    ''')

    # Extraction Results
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS extraction_results (
            id TEXT PRIMARY KEY,
            owner_id TEXT NOT NULL,
            job_id TEXT NOT NULL,
            project_id TEXT NOT NULL,
            chunk_id TEXT NOT NULL,
            categories TEXT,
            confidence REAL,
            extracted_data TEXT,
            result_type TEXT,
            file_path TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (job_id) REFERENCES extraction_jobs (id),
            FOREIGN KEY (project_id) REFERENCES projects (id),
            FOREIGN KEY (chunk_id) REFERENCES chunks (id)
        )
    ''')

    # Dataset Parameters
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS dataset_parameters (
            id TEXT PRIMARY KEY,
            owner_id TEXT NOT NULL,
            project_id TEXT NOT NULL,
            purpose TEXT,
            audience TEXT,
            extraction_rules TEXT,
            dataset_format TEXT,
            question_style TEXT,
            answer_style TEXT,
            negativity_ratio REAL,
            data_augmentation BOOLEAN,
            custom_audience TEXT,
            custom_purpose TEXT,
            complexity_level TEXT,
            domain TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (project_id) REFERENCES projects (id)
        )
    ''')

    # Prompts
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS prompts (
            id TEXT PRIMARY KEY,
            owner_id TEXT NOT NULL,
            project_id TEXT NOT NULL,
            name TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (project_id) REFERENCES projects (id)
        )
    ''')

    # Insert default prompts
    _insert_default_prompts(cursor, owner_id)

    # Dataset Versions
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS dataset_versions (
            id TEXT PRIMARY KEY,
            owner_id TEXT NOT NULL,
            project_id TEXT NOT NULL,
            version TEXT NOT NULL,
            major_version INTEGER,
            minor_version INTEGER,
            patch_version INTEGER,
            dataset_name TEXT,
            description TEXT,
            total_entries INTEGER,
            file_path TEXT,
            file_hash TEXT,
            metadata TEXT,
            is_active BOOLEAN DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            created_by TEXT,
            FOREIGN KEY (project_id) REFERENCES projects (id)
        )
    ''')

    # Dataset Changes
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS dataset_changes (
            id TEXT PRIMARY KEY,
            owner_id TEXT NOT NULL,
            dataset_version_id TEXT NOT NULL,
            change_type TEXT NOT NULL,
            change_description TEXT,
            entries_affected INTEGER,
            changed_by TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (dataset_version_id) REFERENCES dataset_versions (id)
        )
    ''')

    # Dataset Lineage
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS dataset_lineage (
            id TEXT PRIMARY KEY,
            owner_id TEXT NOT NULL,
            dataset_version_id TEXT NOT NULL,
            source_type TEXT NOT NULL,
            source_id TEXT,
            source_name TEXT,
            source_hash TEXT,
            processing_parameters TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (dataset_version_id) REFERENCES dataset_versions (id)
        )
    ''')

    # Processed Outputs
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS processed_outputs (
            id TEXT PRIMARY KEY,
            owner_id TEXT NOT NULL,
            project_id TEXT NOT NULL,
            output_type TEXT NOT NULL,
            content TEXT,
            metadata TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (project_id) REFERENCES projects (id)
        )
    ''')

    # Benchmark Jobs
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS benchmark_jobs (
            id TEXT PRIMARY KEY,
            owner_id TEXT NOT NULL,
            model_name TEXT NOT NULL,
            status TEXT NOT NULL,
            parameters TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            started_at TIMESTAMP,
            completed_at TIMESTAMP,
            error_message TEXT,
            progress REAL DEFAULT 0.0
        )
    ''')

    # Benchmark Results
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS benchmark_results (
            id TEXT PRIMARY KEY,
            owner_id TEXT NOT NULL,
            job_id TEXT NOT NULL,
            metrics TEXT,
            metadata TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (job_id) REFERENCES benchmark_jobs (id)
        )
    ''')

    # Quality Jobs
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS quality_jobs (
            id TEXT PRIMARY KEY,
            owner_id TEXT NOT NULL,
            project_id TEXT NOT NULL,
            dataset_file TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            parameters TEXT DEFAULT '{}',
            results TEXT,
            report_files TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP,
            error_message TEXT,
            started_at TIMESTAMP,
            progress REAL DEFAULT 0.0,
            FOREIGN KEY (project_id) REFERENCES projects (id) ON DELETE CASCADE
        )
    ''')

    # Taxonomies
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS taxonomies (
            id TEXT PRIMARY KEY,
            owner_id TEXT NOT NULL,
            project_id TEXT NOT NULL,
            name TEXT NOT NULL,
            structure TEXT,
            file_path TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (project_id) REFERENCES projects (id)
        )
    ''')

    # Embed (embedding plugin chunk-focused taxonomy cache)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS embed (
            id TEXT PRIMARY KEY,
            owner_id TEXT NOT NULL,
            project_id TEXT NOT NULL,
            document_id TEXT NOT NULL,
            chunk_id TEXT NOT NULL,
            chunk_index INTEGER,
            chunk_hash TEXT NOT NULL,
            taxonomy_tree TEXT NOT NULL,
            leaf_ids TEXT,
            taxonomy_provider TEXT,
            taxonomy_model TEXT,
            generation_mode TEXT DEFAULT 'chunk',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(owner_id, chunk_id),
            FOREIGN KEY (project_id) REFERENCES projects (id),
            FOREIGN KEY (document_id) REFERENCES documents (id),
            FOREIGN KEY (chunk_id) REFERENCES chunks (id)
        )
    ''')

    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_embed_owner_project_chunk
        ON embed(owner_id, project_id, chunk_index)
    ''')

    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_embed_owner_document_chunk
        ON embed(owner_id, document_id, chunk_index)
    ''')

    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_embed_owner_chunk_hash
        ON embed(owner_id, chunk_id, chunk_hash)
    ''')

    # Parsed Documents
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS parsed_documents (
            id TEXT PRIMARY KEY,
            owner_id TEXT NOT NULL,
            project_id TEXT NOT NULL,
            document_id TEXT NOT NULL,
            filename TEXT NOT NULL,
            content TEXT,
            metadata TEXT,
            parser_used TEXT,
            parse_config TEXT,
            total_pages INTEGER,
            parsing_time REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (project_id) REFERENCES projects (id),
            FOREIGN KEY (document_id) REFERENCES documents (id)
        )
    ''')

    # Parsed Files
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS parsed_files (
            id TEXT PRIMARY KEY,
            owner_id TEXT NOT NULL,
            parsed_document_id TEXT NOT NULL,
            filename TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_type TEXT,
            page_number INTEGER,
            content_length INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (parsed_document_id) REFERENCES parsed_documents (id)
        )
    ''')

    # GUI Settings (tenant-specific, with encrypted values for sensitive data)
    # Note: Using empty string '' instead of NULL for system settings to ensure
    # UNIQUE constraint works properly (SQLite treats NULL != NULL in unique constraints)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS gui_settings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key TEXT NOT NULL,
            value TEXT,
            is_encrypted BOOLEAN DEFAULT 0,
            owner_id TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(key, owner_id)
        )
    ''')

    # Plugins
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS plugins (
            id TEXT PRIMARY KEY,
            owner_id TEXT NOT NULL,
            name TEXT NOT NULL,
            version TEXT NOT NULL,
            author TEXT,
            description TEXT,
            entry_point TEXT,
            installed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Job History Table - for retaining metadata of completed/failed jobs
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS job_history (
            id TEXT PRIMARY KEY,
            owner_id TEXT NOT NULL,
            project_id TEXT,
            document_id TEXT,
            job_type TEXT NOT NULL,
            status TEXT NOT NULL,
            parameters TEXT,
            progress REAL,
            result_summary TEXT,
            error_summary TEXT,
            started_at TIMESTAMP,
            completed_at TIMESTAMP,
            duration_seconds INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Index for efficient job history queries
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_job_history_owner_project
        ON job_history(owner_id, project_id, created_at DESC)
    ''')
    
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_job_history_job_type
        ON job_history(job_type, created_at DESC)
    ''')

    conn.commit()


def _insert_default_prompts(cursor: sqlite3.Cursor, owner_id: str) -> None:
    """Insert default prompts for a tenant."""
    import uuid
    
    default_prompts = [
        {
            "name": "qa_pairs",
            "content": """Generate at most {datasets_per_chunk} distinct question-answer pairs based strictly on the following content.
Focus on capturing key information, facts, and concepts. Do not generate more questions than the content supports.

Content: {chunk}

CRITICAL CONSTRAINTS:
- ONLY use information explicitly present in the provided content.
- Each question must cover a DIFFERENT topic or aspect.
- DO NOT generate duplicate questions.
- Generate at most {datasets_per_chunk} pairs.
- For the reasoning field: provide a DETAILED explanation of the causal/logical reasoning behind the answer based on the text. DO NOT just cite "the text states...". Instead, elucidate the actual underlying mechanism, process, or logic described.
- If the text does not contain sufficient information to deduce detailed reasoning, leave the reasoning field empty (null or empty string). DO NOT hallucinate a reason.

IMPORTANT: Provide your response as a valid JSON array of objects:
[
  {
    "question": "The question",
    "answer": "The answer",
    "reasoning": "Detailed explanation of causal/logical reasoning (or empty/null if unsupported by text)",
    "confidence_score": 0.9,
    "reasoning_steps": ["Specific logical step 1", "Specific logical step 2"]
  }
]"""
        },
        {
            "name": "question_generation",
            "content": """Generate at most {datasets_per_chunk} high-quality questions based ONLY on the following text.
Do not generate more questions than the content supports.
Each question must be a single JSON object with a 'question' field.

Content: {chunk}

IMPORTANT: Provide your response as a valid JSON array."""
        },
        {
            "name": "answer_generation",
            "content": """Based on the following text, provide comprehensive answers to the most important questions.
Generate at most {datasets_per_chunk} items.
The output must be a JSON array of objects with an 'answer' field.

Content: {chunk}"""
        },
        {
            "name": "summarization",
            "content": """Summarize the following text accurately and concisely.
Generate at most {datasets_per_chunk} summaries (usually 1).
The output must be a single JSON object with a 'summary' field.

Content: {chunk}"""
        },
        {
            "name": "instruction_following",
            "content": """Create at most {datasets_per_chunk} natural instruction-response pairs based on the provided content.
Do not generate more pairs than the content supports.
The output must be a JSON array with 'instruction', 'input', and 'output' fields.

Content: {chunk}"""
        },
        {
            "name": "classification",
            "content": """Classify the following text based on the provided schema.

Content: {chunk}
Schema: {classification_schema}"""
        },
        {
            "name": "advanced_extraction",
            "content": """Extract structured data from the following text.

Content: {chunk}"""
        }
    ]

    for prompt in default_prompts:
        cursor.execute("SELECT id FROM prompts WHERE name = ? AND owner_id = ?", (prompt["name"], owner_id))
        if not cursor.fetchone():
            prompt_id = str(uuid.uuid4())
            cursor.execute(
                "INSERT INTO prompts (id, owner_id, project_id, name, content) VALUES (?, ?, ?, ?, ?)",
                (prompt_id, owner_id, 'system', prompt["name"], prompt["content"])
            )
