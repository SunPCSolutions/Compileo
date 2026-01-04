"""
Database initialization utilities.
"""

import uuid
from datetime import datetime
from ..database import get_db_connection
from ....core.logging import get_logger

logger = get_logger(__name__)


def setup_database():
    """
    Initialize the database with all required tables.
    """
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Create projects table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Create documents table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                file_name TEXT NOT NULL,
                source_file_path TEXT NOT NULL,
                file_size INTEGER,
                mime_type TEXT,
                status TEXT DEFAULT 'uploaded',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES projects (id)
            )
        ''')

        # Create chunks table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
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

        # Add missing columns to existing chunks table if they don't exist
        try:
            cursor.execute("ALTER TABLE chunks ADD COLUMN chunk_strategy TEXT")
        except:
            pass  # Column might already exist

        try:
            cursor.execute("ALTER TABLE chunks ADD COLUMN status TEXT DEFAULT 'active'")
        except:
            pass  # Column might already exist

        # Create dataset_jobs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS dataset_jobs (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                job_type TEXT NOT NULL,
                status TEXT NOT NULL,
                parameters TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES projects (id)
            )
        ''')

        # Create extraction_jobs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS extraction_jobs (
                id TEXT PRIMARY KEY,
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

        # Add missing columns to existing extraction_jobs table if they don't exist
        try:
            cursor.execute("ALTER TABLE extraction_jobs ADD COLUMN progress REAL DEFAULT 0.0")
        except:
            pass  # Column might already exist

        try:
            cursor.execute("ALTER TABLE extraction_jobs ADD COLUMN started_at TIMESTAMP")
        except:
            pass  # Column might already exist

        try:
            cursor.execute("ALTER TABLE extraction_jobs ADD COLUMN completed_at TIMESTAMP")
        except:
            pass  # Column might already exist

        try:
            cursor.execute("ALTER TABLE extraction_jobs ADD COLUMN error_message TEXT")
        except:
            pass  # Column might already exist

        # Create extraction_results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS extraction_results (
                id TEXT PRIMARY KEY,
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

        # Add missing columns to existing extraction_results table if they don't exist
        try:
            cursor.execute("ALTER TABLE extraction_results ADD COLUMN job_id TEXT")
        except:
            pass  # Column might already exist

        try:
            cursor.execute("ALTER TABLE extraction_results ADD COLUMN project_id TEXT")
        except:
            pass  # Column might already exist

        try:
            cursor.execute("ALTER TABLE extraction_results ADD COLUMN categories TEXT")
        except:
            pass  # Column might already exist

        try:
            cursor.execute("ALTER TABLE extraction_results ADD COLUMN confidence REAL")
        except:
            pass  # Column might already exist

        try:
            cursor.execute("ALTER TABLE extraction_results ADD COLUMN extracted_data TEXT")
        except:
            pass  # Column might already exist

        try:
            cursor.execute("ALTER TABLE extraction_results ADD COLUMN updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
        except:
            pass  # Column might already exist

        try:
            cursor.execute("ALTER TABLE extraction_results ADD COLUMN file_path TEXT")
        except:
            pass  # Column might already exist

        # Create dataset_parameters table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS dataset_parameters (
                id TEXT PRIMARY KEY,
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

        # Create prompts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS prompts (
                id TEXT PRIMARY KEY,
                project_id TEXT,
                name TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES projects (id)
            )
        ''')

        # Create dataset_versions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS dataset_versions (
                id TEXT PRIMARY KEY,
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

        # Create dataset_changes table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS dataset_changes (
                id TEXT PRIMARY KEY,
                dataset_version_id TEXT NOT NULL,
                change_type TEXT NOT NULL,
                change_description TEXT,
                entries_affected INTEGER,
                changed_by TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (dataset_version_id) REFERENCES dataset_versions (id)
            )
        ''')

        # Create dataset_lineage table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS dataset_lineage (
                id TEXT PRIMARY KEY,
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

        # Create processed_outputs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS processed_outputs (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                output_type TEXT NOT NULL,
                content TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES projects (id)
            )
        ''')

        # Create benchmark_jobs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS benchmark_jobs (
                id TEXT PRIMARY KEY,
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

        # Create benchmark_results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS benchmark_results (
                id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL,
                metrics TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (job_id) REFERENCES benchmark_jobs (id)
            )
        ''')


        # Create quality_jobs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS quality_jobs (
                id TEXT PRIMARY KEY,
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

        # Create taxonomies table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS taxonomies (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                name TEXT NOT NULL,
                structure TEXT,
                file_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES projects (id)
            )
        ''')

        # Add file_path column to existing taxonomies table if it doesn't exist
        try:
            cursor.execute("ALTER TABLE taxonomies ADD COLUMN file_path TEXT")
        except:
            pass  # Column might already exist

        # Create parsed_documents table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS parsed_documents (
                id TEXT PRIMARY KEY,
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

        # Create parsed_files table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS parsed_files (
                id TEXT PRIMARY KEY,
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

        # Create gui_settings table (for settings storage)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS gui_settings (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Populate default prompts if they don't exist
        cursor.execute("SELECT COUNT(*) FROM prompts")
        count = cursor.fetchone()[0]
        if count == 0:
            logger.info("Populating default prompts...")

            default_prompts = [
                {
                    "id": str(uuid.uuid4()),
                    "project_id": None,  # Global prompts
                    "name": "qa_pairs",
                    "content": """Generate {datasets_per_chunk} question-answer pairs from the following content.

Content: {chunk}

Instructions:
- Create diverse, high-quality questions that test understanding of the content
- Provide accurate, comprehensive answers based only on the given content
- Ensure questions are answerable using only the provided information
- Vary question types (factual, analytical, applicative)
- Focus on key concepts, important details, and relationships in the content

{taxonomy_instructions}

{purpose_context}{audience_context}{complexity_level_context}

Format your response as a JSON array of objects with this structure:
[
  {{
    "question": "Your question here",
    "answer": "Your answer here",
    "category": "content_category",
    "difficulty": "easy|medium|hard"
  }}
]""",
                    "created_at": datetime.utcnow().isoformat(),
                    "updated_at": datetime.utcnow().isoformat()
                },
                {
                    "id": str(uuid.uuid4()),
                    "project_id": None,  # Global prompts
                    "name": "question_generation",
                    "content": """Generate {datasets_per_chunk} questions from the following content.

Content: {chunk}

Instructions:
- Create diverse, high-quality questions that test understanding of the content
- Ensure questions are answerable using only the provided information
- Vary question types (factual, analytical, applicative)
- Focus on key concepts, important details, and relationships in the content

{taxonomy_instructions}

{purpose_context}{audience_context}{complexity_level_context}

Format your response as a JSON array of objects with this structure:
[
  {{
    "question": "Your question here",
    "category": "content_category",
    "difficulty": "easy|medium|hard"
  }}
]""",
                    "created_at": datetime.utcnow().isoformat(),
                    "updated_at": datetime.utcnow().isoformat()
                },
                {
                    "id": str(uuid.uuid4()),
                    "project_id": None,  # Global prompts
                    "name": "answer_generation",
                    "content": """Generate {datasets_per_chunk} answers for the following question based on the content.

Question: {question}
Content: {chunk}

Instructions:
- Provide accurate, comprehensive answers based only on the given content
- Ensure answers are supported by the provided information
- Include relevant details and context from the content

{taxonomy_instructions}

{purpose_context}{audience_context}{complexity_level_context}

Format your response as a JSON array of objects with this structure:
[
  {{
    "answer": "Your answer here",
    "category": "content_category",
    "confidence": 0.0-1.0
  }}
]""",
                    "created_at": datetime.utcnow().isoformat(),
                    "updated_at": datetime.utcnow().isoformat()
                },
                {
                    "id": str(uuid.uuid4()),
                    "project_id": None,  # Global prompts
                    "name": "summarization",
                    "content": """Generate {datasets_per_chunk} summaries from the following content.

Content: {chunk_text}

Instructions:
- Create concise, accurate summaries that capture the main points
- Focus on key concepts and important details
- Maintain factual accuracy from the source content

{taxonomy_instructions}

{purpose_context}{audience_context}{complexity_level_context}

Format your response as a JSON array of objects with this structure:
[
  {{
    "summary": "Your summary here",
    "key_points": ["Point 1", "Point 2", "Point 3"],
    "category": "content_category"
  }}
]""",
                    "created_at": datetime.utcnow().isoformat(),
                    "updated_at": datetime.utcnow().isoformat()
                },
                {
                    "id": str(uuid.uuid4()),
                    "project_id": None,  # Global prompts
                    "name": "instruction_following",
                    "content": """Generate {datasets_per_chunk} instruction-response pairs based on the following content.

Content: {chunk}

Instructions:
- Create realistic, actionable instructions that someone might give about this content
- Provide comprehensive, helpful responses that directly address the instructions
- Ensure instructions and responses are based only on the provided content
- Vary instruction types (explain, describe, analyze, summarize, create, etc.)

{taxonomy_instructions}

{purpose_context}{audience_context}{complexity_level_context}

Format your response as a JSON array of objects with this structure:
[
  {{
    "instruction": "Your instruction here",
    "input": "",
    "output": "Your response here"
  }}
]""",
                    "created_at": datetime.utcnow().isoformat(),
                    "updated_at": datetime.utcnow().isoformat()
                }
            ]

            for prompt in default_prompts:
                cursor.execute('''
                    INSERT INTO prompts (id, project_id, name, content, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    prompt["id"],
                    prompt["project_id"],
                    prompt["name"],
                    prompt["content"],
                    prompt["created_at"],
                    prompt["updated_at"]
                ))

            logger.info(f"Inserted {len(default_prompts)} default prompts")
        else:
            logger.info(f"Database already has {count} prompts, skipping population")

        conn.commit()
        logger.info("Database initialized successfully")

    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()