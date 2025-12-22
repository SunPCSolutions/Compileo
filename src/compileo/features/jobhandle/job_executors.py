from typing import Dict, Any
import os
import json
import uuid
import sys
from datetime import datetime
from ...core.logging import get_logger

logger = get_logger(__name__)

def _execute_parse_documents(parameters: Dict[str, Any], progress_context, db_connection: Any) -> Dict[str, Any]:
    """Execute document parsing with PDF splitting support."""
    from datetime import datetime
    import json

    # DEBUG: [DEBUG_20251003_RQ_FLOW] - Starting _execute_parse_documents
    from datetime import datetime
    import json

    debug_context = {
        "timestamp": datetime.now().isoformat(),
        "level": "INFO",
        "component": "_execute_parse_documents",
        "message": "Starting _execute_parse_documents",
        "context": {
            "parameters_keys": list(parameters.keys()),
            "project_id": parameters.get('project_id'),
            "document_ids": parameters.get('document_ids'),
            "parser": parameters.get('parser')
        }
    }
    logger.debug(f"DEBUG_JOB_EXEC: {json.dumps(debug_context)}")

    from src.compileo.features.ingestion.main import parse_document
    from ...core.settings import backend_settings

    project_id = parameters.get("project_id")
    document_ids = parameters.get("document_ids", [])
    parser = parameters.get("parser", "grok")
    pages_per_split = parameters.get("pages_per_split", 200)
    overlap_pages = parameters.get("overlap_pages", 1)

    # DEBUG: [DEBUG_20251003_RQ_FLOW] - Job parameters extracted
    debug_context = {
        "timestamp": datetime.now().isoformat(),
        "level": "INFO",
        "component": "_execute_parse_documents",
        "message": "Job parameters extracted",
        "context": {
            "project_id": project_id,
            "document_ids": document_ids,
            "document_count": len(document_ids),
            "parser": parser,
            "pages_per_split": pages_per_split,
            "overlap_pages": overlap_pages
        }
    }
    logger.debug(json.dumps(debug_context))

    # Get database connection (create our own since SQLite connections can't be shared across processes)
    from ...storage.src.database import get_db_connection
    db = get_db_connection()
    if not db:
        raise Exception("Database connection not available")

    total_docs = len(document_ids)
    processed_docs = 0
    results = []

    for i, doc_id in enumerate(document_ids):
        try:
            # Update progress at document level
            progress_context.update_progress_percent((i / total_docs) * 100, f"Processing document {doc_id}")

            # Get document info from database
            cursor = db.cursor()
            cursor.execute("""
                SELECT file_name, source_file_path FROM documents
                WHERE id = ? AND project_id = ?
            """, (doc_id, project_id))

            doc_record = cursor.fetchone()
            if not doc_record:
                results.append({
                    "document_id": doc_id,
                    "status": "failed",
                    "error": f"Document not found"
                })
                continue

            file_name, file_path = doc_record

            # DEBUG: Log file_name and file_path
            logger.debug(f"file_name = {repr(file_name)}, file_path = {repr(file_path)}")

            # Check if file exists
            if not file_path or not os.path.exists(file_path):
                error_msg = f"File not found: {file_path}"
                results.append({
                    "document_id": doc_id,
                    "status": "failed",
                    "error": error_msg
                })
                continue

            # Get API key and specific model for parser
            # Propagate from parameters if available, otherwise get from database settings
            api_key = None
            specific_model = parameters.get("model")
            ollama_options = None

            if parser == "grok":
                api_key = backend_settings.get_setting("grok_api_key")
                if not specific_model:
                    specific_model = backend_settings.get_parsing_grok_model()
            elif parser == "gemini":
                api_key = backend_settings.get_setting("gemini_api_key")
                if not specific_model:
                    specific_model = backend_settings.get_parsing_gemini_model()
            elif parser == "openai":
                api_key = backend_settings.get_setting("openai_api_key")
                if not specific_model:
                    specific_model = backend_settings.get_parsing_openai_model()
            elif parser == "ollama":
                if not specific_model:
                    specific_model = backend_settings.get_parsing_ollama_model()
                ollama_options = {
                    "temperature": backend_settings.get_parsing_ollama_temperature(),
                    "repeat_penalty": backend_settings.get_parsing_ollama_repeat_penalty(),
                    "top_p": backend_settings.get_parsing_ollama_top_p(),
                    "top_k": backend_settings.get_parsing_ollama_top_k(),
                    "num_predict": backend_settings.get_parsing_ollama_num_predict(),
                    "seed": backend_settings.get_parsing_ollama_seed()
                }
            elif parser == "huggingface":
                api_key = backend_settings.get_setting("huggingface_api_key")

            # Use the new ingestion pipeline that handles multi-file parsing
            from src.compileo.features.ingestion.main import parse_document

            try:
                # Prepare options for Ollama if applicable
                parse_options = ollama_options if parser == "ollama" else None

                # Call the ingestion pipeline without progress_context - parsers focus only on parsing
                logger.debug(f"DEBUG_JOB_EXECUTOR_CALL: About to call parse_document for doc {doc_id}, file {file_path}")
                parsed_content = parse_document(
                    file_path=file_path,
                    parser=parser,
                    api_key=api_key,
                    model=specific_model,
                    document_id=doc_id,
                    pages_per_split=pages_per_split,
                    overlap_pages=overlap_pages,
                    options=parse_options
                )
                logger.debug(f"DEBUG_JOB_EXECUTOR_RESULT: parse_document returned {type(parsed_content)} for doc {doc_id}")

                # parse_document() now handles all file creation and database updates
                # Just update the status to indicate parsing is complete
                cursor.execute("""
                    UPDATE documents
                    SET status = 'parsed'
                    WHERE id = ?
                """, (doc_id,))

                processed_docs += 1
                results.append({
                    "document_id": doc_id,
                    "status": "completed",
                    "parser": parser,
                    "pages_per_split": pages_per_split,
                    "overlap_pages": overlap_pages,
                    "parsed_content_length": len(parsed_content) if parsed_content else 0,
                    "paginated": file_path.lower().endswith('.pdf')
                })

            except Exception as parse_error:
                cursor.execute("UPDATE documents SET status = 'parse_failed' WHERE id = ?", (doc_id,))
                results.append({
                    "document_id": doc_id,
                    "status": "failed",
                    "error": str(parse_error)
                })

        except Exception as e:
            try:
                cursor.execute("UPDATE documents SET status = 'parse_failed' WHERE id = ?", (doc_id,))
            except Exception as db_e:
                pass
            results.append({
                "document_id": doc_id,
                "status": "failed",
                "error": str(e)
            })

    # Commit all changes
    try:
        db.commit()
    except:
        db.rollback()

    progress_context.update_progress_percent(100, f"Completed parsing {processed_docs}/{total_docs} documents")

    final_result = {
        "status": "completed",
        "operation": "parse_documents",
        "project_id": project_id,
        "total_documents": total_docs,
        "processed_documents": processed_docs,
        "parser": parser,
        "pages_per_split": pages_per_split,
        "overlap_pages": overlap_pages,
        "results": results
    }

    logger.debug(f"DEBUG_JOB_EXECUTOR_RETURN: Returning final_result with status={final_result['status']}, processed={final_result['processed_documents']}/{final_result['total_documents']}")
    return final_result







def _execute_taxonomy_processing(parameters: Dict[str, Any], progress_context) -> Dict[str, Any]:
    """Execute taxonomy processing with progress tracking."""
    from ...core.settings import backend_settings

    # Taxonomy parameters
    chunks = parameters.get("chunks", [])
    domain = parameters.get("domain", "general")
    depth = parameters.get("depth", 3)
    sample_size = parameters.get("sample_size")
    category_limits = parameters.get("category_limits")
    specificity_level = parameters.get("specificity_level", 1)
    generator_name = parameters.get("generator", "ollama")
    processing_mode = parameters.get("processing_mode", "fast")

    # Initialize taxonomy generator
    # Propagate model from parameters if available, otherwise get from database settings
    model = parameters.get("model")
    
    if generator_name == "gemini":
        from ...features.taxonomy.gemini_generator import GeminiTaxonomyGenerator
        api_key = backend_settings.get_setting("gemini_api_key")
        if not model:
            model = backend_settings.get_taxonomy_gemini_model()
        generator = GeminiTaxonomyGenerator(api_key=api_key, model=model)
        taxonomy_options = {}
    elif generator_name == "grok":
        from ...features.taxonomy.grok_generator import GrokTaxonomyGenerator
        api_key = backend_settings.get_setting("grok_api_key")
        if not model:
            model = backend_settings.get_taxonomy_grok_model()
        generator = GrokTaxonomyGenerator(grok_api_key=api_key, model=model)
        taxonomy_options = {}
    elif generator_name == "openai":
        from ...features.taxonomy.openai_generator import OpenAITaxonomyGenerator
        api_key = backend_settings.get_openai_api_key()
        if not model:
            model = backend_settings.get_taxonomy_openai_model()
        generator = OpenAITaxonomyGenerator(api_key=api_key, model=model)
        taxonomy_options = {}
    else:
        from ...features.taxonomy.ollama_generator import OllamaTaxonomyGenerator
        if not model:
            model = backend_settings.get_taxonomy_ollama_model()
        generator = OllamaTaxonomyGenerator(model=model)
        # Retrieve taxonomy Ollama parameters
        taxonomy_options = {
            "temperature": backend_settings.get_taxonomy_ollama_temperature(),
            "repeat_penalty": backend_settings.get_taxonomy_ollama_repeat_penalty(),
            "top_p": backend_settings.get_taxonomy_ollama_top_p(),
            "top_k": backend_settings.get_taxonomy_ollama_top_k(),
            "num_predict": backend_settings.get_taxonomy_ollama_num_predict(),
            "seed": backend_settings.get_taxonomy_ollama_seed()
        }

    # Generate taxonomy
    progress_context.update_progress(message="Generating taxonomy")
    
    # Common generation arguments
    gen_kwargs = {
        "chunks": chunks,
        "domain": domain,
        "depth": depth,
        "batch_size": sample_size,
        "category_limits": category_limits,
        "specificity_level": specificity_level,
        "processing_mode": processing_mode
    }
    
    # Add options for Ollama
    if generator_name == "ollama":
        gen_kwargs["options"] = taxonomy_options
        
    result = generator.generate_taxonomy(**gen_kwargs)

    progress_context.update_progress(message="Taxonomy generation completed")

    return {
        "status": "completed",
        "taxonomy_result": result,
        **parameters
    }

def _execute_dataset_generation(parameters: Dict[str, Any], progress_context) -> Dict[str, Any]:
    """Execute dataset generation with progress tracking."""
    from ...core.settings import backend_settings
    from ...features.datasetgen.generator import DatasetGenerator
    from ...features.datasetgen.prompt_builder import PromptBuilder
    from ...features.datasetgen.output_formatter import OutputFormatter
    from ...features.datasetgen.llm_interaction import LLMInteraction

    # Retrieve generation Ollama parameters
    generation_options = {
        "temperature": backend_settings.get_generation_ollama_temperature(),
        "repeat_penalty": backend_settings.get_generation_ollama_repeat_penalty(),
        "top_p": backend_settings.get_generation_ollama_top_p(),
        "top_k": backend_settings.get_generation_ollama_top_k(),
        "num_predict": backend_settings.get_generation_ollama_num_predict(),
        "seed": backend_settings.get_generation_ollama_seed()
    }

    # Extract parameters
    project_id = parameters.get("project_id")
    chunks = parameters.get("chunks")
    prompt_name = parameters.get("prompt_name", "qa_pairs")
    format_type = parameters.get("format_type", "json")
    concurrency = parameters.get("concurrency", 1)
    taxonomy_project = parameters.get("taxonomy_project")
    taxonomy_name = parameters.get("taxonomy_name")
    dataset_name = parameters.get("dataset_name")
    enable_versioning = parameters.get("enable_versioning", False)
    datasets_per_chunk = parameters.get("datasets_per_chunk", 3)
    prefer_extracted = parameters.get("prefer_extracted", True)

    # Get LLM provider and model from parameters
    # Propagate generation_model if available, otherwise get from database settings
    llm_provider = parameters.get("classification_provider") or parameters.get("classification_model", "ollama")
    generation_model = parameters.get("generation_model")
    
    # Get specific model and API key based on provider
    api_key = None
    
    if llm_provider == "gemini":
        api_key = backend_settings.get_setting("gemini_api_key")
        if not generation_model:
            generation_model = backend_settings.get_generation_gemini_model()
    elif llm_provider == "grok":
        api_key = backend_settings.get_setting("grok_api_key")
        if not generation_model:
            generation_model = backend_settings.get_generation_grok_model()
    elif llm_provider == "openai":
        api_key = backend_settings.get_openai_api_key()
        if not generation_model:
            generation_model = backend_settings.get_generation_openai_model()
    elif llm_provider == "ollama":
        if not generation_model:
            generation_model = backend_settings.get_generation_ollama_model()

    # Initialize components
    prompt_builder = PromptBuilder()
    output_formatter = OutputFormatter()
    llm_interaction = LLMInteraction(llm_provider=llm_provider, api_key=api_key, model=generation_model)

    # Create dataset generator
    generator = DatasetGenerator(
        prompt_builder=prompt_builder,
        output_formatter=output_formatter,
        llm_interaction=llm_interaction,
        document_repository=None,
        taxonomy_loader=None
    )

    # Generate dataset
    progress_context.update_progress(message="Generating dataset")
    result = generator.generate_dataset(
        project_id=project_id,
        chunks=chunks,
        prompt_name=prompt_name,
        format_type=format_type,
        concurrency=concurrency,
        taxonomy_project=taxonomy_project,
        taxonomy_name=taxonomy_name,
        dataset_name=dataset_name,
        enable_versioning=enable_versioning,
        datasets_per_chunk=datasets_per_chunk,
        prefer_extracted=prefer_extracted,
        options=generation_options
    )

    progress_context.update_progress(message="Dataset generation completed")

    return {
        "status": "completed",
        "dataset_result": result,
        **parameters
    }

def _execute_chunk_documents(parameters: Dict[str, Any], progress_context, db_connection: Any) -> Dict[str, Any]:
    """Execute document chunking with parsed documents using sliding window or sequential processing for multi-file documents."""
    from ...core.settings import backend_settings
    from ...features.chunk.cross_file_chunker import CrossFileChunker
    from ...features.chunk.engine import chunk_document

    project_id = parameters.get("project_id")
    if not isinstance(project_id, (str, int)) or not project_id:
        raise ValueError(f"Invalid project_id: {project_id}")

    document_ids = parameters.get("document_ids", [])
    chunker = parameters.get("chunker", "gemini")
    # Normalize strategy to lowercase to handle potential AI capitalization issues
    chunk_strategy = parameters.get("chunk_strategy", "character").lower()

    # Retrieve and set API keys for chunkers that need them
    # Only set API keys if the chunk strategy actually requires AI models
    needs_api_key = chunk_strategy in ["semantic", "llm_prompt"] or (chunk_strategy == "token" and chunker in ["gemini", "grok"])

    if needs_api_key:
        if chunker == "gemini":
            gemini_api_key = backend_settings.get_setting("gemini_api_key")
            if not gemini_api_key:
                raise ValueError("Gemini API key not configured. Please set gemini_api_key in settings.")
            os.environ['GOOGLE_API_KEY'] = gemini_api_key
        elif chunker == "grok":
            grok_api_key = backend_settings.get_setting("grok_api_key")
            if not grok_api_key:
                raise ValueError("Grok API key not configured. Please set grok_api_key in settings.")
            os.environ['GROK_API_KEY'] = grok_api_key
        elif chunker == "openai":
            openai_api_key = backend_settings.get_setting("openai_api_key")
            if not openai_api_key:
                raise ValueError("OpenAI API key not configured. Please set openai_api_key in settings.")
            os.environ['OPENAI_API_KEY'] = openai_api_key
        # For ollama, no API key needed
    chunk_size = parameters.get("chunk_size", 512)
    overlap = parameters.get("overlap", 50)
    delimiters = parameters.get("delimiters", ["\n\n", "\n"])
    semantic_prompt = parameters.get("semantic_prompt", "")
    schema_json = parameters.get("schema_definition", "")
    skip_parsing = parameters.get("skip_parsing", True)
    manifest_data = parameters.get("manifest_data")
    selected_files_map = parameters.get("selected_files")
    logger.debug(f"DEBUG_MANIFEST: manifest_data present={manifest_data is not None}")
    logger.debug(f"DEBUG_SELECTION: selected_files present={selected_files_map is not None}")
    # Get specific model name for chunker from database settings
    logger.debug(f"DEBUG_MODEL_SETUP: At start - chunker={chunker}")
    chunking_options = None
    from ...core.settings import BackendSettings
    
    # Propagate model from parameters if available
    model = parameters.get("model")
    
    if chunker == "ollama":
        if not model:
            model = BackendSettings.get_chunking_ollama_model()
        num_ctx = parameters.get("num_ctx") or BackendSettings.get_chunking_ollama_num_ctx()
        chunking_options = {"num_ctx": num_ctx}
        logger.debug(f"DEBUG_MODEL_SETUP: Ollama model set to: {model}, num_ctx: {num_ctx}")
    elif chunker == "gemini":
        if not model:
            model = BackendSettings.get_chunking_gemini_model()
        logger.debug(f"DEBUG_MODEL_SETUP: Gemini model set to: {model}")
    elif chunker == "grok":
        if not model:
            model = BackendSettings.get_chunking_grok_model()
        logger.debug(f"DEBUG_MODEL_SETUP: Grok model set to: {model}")
    elif chunker == "openai":
        if not model:
            model = BackendSettings.get_chunking_openai_model()
        logger.debug(f"DEBUG_MODEL_SETUP: OpenAI model set to: {model}")
    else:
        model = chunker  # Fallback to provider name if no model configured
        logger.debug(f"DEBUG_MODEL_SETUP: Fallback model set to: {model}")
    logger.debug(f"DEBUG_MODEL_SETUP: Final model value: {model}")

    # Get database connection (create our own since SQLite connections can't be shared across processes)
    from ...storage.src.database import get_db_connection
    db = get_db_connection()
    if not db:
        raise Exception("Database connection not available")

    # If skip_parsing is True, we need to retrieve manifest_data for multi-file documents
    if skip_parsing and not manifest_data and document_ids:
        logger.debug(f"DEBUG_MANIFEST: skip_parsing=True, retrieving manifest_data for document {document_ids[0]}")
        cursor = db.cursor()
        try:
            cursor.execute("""
                SELECT pf.file_path
                FROM parsed_documents pd
                JOIN parsed_files pf ON pd.id = pf.parsed_document_id
                WHERE pd.document_id = ?
                ORDER BY pf.page_number
            """, (document_ids[0],))
            parsed_files_rows = cursor.fetchall()
            if parsed_files_rows:
                manifest_data = {
                    "parsed_files": [row[0] for row in parsed_files_rows],
                    "document_id": document_ids[0]
                }
                logger.debug(f"DEBUG_MANIFEST: Retrieved manifest_data with {len(parsed_files_rows)} files")
            else:
                logger.debug(f"DEBUG_MANIFEST: No parsed files found for document {document_ids[0]}")
        except Exception as e:
            logger.debug(f"DEBUG_MANIFEST: Error retrieving manifest_data: {e}")
            raise

    total_docs = len(document_ids)
    processed_docs = 0
    total_chunks_created = 0
    results = []

    for i, doc_id in enumerate(document_ids):
        logger.debug(f"DEBUG_CHUNK_LOOP: Starting SEQUENTIAL processing of document {doc_id} (index {i})")
        try:
            progress_context.update_progress_percent((i / total_docs) * 100, f"Processing document {doc_id}")

            # Get document info from database
            cursor = db.cursor()
            cursor.execute("""
                SELECT file_name FROM documents
                WHERE id = ? AND project_id = ?
            """, (doc_id, project_id))

            doc_record = cursor.fetchone()
            if not doc_record:
                results.append({
                    "document_id": doc_id,
                    "status": "failed",
                    "error": f"Document not found"
                })
                continue

            file_name = doc_record[0]

            # Check if this is a multi-file document (has parsed_files) or single-file document
            # Query parsed_files table for this document
            cursor.execute("""
                SELECT pf.file_path FROM parsed_files pf
                JOIN parsed_documents pd ON pf.parsed_document_id = pd.id
                WHERE pd.document_id = ?
                ORDER BY pf.page_number
            """, (doc_id,))

            parsed_file_records = cursor.fetchall()
            parsed_files = [record[0] for record in parsed_file_records]

            if parsed_files:
                # Multi-file document: use cross-file chunking
                logger.debug(f"DEBUG_MULTI_FILE: Multi-file document detected for {doc_id}, {len(parsed_files)} files")

                # Apply file selection filtering if active
                if selected_files_map:
                    doc_id_str = str(doc_id)
                    if doc_id_str in selected_files_map:
                        selected_paths = set(selected_files_map[doc_id_str])
                        original_count = len(parsed_files)
                        # Filter parsed_files keeping original order
                        parsed_files = [f for f in parsed_files if f in selected_paths]
                        logger.debug(f"DEBUG_FILTER: Filtered document {doc_id} from {original_count} to {len(parsed_files)} files based on selection")

                # Create manifest data from parsed files
                manifest_data = {
                    "parsed_files": parsed_files,
                    "total_pages": len(parsed_files),
                    "config": {
                        "pages_per_split": 200,  # Default values
                        "overlap_pages": 1
                    }
                }

                try:
                    logger.debug(f"DEBUG_CHUNK_EXEC: About to create CrossFileChunker for doc {doc_id}")
                    cross_file_chunker = CrossFileChunker(db)
                    logger.debug(f"DEBUG_CHUNK_EXEC: About to call chunk_document for doc {doc_id}")
                    chunk_result = cross_file_chunker.chunk_document(
                        doc_id, project_id, manifest_data, chunk_strategy,
                        chunk_size, overlap, model, chunker, chunking_options,
                        schema_json, delimiters
                    )
                    logger.debug(f"DEBUG_CHUNK_EXEC: chunk_document returned: {chunk_result}")
                except Exception as e:
                    logger.error(f"DEBUG_CHUNK_EXEC: Exception in cross-file chunking for doc {doc_id}: {e}", exc_info=True)
                    results.append({
                        "document_id": doc_id,
                        "status": "failed",
                        "error": f"Cross-file chunking failed: {str(e)}"
                    })
                    continue

                if chunk_result is None:
                    # Handle cases where chunking returns None
                    doc_chunks_created = 0
                else:
                    doc_chunks_created = chunk_result.get("chunks_created", 0)

            else:
                # SINGLE-FILE DOCUMENT: Traditional processing
                logger.debug(f"DEBUG_SINGLE_FILE: Single-file document detected for {doc_id}, using traditional processing")

                # Get parsed file path for single-file documents from parsed_files table
                cursor.execute("""
                    SELECT pf.file_path FROM parsed_files pf
                    JOIN parsed_documents pd ON pf.parsed_document_id = pd.id
                    WHERE pd.document_id = ?
                    ORDER BY pf.page_number
                    LIMIT 1
                """, (doc_id,))

                single_file_record = cursor.fetchone()
                if not single_file_record or not single_file_record[0]:
                    results.append({
                        "document_id": doc_id,
                        "status": "failed",
                        "error": "No parsed file found for single-file document"
                    })
                    continue

                parsed_file_path = single_file_record[0]
                if not os.path.exists(parsed_file_path):
                    results.append({
                        "document_id": doc_id,
                        "status": "failed",
                        "error": f"Parsed file not found: {parsed_file_path}"
                    })
                    continue

                logger.debug(f"DEBUG_SINGLE_FILE: Processing single file: {os.path.basename(parsed_file_path)}")

                try:
                    # Create chunking strategy object
                    from ...features.chunk.schema import CharacterStrategy, LLMPromptStrategy, SchemaStrategy, DelimiterStrategy, TokenStrategy

                    if chunk_strategy == "character":
                        strategy = CharacterStrategy(
                            strategy="character",
                            chunk_size=chunk_size,
                            overlap=overlap
                        )
                    elif chunk_strategy == "token":
                        strategy = TokenStrategy(
                            strategy="token",
                            chunk_size=chunk_size,
                            overlap=overlap,
                            model="cl100k_base"
                        )
                    elif chunk_strategy == "semantic":
                        user_instruction = parameters.get("semantic_prompt", "Split the document by topic.")

                        # Detect if this is logical chunking (asking for specific number of chunks)
                        # or boundary-based chunking (asking to find existing headings)
                        is_logical_chunking = any(phrase in user_instruction.lower() for phrase in [
                            "split into", "create chunks", "major sections", "logical chunks"
                        ]) and any(char.isdigit() for char in user_instruction)

                        if is_logical_chunking:
                            # Logical chunking: AI should return actual chunks
                            LOGICAL_CHUNK_PROMPT_TEMPLATE = """You are an expert document analysis tool. Your task is to split a document into logical chunks based on the user's instruction.

**User Instruction:**
{user_instruction}

**Output Requirements:**
- Analyze the document and create logical chunks according to the instruction
- Return the chunks as a JSON array of strings
- Each chunk should be a coherent, self-contained section
- Ensure chunks flow logically and maintain semantic coherence
- Do not include chunk numbers or headers in the output, just the chunk content

**Example Output Format:**
["First chunk content here...", "Second chunk content here...", "Third chunk content here..."]
"""
                            strategy = LLMPromptStrategy(
                                strategy="llm_prompt",
                                model=parameters.get("model", model),
                                prompt_template=LOGICAL_CHUNK_PROMPT_TEMPLATE,
                                user_instruction=user_instruction,
                                options=chunking_options,
                                system_instruction="You are an expert document analysis tool. Follow the user's instructions precisely to split the document into logical chunks."
                            )
                        else:
                            # Boundary-based chunking: Find existing headings
                            SEMANTIC_CHUNK_PROMPT_TEMPLATE = """You are an expert document analysis tool. Your task is to split a document into logical chunks based on a user's instruction. You will be given an instruction and the document text. You must identify the exact headings or titles that mark the beginning of a new chunk according to the instruction.

**User Instruction:**
{user_instruction}

**Output Requirements:**
- Return ONLY a comma-separated list of the exact heading strings that should start a new chunk.
- Do not include any other text, explanations, or formatting.

**Example:**
If the instruction is "Split by chapter" and the text contains "# Chapter 1" and "# Chapter 2", your output should be:
# Chapter 1,# Chapter 2
"""
                            strategy = LLMPromptStrategy(
                                strategy="llm_prompt",
                                model=parameters.get("model", model),
                                prompt_template=SEMANTIC_CHUNK_PROMPT_TEMPLATE,
                                user_instruction=user_instruction,
                                options=chunking_options,
                                system_instruction="You are an expert document analysis tool. Follow the user's instructions precisely to identify chunk boundaries."
                            )
                    elif chunk_strategy == "schema":
                        strategy = SchemaStrategy(
                            strategy="schema",
                            json_schema=schema_json
                        )
                    elif chunk_strategy == "delimiter":
                        strategy = DelimiterStrategy(
                            strategy="delimiter",
                            delimiter=delimiters if delimiters else "\n\n"
                        )
                    else:
                        strategy = CharacterStrategy(
                            strategy="character",
                            chunk_size=chunk_size,
                            overlap=overlap
                        )

                    # Chunk the single file
                    with open(parsed_file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    file_chunks = chunk_document(content, strategy, None)
                    logger.debug(f"DEBUG_SINGLE_FILE: Generated {len(file_chunks)} chunks from single file")

                    # Save chunks to database and filesystem
                    try:
                        from ...storage.src.project.file_manager import FileManager
                        from ...storage.src.project.database_repositories import ChunkRepository
                        import uuid
                        from datetime import datetime

                        file_manager = FileManager()
                        chunk_repo = ChunkRepository(db)

                        for chunk_idx, chunk in enumerate(file_chunks):
                            chunk_file_name = f"chunk_{chunk_idx + 1}.md"
                            
                            # Ensure chunks directory exists
                            chunk_storage_path = file_manager.get_storage_base_path() / "chunks" / str(project_id) / str(doc_id)
                            chunk_storage_path.mkdir(parents=True, exist_ok=True)
                            
                            # Save chunk file
                            full_chunk_path = chunk_storage_path / chunk_file_name
                            with open(full_chunk_path, "wb") as f:
                                f.write(chunk.encode("utf-8"))
                            
                            # Store relative path
                            relative_chunk_path = str(full_chunk_path.relative_to(file_manager.get_storage_base_path()))
                            
                            chunk_repo.create_chunk({
                                "id": str(uuid.uuid4()),
                                "document_id": doc_id,
                                "chunk_index": chunk_idx + 1,  # Use 1-based indexing for consistency with previous implementation
                                "content": chunk,
                                "metadata": {"strategy": chunk_strategy, "file_path": relative_chunk_path},
                                "created_at": datetime.utcnow().isoformat()
                            })

                        doc_chunks_created = len(file_chunks)

                    except Exception as chunk_save_error:
                        logger.error(f"DEBUG_SINGLE_FILE: Failed to save chunks for doc {doc_id}: {chunk_save_error}")
                        results.append({
                            "document_id": doc_id,
                            "status": "failed",
                            "error": f"Chunking succeeded but saving failed: {str(chunk_save_error)}"
                        })
                        continue

                except Exception as file_chunk_error:
                    logger.error(f"DEBUG_SINGLE_FILE: Failed to chunk single file for doc {doc_id}: {file_chunk_error}")
                    results.append({
                        "document_id": doc_id,
                        "status": "failed",
                        "error": str(file_chunk_error)
                    })
                    continue

            # Update document status to 'chunked' after successful chunking
            cursor.execute("""
                UPDATE documents
                SET status = 'chunked'
                WHERE id = ?
            """, (doc_id,))

            total_chunks_created += doc_chunks_created
            processed_docs += 1

            processing_mode = "sliding_window" if len(parsed_files) > 1 else ("sequential" if 'parsed_files' in locals() else "single_file")
            results.append({
                "document_id": doc_id,
                "status": "completed",
                "chunks_created": doc_chunks_created,
                "parsed_files_processed": len(parsed_files) if 'parsed_files' in locals() else 1,
                "chunk_strategy": chunk_strategy,
                "processing_mode": processing_mode
            })

            logger.debug(f"DEBUG_CHUNKING: Document {doc_id} completed - {doc_chunks_created} chunks created")

        except Exception as e:
            from datetime import datetime
            debug_context = {
                "timestamp": datetime.now().isoformat(),
                "level": "ERROR",
                "component": "_execute_chunk_documents",
                "message": "Exception during document chunking",
                "context": {
                    "document_id": doc_id,
                    "project_id": project_id,
                    "chunk_strategy": chunk_strategy,
                    "error": str(e)
                }
            }
            logger.error(json.dumps(debug_context))

            results.append({
                "document_id": doc_id,
                "status": "failed",
                "error": str(e)
            })

    # Commit all changes
    try:
        db.commit()
    except:
        db.rollback()

    progress_context.update_progress_percent(100, f"Completed chunking {processed_docs}/{total_docs} documents")

    final_result = {
        "status": "completed",
        "operation": "chunk_documents",
        "project_id": project_id,
        "total_documents": total_docs,
        "processed_documents": processed_docs,
        "total_chunks_created": total_chunks_created,
        "chunk_strategy": chunk_strategy,
        "processing_mode": "sequential_multi_file",
        "results": results
    }

    from datetime import datetime
    debug_context = {
        "timestamp": datetime.now().isoformat(),
        "level": "INFO",
        "component": "_execute_chunk_documents",
        "message": "Document chunking job execution completed",
        "context": {
            "project_id": project_id,
            "total_documents": total_docs,
            "processed_documents": processed_docs,
            "total_chunks_created": total_chunks_created,
            "failed_documents": total_docs - processed_docs
        }
    }
    logger.debug(json.dumps(debug_context))

    return final_result


def execute_benchmark_job(parameters: Dict[str, Any], progress_context) -> Dict[str, Any]:
    """Execute benchmark evaluation job."""
    from ...core.settings import backend_settings
    from ...features.benchmarking.evaluator import BenchmarkEvaluator
    from ...features.benchmarking.suites.glue import GLUEBenchmark
    from ...storage.src.project.database_repositories import BenchmarkJobRepository, BenchmarkResultRepository
    from ...storage.src.database import get_db_connection

    # Extract parameters
    job_id = progress_context.job.job_id  # Get job_id from progress context
    benchmark_suite = parameters.get("benchmark_suite", "glue")
    ai_config = parameters.get("ai_config", {})
    benchmark_params = parameters.get("benchmark_params", {})

    if not job_id:
        raise ValueError("job_id is required")

    # Get database connection
    db = get_db_connection()
    if not db:
        raise Exception("Database connection not available")

    # Initialize repositories
    job_repo = BenchmarkJobRepository(db)
    result_repo = BenchmarkResultRepository(db)

    try:
        # Update job status to running
        job_repo.update_job_status(job_id, "running")

        # Initialize evaluator and register benchmarks
        evaluator = BenchmarkEvaluator()

        if benchmark_suite == "glue":
            benchmark = GLUEBenchmark()
            evaluator.register_benchmark("glue", benchmark)
            evaluator.register_metric("accuracy", None)  # TODO: Implement metrics
            evaluator.register_metric("f1", None)

        # Get AI config from environment variables (injected by RQ worker)
        import os
        gemini_key = os.getenv('GOOGLE_API_KEY') or backend_settings.get_setting('gemini_api_key')
        grok_key = os.getenv('GROK_API_KEY') or backend_settings.get_setting('grok_api_key')
        openai_key = os.getenv('OPENAI_API_KEY') or backend_settings.get_setting('openai_api_key')

        ai_config_full = {
            'gemini_api_key': gemini_key,
            'grok_api_key': grok_key,
            'openai_api_key': openai_key,
            'ollama_available': backend_settings.get_setting('ollama_available', True),
            'model_name': ai_config.get('selected_model', ai_config.get('model_name', 'unknown')),
            'provider': ai_config.get('provider')
        }

        # Run benchmarks
        progress_context.update_progress(message="Running benchmark evaluation")
        benchmark_results = evaluator.run_benchmarks(
            ai_config=ai_config_full,
            benchmark_names=[benchmark_suite],
            **benchmark_params
        )

        # Store results in database
        for result in benchmark_results:
            result_repo.create_result({
                'id': str(uuid.uuid4()),
                'job_id': job_id,
                'metrics': result.metrics,
                'metadata': result.metadata,
                'created_at': datetime.utcnow()
            })

        # Update job status to completed
        job_repo.update_job_status(job_id, "completed")

        progress_context.update_progress(message="Benchmark evaluation completed")

        return {
            "status": "completed",
            "job_id": job_id,
            "benchmark_suite": benchmark_suite,
            "results_count": len(benchmark_results)
        }

    except Exception as e:
        # Update job status to failed
        job_repo.update_job_status(job_id, "failed", str(e))
        raise








