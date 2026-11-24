from typing import Dict, Any, Optional, List, Union, Tuple
import os
import json
import uuid
from datetime import datetime

from ...core.logging import get_logger

logger = get_logger(__name__)

def _execute_parse_documents(parameters: Dict[str, Any], progress_context, db_connection: Any) -> Dict[str, Any]:
    """Execute document parsing with PDF splitting support.
    
    This executor now handles single-document jobs. Each document is submitted
    as its own job by the API/GUI, so this function processes exactly one document.
    """
    from datetime import datetime
    import json
    import sys

    from src.compileo.features.ingestion.main import parse_document
    from ...core.settings import BackendSettings

    project_id = parameters.get("project_id")
    owner_id = parameters.get("owner_id")
    document_ids = parameters.get("document_ids", [])
    parser = parameters.get("parser", "grok")
    pages_per_split = parameters.get("pages_per_split", 50)
    overlap_pages = parameters.get("overlap_pages", 0)
    page_selection = parameters.get("page_selection")

    # Support both single-document and multi-document formats for backward compatibility
    if len(document_ids) == 0:
        # Single document mode - check for document_id (singular)
        doc_id = parameters.get("document_id")
        if doc_id:
            document_ids = [doc_id]
        else:
            raise ValueError("No document_ids provided for parse_documents")

    # Debug: Show parser selection
    logger.debug(f"DEBUG_PARSER_SELECTED: parser={parser}, model={parameters.get('model')}, file_count={len(document_ids)}")

    # Use user-aware settings
    user_settings = BackendSettings(owner_id=owner_id)

    # Get database connection using context manager
    from ...storage.src.database import get_tenant_db_connection

    total_docs = len(document_ids)
    processed_docs = 0
    results = []

    if not owner_id:
        raise ValueError("owner_id is required for parse_documents")

    for i, doc_id in enumerate(document_ids):
        try:
            # Update progress at document level
            progress_context.update_progress_percent((i / total_docs) * 100, f"Processing document {doc_id}")

            file_name = None
            file_path = None

            # Get document info with short-lived connection
            with get_tenant_db_connection(owner_id) as db:
                cursor = db.cursor()
                cursor.execute("""
                    SELECT file_name, source_file_path FROM documents
                    WHERE id = ? AND project_id = ? AND owner_id = ?
                """, (doc_id, project_id, owner_id))

                doc_record = cursor.fetchone()
                if not doc_record:
                    results.append({
                        "document_id": doc_id,
                        "status": "failed",
                        "error": f"Document not found or access denied"
                    })
                    continue

                file_name, file_path = doc_record

            # Check if file exists
            if not file_path or not os.path.exists(file_path):
                error_msg = f"File not found: {file_path}"
                results.append({
                    "document_id": doc_id,
                    "status": "failed",
                    "error": error_msg
                })
                continue

            # Get API key and specific model for parser from user settings
            # Propagate from parameters if available, otherwise get from database settings
            api_key = None
            specific_model = parameters.get("model")
            ollama_options = None

            if parser == "grok":
                api_key = user_settings.get_setting("grok_api_key")
                if api_key:
                    os.environ['GROK_API_KEY'] = api_key
                if not specific_model:
                    specific_model = user_settings.get_setting("parsing_grok_model", "grok-4-fast-reasoning")
            elif parser == "gemini":
                api_key = user_settings.get_setting("gemini_api_key")
                if api_key:
                    os.environ['GEMINI_API_KEY'] = api_key
                    os.environ['GOOGLE_API_KEY'] = api_key
                if not specific_model:
                    specific_model = user_settings.get_setting("parsing_gemini_model", "gemini-2.5-flash")
            elif parser == "openai":
                api_key = user_settings.get_openai_api_key()
                if api_key:
                    os.environ['OPENAI_API_KEY'] = api_key
                if not specific_model:
                    specific_model = user_settings.get_parsing_openai_model()
            elif parser == "ollama":
                if not specific_model:
                    specific_model = user_settings.get_setting("parsing_ollama_model", "benhaotang/Nanonets-OCR-s:latest")
                ollama_options = {
                    "temperature": user_settings.get_parsing_ollama_temperature(),
                    "repeat_penalty": user_settings.get_parsing_ollama_repeat_penalty(),
                    "top_p": user_settings.get_parsing_ollama_top_p(),
                    "top_k": user_settings.get_parsing_ollama_top_k(),
                    "num_predict": user_settings.get_parsing_ollama_num_predict(),
                    "seed": user_settings.get_parsing_ollama_seed()
                }
            elif parser == "huggingface":
                api_key = user_settings.get_setting("huggingface_api_key")
                if api_key:
                    os.environ['HUGGING_FACE_HUB_TOKEN'] = api_key

            try:
                # Prepare options for Ollama if applicable
                parse_options = ollama_options if parser == "ollama" else None

                # Debug: Show parsing start
                is_pdf = file_path.lower().endswith('.pdf')
                logger.debug(f"DEBUG_PARSING_START: doc_id={doc_id}, parser={parser}, model={specific_model}, is_pdf={is_pdf}, file={os.path.basename(file_path)}")

                # Call the ingestion pipeline - handles its own DB connection
                parsed_content = parse_document(
                    file_path=file_path,
                    parser=parser,
                    api_key=api_key,
                    model=specific_model,
                    document_id=doc_id,
                    project_id=project_id,
                    owner_id=owner_id,
                    pages_per_split=pages_per_split,
                    overlap_pages=overlap_pages,
                    page_selection=page_selection,
                    options=parse_options
                )

                # Debug: Show parsing complete
                content_len = len(parsed_content) if parsed_content else 0
                logger.debug(f"DEBUG_PARSING_COMPLETE: doc_id={doc_id}, parser={parser}, content_length={content_len}")

                # Status update with short-lived connection
                with get_tenant_db_connection(owner_id) as db:
                    cursor = db.cursor()
                    cursor.execute("""
                        UPDATE documents
                        SET status = 'parsed'
                        WHERE id = ? AND owner_id = ?
                    """, (doc_id, owner_id))
                    db.commit()

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
                with get_tenant_db_connection(owner_id) as db:
                    cursor = db.cursor()
                    cursor.execute("UPDATE documents SET status = 'parse_failed' WHERE id = ? AND owner_id = ?", (doc_id, owner_id))
                    db.commit()
                results.append({
                    "document_id": doc_id,
                    "status": "failed",
                    "error": str(parse_error)
                })

        except Exception as e:
            try:
                with get_tenant_db_connection(owner_id) as db:
                    cursor = db.cursor()
                    cursor.execute("UPDATE documents SET status = 'parse_failed' WHERE id = ? AND owner_id = ?", (doc_id, owner_id))
                    db.commit()
            except:
                pass
            results.append({
                "document_id": doc_id,
                "status": "failed",
                "error": str(e)
            })

    progress_context.update_progress_percent(100, f"Completed parsing {processed_docs}/{total_docs} documents")

    return {
        "status": "completed",
        "operation": "parse_documents",
        "project_id": project_id,
        "total_documents": total_docs,
        "processed_documents": processed_docs,
        "results": results
    }

def execute_benchmark_job(parameters: Dict[str, Any], progress_context) -> Dict[str, Any]:
    """Execute benchmark evaluation job with user isolation."""
    from ...core.settings import BackendSettings
    from ...features.benchmarking.evaluator import BenchmarkEvaluator
    from ...features.benchmarking.suites.glue import GLUEBenchmark
    from ...storage.src.project.database_repositories import BenchmarkJobRepository, BenchmarkResultRepository
    from ...storage.src.database import get_tenant_db_connection

    job_id = progress_context.job.job_id
    owner_id = progress_context.job.user_id
    benchmark_suite = parameters.get("benchmark_suite", "glue")
    ai_config = parameters.get("ai_config", {})
    benchmark_params = parameters.get("benchmark_params", {})

    if not owner_id:
        raise ValueError("owner_id is required for benchmarking job")

    user_settings = BackendSettings(owner_id=owner_id)

    with get_tenant_db_connection(owner_id) as db:
        job_repo = BenchmarkJobRepository(owner_id)
        result_repo = BenchmarkResultRepository(owner_id)

        try:
            job_repo.update_job_status(job_id, "running")
            evaluator = BenchmarkEvaluator()

            if benchmark_suite == "glue":
                benchmark = GLUEBenchmark()
                evaluator.register_benchmark("glue", benchmark)

            gemini_key = os.getenv('GOOGLE_API_KEY') or user_settings.get_setting('gemini_api_key')
            grok_key = os.getenv('GROK_API_KEY') or user_settings.get_setting('grok_api_key')
            openai_key = os.getenv('OPENAI_API_KEY') or user_settings.get_setting('openai_api_key')

            ai_config_full = {
                'gemini_api_key': gemini_key,
                'grok_api_key': grok_key,
                'openai_api_key': openai_key,
                'ollama_available': user_settings.get_setting('ollama_available', True),
                'model_name': ai_config.get('selected_model', 'unknown'),
                'provider': ai_config.get('provider'),
                'ollama_params': ai_config.get('ollama_params', {})
            }

            results = evaluator.run_benchmarks(ai_config=ai_config_full, benchmark_names=[benchmark_suite], **benchmark_params)

            for result in results:
                result_repo.create_result({
                    'id': str(uuid.uuid4()),
                    'owner_id': owner_id,
                    'job_id': job_id,
                    'metrics': result.metrics,
                    'metadata': result.metadata,
                    'created_at': datetime.utcnow()
                })

            job_repo.update_job_status(job_id, "completed")
            return {"status": "completed", "job_id": job_id, "results_count": len(results)}
        except Exception as e:
            job_repo.update_job_status(job_id, "failed", str(e))
            raise

def _execute_taxonomy_processing(parameters: Dict[str, Any], progress_context) -> Dict[str, Any]:
    """Execute taxonomy processing with user-scoped settings."""
    from ...core.settings import BackendSettings
    
    owner_id = parameters.get("owner_id")
    user_settings = BackendSettings(owner_id=owner_id)

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
    model = parameters.get("model")
    
    if generator_name == "gemini":
        from ...features.taxonomy.gemini_generator import GeminiTaxonomyGenerator
        api_key = user_settings.get_setting("gemini_api_key")
        if api_key:
            os.environ['GEMINI_API_KEY'] = api_key
            os.environ['GOOGLE_API_KEY'] = api_key
        if not model:
            model = user_settings.get_taxonomy_gemini_model()
        generator = GeminiTaxonomyGenerator(api_key=api_key, model=model)
        taxonomy_options = {}
    elif generator_name == "grok":
        from ...features.taxonomy.grok_generator import GrokTaxonomyGenerator
        api_key = user_settings.get_setting("grok_api_key")
        if api_key:
            os.environ['GROK_API_KEY'] = api_key
        if not model:
            model = user_settings.get_taxonomy_grok_model()
        generator = GrokTaxonomyGenerator(grok_api_key=api_key, model=model)
        taxonomy_options = {}
    elif generator_name == "openai":
        from ...features.taxonomy.openai_generator import OpenAITaxonomyGenerator
        api_key = user_settings.get_openai_api_key()
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key
        if not model:
            model = user_settings.get_taxonomy_openai_model()
        generator = OpenAITaxonomyGenerator(api_key=api_key, model=model)
        taxonomy_options = {}
    else:
        from ...features.taxonomy.ollama_generator import OllamaTaxonomyGenerator
        if not model:
            model = user_settings.get_taxonomy_ollama_model()
        generator = OllamaTaxonomyGenerator(model=model)
        # Retrieve taxonomy Ollama parameters
        taxonomy_options = {
            "temperature": user_settings.get_taxonomy_ollama_temperature(),
            "repeat_penalty": user_settings.get_taxonomy_ollama_repeat_penalty(),
            "top_p": user_settings.get_taxonomy_ollama_top_p(),
            "top_k": user_settings.get_taxonomy_ollama_top_k(),
            "num_predict": user_settings.get_taxonomy_ollama_num_predict(),
            "seed": user_settings.get_taxonomy_ollama_seed()
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
        "taxonomy_result": result
    }

def _execute_dataset_generation(parameters: Dict[str, Any], progress_context) -> Dict[str, Any]:
    """Execute dataset generation with progress tracking, batching and versioning support."""
    from ...core.settings import BackendSettings
    from ...features.datasetgen.generator import DatasetGenerator
    from ...features.datasetgen.prompt_builder import PromptBuilder
    from ...features.datasetgen.output_formatter import OutputFormatter
    from ...features.datasetgen.llm_interaction import LLMInteraction
    from ...storage.src.project.database_repositories import (
        PromptRepository, DatasetParameterRepository, DocumentRepository,
        ChunkRepository, ExtractionResultRepository, TaxonomyRepository,
        DatasetJobRepository, ProjectRepository
    )
    from ...storage.src.database import get_tenant_db_connection

    # Get job ID and basic params
    job_id = progress_context.job.job_id
    project_id = parameters.get("project_id")
    owner_id = parameters.get("owner_id")
    
    if not owner_id:
         raise ValueError("owner_id must be provided for dataset generation job")

    with get_tenant_db_connection(owner_id) as db:
        # Initialize repositories with owner_id
        prompt_repo = PromptRepository(owner_id)
        dataset_param_repo = DatasetParameterRepository(owner_id)
        document_repo = DocumentRepository(owner_id)
        chunk_repo = ChunkRepository(owner_id)
        extraction_result_repo = ExtractionResultRepository(owner_id)
        job_repo = DatasetJobRepository(owner_id)
        project_repo = ProjectRepository(owner_id)

        # Retrieve project name
        cursor = db.cursor()
        cursor.execute("SELECT name FROM projects WHERE id = ? AND owner_id = ?", (project_id, owner_id))
        project_row = cursor.fetchone()
        if not project_row:
            raise ValueError(f"Project with ID {project_id} not found or access denied")
        project_name = project_row[0]

        # Retrieve user-specific settings
        user_settings = BackendSettings(owner_id=owner_id)

        # Retrieve generation Ollama parameters
        generation_options = {
            "temperature": user_settings.get_generation_ollama_temperature(),
            "repeat_penalty": user_settings.get_generation_ollama_repeat_penalty(),
            "top_p": user_settings.get_generation_ollama_top_p(),
            "top_k": user_settings.get_generation_ollama_top_k(),
            "num_predict": user_settings.get_generation_ollama_num_predict(),
            "seed": user_settings.get_generation_ollama_seed()
        }

        # Extract generation parameters
        prompt_name = parameters.get("prompt_name", "qa_pairs")
        format_type = parameters.get("format_type", "json")
        concurrency = parameters.get("concurrency", 1)
        batch_size_param = parameters.get("batch_size", 50)
        taxonomy_project = parameters.get("taxonomy_project")
        taxonomy_name = parameters.get("taxonomy_name")
        dataset_name = parameters.get("dataset_name")
        enable_versioning = parameters.get("enable_versioning", False)
        datasets_per_chunk = parameters.get("datasets_per_chunk", 3)
        data_source = parameters.get("data_source", "Chunks Only")
        custom_prompt = parameters.get("custom_prompt")
        generation_mode = parameters.get("generation_mode", "default")
        
        # Standardize generation_mode format to match generator.py _strategy_map
        if generation_mode == "instruction following":
            generation_mode = "instruction_following"
            
        extraction_file_id = parameters.get("extraction_file_id")
        selected_categories = parameters.get("selected_categories")
        only_validated = parameters.get("only_validated", False)

        # Resolve models
        llm_provider = parameters.get("classification_provider") or parameters.get("classification_model", "ollama")
        generation_model = parameters.get("generation_model")
        
        # Resolve default model from user settings if not provided
        if not generation_model:
            if llm_provider == "gemini":
                generation_model = user_settings.get_generation_gemini_model()
            elif llm_provider == "grok":
                generation_model = user_settings.get_generation_grok_model()
            elif llm_provider == "openai":
                generation_model = user_settings.get_generation_openai_model()
            elif llm_provider == "ollama":
                generation_model = user_settings.get_generation_ollama_model()

        api_key = None
        if llm_provider == "gemini":
            api_key = user_settings.get_setting("gemini_api_key")
            if api_key:
                os.environ['GEMINI_API_KEY'] = api_key
                os.environ['GOOGLE_API_KEY'] = api_key
        elif llm_provider == "grok":
            api_key = user_settings.get_setting("grok_api_key")
            if api_key:
                os.environ['GROK_API_KEY'] = api_key
        elif llm_provider == "openai":
            api_key = user_settings.get_openai_api_key()
            if api_key:
                os.environ['OPENAI_API_KEY'] = api_key

        if not api_key and llm_provider != "ollama":
            raise ValueError(f"API key not configured for {llm_provider}")

        # Initialize core components
        prompt_builder = PromptBuilder(prompt_repo, dataset_param_repo)
        output_formatter = OutputFormatter()
        llm_interaction = LLMInteraction(
            llm_provider=llm_provider,
            api_key=str(api_key) if api_key else None,
            model=generation_model
        )

        # Initialize taxonomy loader if needed
        taxonomy_loader = None
        if taxonomy_project and taxonomy_name:
            taxonomy_repo = TaxonomyRepository(owner_id)
            from ...features.taxonomy.loader import TaxonomyLoader
            taxonomy_loader = TaxonomyLoader(taxonomy_repo, project_repo)

        # Initialize version manager if needed
        version_manager = None
        if enable_versioning:
            from ...features.datasetgen.version_manager import DatasetVersionManager
            from ...storage.src.project.database_repositories import DatasetVersionRepository, DatasetChangeRepository, DatasetLineageRepository
            version_repo = DatasetVersionRepository(owner_id)
            change_repo = DatasetChangeRepository(owner_id)
            lineage_repo = DatasetLineageRepository(owner_id)
            version_manager = DatasetVersionManager(version_repo, change_repo, lineage_repo)

        # Create dataset generator
        dataset_generator = DatasetGenerator(
            prompt_builder=prompt_builder,
            output_formatter=output_formatter,
            llm_interaction=llm_interaction,
            document_repository=document_repo,
            extraction_result_repository=extraction_result_repo,
            taxonomy_loader=taxonomy_loader,
            version_manager=version_manager,
        )

        progress_context.update_progress_percent(10, "Loading project chunks")
        job_repo.update_job_progress(job_id, 0.1)

        # Get chunks belonging to user, filtered by documents if specified
        document_ids = parameters.get("document_ids")
        if document_ids:
            placeholders = ','.join(['?'] * len(document_ids))
            query = f"""
                SELECT c.file_path FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE d.project_id = ? AND c.owner_id = ? AND d.owner_id = ?
                AND c.document_id IN ({placeholders})
            """
            cursor.execute(query, [project_id, owner_id, owner_id] + document_ids)
        else:
            cursor.execute("""
                SELECT c.file_path FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE d.project_id = ? AND c.owner_id = ? AND d.owner_id = ?
            """, (project_id, owner_id, owner_id))
        
        chunk_files = cursor.fetchall()
        
        chunks = []
        from ...storage.src.project.file_manager import FileManager
        storage_base_path = FileManager.get_storage_base_path()

        for chunk_file_row in chunk_files:
            try:
                chunk_file_path = chunk_file_row[0]
                if not os.path.isabs(chunk_file_path):
                    chunk_file_path = storage_base_path / chunk_file_path
                if os.path.exists(chunk_file_path):
                    with open(chunk_file_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if content: chunks.append(content)
            except: pass

        if not chunks and data_source != "Extract":
            raise ValueError(f"No chunks found for project {project_id}")

        progress_context.update_progress_percent(25, "Processing data")
        job_repo.update_job_progress(job_id, 0.25)

        all_results = []
        batch_files = []
        
        # Handle Extract mode
        if data_source == "Extract":
            def update_gen_progress(val, msg):
                progress_context.update_progress_percent(val * 100, msg)
                job_repo.update_job_progress(job_id, val)

            update_gen_progress(0.4, "Generating from extracted entities")
            
            batch_result = dataset_generator.generate_dataset(
                project_id=int(project_id) if str(project_id).isdigit() else project_id,
                owner_id=owner_id,
                prompt_name=prompt_name,
                format_type="raw_list",
                concurrency=concurrency,
                datasets_per_chunk=datasets_per_chunk,
                extraction_file_id=extraction_file_id,
                custom_prompt=custom_prompt,
                generation_mode=generation_mode,
                selected_categories=selected_categories,
                only_validated=only_validated,
                prefer_extracted=True
            )
            
            batch_valid_results = []
            if isinstance(batch_result, list):
                batch_valid_results = [r for r in batch_result if r]
                all_results.extend(batch_valid_results)

            if batch_valid_results:
                file_manager = FileManager(storage_type='datasets')
                ext = "json"
                try:
                    from ...features.plugin.manager import plugin_manager
                    metadata = plugin_manager.get_format_metadata(format_type)
                    if metadata and 'file_extension' in metadata:
                        ext = metadata['file_extension']
                except: pass
                
                batch_filename = f"dataset_{job_id}_extract.{ext}"
                
                serializable_results = []
                for item in batch_valid_results:
                    if not isinstance(item, dict): continue
                    clean_item = {k: (v.decode('utf-8') if isinstance(v, bytes) else v) for k, v in item.items()}
                    serializable_results.append(clean_item)

                batch_content = output_formatter.format_dataset(serializable_results, format_type, owner_id)
                if isinstance(batch_content, str): batch_content = batch_content.encode('utf-8')
                
                batch_file_path = file_manager.store_file(
                    project_id=project_id,
                    document_id=None,
                    file_name=batch_filename,
                    file_content=batch_content, # type: ignore
                    owner_id=owner_id
                )
                batch_files.append({
                    "batch_index": 0, "file_path": str(batch_file_path),
                    "entries_count": len(batch_valid_results), "chunks_processed": 0
                })
        else:
            # Batch processing for other modes
            total_chunks = len(chunks)
            batch_size = total_chunks if batch_size_param == 0 else batch_size_param
            num_batches = (total_chunks + batch_size - 1) // batch_size

            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, total_chunks)
                batch_chunks = chunks[start_idx:end_idx]

                batch_progress = (25 + (batch_idx / num_batches) * 50) / 100.0
                progress_context.update_progress_percent(batch_progress * 100, f"Processing batch {batch_idx + 1}/{num_batches}")
                job_repo.update_job_progress(job_id, batch_progress)

                # Generate for this batch
                batch_result = dataset_generator.generate_dataset(
                    project_id=project_id,
                    owner_id=owner_id,
                    chunks=batch_chunks,
                    prompt_name=prompt_name,
                    format_type="raw_list",
                    concurrency=concurrency,
                    taxonomy_project=taxonomy_project,
                    taxonomy_name=taxonomy_name,
                    datasets_per_chunk=datasets_per_chunk,
                    only_validated=only_validated,
                    custom_prompt=custom_prompt,
                    generation_mode=generation_mode,
                    prefer_extracted=False
                )

                batch_valid_results = []
                if isinstance(batch_result, list):
                    batch_valid_results = [r for r in batch_result if r]
                    all_results.extend(batch_valid_results)

                if batch_valid_results:
                    file_manager = FileManager(storage_type='datasets')
                    ext = "json"
                    try:
                        from ...features.plugin.manager import plugin_manager
                        metadata = plugin_manager.get_format_metadata(format_type)
                        if metadata and 'file_extension' in metadata:
                            ext = metadata['file_extension']
                    except: pass
                    
                    batch_filename = f"dataset_{job_id}_batch_{batch_idx}.{ext}"
                    
                    serializable_results = []
                    for item in batch_valid_results:
                        if not isinstance(item, dict): continue
                        clean_item = {k: (v.decode('utf-8') if isinstance(v, bytes) else v) for k, v in item.items()}
                        serializable_results.append(clean_item)

                    batch_content = output_formatter.format_dataset(serializable_results, format_type, owner_id)
                    if isinstance(batch_content, str): batch_content = batch_content.encode('utf-8')
                    
                    batch_file_path = file_manager.store_file(
                        project_id=project_id,
                        document_id=None,
                        file_name=batch_filename,
                        file_content=batch_content, # type: ignore
                        owner_id=owner_id
                    )
                    batch_files.append({
                        "batch_index": batch_idx, "file_path": str(batch_file_path),
                        "entries_count": len(batch_valid_results), "chunks_processed": len(batch_chunks)
                    })

        # Wrap up
        total_entries_count = len(all_results)
        total_size = 0
        for batch in batch_files:
            try: total_size += os.path.getsize(batch["file_path"])
            except: pass

        # Register in dataset_versions
        dataset_version_id = str(uuid.uuid4())
        v_name = dataset_name or f"dataset_{job_id}"
        
        # Get next version
        v_str = "1.0.0"
        if enable_versioning:
            try:
                from ...storage.src.project.database_repositories import DatasetVersionRepository
                v_repo = DatasetVersionRepository(owner_id)
                versions = v_repo.get_versions_by_project(project_id)
                if versions:
                    last_v = versions[0][1] if isinstance(versions[0], tuple) else versions[0]['version']
                    ma, mi, pa = map(int, last_v.split('.'))
                    v_str = f"{ma}.{mi}.{pa + 1}"
            except: pass

        major, minor, patch = map(int, v_str.split('.'))
        
        cursor.execute("""
            INSERT INTO dataset_versions (
                id, owner_id, project_id, version, major_version, minor_version, patch_version,
                dataset_name, description, total_entries, file_path, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            dataset_version_id, owner_id, project_id, v_str, major, minor, patch,
            v_name, f"Generated dataset (Source: {data_source})", total_entries_count,
            json.dumps([b["file_path"] for b in batch_files]),
            json.dumps({"total_size": total_size, "completed_at": datetime.utcnow().isoformat()})
        ))
        db.commit()

        progress_context.update_progress_percent(100, "Dataset generation complete")
        job_repo.update_job_progress(job_id, 1.0)
        job_repo.update_job_status(job_id, 'completed')

        result_data = {
            "dataset_id": dataset_version_id,
            "entries_count": total_entries_count,
            "total_size": total_size,
            "batch_files": batch_files
        }
        job_repo.update_job_result(job_id, result_data)
        
        return {"status": "completed", "result": result_data, **parameters}

def _execute_chunk_documents(parameters: Dict[str, Any], progress_context, db_connection: Any) -> Dict[str, Any]:
    """Execute document chunking with parsed documents using sliding window or sequential processing for multi-file documents."""
    from ...core.settings import BackendSettings
    from ...features.chunk.cross_file_chunker import CrossFileChunker
    from ...features.chunk.engine import chunk_document

    project_id = parameters.get("project_id")
    owner_id = parameters.get("owner_id")
    if not isinstance(project_id, (str, int)) or not project_id:
        raise ValueError(f"Invalid project_id: {project_id}")

    document_ids = parameters.get("document_ids", [])
    chunker = parameters.get("chunker", "gemini")
    chunk_strategy = parameters.get("chunk_strategy", "character").strip().lower()

    if not owner_id:
        raise ValueError("owner_id must be provided for chunking job")

    # Use user-aware settings
    user_settings = BackendSettings(owner_id=owner_id)

    # Retrieve and set API keys for chunkers that need them
    needs_api_key = chunk_strategy in ["semantic", "llm_prompt"] or (chunk_strategy == "token" and chunker in ["gemini", "grok"])

    if needs_api_key:
        if chunker == "gemini":
            gemini_api_key = user_settings.get_setting("gemini_api_key")
            if not gemini_api_key:
                raise ValueError("Gemini API key not configured for user")
            os.environ['GOOGLE_API_KEY'] = gemini_api_key
        elif chunker == "grok":
            grok_api_key = user_settings.get_setting("grok_api_key")
            if not grok_api_key:
                raise ValueError("Grok API key not configured for user")
            os.environ['GROK_API_KEY'] = grok_api_key
        elif chunker == "openai":
            openai_api_key = user_settings.get_setting("openai_api_key")
            if not openai_api_key:
                raise ValueError("OpenAI API key not configured for user")
            os.environ['OPENAI_API_KEY'] = openai_api_key

    chunk_size = parameters.get("chunk_size", 512)
    overlap = parameters.get("overlap", 50)
    delimiters = parameters.get("delimiters", ["\n\n", "\n"])
    semantic_prompt = parameters.get("semantic_prompt", "")
    schema_json = parameters.get("schema_definition", "")
    if chunk_strategy == "schema":
        from ...features.chunk.schema import parse_chunking_schema

        parse_chunking_schema(schema_json)
    skip_parsing = parameters.get("skip_parsing", True)
    manifest_data = parameters.get("manifest_data")
    selected_files_map = parameters.get("selected_files")
    chunking_options = None
    
    # Propagate model from parameters if available
    model = parameters.get("model")
    
    if chunker == "ollama":
        if not model:
            model = user_settings.get_chunking_ollama_model()
        num_ctx = parameters.get("num_ctx") or user_settings.get_chunking_ollama_num_ctx()
        chunking_options = {"num_ctx": num_ctx}
    elif chunker == "gemini":
        if not model:
            model = user_settings.get_chunking_gemini_model()
    elif chunker == "grok":
        if not model:
            model = user_settings.get_chunking_grok_model()
    elif chunker == "openai":
        if not model:
            model = user_settings.get_chunking_openai_model()
    else:
        model = parameters.get("model", chunker)

    # Get database connection using context manager
    from ...storage.src.database import get_tenant_db_connection

    total_docs = len(document_ids)
    processed_docs = 0
    total_chunks_created = 0
    results = []

    for i, doc_id in enumerate(document_ids):
        try:
            progress_context.update_progress_percent((i / total_docs) * 100, f"Processing document {doc_id}")

            parsed_files = []
            file_name = ""
            
            # Fetch info with short-lived connection to avoid locking
            with get_tenant_db_connection(owner_id) as db:
                # Get document info, ensuring ownership
                cursor = db.cursor()
                cursor.execute("""
                    SELECT file_name FROM documents
                    WHERE id = ? AND project_id = ? AND owner_id = ?
                """, (doc_id, project_id, owner_id))

                doc_record = cursor.fetchone()
                if not doc_record:
                    logger.debug(f"DEBUG_CHUNK_EXEC: Document {doc_id} not found or access denied for owner {owner_id}")
                    results.append({
                        "document_id": doc_id,
                        "status": "failed",
                        "error": f"Document not found or access denied"
                    })
                    continue

                file_name = doc_record[0]
                logger.debug(f"DEBUG_CHUNK_EXEC: Processing document {doc_id} ({file_name})")

                # Query parsed_files table for this document, ensuring they belong to the correct parsed_document
                cursor.execute("""
                    SELECT pf.file_path FROM parsed_files pf
                    JOIN parsed_documents pd ON pf.parsed_document_id = pd.id
                    WHERE pd.document_id = ? AND pd.owner_id = ?
                    ORDER BY pf.page_number, pf.file_path
                """, (doc_id, owner_id))

                parsed_file_records = cursor.fetchall()
                parsed_files = [record[0] for record in parsed_file_records]
            
            # DB connection closed here, safe to start chunking which uses its own connection
            
            logger.debug(f"DEBUG_CHUNK_EXEC: Found {len(parsed_files)} parsed files for document {doc_id}")
            if parsed_files:
                for j, pf in enumerate(parsed_files[:5]):  # Log first 5
                    logger.debug(f"DEBUG_CHUNK_EXEC:   Parsed file {j}: {pf}")
                if len(parsed_files) > 5:
                    logger.debug(f"DEBUG_CHUNK_EXEC:   ... and {len(parsed_files) - 5} more")

            if parsed_files:
                # Multi-file document
                if selected_files_map:
                    doc_id_str = str(doc_id)
                    if doc_id_str in selected_files_map:
                        selected_paths = set(selected_files_map[doc_id_str])
                        parsed_files = [f for f in parsed_files if f in selected_paths]

                manifest_data = {
                    "parsed_files": parsed_files,
                    "total_pages": len(parsed_files),
                    "config": {"pages_per_split": 200, "overlap_pages": 1}
                }

                from ...features.chunk.cross_file_chunker import CrossFileChunker
                # Don't pass db connection, let it manage its own via repositories
                cross_file_chunker = CrossFileChunker(None)
                chunk_result = cross_file_chunker.chunk_document(
                    doc_id, project_id, manifest_data, chunk_strategy,
                    chunk_size, overlap, model, chunker, owner_id,
                    chunking_options, schema_json, delimiters
                )
                doc_chunks_created = chunk_result.get("chunks_created", 0) if chunk_result else 0
            else:
                # SINGLE-FILE DOCUMENT fallback
                results.append({
                    "document_id": doc_id,
                    "status": "failed",
                    "error": "No parsed files found for document"
                })
                continue

            total_chunks_created += doc_chunks_created
            processed_docs += 1
            results.append({
                "document_id": doc_id,
                "status": "completed",
                "chunks_created": doc_chunks_created,
                "chunk_strategy": chunk_strategy
            })

        except Exception as e:
            results.append({
                "document_id": doc_id,
                "status": "failed",
                "error": str(e)
            })

    if chunk_strategy == "schema" and processed_docs != total_docs:
        errors = "; ".join(f"{result['document_id']}: {result['error']}" for result in results if result["status"] == "failed")
        raise ValueError(f"Schema chunking completed for {processed_docs}/{total_docs} documents. Failed: {errors}")

    progress_context.update_progress_percent(100, f"Completed chunking {processed_docs}/{total_docs} documents")

    return {
        "status": "completed",
        "operation": "chunk_documents",
        "project_id": project_id,
        "total_documents": total_docs,
        "processed_documents": processed_docs,
        "total_chunks_created": total_chunks_created,
        "results": results
    }

def _execute_extraction_job(parameters: Dict[str, Any], progress_context, db_connection: Any) -> Dict[str, Any]:
    """Execute taxonomy extraction for current user."""
    from ...core.settings import BackendSettings
    from ...features.extraction.taxonomy_extractor import TaxonomyExtractor
    from ...storage.src.project.database_repositories import TaxonomyRepository, ChunkRepository, ExtractionJobRepository, ExtractionResultRepository
    from ...features.extraction.filesystem_storage import HybridStorageManager, FilesystemStorageManager, ExtractionFileManager

    taxonomy_id = parameters.get('taxonomy_id')
    owner_id = parameters.get('owner_id')
    project_id = parameters.get('project_id')
    
    if not owner_id:
         raise ValueError("owner_id must be provided for extraction job")

    user_settings = BackendSettings(owner_id=owner_id)

    grok_key = user_settings.get_setting("grok_api_key")
    if grok_key:
        os.environ['GROK_API_KEY'] = grok_key
    gemini_key = user_settings.get_setting("gemini_api_key")
    if gemini_key:
        os.environ['GEMINI_API_KEY'] = gemini_key
        os.environ['GOOGLE_API_KEY'] = gemini_key
    openai_key = user_settings.get_openai_api_key()
    if openai_key:
        os.environ['OPENAI_API_KEY'] = openai_key

    extractor = TaxonomyExtractor(
        taxonomy_repo=TaxonomyRepository(owner_id),
        chunk_repo=ChunkRepository(owner_id),
        file_manager=ExtractionFileManager(),
        grok_api_key=grok_key,
        gemini_api_key=gemini_key,
        openai_api_key=openai_key
    )

    progress_context.update_progress(message="Performing extraction")
    # This is a placeholder for the actual extraction logic
    return {"status": "completed", "operation": "extraction", "owner_id": owner_id}
