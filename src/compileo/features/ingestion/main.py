"""
Document ingestion module for job processing.
Provides parse_document function with multi-file parsing support for large PDFs.
"""

import os
import json
import uuid
import logging
from typing import Optional, Dict, Any, List, Tuple, Union
from datetime import datetime

logger = logging.getLogger(__name__)

def parse_document(
    file_path: str,
    parser: str = "gemini",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    document_id: Optional[Union[int, str]] = None,
    project_id: Optional[str] = None,
    owner_id: Optional[str] = None,
    pages_per_split: int = 200,
    overlap_pages: int = 0,
    page_selection: Optional[str] = None,
    **kwargs
) -> Optional[str]:
    """
    Parse a document and return the extracted text.
    For large PDFs, splits into chunks and parses each separately.

    Args:
        file_path: Path to the document file.
        parser: Parser to use for extraction.
        api_key: API key for cloud-based parsers.
        model: Specific model name to use.
        document_id: Document ID for database tracking.
        project_id: Project ID for organization.
        owner_id: User/owner ID for tenant isolation.
        pages_per_split: Number of pages per split chunk.
        overlap_pages: Number of overlapping pages between chunks.
        page_selection: Optional page selection expression
            (e.g., "1", "1-20", "1, 1-20, 30-120"). Null means all pages.
        **kwargs: Additional parser-specific arguments.

    Returns:
        Extracted text content, or None for PDFs (parsed files are saved separately).
    """
    try:
        logger.info(f"Parsing document: {file_path} with parser: {parser}")
        if page_selection:
            logger.info(f"Page selection specified: {page_selection}")

        # Try to determine owner_id and project_id from path if not provided
        if owner_id is None or project_id is None:
            path_parts = file_path.split(os.sep)
            if "storage" in path_parts:
                s_idx = path_parts.index("storage")
                if s_idx + 3 < len(path_parts) and path_parts[s_idx + 2] == "uploads":
                    if owner_id is None: owner_id = path_parts[s_idx + 1]
                    if project_id is None: project_id = path_parts[s_idx + 3]

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Document file not found: {file_path}")

        # Check if this is a PDF that needs splitting
        logger.debug(f"DEBUG_INGESTION: parse_document called for {os.path.basename(file_path)}, parser={parser}")
        if file_path.lower().endswith('.pdf'):
            logger.debug(f"DEBUG_INGESTION: PDF detected, calling _parse_pdf_with_splitting")
            return _parse_pdf_with_splitting(
                file_path=file_path,
                parser=parser,
                api_key=api_key,
                model=model,
                document_id=document_id,
                project_id=project_id,
                owner_id=owner_id,
                pages_per_split=pages_per_split,
                overlap_pages=overlap_pages,
                page_selection=page_selection,
                **kwargs
            )
        else:
            return _parse_single_file(
                file_path=file_path,
                parser=parser,
                api_key=api_key,
                model=model,
                document_id=document_id,
                project_id=project_id,
                owner_id=owner_id,
                **kwargs
            )

    except Exception as e:
        logger.error(f"Error parsing document {file_path}: {e}", exc_info=True)
        raise

def _parse_pdf_with_splitting(
    file_path: str,
    parser: str,
    api_key: Optional[str],
    model: Optional[str],
    document_id: Optional[Union[int, str]],
    project_id: Optional[str],
    owner_id: Optional[str],
    pages_per_split: int,
    overlap_pages: int,
    page_selection: Optional[str] = None,
    **kwargs
) -> Optional[str]:
    """Parse a PDF with splitting support."""
    from .processing.pdf_splitter import pre_split_pdf, extract_pages_to_pdf
    from .processing.page_selection import parse_page_selection

    logger.debug(f"DEBUG_INGESTION: _parse_pdf_with_splitting started for {os.path.basename(file_path)}")
    logger.info(f"Processing PDF with splitting: {file_path}")

    original_file_path = file_path
    original_total_pages = 0
    selected_pages = []

    # Handle page selection for PDFs
    if page_selection:
        try:
            import fitz
            doc = fitz.open(file_path)
            original_total_pages = doc.page_count
            doc.close()

            selected_pages = parse_page_selection(page_selection, original_total_pages)
            if selected_pages:
                logger.info(
                    f"Page selection applied: {page_selection} -> "
                    f"{len(selected_pages)} pages selected from {original_total_pages} total pages"
                )

                # Create a selected-pages PDF in the uploads directory
                # so that pre_split_pdf produces chunks and manifest in the
                # same location as the normal no-selection flow.
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                selected_pdf_path = os.path.join(
                    os.path.dirname(file_path), f"{base_name}_selected.pdf"
                )

                file_path = extract_pages_to_pdf(
                    source_path=original_file_path,
                    page_numbers=selected_pages,
                    output_path=selected_pdf_path
                )
                logger.info(f"Created selected-pages PDF: {file_path}")
            else:
                logger.warning(
                    f"Page selection '{page_selection}' resulted in no valid pages "
                    f"for document with {original_total_pages} pages. Processing all pages."
                )
        except Exception as e:
            logger.error(f"Failed to apply page selection '{page_selection}': {e}")
            logger.info("Falling back to processing all pages.")

    # Split the PDF if needed
    split_result = pre_split_pdf(file_path, pages_per_split, overlap_pages)
    split_files = split_result["split_files"]
    manifest_path = split_result["manifest_path"]

    # Inject page selection metadata into manifest if applicable
    if page_selection and selected_pages:
        try:
            if manifest_path and os.path.exists(manifest_path):
                with open(manifest_path, 'r', encoding='utf-8') as f:
                    manifest_data = json.load(f)
                manifest_data["page_selection_applied"] = True
                manifest_data["selected_pages"] = page_selection
                manifest_data["extracted_pages_count"] = len(selected_pages)
                manifest_data["original_total_pages"] = original_total_pages
                manifest_data["original_file"] = original_file_path
                with open(manifest_path, 'w', encoding='utf-8') as f:
                    json.dump(manifest_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to update manifest with page selection metadata: {e}")

    logger.info(f"PDF split into {len(split_files)} files, manifest: {manifest_path}")

    # Special handling for HuggingFace parser - load model once and reuse
    preloaded_model = None
    preloaded_processor = None
    logger.info(f"DEBUG_MODEL_REUSE: Checking parser='{parser}', split_files_count={len(split_files)}")
    if parser == "huggingface" and len(split_files) > 1:
        logger.info("DEBUG_MODEL_REUSE: HuggingFace parser detected with multiple chunks - loading model once for reuse")
        try:
            from .processing.huggingface_parser import load_model
            preloaded_model, preloaded_processor, _ = load_model()
            logger.info(f"DEBUG_MODEL_REUSE: Model loaded successfully for chunk reuse - model={type(preloaded_model)}, processor={type(preloaded_processor)}")
        except Exception as e:
            logger.warning(f"DEBUG_MODEL_REUSE: Failed to preload model for HuggingFace chunks: {e}")
            preloaded_model = None
            preloaded_processor = None
    else:
        logger.info(f"DEBUG_MODEL_REUSE: Skipping model preload - parser='{parser}', split_files={len(split_files)}")

    # Perform Two-Pass VLM Structure Analysis (The Skim)
    # Supported VLM parsers: Grok, Gemini, OpenAI, Ollama, HuggingFace
    style_guide = None
    vlm_parsers = ["grok", "gemini", "openai", "ollama", "huggingface"]

    if parser in vlm_parsers and split_files:
        try:
            # Pick the middle chunk to avoid index/glossary pages
            middle_index = len(split_files) // 2
            middle_chunk_path = split_files[middle_index]
            logger.info(f"Performing structure analysis on middle chunk ({parser}): {middle_chunk_path}")

            guide = ""
            if parser == "grok":
                from .processing.grok_parser import analyze_document_structure
                guide = analyze_document_structure(middle_chunk_path, api_key=api_key, model=model)
            elif parser == "gemini":
                from .processing.gemini_parser import analyze_document_structure
                guide = analyze_document_structure(middle_chunk_path, api_key=api_key, model=model)
            elif parser == "openai":
                from .processing.openai_parser import analyze_document_structure
                guide = analyze_document_structure(middle_chunk_path, api_key=api_key, model=model)
            elif parser == "ollama":
                from .processing.ollama_parser import analyze_document_structure
                guide = analyze_document_structure(middle_chunk_path, model=model)
            elif parser == "huggingface":
                from .processing.huggingface_parser import analyze_document_structure
                # Pass preloaded model/processor if available
                guide = analyze_document_structure(
                    middle_chunk_path,
                    api_key=api_key,
                    preloaded_model=preloaded_model,
                    preloaded_processor=preloaded_processor
                )

            if guide:
                style_guide = guide
                logger.info(f"Structure analysis successful. Guide length: {len(style_guide)} chars")
                logger.debug(f"Style Guide: {style_guide[:100]}...")
            else:
                logger.warning("Structure analysis returned empty guide")

        except Exception as e:
            logger.warning(f"Failed to perform structure analysis: {e}")

    # Parse each split file
    parsed_contents = []
    parsed_file_paths = []

    # Load manifest to get overlap information
    overlap_info = {}
    manifest_data = {}
    if manifest_path and os.path.exists(manifest_path):
        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest_data = json.load(f)
                # Create mapping of chunk number to overlap info
                for split in manifest_data.get("splits", []):
                    chunk_num = split["chunk_number"]
                    overlap_info[chunk_num] = {
                        "overlap_with_next": split.get("overlap", {}).get("with_next"),
                        "page_range": split.get("page_range", {})
                    }
        except Exception as e:
            logger.warning(f"Failed to load overlap info from manifest: {e}")

    # Store overlap content from previous chunks
    previous_overlap_content = {}

    for i, split_file in enumerate(split_files, 1):
        logger.info(f"Parsing split file {i}/{len(split_files)}: {split_file}")

        # Get chunk metadata
        chunk_metadata = overlap_info.get(i, {})

        # Get overlap content from previous chunk if it exists
        overlap_text = ""
        if i > 1:
            # Get overlap content that was stored from the previous chunk
            prev_chunk_num = i - 1
            overlap_text = previous_overlap_content.get(prev_chunk_num, "")

        # Parse the split file with the AI parser, including overlap context from previous chunk
        content = _parse_single_file(
            file_path=split_file,
            parser=parser,
            api_key=api_key,
            model=model,
            document_id=document_id,
            project_id=project_id,
            owner_id=owner_id,
            preloaded_model=preloaded_model,
            preloaded_processor=preloaded_processor,
            overlap_text=overlap_text,
            style_guide=style_guide,
            is_chunk=True,
            **kwargs
        )

        if content is not None:
            parsed_contents.append(content)

        # Extract and store overlap content for the NEXT chunk
        # Each split file contains: Y pages (overlap from previous) + (X-Y) pages (fresh content)
        # The first Y pages are overlap content that this chunk received from previous chunk
        # We need to extract the last Y pages of this chunk for the next chunk
        overlap_content_for_next = ""
        overlap_pages_count = manifest_data.get("overlap_pages", overlap_pages)  # Y

        if chunk_metadata.get("overlap_with_next") and overlap_pages_count > 0:
            from .processing.pypdf_parser import parse_pdf_with_pypdf
            # Extract the overlap portion (LAST Y pages) for the next chunk
            # Get total pages in this split file
            import fitz
            doc = fitz.open(split_file)
            total_pages = doc.page_count
            doc.close()

            start_page = total_pages - overlap_pages_count + 1
            overlap_content_for_next = parse_pdf_with_pypdf(split_file, start_page=start_page, end_page=None)

            # Store it for the next chunk
            previous_overlap_content[i] = overlap_content_for_next

        # Save parsed content - only include overlap when overlap_pages > 0
        received_overlap_content = overlap_text if overlap_text else ""
        include_overlap_in_file = overlap_pages > 0  # Only include overlap when explicitly requested

        if document_id is not None:
            parsed_file_path = _save_parsed_file(
                content=content,
                document_id=document_id,
                sequence=i,
                original_file_path=file_path,
                overlap_content=received_overlap_content if include_overlap_in_file else None,
                chunk_metadata=chunk_metadata,
                include_overlap=include_overlap_in_file,
                project_id=project_id,
                owner_id=owner_id
            )
            parsed_file_paths.append(parsed_file_path)

    # Combine all parsed content for backward compatibility
    combined_content = "\n\n".join(parsed_contents)

    # Save manifest alongside parsed files if we have a document_id
    if document_id is not None and manifest_path:
        _save_manifest_for_document(
            manifest_path=manifest_path,
            document_id=document_id,
            parsed_file_paths=parsed_file_paths,
            project_id=project_id,
            owner_id=owner_id
        )

    # Clean up the intermediate selected-pages PDF if one was created.
    # The chunk PDFs and manifest produced by pre_split_pdf are kept in the
    # uploads directory so they remain discoverable.
    if page_selection and selected_pages and original_file_path != file_path:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Cleaned up intermediate selected-pages PDF: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to clean up intermediate PDF {file_path}: {e}")

    logger.info(f"Completed multi-file parsing for document {document_id or 'unknown'}: {len(split_files)} files parsed")
    # Always return None for PDFs to prevent consolidated file creation
    return None

def _parse_single_file(
    file_path: str,
    parser: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    document_id: Optional[Union[int, str]] = None,
    project_id: Optional[str] = None,
    owner_id: Optional[str] = None,
    preloaded_model=None,
    preloaded_processor=None,
    overlap_text: Optional[str] = None,
    main_content_text: Optional[str] = None,
    is_chunk: bool = False,
    **kwargs
) -> Optional[str]:
    """
    Parse a single file using the specified parser.
    """
    logger.info(f"Parsing single file: {file_path} with parser: {parser}")

    # Handle structure analysis for single files (if not already provided by splitting logic)
    style_guide = kwargs.get("style_guide")
    if not style_guide and file_path.lower().endswith('.pdf'):
        vlm_parsers = ["grok", "gemini", "openai", "ollama", "huggingface"]
        if parser in vlm_parsers:
            try:
                logger.info(f"Performing structure analysis on single file ({parser}): {file_path}")
                guide = ""
                if parser == "grok":
                    from .processing.grok_parser import analyze_document_structure
                    guide = analyze_document_structure(file_path, api_key=api_key, model=model)
                elif parser == "gemini":
                    from .processing.gemini_parser import analyze_document_structure
                    guide = analyze_document_structure(file_path, api_key=api_key, model=model)
                elif parser == "openai":
                    from .processing.openai_parser import analyze_document_structure
                    guide = analyze_document_structure(file_path, api_key=api_key, model=model)
                elif parser == "ollama":
                    from .processing.ollama_parser import analyze_document_structure
                    guide = analyze_document_structure(file_path, model=model)
                elif parser == "huggingface":
                    from .processing.huggingface_parser import analyze_document_structure
                    guide = analyze_document_structure(
                        file_path,
                        api_key=api_key,
                        preloaded_model=preloaded_model,
                        preloaded_processor=preloaded_processor
                    )

                if guide:
                    style_guide = guide
                    kwargs["style_guide"] = style_guide # Update kwargs for consistency
                    logger.info(f"Structure analysis successful. Guide length: {len(style_guide)} chars")
                else:
                    logger.warning("Structure analysis returned empty guide")
            except Exception as e:
                logger.warning(f"Failed to perform structure analysis on single file: {e}")

    # Import parser functions
    logger.debug(f"DEBUG_INGESTION: _parse_single_file calling parser={parser} for {os.path.basename(file_path)}")
    if parser == "gemini":
        from .processing.gemini_parser import parse_document_with_gemini
        logger.debug(f"DEBUG_INGESTION: Calling parse_document_with_gemini")
        content = parse_document_with_gemini(file_path, api_key, model, overlap_text, style_guide=style_guide)
    elif parser == "grok":
        from .processing.grok_parser import parse_pdf_with_grok
        style_guide = kwargs.get("style_guide")
        content = parse_pdf_with_grok(file_path, api_key, model, overlap_text, style_guide=style_guide)
    elif parser == "openai":
        from .processing.openai_parser import parse_document_with_openai
        style_guide = kwargs.get("style_guide")
        content = parse_document_with_openai(file_path, api_key, model, overlap_text, style_guide=style_guide)
    elif parser == "ollama":
        from .processing.ollama_parser import parse_document_with_ollama
        style_guide = kwargs.get("style_guide")
        content = parse_document_with_ollama(file_path, model, None, overlap_text, style_guide=style_guide)
    elif parser == "huggingface":
        logger.info(f"DEBUG_HF_PARAMS: Calling parse_pdf_with_vlm with preloaded_model={preloaded_model is not None}, preloaded_processor={preloaded_processor is not None}")
        from .processing.huggingface_parser import parse_pdf_with_vlm
        style_guide = kwargs.get("style_guide")
        content = parse_pdf_with_vlm(file_path, api_key, preloaded_model, preloaded_processor, overlap_text, style_guide=style_guide)
    elif parser == "unstructured":
        from .processing.unstructured_parser import parse_document_with_unstructured
        content = parse_document_with_unstructured(file_path)
    elif parser == "pypdf":
        from .processing.pypdf_parser import parse_pdf_with_pypdf
        content = parse_pdf_with_pypdf(file_path)
    elif parser == "novlm":
        from .processing.novlm_parser import parse_pdf_novlm
        content = parse_pdf_novlm(file_path)
    else:
        # Fallback to basic text reading
        logger.warning(f"Unknown parser '{parser}', falling back to basic text reading")
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

    # Markdown formatting handled by:
    # - AI parsers: Pre-processing instructions in prompts
    # - Unstructured: Native markdown output via markdownify
    # - novlm: Delegates to unstructured or pypdf, both handle markdown appropriately
    # - pypdf: Direct text extraction (no post-processing needed)
    # No post-processing required - all parsers handle markdown through their respective methods

    # Create manifest for single file parsing if document_id is provided
    if document_id is not None and not is_chunk:
        _create_single_file_manifest(file_path, parser, document_id, content, project_id, owner_id)

    # Smart Fallback: If content is empty/None for non-PDF files, read as raw text
    if (not content or not content.strip()) and not file_path.lower().endswith('.pdf'):
        try:
            logger.info(f"Parser '{parser}' returned empty content for non-PDF. Falling back to raw text reading.")
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            logger.warning(f"Fallback raw text reading failed: {e}")

    return content

def _save_parsed_file(
    content: Optional[str],
    document_id: Union[int, str],
    sequence: int,
    original_file_path: str,
    overlap_content: Optional[str] = None,
    chunk_metadata: Optional[Dict[str, Any]] = None,
    include_overlap: bool = False,
    project_id: Optional[str] = None,
    owner_id: Optional[str] = None
) -> Optional[str]:
    """
    Save parsed content as JSON structure with main_content and overlap_content.
    Naming convention: {document_id}_{sequence}.json
    """
    # Determine project ID from original file path
    # Path format: storage/{owner_id}/uploads/{project_id}/{filename}
    if not project_id:
        path_parts = original_file_path.split(os.sep)
        if "storage" in path_parts and "uploads" in path_parts:
            storage_idx = path_parts.index("storage")
            if storage_idx + 3 < len(path_parts):
                project_id = path_parts[storage_idx + 3]  # uploads/{project_id}

    # Create parsed directory
    parsed_dir = f"storage/{owner_id}/parsed/{project_id}"
    os.makedirs(parsed_dir, exist_ok=True)

    # Create filename using original document name
    # Extract base name from original file path
    original_filename = os.path.basename(original_file_path)
    base_name = os.path.splitext(original_filename)[0]
    filename = f"{base_name}_{sequence}.json"
    file_path = os.path.join(parsed_dir, filename)

    # Create JSON structure - conditionally include overlap based on include_overlap flag
    json_structure = {
        "content_type": "main_content_only",  # Changed default content type
        "main_content": content or "",
        "metadata": {
            "chunk_number": sequence,
            "document_id": str(document_id),
            "processing_stage": "parsing",
            **(chunk_metadata or {})
        }
    }

    # Only include overlap_content if explicitly requested (for backward compatibility)
    if include_overlap and overlap_content:
        json_structure["overlap_content"] = overlap_content
        json_structure["content_type"] = "chunk_with_overlap"

    # Save as JSON
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(json_structure, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved parsed JSON file: {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"Failed to save JSON file {file_path}: {e}")
        return None

def _save_manifest_for_document(
    manifest_path: str,
    document_id: Union[int, str],
    parsed_file_paths: List[str],
    project_id: Optional[str] = None,
    owner_id: Optional[str] = None
) -> None:
    """Update and save document manifest, then update database."""
    try:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest_data = json.load(f)

        manifest_data["document_id"] = document_id
        manifest_data["parsed_files"] = parsed_file_paths
        manifest_data["project_id"] = project_id
        manifest_data["owner_id"] = owner_id

        if parsed_file_paths:
            parsed_dir = os.path.dirname(parsed_file_paths[0])
            manifest_filename = f"{document_id}_manifest.json"
            new_manifest_path = os.path.join(parsed_dir, manifest_filename)

            with open(new_manifest_path, 'w', encoding='utf-8') as f:
                json.dump(manifest_data, f, indent=2, ensure_ascii=False)

            _update_document_with_manifest(document_id, new_manifest_path, manifest_data, owner_id)
    except Exception as e:
        logger.error(f"Error saving manifest for document {document_id}: {e}")

def _create_single_file_manifest(
    file_path: str,
    parser: str,
    document_id: Union[int, str],
    content: Optional[str],
    project_id: Optional[str] = None,
    owner_id: Optional[str] = None
) -> None:
    """Create a manifest file for single file parsing operations."""
    parsed_dir = f"storage/{owner_id}/parsed/{project_id}"
    os.makedirs(parsed_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    parsed_filename = f"{base_name}_1.json"
    parsed_file_path = os.path.join(parsed_dir, parsed_filename)

    json_structure = {
        "main_content": content or "",
        "metadata": {"chunk_number": 1, "document_id": str(document_id)}
    }
    with open(parsed_file_path, 'w', encoding='utf-8') as f:
        json.dump(json_structure, f, indent=2, ensure_ascii=False)

    manifest_data = {
        "original_file": file_path,
        "total_pages": 1,
        "parser_used": parser,
        "document_id": document_id,
        "project_id": project_id,
        "owner_id": owner_id,
        "parsed_files": [parsed_file_path]
    }
    _update_document_with_single_file_manifest(document_id, manifest_data, owner_id)

def _update_document_with_manifest(document_id, manifest_path, manifest_data, owner_id):
    """Update database records with parsed document information."""
    from ...storage.src.project.database_repositories import ParsedDocumentRepository, ParsedFileRepository
    
    project_id = manifest_data.get("project_id")
    
    parsed_repo = ParsedDocumentRepository(owner_id)
    parsed_doc_id = parsed_repo.create_document({
        "id": str(uuid.uuid4()),
        "owner_id": owner_id,
        "project_id": project_id,
        "document_id": document_id,
        "filename": os.path.basename(manifest_data.get("original_file", "")),
        "content": "",
        "metadata": {"manifest_path": manifest_path},
        "parser_used": manifest_data.get("parser_used", "multi-file"),
        "parse_config": "{}",
        "total_pages": manifest_data.get("total_pages", 0),
        "parsing_time": 0.0,
        "created_at": datetime.utcnow()
    })

    parsed_file_repo = ParsedFileRepository(owner_id)
    for i, path in enumerate(manifest_data.get("parsed_files", []), 1):
        parsed_file_repo.create_file({
            "id": str(uuid.uuid4()),
            "owner_id": owner_id,
            "parsed_document_id": parsed_doc_id,
            "filename": os.path.basename(path),
            "file_path": path,
            "file_type": "json",
            "page_number": i,
            "content_length": 0,
            "created_at": datetime.utcnow()
        })

def _update_document_with_single_file_manifest(document_id, manifest_data, owner_id):
    """Update database records for single file parsing."""
    _update_document_with_manifest(document_id, "single_file_manifest", manifest_data, owner_id)
    # Update status to parsed in documents table
    from ...storage.src.database import get_tenant_db_connection
    with get_tenant_db_connection(owner_id) as db:
        cursor = db.cursor()
        cursor.execute("UPDATE documents SET status = 'parsed' WHERE id = ? AND owner_id = ?", (document_id, owner_id))
        db.commit()
