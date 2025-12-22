"""
CLI Command Handlers.
Contains all the implementation logic for CLI commands.
"""

import click
import os
import json
from datetime import datetime
from typing import List, Dict, Any


# Document command handlers
def handle_upload_documents(file_paths, project_id):
    """Handle document upload command."""
    try:
        from .features.gui.services.api_client import api_client

        click.echo(f"📤 Uploading {len(file_paths)} documents to project {project_id}...")

        # Validate all files exist and are supported types
        allowed_extensions = {'.pdf', '.docx', '.txt', '.md', '.csv', '.json', '.xml'}
        validated_files = []

        for file_path in file_paths:
            if not os.path.exists(file_path):
                raise click.BadParameter(f"File does not exist: {file_path}")

            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext not in allowed_extensions:
                raise click.BadParameter(f"Unsupported file type: {file_ext}. Allowed: {', '.join(allowed_extensions)}")

            validated_files.append(file_path)

        # Upload documents
        upload_response = api_client.upload_documents(project_id, validated_files)

        if upload_response and "job_id" in upload_response:
            job_id = upload_response["job_id"]
            files_count = upload_response.get("files_count", len(validated_files))
            click.echo(f"✅ Documents uploaded successfully. Job ID: {job_id}")
            click.echo(f"📊 Files uploaded: {files_count}")

            # Show status
            click.echo("⏳ Processing upload...")
            import time
            max_wait = 60
            start_time = time.time()

            while time.time() - start_time < max_wait:
                status_response = api_client.get(f"/api/v1/documents/upload/{job_id}/status")
                if status_response and status_response.get("status") == "completed":
                    processed_files = status_response.get("processed_files", [])
                    click.echo(f"✅ Upload completed. {len(processed_files)} documents processed.")
                    for doc in processed_files[:5]:  # Show first 5
                        click.echo(f"  - {doc.get('file_name', 'Unknown')} (ID: {doc.get('id', 'N/A')})")
                    if len(processed_files) > 5:
                        click.echo(f"  ... and {len(processed_files) - 5} more")
                    return
                time.sleep(2)

            click.echo("⚠️ Upload is still processing. Check status later with: compileo documents status --job-id {job_id}")
        else:
            raise click.ClickException("Failed to upload documents")

    except Exception as e:
        click.echo(f"❌ Error uploading documents: {e}", err=True)
        raise click.Abort()


def handle_process_documents(project_id, document_ids, parser, chunk_strategy, chunk_size, overlap, semantic_prompt, schema_definition, character_chunk_size, character_overlap, skip_parsing):
    """Handle document processing command."""
    try:
        from .features.gui.services.api_client import api_client

        # Parse document IDs
        try:
            doc_ids = [int(x.strip()) for x in document_ids.split(',')]
        except ValueError:
            raise click.BadParameter("--document-ids must be a comma-separated list of integers")

        click.echo(f"⚙️ Processing {len(doc_ids)} documents in project {project_id}")
        click.echo(f"🔍 Parser: {parser}")
        click.echo(f"✂️ Chunk Strategy: {chunk_strategy}")

        # Prepare request data
        request_data = {
            "project_id": project_id,
            "document_ids": doc_ids,
            "parser": parser,
            "chunk_strategy": chunk_strategy,
            "chunk_size": chunk_size,
            "overlap": overlap
        }

        # Add strategy-specific parameters
        if chunk_strategy == "semantic" and semantic_prompt:
            request_data["semantic_prompt"] = semantic_prompt
        elif chunk_strategy == "schema" and schema_definition:
            request_data["schema_definition"] = schema_definition
        elif chunk_strategy == "character":
            if character_chunk_size is not None:
                request_data["character_chunk_size"] = character_chunk_size
            if character_overlap is not None:
                request_data["character_overlap"] = character_overlap

        # Process documents
        process_response = api_client.post("/api/v1/documents/process", data=request_data)

        if process_response and "job_id" in process_response:
            job_id = process_response["job_id"]
            click.echo(f"✅ Processing started. Job ID: {job_id}")

            # Wait for completion
            click.echo("⏳ Waiting for processing completion...")
            import time
            max_wait = 120  # 2 minutes for processing
            start_time = time.time()

            while time.time() - start_time < max_wait:
                status_response = api_client.get(f"/api/v1/documents/process/{job_id}/status")
                if status_response:
                    status = status_response.get("status")
                    if status == "completed":
                        result = status_response.get("result", {})
                        processed_docs = result.get("processed_documents", 0)
                        total_chunks = result.get("total_chunks", 0)
                        click.echo(f"✅ Processing completed successfully!")
                        click.echo(f"📊 Results: {processed_docs} documents processed, {total_chunks} chunks created")
                        return
                    elif status == "failed":
                        error = status_response.get("error", "Unknown error")
                        raise click.ClickException(f"Processing failed: {error}")

                time.sleep(3)

            click.echo("⚠️ Processing is still running. Check status later with: compileo documents status --job-id {job_id}")
        else:
            raise click.ClickException("Failed to start document processing")

    except Exception as e:
        click.echo(f"❌ Error processing documents: {e}", err=True)
        raise click.Abort()


def handle_list_documents(project_id, output_format):
    """Handle document listing command."""
    try:
        from .features.gui.services.api_client import api_client

        params = {}
        if project_id:
            params["project_id"] = project_id

        response = api_client.get("/api/v1/documents/", params=params)
        documents = response.get("documents", [])
        total = response.get("total", len(documents))

        if not documents:
            click.echo("No documents found.")
            return

        if output_format == "json":
            import json
            click.echo(json.dumps({"documents": documents, "total": total}, indent=2, default=str))
        else:
            # Table format
            click.echo(f"Documents (Total: {total}):")
            click.echo("-" * 100)
            click.echo(f"{'ID':<5} {'Project':<8} {'Name':<40} {'Status':<10} {'Created'}")
            click.echo("-" * 100)
            for doc in documents:
                name = doc.get('file_name', 'Unknown')[:39]
                created = doc.get('created_at', 'Unknown')
                if created and len(created) > 10:
                    created = created[:10]
                click.echo(f"{doc.get('id', 0):<5} {doc.get('project_id', 0):<8} {name:<40} {doc.get('status', 'unknown'):<10} {created}")

    except Exception as e:
        click.echo(f"❌ Error listing documents: {e}", err=True)
        raise click.Abort()


def handle_delete_document(document_id, confirm):
    """Handle document deletion command."""
    try:
        from .features.gui.services.api_client import api_client

        if not confirm:
            if not click.confirm(f"Are you sure you want to delete document {document_id}? This will also delete all associated chunks."):
                return

        response = api_client.delete(f"/api/v1/documents/{document_id}")

        if response and "message" in response:
            click.echo(f"✅ {response['message']}")
        else:
            click.echo("✅ Document deleted successfully")

    except Exception as e:
        click.echo(f"❌ Error deleting document: {e}", err=True)
        raise click.Abort()


def handle_parse_documents(project_id, document_ids, parser):
    """Handle document parsing command."""
    try:
        from .features.gui.services.api_client import api_client

        # Parse document IDs
        try:
            doc_ids = [int(x.strip()) for x in document_ids.split(',')]
        except ValueError:
            raise click.BadParameter("--document-ids must be a comma-separated list of integers")

        click.echo(f"📄 Parsing {len(doc_ids)} documents in project {project_id} with {parser}")
        click.echo(f"Parser: {parser}")

        # Prepare request data for parsing only (no chunking)
        request_data = {
            "project_id": project_id,
            "document_ids": doc_ids,
            "parser": parser,
            "skip_parsing": False  # Explicitly enable parsing
        }

        # Parse documents using /process endpoint
        parse_response = api_client.post("/api/v1/documents/process", data=request_data)

        if parse_response and "job_id" in parse_response:
            job_id = parse_response["job_id"]
            click.echo(f"✅ Parsing started. Job ID: {job_id}")

            # Wait for completion
            click.echo("⏳ Waiting for parsing completion...")
            import time
            max_wait = 120  # 2 minutes for parsing
            start_time = time.time()

            while time.time() - start_time < max_wait:
                status_response = api_client.get(f"/api/v1/documents/process/{job_id}/status")
                if status_response:
                    status = status_response.get("status")
                    if status == "completed":
                        result = status_response.get("result", {})
                        processed_docs = result.get("processed_documents", 0)
                        click.echo(f"✅ Parsing completed successfully!")
                        click.echo(f"📊 Results: {processed_docs} documents parsed to markdown")
                        return
                    elif status == "failed":
                        error = status_response.get("error", "Unknown error")
                        raise click.ClickException(f"Parsing failed: {error}")

                time.sleep(3)

            click.echo("⚠️ Parsing is still running. Check status later with: compileo documents status --job-id {job_id} --type process")
        else:
            raise click.ClickException("Failed to start document parsing")

    except Exception as e:
        click.echo(f"❌ Error parsing documents: {e}", err=True)
        raise click.Abort()


def handle_chunk_documents(project_id, document_ids, chunk_strategy, chunk_size, overlap, chunker, semantic_prompt, schema_definition, character_chunk_size, character_overlap, num_ctx, system_instruction, sliding_window):
    """Handle document chunking command."""
    try:
        from .features.gui.services.api_client import api_client

        # Parse document IDs
        try:
            doc_ids = [int(x.strip()) for x in document_ids.split(',')]
        except ValueError:
            raise click.BadParameter("--document-ids must be a comma-separated list of integers")

        click.echo(f"✂️ Chunking {len(doc_ids)} documents in project {project_id}")
        click.echo(f"Strategy: {chunk_strategy}, Model: {chunker}")

        # Prepare request data
        request_data = {
            "project_id": project_id,
            "document_ids": doc_ids,
            "chunk_strategy": chunk_strategy,
            "chunk_size": chunk_size,
            "overlap": overlap,
            "chunker": chunker,
            "skip_parsing": True,  # Only chunk, don't parse
            "num_ctx": num_ctx,
            "system_instruction": system_instruction,
            "sliding_window": sliding_window
        }

        # Add strategy-specific parameters
        if chunk_strategy == "semantic" and semantic_prompt:
            request_data["semantic_prompt"] = semantic_prompt
        elif chunk_strategy == "schema" and schema_definition:
            request_data["schema_definition"] = schema_definition
        elif chunk_strategy == "character":
            if character_chunk_size is not None:
                request_data["character_chunk_size"] = character_chunk_size
            if character_overlap is not None:
                request_data["character_overlap"] = character_overlap

        # Process documents
        process_response = api_client.post("/api/v1/documents/process", data=request_data)

        if process_response and "job_id" in process_response:
            job_id = process_response["job_id"]
            click.echo(f"✅ Chunking started. Job ID: {job_id}")

            # Wait for completion
            click.echo("⏳ Waiting for chunking completion...")
            import time
            max_wait = 120  # 2 minutes for chunking
            start_time = time.time()

            while time.time() - start_time < max_wait:
                status_response = api_client.get(f"/api/v1/documents/process/{job_id}/status")
                if status_response:
                    status = status_response.get("status")
                    if status == "completed":
                        result = status_response.get("result", {})
                        processed_docs = result.get("processed_documents", 0)
                        total_chunks = result.get("total_chunks", 0)
                        click.echo(f"✅ Chunking completed successfully!")
                        click.echo(f"📊 Results: {processed_docs} documents processed, {total_chunks} chunks created")
                        return
                    elif status == "failed":
                        error = status_response.get("error", "Unknown error")
                        raise click.ClickException(f"Chunking failed: {error}")

                time.sleep(3)

            click.echo("⚠️ Chunking is still running. Check status later with: compileo documents status --job-id {job_id} --type process")
        else:
            raise click.ClickException("Failed to start document chunking")

    except Exception as e:
        click.echo(f"❌ Error chunking documents: {e}", err=True)
        raise click.Abort()


def handle_view_document_content(document_id, page, page_size, output_file):
    """Handle document content viewing command."""
    try:
        from .features.gui.services.api_client import api_client

        # Fetch document content
        params = {"page": page, "page_size": page_size}
        response = api_client.get(f"/api/v1/documents/{document_id}/content", params=params)

        if response and "content" in response:
            content = response["content"]
            total_pages = response.get("total_pages", 1)
            total_length = response.get("total_length", 0)
            word_count = response.get("word_count", 0)
            line_count = response.get("line_count", 0)

            if output_file:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                click.echo(f"✅ Content saved to {output_file}")
            else:
                # Display content info
                click.echo(f"Document {document_id} - Page {page} of {total_pages}")
                click.echo(f"Total: {total_length:,} characters, {word_count:,} words, {line_count:,} lines")
                click.echo("-" * 80)

                # Display content
                click.echo(content)

                # Show pagination info
                if total_pages > 1:
                    click.echo("-" * 80)
                    click.echo(f"Page {page} of {total_pages}")
                    if page > 1:
                        click.echo(f"Previous: compileo documents content {document_id} --page {page-1}")
                    if page < total_pages:
                        click.echo(f"Next: compileo documents content {document_id} --page {page+1}")
        else:
            click.echo("❌ Failed to retrieve document content")

    except Exception as e:
        click.echo(f"❌ Error viewing document content: {e}", err=True)
        raise click.Abort()


def handle_split_pdf(pdf_path, pages_per_split, overlap_pages):
    """Handle PDF splitting command."""
    try:
        from .features.gui.services.api_client import api_client

        click.echo(f"📄 Splitting PDF: {pdf_path}")
        click.echo(f"Pages per split: {pages_per_split}")
        click.echo(f"Overlap pages: {overlap_pages}")

        request_data = {
            "pdf_path": pdf_path,
            "pages_per_split": pages_per_split,
            "overlap_pages": overlap_pages
        }

        response = api_client.post("/api/v1/documents/split-pdf", data=request_data)

        if response and "split_files" in response:
            split_files = response["split_files"]
            total_splits = response.get("total_splits", len(split_files))

            click.echo(f"✅ PDF split successfully into {total_splits} files:")
            for i, split_file in enumerate(split_files, 1):
                click.echo(f"  {i}. {split_file}")
        else:
            click.echo("❌ Failed to split PDF")

    except Exception as e:
        click.echo(f"❌ Error splitting PDF: {e}", err=True)
        raise click.Abort()


def handle_check_job_status(job_id, job_type):
    """Handle job status checking command."""
    try:
        from .features.gui.services.api_client import api_client

        endpoint = f"/api/v1/documents/{job_type}/{job_id}/status"
        response = api_client.get(endpoint)

        if response:
            status = response.get("status", "unknown")
            progress = response.get("progress", 0)
            current_step = response.get("current_step", "Unknown")

            click.echo(f"Job Status: {status.upper()}")
            click.echo(f"Progress: {progress}%")
            click.echo(f"Current Step: {current_step}")

            if status == "completed":
                if job_type == "upload":
                    processed_files = response.get("processed_files", [])
                    click.echo(f"Files Processed: {len(processed_files)}")
                elif job_type in ["process", "parse"]:
                    if job_type == "parse":
                        parsed_docs = response.get("parsed_documents", 0)
                        click.echo(f"Documents Parsed: {parsed_docs}")
                    else:  # process
                        result = response.get("result", {})
                        processed_docs = result.get("processed_documents", 0)
                        total_chunks = result.get("total_chunks", 0)
                        click.echo(f"Documents Processed: {processed_docs}")
                        click.echo(f"Total Chunks Created: {total_chunks}")
        else:
            click.echo("❌ Failed to get job status")

    except Exception as e:
        click.echo(f"❌ Error checking job status: {e}", err=True)
        raise click.Abort()


def handle_process_single_file(file_path, project_id, parser, chunk_strategy, chunk_size, overlap, semantic_prompt, schema_definition, character_chunk_size, character_overlap, skip_parsing=False):
    """Handle single file processing command."""
    try:
        import os
        from .features.gui.services.api_client import api_client

        # Validate file exists
        if not os.path.exists(file_path):
            raise click.BadParameter(f"File does not exist: {file_path}")

        # Validate file type
        allowed_extensions = {'.pdf', '.docx', '.txt', '.md', '.csv', '.json', '.xml'}
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext not in allowed_extensions:
            raise click.BadParameter(f"Unsupported file type: {file_ext}. Allowed: {', '.join(allowed_extensions)}")

        click.echo(f"📄 Processing file: {file_path}")
        click.echo(f"📁 Project ID: {project_id}")
        click.echo(f"🔍 Parser: {parser}")
        click.echo(f"✂️ Chunk Strategy: {chunk_strategy}")

        # Prepare request data
        request_data = {
            "project_id": project_id,
            "parser": parser,
            "chunk_strategy": chunk_strategy,
            "chunk_size": chunk_size,
            "overlap": overlap
        }

        # Add strategy-specific parameters
        if chunk_strategy == "semantic" and semantic_prompt:
            request_data["semantic_prompt"] = semantic_prompt
        elif chunk_strategy == "schema" and schema_definition:
            request_data["schema_definition"] = schema_definition
        elif chunk_strategy == "character":
            if character_chunk_size is not None:
                request_data["character_chunk_size"] = character_chunk_size
            if character_overlap is not None:
                request_data["character_overlap"] = character_overlap

        # Add skip_parsing parameter
        request_data["skip_parsing"] = skip_parsing

        # Upload the file first
        click.echo("📤 Uploading document...")
        upload_response = api_client.upload_documents(project_id, [file_path])

        if not upload_response or "job_id" not in upload_response:
            raise click.ClickException("Failed to upload document")

        upload_job_id = upload_response["job_id"]
        click.echo(f"✅ Document uploaded successfully. Job ID: {upload_job_id}")

        # Wait for upload completion
        click.echo("⏳ Waiting for upload completion...")
        import time
        max_wait = 60  # 60 seconds
        start_time = time.time()

        while time.time() - start_time < max_wait:
            status_response = api_client.get(f"/api/v1/documents/upload/{upload_job_id}/status")
            if status_response and status_response.get("status") == "completed":
                processed_files = status_response.get("processed_files", [])
                if processed_files:
                    document_id = processed_files[0]["id"]
                    click.echo(f"✅ Upload completed. Document ID: {document_id}")
                    break
            time.sleep(2)
        else:
            raise click.ClickException("Upload timeout")

        # Process the document
        click.echo("⚙️ Processing document...")
        request_data["document_ids"] = [document_id]

        process_response = api_client.post("/api/v1/documents/process", data=request_data)

        if process_response and "job_id" in process_response:
            process_job_id = process_response["job_id"]
            click.echo(f"✅ Processing started. Job ID: {process_job_id}")

            # Wait for processing completion
            click.echo("⏳ Waiting for processing completion...")
            start_time = time.time()

            while time.time() - start_time < max_wait:
                status_response = api_client.get(f"/api/v1/documents/process/{process_job_id}/status")
                if status_response and status_response.get("status") == "completed":
                    result = status_response.get("result", {})
                    processed_docs = result.get("processed_documents", 0)
                    total_chunks = result.get("total_chunks", 0)
                    click.echo(f"✅ Processing completed successfully!")
                    click.echo(f"📊 Results: {processed_docs} documents processed, {total_chunks} chunks created")
                    return
                elif status_response and status_response.get("status") == "failed":
                    error = status_response.get("error", "Unknown error")
                    raise click.ClickException(f"Processing failed: {error}")
                time.sleep(2)

            raise click.ClickException("Processing timeout")
        else:
            raise click.ClickException("Failed to start document processing")

    except Exception as e:
        click.echo(f"❌ Error processing document: {e}", err=True)
        raise click.Abort()


# Dataset generation handler
def handle_generate_dataset(project_id, prompt_name, format_type, concurrency, batch_size, include_evaluation_sets, taxonomy_project, taxonomy_name, data_source, extraction_file_id, output_dir, analyze_quality, quality_threshold, quality_config, enable_versioning, dataset_name, run_benchmarks, benchmark_suite, benchmark_config, category_limits, specificity_level, custom_audience, custom_purpose, complexity_level, domain, datasets_per_chunk, selected_categories):
    """Handle dataset generation command."""
    try:
        from .features.gui.services.api_client import api_client

        click.echo(f"🤖 Starting dataset generation for project {project_id}")
        click.echo(f"📊 Batch size: {batch_size} chunks, Workers: {concurrency}")
        click.echo(f"📄 Format: {format_type}, Prompt: {prompt_name}")

        # Prepare request data
        request_data = {
            "project_id": project_id,
            "prompt_name": prompt_name,
            "format_type": format_type,
            "concurrency": concurrency,
            "batch_size": batch_size,
            "data_source": data_source,
            "analyze_quality": analyze_quality,
            "quality_threshold": quality_threshold,
            "datasets_per_chunk": datasets_per_chunk
        }

        # Add optional parameters
        if taxonomy_project and taxonomy_name:
            request_data["taxonomy_project"] = taxonomy_project
            request_data["taxonomy_name"] = taxonomy_name

        if extraction_file_id:
            request_data["extraction_file_id"] = extraction_file_id

        if selected_categories:
            # Parse comma-separated list
            categories = [cat.strip() for cat in selected_categories.split(',')]
            request_data["selected_categories"] = categories

        if custom_audience:
            request_data["custom_audience"] = custom_audience
        if custom_purpose:
            request_data["custom_purpose"] = custom_purpose
        if complexity_level:
            request_data["complexity_level"] = complexity_level
        if domain:
            request_data["domain"] = domain

        # Start dataset generation
        response = api_client.post("/api/v1/datasets/generate", request_data)

        if response and "job_id" in response:
            job_id = response["job_id"]
            click.echo(f"✅ Dataset generation started successfully!")
            click.echo(f"📋 Job ID: {job_id}")
            click.echo(f"🔍 Monitor progress: compileo jobs poll {job_id}")
            click.echo(f"📊 Check status: compileo jobs status {job_id}")
        else:
            click.echo("❌ Failed to start dataset generation")
            raise click.Abort()

    except Exception as e:
        click.echo(f"❌ Error starting dataset generation: {e}", err=True)
        raise click.Abort()


# Quality analysis handler
def handle_analyze_quality(dataset_file, quality_config, threshold, output_format, output_file):
    """Handle quality analysis command."""
    try:
        import json
        from pathlib import Path

        # Import quality analysis
        try:
            from .features.datasetqual import QualityAnalyzer, QualityConfig, DEFAULT_CONFIG, QualityReporter
        except ImportError:
            click.echo("Error: datasetqual module not available. Please install the quality analysis module.", err=True)
            raise click.Abort()

        # Load dataset
        dataset_path = Path(dataset_file)
        if dataset_path.suffix.lower() == '.jsonl':
            dataset = []
            with open(dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        dataset.append(json.loads(line))
        elif dataset_path.suffix.lower() == '.json':
            with open(dataset_path, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
            if not isinstance(dataset, list):
                dataset = [dataset]
        else:
            click.echo(f"Unsupported file format: {dataset_path.suffix}", err=True)
            raise click.Abort()

        click.echo(f"Loaded dataset with {len(dataset)} items from {dataset_file}")

        # Load configuration
        if quality_config:
            with open(quality_config, 'r') as f:
                config_dict = json.load(f)
            config = QualityConfig(**config_dict)
        else:
            config = DEFAULT_CONFIG
            config.enabled = True  # Enable for standalone analysis

        # Run analysis
        analyzer = QualityAnalyzer(config)
        results = analyzer.analyze_dataset(dataset)

        # Format and display results
        if output_file:
            QualityReporter.save_report(results, output_file, output_format)
            click.echo(f"Quality report saved to {output_file}")
        else:
            report = QualityReporter.get_report(results, output_format)
            click.echo(report)

        # Check threshold
        summary = results.get('summary', {})
        overall_score = summary.get('overall_score')
        if isinstance(overall_score, (int, float)) and overall_score < threshold:
            click.echo(f"Warning: Quality score {overall_score} below threshold {threshold}", err=True)
            raise click.Abort()

    except Exception as e:
        click.echo(f"Error analyzing quality: {e}", err=True)
        raise click.Abort()


# Taxonomy handlers
def handle_list_taxonomies(project_id, output_format):
    """Handle taxonomy listing command."""
    try:
        from .features.gui.services.api_client import api_client

        params = {}
        if project_id:
            params["project_id"] = project_id

        response = api_client.get("/api/v1/taxonomy/", params=params)
        taxonomies = response.get("taxonomies", [])

        if not taxonomies:
            click.echo("No taxonomies found.")
            return

        if output_format == "json":
            import json
            click.echo(json.dumps(taxonomies, indent=2, default=str))
        else:
            # Table format
            click.echo(f"{'ID':<5} {'Name':<30} {'Categories':<10} {'Confidence':<12} {'Created'}")
            click.echo("-" * 80)
            for tax in taxonomies:
                name = tax.get('name', 'Unknown')[:29]
                created = tax.get('created_at', 'Unknown')
                if created and len(created) > 10:
                    created = created[:10]
                click.echo(f"{tax.get('id', 0):<5} {name:<30} {tax.get('categories_count', 0):<10} {tax.get('confidence_score', 0.0):<12.2f} {created}")

    except Exception as e:
        click.echo(f"Error listing taxonomies: {e}", err=True)
        raise click.Abort()


def handle_create_taxonomy(project_id, name, description, taxonomy_file):
    """Handle taxonomy creation command."""
    try:
        from .features.gui.services.api_client import api_client
        import json

        # Load taxonomy structure from file or create basic structure
        if taxonomy_file:
            with open(taxonomy_file, 'r') as f:
                taxonomy_data = json.load(f)
        else:
            taxonomy_data = {
                "name": name,
                "description": description or "",
                "children": []
            }

        request_data = {
            "name": name,
            "description": description or "",
            "project_id": project_id,
            "taxonomy": taxonomy_data
        }

        response = api_client.post("/api/v1/taxonomy/", data=request_data)
        click.echo(f"Taxonomy '{name}' created successfully with ID: {response.get('id')}")

    except Exception as e:
        click.echo(f"Error creating taxonomy: {e}", err=True)
        raise click.Abort()


def handle_generate_taxonomy(project_id, name, documents, depth, generator, domain, batch_size, category_limits, specificity_level, processing_mode):
    """Handle taxonomy generation command."""
    try:
        from .features.gui.services.api_client import api_client

        # Parse document IDs
        try:
            document_ids = [int(x.strip()) for x in documents.split(',')]
        except ValueError:
            raise click.BadParameter("--documents must be a comma-separated list of integers")

        # Parse category limits if provided
        parsed_limits = None
        if category_limits:
            try:
                parsed_limits = [int(x.strip()) for x in category_limits.split(',')]
            except ValueError:
                raise click.BadParameter("--category-limits must be a comma-separated list of integers")

        request_data = {
            "project_id": project_id,
            "name": name,
            "documents": document_ids,
            "depth": depth,
            "generator": generator,
            "domain": domain,
            "batch_size": batch_size,
            "specificity_level": specificity_level,
            "processing_mode": processing_mode
        }

        if parsed_limits:
            request_data["category_limits"] = parsed_limits

        response = api_client.post("/api/v1/taxonomy/generate", data=request_data)
        click.echo(f"Taxonomy '{name}' generation started in {processing_mode} mode. Check status with taxonomy load command.")

    except Exception as e:
        click.echo(f"Error generating taxonomy: {e}", err=True)
        raise click.Abort()


def handle_extend_taxonomy(taxonomy_data, project_id, additional_depth, generator, domain, batch_size, documents, processing_mode):
    """Handle taxonomy extension command."""
    try:
        from .features.gui.services.api_client import api_client
        import json

        if not taxonomy_data:
            raise click.BadParameter("--taxonomy-data is required for extension")

        # Load taxonomy data from file
        with open(taxonomy_data, 'r') as f:
            taxonomy_structure = json.load(f)

        request_data = {
            "taxonomy_data": taxonomy_structure,
            "project_id": project_id,
            "additional_depth": additional_depth,
            "generator": generator,
            "domain": domain,
            "batch_size": batch_size,
            "processing_mode": processing_mode
        }

        if documents:
            try:
                document_ids = [int(x.strip()) for x in documents.split(',')]
                request_data["documents"] = document_ids
            except ValueError:
                raise click.BadParameter("--documents must be a comma-separated list of integers")

        response = api_client.post("/api/v1/taxonomy/extend", data=request_data)
        click.echo(f"Taxonomy extension completed in {processing_mode} mode. Taxonomy updated in place with ID: {response.get('id')}")

    except Exception as e:
        click.echo(f"Error extending taxonomy: {e}", err=True)
        raise click.Abort()


def handle_load_taxonomy(taxonomy_id, output_format, output_file):
    """Handle taxonomy loading command."""
    try:
        from .features.gui.services.api_client import api_client
        import json

        response = api_client.get(f"/api/v1/taxonomy/{taxonomy_id}")

        if output_file:
            with open(output_file, 'w') as f:
                json.dump(response, f, indent=2)
            click.echo(f"Taxonomy saved to {output_file}")
        elif output_format == "json":
            click.echo(json.dumps(response, indent=2, default=str))
        else:
            # Text format
            taxonomy = response.get('taxonomy', {})
            click.echo(f"Taxonomy: {taxonomy.get('name', 'Unknown')}")
            click.echo(f"Description: {taxonomy.get('description', 'None')}")
            click.echo(f"Categories: {len(taxonomy.get('children', []))}")

    except Exception as e:
        click.echo(f"Error loading taxonomy: {e}", err=True)
        raise click.Abort()


def handle_update_taxonomy(taxonomy_id, name):
    """Handle taxonomy update command."""
    try:
        from .features.gui.services.api_client import api_client

        if not name:
            raise click.BadParameter("--name is required for update")

        request_data = {"name": name}
        response = api_client.put(f"/api/v1/taxonomy/{taxonomy_id}", data=request_data)
        click.echo(f"Taxonomy {taxonomy_id} updated successfully")

    except Exception as e:
        click.echo(f"Error updating taxonomy: {e}", err=True)
        raise click.Abort()


def handle_delete_taxonomy(taxonomy_id, confirm):
    """Handle taxonomy deletion command."""
    try:
        from .features.gui.services.api_client import api_client

        if not confirm:
            if not click.confirm(f"Are you sure you want to delete taxonomy {taxonomy_id}?"):
                return

        api_client.delete(f"/api/v1/taxonomy/{taxonomy_id}")
        click.echo(f"Taxonomy {taxonomy_id} deleted successfully")

    except Exception as e:
        click.echo(f"Error deleting taxonomy: {e}", err=True)
        raise click.Abort()


def handle_bulk_delete_taxonomies(taxonomy_ids, confirm):
    """Handle bulk taxonomy deletion command."""
    try:
        from .features.gui.services.api_client import api_client

        # Parse taxonomy IDs
        try:
            ids = [int(x.strip()) for x in taxonomy_ids.split(',')]
        except ValueError:
            raise click.BadParameter("--taxonomy-ids must be a comma-separated list of integers")

        if not confirm:
            if not click.confirm(f"Are you sure you want to delete {len(ids)} taxonomies?"):
                return

        request_data = {"taxonomy_ids": ids}
        response = api_client.delete("/api/v1/taxonomy/", data=request_data)

        deleted = len(response.get("deleted", []))
        failed = len(response.get("failed", []))
        click.echo(f"Deleted {deleted} taxonomies successfully")
        if failed > 0:
            click.echo(f"Failed to delete {failed} taxonomies", err=True)

    except Exception as e:
        click.echo(f"Error during bulk deletion: {e}", err=True)
        raise click.Abort()


# Job handlers
def handle_job_status(job_id, poll, timeout):
    """Handle job status command."""
    try:
        from .features.gui.services.api_client import api_client
        import time

        if poll:
            click.echo(f"Polling job {job_id} status (timeout: {timeout}s)...")
            start_time = time.time()

            while time.time() - start_time < timeout:
                response = api_client.get(f"/api/v1/jobs/status/{job_id}")
                if response:
                    status = response.get("status", "unknown")
                    progress = response.get("progress", 0)
                    click.echo(f"Status: {status.upper()} | Progress: {progress}%")

                    if status in ["completed", "failed", "cancelled"]:
                        if status == "completed":
                            click.echo("✅ Job completed successfully")
                        elif status == "failed":
                            error = response.get("error", "Unknown error")
                            click.echo(f"❌ Job failed: {error}", err=True)
                        else:
                            click.echo("⚠️ Job was cancelled")
                        return

                time.sleep(2)

            click.echo("⚠️ Polling timeout reached. Job may still be running.")
        else:
            response = api_client.get(f"/api/v1/jobs/status/{job_id}")
            if response:
                status = response.get("status", "unknown")
                progress = response.get("progress", 0)
                created_at = response.get("created_at", "Unknown")
                started_at = response.get("started_at", "Not started")
                completed_at = response.get("completed_at", "Not completed")

                click.echo(f"Job ID: {job_id}")
                click.echo(f"Status: {status.upper()}")
                click.echo(f"Progress: {progress}%")
                click.echo(f"Created: {created_at}")
                click.echo(f"Started: {started_at}")
                click.echo(f"Completed: {completed_at}")

                if response.get("error"):
                    click.echo(f"Error: {response['error']}")

                metrics = response.get("metrics", {})
                if metrics:
                    click.echo("Metrics:")
                    for key, value in metrics.items():
                        click.echo(f"  {key}: {value}")
            else:
                click.echo("❌ Failed to get job status")

    except Exception as e:
        click.echo(f"❌ Error getting job status: {e}", err=True)
        raise click.Abort()


def handle_cancel_job(job_id, confirm):
    """Handle job cancellation command."""
    try:
        from .features.gui.services.api_client import api_client

        if not confirm:
            if not click.confirm(f"Are you sure you want to cancel job {job_id}?"):
                return

        response = api_client.post(f"/api/v1/jobs/cancel/{job_id}")
        if response and response.get("status") == "cancelled":
            click.echo(f"✅ Job {job_id} cancelled successfully")
        else:
            click.echo("❌ Failed to cancel job")

    except Exception as e:
        click.echo(f"❌ Error cancelling job: {e}", err=True)
        raise click.Abort()


def handle_restart_job(job_id, confirm):
    """Handle job restart command."""
    try:
        from .features.gui.services.api_client import api_client

        if not confirm:
            if not click.confirm(f"Are you sure you want to restart job {job_id}?"):
                return

        response = api_client.post(f"/api/v1/jobs/restart/{job_id}")
        if response and response.get("status") == "pending":
            click.echo(f"✅ Job {job_id} restarted successfully")
        else:
            click.echo("❌ Failed to restart job")

    except Exception as e:
        click.echo(f"❌ Error restarting job: {e}", err=True)
        raise click.Abort()


def handle_queue_stats(output_format):
    """Handle queue statistics command."""
    try:
        from .features.gui.services.api_client import api_client
        import json

        response = api_client.get("/api/v1/jobs/queue/stats")
        if response:
            if output_format == "json":
                click.echo(json.dumps(response, indent=2))
            else:
                click.echo("--- Job Queue Statistics ---")
                click.echo(f"Pending Jobs: {response.get('pending_jobs', 0)}")
                click.echo(f"Running Jobs: {response.get('running_jobs', 0)}")
                click.echo(f"Scheduled Jobs: {response.get('scheduled_jobs', 0)}")
                click.echo(f"Completed Jobs: {response.get('completed_jobs', 0)}")
                click.echo(f"Failed Jobs: {response.get('failed_jobs', 0)}")
                click.echo(f"Total Jobs: {response.get('total_jobs', 0)}")
                click.echo(f"Queue Type: {response.get('queue_type', 'unknown')}")
                click.echo(f"Active Workers: {response.get('active_workers', 0)}")
                click.echo(f"CPU Usage: {response.get('cpu_usage_percent', 0):.1f}%")
                click.echo(f"Memory Usage: {response.get('memory_usage_mb', 0):.1f} MB")
                click.echo(f"Global Max Concurrent Jobs: {response.get('global_max_concurrent_jobs', 'N/A')}")
                # TODO: Uncomment when multi-user architecture is implemented
                # click.echo(f"Per-User Max Concurrent Jobs: {response.get('per_user_max_concurrent_jobs', 'N/A')}")
        else:
            click.echo("❌ Failed to get queue statistics")

    except Exception as e:
        click.echo(f"❌ Error getting queue stats: {e}", err=True)
        raise click.Abort()


def handle_poll_job_status(job_id, timeout, interval, quiet):
    """Handle job status polling command."""
    try:
        from .features.gui.services.api_client import api_client
        import time

        click.echo(f"🔄 Polling job {job_id} status (timeout: {timeout}s, interval: {interval}s)...")
        start_time = time.time()
        last_progress = -1

        while time.time() - start_time < timeout:
            response = api_client.get(f"/api/v1/jobs/status/{job_id}")

            if response:
                status = response.get("status", "unknown")
                progress = response.get("progress", 0)
                current_step = response.get("current_step", "")

                if not quiet:
                    if progress != last_progress:
                        click.echo(f"[{int(time.time() - start_time)}s] Status: {status.upper()} | Progress: {progress}% | Step: {current_step}")
                        last_progress = progress
                    else:
                        click.echo(f"[{int(time.time() - start_time)}s] Status: {status.upper()}")

                # Check if job is complete
                if status in ["completed", "failed", "cancelled"]:
                    if status == "completed":
                        click.echo("✅ Job completed successfully")
                        if not quiet:
                            result = response.get("result", {})
                            if result:
                                click.echo(f"📊 Result: {result}")
                    elif status == "failed":
                        error = response.get("error", "Unknown error")
                        click.echo(f"❌ Job failed: {error}", err=True)
                    else:
                        click.echo("⚠️ Job was cancelled")
                    return

            time.sleep(interval)

        click.echo("⚠️ Polling timeout reached. Job may still be running.")
        click.echo(f"Check status manually: compileo jobs status {job_id}")

    except Exception as e:
        click.echo(f"❌ Error polling job status: {e}", err=True)
        raise click.Abort()


def handle_start_standalone_worker(redis_url, queue_name, worker_name):
    """
    Handle standalone worker start command.
    Initializes the job queue manager without auto-starting the worker,
    then starts the RQ worker with proper configuration.
    """
    try:
        from .features.jobhandle.enhanced_job_queue import initialize_job_queue_manager, start_enhanced_worker
        from .storage.src.database import get_db_connection

        click.echo("🔧 Initializing job queue manager for standalone worker...")
        db_conn = get_db_connection()
        initialize_job_queue_manager(
            db_connection=db_conn
        )
        click.echo("✅ Job queue manager initialized.")

        click.echo(f"🚀 Starting RQ worker '{worker_name or 'default'}'...")
        click.echo(f"Listening on queue: '{queue_name}'")
        click.echo(f"Connecting to Redis at: {redis_url}")
        click.echo("Worker is now running. Press Ctrl+C to stop.")

        start_enhanced_worker(
            redis_url=redis_url,
            queue_name=queue_name,
            worker_name=worker_name
        )

    except Exception as e:
        click.echo(f"❌ Error starting standalone worker: {e}", err=True)
        raise click.Abort()


# Dataset version handlers
def handle_list_versions(project_id, dataset_name, output_format):
    """Handle dataset version listing command."""
    try:
        from .storage.src.project.database_repositories import DatasetVersionRepository
        from .storage.src.database import get_db_connection
        from .features.datasetgen.version_manager import DatasetVersionManager

        db_connection = get_db_connection()
        version_repo = DatasetVersionRepository(db_connection)
        version_manager = DatasetVersionManager(version_repo)

        history = version_manager.get_version_history(project_id, dataset_name)

        if not history:
            click.echo(f"No versions found for dataset '{dataset_name}' in project {project_id}")
            return

        if output_format == "json":
            import json
            click.echo(json.dumps(history, indent=2, default=str))
        else:
            # Table format
            click.echo(f"Versions for dataset '{dataset_name}':")
            click.echo("-" * 80)
            click.echo(f"{'Version':<12} {'Entries':<8} {'Active':<8} {'Created':<20} {'Description'}")
            click.echo("-" * 80)
            for version in history:
                active = "✓" if version["is_active"] else " "
                created = version["created_at"].strftime("%Y-%m-%d %H:%M") if version["created_at"] else "N/A"
                desc = version.get("description", "")[:30] + "..." if version.get("description", "") and len(version.get("description", "")) > 30 else version.get("description", "")
                click.echo(f"{version['version']:<12} {version['total_entries']:<8} {active:<8} {created:<20} {desc}")

    except Exception as e:
        click.echo(f"Error listing versions: {e}", err=True)
        raise click.Abort()


def handle_compare_versions(project_id, dataset_name, version1, version2):
    """Handle dataset version comparison command."""
    try:
        from .storage.src.project.database_repositories import DatasetVersionRepository
        from .storage.src.database import get_db_connection
        from .features.datasetgen.version_manager import DatasetVersionManager

        db_connection = get_db_connection()
        version_repo = DatasetVersionRepository(db_connection)
        version_manager = DatasetVersionManager(version_repo)

        comparison = version_manager.compare_versions(project_id, dataset_name, version1, version2)

        click.echo(f"Comparison between {version1} and {version2}:")
        click.echo("=" * 60)

        click.echo(f"Version {version1}:")
        click.echo(f"  Entries: {comparison['version1']['total_entries']}")
        click.echo(f"  Created: {comparison['version1']['created_at']}")
        click.echo(f"  Changes: {comparison['version1']['changes_count']}")

        click.echo(f"\nVersion {version2}:")
        click.echo(f"  Entries: {comparison['version2']['total_entries']}")
        click.echo(f"  Created: {comparison['version2']['created_at']}")
        click.echo(f"  Changes: {comparison['version2']['changes_count']}")

        click.echo(f"\nComparison:")
        click.echo(f"  Entries difference: {comparison['comparison']['entries_difference']}")
        click.echo(f"  Version relationship: {comparison['comparison']['version_difference']}")

    except Exception as e:
        click.echo(f"Error comparing versions: {e}", err=True)
        raise click.Abort()


def handle_rollback_version(project_id, dataset_name, target_version, confirm):
    """Handle dataset version rollback command."""
    try:
        from .storage.src.project.database_repositories import DatasetVersionRepository
        from .storage.src.database import get_db_connection
        from .features.datasetgen.version_manager import DatasetVersionManager

        if not confirm:
            if not click.confirm(f"Are you sure you want to rollback dataset '{dataset_name}' to version {target_version}? This will deactivate all newer versions."):
                return

        db_connection = get_db_connection()
        version_repo = DatasetVersionRepository(db_connection)
        version_manager = DatasetVersionManager(version_repo)

        success = version_manager.rollback_to_version(project_id, dataset_name, target_version)

        if success:
            click.echo(f"Successfully rolled back dataset '{dataset_name}' to version {target_version}")
        else:
            click.echo(f"Failed to rollback. Version {target_version} not found or rollback failed.", err=True)
            raise click.Abort()

    except Exception as e:
        click.echo(f"Error during rollback: {e}", err=True)
        raise click.Abort()


def handle_increment_version(project_id, dataset_name, version_type, description):
    """Handle dataset version increment command."""
    try:
        from .storage.src.project.database_repositories import DatasetVersionRepository
        from .storage.src.database import get_db_connection
        from .features.datasetgen.version_manager import DatasetVersionManager

        db_connection = get_db_connection()
        version_repo = DatasetVersionRepository(db_connection)
        version_manager = DatasetVersionManager(version_repo)

        new_version = version_manager.increment_version(
            project_id=project_id,
            dataset_name=dataset_name,
            version_type=version_type,
            description=description
        )

        click.echo(f"Successfully incremented dataset '{dataset_name}' to version {new_version}")

    except Exception as e:
        click.echo(f"Error incrementing version: {e}", err=True)
        raise click.Abort()


# Extraction handlers
def handle_create_extraction(project_id, taxonomy_id, selected_categories, effective_classifier, enable_validation_stage, validation_classifier, confidence_threshold, max_chunks, extraction_type, extraction_mode):
    """Handle extraction creation command."""
    try:
        from .features.gui.services.api_client import api_client

        # Parse selected categories
        try:
            categories = [cat.strip() for cat in selected_categories.split(',')]
        except Exception:
            raise click.BadParameter("--selected-categories must be a comma-separated list of category names")

        click.echo(f"🚀 Creating extraction job for project {project_id}")
        click.echo(f"📋 Taxonomy ID: {taxonomy_id}")
        click.echo(f"🏷️ Categories: {', '.join(categories)}")
        click.echo(f"🤖 Classifier: {effective_classifier}")
        click.echo(f"📝 Extraction Type: {extraction_type}")

        # Prepare request data
        request_data = {
            "project_id": project_id,
            "taxonomy_id": taxonomy_id,
            "selected_categories": categories,
            "effective_classifier": effective_classifier,
            "enable_validation_stage": enable_validation_stage,
            "confidence_threshold": confidence_threshold,
            "extraction_type": extraction_type,
            "extraction_mode": extraction_mode
        }

        # Add optional parameters
        if validation_classifier:
            request_data["validation_classifier"] = validation_classifier
        if max_chunks:
            request_data["max_chunks"] = max_chunks

        # Create extraction job
        response = api_client.post("/api/v1/extraction/", data=request_data)

        if response and "job_id" in response:
            job_id = response["job_id"]
            click.echo(f"✅ Extraction job created successfully!")
            click.echo(f"📋 Job ID: {job_id}")
            click.echo(f"🔍 Monitor progress: compileo jobs poll {job_id}")
            click.echo(f"📊 Check status: compileo jobs status {job_id}")
        else:
            click.echo("❌ Failed to create extraction job")
            raise click.Abort()

    except Exception as e:
        click.echo(f"❌ Error creating extraction job: {e}", err=True)
        raise click.Abort()

def handle_delete_chunks(chunk_id, document_id, confirm):
    """Handle chunk deletion command."""
    try:
        if not chunk_id and not document_id:
             click.echo("❌ Error: Must specify either --chunk-id or --document-id")
             return

        from .storage.src.database import get_db_connection
        from .features.chunk.service import ChunkService

        if not confirm:
            target = f"chunk {chunk_id}" if chunk_id else f"all chunks for document {document_id}"
            if not click.confirm(f"Are you sure you want to delete {target}? This cannot be undone."):
                return

        db_connection = get_db_connection()
        service = ChunkService(db_connection)

        if chunk_id:
            if service.delete_chunk(chunk_id):
                click.echo(f"✅ Chunk {chunk_id} deleted successfully")
            else:
                click.echo(f"❌ Chunk {chunk_id} not found or could not be deleted")
        elif document_id:
            count = service.delete_document_chunks(document_id)
            click.echo(f"✅ Deleted {count} chunks for document {document_id}")

    except Exception as e:
        click.echo(f"❌ Error deleting chunks: {e}", err=True)
        raise click.Abort()


# Benchmarking handlers
def handle_run_benchmark(project_id, suite, provider, model, ollama_params, custom_config, tasks):
    """Handle benchmark run command."""
    try:
        from .features.gui.services.api_client import api_client
        import json

        click.echo(f"🚀 Starting benchmark evaluation")
        click.echo(f"📊 Suite: {suite}")
        click.echo(f"🤖 Provider: {provider}")
        click.echo(f"🧠 Model: {model}")

        # Prepare request data
        request_data = {
            "project_id": project_id,
            "suite": suite,
            "config": {
                "provider": provider,
                "model": model
            }
        }

        # Add Ollama parameters if applicable
        if provider == "ollama" and ollama_params:
            request_data["config"].update(ollama_params)

        # Add custom config if provided
        if custom_config:
            try:
                custom_config_dict = json.loads(custom_config)
                request_data["config"].update(custom_config_dict)
            except json.JSONDecodeError:
                raise click.BadParameter("Invalid JSON in --custom-config")

        # Add tasks if specified
        if tasks:
            request_data["config"]["tasks"] = tasks

        # Start benchmark
        response = api_client.post("/api/v1/benchmarking/run", data=request_data)

        if response and "job_id" in response:
            job_id = response["job_id"]
            click.echo(f"✅ Benchmarking started successfully!")
            click.echo(f"📋 Job ID: {job_id}")
            click.echo(f"🔍 Monitor progress: compileo benchmark status {job_id}")
            click.echo(f"📊 Check results: compileo benchmark results {job_id}")
        else:
            click.echo("❌ Failed to start benchmarking")
            raise click.Abort()

    except Exception as e:
        click.echo(f"❌ Error starting benchmark: {e}", err=True)
        raise click.Abort()


def handle_get_benchmark_results(job_id, output_format):
    """Handle benchmark results retrieval command."""
    try:
        from .features.gui.services.api_client import api_client
        import json

        response = api_client.get(f"/api/v1/benchmarking/results/{job_id}")

        if response:
            if output_format == "json":
                click.echo(json.dumps(response, indent=2, default=str))
            else:
                # Table format
                click.echo(f"Benchmark Results for Job {job_id}")
                click.echo("=" * 50)

                status = response.get("status", "unknown")
                click.echo(f"Status: {status.upper()}")

                if status == "completed":
                    summary = response.get("summary", {})
                    performance_data = response.get("performance_data", {})

                    click.echo(f"Total Evaluations: {summary.get('total_evaluations', 0)}")
                    click.echo(f"Benchmarks Run: {', '.join(summary.get('benchmarks_run', []))}")
                    click.echo(f"Models Evaluated: {summary.get('models_evaluated', 0)}")
                    click.echo(f"Total Time: {summary.get('total_time_seconds', 0):.1f}s")

                    # Show performance metrics
                    if performance_data:
                        click.echo("\nPerformance Metrics:")
                        for benchmark, metrics in performance_data.items():
                            click.echo(f"  {benchmark.upper()}:")
                            for metric, values in metrics.items():
                                if isinstance(values, dict):
                                    mean_val = values.get('mean', 'N/A')
                                    std_val = values.get('std', 'N/A')
                                    click.echo(f"    {metric}: {mean_val:.3f} ± {std_val:.3f}")
                                else:
                                    click.echo(f"    {metric}: {values}")
                else:
                    click.echo("Job not completed yet or failed")
        else:
            click.echo("❌ Failed to retrieve benchmark results")

    except Exception as e:
        click.echo(f"❌ Error retrieving benchmark results: {e}", err=True)
        raise click.Abort()


def handle_list_benchmarks(model, suite, status, limit, output_format):
    """Handle benchmark listing command."""
    try:
        from .features.gui.services.api_client import api_client
        import json

        # Build query parameters
        params = {"limit": limit}
        if model:
            params["model_name"] = model
        if suite:
            params["suite"] = suite
        if status:
            params["status"] = status

        response = api_client.get("/api/v1/benchmarking/results", params=params)

        if response and "results" in response:
            results = response["results"]
            total = response.get("total", len(results))

            if not results:
                click.echo("No benchmark results found matching criteria.")
                return

            if output_format == "json":
                click.echo(json.dumps({"results": results, "total": total}, indent=2, default=str))
            else:
                # Table format
                click.echo(f"Benchmark Results (Total: {total}):")
                click.echo("-" * 120)
                click.echo(f"{'Job ID':<40} {'Model':<20} {'Suite':<10} {'Status':<10} {'Completed'}")
                click.echo("-" * 120)

                for result in results:
                    job_id = str(result.get('job_id', ''))[:39]
                    model_name = result.get('model_name', 'Unknown')[:19]
                    suite_name = result.get('benchmark_suite', 'Unknown')[:9]
                    status_val = result.get('status', 'unknown')[:9]
                    completed_at = result.get('completed_at', '')
                    if completed_at and len(str(completed_at)) > 10:
                        completed_at = str(completed_at)[:10]

                    click.echo(f"{job_id:<40} {model_name:<20} {suite_name:<10} {status_val:<10} {completed_at}")
        else:
            click.echo("❌ Failed to retrieve benchmark list")

    except Exception as e:
        click.echo(f"❌ Error listing benchmarks: {e}", err=True)
        raise click.Abort()


def handle_compare_models(model_ids, suite, metrics, output_format):
    """Handle model comparison command."""
    try:
        from .features.gui.services.api_client import api_client
        import json

        request_data = {
            "model_ids": model_ids,
            "benchmark_suite": suite
        }

        if metrics:
            request_data["metrics"] = metrics

        response = api_client.post("/api/v1/benchmarking/compare", data=request_data)

        if response and "comparison" in response:
            comparison = response["comparison"]

            if output_format == "json":
                click.echo(json.dumps(comparison, indent=2, default=str))
            else:
                # Formatted output
                click.echo("Model Comparison Results")
                click.echo("=" * 40)

                click.echo(f"Models Compared: {', '.join(comparison.get('models_compared', []))}")
                click.echo(f"Suite: {comparison.get('suite', 'Unknown')}")

                results = comparison.get('results', {})
                if results:
                    click.echo(f"Best Performing Model: {results.get('best_performing', 'N/A')}")
                    click.echo(f"Performance Gap: {results.get('performance_gap', 0):.3f}")
                    click.echo(f"Statistical Significance: {results.get('statistical_significance', 'N/A')}")

                recommendations = comparison.get('recommendations', [])
                if recommendations:
                    click.echo("\nRecommendations:")
                    for rec in recommendations:
                        click.echo(f"  • {rec}")
        else:
            click.echo("❌ Failed to perform model comparison")

    except Exception as e:
        click.echo(f"❌ Error comparing models: {e}", err=True)
        raise click.Abort()


def handle_benchmark_history(model, days, output_format):
    """Handle benchmark history command."""
    try:
        from .features.gui.services.api_client import api_client
        import json

        params = {"days": days}
        if model:
            params["model_name"] = model

        response = api_client.get("/api/v1/benchmarking/history", params=params)

        if response and "history" in response:
            history = response["history"]
            total_runs = response.get("total_runs", len(history))

            if not history:
                click.echo("No benchmark history found.")
                return

            if output_format == "json":
                click.echo(json.dumps({"history": history, "total_runs": total_runs}, indent=2, default=str))
            else:
                # Table format
                click.echo(f"Benchmark History (Last {days} days, Total: {total_runs}):")
                click.echo("-" * 100)
                click.echo(f"{'Job ID':<40} {'Model':<15} {'Suite':<10} {'Status':<10} {'Date'}")
                click.echo("-" * 100)

                for item in history:
                    job_id = str(item.get('job_id', ''))[:39]
                    model_name = item.get('model_name', 'Unknown')[:14]
                    suite_name = item.get('benchmark_suite', 'Unknown')[:9]
                    status_val = item.get('status', 'unknown')[:9]
                    created_at = item.get('created_at', '')
                    if created_at and len(str(created_at)) > 10:
                        created_at = str(created_at)[:10]

                    click.echo(f"{job_id:<40} {model_name:<15} {suite_name:<10} {status_val:<10} {created_at}")
        else:
            click.echo("❌ Failed to retrieve benchmark history")

    except Exception as e:
        click.echo(f"❌ Error retrieving benchmark history: {e}", err=True)
        raise click.Abort()


def handle_benchmark_leaderboard(suite, metric, limit, output_format):
    """Handle benchmark leaderboard command."""
    try:
        from .features.gui.services.api_client import api_client
        import json

        params = {
            "suite": suite,
            "metric": metric,
            "limit": limit
        }

        response = api_client.get("/api/v1/benchmarking/leaderboard", params=params)

        if response and "leaderboard" in response:
            leaderboard = response["leaderboard"]
            total_models = response.get("total_models", len(leaderboard))

            if not leaderboard:
                click.echo("No leaderboard data available.")
                return

            if output_format == "json":
                click.echo(json.dumps({
                    "suite": suite,
                    "metric": metric,
                    "leaderboard": leaderboard,
                    "total_models": total_models
                }, indent=2, default=str))
            else:
                # Table format
                click.echo(f"🏆 {suite.upper()} Leaderboard (Top {limit}, Metric: {metric})")
                click.echo("=" * 60)

                for entry in leaderboard:
                    rank = entry.get('rank', 0)
                    model = entry.get('model', 'Unknown')
                    score = entry.get('score', 0)
                    evaluations = entry.get('evaluation_count', 0)

                    medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"#{rank}"
                    click.echo(f"{medal} {model}: {score:.4f} ({evaluations} evaluations)")

                click.echo(f"\nTotal models evaluated: {total_models}")
        else:
            click.echo("❌ Failed to retrieve leaderboard")

    except Exception as e:
        click.echo(f"❌ Error retrieving leaderboard: {e}", err=True)
        raise click.Abort()


def handle_benchmark_status(job_id, poll, timeout):
    """Handle benchmark status command."""
    try:
        from .features.gui.services.api_client import api_client
        import time

        if poll:
            click.echo(f"🔄 Polling benchmark job {job_id} status (timeout: {timeout}s)...")
            start_time = time.time()

            while time.time() - start_time < timeout:
                response = api_client.get(f"/api/v1/benchmarking/results/{job_id}")
                if response:
                    status = response.get("status", "unknown")
                    click.echo(f"Status: {status.upper()}")

                    if status in ["completed", "failed"]:
                        if status == "completed":
                            click.echo("✅ Benchmark completed successfully")
                            # Show summary
                            summary = response.get("summary", {})
                            if summary:
                                click.echo(f"📊 Evaluations: {summary.get('total_evaluations', 0)}")
                                click.echo(f"🤖 Models: {summary.get('models_evaluated', 0)}")
                        else:
                            error = response.get("error", "Unknown error")
                            click.echo(f"❌ Benchmark failed: {error}", err=True)
                        return

                time.sleep(2)

            click.echo("⚠️ Polling timeout reached. Job may still be running.")
        else:
            response = api_client.get(f"/api/v1/benchmarking/results/{job_id}")
            if response:
                status = response.get("status", "unknown")
                summary = response.get("summary", {})

                click.echo(f"Job ID: {job_id}")
                click.echo(f"Status: {status.upper()}")

                if summary:
                    click.echo(f"Evaluations: {summary.get('total_evaluations', 0)}")
                    click.echo(f"Models: {summary.get('models_evaluated', 0)}")
                    click.echo(f"Benchmarks: {', '.join(summary.get('benchmarks_run', []))}")

                if response.get("error"):
                    click.echo(f"Error: {response['error']}")
            else:
                click.echo("❌ Failed to get benchmark status")

    except Exception as e:
        click.echo(f"❌ Error checking benchmark status: {e}", err=True)
        raise click.Abort()

# Plugin handlers
def handle_list_plugins(output_format):
    """Handle plugin listing command."""
    try:
        from .features.gui.services.api_client import api_client
        import json

        response = api_client.get("/api/v1/plugins/")
        
        if not response:
            click.echo("No plugins installed.")
            return

        if output_format == "json":
            click.echo(json.dumps(response, indent=2))
        else:
            # Table format
            click.echo(f"Installed Plugins ({len(response)}):")
            click.echo("-" * 80)
            click.echo(f"{'ID':<25} {'Name':<25} {'Version':<10} {'Author'}")
            click.echo("-" * 80)
            
            for plugin in response:
                pid = plugin.get('id', 'Unknown')[:24]
                name = plugin.get('name', 'Unknown')[:24]
                version = plugin.get('version', 'Unknown')[:9]
                author = plugin.get('author', 'Unknown')
                click.echo(f"{pid:<25} {name:<25} {version:<10} {author}")

    except Exception as e:
        click.echo(f"❌ Error listing plugins: {e}", err=True)
        raise click.Abort()

def handle_install_plugin(plugin_file):
    """Handle plugin installation command."""
    try:
        import os
        from .features.gui.services.api_client import api_client
        
        if not os.path.exists(plugin_file):
            raise click.BadParameter(f"File not found: {plugin_file}")
            
        if not plugin_file.endswith('.zip'):
            raise click.BadParameter("Plugin must be a .zip file")

        click.echo(f"📦 Installing plugin from {plugin_file}...")
        
        # We need to manually construct the request since api_client might not have a generic file upload method 
        # that fits this endpoint perfectly or we want to use the unified client.
        # Let's check api_client... it has upload_documents but that's specific.
        # We'll use the underlying session or requests logic if possible, or just standard requests 
        # using the base_url from api_client.
        
        import requests
        
        base_url = api_client.base_url
        url = f"{base_url}/api/v1/plugins/upload"
        
        files = {'file': open(plugin_file, 'rb')}
        headers = {}
        if api_client.api_key:
            headers["X-API-Key"] = str(api_client.api_key)
            
        response = requests.post(url, files=files, headers=headers)
        
        if response.status_code == 200:
            result = response.json()
            click.echo(f"✅ Plugin installed successfully!")
            click.echo(f"ID: {result.get('plugin_id')}")
        else:
            click.echo(f"❌ Installation failed: {response.text}")
            raise click.Abort()

    except Exception as e:
        click.echo(f"❌ Error installing plugin: {e}", err=True)
        raise click.Abort()

def handle_uninstall_plugin(plugin_id, confirm):
    """Handle plugin uninstallation command."""
    try:
        from .features.gui.services.api_client import api_client
        
        if not confirm:
            if not click.confirm(f"Are you sure you want to uninstall plugin '{plugin_id}'?"):
                return

        click.echo(f"🗑️ Uninstalling plugin {plugin_id}...")
        
        response = api_client.delete(f"/api/v1/plugins/{plugin_id}")
        
        if response and response.get("status") == "success":
            click.echo(f"✅ Plugin uninstalled successfully")
        else:
            click.echo("❌ Failed to uninstall plugin")
            
    except Exception as e:
        click.echo(f"❌ Error uninstalling plugin: {e}", err=True)
        raise click.Abort()
