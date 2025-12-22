"""
CLI Command Definitions.
Contains all Click command definitions and decorators.
"""

import click
from . import __version__


# Main CLI group
@click.group()
@click.version_option(__version__)
@click.option("--log-level", type=click.Choice(["none", "error", "debug"]), help="Set the log level")
def cli(log_level):
    """Compileo: A modular pipeline for dataset creation and curation."""
    from .core.logging import setup_logging
    from .core.settings import LogLevel, BackendSettings
    
    if log_level:
        # Override setting for this run and update persistent setting
        lvl = LogLevel(log_level)
        BackendSettings.set_log_level(lvl)
        setup_logging(lvl)
    else:
        # Use existing setting
        setup_logging()


# Hello command
@cli.command()
def hello():
    """Say hello from Compileo."""
    click.echo("Hello from Compileo!")


# Shutdown command
@cli.command()
@click.option("--api-url", default="http://localhost:8000", help="API server URL")
@click.option("--timeout", default=30, type=int, help="Request timeout in seconds")
@click.confirmation_option(prompt="This will shutdown the Compileo API server and all workers. Continue?")
def shutdown(api_url, timeout):
    """Shutdown the Compileo API server and all background workers."""
    import requests

    try:
        click.echo("Initiating Compileo shutdown...")

        # Call the shutdown endpoint
        response = requests.post(f"{api_url}/shutdown", timeout=timeout)

        if response.status_code == 200:
            click.echo("✅ Shutdown initiated successfully!")
            click.echo("The API server and workers will shutdown shortly.")
            click.echo("You may need to restart the services manually.")
        else:
            click.echo(f"❌ Shutdown failed: HTTP {response.status_code}")
            click.echo(f"Response: {response.text[:500]}")

    except requests.exceptions.RequestException as e:
        click.echo(f"❌ Could not connect to API server: {e}")
        click.echo("Make sure the Compileo API server is running and accessible.")
    except Exception as e:
        click.echo(f"❌ Unexpected error during shutdown: {e}")


# Documents command group
@cli.group()
def documents():
    """Manage documents for processing and chunking."""
    pass


# Documents subcommands
@documents.command("upload")
@click.argument("file_paths", nargs=-1, required=True)
@click.option("--project-id", required=True, type=int, help="Project ID to upload documents to")
def upload_documents(file_paths, project_id):
    """Upload one or more documents to a project."""
    from .cli_handlers import handle_upload_documents
    handle_upload_documents(file_paths, project_id)


@documents.command("process")
@click.option("--project-id", required=True, type=int, help="Project ID containing documents to process")
@click.option("--document-ids", required=True, help="Comma-separated list of document IDs to process")
@click.option("--parser", default="gemini", type=click.Choice(["gemini", "grok", "ollama", "openai", "pypdf", "unstructured", "huggingface", "novlm"]), help="Document parser to use")
@click.option("--chunk-strategy", default="token", type=click.Choice(["token", "character", "semantic", "delimiter", "schema"]), help="Text chunking strategy")
@click.option("--chunk-size", default=512, type=int, help="Chunk size (tokens for token strategy, characters for character strategy)")
@click.option("--overlap", default=50, type=int, help="Overlap between chunks (tokens or characters)")
@click.option("--semantic-prompt", help="Custom prompt for semantic chunking")
@click.option("--schema-definition", help="JSON schema definition for schema-based chunking")
@click.option("--character-chunk-size", type=int, help="Character chunk size (overrides --chunk-size for character strategy)")
@click.option("--character-overlap", type=int, help="Character overlap (overrides --overlap for character strategy)")
@click.option("--skip-parsing", is_flag=True, help="Skip parsing if documents are already parsed")
def process_documents(project_id, document_ids, parser, chunk_strategy, chunk_size, overlap, semantic_prompt, schema_definition, character_chunk_size, character_overlap, skip_parsing):
    """Process documents with specified parsing and chunking options."""
    from .cli_handlers import handle_process_documents
    handle_process_documents(project_id, document_ids, parser, chunk_strategy, chunk_size, overlap, semantic_prompt, schema_definition, character_chunk_size, character_overlap, skip_parsing)


@documents.command("list")
@click.option("--project-id", type=int, help="Filter documents by project ID")
@click.option("--format", "output_format", default="table", type=click.Choice(["table", "json"]), help="Output format")
def list_documents(project_id, output_format):
    """List documents, optionally filtered by project."""
    from .cli_handlers import handle_list_documents
    handle_list_documents(project_id, output_format)


@documents.command("delete")
@click.argument("document_id", type=int)
@click.option("--confirm", is_flag=True, help="Skip confirmation prompt")
def delete_document(document_id, confirm):
    """Delete a document and all its associated chunks."""
    from .cli_handlers import handle_delete_document
    handle_delete_document(document_id, confirm)


@documents.command("parse")
@click.option("--project-id", required=True, type=int, help="Project ID containing documents to parse")
@click.option("--document-ids", required=True, help="Comma-separated list of document IDs to parse")
@click.option("--parser", default="gemini", type=click.Choice(["gemini", "grok", "ollama", "openai", "pypdf", "unstructured", "huggingface", "novlm"]), help="Document parser to use")
def parse_documents(project_id, document_ids, parser):
    """Parse documents to markdown without chunking."""
    from .cli_handlers import handle_parse_documents
    handle_parse_documents(project_id, document_ids, parser)


@documents.command("chunk")
@click.option("--project-id", required=True, type=int, help="Project ID containing documents to chunk")
@click.option("--document-ids", required=True, help="Comma-separated list of document IDs to chunk")
@click.option("--chunk-strategy", default="token", type=click.Choice(["token", "character", "semantic", "delimiter", "schema"]), help="Text chunking strategy")
@click.option("--chunk-size", default=512, type=int, help="Chunk size (tokens for token strategy, characters for character strategy)")
@click.option("--overlap", default=50, type=int, help="Overlap between chunks (tokens or characters)")
@click.option("--chunker", default="gemini", type=click.Choice(["gemini", "grok", "ollama", "openai"]), help="AI model for intelligent chunking")
@click.option("--semantic-prompt", help="Custom prompt for semantic chunking")
@click.option("--schema-definition", help="JSON schema definition for schema-based chunking")
@click.option("--character-chunk-size", type=int, help="Character chunk size (overrides --chunk-size for character strategy)")
@click.option("--character-overlap", type=int, help="Character overlap (overrides --overlap for character strategy)")
@click.option("--num-ctx", type=int, help="Context window size for Ollama models (overrides default setting)")
@click.option("--system-instruction", help="System-level instructions to guide the model's behavior, especially for Gemini")
@click.option("--sliding-window", is_flag=True, help="Use sliding window chunking for multi-file documents (mandatory for semantic coherence)")
def chunk_documents(project_id, document_ids, chunk_strategy, chunk_size, overlap, chunker, semantic_prompt, schema_definition, character_chunk_size, character_overlap, num_ctx, system_instruction, sliding_window):
    """Chunk already parsed documents using specified chunking strategy."""
    from .cli_handlers import handle_chunk_documents
    handle_chunk_documents(project_id, document_ids, chunk_strategy, chunk_size, overlap, chunker, semantic_prompt, schema_definition, character_chunk_size, character_overlap, num_ctx, system_instruction, sliding_window)


@documents.command("content")
@click.argument("document_id", type=int)
@click.option("--page", default=1, type=int, help="Page number to view")
@click.option("--page-size", default=3000, type=int, help="Characters per page")
@click.option("--output", "output_file", help="Save content to file instead of displaying")
def view_document_content(document_id, page, page_size, output_file):
    """View parsed content of a document with pagination support."""
    from .cli_handlers import handle_view_document_content
    handle_view_document_content(document_id, page, page_size, output_file)


@documents.command("split-pdf")
@click.argument("pdf_path", type=click.Path(exists=True))
@click.option("--pages-per-split", default=200, type=int, help="Number of pages per split chunk")
@click.option("--overlap-pages", default=1, type=int, help="Number of overlapping pages between splits")
def split_pdf(pdf_path, pages_per_split, overlap_pages):
    """Split a large PDF into smaller chunks for processing."""
    from .cli_handlers import handle_split_pdf
    handle_split_pdf(pdf_path, pages_per_split, overlap_pages)


@documents.command("status")
@click.option("--job-id", required=True, help="Job ID to check status for")
@click.option("--type", "job_type", default="process", type=click.Choice(["upload", "process", "parse"]), help="Job type")
def check_job_status(job_id, job_type):
    """Check the status of an upload, parsing, or processing job."""
    from .cli_handlers import handle_check_job_status
    handle_check_job_status(job_id, job_type)


# Single process command
@cli.command()
@click.argument("file_path")
@click.option("--project-id", required=True, type=int, help="Project ID to upload document to")
@click.option("--parser", default="gemini", type=click.Choice(["gemini", "grok", "ollama", "openai", "pypdf", "unstructured", "huggingface", "novlm"]), help="Document parser to use")
@click.option("--chunk-strategy", default="token", type=click.Choice(["token", "character", "semantic", "delimiter", "schema"]), help="Text chunking strategy")
@click.option("--chunk-size", default=512, type=int, help="Chunk size (tokens for token strategy, characters for character strategy)")
@click.option("--overlap", default=50, type=int, help="Overlap between chunks (tokens or characters)")
@click.option("--semantic-prompt", help="Custom prompt for semantic chunking")
@click.option("--schema-definition", help="JSON schema definition for schema-based chunking")
@click.option("--character-chunk-size", type=int, help="Character chunk size (overrides --chunk-size for character strategy)")
@click.option("--character-overlap", type=int, help="Character overlap (overrides --overlap for character strategy)")
@click.option("--skip-parsing", is_flag=True, help="Skip parsing if document is already parsed")
def process(file_path, project_id, parser, chunk_strategy, chunk_size, overlap, semantic_prompt, schema_definition, character_chunk_size, character_overlap, skip_parsing):
    """Process a single document file with specified parsing and chunking options."""
    from .cli_handlers import handle_process_single_file
    handle_process_single_file(file_path, project_id, parser, chunk_strategy, chunk_size, overlap, semantic_prompt, schema_definition, character_chunk_size, character_overlap, skip_parsing)


# Dataset generation command
@cli.command()
@click.option("--project-id", required=True, type=int, help="Project ID for dataset generation")
@click.option("--prompt-name", default="default", help="Name of the prompt to use")
@click.option("--format-type", default="jsonl", type=click.Choice(["jsonl", "parquet"]), help="Output format")
@click.option("--concurrency", default=2, type=int, help="Number of parallel workers within each batch (1-5)")
@click.option("--batch-size", default=50, type=int, help="Number of chunks to process per batch (0 = all at once)")
@click.option("--include-evaluation-sets", is_flag=True, help="Generate comprehensive evaluation datasets")
@click.option("--taxonomy-project", help="Project name containing the taxonomy to use")
@click.option("--taxonomy-name", help="Name of the taxonomy to use for dataset generation")
@click.option("--output-dir", default=".", help="Output directory for generated datasets")
@click.option("--analyze-quality", is_flag=True, help="Enable quality analysis of generated dataset")
@click.option("--quality-threshold", type=float, default=0.7, help="Quality threshold for pass/fail (0-1)")
@click.option("--quality-config", help="Path to quality configuration JSON file")
@click.option("--enable-versioning", is_flag=True, help="Enable dataset versioning")
@click.option("--dataset-name", help="Name for the dataset (required for versioning)")
@click.option("--run-benchmarks", is_flag=True, help="Run AI model benchmarks after dataset generation")
@click.option("--benchmark-suite", default="glue", help="Benchmark suite to run (glue, superglue, mmlu, medical)")
@click.option("--benchmark-config", help="Path to benchmarking configuration JSON file")
@click.option("--category-limits", help="Comma-separated list of integers for max categories per taxonomy level (e.g., '5,10,15')")
@click.option("--specificity-level", default=1, type=click.IntRange(1, 5), help="Base specificity level for taxonomy generation (1-5)")
@click.option("--custom-audience", help="Target audience description (e.g., 'medical residents preparing for board exams')")
@click.option("--custom-purpose", help="Dataset purpose description (e.g., 'create practice questions that match symptoms to specific diagnoses')")
@click.option("--complexity-level", type=click.Choice(["auto", "basic", "intermediate", "advanced"]), default="intermediate", help="Complexity level for generated content")
@click.option("--domain", default="general", help="Knowledge domain or specialty area")
@click.option("--data-source", default="Chunks Only", type=click.Choice(["Chunks Only", "Taxonomy", "Extract"]), help="Data source for dataset generation")
@click.option("--extraction-file-id", help="Specific extraction file ID when data_source is 'Extract'")
@click.option("--datasets-per-chunk", default=3, type=click.IntRange(1, 50), help="Maximum number of datasets to generate per text chunk")
@click.option("--selected-categories", help="Comma-separated list of category names to filter by (only for Extract mode)")
def generate_dataset(project_id, prompt_name, format_type, concurrency, batch_size, include_evaluation_sets, taxonomy_project, taxonomy_name, data_source, extraction_file_id, output_dir, analyze_quality, quality_threshold, quality_config, enable_versioning, dataset_name, run_benchmarks, benchmark_suite, benchmark_config, category_limits, specificity_level, custom_audience, custom_purpose, complexity_level, domain, datasets_per_chunk, selected_categories):
    """Generate a dataset from processed chunks."""
    from .cli_handlers import handle_generate_dataset
    handle_generate_dataset(project_id, prompt_name, format_type, concurrency, batch_size, include_evaluation_sets, taxonomy_project, taxonomy_name, data_source, extraction_file_id, output_dir, analyze_quality, quality_threshold, quality_config, enable_versioning, dataset_name, run_benchmarks, benchmark_suite, benchmark_config, category_limits, specificity_level, custom_audience, custom_purpose, complexity_level, domain, datasets_per_chunk, selected_categories)


# Quality analysis command
@cli.command()
@click.argument("dataset_file", type=click.Path(exists=True))
@click.option("--config", "quality_config", help="Path to quality configuration JSON file")
@click.option("--threshold", type=float, default=0.7, help="Quality threshold for pass/fail (0-1)")
@click.option("--format", "output_format", default="text", type=click.Choice(["json", "text", "markdown"]), help="Output format")
@click.option("--output", "output_file", help="Save report to file")
def analyze_quality(dataset_file, quality_config, threshold, output_format, output_file):
    """Analyze quality of an existing dataset."""
    from .cli_handlers import handle_analyze_quality
    handle_analyze_quality(dataset_file, quality_config, threshold, output_format, output_file)


# Taxonomy command group
@cli.group()
def taxonomy():
    """Manage taxonomies for content categorization."""
    pass


# Taxonomy subcommands
@taxonomy.command()
@click.option("--project-id", type=int, help="Filter taxonomies by project ID")
@click.option("--format", "output_format", default="table", type=click.Choice(["table", "json"]), help="Output format")
def list(project_id, output_format):
    """List all available taxonomies."""
    from .cli_handlers import handle_list_taxonomies
    handle_list_taxonomies(project_id, output_format)


@taxonomy.command()
@click.option("--project-id", required=True, type=int, help="Project ID for the taxonomy")
@click.option("--name", required=True, help="Name of the taxonomy")
@click.option("--description", help="Description of the taxonomy")
@click.option("--file", "taxonomy_file", type=click.Path(exists=True), help="JSON file containing taxonomy structure")
def create(project_id, name, description, taxonomy_file):
    """Create a new manual taxonomy."""
    from .cli_handlers import handle_create_taxonomy
    handle_create_taxonomy(project_id, name, description, taxonomy_file)


@taxonomy.command()
@click.option("--project-id", required=True, type=int, help="Project ID for taxonomy generation")
@click.option("--name", required=True, help="Name for the generated taxonomy")
@click.option("--documents", required=True, help="Comma-separated list of document IDs to analyze")
@click.option("--depth", default=3, type=int, help="Taxonomy hierarchy depth")
@click.option("--generator", default="gemini", type=click.Choice(["gemini", "grok", "ollama", "openai"]), help="AI generator to use")
@click.option("--domain", default="general", help="Content domain for generation")
@click.option("--batch-size", default=10, type=int, help="Number of complete chunks to process in this batch")
@click.option("--category-limits", help="Comma-separated list of integers for max categories per level")
@click.option("--specificity-level", default=1, type=click.IntRange(1, 5), help="Specificity level (1-5)")
@click.option("--processing-mode", default="fast", type=click.Choice(["fast", "complete"]), help="Processing mode: fast (sampled) or complete (all content)")
def generate(project_id, name, documents, depth, generator, domain, batch_size, category_limits, specificity_level, processing_mode):
    """Generate a new taxonomy using AI."""
    from .cli_handlers import handle_generate_taxonomy
    handle_generate_taxonomy(project_id, name, documents, depth, generator, domain, batch_size, category_limits, specificity_level, processing_mode)


@taxonomy.command()
@click.option("--taxonomy-data", type=click.Path(exists=True), help="JSON file containing taxonomy/category data to extend")
@click.option("--project-id", required=True, type=int, help="Project ID")
@click.option("--additional-depth", default=2, type=int, help="Number of additional levels to add")
@click.option("--generator", default="gemini", type=click.Choice(["gemini", "grok", "ollama", "openai"]), help="AI generator to use")
@click.option("--domain", default="general", help="Content domain")
@click.option("--batch-size", default=10, type=int, help="Number of complete chunks to process in this batch")
@click.option("--documents", help="Comma-separated list of document IDs to analyze")
@click.option("--processing-mode", default="fast", type=click.Choice(["fast", "complete"]), help="Processing mode: fast (sampled) or complete (all content)")
def extend(taxonomy_data, project_id, additional_depth, generator, domain, batch_size, documents, processing_mode):
    """Extend an existing taxonomy or category with AI-generated subcategories."""
    from .cli_handlers import handle_extend_taxonomy
    handle_extend_taxonomy(taxonomy_data, project_id, additional_depth, generator, domain, batch_size, documents, processing_mode)


@taxonomy.command()
@click.argument("taxonomy_id", type=int)
@click.option("--format", "output_format", default="json", type=click.Choice(["json", "text"]), help="Output format")
@click.option("--output", "output_file", help="Save taxonomy to file")
def load(taxonomy_id, output_format, output_file):
    """Load detailed information about a specific taxonomy."""
    from .cli_handlers import handle_load_taxonomy
    handle_load_taxonomy(taxonomy_id, output_format, output_file)


@taxonomy.command()
@click.argument("taxonomy_id", type=int)
@click.option("--name", help="New name for the taxonomy")
def update(taxonomy_id, name):
    """Update taxonomy information."""
    from .cli_handlers import handle_update_taxonomy
    handle_update_taxonomy(taxonomy_id, name)


@taxonomy.command()
@click.argument("taxonomy_id", type=int)
@click.option("--confirm", is_flag=True, help="Confirm deletion")
def delete(taxonomy_id, confirm):
    """Delete a taxonomy."""
    from .cli_handlers import handle_delete_taxonomy
    handle_delete_taxonomy(taxonomy_id, confirm)


@taxonomy.command()
@click.option("--taxonomy-ids", required=True, help="Comma-separated list of taxonomy IDs to delete")
@click.option("--confirm", is_flag=True, help="Confirm bulk deletion")
def bulk_delete(taxonomy_ids, confirm):
    """Delete multiple taxonomies at once."""
    from .cli_handlers import handle_bulk_delete_taxonomies
    handle_bulk_delete_taxonomies(taxonomy_ids, confirm)


# Jobs command group
@cli.group()
def jobs():
    """Manage background jobs and queue operations."""
    pass


# Jobs subcommands
@jobs.command("status")
@click.argument("job_id")
@click.option("--poll", is_flag=True, help="Poll for status changes until completion")
@click.option("--timeout", default=30, type=int, help="Timeout for polling in seconds")
def job_status(job_id, poll, timeout):
    """Get the status of a specific job."""
    from .cli_handlers import handle_job_status
    handle_job_status(job_id, poll, timeout)


@jobs.command("cancel")
@click.argument("job_id")
@click.option("--confirm", is_flag=True, help="Skip confirmation prompt")
def cancel_job(job_id, confirm):
    """Cancel a pending or running job."""
    from .cli_handlers import handle_cancel_job
    handle_cancel_job(job_id, confirm)


@jobs.command("restart")
@click.argument("job_id")
@click.option("--confirm", is_flag=True, help="Skip confirmation prompt")
def restart_job(job_id, confirm):
    """Restart a failed or cancelled job."""
    from .cli_handlers import handle_restart_job
    handle_restart_job(job_id, confirm)


@jobs.command("stats")
@click.option("--format", "output_format", default="text", type=click.Choice(["text", "json"]), help="Output format")
def queue_stats(output_format):
    """Get comprehensive job queue statistics."""
    from .cli_handlers import handle_queue_stats
    handle_queue_stats(output_format)


@jobs.command("poll")
@click.argument("job_id")
@click.option("--timeout", default=300, type=int, help="Maximum time to poll in seconds (default: 300)")
@click.option("--interval", default=5, type=int, help="Polling interval in seconds (default: 5)")
@click.option("--quiet", is_flag=True, help="Only show final status, suppress progress updates")
def poll_job_status(job_id, timeout, interval, quiet):
    """Poll job status until completion or timeout."""
    from .cli_handlers import handle_poll_job_status
    handle_poll_job_status(job_id, timeout, interval, quiet)


# Worker command group
@cli.group()
def worker():
    """Manage RQ worker processes."""
    pass


@worker.command("start")
@click.option("--redis-url", default="redis://localhost:6379/0", help="Redis URL for job queue")
@click.option("--queue-name", default="extraction_jobs", help="Name of the RQ queue to listen to")
@click.option("--worker-name", help="Unique name for this worker (optional)")
def start_worker(redis_url, queue_name, worker_name):
    """Start a standalone RQ worker process."""
    from .cli_handlers import handle_start_standalone_worker
    handle_start_standalone_worker(redis_url, queue_name, worker_name)


# Dataset version command group
@cli.group()
def dataset_version():
    """Manage dataset versions."""
    pass


# Dataset version subcommands
@dataset_version.command()
@click.option("--project-id", required=True, type=int, help="Project ID")
@click.option("--dataset-name", required=True, help="Name of the dataset")
@click.option("--format", "output_format", default="table", type=click.Choice(["table", "json"]), help="Output format")
def list_versions(project_id, dataset_name, output_format):
    """List all versions of a dataset."""
    from .cli_handlers import handle_list_versions
    handle_list_versions(project_id, dataset_name, output_format)


@dataset_version.command()
@click.option("--project-id", required=True, type=int, help="Project ID")
@click.option("--dataset-name", required=True, help="Name of the dataset")
@click.option("--version1", required=True, help="First version to compare")
@click.option("--version2", required=True, help="Second version to compare")
def compare_versions(project_id, dataset_name, version1, version2):
    """Compare two dataset versions."""
    from .cli_handlers import handle_compare_versions
    handle_compare_versions(project_id, dataset_name, version1, version2)


@dataset_version.command()
@click.option("--project-id", required=True, type=int, help="Project ID")
@click.option("--dataset-name", required=True, help="Name of the dataset")
@click.option("--target-version", required=True, help="Version to rollback to")
@click.option("--confirm", is_flag=True, help="Confirm rollback operation")
def rollback(project_id, dataset_name, target_version, confirm):
    """Rollback dataset to a previous version."""
    from .cli_handlers import handle_rollback_version
    handle_rollback_version(project_id, dataset_name, target_version, confirm)


@dataset_version.command()
@click.option("--project-id", required=True, type=int, help="Project ID")
@click.option("--dataset-name", required=True, help="Name of the dataset")
@click.option("--version-type", default="patch", type=click.Choice(["major", "minor", "patch"]), help="Type of version increment")
@click.option("--description", help="Description of the version change")
def increment_version(project_id, dataset_name, version_type, description):
    """Increment the version of a dataset."""
    from .cli_handlers import handle_increment_version
    handle_increment_version(project_id, dataset_name, version_type, description)


# Extraction command group
@cli.group()
def extraction():
    """Manage content extraction jobs."""
    pass


# Extraction subcommands
@extraction.command("create")
@click.option("--project-id", required=True, type=int, help="Project ID for the extraction")
@click.option("--taxonomy-id", required=True, type=int, help="Taxonomy ID to use for extraction")
@click.option("--selected-categories", required=True, help="Comma-separated list of categories to extract")
@click.option("--effective-classifier", required=True, type=click.Choice(["grok", "gemini", "openai", "ollama"]), help="Unified AI model for both filtering and classification")
@click.option("--enable-validation-stage", is_flag=True, help="Enable validation stage")
@click.option("--validation-classifier", help="Classifier to use for validation (optional)")
@click.option("--confidence-threshold", type=float, default=0.5, help="Minimum confidence score for classifications")
@click.option("--max-chunks", type=int, help="Maximum number of chunks to process")
@click.option("--extraction-type", default="ner", type=click.Choice(["ner", "whole_text"]), help="Type of extraction: 'ner' for named entities, 'whole_text' for complete text portions")
@click.option("--extraction-mode", type=click.Choice(["contextual", "document-wide"]), default="contextual", help="Extraction mode: contextual (default) or document-wide")
def create_extraction(project_id, taxonomy_id, selected_categories, effective_classifier, enable_validation_stage, validation_classifier, confidence_threshold, max_chunks, extraction_type, extraction_mode):
    """Create a new extraction job using unified AI model selection."""
    from .cli_handlers import handle_create_extraction
    handle_create_extraction(project_id, taxonomy_id, selected_categories, effective_classifier, enable_validation_stage, validation_classifier, confidence_threshold, max_chunks, extraction_type, extraction_mode)


# Benchmarking command group
@cli.group()
def benchmark():
    """Run and manage AI model benchmarking operations."""
    pass


# Benchmarking subcommands
@benchmark.command("run")
@click.option("--project-id", required=True, type=int, help="Project ID for benchmarking")
@click.option("--suite", default="glue", type=click.Choice(["glue", "superglue", "mmlu", "medical"]), help="Benchmark suite to run")
@click.option("--provider", default="ollama", type=click.Choice(["ollama", "gemini", "grok", "openai"]), help="AI provider to use")
@click.option("--model", required=True, help="Specific model name to benchmark")
@click.option("--ollama-temperature", type=float, help="Temperature for Ollama models (0.0-2.0)")
@click.option("--ollama-repeat-penalty", type=float, help="Repeat penalty for Ollama models (0.0-2.0)")
@click.option("--ollama-top-p", type=float, help="Top P for Ollama models (0.0-1.0)")
@click.option("--ollama-top-k", type=int, help="Top K for Ollama models (0-100)")
@click.option("--ollama-num-predict", type=int, help="Num predict for Ollama models (1-4096)")
@click.option("--ollama-num-ctx", type=int, help="Context window for Ollama models")
@click.option("--ollama-seed", type=int, help="Seed for Ollama models (0-4294967295)")
@click.option("--custom-config", help="Custom configuration JSON for advanced options")
@click.option("--tasks", help="Comma-separated list of specific tasks to run (optional)")
def run_benchmark(project_id, suite, provider, model, ollama_temperature, ollama_repeat_penalty, ollama_top_p, ollama_top_k, ollama_num_predict, ollama_num_ctx, ollama_seed, custom_config, tasks):
    """Run AI model benchmarks with specified configuration."""
    from .cli_handlers import handle_run_benchmark
    handle_run_benchmark(
        project_id=project_id,
        suite=suite,
        provider=provider,
        model=model,
        ollama_params={
            "temperature": ollama_temperature,
            "repeat_penalty": ollama_repeat_penalty,
            "top_p": ollama_top_p,
            "top_k": ollama_top_k,
            "num_predict": ollama_num_predict,
            "num_ctx": ollama_num_ctx,
            "seed": ollama_seed
        } if provider == "ollama" else None,
        custom_config=custom_config,
        tasks=tasks.split(',') if tasks else None
    )


@benchmark.command("results")
@click.argument("job_id")
@click.option("--format", "output_format", default="table", type=click.Choice(["table", "json"]), help="Output format")
def get_benchmark_results(job_id, output_format):
    """Get results for a specific benchmark job."""
    from .cli_handlers import handle_get_benchmark_results
    handle_get_benchmark_results(job_id, output_format)


@benchmark.command("list")
@click.option("--model", help="Filter by model name")
@click.option("--suite", type=click.Choice(["glue", "superglue", "mmlu", "medical"]), help="Filter by benchmark suite")
@click.option("--status", type=click.Choice(["pending", "running", "completed", "failed"]), help="Filter by job status")
@click.option("--limit", default=20, type=int, help="Maximum number of results to show")
@click.option("--format", "output_format", default="table", type=click.Choice(["table", "json"]), help="Output format")
def list_benchmarks(model, suite, status, limit, output_format):
    """List benchmark jobs with optional filtering."""
    from .cli_handlers import handle_list_benchmarks
    handle_list_benchmarks(model, suite, status, limit, output_format)


@benchmark.command("compare")
@click.argument("model_ids", nargs=-1, required=True)
@click.option("--suite", default="glue", type=click.Choice(["glue", "superglue", "mmlu", "medical"]), help="Benchmark suite to compare on")
@click.option("--metrics", help="Comma-separated list of metrics to compare (optional)")
@click.option("--format", "output_format", default="table", type=click.Choice(["table", "json"]), help="Output format")
def compare_models(model_ids, suite, metrics, output_format):
    """Compare performance across multiple AI models."""
    from .cli_handlers import handle_compare_models
    handle_compare_models(
        model_ids=list(model_ids),
        suite=suite,
        metrics=metrics.split(',') if metrics else None,
        output_format=output_format
    )


@benchmark.command("history")
@click.option("--model", help="Filter by model name")
@click.option("--days", default=30, type=int, help="Number of days of history to show")
@click.option("--format", "output_format", default="table", type=click.Choice(["table", "json"]), help="Output format")
def benchmark_history(model, days, output_format):
    """Get benchmarking history with optional filtering."""
    from .cli_handlers import handle_benchmark_history
    handle_benchmark_history(model, days, output_format)


@benchmark.command("leaderboard")
@click.option("--suite", default="glue", type=click.Choice(["glue", "superglue", "mmlu", "medical"]), help="Benchmark suite for leaderboard")
@click.option("--metric", default="accuracy", type=click.Choice(["accuracy", "f1", "precision", "recall"]), help="Metric to rank by")
@click.option("--limit", default=10, type=int, help="Number of top models to show")
@click.option("--format", "output_format", default="table", type=click.Choice(["table", "json"]), help="Output format")
def benchmark_leaderboard(suite, metric, limit, output_format):
    """Get model leaderboard for a benchmark suite."""
    from .cli_handlers import handle_benchmark_leaderboard
    handle_benchmark_leaderboard(suite, metric, limit, output_format)


@benchmark.command("status")
@click.argument("job_id")
@click.option("--poll", is_flag=True, help="Poll for status changes until completion")
@click.option("--timeout", default=300, type=int, help="Timeout for polling in seconds")
def benchmark_status(job_id, poll, timeout):
    """Check the status of a benchmark job."""
    from .cli_handlers import handle_benchmark_status
    handle_benchmark_status(job_id, poll, timeout)


@benchmark.command("cancel")
@click.argument("job_id")
def benchmark_cancel(job_id):
    """Cancel a running benchmark job."""
    from .cli_handlers import handle_cancel_job
    # Reuse the generic job cancel handler
    handle_cancel_job(job_id, confirm=True)
# Plugin command group
@cli.group()
def plugins():
    """Manage plugins."""
    pass

# Load Plugin Commands
try:
    from src.compileo.features.plugin.manager import plugin_manager
    plugin_commands = plugin_manager.get_extensions("compileo.cli.command")
    for plugin_id, command_obj in plugin_commands.items():
        # Check if it's a click command/group
        if isinstance(command_obj, (click.Command, click.Group)):
            cli.add_command(command_obj)
        else:
            # Maybe it's a function that returns a command
            try:
                cmd = command_obj()
                if isinstance(cmd, (click.Command, click.Group)):
                    cli.add_command(cmd)
            except Exception:
                pass
except Exception as e:
    # Logger might not be configured, just pass or print
    pass


@plugins.command("list")
@click.option("--format", "output_format", default="table", type=click.Choice(["table", "json"]), help="Output format")
def list_plugins(output_format):
    """List all installed plugins."""
    from .cli_handlers import handle_list_plugins
    handle_list_plugins(output_format)


@plugins.command("install")
@click.argument("plugin_file", type=click.Path(exists=True))
def install_plugin(plugin_file):
    """Install a plugin from a .zip file."""
    from .cli_handlers import handle_install_plugin
    handle_install_plugin(plugin_file)


@plugins.command("uninstall")
@click.argument("plugin_id")
@click.option("--confirm", is_flag=True, help="Confirm uninstallation")
def uninstall_plugin(plugin_id, confirm):
    """Uninstall a plugin."""
    from .cli_handlers import handle_uninstall_plugin
    handle_uninstall_plugin(plugin_id, confirm)