import concurrent.futures
from typing import Union, List, Dict, Any, Optional, Callable
from functools import partial
import logging

from .prompt_builder import PromptBuilder
from .output_formatter import OutputFormatter
from .llm_interaction import LLMInteraction
from .evaluation_dataset_generator import EvaluationDatasetGenerator
from .version_manager import DatasetVersionManager
from .data_loader import DataLoader
from .strategies.chunk_strategy import ChunkGenerationStrategy
from .strategies.extraction_strategy import ExtractionGenerationStrategy
from .strategies.instruction_strategy import InstructionGenerationStrategy
from .strategies.custom_strategy import CustomPromptStrategy
from .constants import STRATEGY_METHOD_MAP

from src.compileo.storage.src.project.database_repositories import (
    DocumentRepository, ExtractionResultRepository, ExtractionJobRepository
)
from src.compileo.features.taxonomy.loader import TaxonomyLoader
from src.compileo.features.extraction.filesystem_storage import FilesystemStorageManager

logger = logging.getLogger(__name__)


class DatasetGenerator:
    """Facade class for dataset generation, orchestrating data loading and generation strategies."""

    def __init__(
        self,
        prompt_builder: PromptBuilder,
        output_formatter: OutputFormatter,
        llm_interaction: LLMInteraction,
        document_repository: DocumentRepository,
        extraction_result_repository: Optional[ExtractionResultRepository] = None,
        taxonomy_loader: Optional[TaxonomyLoader] = None,
        version_manager: Optional[DatasetVersionManager] = None,
        filesystem_storage_manager: Optional[FilesystemStorageManager] = None,
        enable_filesystem_storage: bool = True,
        extraction_job_repository: Optional[ExtractionJobRepository] = None,
    ):
        self.prompt_builder = prompt_builder
        self.output_formatter = output_formatter
        self.llm_interaction = llm_interaction
        self.document_repository = document_repository
        self.version_manager = version_manager
        
        # Initialize data loader
        self.data_loader = DataLoader(
            extraction_result_repository=extraction_result_repository,
            extraction_job_repository=extraction_job_repository,
            filesystem_storage_manager=filesystem_storage_manager,
            enable_filesystem_storage=enable_filesystem_storage
        )
        
        # Initialize strategies
        self.instruction_strategy = InstructionGenerationStrategy(prompt_builder, llm_interaction)
        self.chunk_strategy = ChunkGenerationStrategy(prompt_builder, llm_interaction)
        self.extraction_strategy = ExtractionGenerationStrategy(
            prompt_builder, llm_interaction, self.instruction_strategy
        )
        self.custom_prompt_strategy = CustomPromptStrategy(prompt_builder, llm_interaction)
        
        # Strategy dispatcher mapping
        self._strategy_map = {
            "question": (self.chunk_strategy, "generate_questions"),
            "answer": (self.chunk_strategy, "generate_answers"),
            "summarization": (self.chunk_strategy, "generate_summary"),
            "instruction_following": (self.instruction_strategy, "generate_instruction_following"),
            "custom": (self.custom_prompt_strategy, "generate_with_custom_prompt"),
        }

    def generate_dataset(
        self,
        project_id: str,
        owner_id: str,
        chunks: Optional[List[str]] = None,
        prompt_name: str = "qa_pairs",
        format_type: str = "json",
        concurrency: int = 1,
        taxonomy_project: Optional[str] = None,
        taxonomy_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        enable_versioning: bool = False,
        datasets_per_chunk: int = 3,
        prefer_extracted: bool = True,
        only_validated: bool = False,
        custom_prompt: Optional[str] = None,
        extraction_file_id: Optional[str] = None,
        generation_mode: str = "question and answer",
        selected_categories: Optional[List[str]] = None,
        options=None
    ) -> Union[str, bytes, Dict]:
        """Main entry point for generating a dataset."""
        logger.info(f"Generating dataset for project {project_id} (owner: {owner_id}, mode: {generation_mode})")
        
        results = []
        
        # Generate based on data source preference
        if prefer_extracted:
            if not extraction_file_id:
                raise ValueError("Extraction file ID is required when prefer_extracted=True")
            
            results = self._generate_from_extraction(
                project_id, owner_id, extraction_file_id, only_validated,
                selected_categories, prompt_name, datasets_per_chunk,
                custom_prompt, generation_mode, concurrency
            )
        elif chunks:
            results = self._generate_from_chunks(
                project_id, owner_id, chunks, generation_mode, custom_prompt,
                prompt_name, taxonomy_project, taxonomy_name, datasets_per_chunk,
                options, concurrency
            )
        else:
            raise ValueError("Either extraction_file_id (with prefer_extracted=True) or chunks must be provided")

        # Validate results
        if not results:
            error_msg = "No content could be generated from provided sources."
            if extraction_file_id:
                error_msg += f" (Extraction file {extraction_file_id} used)"
            if selected_categories:
                error_msg += f" (Filtered by categories: {selected_categories})"
            raise ValueError(error_msg)

        # Format and save
        formatted = self.output_formatter.format_dataset(results, format_type, owner_id)
        
        # Handle versioning
        if enable_versioning and self.version_manager and dataset_name:
            try:
                self.version_manager.create_new_version(
                    project_id, owner_id, dataset_name, results
                )
            except Exception as e:
                logger.warning(f"Versioning failed: {e}")

        self._trigger_post_generation_hooks(project_id, owner_id, results)
        return formatted

    def _generate_from_extraction(
        self,
        project_id: str,
        owner_id: str,
        extraction_file_id: Optional[str],
        only_validated: bool,
        selected_categories: Optional[List[str]],
        prompt_name: str,
        datasets_per_chunk: int,
        custom_prompt: Optional[str],
        generation_mode: str,
        concurrency: int
    ) -> List[Dict]:
        """Generate dataset from extraction results."""
        # Load extraction data
        logger.info(f"Loading extraction data for job {extraction_file_id}, owner {owner_id}")
        
        if extraction_file_id:
            data, total_count = self.data_loader.load_extracted_data_by_job_id(
                extraction_file_id, owner_id, only_validated=only_validated
            )
            logger.info(f"Loaded {len(data) if data else 0} items (total: {total_count}) from extraction job {extraction_file_id}")
        else:
            data = self.data_loader.load_extracted_data(
                project_id, owner_id, only_validated=only_validated
            )
            logger.info(f"Loaded {len(data) if data else 0} items from project {project_id}")
        
        if not data:
            logger.error("No extraction data loaded - data is None or empty")
            raise ValueError(f"No extraction data found for file {extraction_file_id}")
        
        # Aggregate by category
        logger.info(f"Aggregating {len(data)} items by category (filter: {selected_categories})")
        aggregated_items = self.extraction_strategy.aggregate_entities_by_category(
            data, category_names=selected_categories
        )
        logger.info(f"Aggregation produced {len(aggregated_items)} category groups")
        
        if not aggregated_items:
            if selected_categories:
                logger.error(f"No categories matched filter: {selected_categories}")
                raise ValueError(f"No extraction data found matching categories: {selected_categories}")
            else:
                logger.error("No entities found in extraction data")
                raise ValueError("No entities found in extraction data - check extraction results")
        
        # Generate from aggregated items
        logger.info(f"Generating content from {len(aggregated_items)} aggregated items")
        results = self._run_concurrent_or_sequential(
            aggregated_items,
            lambda item: self.extraction_strategy.generate_content_from_extracted_data(
                project_id, owner_id, prompt_name, item,
                datasets_per_chunk=datasets_per_chunk,
                custom_prompt=custom_prompt,
                generation_mode=generation_mode,
                category_names=selected_categories
            ),
            concurrency
        )
        
        logger.info(f"Generated {len(results)} results from extraction data")
        return results

    def _generate_from_chunks(
        self,
        project_id: str,
        owner_id: str,
        chunks: List[str],
        generation_mode: str,
        custom_prompt: Optional[str],
        prompt_name: str,
        taxonomy_project: Optional[str],
        taxonomy_name: Optional[str],
        datasets_per_chunk: int,
        options: Optional[Dict],
        concurrency: int
    ) -> List[Dict]:
        """Generate dataset from raw text chunks."""
        # Determine strategy and method
        if custom_prompt:
            strategy = self.custom_prompt_strategy
            method_name = "generate_with_custom_prompt"
            kwargs = {
                "custom_prompt": custom_prompt,
                "generation_mode": generation_mode
            }
        elif generation_mode in self._strategy_map:
            strategy, method_name = self._strategy_map[generation_mode]
            kwargs = {}
        else:
            # Default to chunk strategy generate_qa
            strategy = self.chunk_strategy
            method_name = "generate_qa"
            kwargs = {
                "prompt_name": prompt_name,
                "taxonomy_project": taxonomy_project,
                "taxonomy_name": taxonomy_name,
                "options": options
            }
        
        method = getattr(strategy, method_name)
        
        # Build kwargs for each call
        def process_chunk(chunk: str) -> List[Dict]:
            call_kwargs = {
                "project_id": project_id,
                "chunk": chunk,
                "owner_id": owner_id,
                "datasets_per_chunk": datasets_per_chunk,
                **kwargs
            }
            return method(**call_kwargs)
        
        return self._run_concurrent_or_sequential(chunks, process_chunk, concurrency)

    def _run_concurrent_or_sequential(
        self,
        items: List[Any],
        process_func: Callable[[Any], List[Dict]],
        concurrency: int
    ) -> List[Dict]:
        """Execute processing function concurrently or sequentially."""
        results = []
        
        if concurrency > 1 and len(items) > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                for res in executor.map(process_func, items):
                    results.extend(res)
        else:
            for item in items:
                results.extend(process_func(item))
        
        return results

    def generate_evaluation_dataset(
        self,
        project_id: str,
        owner_id: str,
        chunks: List[str],
        prompt_name: str,
        format_type: str,
        concurrency: int = 1,
        datasets_per_chunk: int = 3,
        custom_prompt: Optional[str] = None
    ) -> Union[str, bytes, Dict]:
        """Specialized method for evaluation datasets."""
        from typing import cast
        
        raw_data = self.generate_dataset(
            project_id=project_id,
            owner_id=owner_id,
            chunks=chunks,
            prompt_name=prompt_name,
            format_type="raw_list",
            concurrency=concurrency,
            datasets_per_chunk=datasets_per_chunk,
            custom_prompt=custom_prompt,
            prefer_extracted=False
        )
        
        eval_gen = EvaluationDatasetGenerator()
        eval_data = eval_gen.create_evaluation_datasets(
            dataset=cast(List[Dict], raw_data),
            llm_interaction=self.llm_interaction
        )
        return self.output_formatter.format_dataset(eval_data, format_type, owner_id)

    def _trigger_post_generation_hooks(self, project_id: str, owner_id: str, results: List[Dict]):
        """Trigger optional tasks like auto-benchmarking."""
        try:
            from src.compileo.core.settings import BackendSettings
            from src.compileo.features.jobhandle.enhanced_job_queue import (
                enhanced_job_queue_manager, JobType, JobPriority
            )
            
            settings = BackendSettings(owner_id=owner_id)
            if settings.get_setting(f"project_{project_id}_auto_benchmark", False):
                if enhanced_job_queue_manager:
                    params = {
                        "project_id": project_id,
                        "benchmark_suite": "glue",
                        "ai_config": {"selected_model": "mistral:latest", "provider": "ollama"},
                        "benchmark_params": {"auto_generated": True}
                    }
                    enhanced_job_queue_manager.submit_job(
                        JobType.BENCHMARKING,
                        params,
                        JobPriority.NORMAL,
                        user_id=owner_id
                    )
        except Exception as e:
            logger.warning(f"Post-generation hooks failed: {e}")
