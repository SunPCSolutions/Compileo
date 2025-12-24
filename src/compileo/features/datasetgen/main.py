import os
from dotenv import load_dotenv
from .generator import DatasetGenerator
from .prompt_builder import PromptBuilder
from .output_formatter import OutputFormatter
from .llm_interaction import LLMInteraction
from src.compileo.storage.src.project.database_repositories import (
    PromptRepository,
    DatasetParameterRepository,
    DocumentRepository,
    ExtractionResultRepository,
)
from src.compileo.storage.src.database import get_db_connection


def main():
    """
    Main function to generate a dataset.
    """
    load_dotenv()
    api_key = os.getenv("API_KEY")

    # Establish database connection
    db_connection = get_db_connection()

    # Initialize repositories
    prompt_repo = PromptRepository(db_connection)
    dataset_param_repo = DatasetParameterRepository(db_connection)
    document_repo = DocumentRepository(db_connection)
    extraction_result_repo = ExtractionResultRepository(db_connection)

    # Initialize core components
    prompt_builder = PromptBuilder(prompt_repo, dataset_param_repo)
    output_formatter = OutputFormatter()

    if not api_key:
        raise ValueError("API_KEY not found in environment variables.")

    from src.compileo.core.settings import backend_settings
    gemini_model = backend_settings.get_generation_gemini_model()
    llm_interaction = LLMInteraction(llm_provider="gemini", api_key=api_key, model=gemini_model)

    # Initialize the dataset generator
    dataset_generator = DatasetGenerator(
        prompt_builder=prompt_builder,
        output_formatter=output_formatter,
        llm_interaction=llm_interaction,
        document_repository=document_repo,
        extraction_result_repository=extraction_result_repo,
    )

    # Generate the dataset
    project_id = 1  # Example project ID
    prompt_name = "example_prompt"  # Example prompt name
    format_type = "jsonl"  # Example format type
    concurrency = 4 # Example concurrency

    formatted_dataset = dataset_generator.generate_dataset(
        project_id=project_id,
        prompt_name=prompt_name,
        format_type=format_type,
        concurrency=concurrency,
    )

    # Print or save the dataset
    print(formatted_dataset)

if __name__ == "__main__":
    main()