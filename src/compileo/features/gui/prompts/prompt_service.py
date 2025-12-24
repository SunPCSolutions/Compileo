from src.compileo.storage.src.project.database_repositories import PromptRepository

class PromptService:
    """
    Service for handling prompt-related business logic.
    """

    def __init__(self, db_connection):
        """
        Initializes the PromptService.

        Args:
            db_connection: The database connection object.
        """
        self.prompt_repository = PromptRepository(db_connection)

    def create_prompt(self, name: str, content: str) -> int:
        """
        Creates a new prompt.

        Args:
            name (str): The name of the prompt.
            content (str): The content of the prompt.

        Returns:
            int: The ID of the newly created prompt.
        """
        return self.prompt_repository.create(name, content)

    def get_prompt_by_name(self, name: str):
        """
        Retrieves a prompt by its name.

        Args:
            name (str): The name of the prompt to retrieve.

        Returns:
            The prompt object, or None if not found.
        """
        return self.prompt_repository.get_by_name(name)

    def update_prompt(self, prompt_id: int, name: str, content: str):
        """
        Updates a prompt.

        Args:
            prompt_id (int): The ID of the prompt to update.
            name (str): The new name of the prompt.
            content (str): The new content of the prompt.
        """
        self.prompt_repository.update(prompt_id, name, content)

    def delete_prompt(self, prompt_id: int):
        """
        Deletes a prompt.

        Args:
            prompt_id (int): The ID of the prompt to delete.
        """
        self.prompt_repository.delete(prompt_id)