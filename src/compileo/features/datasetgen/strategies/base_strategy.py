from typing import Any, Dict, List, Optional
from ..prompt_builder import PromptBuilder
from ..llm_interaction import LLMInteraction

class BaseGenerationStrategy:
    """
    Base class for dataset generation strategies.
    """
    def __init__(self, prompt_builder: PromptBuilder, llm_interaction: LLMInteraction):
        self.prompt_builder = prompt_builder
        self.llm_interaction = llm_interaction

    def generate(self, *args, **kwargs) -> List[Dict[str, Any]]:
        raise NotImplementedError("Subclasses must implement generate method.")
