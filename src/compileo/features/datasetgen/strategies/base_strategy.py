from typing import List, Dict, Any, Union, Optional
from ..prompt_builder import PromptBuilder
from ..llm_interaction import LLMInteraction


class ResultHandlerMixin:
    """Mixin for standardized result handling across generation strategies.
    
    This mixin eliminates code duplication in result processing by providing
    a common method to handle LLM responses (both single items and lists).
    """
    
    def _process_result(
        self,
        result: Union[Dict[str, Any], List[Dict[str, Any]]],
        source_type: str,
        max_items: int,
        **metadata
    ) -> List[Dict[str, Any]]:
        """Process LLM result into standardized format.
        
        Args:
            result: LLM response (dict or list of dicts)
            source_type: Type identifier for the source (e.g., 'chunk', 'extraction')
            max_items: Maximum number of items to return (0 = AI-determined, no limit)
            **metadata: Additional metadata to add to each item
            
        Returns:
            List of processed result dictionaries
        """
        results = []
        
        # Normalize to list
        items = result if isinstance(result, list) else [result]
        
        # For AI-determined mode (max_items == 0), process all items without limit
        items_to_process = items if max_items == 0 else items[:max_items]
        
        for item in items_to_process:
            if not isinstance(item, dict):
                continue
                
            item_copy = item.copy()
            item_copy["source_type"] = source_type
            # Add ai_determined flag when max_items is 0
            if max_items == 0:
                item_copy["ai_determined"] = True
            item_copy.update(metadata)
            results.append(item_copy)
        
        # Only apply limit for fixed-count mode (max_items > 0)
        return results if max_items == 0 else results[:max_items]


class BaseGenerationStrategy(ResultHandlerMixin):
    """Abstract base class for dataset generation strategies.
    
    All generation strategies should inherit from this class and implement
    the generate() method. The class includes ResultHandlerMixin for
    standardized result processing.
    """
    
    def __init__(self, prompt_builder: PromptBuilder, llm_interaction: LLMInteraction):
        self.prompt_builder = prompt_builder
        self.llm_interaction = llm_interaction

    def generate(self, *args, **kwargs) -> List[Dict[str, Any]]:
        """Main generation method to be implemented by subclasses.
        
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement generate()")
