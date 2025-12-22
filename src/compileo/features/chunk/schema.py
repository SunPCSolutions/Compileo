from pydantic import BaseModel, Field, field_validator
from typing import Literal, Optional, Dict, Any, List, Union
import json
import re
import tiktoken

class ManifestData(BaseModel):
    """
    Defines the schema for manifest data used in multi-file document chunking.
    """
    total_pages: Optional[int] = Field(default=None, description="Total number of pages in the document.")
    pages_per_split: Optional[int] = Field(default=None, description="Pages per file split.")
    overlap_pages: Optional[int] = Field(default=None, description="Overlap pages between splits.")
    splits: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="List of file splits.")

class LLMPromptStrategy(BaseModel):
    """
    Defines the schema for an LLM-mediated chunking strategy.
    """
    strategy: Literal['llm_prompt'] = 'llm_prompt'
    model: str = Field(..., description="The identifier for the LLM to be used (e.g., 'gemini-1.5-pro').")
    prompt_template: str = Field(..., description="A template for the prompt, with placeholders for user instructions.")
    user_instruction: str = Field(..., description="A user-defined instruction that specifies the chunking logic.")
    options: Optional[Dict[str, Any]] = Field(default=None, description="Additional options for the LLM, such as num_ctx for Ollama.")
    system_instruction: Optional[str] = Field(default=None, description="System-level instructions to guide the model's behavior, especially for Gemini.")

    # Model validation removed - models come from validated GUI settings

class DelimiterStrategy(BaseModel):
    """
    Defines the schema for a simple delimiter-based chunking strategy.
    """
    strategy: Literal['delimiter'] = 'delimiter'
    delimiter: str = Field(..., description="The string to split the text by.")

    @field_validator('delimiter')
    @classmethod
    def validate_delimiter(cls, v):
        if not v or not v.strip():
            raise ValueError("Delimiter cannot be empty or whitespace-only")
        return v

class CharacterStrategy(BaseModel):
    """
    Defines the schema for a character-based chunking strategy.
    """
    strategy: Literal['character'] = 'character'
    chunk_size: int = Field(..., description="The maximum number of characters per chunk.")
    overlap: int = Field(..., description="The number of characters to overlap between consecutive chunks.")

    @field_validator('chunk_size')
    @classmethod
    def validate_chunk_size(cls, v):
        if v <= 0:
            raise ValueError("chunk_size must be greater than 0")
        return v

    @field_validator('overlap')
    @classmethod
    def validate_overlap(cls, v):
        if v < 0:
            raise ValueError("overlap must be greater than or equal to 0")
        return v

class TokenStrategy(BaseModel):
    """
    Defines the schema for a token-based chunking strategy.
    """
    strategy: Literal['token'] = 'token'
    chunk_size: int = Field(..., description="The maximum number of tokens per chunk.")
    overlap: int = Field(..., description="The number of tokens to overlap between consecutive chunks.")
    model: str = Field(default="cl100k_base", description="The tokenizer model to use (e.g., 'cl100k_base' for GPT models).")

    @field_validator('chunk_size')
    @classmethod
    def validate_chunk_size(cls, v):
        if v <= 0:
            raise ValueError("chunk_size must be greater than 0")
        return v

    @field_validator('overlap')
    @classmethod
    def validate_overlap(cls, v):
        if v < 0:
            raise ValueError("overlap must be greater than or equal to 0")
        return v

    @field_validator('model')
    @classmethod
    def validate_model(cls, v):
        try:
            tiktoken.get_encoding(v)
        except KeyError:
            # Try fallback encodings
            fallback_encodings = ["cl100k_base", "p50k_base", "r50k_base"]
            if v not in fallback_encodings:
                raise ValueError(f"Unknown tiktoken model '{v}'. Supported models: {', '.join(fallback_encodings)}")
        return v

class SchemaStrategy(BaseModel):
    """
    Defines the schema for a JSON schema-based chunking strategy.
    """
    strategy: Literal['schema'] = 'schema'
    json_schema: str = Field(..., description="A JSON schema string that defines flexible document structure patterns for splitting, combining multiple splitting criteria like patterns, delimiters, and structural elements. Supports 'include_pattern' flag to control whether matched patterns are included in chunks.")

    @field_validator('json_schema')
    @classmethod
    def validate_json_schema(cls, v):
        """Validate JSON schema string. Supports both pattern and delimiter rules."""
        if not v or not v.strip():
            raise ValueError("JSON schema cannot be empty")

        try:
            schema = json.loads(v)
            # Validate basic structure
            if not isinstance(schema, dict) or "rules" not in schema:
                raise ValueError("JSON schema must be an object with a 'rules' array")

            # Validate that rules are properly structured
            for rule in schema["rules"]:
                if not isinstance(rule, dict) or "type" not in rule or "value" not in rule:
                    raise ValueError("Each rule must have 'type' and 'value' fields")

                rule_type = rule.get("type")
                if rule_type not in ["pattern", "delimiter"]:
                    raise ValueError("Rule type must be 'pattern' or 'delimiter'")

            # Validate combine field
            if "combine" not in schema or schema["combine"] not in ["any", "all"]:
                raise ValueError("Schema must have a 'combine' field with value 'any' or 'all'")

            # Validate include_pattern field (optional, defaults to False)
            if "include_pattern" in schema and not isinstance(schema["include_pattern"], bool):
                raise ValueError("Schema 'include_pattern' field must be a boolean")

            return v
        except json.JSONDecodeError as e:
            error_msg = str(e)
            if "Invalid \\escape" in error_msg:
                raise ValueError(
                    "JSON schema contains corrupted backslash escapes in regex patterns. "
                    "This commonly happens when copying AI-recommended schemas into the GUI. "
                    "Try: 1) Re-paste the JSON from AI recommendations, or 2) Use double backslashes (\\\\) for regex patterns like \\s, \\n, etc., or 3) The GUI should auto-fix this - check for the 'Auto-fixed' message."
                )
            else:
                raise ValueError(f"Invalid JSON schema: {error_msg}")


ChunkingStrategy = LLMPromptStrategy | DelimiterStrategy | CharacterStrategy | TokenStrategy | SchemaStrategy