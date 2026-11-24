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

def parse_chunking_schema(value: str) -> dict:
    """Validate a splitting schema and report errors against the offending rule."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Chunking schema JSON cannot be empty")
    try:
        schema = json.loads(value)
    except json.JSONDecodeError as exc:
        hint = " Use double backslashes for regex escapes in JSON." if "escape" in exc.msg else ""
        raise ValueError(f"Invalid chunking schema JSON: {exc}.{hint}") from exc

    if not isinstance(schema, dict) or not isinstance(schema.get("rules"), list) or not schema["rules"]:
        raise ValueError("Chunking schema must have a nonempty 'rules' array")
    for index, rule in enumerate(schema["rules"], 1):
        if not isinstance(rule, dict) or rule.get("type") not in ("pattern", "delimiter"):
            raise ValueError(f"Rule {index}: 'type' must be 'pattern' or 'delimiter'")
        rule_value = rule.get("value")
        if not isinstance(rule_value, str) or not rule_value:
            raise ValueError(f"Rule {index}: 'value' must be a nonempty string")
        if rule["type"] == "pattern":
            try:
                re.compile(rule_value, re.MULTILINE)
            except (re.error, OverflowError, RecursionError) as exc:
                raise ValueError(f"Rule {index}: invalid regex: {exc}") from exc
    if schema.get("combine") not in ("any", "all"):
        raise ValueError("Schema 'combine' must be 'any' or 'all'")
    if "include_pattern" in schema and not isinstance(schema["include_pattern"], bool):
        raise ValueError("Schema 'include_pattern' must be a boolean")
    return schema


class SchemaStrategy(BaseModel):
    """
    Defines the schema for a JSON schema-based chunking strategy.
    """
    strategy: Literal['schema'] = 'schema'
    json_schema: str = Field(..., description="JSON splitting rules. 'any' unions split positions; 'all' intersects them. 'include_pattern' cuts before regex matches when true and after them when false. Neither setting discards matched text.")

    @field_validator('json_schema')
    @classmethod
    def validate_json_schema(cls, v):
        parse_chunking_schema(v)
        return v


ChunkingStrategy = LLMPromptStrategy | DelimiterStrategy | CharacterStrategy | TokenStrategy | SchemaStrategy
