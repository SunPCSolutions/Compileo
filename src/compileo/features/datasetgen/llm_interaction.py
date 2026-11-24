from typing import Dict, Any, Optional, Union, List
import ollama
import google.genai as genai
from google.genai import types
import requests
import json
import openai
import logging
from datetime import datetime, timezone

class LLMInteraction:
    """
    Handles interaction with different LLMs.
    """

    def __init__(self, llm_provider: str, api_key: Optional[str] = None, model: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        """
        Initializes the LLMInteraction.

        Args:
            llm_provider: The LLM provider to use ('ollama', 'gemini', 'grok', or 'openai').
            api_key: The API key for the LLM provider.
            model: Specific model to use. If None, uses settings defaults.
        """
        self.llm_provider = llm_provider
        self.model = model  # Store the model parameter
        self.api_key = api_key  # Store API key for all providers
        self._gemini_client = None

        if llm_provider == 'gemini':
            if not api_key:
                raise ValueError("API key required for Gemini provider")
        elif llm_provider == 'grok':
            if not api_key:
                raise ValueError("API key required for Grok provider")
            if not model:
                raise ValueError("model is required for grok provider")
            self.grok_api_key = api_key
            self.grok_api_url = "https://api.x.ai/v1/chat/completions"
            self.grok_model = model
        elif llm_provider == 'openai':
            if not api_key:
                raise ValueError("API key required for OpenAI provider")
            if not model:
                raise ValueError("model is required for openai provider")
            self.openai_client = openai.OpenAI(api_key=api_key)
            self.openai_model = model
        elif llm_provider == 'ollama':
            if not model:
                raise ValueError("model is required for ollama provider")
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")

    def generate(self, prompt: str, options: Optional[Dict[str, Any]] = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Generates content using the selected LLM.

        Args:
            prompt: The prompt to send to the LLM.
            options: Optional Ollama API options to override defaults

        Returns:
            A dictionary or list of dictionaries containing the generated response with reasoning metadata.
        """
        timestamp = datetime.now(timezone.utc).isoformat() + "Z"

        if self.llm_provider == 'ollama':
            from ...core.settings import backend_settings
            # Model is already validated in __init__, use it directly
            ollama_model = self.model

            # Default options from settings
            default_options = {
                "temperature": backend_settings.get_generation_ollama_temperature(),
                "repeat_penalty": backend_settings.get_generation_ollama_repeat_penalty(),
                "top_p": backend_settings.get_generation_ollama_top_p(),
                "top_k": backend_settings.get_generation_ollama_top_k(),
                "num_predict": backend_settings.get_generation_ollama_num_predict(),
                "num_ctx": 32768  # Keep num_ctx as default, can be overridden
            }

            # Add seed if set
            seed = backend_settings.get_generation_ollama_seed()
            if seed is not None:
                default_options["seed"] = seed

            # Merge provided options with defaults
            merged_options = {**default_options, **(options or {})}

            # Check if prompt expects an array response
            prompt_lower = prompt.lower()
            expects_array = (
                "json array" in prompt_lower
                or "question-answer pairs" in prompt_lower
                or "instruction-response pairs" in prompt_lower
                or "array of objects" in prompt_lower
                or "array of instruction" in prompt_lower
            )

            # Define the response schema for a single QA item
            qa_item_schema = {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "answer": {"type": "string"},
                    "reasoning": {"type": "string"},
                    "confidence_score": {"type": "number"},
                    "reasoning_steps": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["question", "answer", "reasoning", "confidence_score", "reasoning_steps"]
            }

            if expects_array:
                # For array responses, wrap in an object with a field name (Ollama best practice)
                ollama_schema = {
                    "type": "object",
                    "properties": {
                        "qa_pairs": {
                            "type": "array",
                            "items": qa_item_schema
                        }
                    },
                    "required": ["qa_pairs"]
                }
                # Don't add schema to prompt text - the format parameter handles it
                final_prompt = prompt
            else:
                # Single object response
                ollama_schema = qa_item_schema
                # Simple field description instead of full schema
                final_prompt = f"{prompt}\n\nOutput MUST be a single JSON object with these fields: question, answer, reasoning, confidence_score (0-1), reasoning_steps (array of strings)."

            request_data = {
                "model": ollama_model,
                "messages": [{'role': 'user', 'content': final_prompt}],
                "stream": False,
                "format": ollama_schema,
                "options": merged_options
            }

            try:
                response = ollama.chat(**request_data)
                content = response.get('message', {}).get('content', '')
                
                # Fallback: If empty, retry without schema (some older Ollama versions/models struggle with schemas)
                if not content:
                    import logging
                    logging.warning(f"Ollama returned empty content with schema, retrying with format='json'...")
                    request_data["format"] = "json"
                    request_data["messages"][0]["content"] = prompt # Original prompt
                    response = ollama.chat(**request_data)
                    content = response.get('message', {}).get('content', '')

                # DEBUG: Log raw content from Ollama
                self.logger.info(f"Ollama raw response length: {len(content)}")
                if not content:
                    self.logger.warning(f"Ollama returned empty content for model {ollama_model}")
                return self._parse_response(content, ollama_model, timestamp)
            except Exception as e:
                self.logger.error(f"Ollama chat failed: {e}")
                raise
        elif self.llm_provider == 'gemini':
            # Model must be provided - no fallback to settings
            if not self.model:
                raise ValueError("model is required for gemini provider")
            gemini_model = self.model

            # Use cached client or create new one
            if self._gemini_client is None:
                # Set environment variable for Google GenAI
                import os
                if self.api_key:
                    os.environ['GOOGLE_API_KEY'] = self.api_key
                self._gemini_client = genai.Client(api_key=self.api_key)

            try:
                # Use structured output for Gemini to ensure JSON consistency
                # Define schema for a single QA item
                qa_item_schema = {
                    "type": "OBJECT",
                    "properties": {
                        "question": {"type": "STRING"},
                        "answer": {"type": "STRING"},
                        "reasoning": {"type": "STRING"},
                        "confidence_score": {"type": "NUMBER"},
                        "reasoning_steps": {
                            "type": "ARRAY",
                            "items": {"type": "STRING"}
                        }
                    },
                    "required": ["question", "answer", "reasoning", "confidence_score", "reasoning_steps"]
                }

                # Check if we expect a list (based on prompt instructions)
                prompt_lower = prompt.lower()
                if (
                    "json array" in prompt_lower
                    or "instruction-response pairs" in prompt_lower
                    or "array of objects" in prompt_lower
                    or "array of instruction" in prompt_lower
                ):
                    response_schema = {
                        "type": "ARRAY",
                        "items": qa_item_schema
                    }
                else:
                    response_schema = qa_item_schema

                response = self._gemini_client.models.generate_content(
                    model=gemini_model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=response_schema
                    )
                )
                content = response.text or ""
                return self._parse_response(content, gemini_model, timestamp)
            except Exception as e:
                # Check if the error is about client being closed
                if "client has been closed" in str(e).lower():
                    self.logger.debug(f"Gemini client was closed, recreating client and retrying...")
                    # Recreate the client and retry once
                    self._gemini_client = genai.Client(api_key=self.api_key)
                    response = self._gemini_client.models.generate_content(
                        model=gemini_model,
                        contents=prompt
                    )
                    content = response.text or ""
                    return self._parse_response(content, gemini_model, timestamp)
                else:
                    # Re-raise other exceptions
                    raise
        elif self.llm_provider == 'grok':
            content = self._call_grok_api(prompt)
            return self._parse_response(content, self.grok_model, timestamp)
        elif self.llm_provider == 'openai':
            content = self._call_openai_api(prompt)
            return self._parse_response(content, self.openai_model, timestamp)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")

    def _parse_response(self, content: str, model: str, timestamp: str, include_raw: bool = False) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Parse the LLM response, attempting to extract reasoning metadata from JSON.
        Handles thinking models that include <think> tags.

        Args:
            content: Raw response content from LLM
            model: Model name used
            timestamp: Generation timestamp

        Returns:
            Parsed response dictionary with reasoning metadata
        """
        # Extract thinking content if present (for models like Ollama with thinking)
        thinking_content = ""
        json_content = content

        # Check for <think> tags (Ollama thinking format)
        if "<think>" in content and "</think>" in content:
            import re
            think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
            if think_match:
                thinking_content = think_match.group(1).strip()
                # Remove thinking tags from JSON content
                json_content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

        # Try to parse as JSON
        try:
            # Clean up the content - remove any leading/trailing whitespace and potential markdown
            json_content = json_content.strip()
            if json_content.startswith('```json'):
                json_content = json_content[7:]
            if json_content.endswith('```'):
                json_content = json_content[:-3]
            json_content = json_content.strip()

            if not json_content:
                self.logger.debug(f"DEBUG: Empty JSON content from LLM")
                raise ValueError("Empty JSON content")

            parsed = json.loads(json_content)
            if isinstance(parsed, list):
                # Handle raw JSON array - model returned multiple items as array
                self.logger.debug(f"DEBUG: Successfully parsed JSON array with {len(parsed)} items")
                results = []
                for item in parsed:
                    if isinstance(item, dict):
                        result_item = {
                            "question": item.get("question", item.get("instruction", "")),
                            "answer": item.get("answer", item.get("output", item.get("summary", ""))),
                            "thinking": thinking_content,
                            "reasoning": item.get("reasoning", ""),
                            "confidence_score": item.get("confidence_score", 0.8),
                            "reasoning_steps": item.get("reasoning_steps", []),
                            "generation_model": model,
                            "generation_timestamp": timestamp
                        }
                        if include_raw:
                            result_item["raw_response"] = content
                        results.append(result_item)
                if results:
                    return results
                # If no valid items, fall through to dict handling
                parsed = {"items": parsed}  # Wrap for fallback handling
            
            if isinstance(parsed, dict):
                # Check for common nested list patterns (e.g., {"qa_pairs": [...]}, {"qa": [...]})
                # This handles cases where models return multiple pairs in one call despite instructions
                for key in ["qa_pairs", "qa", "pairs", "results", "entries", "data", "items"]:
                    if key in parsed and isinstance(parsed[key], list):
                        self.logger.info(f"Found nested list in key '{key}' with {len(parsed[key])} items")
                        results = []
                        for item in parsed[key]:
                            if isinstance(item, dict):
                                result_item = {
                                    "question": item.get("question", item.get("instruction", "")),
                                    "answer": item.get("answer", item.get("output", item.get("summary", ""))),
                                    "thinking": thinking_content,
                                    "reasoning": item.get("reasoning", ""),
                                    "confidence_score": item.get("confidence_score", 0.8),
                                    "reasoning_steps": item.get("reasoning_steps", []),
                                    "generation_model": model,
                                    "generation_timestamp": timestamp
                                }
                                if include_raw:
                                    result_item["raw_response"] = content
                                results.append(result_item)
                        if results:
                            return results

                # Extract fields with defaults
                result = {
                    "question": parsed.get("question", parsed.get("instruction", "")),
                    "answer": parsed.get("answer", parsed.get("output", parsed.get("summary", json_content))),
                    "thinking": thinking_content,  # Add thinking field for thinking models
                    "reasoning": parsed.get("reasoning", ""),
                    "confidence_score": parsed.get("confidence_score", 0.5),
                    "reasoning_steps": parsed.get("reasoning_steps", []),
                    "generation_model": model,
                    "generation_timestamp": timestamp
                }
                if include_raw:
                    result["raw_response"] = content
                return result
            elif isinstance(parsed, list):
                # Return list directly if requested
                self.logger.debug(f"DEBUG: Successfully parsed JSON array with {len(parsed)} items")
                # Add metadata to each item
                for item in parsed:
                    if isinstance(item, dict):
                        item.setdefault("generation_model", model)
                        item.setdefault("generation_timestamp", timestamp)
                return parsed
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            # DEBUG: Log full content for analysis
            self.logger.error(f"JSON parsing failed: {e}")
            self.logger.debug(f"FULL LLM CONTENT: {content}")
            self.logger.debug(f"JSON parsing failed: {e}, content: {json_content[:200]}...")

        # Fallback: treat as plain text answer
        result = {
            "question": "",
            "answer": json_content,
            "thinking": thinking_content,  # Always include thinking field
            "reasoning": "",
            "confidence_score": 0.5,
            "reasoning_steps": [],
            "generation_model": model,
            "generation_timestamp": timestamp
        }
        if include_raw:
            result["raw_response"] = content
        return result

    def _call_grok_api(self, prompt: str) -> str:
        """
        Call Grok API for content generation with structured output.

        Args:
            prompt: The prompt to send.

        Returns:
            Generated content string.
        """
        # Check if prompt expects an array response
        prompt_lower = prompt.lower()
        expects_array = (
            "json array" in prompt_lower
            or "question-answer pairs" in prompt_lower
            or "instruction-response pairs" in prompt_lower
            or "array of objects" in prompt_lower
            or "array of instruction" in prompt_lower
        )
        
        # Use OpenAI-compatible structured output format for Grok
        request_data = {
            "model": self.grok_model,
            "messages": [{"role": "user", "content": prompt}]
            # Note: max_tokens and temperature use API defaults per user request
        }
        
        # Use Grok's structured outputs via json_schema
        if not expects_array:
            request_data["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "qa_response",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "The generated question"
                            },
                            "answer": {
                                "type": "string",
                                "description": "The answer to the question"
                            },
                            "reasoning": {
                                "type": "string",
                                "description": "Brief explanation of the reasoning process"
                            },
                            "confidence_score": {
                                "type": "number",
                                "description": "Confidence score between 0.0 and 1.0"
                            },
                            "reasoning_steps": {
                                "type": "array",
                                "items": {
                                    "type": "string"
                                },
                                "description": "Array of reasoning steps"
                            }
                        },
                        "required": ["question", "answer", "reasoning", "confidence_score", "reasoning_steps"],
                        "additionalProperties": False
                    }
                }
            }

        headers = {
            "Authorization": f"Bearer {self.grok_api_key}",
            "Content-Type": "application/json"
        }

        try:
            response = requests.post(
                self.grok_api_url,
                json=request_data,
                headers=headers,
                timeout=60
            )
            if not response.ok:
                error_body = response.text
                raise Exception(f"Grok API request failed with status {response.status_code}: {error_body}")
            
            response.raise_for_status()

            response_json = response.json()
            content = response_json.get("choices", [{}])[0].get("message", {}).get("content", "")

            if not content:
                raise ValueError("Empty response from Grok API")

            return content

        except Exception as e:
            raise Exception(f"Grok API request failed: {e}")

    def _call_openai_api(self, prompt: str) -> str:
        """
        Call OpenAI API for content generation with structured output.

        Args:
            prompt: The prompt to send.

        Returns:
            Generated content string.
        """
        try:
            # Check if prompt expects an array response
            prompt_lower = prompt.lower()
            expects_array = (
                "json array" in prompt_lower
                or "question-answer pairs" in prompt_lower
                or "instruction-response pairs" in prompt_lower
                or "array of objects" in prompt_lower
                or "array of instruction" in prompt_lower
            )
            
            request_kwargs = {
                "model": self.openai_model,
                "messages": [{"role": "user", "content": prompt}]
                # Note: max_tokens and temperature use API defaults per user request
            }
            
            # Only use structured output for single items; arrays are handled via prompt instructions
            if not expects_array:
                request_kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "qa_response",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "question": {
                                    "type": "string",
                                    "description": "The generated question"
                                },
                                "answer": {
                                    "type": "string",
                                    "description": "The answer to the question"
                                },
                                "reasoning": {
                                    "type": "string",
                                    "description": "Brief explanation of the reasoning process"
                                },
                                "confidence_score": {
                                    "type": "number",
                                    "minimum": 0.0,
                                    "maximum": 1.0,
                                    "description": "Confidence score between 0.0 and 1.0"
                                },
                                "reasoning_steps": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    },
                                    "description": "Array of reasoning steps"
                                }
                            },
                            "required": ["question", "answer", "reasoning", "confidence_score", "reasoning_steps"],
                            "additionalProperties": False
                        },
                        "strict": True
                    }
                }
            
            response = self.openai_client.chat.completions.create(**request_kwargs)

            content = response.choices[0].message.content

            if not content:
                raise ValueError("Empty response from OpenAI API")

            return content

        except Exception as e:
            raise Exception(f"OpenAI API request failed: {e}")
