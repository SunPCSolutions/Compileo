from typing import Dict, Any, Optional
import ollama
from google import genai
import requests
import json
import openai
from datetime import datetime, timezone
from ...core.logging import get_logger

logger = get_logger(__name__)

class LLMInteraction:
    """
    Handles interaction with different LLMs.
    """

    def __init__(self, llm_provider: str, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initializes the LLMInteraction.

        Args:
            llm_provider: The LLM provider to use ('ollama', 'gemini', 'grok', or 'openai').
            api_key: The API key for the LLM provider.
            model: Specific model to use. If None, uses settings defaults.
        """
        from ...core.settings import backend_settings

        self.llm_provider = llm_provider
        self.model = model  # Store the model parameter
        self.api_key = api_key
        self._gemini_client = None

        if llm_provider == 'gemini':
            if not api_key:
                raise ValueError("API key required for Gemini provider")
            # API key will be handled by the client
        elif llm_provider == 'grok':
            if not api_key:
                raise ValueError("API key required for Grok provider")
            self.grok_api_key = api_key
            self.grok_api_url = "https://api.x.ai/v1/chat/completions"
            # Use provided model or settings default
            self.grok_model = model or backend_settings.get_generation_grok_model()
        elif llm_provider == 'openai':
            if not api_key:
                raise ValueError("API key required for OpenAI provider")
            self.openai_client = openai.OpenAI(api_key=api_key)
            # Use provided model or settings default
            self.openai_model = model or backend_settings.get_generation_openai_model()
        elif llm_provider == 'ollama':
            pass  # No API key needed for Ollama
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")

    def generate(self, prompt: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generates content using the selected LLM.

        Args:
            prompt: The prompt to send to the LLM.
            options: Optional Ollama API options to override defaults

        Returns:
            A dictionary containing the generated response with reasoning metadata.
        """
        timestamp = datetime.now(timezone.utc).isoformat() + "Z"

        if self.llm_provider == 'ollama':
            from ...core.settings import backend_settings
            ollama_model = self.model or backend_settings.get_generation_ollama_model()

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

            request_data = {
                "model": ollama_model,
                "messages": [{'role': 'user', 'content': prompt}],
                "stream": False,
                "options": merged_options
            }

            response = ollama.chat(**request_data)
            content = response['message']['content']
            return self._parse_response(content, ollama_model, timestamp)
        elif self.llm_provider == 'gemini':
            from ...core.settings import backend_settings
            gemini_model = self.model or backend_settings.get_generation_gemini_model()

            # Use cached client or create new one
            if self._gemini_client is None:
                # Set environment variable for Google GenAI
                import os
                if self.api_key:
                    os.environ['GOOGLE_API_KEY'] = self.api_key
                self._gemini_client = genai.Client(api_key=self.api_key)

            try:
                response = self._gemini_client.models.generate_content(
                    model=gemini_model,
                    contents=prompt
                )
                content = response.text
                return self._parse_response(content, gemini_model, timestamp)
            except Exception as e:
                # Check if the error is about client being closed
                if "client has been closed" in str(e).lower():
                    logger.warning(f"Gemini client was closed, recreating client and retrying...")
                    # Recreate the client and retry once
                    self._gemini_client = genai.Client(api_key=self.api_key)
                    response = self._gemini_client.models.generate_content(
                        model=gemini_model,
                        contents=prompt
                    )
                    content = response.text
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

    def _parse_response(self, content: str, model: str, timestamp: str) -> Dict[str, Any]:
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

            parsed = json.loads(json_content)
            if isinstance(parsed, dict):
                # Extract fields with defaults
                res = {
                    "instruction": parsed.get("instruction", ""),
                    "input": parsed.get("input", ""),
                    "output": parsed.get("output", ""),
                    "question": parsed.get("question", ""),
                    "answer": parsed.get("answer", ""),
                    "thinking": thinking_content,  # Add thinking field for thinking models
                    "reasoning": parsed.get("reasoning", ""),
                    "confidence_score": parsed.get("confidence_score", 0.5),
                    "reasoning_steps": parsed.get("reasoning_steps", []),
                    "generation_model": model,
                    "generation_timestamp": timestamp,
                    "raw_response": content
                }
                
                # Support plural/specialized schemas
                if "questions" in parsed:
                    res["questions"] = parsed["questions"]
                if "answers" in parsed:
                    res["answers"] = parsed["answers"]
                if "summary" in parsed:
                    res["summary"] = parsed["summary"]
                if "key_points" in parsed:
                    res["key_points"] = parsed["key_points"]
                    
                return res
        except (json.JSONDecodeError, TypeError) as e:
            logger.debug(f"JSON parsing failed: {e}, content: {json_content[:200]}...")

        # Fallback: treat as plain text answer
        return {
            "question": "",
            "answer": json_content,
            "thinking": thinking_content,  # Always include thinking field
            "reasoning": "",
            "confidence_score": 0.5,
            "reasoning_steps": [],
            "generation_model": model,
            "generation_timestamp": timestamp,
            "raw_response": content
        }

    def _call_grok_api(self, prompt: str) -> str:
        """
        Call Grok API for content generation with structured output.

        Args:
            prompt: The prompt to send.

        Returns:
            Generated content string.
        """
        # Use OpenAI-compatible structured output format for Grok
        request_data = {
            "model": self.grok_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 4000,
            "temperature": 0.3,
            "response_format": { "type": "json_object" }
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
            response.raise_for_status()

            response_json = response.json()
            content = response_json.get("choices", [{}])[0].get("message", {}).get("content", "")

            if not content:
                raise ValueError("Empty response from Grok API")

            return content

        except requests.exceptions.RequestException as e:
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
            response = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4000,
                temperature=0.3,
                response_format={ "type": "json_object" }
            )

            content = response.choices[0].message.content

            if not content:
                raise ValueError("Empty response from OpenAI API")

            return content

        except Exception as e:
            raise Exception(f"OpenAI API request failed: {e}")
