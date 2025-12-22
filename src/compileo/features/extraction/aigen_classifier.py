import requests
from typing import List, Optional
import json
import time
import re
from ...core.logging import get_logger

logger = get_logger(__name__)

def _sanitize_json_response(text: str) -> str:
    """
    Extracts and sanitizes a JSON string from a text response, handling both objects and arrays.
    """
    # Find the start and end of the JSON structure
    start_chars = ['{', '[']
    end_chars = ['}', ']']

    start_index = -1
    end_index = -1
    start_char = ''

    for i, char in enumerate(text):
        if char in start_chars and start_index == -1:
            start_index = i
            start_char = char
            break

    if start_index == -1:
        raise ValueError("No JSON structure found in the response")

    # Find the matching end
    end_char = end_chars[start_chars.index(start_char)]
    brace_count = 0
    for i in range(start_index, len(text)):
        if text[i] == start_char:
            brace_count += 1
        elif text[i] == end_char:
            brace_count -= 1
            if brace_count == 0:
                end_index = i
                break

    if end_index == -1:
        raise ValueError("Unmatched JSON structure")

    json_text = text[start_index:end_index+1]

    # Remove trailing commas before closing braces/brackets
    json_text = re.sub(r',\s*([}\]])', r'\1', json_text)

    return json_text

def generate_categories(chunk_text: str, num_categories: int, exclude_categories: Optional[List[str]] = None, retries: int = 3, backoff_factor: float = 0.5) -> List[str]:
    """
    Generates a list of high-level categories from a text chunk using Ollama.

    Args:
        chunk_text (str): The text chunk to analyze.
        num_categories (int): The desired number of categories.
        retries (int): The number of times to retry the request.
        backoff_factor (float): The factor to determine the delay between retries.

    Returns:
        """
    from ...core.settings import backend_settings
    api_url = f"{backend_settings.get_ollama_base_url()}/api/generate"
    exclude_text = ""
    if exclude_categories:
        exclude_text = f" Exclude these categories: {', '.join(exclude_categories)}."

    prompt_template = """[INST] You are an expert medical text analyst specializing in clinical content classification. Your task is to analyze medical text and generate high-level categories that capture the core themes and topics relevant to clinical diagnosis and patient care.

Analyze the following text and generate exactly {num_categories} high-level categories that are relevant to the content but distinct from the excluded ones.{exclude_text}

Think step by step:
1. Read the text carefully and identify the main themes, medical conditions, symptoms, treatments, or diagnostic elements mentioned.
2. Consider the clinical context: what diseases, organ systems, or medical specialties are implied?
3. Ensure none of the generated categories match the excluded categories.
4. Generate new, unique categories that are broad enough to encompass related concepts but specific enough to be meaningful for clinical classification.
5. Prioritize categories that would be useful for medical research, diagnosis assistance, or patient record organization.

Few-shot examples:
Example 1:
Text: "The patient presents with acute chest pain, shortness of breath, and elevated troponin levels. ECG shows ST elevation in leads II, III, and aVF."
Categories: ["Cardiovascular Disease", "Acute Coronary Syndrome", "Diagnostic Markers"]

Example 2:
Text: "Patient diagnosed with type 2 diabetes mellitus, currently on metformin and insulin therapy. HbA1c is 8.5%, with complaints of frequent urination and fatigue."
Categories: ["Endocrine Disorders", "Diabetes Management", "Metabolic Markers"]

Return only a JSON array of strings, like this: ["category1", "category2", "category3"]
Do not include any other text.

Text:
{chunk_text}
[/INST]"""
    prompt = prompt_template.format(num_categories=num_categories, exclude_text=exclude_text, chunk_text=chunk_text)

    # Get options from database via settings
    options = {
        "num_ctx": backend_settings.get_classification_ollama_num_ctx(),
        "temperature": backend_settings.get_classification_ollama_temperature(),
        "repeat_penalty": backend_settings.get_classification_ollama_repeat_penalty(),
        "top_p": backend_settings.get_classification_ollama_top_p(),
        "top_k": backend_settings.get_classification_ollama_top_k(),
        "num_predict": backend_settings.get_classification_ollama_num_predict()
    }
    
    seed = backend_settings.get_classification_ollama_seed()
    if seed is not None:
        options["seed"] = seed

    request_data = {
        "model": backend_settings.get_classification_ollama_model(),
        "prompt": prompt,
        "stream": False,
        "raw": True,
        "options": options
    }

    # Log AI interaction
    logger.debug(f"[AI_INTERACTION] - Request Data: {json.dumps(request_data, indent=2)}")

    response = None
    for i in range(retries):
        try:
            response = requests.post(api_url, json=request_data)
            response.raise_for_status()

            response_text = response.json().get("response", "[]")
            # Log AI response
            logger.debug(f"[AI_INTERACTION] - Response Text: {response_text}")

            if not response_text:
                raise ValueError("Empty response from Ollama API")

            # Try to sanitize and parse the response
            try:
                sanitized_text = _sanitize_json_response(response_text)
                data = json.loads(sanitized_text)
            except (ValueError, json.JSONDecodeError):
                # Fallback to direct parsing
                data = json.loads(response_text)

            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and "categories" in data and isinstance(data["categories"], list):
                return data["categories"]
            else:
                return []
        except (requests.exceptions.RequestException, json.JSONDecodeError, ValueError) as e:
            logger.error(f"Error interacting with Ollama (attempt {i+1}/{retries}): {e}")
            logger.debug(f"Request data: {request_data}")
            if response:
                logger.debug(f"Response text: {response.text}")
            if i < retries - 1:
                time.sleep(backoff_factor * (2 ** i))
            else:
                return []
    return []
