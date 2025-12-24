"""
This module provides a function to classify a text chunk based on a hierarchical taxonomy
using the xAI Grok API.
"""

import json
import re
import requests
import time
from typing import Dict, List, Any, Optional
from .context_models import DocumentContext, ChunkContext, HierarchicalCategory
from ...core.settings import backend_settings
from ...core.logging import get_logger

logger = get_logger(__name__)


def _sanitize_json(text: str) -> str:
    """
    Extracts and sanitizes a JSON string from a text response, handling both objects and arrays.
    Handles reasoning models, particularly DeepSeek R1's <think> format.
    """
    import re

    # Remove DeepSeek R1 reasoning tags (confirmed format from user logs)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

    # Remove other potential reasoning markers (conservative approach)
    reasoning_patterns = [
        r'```json\s*',                   # Markdown JSON blocks
        r'```\s*',                       # Generic code blocks
        r'JSON Response:\s*',            # Common prefixes
        r'Here is the JSON:\s*',         # Response prefixes
    ]

    for pattern in reasoning_patterns:
        text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)

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

    # Try to parse as is first to avoid corrupting valid JSON with regexes
    try:
        json.loads(json_text)
        return json_text
    except json.JSONDecodeError:
        pass

    # Remove trailing commas before closing braces/brackets
    json_text = re.sub(r',\s*([}\]])', r'\1', json_text)

    # Fix unquoted keys that sometimes appear
    # This regex looks for word characters followed by a colon, ensuring they aren't already quoted or inside quotes
    # It's a heuristic and might need adjustment for complex cases
    json_text = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', json_text)

    # Try to parse again after trailing comma fix
    try:
        json.loads(json_text)
        return json_text
    except json.JSONDecodeError:
        pass

    # Fix unquoted keys that sometimes appear
    # Note: This is risky as it can match text inside strings, so we only do it if parsing failed
    json_text = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', json_text)

    return json_text


def generate_categories(chunk_text: str, num_categories: int, api_key: str, model: Optional[str] = None, exclude_categories: Optional[List[str]] = None, retries: int = 3, backoff_factor: float = 0.5) -> List[str]:
    """
    Generates a list of high-level categories from a text chunk using xAI Grok API.

    Args:
        chunk_text (str): The text chunk to analyze.
        num_categories (int): The desired number of categories.
        api_key (str): The xAI API key for authentication.
        model (Optional[str]): The Grok model to use. If None, uses default from settings.
        exclude_categories (Optional[List[str]]): Categories to exclude.
        retries (int): The number of times to retry the request.
        backoff_factor (float): The factor to determine the delay between retries.

    Returns:
        List[str]: The generated categories.
    """
    api_url = "https://api.x.ai/v1/chat/completions"
    exclude_text = ""
    if exclude_categories:
        exclude_text = f" Exclude these categories: {', '.join(exclude_categories)}."

    prompt_template = """You are an expert text analyst specializing in content classification. Your task is to analyze text and generate high-level categories that capture the core themes and topics relevant to the content.

Analyze the following text and generate exactly {num_categories} high-level categories that are relevant to the content but distinct from the excluded ones.{exclude_text}

Think step by step:
1. Read the text carefully and identify the main themes, topics, or key elements mentioned.
2. Consider the context: what subjects, concepts, or areas are implied?
3. Ensure none of the generated categories match the excluded categories.
4. Generate new, unique categories that are broad enough to encompass related concepts but specific enough to be meaningful for content classification.
5. Prioritize categories that would be useful for content organization, analysis, or research.

Few-shot examples:
Example 1:
Text: "The company reported quarterly earnings of $2.5 billion, up 15% from last year. Revenue growth was driven by strong performance in the technology sector."
Categories: ["Financial Performance", "Business Growth", "Market Analysis"]

Example 2:
Text: "The new software update includes enhanced security features, improved user interface, and better integration with cloud services."
Categories: ["Software Development", "Security Enhancements", "User Experience"]

Return only a JSON array of strings, like this: ["category1", "category2", "category3"]
Do not include any other text.

Text:
{chunk_text}"""
    prompt = prompt_template.format(num_categories=num_categories, exclude_text=exclude_text, chunk_text=chunk_text)

    # Use provided model or get from settings
    grok_model = model if model is not None else backend_settings.get_classification_grok_model()

    request_data = {
        "model": grok_model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1000,
        "temperature": 0.2
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    response = None
    for i in range(retries):
        try:
            response = requests.post(api_url, json=request_data, headers=headers)
            response.raise_for_status()

            response_json = response.json()
            response_text = response_json.get("choices", [{}])[0].get("message", {}).get("content", "[]")

            if not response_text:
                raise ValueError("Empty response from Grok API")

            # Try to sanitize and parse the response
            try:
                sanitized_text = _sanitize_json(response_text)
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
            logger.error(f"Error interacting with Grok API (attempt {i+1}/{retries}): {e}")
            logger.debug(f"Request data: {request_data}")
            if response:
                logger.debug(f"Response text: {response.text}")
            if i < retries - 1:
                time.sleep(backoff_factor * (2 ** i))
            else:
                return []
    return []


def _classify_with_categories(
    text: str,
    categories: List[str],
    api_key: str,
    model: Optional[str] = None,
    document_context: Optional[DocumentContext] = None,
    chunk_context: Optional[ChunkContext] = None,
    taxonomy: Optional[HierarchicalCategory] = None
) -> Dict[str, Any]:
    """
    Classifies text using provided categories via Grok API with context awareness.
    """
    api_url = "https://api.x.ai/v1/chat/completions"

    # Build context information
    context_info = ""
    if document_context:
        context_parts = []
        if document_context.title:
            context_parts.append(f"Document Title: {document_context.title}")
        if document_context.summary:
            context_parts.append(f"Document Summary: {document_context.summary}")
        if document_context.author:
            context_parts.append(f"Author: {document_context.author}")
        if document_context.keywords:
            context_parts.append(f"Keywords: {', '.join(document_context.keywords)}")
        if context_parts:
            context_info += "\n\nDocument Context:\n" + "\n".join(context_parts)

    if chunk_context and (chunk_context.previous_chunks or chunk_context.next_chunks):
        context_info += "\n\nAdjacent Content for Context:"
        if chunk_context.previous_chunks:
            context_info += f"\nPrevious Content: {' '.join(chunk_context.previous_chunks[-1:])}"
        if chunk_context.next_chunks:
            context_info += f"\nNext Content: {' '.join(chunk_context.next_chunks[:1])}"

    # Convert categories list to string representation
    categories_str = json.dumps(categories)

    prompt_template = f"""You are an expert classifier specializing in extracting and organizing information from texts. Your role is to analyze documents and narratives to identify and extract relevant details for specified categories, ensuring accuracy and relevance.{context_info}

Extract information from the following text for each of the provided categories.

Think step by step:
1. Read the text thoroughly and understand the context, including key details, descriptions, and information.
2. Consider the document context and adjacent content to better understand the full picture.
3. For each provided category, scan the text for any mentions, descriptions, or implications related to that category.
4. Extract specific details, measurements, observations, or key facts that directly pertain to each category.
5. If a category has no relevant information in the text, use an empty string or null value for that key.
6. Summarize or list the extracted information concisely, maintaining appropriate terminology where relevant.
7. Ensure the extracted information is directly supported by the text and meaningful.
8. Organize the results into a clean JSON object with category names as keys and extracted information as values.

Few-shot examples:
Example 1:
Text: "The product launch event attracted 500 attendees and generated $1.2 million in sales. The keynote speaker discussed emerging trends in digital marketing."
Categories: ["Event Attendance", "Sales Performance", "Industry Trends"]
Output: {{"Event Attendance": "500 attendees", "Sales Performance": "$1.2 million in sales", "Industry Trends": "Emerging trends in digital marketing"}}

Example 2:
Text: "The research study involved 200 participants aged 18-65. The methodology included surveys and interviews, with results showing 75% satisfaction rate."
Categories: ["Study Demographics", "Research Methods", "Key Findings"]
Output: {{"Study Demographics": "200 participants aged 18-65", "Research Methods": "Surveys and interviews", "Key Findings": "75% satisfaction rate"}}

Text: {text}

Categories: {categories_str}

Return only a valid JSON object, nothing else."""
    prompt = prompt_template

    # Use provided model or get from settings
    grok_model = model if model is not None else backend_settings.get_classification_grok_model()

    request_data = {
        "model": grok_model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 2000,
        "temperature": 0.2
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(api_url, json=request_data, headers=headers)
        response.raise_for_status()

        response_json = response.json()
        response_content = response_json.get("choices", [{}])[0].get("message", {}).get("content", "{}")

        # Try to sanitize and parse
        try:
            sanitized_content = _sanitize_json(response_content)
            final_json = json.loads(sanitized_content)
        except (ValueError, json.JSONDecodeError):
            final_json = json.loads(response_content)

        if isinstance(final_json, dict):
            return final_json
        else:
            return {}

    except (requests.exceptions.RequestException, json.JSONDecodeError, ValueError) as e:
        logger.error(f"Error interacting with Grok API: {e}")
        return {}


def classify_chunk(
    chunk_text: str,
    api_key: str,
    model: Optional[str] = None,
    high_level_categories: Optional[List[str]] = None,
    num_categories: int = 0,
    document_context: Optional[DocumentContext] = None,
    chunk_context: Optional[ChunkContext] = None,
    taxonomy: Optional[HierarchicalCategory] = None
) -> Dict[str, Any]:
    """
    Generates a hierarchical classification for a text chunk using xAI Grok API with context awareness.

    Args:
        chunk_text (str): The text chunk to classify.
        api_key (str): The xAI API key for authentication.
        model (Optional[str]): The Grok model to use. If None, uses default from settings.
        high_level_categories (Optional[List[str]]): A list of high-level categories to guide the model.
        num_categories (int): The number of categories to generate for aigen.
        document_context (Optional[DocumentContext]): Document-level metadata and context.
        chunk_context (Optional[ChunkContext]): Context from adjacent chunks.
        taxonomy (Optional[HierarchicalCategory]): Hierarchical category taxonomy for classification.

    Returns:
        Dict[str, Any]: A dictionary with 'custom' and 'aigen' classifications.
    """
    output = {
        "custom": {},
        "aigen": {}
    }

    if high_level_categories:
        output["custom"] = _classify_with_categories(
            chunk_text, high_level_categories, api_key, model, document_context, chunk_context, taxonomy
        )

    if num_categories > 0:
        aigen_categories = generate_categories(chunk_text, num_categories, api_key, model, exclude_categories=high_level_categories)
        if aigen_categories:
            output["aigen"] = _classify_with_categories(
                chunk_text, aigen_categories, api_key, model, document_context, chunk_context, taxonomy
            )

    return output