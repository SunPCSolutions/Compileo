"""
AI-Assisted Document Chunking Analyzer
Handles AI analysis for optimal chunking strategy recommendations.
"""

import json
import streamlit as st
from typing import Dict, Any, Optional
import traceback


def analyze_document_for_chunking(sample_content: str, user_instructions: str, selected_chunker: str) -> Optional[Dict[str, Any]]:
    """Analyze document sample and provide chunking recommendations using AI."""
    try:
        from src.compileo.core.settings import backend_settings

        # Get the selected model configuration
        model_name = None
        api_key = None

        if selected_chunker == "gemini":
            api_key = backend_settings.get_setting("gemini_api_key")
            model_name = backend_settings.get_setting("chunking_gemini_model", "gemini-2.5-flash")
        elif selected_chunker == "grok":
            api_key = backend_settings.get_setting("grok_api_key")
            model_name = backend_settings.get_setting("chunking_grok_model", "grok-4-fast-reasoning")
        elif selected_chunker == "openai":
            api_key = backend_settings.get_setting("openai_api_key")
            model_name = backend_settings.get_setting("chunking_openai_model", "gpt-4o")
        elif selected_chunker == "ollama":
            model_name = backend_settings.get_setting("chunking_ollama_model", "mistral:latest")

        if not model_name:
            st.error("❌ No chunking model configured. Please check your settings.")
            return None

        # Create an improved prompt with role-playing, chain-of-thought reasoning, and better validation
        analysis_prompt = fr"""You are a senior document processing expert with 15+ years of experience in text analysis, information retrieval, and content structuring. You have worked with medical textbooks, legal documents, technical manuals, and research papers across multiple domains.

Your task is to analyze the user's document chunking requirements and provide the most accurate recommendations based on systematic analysis.

=== STEP-BY-STEP ANALYSIS PROCESS ===

Follow this chain-of-thought reasoning process:

1. **CONTENT ANALYSIS**: Examine the document structure, patterns, and content type
2. **REQUIREMENT INTERPRETATION**: Understand what the user wants to achieve with chunking
3. **STRATEGY EVALUATION**: Consider all available strategies and their suitability
4. **PATTERN IDENTIFICATION**: Identify specific patterns that define chunk boundaries
5. **VALIDATION AGAINST COMMON FAILURES**: Check recommendations against known failure modes
6. **CONFIDENCE ASSESSMENT**: Evaluate how well the recommendation matches the content

=== USER REQUIREMENTS ===
{user_instructions}

=== DOCUMENT SAMPLE ===
{sample_content}

=== COMMON FAILURE MODES TO AVOID ===

- **Over-chunking**: Creating too many small chunks that lose context
- **Under-chunking**: Creating chunks too large for effective retrieval
- **Pattern mismatch**: Using regex patterns that don't match the actual content structure
- **Boundary confusion**: Not clearly defining where chunks should start vs. end
- **Domain bias**: Assuming content follows patterns from other domains

=== AVAILABLE STRATEGIES ===

1. **CHARACTER CHUNKING**: Fixed-size chunks by character count
   - Best for: Uniform content, simple documents
   - Parameters: chunk_size (int), overlap (int)

2. **TOKEN CHUNKING**: Fixed-size chunks by token count
   - Best for: LLM processing, when token limits matter
   - Parameters: chunk_size (int), overlap (int)

3. **DELIMITER CHUNKING**: Split on specific delimiter patterns
   - Best for: Documents with clear structural delimiters
   - Parameters: delimiters (list of strings)

4. **SCHEMA CHUNKING**: Advanced pattern-based splitting with rules
   - Best for: Complex structured documents, domain-specific patterns
   - Parameters: json_schema (JSON string with rules, combine, include_pattern)

=== SCHEMA CHUNKING DETAILS ===

Schema chunking uses pattern matching to split content. Key concepts:

- **include_pattern**: true = include matched pattern in chunk, false = exclude it
- **rules**: Array of pattern matching rules
- **combine**: "any" (match any rule) or "all" (match all rules)

**CRITICAL RULE FOR PATTERN VARIABILITY**:
If the user describes ANY variability in the pattern structure (e.g. optional lines, variable spacing, alternative keywords), you **MUST** generate separate rules for each concrete variation using `"combine": "any"` rather than a single complex regex.
- DO NOT use `\s+` or `[\s\S]*?` to cover multiple structural variations in one rule if it makes the regex complex.
- DO create Rule 1 for "Variation A" and Rule 2 for "Variation B".

For chunks to START with a pattern (most common user request):
- Set include_pattern: true
- Create regex that matches the desired start pattern
- Splits occur BEFORE the pattern, keeping it in the following chunk

Example: User wants chunks starting with "# DISEASE NAME"
- include_pattern: true
- regex: "# [A-Z ]+"

=== RESPONSE FORMAT ===

Return ONLY a valid JSON object with this exact structure:

{{
  "recommended_strategy": "STRATEGY_NAME",
  "parameters": {{
    "PARAMETER_NAME": "VALUE"
  }},
  "reasoning": "Detailed explanation of analysis and choice",
  "confidence": 0.95
}}

=== VALIDATION REQUIREMENTS ===

Before finalizing your response, validate:
- Strategy matches content characteristics
- Parameters are appropriate for the content type
- Reasoning explains the systematic analysis process
- Confidence reflects actual suitability (not artificially high)

=== EXAMPLES ===

Medical Textbook Example (Variable Formatting):
{{
  "recommended_strategy": "schema",
  "parameters": {{
    "json_schema": "{{\\"rules\\": [{{\\"type\\": \\"pattern\\", \\"value\\": \\"# [A-Z ]+\\\\\\\\n[A-Za-z\\\\\\\\s,\\\\\\\\.]+, [A-Z]+\\\\\\\\n# BASICS\\"}}, {{\\"type\\": \\"pattern\\", \\"value\\": \\"# [A-Z ]+\\\\\\\\n\\\\\\\\n[A-Za-z\\\\\\\\s,\\\\\\\\.]+, [A-Z]+\\\\\\\\n# BASICS\\"}}], \\"combine\\": \\"any\\", \\"include_pattern\\": true}}"
  }},
  "reasoning": "Analysis shows the document uses '# DISEASE NAME' followed by authors and '# BASICS'. The spacing varies (sometimes one newline, sometimes two). I've created two separate rules combined with 'any' to robustly catch both formatting variations without using an overly complex or brittle regex.",
  "confidence": 0.95
}}

Simple Document Example:
{{
  "recommended_strategy": "character",
  "parameters": {{
    "chunk_size": 1000,
    "overlap": 100
  }},
  "reasoning": "Content analysis reveals uniform text without strong structural patterns. Character chunking provides consistent, predictable chunk sizes suitable for general-purpose retrieval.",
  "confidence": 0.78
}}

Return ONLY the JSON object. No explanations, no markdown, no code blocks.
"""

        # Call the appropriate AI model
        response = None
        if selected_chunker == "gemini":
            response = call_gemini_api(analysis_prompt, api_key, model_name)
        elif selected_chunker == "grok":
            response = call_grok_api(analysis_prompt, api_key, model_name)
        elif selected_chunker == "openai":
            response = call_openai_api(analysis_prompt, api_key, model_name)
        elif selected_chunker == "ollama":
            response = call_ollama_api(analysis_prompt, model_name)

        if response:
            try:
                # Clean the response to ensure it's a single JSON object
                response = response.strip()
                if response.startswith("```json"):
                    response = response[7:]
                if response.endswith("```"):
                    response = response[:-3]
                response = response.strip()
                
                result = json.loads(response)
                # Convert json_schema dict to JSON string if it's a dict
                if isinstance(result.get('parameters', {}).get('json_schema'), dict):
                    result['parameters']['json_schema'] = json.dumps(result['parameters']['json_schema'])
                return result
            except json.JSONDecodeError:
                st.error("❌ AI returned invalid response format")
                return None
        else:
            return None

    except Exception as e:
        st.error(f"❌ Error during AI analysis: {str(e)}")
        st.error(traceback.format_exc())
        return None


def display_ai_recommendations(recommendations: Dict[str, Any]) -> None:
    """Display AI recommendations and provide accept/adjust options."""
    st.markdown("### 🎯 AI Recommendations")

    strategy = recommendations.get("recommended_strategy", "")
    parameters = recommendations.get("parameters", {})
    reasoning = recommendations.get("reasoning", "")
    confidence = recommendations.get("confidence", 0)

    # Display recommendation
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"**Recommended Strategy:** {strategy}")
        # Handle confidence as string or number
        try:
            if isinstance(confidence, str):
                confidence_val = float(confidence)
            else:
                confidence_val = confidence
            st.markdown(f"**Confidence:** {confidence_val:.1%}")
        except (ValueError, TypeError):
            st.markdown(f"**Confidence:** {confidence}")
    with col2:
        # Handle confidence comparison for display
        try:
            if isinstance(confidence, str):
                confidence_val = float(confidence)
            else:
                confidence_val = confidence
            if confidence_val >= 0.8:
                st.markdown("🟢 **High Confidence**")
            elif confidence_val >= 0.6:
                st.markdown("🟡 **Medium Confidence**")
            else:
                st.markdown("🔴 **Low Confidence**")
        except (ValueError, TypeError):
            st.markdown("⚪ **Confidence Unknown**")

    # Display reasoning
    with st.expander("📋 AI Reasoning"):
        st.markdown(reasoning)

    # Display parameters
    st.markdown("**📊 Recommended Parameters:**")
    params_display = "\n".join([f"- **{k}**: {v}" for k, v in parameters.items()])
    st.markdown(params_display)

    # Accept/Adjust buttons
    col_accept, col_adjust, col_refine = st.columns([1, 1, 1])
    with col_accept:
        if st.button("✅ Accept & Apply", type="primary"):
            apply_ai_recommendations(strategy, parameters)
            st.success("✅ Recommendations applied to configuration!")
            st.rerun()

    with col_adjust:
        if st.button("🔧 Adjust Parameters"):
            st.session_state.show_parameter_adjustment = True

    with col_refine:
        if st.button("🔄 Refine Analysis"):
            st.session_state.refine_analysis = True

    # Parameter adjustment section
    if st.session_state.get("show_parameter_adjustment", False):
        st.markdown("### 🔧 Adjust Parameters")
        adjusted_params = {}
        for param_name, param_value in parameters.items():
            if isinstance(param_value, int):
                adjusted_params[param_name] = st.slider(
                    f"{param_name}",
                    min_value=max(1, param_value // 2),
                    max_value=param_value * 2,
                    value=param_value
                )
            elif isinstance(param_value, float):
                adjusted_params[param_name] = st.slider(
                    f"{param_name}",
                    min_value=0.0,
                    max_value=1.0,
                    value=param_value,
                    step=0.1
                )
            else:
                adjusted_params[param_name] = st.text_input(
                    f"{param_name}",
                    value=str(param_value)
                )

        if st.button("✅ Apply Adjusted Parameters"):
            apply_ai_recommendations(strategy, adjusted_params)
            st.success("✅ Adjusted parameters applied!")
            st.session_state.show_parameter_adjustment = False
            st.rerun()


def apply_ai_recommendations(strategy: str, parameters: Dict[str, Any]) -> None:
    """Apply AI recommendations to the session state for processing."""
    # Store recommendations in session state for use in processing
    st.session_state.ai_recommended_strategy = strategy
    st.session_state.ai_recommended_params = parameters


def call_gemini_api(prompt: str, api_key: str, model_name: str) -> Optional[str]:
    """Call Gemini API for analysis."""
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        st.error(f"❌ Gemini API error: {str(e)}")
        return None


def call_grok_api(prompt: str, api_key: str, model_name: str) -> Optional[str]:
    """Call Grok API for analysis."""
    try:
        import requests
        api_url = "https://api.x.ai/v1/chat/completions"

        request_data = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1000,
            "temperature": 0.1
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        response = requests.post(api_url, json=request_data, headers=headers)
        response.raise_for_status()
        response_json = response.json()
        return response_json.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    except Exception as e:
        st.error(f"❌ Grok API error: {str(e)}")
        return None


def call_openai_api(prompt: str, api_key: str, model_name: str) -> Optional[str]:
    """Call OpenAI API for analysis."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1000
        )

        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"❌ OpenAI API error: {str(e)}")
        return None


def call_ollama_api(prompt: str, model_name: str) -> Optional[str]:
    """Call Ollama API for analysis."""
    try:
        import requests
        api_url = "http://localhost:11434/api/generate"

        request_data = {
            "model": model_name,
            "prompt": prompt,
            "stream": False
        }

        response = requests.post(api_url, json=request_data)
        response.raise_for_status()
        response_json = response.json()
        return response_json.get("response", "").strip()
    except Exception as e:
        st.error(f"❌ Ollama API error: {str(e)}")
        return None