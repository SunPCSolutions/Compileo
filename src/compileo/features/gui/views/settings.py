"""Settings page for the Compileo GUI."""

import streamlit as st
import requests
from src.compileo.features.gui.state.session_state import session_state
from src.compileo.features.gui.config import config
from src.compileo.features.gui.utils.settings_storage import settings_storage
from src.compileo.features.gui.utils.ollama_utils import (
    get_ollama_models,
    test_ollama_connection,
    get_gemini_models,
    get_grok_models,
    get_openai_models
)
from src.compileo.features.gui.services.api_client import api_client
from src.compileo.features.gui.views.plugins_tab import render_plugins_tab

def render_settings():
    """Render the settings page."""
    st.title("⚙️ Settings")
    
    tab_general, tab_plugins = st.tabs(["General", "Plugins"])
    
    with tab_plugins:
        render_plugins_tab()
        
    with tab_general:
        render_general_settings()

def render_general_settings():
    """Render the general settings tab."""
    st.markdown("Configure application preferences and API settings.")

    # Load saved settings
    saved_settings = settings_storage.load_settings()

    # API Settings
    st.subheader("🔗 API Configuration")
    with st.expander("API Settings", expanded=True):
        # API Key input (already in sidebar, but also here for completeness)
        api_key = st.text_input(
            "API Key",
            value=session_state.api_key or saved_settings.get("api_key", ""),
            type="password",
            help="Enter your API key for backend authentication",
            key="settings_api_key"
        )
        if api_key != (session_state.api_key or ""):
            session_state.api_key = api_key

        # API Base URL
        api_url = st.text_input(
            "API Base URL",
            value=saved_settings.get("api_base_url", config.api_base_url),
            help="Base URL for the Compileo API",
            key="settings_api_url"
        )

        # Ollama Base URL
        ollama_url = st.text_input(
            "Ollama Base URL",
            value=saved_settings.get("ollama_base_url", "http://localhost:11434"),
            help="Base URL for Ollama API server",
            key="settings_ollama_url"
        )

        # Test Ollama connection
        if st.button("Test Ollama Connection", key="test_ollama"):
            if test_ollama_connection(ollama_url):
                st.success("✅ Ollama connection successful!")
            else:
                st.error("❌ Could not connect to Ollama. Please check the URL and ensure Ollama is running.")

    # AI Model Configuration
    st.subheader("🤖 AI Model Configuration")
    with st.expander("Model Settings", expanded=True):
        st.markdown("Configure API keys and model assignments for different tasks.")

        # API Keys
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            gemini_key = st.text_input(
                "Google Gemini API Key",
                value=saved_settings.get("gemini_api_key", ""),
                type="password",
                help="API key for Google Gemini models",
                key="gemini_api_key"
            )
        with col2:
            grok_key = st.text_input(
                "xAI Grok API Key",
                value=saved_settings.get("grok_api_key", ""),
                type="password",
                help="API key for xAI Grok models",
                key="grok_api_key"
            )
        with col3:
            huggingface_key = st.text_input(
                "HuggingFace API Key",
                value=saved_settings.get("huggingface_api_key", ""),
                type="password",
                help="API key for HuggingFace models",
                key="huggingface_api_key"
            )
        with col4:
            openai_key = st.text_input(
                "OpenAI API Key",
                value=saved_settings.get("openai_api_key", ""),
                type="password",
                help="API key for OpenAI models (ChatGPT)",
                key="openai_api_key"
            )

        # Model Assignments
        st.markdown("### Model Assignments")
        st.markdown("Configure preferred models for each AI provider. These will be used when you select that provider in processing tabs.")

        # Define tasks and providers
        tasks = [
            ("parsing", "Document Parsing"),
            ("chunking", "Text Chunking"),
            ("taxonomy", "Taxonomy Generation"),
            ("classification", "Document Classification"),
            ("generation", "Dataset Generation"),
            ("quality", "Dataset Quality Analysis"),
            ("benchmarking", "Benchmarking"),
        ]

        providers = ["ollama", "gemini", "grok", "openai"]

        # Initialize model variables
        parsing_ollama_model = None
        parsing_gemini_model = None
        parsing_grok_model = None
        parsing_openai_model = None
        chunking_ollama_model = None
        chunking_gemini_model = None
        chunking_grok_model = None
        chunking_openai_model = None
        taxonomy_ollama_model = None
        taxonomy_gemini_model = None
        taxonomy_grok_model = None
        taxonomy_openai_model = None
        classification_ollama_model = None
        classification_gemini_model = None
        classification_grok_model = None
        classification_openai_model = None
        generation_ollama_model = None
        generation_gemini_model = None
        generation_grok_model = None
        generation_openai_model = None
        quality_ollama_model = None
        quality_gemini_model = None
        quality_grok_model = None
        quality_openai_model = None
        benchmarking_ollama_model = None
        benchmarking_gemini_model = None
        benchmarking_grok_model = None
        benchmarking_openai_model = None

        # Initialize Ollama parameter variables
        parsing_ollama_temperature = None
        parsing_ollama_repeat_penalty = None
        parsing_ollama_top_p = None
        parsing_ollama_top_k = None
        parsing_ollama_num_predict = None
        parsing_ollama_seed = None
        parsing_ollama_num_ctx = None

        chunking_ollama_temperature = None
        chunking_ollama_repeat_penalty = None
        chunking_ollama_top_p = None
        chunking_ollama_top_k = None
        chunking_ollama_num_predict = None
        chunking_ollama_seed = None
        chunking_ollama_num_ctx = None

        taxonomy_ollama_temperature = None
        taxonomy_ollama_repeat_penalty = None
        taxonomy_ollama_top_p = None
        taxonomy_ollama_top_k = None
        taxonomy_ollama_num_predict = None
        taxonomy_ollama_seed = None
        taxonomy_ollama_num_ctx = None

        classification_ollama_temperature = None
        classification_ollama_repeat_penalty = None
        classification_ollama_top_p = None
        classification_ollama_top_k = None
        classification_ollama_num_predict = None
        classification_ollama_seed = None
        classification_ollama_num_ctx = None

        generation_ollama_temperature = None
        generation_ollama_repeat_penalty = None
        generation_ollama_top_p = None
        generation_ollama_top_k = None
        generation_ollama_num_predict = None
        generation_ollama_seed = None
        generation_ollama_num_ctx = None

        # Get model lists once
        ollama_models = get_ollama_models(ollama_url)
        if not ollama_models:
            st.warning("Could not fetch Ollama models. Please check your Ollama server connection.")
            ollama_models = ["mistral:latest", "llama2:latest"]

        gemini_models = get_gemini_models(gemini_key)
        if not gemini_models:
            st.warning("Could not fetch Gemini models. Using default model.")
            gemini_models = ["gemini-2.5-flash"]

        grok_models = get_grok_models(grok_key)
        if not grok_models:
            st.warning("Could not fetch Grok models. Using default model.")
            grok_models = ["grok-4-fast-reasoning"]

        openai_models = get_openai_models(openai_key)
        if not openai_models:
             # Only show warning if key is present but fetch failed, or just fallback silently if empty
            if openai_key:
                st.warning("Could not fetch OpenAI models. Using default list.")
            openai_models = ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]

        # Create a grid for model configuration
        provider_models = {
            "ollama": ollama_models,
            "gemini": gemini_models,
            "grok": grok_models,
            "openai": openai_models
        }

        # Display model configuration grid
        for task_key, display_name in tasks:
            st.markdown(f"**{display_name}**")

            cols = st.columns(len(providers))
            for i, provider in enumerate(providers):
                with cols[i]:
                    model_options = provider_models[provider]
                    model_key = f"{task_key}_{provider}_model"

                    default_model = saved_settings.get(model_key, model_options[0])
                    if default_model not in model_options:
                        default_model = model_options[0]

                    model = st.selectbox(
                        f"{provider.title()} Model",
                        options=model_options,
                        index=model_options.index(default_model),
                        help=f"{provider.title()} model for {display_name.lower()}",
                        key=f"{task_key}_{provider}_model_select"
                    )

                    # Store the selected model in the appropriate variable
                    if task_key == "parsing":
                        if provider == "ollama":
                            parsing_ollama_model = model
                        elif provider == "gemini":
                            parsing_gemini_model = model
                        elif provider == "grok":
                            parsing_grok_model = model
                        elif provider == "openai":
                            parsing_openai_model = model
                    elif task_key == "chunking":
                        if provider == "ollama":
                            chunking_ollama_model = model
                        elif provider == "gemini":
                            chunking_gemini_model = model
                        elif provider == "grok":
                            chunking_grok_model = model
                        elif provider == "openai":
                            chunking_openai_model = model
                    elif task_key == "taxonomy":
                        if provider == "ollama":
                            taxonomy_ollama_model = model
                        elif provider == "gemini":
                            taxonomy_gemini_model = model
                        elif provider == "grok":
                            taxonomy_grok_model = model
                        elif provider == "openai":
                            taxonomy_openai_model = model
                    elif task_key == "classification":
                        if provider == "ollama":
                            classification_ollama_model = model
                        elif provider == "gemini":
                            classification_gemini_model = model
                        elif provider == "grok":
                            classification_grok_model = model
                        elif provider == "openai":
                            classification_openai_model = model
                    elif task_key == "generation":
                        if provider == "ollama":
                            generation_ollama_model = model
                        elif provider == "gemini":
                            generation_gemini_model = model
                        elif provider == "grok":
                            generation_grok_model = model
                        elif provider == "openai":
                            generation_openai_model = model
                    elif task_key == "quality":
                        if provider == "ollama":
                            quality_ollama_model = model
                        elif provider == "gemini":
                            quality_gemini_model = model
                        elif provider == "grok":
                            quality_grok_model = model
                        elif provider == "openai":
                            quality_openai_model = model
                    elif task_key == "benchmarking":
                        if provider == "ollama":
                            benchmarking_ollama_model = model
                        elif provider == "gemini":
                            benchmarking_gemini_model = model
                        elif provider == "grok":
                            benchmarking_grok_model = model
                        elif provider == "openai":
                            benchmarking_openai_model = model

                    # Add Ollama parameter inputs below the model dropdown
                    if provider == "ollama":
                        st.markdown("**Ollama Parameters:**")

                        # Create 4-column grid for parameters
                        param_cols = st.columns(4)

                        # Row 1: temperature, repeat_penalty, top_p, top_k
                        with param_cols[0]:
                            temp_key = f"{task_key}_ollama_temperature"
                            default_temp = saved_settings.get(temp_key, "0.1")
                            temp_val = st.text_input(
                                "Temperature",
                                value=str(default_temp),
                                help="(0.0-2.0)",
                                key=f"{task_key}_ollama_temp_input"
                            )

                        with param_cols[1]:
                            repeat_key = f"{task_key}_ollama_repeat_penalty"
                            default_repeat = saved_settings.get(repeat_key, "1.1")
                            repeat_val = st.text_input(
                                "Repeat Penalty",
                                value=str(default_repeat),
                                help="(0.0-2.0)",
                                key=f"{task_key}_ollama_repeat_input"
                            )

                        with param_cols[2]:
                            topp_key = f"{task_key}_ollama_top_p"
                            default_topp = saved_settings.get(topp_key, "0.9")
                            topp_val = st.text_input(
                                "Top P",
                                value=str(default_topp),
                                help="(0.0-1.0)",
                                key=f"{task_key}_ollama_topp_input"
                            )

                        with param_cols[3]:
                            topk_key = f"{task_key}_ollama_top_k"
                            default_topk = saved_settings.get(topk_key, "40")
                            topk_val = st.text_input(
                                "Top K",
                                value=str(default_topk),
                                help="(0-100)",
                                key=f"{task_key}_ollama_topk_input"
                            )

                        # Row 2: num_predict, seed, (empty), (empty)
                        param_cols2 = st.columns(4)

                        with param_cols2[0]:
                            numpred_key = f"{task_key}_ollama_num_predict"
                            default_numpred = saved_settings.get(numpred_key, "1024")
                            numpred_val = st.text_input(
                                "Num Predict",
                                value=str(default_numpred),
                                help="(1-4096)",
                                key=f"{task_key}_ollama_numpred_input"
                            )

                        with param_cols2[1]:
                            numctx_key = f"{task_key}_ollama_num_ctx"
                            default_numctx = saved_settings.get(numctx_key, "4096")
                            numctx_val = st.text_input(
                                "Num Ctx",
                                value=str(default_numctx),
                                help="Context window size in tokens",
                                key=f"{task_key}_ollama_numctx_input"
                            )

                        with param_cols2[2]:
                            seed_key = f"{task_key}_ollama_seed"
                            default_seed = saved_settings.get(seed_key, "")
                            seed_val = st.text_input(
                                "Seed",
                                value=str(default_seed) if default_seed is not None else "",
                                help="(0-4294967295)",
                                key=f"{task_key}_ollama_seed_input"
                            )

                        # Column 3 in row 2 is empty

                        # Store parameter values
                        if task_key == "parsing":
                            parsing_ollama_temperature = temp_val
                            parsing_ollama_repeat_penalty = repeat_val
                            parsing_ollama_top_p = topp_val
                            parsing_ollama_top_k = topk_val
                            parsing_ollama_num_predict = numpred_val
                            parsing_ollama_seed = seed_val if seed_val else None
                            parsing_ollama_num_ctx = numctx_val
                        elif task_key == "chunking":
                            chunking_ollama_temperature = temp_val
                            chunking_ollama_repeat_penalty = repeat_val
                            chunking_ollama_top_p = topp_val
                            chunking_ollama_top_k = topk_val
                            chunking_ollama_num_predict = numpred_val
                            chunking_ollama_seed = seed_val if seed_val else None
                            chunking_ollama_num_ctx = numctx_val
                        elif task_key == "taxonomy":
                            taxonomy_ollama_temperature = temp_val
                            taxonomy_ollama_repeat_penalty = repeat_val
                            taxonomy_ollama_top_p = topp_val
                            taxonomy_ollama_top_k = topk_val
                            taxonomy_ollama_num_predict = numpred_val
                            taxonomy_ollama_seed = seed_val if seed_val else None
                            taxonomy_ollama_num_ctx = numctx_val
                        elif task_key == "classification":
                            classification_ollama_temperature = temp_val
                            classification_ollama_repeat_penalty = repeat_val
                            classification_ollama_top_p = topp_val
                            classification_ollama_top_k = topk_val
                            classification_ollama_num_predict = numpred_val
                            classification_ollama_seed = seed_val if seed_val else None
                            classification_ollama_num_ctx = numctx_val
                        elif task_key == "generation":
                            generation_ollama_temperature = temp_val
                            generation_ollama_repeat_penalty = repeat_val
                            generation_ollama_top_p = topp_val
                            generation_ollama_top_k = topk_val
                            generation_ollama_num_predict = numpred_val
                            generation_ollama_seed = seed_val if seed_val else None
                            generation_ollama_num_ctx = numctx_val

            st.markdown("---")

        # Advanced Settings
        with st.expander("Advanced Model Settings"):
            taxonomy_depth = st.slider(
                "Taxonomy Hierarchy Depth",
                min_value=1,
                max_value=5,
                value=saved_settings.get("taxonomy_depth", 3),
                help="Maximum depth for generated taxonomy hierarchies",
                key="taxonomy_depth"
            )

            sample_size = st.slider(
                "Taxonomy Sample Size",
                min_value=10,
                max_value=500,
                value=saved_settings.get("sample_size", 100),
                help="Number of chunks to sample for taxonomy generation",
                key="sample_size"
            )

    # UI Settings
    st.subheader("🎨 User Interface")
    with st.expander("Display Settings"):
        theme = st.selectbox(
            "Theme",
            options=["Light", "Dark", "Auto"],
            index=["Light", "Dark", "Auto"].index(
                saved_settings.get("theme", "Light")
            ) if saved_settings.get("theme") in ["Light", "Dark", "Auto"] else 0,
            help="Choose the application theme"
        )

        page_size = st.slider(
            "Default Page Size",
            min_value=10,
            max_value=100,
            value=saved_settings.get("default_page_size", config.default_page_size),
            help="Number of items to display per page"
        )

    # Processing Settings
    st.subheader("⚡ Processing Configuration")
    with st.expander("Processing Settings"):
        col1, col2 = st.columns(2)

        with col1:
            max_concurrent = st.number_input(
                "Max Concurrent Jobs (Global)",
                min_value=1,
                max_value=50,
                value=saved_settings.get("max_concurrent_jobs", config.max_concurrent_jobs),
                help="Maximum number of concurrent processing jobs across all users",
                key="global_jobs_input"
            )

        with col2:
            # TODO: Uncomment when multi-user architecture is implemented
            # max_per_user_default = min(
            #     saved_settings.get("max_concurrent_jobs_per_user", 3),
            #     max_concurrent
            # )
            # max_concurrent_per_user = st.number_input(
            #     "Max Concurrent Jobs Per User",
            #     min_value=1,
            #     max_value=max_concurrent,  # Dynamic max based on global limit
            #     value=max_per_user_default,
            #     help="Maximum number of concurrent processing jobs per user",
            #     key="per_user_jobs_input"
            # )
            # Single-user mode: per-user limit equals global limit
            max_concurrent_per_user = max_concurrent

        quality_threshold = st.slider(
            "Quality Threshold",
            min_value=0.0,
            max_value=1.0,
            value=saved_settings.get("quality_threshold", config.default_quality_threshold),
            step=0.1,
            help="Minimum quality score for dataset entries"
        )

    # File Upload Settings
    st.subheader("📁 File Upload")
    with st.expander("Upload Settings"):
        max_file_size = st.number_input(
            "Max File Size (MB)",
            min_value=1,
            max_value=1000,
            value=saved_settings.get("max_file_size_mb", config.max_file_size_mb),
            help="Maximum file size for uploads"
        )

        st.write("**Allowed File Types:**")
        st.code(", ".join(config.allowed_file_types))

    # Validation before save
    validation_errors = []

    # Check job limits validation
    if max_concurrent_per_user > max_concurrent:
        validation_errors.append(f"Per-user concurrent jobs limit ({max_concurrent_per_user}) cannot exceed the global limit ({max_concurrent}).")
        # Auto-correct the value
        max_concurrent_per_user = max_concurrent

    # Logging Settings
    st.subheader("📝 Logging Configuration")
    with st.expander("Logging Settings", expanded=True):
        log_level = st.selectbox(
            "Log Level",
            options=["none", "error", "debug"],
            index=["none", "error", "debug"].index(
                saved_settings.get("log_level", "debug")
            ),
            help="Choose the log level for the application. 'none' disables all logs, 'error' shows only errors, and 'debug' shows extensive reporting.",
            key="settings_log_level"
        )

    # Save Settings
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        save_disabled = len(validation_errors) > 0
        if st.button("💾 Save Settings", type="primary", width='stretch', disabled=save_disabled):
            # Collect all settings
            settings_to_save = {
                "api_key": api_key,
                "api_base_url": api_url,
                "ollama_base_url": ollama_url,
                "gemini_api_key": gemini_key,
                "grok_api_key": grok_key,
                "huggingface_api_key": huggingface_key,
                "openai_api_key": openai_key,
                "log_level": log_level,
                "taxonomy_depth": taxonomy_depth,
                "sample_size": sample_size,
                "theme": theme,
                "default_page_size": page_size,
                "max_concurrent_jobs": max_concurrent,
                "max_concurrent_jobs_per_user": max_concurrent_per_user,
                "quality_threshold": quality_threshold,
                "max_file_size_mb": max_file_size,
            }

            # Add job configuration
            settings_to_save["max_concurrent_jobs"] = max_concurrent
            # TODO: Uncomment when multi-user architecture is implemented
            # settings_to_save["max_concurrent_jobs_per_user"] = max_concurrent_per_user
            settings_to_save["quality_threshold"] = quality_threshold

            # Add all model configurations (always save them)
            if parsing_ollama_model is not None:
                settings_to_save["parsing_ollama_model"] = parsing_ollama_model
            if parsing_gemini_model is not None:
                settings_to_save["parsing_gemini_model"] = parsing_gemini_model
            if parsing_grok_model is not None:
                settings_to_save["parsing_grok_model"] = parsing_grok_model
            if parsing_openai_model is not None:
                settings_to_save["parsing_openai_model"] = parsing_openai_model

            if chunking_ollama_model is not None:
                settings_to_save["chunking_ollama_model"] = chunking_ollama_model
            if chunking_gemini_model is not None:
                settings_to_save["chunking_gemini_model"] = chunking_gemini_model
            if chunking_grok_model is not None:
                settings_to_save["chunking_grok_model"] = chunking_grok_model
            if chunking_openai_model is not None:
                settings_to_save["chunking_openai_model"] = chunking_openai_model

            if taxonomy_ollama_model is not None:
                settings_to_save["taxonomy_ollama_model"] = taxonomy_ollama_model
            if taxonomy_gemini_model is not None:
                settings_to_save["taxonomy_gemini_model"] = taxonomy_gemini_model
            if taxonomy_grok_model is not None:
                settings_to_save["taxonomy_grok_model"] = taxonomy_grok_model
            if taxonomy_openai_model is not None:
                settings_to_save["taxonomy_openai_model"] = taxonomy_openai_model

            if classification_ollama_model is not None:
                settings_to_save["classification_ollama_model"] = classification_ollama_model
            if classification_gemini_model is not None:
                settings_to_save["classification_gemini_model"] = classification_gemini_model
            if classification_grok_model is not None:
                settings_to_save["classification_grok_model"] = classification_grok_model
            if classification_openai_model is not None:
                settings_to_save["classification_openai_model"] = classification_openai_model

            if generation_ollama_model is not None:
                settings_to_save["generation_ollama_model"] = generation_ollama_model
            if generation_gemini_model is not None:
                settings_to_save["generation_gemini_model"] = generation_gemini_model
            if generation_grok_model is not None:
                settings_to_save["generation_grok_model"] = generation_grok_model
            if generation_openai_model is not None:
                settings_to_save["generation_openai_model"] = generation_openai_model
            if quality_ollama_model is not None:
                settings_to_save["quality_ollama_model"] = quality_ollama_model
            if quality_gemini_model is not None:
                settings_to_save["quality_gemini_model"] = quality_gemini_model
            if quality_grok_model is not None:
                settings_to_save["quality_grok_model"] = quality_grok_model
            if quality_openai_model is not None:
                settings_to_save["quality_openai_model"] = quality_openai_model
            if benchmarking_ollama_model is not None:
                settings_to_save["benchmarking_ollama_model"] = benchmarking_ollama_model
            if benchmarking_gemini_model is not None:
                settings_to_save["benchmarking_gemini_model"] = benchmarking_gemini_model
            if benchmarking_grok_model is not None:
                settings_to_save["benchmarking_grok_model"] = benchmarking_grok_model
            if benchmarking_openai_model is not None:
                settings_to_save["benchmarking_openai_model"] = benchmarking_openai_model

            # Add Ollama parameter configurations
            if parsing_ollama_temperature is not None:
                settings_to_save["parsing_ollama_temperature"] = float(parsing_ollama_temperature)
            if parsing_ollama_repeat_penalty is not None:
                settings_to_save["parsing_ollama_repeat_penalty"] = float(parsing_ollama_repeat_penalty)
            if parsing_ollama_top_p is not None:
                settings_to_save["parsing_ollama_top_p"] = float(parsing_ollama_top_p)
            if parsing_ollama_top_k is not None:
                settings_to_save["parsing_ollama_top_k"] = int(parsing_ollama_top_k)
            if parsing_ollama_num_predict is not None:
                settings_to_save["parsing_ollama_num_predict"] = int(parsing_ollama_num_predict)
            if parsing_ollama_seed is not None:
                settings_to_save["parsing_ollama_seed"] = int(parsing_ollama_seed) if parsing_ollama_seed else None
            if parsing_ollama_num_ctx is not None:
                settings_to_save["parsing_ollama_num_ctx"] = int(parsing_ollama_num_ctx)

            if chunking_ollama_temperature is not None:
                settings_to_save["chunking_ollama_temperature"] = float(chunking_ollama_temperature)
            if chunking_ollama_repeat_penalty is not None:
                settings_to_save["chunking_ollama_repeat_penalty"] = float(chunking_ollama_repeat_penalty)
            if chunking_ollama_top_p is not None:
                settings_to_save["chunking_ollama_top_p"] = float(chunking_ollama_top_p)
            if chunking_ollama_top_k is not None:
                settings_to_save["chunking_ollama_top_k"] = int(chunking_ollama_top_k)
            if chunking_ollama_num_predict is not None:
                settings_to_save["chunking_ollama_num_predict"] = int(chunking_ollama_num_predict)
            if chunking_ollama_seed is not None:
                settings_to_save["chunking_ollama_seed"] = int(chunking_ollama_seed) if chunking_ollama_seed else None
            if chunking_ollama_num_ctx is not None:
                settings_to_save["chunking_ollama_num_ctx"] = int(chunking_ollama_num_ctx)

            if taxonomy_ollama_temperature is not None:
                settings_to_save["taxonomy_ollama_temperature"] = float(taxonomy_ollama_temperature)
            if taxonomy_ollama_repeat_penalty is not None:
                settings_to_save["taxonomy_ollama_repeat_penalty"] = float(taxonomy_ollama_repeat_penalty)
            if taxonomy_ollama_top_p is not None:
                settings_to_save["taxonomy_ollama_top_p"] = float(taxonomy_ollama_top_p)
            if taxonomy_ollama_top_k is not None:
                settings_to_save["taxonomy_ollama_top_k"] = int(taxonomy_ollama_top_k)
            if taxonomy_ollama_num_predict is not None:
                settings_to_save["taxonomy_ollama_num_predict"] = int(taxonomy_ollama_num_predict)
            if taxonomy_ollama_seed is not None:
                settings_to_save["taxonomy_ollama_seed"] = int(taxonomy_ollama_seed) if taxonomy_ollama_seed else None
            if taxonomy_ollama_num_ctx is not None:
                settings_to_save["taxonomy_ollama_num_ctx"] = int(taxonomy_ollama_num_ctx)

            if classification_ollama_temperature is not None:
                settings_to_save["classification_ollama_temperature"] = float(classification_ollama_temperature)
            if classification_ollama_repeat_penalty is not None:
                settings_to_save["classification_ollama_repeat_penalty"] = float(classification_ollama_repeat_penalty)
            if classification_ollama_top_p is not None:
                settings_to_save["classification_ollama_top_p"] = float(classification_ollama_top_p)
            if classification_ollama_top_k is not None:
                settings_to_save["classification_ollama_top_k"] = int(classification_ollama_top_k)
            if classification_ollama_num_predict is not None:
                settings_to_save["classification_ollama_num_predict"] = int(classification_ollama_num_predict)
            if classification_ollama_seed is not None:
                settings_to_save["classification_ollama_seed"] = int(classification_ollama_seed) if classification_ollama_seed else None
            if classification_ollama_num_ctx is not None:
                settings_to_save["classification_ollama_num_ctx"] = int(classification_ollama_num_ctx)

            if generation_ollama_temperature is not None:
                settings_to_save["generation_ollama_temperature"] = float(generation_ollama_temperature)
            if generation_ollama_repeat_penalty is not None:
                settings_to_save["generation_ollama_repeat_penalty"] = float(generation_ollama_repeat_penalty)
            if generation_ollama_top_p is not None:
                settings_to_save["generation_ollama_top_p"] = float(generation_ollama_top_p)
            if generation_ollama_top_k is not None:
                settings_to_save["generation_ollama_top_k"] = int(generation_ollama_top_k)
            if generation_ollama_num_predict is not None:
                settings_to_save["generation_ollama_num_predict"] = int(generation_ollama_num_predict)
            if generation_ollama_seed is not None:
                settings_to_save["generation_ollama_seed"] = int(generation_ollama_seed) if generation_ollama_seed else None
            if generation_ollama_num_ctx is not None:
                settings_to_save["generation_ollama_num_ctx"] = int(generation_ollama_num_ctx)

            # Save to database
            if settings_storage.save_settings(settings_to_save):
                # Update API client with new settings
                api_client.update_settings(base_url=api_url, api_key=api_key)
                st.success("Settings saved successfully!")
            else:
                st.error("Failed to save settings. Please try again.")
        elif save_disabled:
            st.error("Please fix the validation errors before saving.")
            for error in validation_errors:
                st.error(f"• {error}")

    # Reset to Defaults
    if st.button("🔄 Reset to Defaults", help="Reset all settings to default values"):
        if settings_storage.reset_settings():
            st.info("Settings reset to defaults")
            st.rerun()  # Refresh to show default values
        else:
            st.error("Failed to reset settings. Please try again.")

    # System Information
    st.subheader("ℹ️ System Information")
    with st.expander("System Info"):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("API Status", "Connected" if session_state.api_key else "Not Configured")
            st.metric("Database", "Connected")
        with col2:
            st.metric("Vector Store", "Connected")
            st.metric("LLM Services", "Available")


