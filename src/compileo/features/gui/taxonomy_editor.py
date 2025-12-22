"""
Taxonomy Editor Module.
Handles taxonomy creation and editing functionality.
"""

import streamlit as st
from typing import Dict, Any, List
from src.compileo.features.gui.services.api_client import api_client, APIError
from src.compileo.features.gui.utils.taxonomy_utils import (
    get_available_projects,
    save_unified_taxonomy,
    save_manual_taxonomy,
    generate_unified_ai_taxonomy,
    extend_taxonomy_with_ai
)
from src.compileo.features.gui.components.taxonomy_components import (
    render_collapsible_category_builder,
    render_taxonomy_category_builder
)


def render_manual_taxonomy_builder():
    """Render the manual taxonomy creation interface."""
    st.subheader("➕ Create Manual Taxonomy")

    # Get available projects
    projects = get_available_projects()
    if not projects:
        st.warning("No projects available. Create a project first.")
        return

    # Initialize session state for manual taxonomy
    if "manual_taxonomy" not in st.session_state:
        st.session_state.manual_taxonomy = {
            "name": "",
            "description": "",
            "project_id": None,
            "depth": 3,
            "categories": []  # List of top-level categories
        }

    taxonomy = st.session_state.manual_taxonomy

    # Project selection
    project_options = {f"{p['name']} (ID: {p['id']})": p for p in projects}
    selected_project_key = st.selectbox(
        "Select Project",
        options=list(project_options.keys()),
        key="manual_taxonomy_project",
        help="Choose the project for this taxonomy"
    )
    if selected_project_key:
        selected_project = project_options[selected_project_key]
        taxonomy["project_id"] = selected_project["id"]

    # Load existing taxonomy option
    st.markdown("---")
    col1, col2 = st.columns([3, 1])
    with col1:
        load_existing = st.checkbox(
            "Load existing taxonomy as template",
            key="load_existing_taxonomy",
            help="Load an existing taxonomy to use as a starting point for manual editing"
        )
    with col2:
        if load_existing:
            if st.button("🔄 Refresh Taxonomies", help="Refresh the list of available taxonomies"):
                st.cache_data.clear()
                st.rerun()

    if load_existing:
        # Get available taxonomies for the selected project
        available_taxonomies = []
        if taxonomy.get("project_id"):
            try:
                response = api_client.get(f"/api/v1/taxonomy?project_id={taxonomy['project_id']}")
                available_taxonomies = response.get("taxonomies", [])
            except APIError:
                available_taxonomies = []

        if available_taxonomies:
            taxonomy_options = {f"{t['name']} (ID: {t['id']})": t for t in available_taxonomies}
            selected_taxonomy_key = st.selectbox(
                "Select Taxonomy to Load",
                options=list(taxonomy_options.keys()),
                key="load_taxonomy_select",
                help="Choose an existing taxonomy to load for editing"
            )

            if selected_taxonomy_key and st.button("📥 Load Taxonomy", type="secondary"):
                selected_taxonomy = taxonomy_options[selected_taxonomy_key]
                try:
                    # Load the full taxonomy details
                    response = api_client.get(f"/api/v1/taxonomy/{selected_taxonomy['id']}")
                    taxonomy_data = response.get('taxonomy', {})

                    # Update the session state with loaded taxonomy
                    taxonomy["name"] = f"{taxonomy_data.get('name', 'Loaded Taxonomy')} (Edited)"
                    taxonomy["description"] = taxonomy_data.get('description', '')
                    taxonomy["categories"] = taxonomy_data.get('children', [])

                    st.success(f"Taxonomy '{selected_taxonomy['name']}' loaded successfully! You can now edit it manually.")
                    st.rerun()

                except APIError as e:
                    st.error(f"Failed to load taxonomy: {e}")
        else:
            st.info("No taxonomies available for the selected project. Create a taxonomy first or select a different project.")

    st.markdown("---")

    # Basic info
    col1, col2 = st.columns(2)
    with col1:
        taxonomy["name"] = st.text_input(
            "Taxonomy Name",
            value=taxonomy["name"],
            key="manual_taxonomy_name",
            help="Enter a name for your taxonomy"
        )
    with col2:
        taxonomy["depth"] = st.slider(
            "Maximum Depth",
            min_value=1,
            max_value=5,
            value=taxonomy["depth"],
            key="manual_taxonomy_depth",
            help="Maximum number of hierarchy levels (1-5)"
        )

    taxonomy["description"] = st.text_area(
        "Description (Optional)",
        value=taxonomy["description"],
        key="manual_taxonomy_description",
        height=100,
        help="Describe the purpose and scope of this taxonomy"
    )

    st.divider()

    # Tree builder
    st.subheader("🏗️ Build Taxonomy Structure")
    st.markdown("Build your hierarchical taxonomy by adding top-level categories and their subcategories.")

    # Add top-level category button
    if st.button("➕ Add Top-Level Category", type="secondary", width='stretch'):
        taxonomy["categories"].append({
            "name": "",
            "description": "",
            "confidence_threshold": 0.8,
            "children": []
        })
        st.rerun()

    # Get extension parameters from session state or defaults
    ext_additional_depth = st.session_state.get("extend_depth", 2)
    ext_generator_type = st.session_state.get("extend_generator", "gemini")
    ext_domain = st.session_state.get("extend_domain", "general")

    # Render all top-level categories
    if taxonomy["categories"]:
        for i, category in enumerate(taxonomy["categories"]):
            render_taxonomy_category_builder(category, 0, taxonomy["depth"], f"cat_{i}", taxonomy, ext_additional_depth, ext_generator_type, ext_domain)
    else:
        st.info("Click 'Add Top-Level Category' to start building your taxonomy.")

    # Generate Downstream Taxonomy section
    st.divider()
    st.subheader("🚀 Generate Downstream Taxonomy")
    st.markdown("Use AI to automatically extend your manually defined taxonomy with additional subcategories.")

    # Extension parameters
    col1, col2, col3 = st.columns(3)
    with col1:
        additional_depth = st.slider(
            "Additional Depth Levels",
            min_value=1,
            max_value=3,
            value=2,
            key="extend_depth",
            help="Number of additional hierarchy levels to generate"
        )
    with col2:
        generator_type = st.selectbox(
            "AI Generator",
            options=["grok", "gemini", "ollama", "openai"],
            index=0,
            key="extend_generator",
            help="AI model to use for taxonomy extension"
        )
    with col3:
        domain = st.selectbox(
            "Domain",
            options=["general", "medical", "legal", "technical", "business", "scientific"],
            index=0,
            key="extend_domain",
            help="Content domain for better categorization"
        )

    # Document selection for AI extension
    selected_documents = []
    if taxonomy.get("project_id"):
        try:
            docs_response = api_client.get(f"/api/v1/documents?project_id={taxonomy['project_id']}")
            documents = docs_response.get('documents', [])

            if documents:
                doc_options = {f"{d['file_name']}": d for d in documents}
                selected_doc_names = st.multiselect(
                    "Select Documents to Analyze (Optional)",
                    options=list(doc_options.keys()),
                    key="manual_extend_docs_multiselect",
                    help="Choose specific documents to analyze for taxonomy extension. If none selected, all project documents will be used."
                )
                selected_documents = [doc_options[name]['id'] for name in selected_doc_names]
            else:
                st.warning("No documents found in selected project for AI extension.")
        except APIError as e:
            st.error(f"Failed to load project documents: {e}")

    # Generate button
    if st.button("🚀 Generate Downstream Taxonomy", type="secondary", width='stretch'):
        if not taxonomy["categories"]:
            st.error("Please add at least one top-level category before extending.")
        else:
            extend_taxonomy_with_ai(taxonomy, additional_depth, generator_type, domain, selected_documents)

    # Save button
    st.divider()
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("💾 Save Taxonomy", type="primary", width='stretch'):
            if not taxonomy["name"].strip():
                st.error("Taxonomy name is required.")
            elif not taxonomy["project_id"]:
                st.error("Please select a project.")
            elif not taxonomy["categories"]:
                st.error("At least one top-level category is required.")
            elif not all(cat["name"].strip() for cat in taxonomy["categories"]):
                st.error("All top-level category names are required.")
            else:
                save_manual_taxonomy(taxonomy)
    with col2:
        if st.button("🔄 Reset Form", width='stretch'):
            del st.session_state.manual_taxonomy
            st.rerun()


def render_unified_taxonomy_builder():
    """Render the unified taxonomy builder interface combining manual and AI creation."""
    # st.subheader("🏗️ Unified Taxonomy Builder") # Removed per user feedback

    # Get available projects
    projects = get_available_projects()
    if not projects:
        st.warning("No projects available. Create a project first.")
        return

    # Initialize session state for unified taxonomy builder
    if "unified_taxonomy" not in st.session_state:
        st.session_state.unified_taxonomy = {
            "name": "",
            "description": "",
            "project_id": None,
            "creation_mode": "hybrid",  # "ai" or "hybrid"
            "depth": 3,
            "categories": [],  # List of top-level categories
            "loaded_taxonomy_id": None,  # Track which taxonomy is currently loaded
            "ai_config": {
                "generator": "grok",
                "domain": "general",
                "batch_size": 10,
                "specificity_level": 1,
                "category_limits": [5, 5, 5],  # Default limits for levels 1, 2, 3
                "selected_documents": []
            }
        }

    taxonomy = st.session_state.unified_taxonomy
    ai_config = taxonomy["ai_config"]

    # Project, Document selection, and Creation mode side-by-side
    col1, col2, col3 = st.columns(3)
    with col1:
        project_options = {f"{p['name']} (ID: {p['id']})": p for p in projects}
        selected_project_key = st.selectbox(
            "Select Project",
            options=list(project_options.keys()),
            key="unified_taxonomy_project",
            help="Choose the project for this taxonomy"
        )
        if selected_project_key:
            selected_project = project_options[selected_project_key]
            # Reset document selection if project changes
            if taxonomy["project_id"] != selected_project["id"]:
                ai_config["selected_documents"] = []
            taxonomy["project_id"] = selected_project["id"]

    with col2:
        if taxonomy.get("project_id"):
            try:
                docs_response = api_client.get(f"/api/v1/documents?project_id={taxonomy['project_id']}")
                documents = docs_response.get('documents', [])

                if documents:
                    doc_options = {f"{d['file_name']}": d for d in documents}
                    # Filter out any selected documents that are no longer in the project
                    valid_selected = [doc for doc in ai_config["selected_documents"] if doc in doc_options]

                    ai_config["selected_documents"] = st.multiselect(
                        "Select Documents",
                        options=list(doc_options.keys()),
                        default=valid_selected,
                        key="unified_taxonomy_docs",
                        help="Choose documents to analyze for taxonomy generation"
                    )
                else:
                    st.warning("No documents found in selected project.")
                    ai_config["selected_documents"] = []
            except APIError as e:
                st.error(f"Failed to load project documents: {e}")
                ai_config["selected_documents"] = []
        else:
            st.info("Select a project to see available documents.")

    with col3:
        # Creation mode selection
        creation_modes = {
            "ai": "🚀 AI Generation",
            "hybrid": "🔄 Hybrid Mode"
        }

        selected_mode = st.radio(
            "Taxonomy Creation Mode",
            options=list(creation_modes.keys()),
            format_func=lambda x: creation_modes[x],
            key="creation_mode_radio",
            help="""**AI Generation**: Automatically analyzes your selected documents to create a complete taxonomy hierarchy using advanced AI algorithms. Best for when you want AI to handle the entire taxonomy creation process.

**Hybrid Mode**: Start with manual taxonomy creation and selectively use AI to extend or enhance specific parts of your taxonomy. Provides maximum control while leveraging AI assistance."""
        )
        taxonomy["creation_mode"] = selected_mode

    # Taxonomy starting point selection
    st.markdown("---")
    if taxonomy.get("project_id"):
        available_taxonomies = []
        try:
            response = api_client.get(f"/api/v1/taxonomy?project_id={taxonomy['project_id']}")
            available_taxonomies = response.get("taxonomies", [])
        except APIError:
            available_taxonomies = []

        # Always include "New Taxonomy" option
        taxonomy_options = {"🆕 New Taxonomy": None}
        if available_taxonomies:
            taxonomy_options.update({f"{t['name']} (ID: {t['id']})": t for t in available_taxonomies})

        selected_taxonomy_key = st.selectbox(
            "Choose Starting Point",
            options=list(taxonomy_options.keys()),
            index=0,  # Default to "New Taxonomy"
            key="saved_taxonomy_select",
            help="Choose 'New Taxonomy' to start from scratch or select an existing taxonomy to use as a starting point"
        )

        if selected_taxonomy_key and selected_taxonomy_key != "🆕 New Taxonomy":
            selected_taxonomy = taxonomy_options[selected_taxonomy_key]
            if taxonomy.get("loaded_taxonomy_id") != selected_taxonomy['id']:
                # Load the taxonomy
                try:
                    response = api_client.get(f"/api/v1/taxonomy/{selected_taxonomy['id']}")
                    saved_data = response.get('taxonomy', {})

                    # Update session state with saved data
                    taxonomy["name"] = f"{saved_data.get('name', 'Saved')} (Modified)"
                    taxonomy["description"] = saved_data.get('description', '')
                    taxonomy["categories"] = saved_data.get('children', [])
                    taxonomy["loaded_taxonomy_id"] = selected_taxonomy['id']

                    st.success(f"Saved taxonomy '{selected_taxonomy['name']}' loaded successfully!")
                    st.rerun()

                except APIError as e:
                    st.error(f"Failed to load saved taxonomy: {e}")
        elif selected_taxonomy_key == "🆕 New Taxonomy":
            # Reset to new taxonomy if switching from loaded one
            if taxonomy.get("loaded_taxonomy_id") is not None:
                taxonomy["name"] = ""
                taxonomy["description"] = ""
                taxonomy["categories"] = []
                taxonomy["loaded_taxonomy_id"] = None
                st.info("Switched to creating a new taxonomy from scratch.")

    st.markdown("---")

    # Basic taxonomy info
    col1, col2 = st.columns(2)
    with col1:
        taxonomy["name"] = st.text_input(
            "Taxonomy Name",
            value=taxonomy["name"],
            key="unified_taxonomy_name",
            help="Enter a name for your taxonomy"
        )
    with col2:
        taxonomy["depth"] = st.slider(
            "Maximum Depth",
            min_value=1,
            max_value=5,
            value=taxonomy["depth"],
            key="unified_taxonomy_depth",
            help="Maximum number of hierarchy levels (1-5)"
        )

    taxonomy["description"] = st.text_area(
        "Description (Optional)",
        value=taxonomy["description"],
        key="unified_taxonomy_description",
        height=80,
        help="Describe the purpose and scope of this taxonomy"
    )

    # AI Configuration (shown for AI mode only)
    if selected_mode == "ai":
        st.markdown("### 🤖 AI Configuration")
        # ai_config is already retrieved above

        col1, col2, col3 = st.columns(3)
        with col1:
            ai_config["generator"] = st.selectbox(
                "AI Generator",
                options=["grok", "gemini", "ollama", "openai"],
                index=["grok", "gemini", "ollama", "openai"].index(ai_config["generator"]) if ai_config["generator"] in ["grok", "gemini", "ollama", "openai"] else 0,
                key="ai_generator_select",
                help="AI model to use for generation"
            )
        with col2:
            ai_config["domain"] = st.selectbox(
                "Domain",
                options=["general", "medical", "legal", "technical", "business", "scientific"],
                index=["general", "medical", "legal", "technical", "business", "scientific"].index(ai_config["domain"]),
                key="ai_domain_select",
                help="Content domain for better categorization"
            )
        with col3:
            processing_mode_options = {
                "fast": "Fast (Sampled)",
                "complete": "Complete (All Content)"
            }
            ai_config["processing_mode"] = st.selectbox(
                "Processing Mode",
                options=["fast", "complete"],
                format_func=lambda x: processing_mode_options[x],
                index=0 if ai_config.get("processing_mode", "fast") == "fast" else 1,
                key="ai_processing_mode",
                help="Choose 'Fast' for quick sampling or 'Complete' to process every chunk iteratively (slower but more comprehensive)."
            )

        # Batch Size and Specificity Level side-by-side
        col_batch, col_specificity = st.columns(2)
        with col_batch:
            ai_config["batch_size"] = st.slider(
                "Chunk Batch Size",
                min_value=1,
                max_value=50,
                value=ai_config.get("batch_size", 10),
                step=1,
                key="ai_batch_size_slider",
                help="Number of complete chunks to process in each batch"
            )
        with col_specificity:
            ai_config["specificity_level"] = st.slider(
                "Specificity Level",
                min_value=1,
                max_value=5,
                value=ai_config["specificity_level"],
                key="ai_specificity_slider",
                help="Each taxonomy level will automatically be 1 specificity level more specific than the previous level"
            )

        # Category limits per level (integrated)
        st.markdown("### 📊 Category Limits per Level")
        st.markdown("Set maximum number of categories for each taxonomy level:")

        # Ensure category_limits has enough entries
        while len(ai_config["category_limits"]) < taxonomy["depth"]:
            ai_config["category_limits"].append(5)

        category_limits_cols = st.columns(min(taxonomy["depth"], 5))
        for level in range(taxonomy["depth"]):
            with category_limits_cols[level]:
                ai_config["category_limits"][level] = st.number_input(
                    f"Level {level + 1}",
                    min_value=1,
                    max_value=100,
                    value=ai_config["category_limits"][level],
                    key=f"level_{level + 1}_limit",
                    help=f"Maximum categories for taxonomy level {level + 1}"
                )

    # Manual taxonomy building section (shown for manual and hybrid modes)
    if selected_mode in ["manual", "hybrid"]:
        st.markdown("---")
        st.markdown("### 🏗️ Manual Taxonomy Structure")

        if selected_mode == "hybrid":
            st.info("💡 In hybrid mode, you can manually define the basic structure and then use AI to expand it.")

        # Add top-level category button
        if st.button("➕ Add Top-Level Category", type="secondary", width='stretch'):
            taxonomy["categories"].append({
                "name": "",
                "description": "",
                "confidence_threshold": 0.8,
                "children": [],
                "depth_limit": taxonomy["depth"] - 1  # Per-category depth control
            })
            st.rerun()

        # Render collapsible taxonomy tree
        if taxonomy["categories"]:
            st.markdown("**Taxonomy Structure:**")
            for i, category in enumerate(taxonomy["categories"]):
                render_collapsible_category_builder(category, 0, taxonomy["depth"], f"cat_{i}", taxonomy)
        else:
            st.info("Click 'Add Top-Level Category' to start building your taxonomy.")

    # Taxonomy-wide parameters section (for hybrid mode)
    if selected_mode == "hybrid" and taxonomy["categories"]:
        st.markdown("---")
        st.markdown("### Taxonomy-wide parameters")

        col1, col2, col3 = st.columns(3)
        with col1:
            extend_depth = st.slider(
                "Additional Depth Levels",
                min_value=1,
                max_value=3,
                value=2,
                key="extend_depth_slider",
                help="Number of additional hierarchy levels to generate"
            )
        with col2:
            extend_generator = st.selectbox(
                "Extension Generator",
                options=["grok", "gemini", "ollama"],
                index=0,
                key="extend_generator_select",
                help="AI model to use for taxonomy extension"
            )
        with col3:
            extend_domain = st.selectbox(
                "Extension Domain",
                options=["general", "medical", "legal", "technical", "business", "scientific"],
                index=0,
                key="extend_domain_select",
                help="Content domain for extension"
            )
        
        # Add processing mode for hybrid extension too
        processing_mode_options = {
            "fast": "Fast (Sampled)",
            "complete": "Complete (All Content)"
        }
        extend_processing_mode = st.selectbox(
            "Processing Mode",
            options=["fast", "complete"],
            format_func=lambda x: processing_mode_options[x],
            index=0,
            key="extend_processing_mode",
            help="Choose 'Fast' for quick sampling or 'Complete' to process every chunk iteratively."
        )

        # Document selection for enhancement (uses unified selection if available, else optional override)
        # For hybrid enhancement, we reuse the unified document selection if user made one
        selected_doc_ids = []
        doc_options = {}
        if taxonomy.get("project_id"):
             # Map unified selection names to IDs
            try:
                # Re-fetch to map names to IDs if needed, or rely on cache/state if optimized.
                # Since we are inside a function, let's just use what we have or re-fetch safely.
                # Assuming ai_config["selected_documents"] contains the NAMES selected in the unified multiselect
                selected_doc_names = ai_config.get("selected_documents", [])
                
                # We need to resolve these names to IDs.
                # Ideally, we should have stored IDs or objects, but Streamlit multiselect returns the options list items.
                if selected_doc_names:
                     # Fetch docs to map names to IDs
                    docs_response = api_client.get(f"/api/v1/documents?project_id={taxonomy['project_id']}")
                    documents = docs_response.get('documents', [])
                    doc_map = {d['file_name']: d['id'] for d in documents}
                    selected_doc_ids = [doc_map[name] for name in selected_doc_names if name in doc_map]
            except APIError:
                pass

        if st.button("🚀 Extend Entire Taxonomy", type="secondary", width='stretch'):
            if not taxonomy["categories"]:
                st.error("Please add at least one top-level category before enhancing.")
            elif not selected_doc_ids:
                  # If no documents selected in the unified selector, warn user
                  st.error("Please select at least one document in the 'Select Documents' section above.")
            else:
                # Add processing_mode to the extension call
                taxonomy["ai_config"]["processing_mode"] = extend_processing_mode
                extend_taxonomy_with_ai(taxonomy, extend_depth, extend_generator, extend_domain, selected_doc_ids)

    # Generate AI Taxonomy button (for AI mode)
    if selected_mode == "ai":
        st.markdown("---")
        if st.button("🚀 Generate AI Taxonomy", type="primary", width='stretch'):
            if not taxonomy.get("project_id"):
                st.error("Please select a project.")
            elif not taxonomy["ai_config"]["selected_documents"]:
                st.error("Please select at least one document for AI generation.")
            else:
                generate_unified_ai_taxonomy(taxonomy)

    # Save Manual/Hybrid Taxonomy button
    if selected_mode in ["manual", "hybrid"]:
        st.markdown("---")
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("💾 Save Taxonomy", type="primary", width='stretch'):
                if not taxonomy["name"].strip():
                    st.error("Taxonomy name is required.")
                elif not taxonomy["project_id"]:
                    st.error("Please select a project.")
                elif not taxonomy["categories"]:
                    st.error("At least one top-level category is required.")
                elif not all(cat["name"].strip() for cat in taxonomy["categories"]):
                    st.error("All top-level category names are required.")
                else:
                    save_unified_taxonomy(taxonomy)
        with col2:
            if st.button("🔄 Reset Form", width='stretch'):
                del st.session_state.unified_taxonomy
                st.rerun()


def render_ai_taxonomy_generator():
    """Render the AI-powered taxonomy generation interface."""
    st.subheader("🚀 Generate AI Taxonomy")
    projects = get_available_projects()
    if projects:
        project_options = {f"{p['name']} (ID: {p['id']})": p for p in projects}
        selected_project_key = st.selectbox("Select Project", options=list(project_options.keys()), key="ai_taxonomy_project_select")
        if selected_project_key:
            selected_project = project_options[selected_project_key]
            try:
                docs_response = api_client.get(f"/api/v1/documents?project_id={selected_project['id']}")
                documents = docs_response.get('documents', [])
                if documents:
                    taxonomy_name = st.text_input("Taxonomy Name", key="ai_taxonomy_name")
                    generator_type = st.selectbox("AI Generator", options=["grok", "gemini", "ollama", "openai"], key="ai_taxonomy_generator")

                    # Add new parameter controls
                    col1, col2 = st.columns(2)
                    with col1:
                        depth = st.slider("Depth", min_value=1, max_value=5, value=3, help="Taxonomy hierarchy depth (1-5 levels)", key="ai_taxonomy_depth")
                        sample_size = st.slider("Sample Size", min_value=10, max_value=500, value=100, step=10, help="Number of chunks to sample for analysis", key="ai_taxonomy_sample_size")
                    with col2:
                        domain_options = ["general", "medical", "technical", "scientific", "business", "legal", "educational", "research"]
                        domain = st.selectbox("Domain", options=domain_options, index=0, help="Content domain for taxonomy generation", key="ai_taxonomy_domain")

                    # Specificity level slider
                    st.markdown("**Specificity Level:** Controls how specific each taxonomy level becomes")
                    specificity_level = st.slider("Specificity Level", min_value=1, max_value=5, value=1, help="Each taxonomy level will automatically be 1 specificity level more specific than the previous level", key="ai_taxonomy_specificity")

                    # Category limits per level
                    st.subheader("📊 Category Limits per Level")
                    st.markdown("Set maximum number of categories for each taxonomy level:")

                    category_limits = []
                    for level in range(1, depth + 1):
                        limit = st.number_input(
                            f"Level {level} max categories",
                            min_value=1,
                            max_value=100,
                            value=5,
                            key=f"ai_taxonomy_level_{level}_limit",
                            help=f"Maximum categories for taxonomy level {level}"
                        )
                        category_limits.append(limit)

                    doc_options = {f"{d['file_name']}": d for d in documents}
                    selected_docs = st.multiselect("Select Documents to Analyze", options=list(doc_options.keys()), key="ai_taxonomy_docs")
                    if st.button("🚀 Generate AI Taxonomy", type="primary", width='stretch'):
                        if not taxonomy_name.strip() or not selected_docs:
                            st.error("Taxonomy name and at least one document are required.")
                        else:
                            from src.compileo.features.taxonomy.generator import TaxonomyGenerator
                            taxonomy_generator = TaxonomyGenerator(api_key=None)  # Will use default API key
                            taxonomy_generator.generate_taxonomy_sync(taxonomy_name.strip(), selected_project['id'], [doc_options[doc]['id'] for doc in selected_docs], depth, generator_type, domain, sample_size, category_limits, specificity_level)
                else:
                    st.warning("No documents found in selected project.")
            except APIError as e:
                st.error(f"Failed to load project documents: {e}")
    else:
        st.warning("No projects available.")