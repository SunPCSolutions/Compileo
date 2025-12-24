"""Utility functions for taxonomy operations."""

import streamlit as st
from typing import Dict, Any, List, Optional
import time

from src.compileo.features.gui.services.api_client import api_client, APIError
from src.compileo.features.gui.services.job_monitoring_service import monitor_job_synchronously


@st.cache_data(ttl=300)
def get_available_taxonomies() -> List[Dict[str, Any]]:
    """Get list of available taxonomies."""
    try:
        response = api_client.get("/api/v1/taxonomy")
        if response:
            return response.get("taxonomies", [])
        return []
    except APIError:
        return []


@st.cache_data(ttl=300)
def get_available_projects() -> List[Dict[str, Any]]:
    """Get list of available projects."""
    try:
        response = api_client.get("/api/v1/projects")
        if response:
            return response.get("projects", [])
        return []
    except APIError:
        return []


def remove_category_from_unified_tree(path: str):
    """Remove a category from the unified taxonomy tree."""
    parts = path.split("_")
    if len(parts) < 2 or parts[0] != "cat":
        return

    # All paths start with "cat", so we need to check the length to determine level
    if len(parts) == 2:
        # Top-level category: "cat_0", "cat_1", etc.
        cat_idx = int(parts[1])
        if cat_idx < len(st.session_state.unified_taxonomy["categories"]):
            st.session_state.unified_taxonomy["categories"].pop(cat_idx)
        return

    # Subcategory: "cat_0_1", "cat_0_1_2", etc.
    cat_idx = int(parts[1])  # Index of top-level category
    if cat_idx >= len(st.session_state.unified_taxonomy["categories"]):
        return

    current = st.session_state.unified_taxonomy["categories"][cat_idx]

    # Navigate through the hierarchy (skip "cat" and top-level index)
    for part in parts[2:-1]:
        if part.isdigit():
            idx = int(part)
            if idx < len(current.get("children", [])):
                current = current["children"][idx]
            else:
                return

    # Remove the last child
    last_idx = int(parts[-1])
    if "children" in current and last_idx < len(current["children"]):
        current["children"].pop(last_idx)


def remove_category_from_tree(path: str):
    """Remove a category from the taxonomy tree based on its path."""
    parts = path.split("_")
    if len(parts) < 2:
        return  # Can't remove if not enough parts

    if parts[0] == "cat":
        # Removing a top-level category
        cat_idx = int(parts[1])
        if cat_idx < len(st.session_state.manual_taxonomy["categories"]):
            st.session_state.manual_taxonomy["categories"].pop(cat_idx)
        return

    # Removing a subcategory
    cat_idx = int(parts[0].replace("cat_", ""))
    if cat_idx >= len(st.session_state.manual_taxonomy["categories"]):
        return

    # Navigate to the parent node
    current = st.session_state.manual_taxonomy["categories"][cat_idx]
    for part in parts[2:-1]:  # Skip 'cat_X' and last index
        if part.isdigit():
            idx = int(part)
            if idx < len(current.get("children", [])):
                current = current["children"][idx]
            else:
                return

    # Remove the last child
    last_idx = int(parts[-1])
    if "children" in current and last_idx < len(current["children"]):
        current["children"].pop(last_idx)


def get_category_by_path(path: str, taxonomy: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Get a category from the taxonomy by its path."""
    parts = path.split("_")
    if len(parts) < 2:
        return None

    if parts[0] == "cat":
        # Navigate to the category
        cat_idx = int(parts[1])
        if cat_idx >= len(taxonomy["categories"]):
            return None

        current = taxonomy["categories"][cat_idx]

        # Navigate deeper if there are more path parts
        for part in parts[2:]:
            if part.isdigit():
                idx = int(part)
                if idx < len(current.get("children", [])):
                    current = current["children"][idx]
                else:
                    return None
            else:
                return None

        return current

    return None


def generate_unified_ai_taxonomy(taxonomy: Dict[str, Any]):
    """Generate taxonomy using AI in the unified interface."""
    try:
        ai_config = taxonomy["ai_config"]
        doc_options = {}
        if taxonomy.get("project_id"):
            try:
                docs_response = api_client.get(f"/api/v1/documents?project_id={taxonomy['project_id']}")
                if docs_response:
                    documents = docs_response.get('documents', [])
                    doc_options = {f"{d['file_name']}": d for d in documents}
            except APIError:
                pass

        # Retrieve selected documents from ai_config and convert to list of IDs
        document_ids = [doc_options[doc]['id'] for doc in ai_config["selected_documents"] if doc in doc_options]

        if not document_ids:
             st.error("No documents selected or documents not found. Please select documents to generate taxonomy.")
             return

        with st.spinner("Generating AI taxonomy..."):
            request_data = {
                "project_id": taxonomy["project_id"],
                "name": taxonomy["name"],
                "documents": document_ids,
                "depth": taxonomy["depth"],
                "generator": ai_config["generator"],
                "domain": ai_config["domain"],
                "batch_size": ai_config["batch_size"],
                "specificity_level": ai_config["specificity_level"],
                "processing_mode": ai_config.get("processing_mode", "fast")
            }
            if ai_config["category_limits"]:
                request_data["category_limits"] = ai_config["category_limits"][:taxonomy["depth"]]

            response = api_client.post("/api/v1/taxonomy/generate", request_data)

        if response is None:
            st.error("Failed to generate AI taxonomy - API call returned no response")
        elif 'job_id' in response:
            job_id = response['job_id']
            st.success(f"✅ AI Taxonomy generation started! Job ID: {job_id}")
            # Monitor job synchronously in-view (Wizard-style)
            success = monitor_job_synchronously(job_id, success_text="Taxonomy generation completed!")
            if success:
                st.cache_data.clear()
                st.rerun()
        elif 'id' in response and 'name' in response:
            taxonomy_name = taxonomy['name'].strip() if taxonomy['name'].strip() else response.get('name', 'AI Generated Taxonomy')
            st.success(f"AI taxonomy '{taxonomy_name}' generated successfully!")
            st.cache_data.clear()
            time.sleep(1)
            st.rerun()
        else:
            st.error("Failed to start AI taxonomy generation")
    except APIError as e:
        st.error(f"Failed to generate AI taxonomy: {e}")


def save_unified_taxonomy(taxonomy: Dict[str, Any], silent: bool = False):
    """Save the unified taxonomy via API. Handles both creation and updates."""
    try:
        # Check for existing ID
        taxonomy_id = taxonomy.get("loaded_taxonomy_id") or taxonomy.get("id")
        
        taxonomy_payload = {
            "name": taxonomy["name"],
            "description": taxonomy["description"],
            "project_id": taxonomy["project_id"],
            "taxonomy": {
                "name": taxonomy["name"],
                "description": taxonomy["description"],
                "children": taxonomy["categories"]
            }
        }

        if taxonomy_id:
            # UPDATE existing taxonomy
            if silent:
                api_client.put(f"/api/v1/taxonomy/{taxonomy_id}", taxonomy_payload)
                st.cache_data.clear()
                return
            
            with st.spinner("Updating taxonomy..."):
                response = api_client.put(f"/api/v1/taxonomy/{taxonomy_id}", taxonomy_payload)
            
            st.success(f"Taxonomy '{taxonomy['name']}' updated successfully!")
        else:
            # CREATE new taxonomy
            if silent:
                response = api_client.post("/api/v1/taxonomy/", taxonomy_payload)
                if response and response.get("id"):
                    taxonomy["id"] = response["id"]
                    taxonomy["loaded_taxonomy_id"] = response["id"]
                st.cache_data.clear()
                return

            with st.spinner("Creating taxonomy..."):
                response = api_client.post("/api/v1/taxonomy/", taxonomy_payload)

            if response and response.get("id"):
                taxonomy["id"] = response["id"]
                taxonomy["loaded_taxonomy_id"] = response["id"]

            st.success(f"Taxonomy '{taxonomy['name']}' created successfully!")
        
        # Only clear state and rerun if not silent
        if not silent:
            if "unified_taxonomy" in st.session_state:
                del st.session_state.unified_taxonomy
            st.cache_data.clear()
            st.rerun()

    except APIError as e:
        if not silent:
            st.error(f"Failed to save taxonomy: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")


def save_manual_taxonomy(taxonomy: Dict[str, Any]):
    """Save the manual taxonomy via API. Handles both creation and updates."""
    try:
        taxonomy_id = taxonomy.get("id")
        
        # Prepare the taxonomy data - wrap categories in a structure expected by API
        taxonomy_payload = {
            "name": taxonomy["name"],
            "description": taxonomy["description"],
            "project_id": taxonomy["project_id"],
            "taxonomy": {
                "name": taxonomy["name"],
                "description": taxonomy["description"],
                "children": taxonomy["categories"]  # API expects a root with children
            }
        }

        if taxonomy_id:
            with st.spinner("Updating taxonomy..."):
                response = api_client.put(f"/api/v1/taxonomy/{taxonomy_id}", taxonomy_payload)
            st.success(f"Manual taxonomy '{taxonomy['name']}' updated successfully!")
        else:
            with st.spinner("Creating taxonomy..."):
                response = api_client.post("/api/v1/taxonomy/", taxonomy_payload)
            st.success(f"Manual taxonomy '{taxonomy['name']}' created successfully!")

        # Clear the form
        if "manual_taxonomy" in st.session_state:
            del st.session_state.manual_taxonomy
        st.cache_data.clear()
        st.rerun()

    except APIError as e:
        st.error(f"Failed to save taxonomy: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")


def generate_ai_taxonomy_sync(name: str, project_id: int, document_ids: List[int], depth: int, generator: str, domain: str, batch_size: int, category_limits: Optional[List[int]] = None, specificity_level: int = 1):
    """Generate taxonomy using AI synchronously."""
    try:
        with st.spinner("Generating AI taxonomy..."):
            request_data = {
                "project_id": project_id,
                "name": name,
                "documents": document_ids,
                "depth": depth,
                "generator": generator,
                "domain": domain,
                "batch_size": batch_size,
                "specificity_level": specificity_level
            }
            if category_limits is not None:
                request_data["category_limits"] = category_limits
            response = api_client.post("/api/v1/taxonomy/generate", request_data)
        if response is None:
            st.error("Failed to generate AI taxonomy - API call returned no response")
        elif 'job_id' in response:
            job_id = response['job_id']
            st.success(f"✅ AI Taxonomy generation started! Job ID: {job_id}")
            # Monitor job synchronously in-view (Wizard-style)
            success = monitor_job_synchronously(job_id, success_text="Taxonomy generation completed!")
            if success:
                st.cache_data.clear()
                st.rerun()
        elif 'id' in response and 'name' in response:
            st.success(f"AI taxonomy '{name}' generated successfully!")
            st.cache_data.clear()
            time.sleep(1)
            st.rerun()
        else:
            st.error("Failed to start AI taxonomy generation")
    except APIError as e:
        st.error(f"Failed to generate AI taxonomy: {e}")


def extend_taxonomy_with_ai(taxonomy: Dict[str, Any], additional_depth: int, generator: str, domain: str, selected_documents: Optional[List[int]] = None):
    """Extend the manual taxonomy using AI."""
    try:
        with st.spinner("Generating downstream taxonomy..."):
            # Prepare the taxonomy data for extension
            taxonomy_data = {
                "name": taxonomy["name"],
                "description": taxonomy["description"],
                "children": taxonomy["categories"]  # API expects children at root level
            }

            # Retrieve batch_size and processing_mode from taxonomy config if available
            ai_config = taxonomy.get("ai_config", {})
            batch_size = ai_config.get("batch_size", 10)
            processing_mode = ai_config.get("processing_mode", "fast")

            # Call the extend API directly with taxonomy data
            request_data = {
                "taxonomy_data": taxonomy_data,
                "project_id": taxonomy["project_id"],  # Already a string
                "additional_depth": additional_depth,
                "generator": generator,
                "domain": domain,
                "batch_size": batch_size,
                "processing_mode": processing_mode
            }

            if selected_documents:
                request_data["documents"] = selected_documents

            extended_response = api_client.post("/api/v1/taxonomy/extend", request_data)

            if extended_response is None:
                st.error("Failed to extend taxonomy - API call returned no response")
            elif 'job_id' in extended_response:
                job_id = extended_response['job_id']
                st.success(f"✅ Taxonomy extension started! Job ID: {job_id}")
                # Monitor job synchronously in-view (Wizard-style)
                success = monitor_job_synchronously(job_id, success_text="Taxonomy extension completed!")
                if success:
                    st.cache_data.clear()
                    st.rerun()
            elif "extended_taxonomy" in extended_response:
                # Handle category extension response (new format for temp taxonomies)
                st.success(f"Taxonomy extended successfully!")

                extended_taxonomy_data = extended_response.get("extended_taxonomy", {})
                extended_category_data = extended_taxonomy_data.get('taxonomy', {})

                # Update the session state with extended categories
                if "children" in extended_category_data:
                    taxonomy["categories"] = extended_category_data["children"]

                st.info("Your manual taxonomy has been updated with AI-generated subcategories. You can continue editing or save the taxonomy.")
                st.rerun()

            elif "id" in extended_response:
                # Handle legacy response format for saved taxonomies
                st.success(f"Taxonomy extended successfully! Taxonomy updated in place.")

                # Update the current taxonomy with the extended version
                try:
                    extended_details = api_client.get(f"/api/v1/taxonomy/{extended_response['id']}")
                    extended_taxonomy_data = {}
                    if extended_details is None:
                        st.warning("Taxonomy extended successfully, but failed to load details")
                    else:
                        extended_taxonomy_data = extended_details.get('taxonomy', {})

                    # Update the session state with extended categories
                    if "children" in extended_taxonomy_data:
                        taxonomy["categories"] = extended_taxonomy_data["children"]

                    st.info("Your manual taxonomy has been updated with AI-generated subcategories. You can continue editing or save the taxonomy.")

                except APIError as e:
                    st.warning(f"Taxonomy extended successfully, but failed to load details: {e}")

                # Clear cache to refresh taxonomy list
                st.cache_data.clear()
                st.rerun()

            else:
                st.error("Failed to extend taxonomy.")

    except APIError as e:
        st.error(f"Failed to extend taxonomy: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")


def extend_category_with_ai(path: str, taxonomy: Dict[str, Any], additional_depth: int, generator: str, domain: str):
    """Extend a specific category in the taxonomy using AI."""
    try:
        with st.spinner("Extending category with AI..."):
            # Find the category to extend based on the path
            category = get_category_by_path(path, taxonomy)
            if not category:
                st.error("Category not found.")
                return

            # Prepare the category data for extension
            category_data = {
                "name": category.get("name", ""),
                "description": category.get("description", ""),
                "children": category.get("children", [])
            }

            # Retrieve batch_size and processing_mode from taxonomy config if available
            ai_config = taxonomy.get("ai_config", {})
            batch_size = ai_config.get("batch_size", 10)
            processing_mode = ai_config.get("processing_mode", "fast")

            # Call the extend API with category-specific data
            request_data = {
                "taxonomy_data": category_data,
                "project_id": taxonomy["project_id"],  # Already a string
                "additional_depth": additional_depth,
                "generator": generator,
                "domain": domain,
                "batch_size": batch_size,
                "processing_mode": processing_mode
            }

            extended_response = api_client.post("/api/v1/taxonomy/extend", request_data)

            if extended_response is None:
                st.error("Failed to extend category - API call returned no response")
            elif "extended_taxonomy" in extended_response:
                # Handle category extension response (new format for temp taxonomies)
                st.success(f"Category extended successfully!")

                extended_taxonomy_data = extended_response.get("extended_taxonomy", {})
                extended_category_data = extended_taxonomy_data.get('taxonomy', {})

                # Update the category in the taxonomy with the extended children
                if "children" in extended_category_data:
                    category["children"] = extended_category_data["children"]

                st.info("Category has been extended with AI-generated subcategories.")
                st.rerun()

            elif "id" in extended_response:
                # Handle legacy response format (shouldn't happen for category extensions)
                st.warning("Unexpected response format - category extension may not have worked correctly")
            else:
                st.error("Failed to extend category.")

    except APIError as e:
        st.error(f"Failed to extend category: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")