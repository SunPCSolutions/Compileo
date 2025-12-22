"""Export utilities for taxonomy operations."""

import streamlit as st
import json
from typing import List, Dict, Any
import time

from src.compileo.features.gui.services.api_client import api_client, APIError


def export_taxonomies_json(taxonomies: List[Dict[str, Any]]):
    """Export taxonomies as JSON file."""
    try:
        # Prepare export data
        export_data = {
            "export_timestamp": str(time.time()),
            "total_taxonomies": len(taxonomies),
            "taxonomies": []
        }

        for taxonomy in taxonomies:
            try:
                # Get full taxonomy details
                response = api_client.get(f"/api/v1/taxonomy/{taxonomy['id']}")
                taxonomy_detail = response
                export_data["taxonomies"].append(taxonomy_detail)
            except APIError:
                # If detailed fetch fails, include basic info
                export_data["taxonomies"].append(taxonomy)

        # Create JSON string
        json_str = json.dumps(export_data, indent=2)

        # Create download button
        st.download_button(
            label="📥 Download JSON Export",
            data=json_str,
            file_name=f"taxonomies_export_{int(time.time())}.json",
            mime="application/json",
            width='stretch'
        )

        st.success(f"Prepared export for {len(taxonomies)} taxonomies")

    except Exception as e:
        st.error(f"Failed to export taxonomies: {str(e)}")


def export_taxonomies_csv(taxonomies: List[Dict[str, Any]]):
    """Export taxonomy summary as CSV file."""
    try:
        import pandas as pd

        # Prepare data for CSV
        csv_data = []
        for taxonomy in taxonomies:
            csv_data.append({
                "ID": taxonomy.get("id"),
                "Name": taxonomy.get("name", "Unknown"),
                "Project_ID": taxonomy.get("project_id"),
                "Categories_Count": taxonomy.get("categories_count", 0),
                "Confidence_Score": taxonomy.get("confidence_score", 0.0),
                "Created_At": taxonomy.get("created_at", "Unknown")
            })

        df = pd.DataFrame(csv_data)
        csv_str = df.to_csv(index=False)

        # Create download button
        st.download_button(
            label="📥 Download CSV Export",
            data=csv_str,
            file_name=f"taxonomies_summary_{int(time.time())}.csv",
            mime="text/csv",
            width='stretch'
        )

        st.success(f"Prepared CSV export for {len(taxonomies)} taxonomies")

    except ImportError:
        st.error("Pandas is required for CSV export. Please install pandas.")
    except Exception as e:
        st.error(f"Failed to export taxonomies as CSV: {str(e)}")