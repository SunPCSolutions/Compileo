"""Quality page for the Compileo GUI."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, Optional
import time
import os

from ..services.api_client import api_client, APIError
from ..services.job_monitoring_service import monitor_job_synchronously


def render_quality():
    """Render the comprehensive quality metrics page."""
    st.title("📊 Dataset Quality Analysis")

    # Initialize session state for quality analysis
    if 'quality_job_id' not in st.session_state:
        st.session_state.quality_job_id = None
    if 'quality_status' not in st.session_state:
        st.session_state.quality_status = None
    if 'quality_results' not in st.session_state:
        st.session_state.quality_results = None

    # Main content in tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Analysis", "📈 Metrics Dashboard", "📋 History"])

    with tab1:
        render_quality_analysis_tab()

    with tab2:
        render_metrics_dashboard_tab()

    with tab3:
        render_quality_history_tab()


def render_quality_analysis_tab():
    """Render the quality analysis tab with dataset selection and analysis trigger."""
    st.header("Dataset Quality Analysis")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Project selection
        st.subheader("📁 Select Project")
        from ..services.document_api_service import get_projects
        projects = get_projects()
        if projects:
            project_options = {f"{p['name']} (ID: {p['id']})": p['id'] for p in projects}
            selected_project_display = st.selectbox(
                "Select Project",
                options=list(project_options.keys()),
                key="quality_project_select",
                help="Choose the project containing datasets for quality analysis"
            )
            selected_project_id = project_options[selected_project_display]
        else:
            st.error("⚠️ No projects available. Please create a project first.")
            return

        # Dataset selection
        st.subheader("Select Dataset")

        # Get available datasets for the selected project
        available_datasets = get_available_datasets(selected_project_id)

        if available_datasets:
            selected_dataset = st.selectbox(
                "Choose a dataset to analyze:",
                options=list(available_datasets.keys()),
                format_func=lambda x: available_datasets[x]['name'],
                help="Select a dataset file for quality analysis"
            )

            dataset_info = available_datasets[selected_dataset]
            st.info(f"📁 **{dataset_info['name']}** - {dataset_info['size']} entries")

            # Configuration options
            with st.expander("⚙️ Analysis Configuration", expanded=False):
                st.subheader("Quality Analysis Model")

                # Model selection for quality analysis
                quality_model = st.selectbox(
                    "AI Model for Quality Analysis",
                    options=["grok", "gemini", "ollama", "openai"],
                    index=0,
                    help="AI model used for quality analysis. Should match the model used for dataset generation for consistency."
                )

                st.subheader("Quality Metrics Configuration")

                col_a, col_b = st.columns(2)

                with col_a:
                    enable_diversity = st.checkbox("Diversity Analysis", value=True,
                                                 help="Analyze lexical and semantic diversity")
                    enable_bias = st.checkbox("Bias Detection", value=True,
                                            help="Detect potential biases in the dataset")
                    enable_difficulty = st.checkbox("Difficulty Assessment", value=True,
                                                  help="Assess question complexity and readability")

                with col_b:
                    threshold = st.slider("Overall Quality Threshold", 0.0, 1.0, 0.7,
                                        help="Minimum quality score required to pass")
                    output_format = st.selectbox("Output Format", ["json", "html", "pdf"],
                                               help="Format for quality reports")

                # Advanced configuration
                with st.expander("Advanced Settings"):
                    diversity_threshold = st.slider("Diversity Threshold", 0.0, 1.0, 0.6)
                    bias_threshold = st.slider("Bias Threshold", 0.0, 1.0, 0.8)
                    difficulty_target = st.slider("Target Difficulty", 0.0, 1.0, 0.5)

            # Analysis trigger
            if st.button("🚀 Start Quality Analysis", type="primary", width='stretch'):
                start_quality_analysis(
                    dataset_info['path'],
                    {
                        "enabled": True,
                        "diversity": {"enabled": enable_diversity, "threshold": diversity_threshold},
                        "bias": {"enabled": enable_bias, "threshold": bias_threshold},
                        "difficulty": {"enabled": enable_difficulty, "threshold": difficulty_target}
                    },
                    threshold,
                    output_format,
                    quality_model
                )
        else:
            st.warning("No datasets found. Please generate or upload a dataset first.")

    with col2:
        # Analysis status
        # Synchronous monitoring is now handled within the button handler
        render_analysis_status()


def render_analysis_status():
    """Render the current analysis status and progress."""
    st.subheader("Analysis Status")

    if st.session_state.quality_job_id:
        try:
            # Get current results from state or API
            if st.session_state.quality_results:
                status = st.session_state.quality_results
            else:
                status = api_client.get_quality_results(st.session_state.quality_job_id)
                
            if status is None:
                st.error("Analysis job not found.")
                return

            job_status = status.get('status', 'unknown')
            
            if job_status == 'running':
                st.info("🔄 Analysis in progress...")
            elif job_status == 'completed':
                st.success("✅ Analysis completed!")
                st.session_state.quality_results = status
                st.session_state.quality_status = 'completed'

                # Show summary
                summary = status.get('summary', {})
                if summary:
                    st.metric("Overall Score", f"{summary.get('overall_score', 0):.3f}")
                    st.metric("Passed Metrics", f"{summary.get('passed_metrics', 0)}/{summary.get('total_metrics', 0)}")
            elif job_status == 'failed':
                st.error("❌ Analysis failed")
                error_details = status.get('details', {}).get('error', status.get('error', 'Unknown error'))
                st.error(f"Error: {error_details}")

        except Exception as e:
            st.error(f"Error displaying analysis status: {e}")
    else:
        st.info("No analysis in progress")


def render_metrics_dashboard_tab():
    """Render the metrics dashboard with interactive visualizations."""
    st.header("Quality Metrics Dashboard")

    if not st.session_state.quality_results:
        st.info("Run a quality analysis first to see the dashboard.")
        return

    results = st.session_state.quality_results
    summary = results.get('summary', {})
    metrics = results.get('metrics', {})
    details = results.get('details', {})

    # Overall summary cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        overall_score = summary.get('overall_score', 0)
        passed = summary.get('passed', False)
        delta_color = "normal" if passed else "inverse"
        st.metric("Overall Quality", f"{overall_score:.3f}",
                 delta="PASS" if passed else "FAIL", delta_color=delta_color)

    with col2:
        st.metric("Dataset Size", results.get('dataset_size', 0))

    with col3:
        passed_metrics = summary.get('passed_metrics', 0)
        total_metrics = summary.get('total_metrics', 0)
        st.metric("Passed Metrics", f"{passed_metrics}/{total_metrics}")

    with col4:
        failed_metrics = summary.get('failed_metrics', 0)
        st.metric("Failed Metrics", failed_metrics)

    # Individual metrics breakdown
    st.subheader("📊 Individual Metrics")

    if metrics:
        # Create metrics dataframe for visualization
        metrics_df = pd.DataFrame([
            {
                'Metric': name.title(),
                'Score': score,
                'Threshold': details.get('results', {}).get(name, {}).get('threshold'),
                'Passed': details.get('results', {}).get(name, {}).get('passed', False)
            }
            for name, score in metrics.items()
        ])

        # Metrics bar chart
        fig = px.bar(
            metrics_df,
            x='Metric',
            y='Score',
            color='Passed',
            color_discrete_map={True: 'green', False: 'red'},
            title='Quality Metrics Scores',
            range_y=[0, 1]
        )
        fig.add_hline(y=0.7, line_dash="dash", line_color="orange",
                     annotation_text="Default Threshold")
        st.plotly_chart(fig, width='stretch')

        # Detailed metrics table
        st.subheader("📋 Detailed Results")
        st.dataframe(
            metrics_df.style.apply(lambda x: ['background-color: lightgreen' if v else 'background-color: lightcoral'
                                            for v in x], subset=['Passed']),
            width='stretch'
        )

    # Issues and recommendations
    issues = summary.get('issues', [])
    if issues:
        st.subheader("⚠️ Issues Found")
        for issue in issues[:5]:  # Show first 5 issues
            st.warning(issue)

    # Detailed breakdowns for each metric
    if details.get('results'):
        render_detailed_metric_breakdowns(details['results'])


def render_detailed_metric_breakdowns(results: Dict[str, Any]):
    """Render detailed breakdowns for each quality metric."""
    st.subheader("🔍 Detailed Analysis")

    for metric_name, metric_data in results.items():
        with st.expander(f"{metric_name.title()} Analysis", expanded=False):
            details = metric_data.get('details', {})

            if metric_name == 'diversity':
                render_diversity_breakdown(details)
            elif metric_name == 'bias':
                render_bias_breakdown(details)
            elif metric_name == 'difficulty':
                render_difficulty_breakdown(details)
            else:
                st.json(details)


def render_diversity_breakdown(details: Dict[str, Any]):
    """Render diversity metric detailed breakdown."""
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Lexical Diversity", f"{details.get('lexical_diversity', 0):.3f}")
        st.metric("Semantic Diversity", f"{details.get('semantic_diversity', 0):.3f}")

    with col2:
        st.metric("Topic Balance", f"{details.get('topic_balance', 0):.3f}")

    # Diversity radar chart
    categories = ['Lexical', 'Semantic', 'Topic Balance']
    values = [
        details.get('lexical_diversity', 0),
        details.get('semantic_diversity', 0),
        details.get('topic_balance', 0)
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Diversity Metrics'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
        title="Diversity Profile"
    )
    st.plotly_chart(fig)


def render_bias_breakdown(details: Dict[str, Any]):
    """Render bias detection detailed breakdown."""
    st.metric("Bias Score", f"{details.get('bias_score', 0):.3f}")

    # Bias indicators
    bias_indicators = details.get('bias_indicators', {})
    if bias_indicators:
        st.subheader("Bias Indicators")
        for indicator, score in bias_indicators.items():
            st.metric(indicator.replace('_', ' ').title(), f"{score:.3f}")


def render_difficulty_breakdown(details: Dict[str, Any]):
    """Render difficulty assessment detailed breakdown."""
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Average Difficulty", f"{details.get('avg_difficulty', 0):.3f}")

    with col2:
        st.metric("Complexity Score", f"{details.get('complexity_score', 0):.3f}")

    with col3:
        st.metric("Readability Score", f"{details.get('readability_score', 0):.3f}")

    # Difficulty distribution
    distribution = details.get('difficulty_distribution', {})
    if distribution:
        dist_df = pd.DataFrame(list(distribution.items()), columns=['Level', 'Count'])
        fig = px.pie(dist_df, values='Count', names='Level', title='Difficulty Distribution')
        st.plotly_chart(fig)


def render_quality_history_tab():
    """Render the quality analysis history tab."""
    st.header("Analysis History")

    try:
        history = api_client.get_quality_history(limit=20)

        if history and history.get('analyses'):
            # Convert to dataframe for display
            history_df = pd.DataFrame(history['analyses'])

            if not history_df.empty:
                # Format the dataframe
                display_df = history_df[[
                    'job_id', 'status', 'summary', 'completed_at'
                ]].copy()

                display_df['completed_at'] = pd.to_datetime(display_df['completed_at']).dt.strftime('%Y-%m-%d %H:%M')

                st.dataframe(display_df, width='stretch')

                # Summary statistics
                status_counts = history_df['status'].value_counts()
                st.subheader("Summary")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Analyses", len(history_df))
                with col2:
                    completed = status_counts.get('completed', 0)
                    st.metric("Completed", completed)
                with col3:
                    failed = status_counts.get('failed', 0)
                    st.metric("Failed", failed)
            else:
                st.info("No analysis history found.")
        else:
            st.info("No analysis history available.")

    except APIError as e:
        st.error(f"Failed to load analysis history: {e}")


def get_available_datasets(project_id: int) -> Dict[str, Dict[str, Any]]:
    """Get list of available datasets for analysis."""
    datasets = {}

    # First try to get datasets from the API (database-backed)
    try:
        response = api_client.list_datasets(project_id=project_id, page=1, page_size=100)
        if response and 'datasets' in response:
            for dataset in response['datasets']:
                file_path = dataset.get('name', '')  # Use filename as key
                if file_path:
                    full_path = f"storage/datasets/{project_id}/{file_path}"
                    datasets[file_path] = {
                        'name': dataset.get('name', file_path),
                        'path': full_path,
                        'size': dataset.get('entries_count', 0),
                        'type': dataset.get('format_type', 'jsonl')
                    }
    except Exception as e:
        # If API call fails, fall back to filesystem scanning
        pass

    # Fallback: Check test_outputs directory for legacy support
    if not datasets:  # Only if no datasets found via API
        test_outputs_dir = "test_outputs"
        if os.path.exists(test_outputs_dir):
            for file in os.listdir(test_outputs_dir):
                if file.endswith('.jsonl'):
                    file_path = os.path.join(test_outputs_dir, file)
                    try:
                        # Count lines for dataset size
                        with open(file_path, 'r', encoding='utf-8') as f:
                            size = sum(1 for _ in f)

                        datasets[file_path] = {
                            'name': file,
                            'path': file_path,
                            'size': size,
                            'type': 'jsonl'
                        }
                    except Exception:
                        continue

    return datasets


def start_quality_analysis(dataset_file: str, config: Dict[str, Any],
                          threshold: float, output_format: str, quality_model: str):
    """Start a quality analysis job."""
    try:
        with st.spinner("Starting quality analysis..."):
            response = api_client.analyze_quality(
                dataset_file=dataset_file,
                config=config,
                threshold=threshold,
                output_format=output_format,
                quality_model=quality_model
            )

        job_id = response.get('job_id')
        if job_id:
            st.session_state.quality_job_id = job_id
            st.session_state.quality_status = 'running'
            st.session_state.quality_results = None
            st.success(f"✅ Analysis started! Job ID: {job_id}")
            
            # Monitor job synchronously in-view (Wizard-style)
            # Use general endpoint as quality uses /quality/{id}/results for status too
            success = monitor_job_synchronously(job_id, success_text="Quality analysis completed!")
            
            if success:
                # Get results and update state
                results = api_client.get_quality_results(job_id)
                st.session_state.quality_results = results
                st.session_state.quality_status = 'completed'
                st.rerun()
        else:
            st.error("Failed to start analysis - no job ID returned")

    except APIError as e:
        st.error(f"Failed to start quality analysis: {e}")

