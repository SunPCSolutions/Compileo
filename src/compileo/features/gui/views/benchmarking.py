"""Benchmarking dashboard for the Compileo GUI."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import json
from io import StringIO
from typing import Dict, Any, Optional, List
import uuid

from ..services.api_client import benchmarking_api_client
from ..services.job_monitoring_service import monitor_job_synchronously
import sqlite3
import os
import threading

# Thread-safe database connection (copied from storage module)
db_lock = threading.Lock()

def get_db_connection():
    """Get database connection with thread safety."""
    db_path = os.getenv('DATABASE_PATH', 'storage/database.db')
    with db_lock:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn


def render_benchmarking():
    """Render the comprehensive benchmarking dashboard."""
    st.title("📊 AI Model Benchmarking Dashboard")

    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Overview", "🔄 Run Benchmarks", "⚖️ Model Comparison",
        "📚 History", "🏆 Leaderboard"
    ])

    with tab1:
        render_overview_tab()

    with tab2:
        render_run_benchmarks_tab()

    with tab3:
        render_model_comparison_tab()

    with tab4:
        render_history_tab()

    with tab5:
        render_leaderboard_tab()


def render_overview_tab():
    """Render the overview dashboard with key metrics and visualizations."""
    st.header("Benchmarking Overview")

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_suite = st.selectbox("Benchmark Suite", ["GLUE", "SuperGLUE", "MMLU", "Medical"], index=0)
    with col2:
        selected_metric = st.selectbox("Primary Metric", ["accuracy", "f1", "precision", "recall"], index=0)
    with col3:
        days_back = st.slider("Days Back", 1, 90, 30)

    # Get recent benchmark results from database
    try:
        db_conn = get_db_connection()
        cursor = db_conn.cursor()

        # Get recent results - handle empty results table
        cursor.execute("""
            SELECT
                bj.id,
                COALESCE(json_extract(bj.parameters, '$.ai_config.selected_model'), json_extract(bj.parameters, '$.ai_config.model_name'), bj.model_name) as model_name,
                json_extract(bj.parameters, '$.benchmark_suite') as benchmark_suite,
                bj.status,
                bj.created_at,
                bj.completed_at,
                CASE WHEN br.metrics IS NOT NULL THEN CAST(json_extract(br.metrics, '$.accuracy') AS FLOAT) ELSE NULL END as accuracy,
                CASE WHEN br.metrics IS NOT NULL THEN CAST(json_extract(br.metrics, '$.f1') AS FLOAT) ELSE NULL END as f1,
                CASE WHEN br.metrics IS NOT NULL THEN CAST(json_extract(br.metrics, '$.precision') AS FLOAT) ELSE NULL END as precision,
                CASE WHEN br.metrics IS NOT NULL THEN CAST(json_extract(br.metrics, '$.recall') AS FLOAT) ELSE NULL END as recall
            FROM benchmark_jobs bj
            LEFT JOIN benchmark_results br ON bj.id = br.job_id
            WHERE bj.created_at >= date('now', '-{} days')
            ORDER BY bj.created_at DESC
            LIMIT 50
        """.format(days_back))

        rows = cursor.fetchall()
        if rows:
            df = pd.DataFrame(rows, columns=['id', 'model_name', 'benchmark_suite', 'status', 'created_at', 'completed_at', 'accuracy', 'f1', 'precision', 'recall'])

            # Key metrics cards
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                avg_score = df[selected_metric].mean() if selected_metric in df.columns and not df[selected_metric].empty else 0
                st.metric(f"Average {selected_metric.title()}", f"{avg_score:.3f}")
            with col2:
                total_runs = len(df)
                st.metric("Total Benchmark Runs", total_runs)
            with col3:
                unique_models = df['model_name'].nunique() if 'model_name' in df.columns else 0
                st.metric("Models Evaluated", unique_models)
            with col4:
                latest_run = pd.to_datetime(df['completed_at']).max() if 'completed_at' in df.columns and not df['completed_at'].isna().all() else pd.Timestamp.now()
                st.metric("Latest Run", latest_run.strftime("%Y-%m-%d") if pd.notna(latest_run) else "N/A")

            # Performance visualization
            st.subheader("Performance Trends")
            if not df.empty:
                fig = create_performance_trend_chart(df, selected_metric)
                st.plotly_chart(fig, width='stretch')

            # Recent results table
            st.subheader("Recent Benchmark Results")
            display_results_table(df.head(10))

            # Export options
            st.subheader("Export Data")
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("📊 Export as CSV", key="export_overview_csv"):
                    export_benchmark_data_csv(df.head(10))
            with col2:
                if st.button("📋 Export as JSON", key="export_overview_json"):
                    export_benchmark_data_json(df.head(10))
            with col3:
                if st.button("📄 Export Report (PDF)", key="export_overview_pdf"):
                    export_benchmark_report_pdf(df.head(10), selected_suite, selected_metric)

        else:
            st.info("No benchmark results found. Run some benchmarks to see data here.")

    except Exception as e:
        st.error(f"Failed to load benchmark data: {str(e)}")


def render_run_benchmarks_tab():
    """Render the tab for running new benchmarks."""
    st.header("Run AI Model Benchmarks")

    # Load saved settings
    from ..utils.settings_storage import settings_storage
    saved_settings = settings_storage.load_settings()

    # AI Provider Selection (outside form for dynamic updates)
    st.subheader("🤖 AI Model Selection")

    # Initialize session state for provider if not exists
    if 'benchmark_provider' not in st.session_state:
        st.session_state.benchmark_provider = "ollama"

    # Provider selection with callback to update session state
    def update_provider():
        st.session_state.benchmark_provider = st.session_state.provider_selector

    provider = st.selectbox(
        "AI Provider",
        ["ollama", "gemini", "grok", "openai"],
        index=["ollama", "gemini", "grok", "openai"].index(st.session_state.benchmark_provider),
        key="provider_selector",
        help="Select the AI provider for benchmarking",
        on_change=update_provider
    )

    # Get API keys and URLs from settings
    ollama_url = saved_settings.get("ollama_base_url", "http://localhost:11434")
    gemini_key = saved_settings.get("gemini_api_key", "")
    grok_key = saved_settings.get("grok_api_key", "")
    openai_key = saved_settings.get("openai_api_key", "")

    # Get available models based on selected provider
    if provider == "ollama":
        from ..utils.ollama_utils import get_ollama_models
        available_models = get_ollama_models(ollama_url)
        if not available_models:
            st.warning("Could not fetch Ollama models. Please check your Ollama server connection.")
            available_models = ["mistral:latest", "llama2:latest"]
    elif provider == "gemini":
        from ..utils.ollama_utils import get_gemini_models
        available_models = get_gemini_models(gemini_key)
        if not available_models:
            st.warning("Could not fetch Gemini models. Please check your API key.")
            available_models = ["gemini-2.5-flash"]
    elif provider == "grok":
        from ..utils.ollama_utils import get_grok_models
        available_models = get_grok_models(grok_key)
        if not available_models:
            st.warning("Could not fetch Grok models. Please check your API key.")
            available_models = ["grok-4-fast-reasoning"]
    elif provider == "openai":
        from ..utils.ollama_utils import get_openai_models
        available_models = get_openai_models(openai_key)
        if not available_models:
            if openai_key:
                st.warning("Could not fetch OpenAI models. Please check your API key.")
            available_models = ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]

    # Model selection (dynamic based on provider)
    selected_model = st.selectbox(
        f"{provider.title()} Model",
        options=available_models,
        index=0,
        help=f"Select a {provider.title()} model for benchmarking"
    )

    with st.form("benchmark_form"):
        col1, col2 = st.columns(2)

        with col1:
            # Display selected provider and model (read-only in form)
            st.write(f"**Selected Provider:** {provider.title()}")
            st.write(f"**Selected Model:** {selected_model}")

            # Ollama parameters (only show for Ollama)
            ollama_params = {}
            if provider == "ollama":
                st.markdown("**Ollama Parameters:**")

                # Create parameter grid
                param_cols = st.columns(4)

                # Row 1: temperature, repeat_penalty, top_p, top_k
                with param_cols[0]:
                    temperature = st.text_input(
                        "Temperature",
                        value=str(saved_settings.get("benchmarking_ollama_temperature", "0.1")),
                        help="(0.0-2.0)",
                        key="benchmark_temp"
                    )

                with param_cols[1]:
                    repeat_penalty = st.text_input(
                        "Repeat Penalty",
                        value=str(saved_settings.get("benchmarking_ollama_repeat_penalty", "1.2")),
                        help="(0.0-2.0)",
                        key="benchmark_repeat"
                    )

                with param_cols[2]:
                    top_p = st.text_input(
                        "Top P",
                        value=str(saved_settings.get("benchmarking_ollama_top_p", "0.9")),
                        help="(0.0-1.0)",
                        key="benchmark_top_p"
                    )

                with param_cols[3]:
                    top_k = st.text_input(
                        "Top K",
                        value=str(saved_settings.get("benchmarking_ollama_top_k", "20")),
                        help="(0-100)",
                        key="benchmark_top_k"
                    )

                # Row 2: num_predict, num_ctx, seed
                param_cols2 = st.columns(4)

                with param_cols2[0]:
                    num_predict = st.text_input(
                        "Num Predict",
                        value=str(saved_settings.get("benchmarking_ollama_num_predict", "1024")),
                        help="(1-4096)",
                        key="benchmark_num_predict"
                    )

                with param_cols2[1]:
                    num_ctx = st.text_input(
                        "Num Ctx",
                        value=str(saved_settings.get("benchmarking_ollama_num_ctx", "32768")),
                        help="Context window size in tokens",
                        key="benchmark_num_ctx"
                    )

                with param_cols2[2]:
                    seed = st.text_input(
                        "Seed",
                        value=str(saved_settings.get("benchmarking_ollama_seed", "")),
                        help="(0-4294967295)",
                        key="benchmark_seed"
                    )

                # Store Ollama parameters
                ollama_params = {
                    "temperature": float(temperature) if temperature else 0.1,
                    "repeat_penalty": float(repeat_penalty) if repeat_penalty else 1.2,
                    "top_p": float(top_p) if top_p else 0.9,
                    "top_k": int(top_k) if top_k else 20,
                    "num_predict": int(num_predict) if num_predict else 1024,
                    "num_ctx": int(num_ctx) if num_ctx else 32768,
                    "seed": int(seed) if seed else None
                }

        with col2:
            st.subheader("Benchmark Configuration")
            benchmark_suite = st.selectbox("Benchmark Suite",
                                         ["glue", "superglue", "mmlu", "medical"],
                                         index=0)
            tasks = st.multiselect("Specific Tasks (optional)",
                                 ["cola", "sst2", "mrpc", "qqp", "mnli", "qnli", "rte", "wnli"],
                                 help="Leave empty to run all tasks in the suite")

        # Advanced options (only for Gemini and Grok)
        custom_config = ""
        if provider in ["gemini", "grok"]:
            with st.expander("Advanced Options"):
                custom_config = st.text_area("Custom Configuration (JSON)",
                                           placeholder='{"batch_size": 32, "max_length": 512}',
                                           height=100)

        submitted = st.form_submit_button("🚀 Start Benchmarking")

    if submitted:
        if not selected_model:
            st.error("Please select a model.")
            return

        # Prepare data
        model_info = {
            "name": selected_model,
            "provider": provider,
            "model": selected_model
        }
        
        config = {"tasks": tasks} if tasks else {}
        if 'custom_config' in locals() and custom_config:
            try:
                import json
                config.update(json.loads(custom_config))
            except json.JSONDecodeError:
                st.error("Invalid JSON in custom configuration.")
                return

        try:
            # Import job queue modules with fallback
            try:
                from src.compileo.features.jobhandle.enhanced_job_queue import enhanced_job_queue_manager
                from src.compileo.features.jobhandle.models import JobType
                from src.compileo.storage.src.project.database_repositories import BenchmarkJobRepository
            except ImportError:
                # Fallback for when running from different directory
                import sys
                import os
                sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
                from src.compileo.features.jobhandle.enhanced_job_queue import enhanced_job_queue_manager
                from src.compileo.features.jobhandle.models import JobType
                from src.compileo.storage.src.project.database_repositories import BenchmarkJobRepository

            # Prepare job parameters
            ai_config = {
                "provider": provider,
                "selected_model": selected_model
            }

            # Add Ollama parameters if Ollama is selected
            if provider == "ollama":
                ai_config["ollama_params"] = ollama_params

            job_params = {
                "benchmark_suite": benchmark_suite,
                "ai_config": ai_config,
                "benchmark_params": config
            }

            # Generate job ID and create record in database first
            job_id = str(uuid.uuid4())
            
            try:
                db_conn = get_db_connection()
                job_repo = BenchmarkJobRepository(db_conn)
                job_repo.create_job({
                    'id': job_id,
                    'model_name': selected_model,  # Use selected model as model_name
                    'status': 'pending',
                    'parameters': {
                        'benchmark_suite': benchmark_suite,
                        'ai_config': ai_config,
                        'benchmark_params': config
                    },
                    'created_at': datetime.utcnow(),
                    'updated_at': datetime.utcnow()
                })
                
                # Submit job to queue with pre-generated ID
                submitted_id = enhanced_job_queue_manager.submit_job(
                    job_type=JobType.BENCHMARKING,
                    parameters=job_params,
                    user_id="gui_user",
                    job_id=job_id
                )

                if submitted_id:
                    st.success(f"✅ Benchmarking started! Job ID: {submitted_id}")
                    # Show progress tracking
                    # Monitor job synchronously in-view (Wizard-style)
                    success = monitor_job_synchronously(submitted_id, success_text="Benchmarking completed!")
                    if success:
                        st.rerun()
                else:
                    st.error("Failed to submit benchmarking job to queue.")
            except Exception as e:
                st.error(f"Failed to create job record: {str(e)}")

        except Exception as e:
            st.error(f"Failed to start benchmarking: {str(e)}")


def render_model_comparison_tab():
    """Render the model comparison interface."""
    st.header("Model Performance Comparison")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Model selection
        st.subheader("Select Models to Compare")

        # Get available models from database
        try:
            db_conn = get_db_connection()
            cursor = db_conn.cursor()

            cursor.execute("""
                SELECT DISTINCT COALESCE(json_extract(parameters, '$.ai_config.selected_model'), json_extract(parameters, '$.ai_config.model_name'), model_name) as model_name
                FROM benchmark_jobs
                WHERE status = 'completed'
                AND (json_extract(parameters, '$.ai_config.selected_model') IS NOT NULL OR json_extract(parameters, '$.ai_config.model_name') IS NOT NULL OR model_name IS NOT NULL)
                ORDER BY model_name
            """)

            available_models = [row[0] for row in cursor.fetchall()]

            if available_models:
                selected_models = st.multiselect(
                    "Choose models to compare",
                    available_models,
                    max_selections=5,
                    help="Select up to 5 models for comparison"
                )
            else:
                st.info("No completed benchmark runs found. Run some benchmarks first.")
                selected_models = []

        except Exception as e:
            st.error(f"Failed to load model history: {str(e)}")
            selected_models = []

    with col2:
        # Comparison settings
        st.subheader("Comparison Settings")
        benchmark_suite = st.selectbox("Benchmark Suite", ["glue", "superglue", "mmlu"], index=0)
        metrics_to_compare = st.multiselect(
            "Metrics to Compare",
            ["accuracy", "f1", "precision", "recall"],
            default=["accuracy", "f1"]
        )

    if selected_models and len(selected_models) >= 2:
        if st.button("🔍 Compare Models", type="primary"):
            try:
                # Get comparison data from database
                db_conn = get_db_connection()
                cursor = db_conn.cursor()

                # Build query for selected models
                placeholders = ','.join('?' * len(selected_models))
                query = f"""
                    SELECT
                        COALESCE(json_extract(bj.parameters, '$.ai_config.selected_model'), json_extract(bj.parameters, '$.ai_config.model_name'), bj.model_name) as model_name,
                        CAST(json_extract(br.metrics, '$.accuracy') AS FLOAT) as accuracy,
                        CAST(json_extract(br.metrics, '$.f1') AS FLOAT) as f1,
                        CAST(json_extract(br.metrics, '$.precision') AS FLOAT) as precision,
                        CAST(json_extract(br.metrics, '$.recall') AS FLOAT) as recall
                    FROM benchmark_jobs bj
                    LEFT JOIN benchmark_results br ON bj.id = br.job_id
                    WHERE bj.status = 'completed'
                    AND json_extract(bj.parameters, '$.benchmark_suite') = ?
                    AND COALESCE(json_extract(bj.parameters, '$.ai_config.selected_model'), json_extract(bj.parameters, '$.ai_config.model_name'), bj.model_name) IN ({placeholders})
                """

                params = [benchmark_suite.lower()] + selected_models
                cursor.execute(query, params)
                rows = cursor.fetchall()

                if rows:
                    # Create comparison data structure
                    df = pd.DataFrame(rows, columns=['model_name', 'accuracy', 'f1', 'precision', 'recall'])

                    # Group by model and calculate averages
                    comparison_data = {}
                    for model in selected_models:
                        model_data = df[df['model_name'] == model]
                        if not model_data.empty:
                            comparison_data[model] = {
                                'accuracy': model_data['accuracy'].mean(),
                                'f1': model_data['f1'].mean(),
                                'precision': model_data['precision'].mean(),
                                'recall': model_data['recall'].mean()
                            }

                    if comparison_data:
                        display_model_comparison(comparison_data, metrics_to_compare)
                    else:
                        st.error("No comparison data available.")
                else:
                    st.error("No data found for selected models.")

            except Exception as e:
                st.error(f"Failed to compare models: {str(e)}")
    elif selected_models and len(selected_models) < 2:
        st.warning("Please select at least 2 models to compare.")


def render_history_tab():
    """Render the benchmarking history view."""
    st.header("Benchmarking History")

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        model_filter = st.text_input("Filter by Model Name", placeholder="e.g., gpt-4")
    with col2:
        suite_filter = st.selectbox("Filter by Suite", ["All", "glue", "superglue", "mmlu", "medical"], index=0)
    with col3:
        status_filter = st.selectbox("Filter by Status", ["All", "completed", "running", "failed"], index=0)

    try:
        # Get history data from database
        db_conn = get_db_connection()
        cursor = db_conn.cursor()

        query = """
            SELECT
                bj.id,
                COALESCE(json_extract(bj.parameters, '$.ai_config.selected_model'), json_extract(bj.parameters, '$.ai_config.model_name'), bj.model_name) as model_name,
                json_extract(bj.parameters, '$.benchmark_suite') as benchmark_suite,
                bj.status,
                bj.created_at,
                bj.completed_at,
                CAST(json_extract(br.metrics, '$.accuracy') AS FLOAT) as accuracy,
                CAST(json_extract(br.metrics, '$.f1') AS FLOAT) as f1
            FROM benchmark_jobs bj
            LEFT JOIN benchmark_results br ON bj.id = br.job_id
            WHERE bj.created_at >= date('now', '-90 days')
        """

        params = []

        if model_filter:
            query += " AND COALESCE(json_extract(bj.parameters, '$.ai_config.selected_model'), json_extract(bj.parameters, '$.ai_config.model_name'), bj.model_name) LIKE ?"
            params.append(f"%{model_filter}%")

        if suite_filter != "All":
            query += " AND json_extract(bj.parameters, '$.benchmark_suite') = ?"
            params.append(suite_filter.lower())

        if status_filter != "All":
            query += " AND bj.status = ?"
            params.append(status_filter)

        query += " ORDER BY bj.created_at DESC"

        cursor.execute(query, params)
        rows = cursor.fetchall()

        if rows:
            # Convert to DataFrame
            history_df = pd.DataFrame(rows, columns=['id', 'model_name', 'benchmark_suite', 'status', 'created_at', 'completed_at', 'accuracy', 'f1'])

            # Apply additional filters if needed
            filtered_history = history_df

            if filtered_history.empty:
                st.info("No history data matches the selected filters.")
            else:
                # Display as table
                st.dataframe(filtered_history, width='stretch')

                # Summary stats
                total_runs = len(filtered_history)
                completed_runs = len(filtered_history[filtered_history['status'] == 'completed'])
                success_rate = (completed_runs / total_runs * 100) if total_runs > 0 else 0

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Runs", total_runs)
                with col2:
                    st.metric("Completed Runs", completed_runs)
                with col3:
                    st.metric("Success Rate", f"{success_rate:.1f}%")

                # Export options for history
                st.subheader("Export History Data")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📊 Export History as CSV", key="export_history_csv"):
                        export_benchmark_data_csv(filtered_history)
                with col2:
                    if st.button("📋 Export History as JSON", key="export_history_json"):
                        export_benchmark_data_json(filtered_history)
        else:
            st.info("No benchmarking history available.")

    except Exception as e:
        st.error(f"Failed to load history: {str(e)}")


def render_leaderboard_tab():
    """Render the model leaderboard."""
    st.header("🏆 Model Leaderboard")

    col1, col2, col3 = st.columns(3)
    with col1:
        suite = st.selectbox("Benchmark Suite", ["glue", "superglue", "mmlu", "medical"], index=0)
    with col2:
        metric = st.selectbox("Ranking Metric", ["accuracy", "f1", "precision", "recall"], index=0)
    with col3:
        limit = st.slider("Show Top N", 5, 50, 10)

    try:
        # Get leaderboard data from database
        db_conn = get_db_connection()
        cursor = db_conn.cursor()

        cursor.execute("""
            SELECT
                COALESCE(json_extract(bj.parameters, '$.ai_config.selected_model'), json_extract(bj.parameters, '$.ai_config.model_name'), bj.model_name) as model_name,
                CAST(json_extract(br.metrics, ?) AS FLOAT) as score,
                COUNT(*) as evaluation_count
            FROM benchmark_jobs bj
            LEFT JOIN benchmark_results br ON bj.id = br.job_id
            WHERE bj.status = 'completed'
            AND json_extract(bj.parameters, '$.benchmark_suite') = ?
            AND json_extract(br.metrics, ?) IS NOT NULL
            GROUP BY COALESCE(json_extract(bj.parameters, '$.ai_config.selected_model'), json_extract(bj.parameters, '$.ai_config.model_name'), bj.model_name)
            ORDER BY score DESC
            LIMIT ?
        """, (f'$.{metric}', suite.lower(), f'$.{metric}', limit))

        rows = cursor.fetchall()
        if rows:
            # Create leaderboard DataFrame
            df = pd.DataFrame(rows, columns=['model_name', 'score', 'evaluation_count'])
            df['rank'] = range(1, len(df) + 1)

            # Add medal emojis for top 3
            df["Rank"] = df["rank"].apply(lambda x: f"🥇 #{x}" if x == 1 else f"🥈 #{x}" if x == 2 else f"🥉 #{x}" if x == 3 else f"#{x}")

            st.dataframe(
                df[["Rank", "model_name", "score", "evaluation_count"]].rename(columns={
                    "model_name": "Model",
                    "score": f"{metric.title()} Score",
                    "evaluation_count": "Evaluations"
                }),
                width='stretch',
                hide_index=True
            )

            # Additional stats
            st.caption(f"Total models evaluated: {len(df)}")

            # Export options for leaderboard
            st.subheader("Export Leaderboard")
            if st.button("📊 Export Leaderboard as CSV", key="export_leaderboard_csv"):
                export_benchmark_data_csv(df)
        else:
            st.info("No leaderboard data available for the selected criteria.")

    except Exception as e:
        st.error(f"Failed to load leaderboard: {str(e)}")


def create_performance_trend_chart(df: pd.DataFrame, metric: str) -> go.Figure:
    """Create a performance trend chart."""
    if df.empty:
        return go.Figure()

    # Prepare data for plotting - use the new flat structure
    plot_data = []
    for _, row in df.iterrows():
        if metric in df.columns and pd.notna(row[metric]):
            plot_data.append({
                "timestamp": pd.to_datetime(row.get("completed_at", datetime.now())),
                "score": row[metric],
                "model": row.get("model_name", "Unknown")
            })

    if plot_data:
        trend_df = pd.DataFrame(plot_data)
        fig = px.line(
            trend_df,
            x="timestamp",
            y="score",
            color="model",
            title=f"{metric.title()} Performance Over Time",
            labels={"timestamp": "Date", "score": f"{metric.title()} Score", "model": "Model"}
        )
        fig.update_layout(height=400)
        return fig

    return go.Figure()


def display_results_table(df: pd.DataFrame):
    """Display benchmark results in a formatted table."""
    if df.empty:
        st.info("No results to display.")
        return

    # Format the dataframe for display
    display_df = df.copy()

    # Extract relevant columns - use the new flat structure
    columns_to_show = []
    if "id" in display_df.columns:
        columns_to_show.append("id")
    if "model_name" in display_df.columns:
        columns_to_show.append("model_name")
    if "benchmark_suite" in display_df.columns:
        columns_to_show.append("benchmark_suite")
    if "status" in display_df.columns:
        columns_to_show.append("status")
    if "completed_at" in display_df.columns:
        columns_to_show.append("completed_at")
    if "accuracy" in display_df.columns:
        columns_to_show.append("accuracy")
    if "f1" in display_df.columns:
        columns_to_show.append("f1")

    if columns_to_show:
        # Rename columns for better display
        column_names = {
            "id": "Job ID",
            "model_name": "Model",
            "benchmark_suite": "Suite",
            "status": "Status",
            "completed_at": "Completed",
            "accuracy": "Accuracy",
            "f1": "F1 Score"
        }
        display_df_renamed = display_df[columns_to_show].rename(columns=column_names)
        st.dataframe(display_df_renamed, width='stretch')
    else:
        st.dataframe(display_df, width='stretch')


def display_model_comparison(comparison_data: Dict[str, Any], metrics: List[str]):
    """Display model comparison results with visualizations."""
    st.subheader("📊 Model Comparison Results")

    # Summary stats
    col1, col2, col3 = st.columns(3)
    with col1:
        best_model = comparison_data.get("best_performing", "N/A")
        st.metric("Best Performing Model", best_model)
    with col2:
        perf_gap = comparison_data.get("performance_gap", 0)
        st.metric("Performance Gap", f"{perf_gap:.3f}")
    with col3:
        sig_test = comparison_data.get("statistical_significance", "N/A")
        st.metric("Statistical Significance", sig_test)

    # Recommendations
    if comparison_data.get("recommendations"):
        st.info("💡 " + " ".join(comparison_data["recommendations"]))

    # Create comparison visualizations
    if metrics:
        # Radar chart for multi-metric comparison
        st.subheader("Multi-Metric Comparison")
        fig = create_radar_comparison_chart(comparison_data, metrics)
        if fig:
            st.plotly_chart(fig, width='stretch')

        # Bar chart for individual metrics
        st.subheader("Individual Metric Comparison")
        fig2 = create_bar_comparison_chart(comparison_data, metrics)
        if fig2:
            st.plotly_chart(fig2, width='stretch')


def create_radar_comparison_chart(comparison_data: Dict[str, Any], metrics: List[str]) -> Optional[go.Figure]:
    """Create a radar chart for model comparison."""
    # This would need actual metric data from the comparison
    # For now, return None as we don't have the detailed data structure
    return None


def create_bar_comparison_chart(comparison_data: Dict[str, Any], metrics: List[str]) -> Optional[go.Figure]:
    """Create a bar chart for model comparison."""
    # This would need actual metric data from the comparison
    # For now, return None as we don't have the detailed data structure
    return None


def show_benchmark_progress(job_id: str):
    """
    Show benchmark progress.
    Legacy non-synchronous display used as fallback or summary.
    """
    st.subheader("🔄 Benchmark Status")

    try:
        db_conn = get_db_connection()
        cursor = db_conn.cursor()

        # Get job status
        cursor.execute("SELECT status, error_message FROM benchmark_jobs WHERE id = ?", (job_id,))
        row = cursor.fetchone()

        if row:
            status_value = row[0]
            error_msg = row[1]

            if status_value == "completed":
                st.success("✅ Benchmarking completed!")
                # Show results summary if available
                cursor.execute("SELECT metrics FROM benchmark_results WHERE job_id = ?", (job_id,))
                result_row = cursor.fetchone()
                if result_row:
                    results = {"status": "completed", "results": json.loads(result_row[0])}
                    display_benchmark_results(results)
            elif status_value == "failed":
                st.error(f"❌ Benchmarking failed: {error_msg}")
            elif status_value == "cancelled":
                st.warning("⚠️ Benchmarking cancelled.")
            else:
                st.info(f"🔄 **Status: {status_value.title()}**")
                st.caption("Job is being processed in the background.")
        else:
            st.error("Job not found")

    except Exception as e:
        st.error(f"Error displaying benchmark progress: {str(e)}")


def display_benchmark_results(results: Dict[str, Any]):
    """Display completed benchmark results."""
    st.subheader("📊 Benchmark Results")

    # Handle the new results structure
    results_data = results.get("results", {})

    if results_data:
        # Summary metrics from results
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            accuracy = results_data.get("accuracy", 0)
            st.metric("Accuracy", f"{accuracy:.3f}")
        with col2:
            f1 = results_data.get("f1", 0)
            st.metric("F1 Score", f"{f1:.3f}")
        with col3:
            precision = results_data.get("precision", 0)
            st.metric("Precision", f"{precision:.3f}")
        with col4:
            recall = results_data.get("recall", 0)
            st.metric("Recall", f"{recall:.3f}")

        # Performance breakdown
        st.subheader("Performance Metrics")

        # Display metrics in a nice format
        metrics_data = {
            "Accuracy": accuracy,
            "F1 Score": f1,
            "Precision": precision,
            "Recall": recall
        }

        metric_cols = st.columns(len(metrics_data))
        for i, (metric_name, value) in enumerate(metrics_data.items()):
            with metric_cols[i]:
                st.metric(metric_name, f"{value:.3f}")

        # Detailed results
        st.subheader("Detailed Results")
        st.json(results_data)
    else:
        st.info("No detailed results available.")


# Export Functions
def export_benchmark_data_csv(df: pd.DataFrame):
    """Export benchmark data as CSV."""
    try:
        # Prepare data for export
        export_df = df.copy()

        # Flatten nested structures for CSV
        if "model_info" in export_df.columns:
            export_df["model_name"] = export_df["model_info"].apply(
                lambda x: x.get("name", "Unknown") if isinstance(x, dict) else "Unknown"
            )
            export_df["model_provider"] = export_df["model_info"].apply(
                lambda x: x.get("provider", "Unknown") if isinstance(x, dict) else "Unknown"
            )
            export_df.drop("model_info", axis=1, inplace=True)

        if "performance_data" in export_df.columns:
            # Extract key metrics
            export_df["accuracy"] = export_df["performance_data"].apply(
                lambda x: x.get("glue", {}).get("accuracy", {}).get("mean", "N/A") if isinstance(x, dict) else "N/A"
            )
            export_df["f1_score"] = export_df["performance_data"].apply(
                lambda x: x.get("glue", {}).get("f1", {}).get("mean", "N/A") if isinstance(x, dict) else "N/A"
            )
            export_df.drop("performance_data", axis=1, inplace=True)

        # Convert to CSV
        csv_buffer = StringIO()
        export_df.to_csv(csv_buffer, index=False)

        # Create download button using Streamlit's secure method
        st.download_button(
            label="📊 Download CSV",
            data=csv_buffer.getvalue(),
            file_name="benchmark_results.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Failed to export CSV: {str(e)}")


def export_benchmark_data_json(df: pd.DataFrame):
    """Export benchmark data as JSON."""
    try:
        # Convert DataFrame to JSON
        json_data = df.to_json(orient="records", indent=2, date_format="iso")

        # Create download button using Streamlit's secure method
        st.download_button(
            label="📋 Download JSON",
            data=json_data,
            file_name="benchmark_results.json",
            mime="application/json"
        )

    except Exception as e:
        st.error(f"Failed to export JSON: {str(e)}")


def export_benchmark_report_pdf(df: pd.DataFrame, suite: str, metric: str):
    """Export benchmark report as PDF."""
    try:
        # For PDF export, we'll create a simple HTML-based report
        # In a real implementation, you'd use a proper PDF library like reportlab

        html_content = f"""
        <html>
        <head>
            <title>Benchmark Report - {suite.upper()}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #2E86AB; }}
                h2 {{ color: #A23B72; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metric {{ font-weight: bold; }}
            </style>
        </head>
        <body>
            <h1>AI Model Benchmarking Report</h1>
            <p><strong>Suite:</strong> {suite.upper()}</p>
            <p><strong>Primary Metric:</strong> {metric}</p>
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

            <h2>Summary Statistics</h2>
            <ul>
                <li>Total Evaluations: {len(df)}</li>
                <li>Models Evaluated: {df.get('model_info', pd.Series()).apply(lambda x: x.get('name', 'Unknown') if isinstance(x, dict) else 'Unknown').nunique() if 'model_info' in df.columns else 'N/A'}</li>
                <li>Average {metric.title()}: {df.get('performance_data', pd.Series()).apply(lambda x: x.get(suite, {}).get(metric, {}).get('mean', 0) if isinstance(x, dict) else 0).mean():.3f}</li>
            </ul>

            <h2>Detailed Results</h2>
            <table>
                <tr>
                    <th>Model</th>
                    <th>Status</th>
                    <th>{metric.title()}</th>
                    <th>Completed At</th>
                </tr>
        """

        for _, row in df.iterrows():
            model_name = "Unknown"
            if "model_info" in row and isinstance(row["model_info"], dict):
                model_name = row["model_info"].get("name", "Unknown")

            score = "N/A"
            if "performance_data" in row and isinstance(row["performance_data"], dict):
                score = row["performance_data"].get(suite, {}).get(metric, {}).get("mean", "N/A")
                if isinstance(score, (int, float)):
                    score = f"{score:.3f}"

            completed_at = row.get("completed_at", "N/A")
            if pd.notna(completed_at):
                completed_at = pd.to_datetime(completed_at).strftime('%Y-%m-%d %H:%M:%S')

            html_content += f"""
                <tr>
                    <td>{model_name}</td>
                    <td>{row.get('status', 'N/A')}</td>
                    <td>{score}</td>
                    <td>{completed_at}</td>
                </tr>
            """

        html_content += """
            </table>
        </body>
        </html>
        """

        # Create download button using Streamlit's secure method
        st.download_button(
            label="📄 Download HTML Report",
            data=html_content,
            file_name="benchmark_report.html",
            mime="text/html"
        )
        st.info("📄 HTML report generated. For PDF export, additional libraries would be needed.")

    except Exception as e:
        st.error(f"Failed to export report: {str(e)}")
