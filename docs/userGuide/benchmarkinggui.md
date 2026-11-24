# Benchmarking in Compileo GUI

## Overview

The Compileo GUI provides a comprehensive web interface for AI model benchmarking with full integration into the asynchronous job processing system. Users can run performance evaluations on Ollama, Gemini, and Grok models, compare results, track historical performance, and generate detailed reports.

## Accessing Benchmarking

1. **Navigate to the Application**: Open Compileo in your web browser
2. **Select Benchmarking**: Click on "📊 AI Model Benchmarking Dashboard" in the sidebar
3. **Choose Operation**: Select from Overview, Run Benchmarks, Model Comparison, History, or Leaderboard tabs

## Interface Components

### Overview Tab

The overview dashboard provides key metrics and visualizations of recent benchmarking activity.

#### Key Metrics Cards
- **Average Accuracy**: Shows mean accuracy scores across completed benchmarks
- **Total Benchmark Runs**: Count of completed evaluations
- **Models Evaluated**: Number of unique models tested
- **Latest Run**: Timestamp of most recent benchmark completion

#### Filters
- **Benchmark Suite**: Select from GLUE (fully supported) or other suites (framework ready)
- **Primary Metric**: Choose accuracy or f1 score
- **Days Back**: Set time range (1-90 days, default: 30)

#### Performance Trends Chart
Visualizes performance over time with:
- Line chart showing accuracy scores by completion date
- Color-coded lines for different models
- Interactive tooltips with detailed information
- Date range filtering

#### Recent Results Table
Displays latest benchmark runs with:
- Job ID and model name
- Benchmark suite and completion status
- Accuracy and F1 scores
- Completion timestamps

### Run Benchmarks Tab

Execute new benchmarking evaluations for AI models using Compileo's job queue system.

#### AI Provider Selection
- **Provider**: Select from Ollama, Gemini, or Grok
- **Dynamic Model Loading**: Available models fetched automatically based on provider
- **Provider-Specific Settings**: Different configuration options per provider

#### Ollama Configuration (Local Models)
- **Model Selection**: Dropdown populated from available Ollama models
- **Temperature**: Sampling temperature (0.0-2.0)
- **Top P**: Nucleus sampling (0.0-1.0)
- **Top K**: Top-k sampling (0-100)
- **Num Predict**: Maximum tokens to generate
- **Num Context**: Context window size
- **Seed**: Random seed (optional)

#### Gemini/Grok Configuration (API Models)
- **Model Selection**: Available models for the provider
- **Custom Configuration**: JSON configuration for advanced options
- **API Key**: Automatically loaded from settings (environment variables)

#### Benchmark Configuration
- **Benchmark Suite**: Currently supports GLUE (fully implemented)
- **Project ID**: Associated project for the benchmark job

#### Benchmark Execution
1. **Select Provider**: Choose Ollama, Gemini, or Grok
2. **Choose Model**: Select from available models for the provider
3. **Configure Parameters**: Set provider-specific options
4. **Start Benchmarking**: Click "🚀 Start Benchmarking"
5. **Monitor Progress**: Real-time progress tracking with:
   - Job ID display
   - Status updates (pending → running → completed/failed)
   - Auto-refresh options (1, 5, 30, or 60 second intervals)
   - Progress percentage and current step information
   - **Stop Job**: Button to cancel running benchmarks

#### Progress Monitoring
- **Real-time Updates**: Automatic status polling
- **Job Persistence**: Jobs continue running even if browser is closed
- **Job Cancellation**: Stop long-running jobs directly from the interface
- **Result Access**: Direct links to results when complete
- **Error Handling**: Clear error messages and recovery options

### Model Comparison Tab

Compare performance across multiple AI models (framework ready - implementation pending).

#### Model Selection
- **Available Models**: List of models with completed benchmarks
- **Multi-select**: Choose multiple models for comparison
- **Suite Selection**: Filter by benchmark suite

#### Comparison Features (Planned)
- **Performance Gap Analysis**: Difference between best and worst models
- **Statistical Significance**: p-value calculations
- **Visual Comparisons**: Charts and graphs for easy interpretation

### History Tab

Review past benchmarking runs with comprehensive filtering and search.

#### Filters
- **Model Name**: Text search for specific models
- **Suite Filter**: Dropdown for benchmark suite selection
- **Status Filter**: Filter by job status (All, completed, running, failed)

#### History Table
Displays filtered results with:
- Job ID and model name
- Benchmark suite and completion status
- Created and completed timestamps
- Export options (CSV, JSON)

#### Summary Statistics
- **Total Runs**: Count of matching benchmark jobs
- **Completed Runs**: Successful evaluations
- **Success Rate**: Percentage of successful runs

### Leaderboard Tab

Ranked performance comparison across all evaluated models (framework ready - implementation pending).

#### Leaderboard Settings
- **Benchmark Suite**: Select evaluation framework (default: GLUE)
- **Ranking Metric**: Choose performance metric (default: accuracy)
- **Show Top N**: Display configurable number of top models

#### Leaderboard Features (Planned)
- **Ranked List**: Ordered by performance metrics
- **Model Details**: Name, provider, and score information
- **Benchmark Count**: Number of evaluations per model
- **Trend Indicators**: Performance change indicators

## Error Handling

### Common Error Types

- **🔌 Connection Error**: Network issues preventing API communication
- **⏱️ Timeout Error**: Benchmark execution exceeding 3-hour limit
- **🤖 Model API Error**: Issues with AI provider APIs (Gemini, Grok)
- **📊 Invalid Configuration**: Incorrect benchmark parameters
- **🐌 Rate Limit Exceeded**: Too many concurrent benchmarking jobs
- **💾 Database Error**: Issues with result storage or retrieval

### Error Recovery

- **Automatic Retry**: RQ worker handles transient failures
- **Job Persistence**: Failed jobs can be restarted
- **Configuration Validation**: Frontend validates parameters before submission
- **Resource Monitoring**: System prevents resource exhaustion

## Export Capabilities

### Data Export Formats
- **CSV Export**: Tabular data for spreadsheet analysis
- **JSON Export**: Structured data for programmatic processing

### Export Locations
- **Overview Tab**: Export recent results and performance data
- **History Tab**: Export filtered historical benchmark data

## Best Practices

### Benchmark Planning
- **Provider Selection**: Choose Ollama for local/private, Gemini/Grok for API models
- **Model Selection**: Use appropriate model sizes for your hardware capabilities
- **Resource Planning**: GLUE benchmarks take 10-30 minutes per model

### Performance Monitoring
- **Progress Tracking**: Use real-time monitoring for job status
- **Resource Awareness**: Monitor system impact during benchmarking
- **Result Validation**: Verify benchmark results are reasonable

### Data Management
- **Regular Exports**: Save important results for future reference
- **Historical Analysis**: Use History tab for performance trend analysis
- **Job Management**: Cancel unnecessary jobs to free queue resources

## Integration with Other Features

### Dataset Generation Workflow
1. **Run Benchmarks**: Evaluate model performance on GLUE tasks
2. **Analyze Results**: Review accuracy and F1 scores
3. **Model Selection**: Choose best-performing models for dataset generation
4. **Quality Validation**: Use benchmark results to inform quality thresholds

### Job Queue Integration
- **Asynchronous Processing**: All benchmarks run via RQ job queue
- **Resource Management**: Automatic resource allocation and monitoring
- **Progress Tracking**: Real-time status updates across all interfaces
- **Error Recovery**: Automatic retry and failure handling

## Troubleshooting

### Jobs Not Starting
- **Queue Full**: Wait for current jobs to complete (max 3 concurrent)
- **Invalid Parameters**: Check model selection and configuration
- **API Keys**: Verify environment variables for Gemini/Grok
- **Worker Status**: Ensure RQ worker is running

### Progress Not Updating
- **Browser Cache**: Hard refresh to clear cached data
- **Connection Issues**: Check network connectivity to Compileo
- **Job Completion**: Very long-running jobs may show delayed updates

### Results Not Appearing
- **Job Status**: Wait for job to reach "completed" status
- **Error Checking**: Review job details for failure reasons
- **Database Issues**: Check system logs for storage problems

### Provider-Specific Issues

#### Ollama Problems
- **Model Not Available**: Check `ollama list` for installed models
- **Connection Failed**: Verify Ollama server is running on port 11434
- **Parameter Errors**: Validate temperature/top_p ranges

#### Gemini/Grok Problems
- **API Key Missing**: Check environment variables `GOOGLE_API_KEY`/`GROK_API_KEY`
- **Rate Limits**: Wait before retrying (provider-specific limits)
- **Model Access**: Verify API key has access to requested models

## Advanced Features

### Provider-Specific Configuration

#### Ollama Advanced Settings
- **Temperature Control**: Fine-tune randomness (0.0-2.0)
- **Context Window**: Adjust token limits for large models
- **Sampling Parameters**: Top-k, top-p, and repetition penalty
- **Seed Control**: Reproducible results with fixed seeds

#### API Provider Options
- **Custom Configuration**: JSON settings for advanced parameters
- **Model Selection**: Latest available models from providers
- **Error Handling**: Automatic retry with exponential backoff

### Job Management
- **Queue Monitoring**: Real-time queue status and worker health
- **Job Cancellation**: Stop running jobs via GUI button or CLI
- **Result Persistence**: All results stored in SQLite database
- **Historical Tracking**: Complete audit trail of all benchmarks

### System Integration
- **Environment Variables**: Secure API key management
- **Database Transactions**: Atomic operations for data consistency
- **Logging**: Comprehensive debug logging for troubleshooting
- **Health Checks**: System monitoring and automatic recovery

This benchmarking interface provides production-ready AI model evaluation capabilities with seamless integration into Compileo's asynchronous processing system.