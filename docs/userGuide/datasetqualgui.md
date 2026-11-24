# Dataset Quality Analysis in Compileo GUI

## Overview

The Compileo GUI provides comprehensive dataset quality analysis capabilities through an intuitive web interface. The quality analysis evaluates datasets across multiple dimensions including diversity, bias detection, difficulty assessment, and consistency validation.

## Accessing Quality Analysis

1. **Navigate to the Application**: Open Compileo in your web browser
2. **Select Quality Analysis**: Click on "📊 Quality Metrics" in the sidebar or main navigation
3. **Choose Analysis Type**: Select from Analysis, Metrics Dashboard, or History tabs

## Interface Components

### Analysis Tab

The main analysis interface for running quality assessments on datasets.

#### Dataset Selection
- **Available Datasets**: Dropdown showing all JSONL dataset files in the test_outputs directory
- **Dataset Information**: Displays dataset name and number of entries
- **File Format**: Supports JSONL format with question/answer structure

#### Analysis Configuration

**Basic Settings:**
- **Quality Metrics Selection**:
  - Diversity Analysis: Evaluates lexical and semantic variety
  - Bias Detection: Identifies potential demographic and content biases
  - Difficulty Assessment: Measures question complexity and readability
- **Overall Quality Threshold**: Minimum passing score (0.0-1.0, default 0.7)
- **Output Format**: JSON, HTML, or PDF report formats

**Advanced Settings:**
- **Diversity Threshold**: Minimum diversity score required
- **Bias Threshold**: Maximum acceptable bias score
- **Target Difficulty**: Desired difficulty level for the dataset

#### Analysis Execution
1. **Configure Settings**: Select metrics and adjust thresholds
2. **Start Analysis**: Click "🚀 Start Quality Analysis"
3. **Monitor Progress**: View real-time status updates
4. **Review Results**: Access completed analysis in dashboard

### Metrics Dashboard Tab

Interactive visualization and detailed breakdown of quality analysis results.

#### Overall Summary Metrics
- **Overall Quality Score**: Weighted average across all enabled metrics
- **Dataset Size**: Total number of entries analyzed
- **Passed/Failed Metrics**: Count of metrics meeting thresholds
- **Failed Metrics Count**: Number of metrics below threshold

#### Individual Metrics Visualization
- **Metrics Bar Chart**: Color-coded bars showing scores for each metric
  - Green bars: Passed metrics
  - Red bars: Failed metrics
  - Orange threshold line: Default quality threshold
- **Detailed Results Table**: Tabular view with scores, thresholds, and pass/fail status

#### Metric-Specific Breakdowns

**Diversity Analysis:**
- Lexical Diversity: Vocabulary richness and variety
- Semantic Diversity: Concept and meaning coverage
- Topic Balance: Subject matter distribution
- **Radar Chart**: Visual representation of diversity profile

**Bias Detection:**
- Overall Bias Score: Composite bias measurement
- Bias Indicators: Breakdown by demographic categories
- Content Balance: Topic and perspective distribution

**Difficulty Assessment:**
- Average Difficulty: Mean complexity score
- Complexity Score: Cognitive load assessment
- Readability Score: Text comprehension metrics
- **Distribution Chart**: Pie chart showing difficulty level distribution

### History Tab

Review past quality analysis runs and track performance over time.

#### Analysis History Table
- **Job ID**: Unique identifier for each analysis run
- **Status**: Completion status (completed, failed, running)
- **Summary Scores**: Overall quality metrics
- **Completion Timestamp**: When analysis finished

#### Summary Statistics
- **Total Analyses**: Number of quality assessments run
- **Completed Analyses**: Successfully finished evaluations
- **Failed Analyses**: Analyses that encountered errors

## Quality Metrics Explained

### Diversity Metrics
Evaluates content variety and representation:
- **Lexical Diversity**: Measures vocabulary richness and word variety
- **Semantic Diversity**: Assesses concept coverage and meaning variety
- **Topic Coverage**: Analyzes subject matter distribution and balance

### Bias Detection
Identifies potential biases in dataset content:
- **Demographic Bias**: Gender, ethnicity, age, and cultural representation
- **Content Bias**: Topic selection and perspective balance
- **Language Bias**: Formal/informal tone and register distribution

### Difficulty Assessment
Measures question and content complexity:
- **Reading Level**: Text complexity using readability formulas
- **Cognitive Load**: Reasoning and comprehension requirements
- **Domain Expertise**: Required knowledge level for correct answers

### Consistency Validation
Ensures logical and factual coherence:
- **Factual Consistency**: Accuracy of information and claims
- **Logical Consistency**: Reasoning validity and coherence
- **Format Consistency**: Structural uniformity across entries

## Analysis Workflow

### Preparing for Analysis
1. **Generate Dataset**: Create or upload dataset in JSONL format
2. **Review Content**: Ensure proper question/answer structure
3. **Check Size**: Verify sufficient entries for reliable analysis (minimum 10-50 recommended)

### Running Quality Analysis
1. **Select Dataset**: Choose from available JSONL files
2. **Configure Metrics**: Enable desired quality checks
3. **Set Thresholds**: Adjust passing criteria as needed
4. **Execute Analysis**: Start background processing
5. **Monitor Progress**: Track real-time status updates

### Interpreting Results
1. **Review Overall Score**: Check if dataset meets quality requirements
2. **Analyze Failed Metrics**: Identify specific quality issues
3. **Examine Details**: Review metric-specific breakdowns
4. **Address Issues**: Modify dataset based on recommendations

## Error Handling

### Common Issues

- **Dataset Not Found**: Ensure JSONL files exist in test_outputs directory
- **Invalid Format**: Verify question/answer structure in dataset
- **Analysis Timeout**: Large datasets may take time to process
- **Metric Failures**: Individual metrics may fail while others succeed

### Recovery Actions
- **Restart Analysis**: Failed analyses can be restarted
- **Modify Configuration**: Adjust thresholds or disable problematic metrics
- **Check Dataset**: Validate data format and content
- **Contact Support**: For persistent technical issues

## Best Practices

### Dataset Preparation
- Use consistent question/answer formats
- Include diverse, representative content
- Add metadata for enhanced analysis
- Validate data quality before analysis

### Analysis Configuration
- Enable all relevant metrics for comprehensive evaluation
- Set appropriate thresholds based on use case
- Consider domain-specific quality requirements
- Use advanced settings for fine-tuned analysis

### Result Utilization
- Address failed metrics before dataset deployment
- Use detailed breakdowns for targeted improvements
- Track quality trends across dataset versions
- Compare quality scores between datasets

### Performance Optimization
- Analyze datasets during off-peak hours for large files
- Use appropriate metric subsets for quick assessments
- Cache results for repeated analysis of same datasets
- Monitor system resources during analysis

## Integration with Other Features

### Dataset Generation Workflow
1. **Generate Dataset**: Create training data using dataset generation tools
2. **Run Quality Analysis**: Evaluate generated content quality
3. **Review Results**: Identify areas for improvement
4. **Iterate Generation**: Refine prompts and parameters based on analysis
5. **Validate Improvements**: Re-run analysis to confirm quality gains

### Benchmarking Integration
1. **Complete Quality Analysis**: Ensure dataset meets quality standards
2. **Run Benchmarking**: Evaluate model performance on quality-assessed data
3. **Correlate Results**: Compare quality metrics with benchmark performance
4. **Optimize Dataset**: Use insights to improve both quality and performance

## Troubleshooting

### Analysis Not Starting
- Verify dataset file exists and is readable
- Check JSONL format and content structure
- Ensure sufficient system resources available

### Unexpected Results
- Review dataset content for consistency issues
- Check metric thresholds are appropriate for content type
- Validate analysis configuration settings

### Performance Issues
- Reduce dataset size for faster analysis
- Disable unnecessary metrics for quick checks
- Run analysis during low-usage periods

This quality analysis interface provides comprehensive evaluation capabilities to ensure datasets meet the highest standards for AI training and evaluation.