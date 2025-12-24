# Dataset Quality API in Compileo

## Overview

The Compileo Dataset Quality API provides comprehensive quality assessment capabilities for AI training datasets. It evaluates datasets across multiple quality dimensions including diversity, bias detection, difficulty assessment, and consistency checking.

## Base URL: `/api/v1/quality`

---

## 1. Analyze Dataset Quality

**Endpoint:** `POST /analyze`

**Description:** Performs comprehensive quality analysis on a dataset using configured metrics.

**Request Body:**
```json
{
  "dataset": [
    {
      "question": "What is the capital of France?",
      "answer": "Paris",
      "metadata": {
        "difficulty": "easy",
        "topic": "geography"
      }
    }
  ],
  "config": {
    "enabled_metrics": ["diversity", "bias", "difficulty", "consistency"],
    "thresholds": {
      "diversity": 0.7,
      "bias": 0.8,
      "difficulty": 0.6,
      "consistency": 0.9
    }
  },
  "quality_model": "gemini"
}
```

**Parameters:**
- `dataset` (array, required): Array of dataset items
- `config` (object, optional): Quality analysis configuration
  - `enabled_metrics` (array): Metrics to run
  - `thresholds` (object): Custom thresholds per metric
- `quality_model` (string, optional): AI model for quality analysis (`gemini`, `grok`, `ollama`, `openai`) (default: `gemini`)

**Success Response (200 OK):**
```json
{
  "enabled": true,
  "dataset_size": 100,
  "metrics_run": ["diversity", "bias", "difficulty", "consistency"],
  "results": {
    "diversity": {
      "name": "diversity",
      "score": 0.85,
      "threshold": 0.7,
      "passed": true,
      "details": {
        "lexical_diversity": 0.78,
        "semantic_diversity": 0.92,
        "topic_coverage": 0.85
      }
    }
  },
  "summary": {
    "overall_score": 0.82,
    "passed": true,
    "passed_metrics": 4,
    "failed_metrics": 0,
    "total_metrics": 4,
    "issues": []
  }
}
```

---

## 2. Get Quality Metrics

**Endpoint:** `GET /metrics`

**Description:** Retrieves information about available quality metrics and their configurations.

**Success Response (200 OK):**
```json
{
  "available_metrics": [
    {
      "name": "diversity",
      "description": "Evaluates lexical and semantic diversity",
      "default_threshold": 0.7,
      "enabled": true
    },
    {
      "name": "bias",
      "description": "Detects demographic and content bias",
      "default_threshold": 0.8,
      "enabled": true
    },
    {
      "name": "difficulty",
      "description": "Assesses question/answer complexity",
      "default_threshold": 0.6,
      "enabled": true
    },
    {
      "name": "consistency",
      "description": "Checks factual and logical consistency",
      "default_threshold": 0.9,
      "enabled": true
    }
  ],
  "default_config": {
    "enabled": true,
    "fail_on_any_failure": false,
    "output_format": "json"
  }
}
```

---

## 3. Validate Dataset

**Endpoint:** `POST /validate`

**Description:** Quick validation check to ensure dataset format and basic quality requirements.

**Request Body:**
```json
{
  "dataset": [
    {
      "question": "Sample question?",
      "answer": "Sample answer",
      "metadata": {}
    }
  ],
  "strict_mode": false
}
```

**Success Response (200 OK):**
```json
{
  "valid": true,
  "issues": [],
  "warnings": [
    "Dataset size is small (1 items). Consider larger datasets for reliable analysis."
  ],
  "recommendations": [
    "Add more diverse examples",
    "Include metadata for better analysis"
  ]
}
```

---

## Quality Metrics Details

### Diversity Metric
Evaluates content variety and coverage:
- **Lexical Diversity**: Vocabulary richness and variety
- **Semantic Diversity**: Meaning and concept coverage
- **Topic Coverage**: Subject matter distribution

### Bias Metric
Detects potential biases in content:
- **Demographic Bias**: Gender, ethnicity, age representation
- **Content Bias**: Topic or perspective imbalance
- **Language Bias**: Formal/informal tone distribution

### Difficulty Metric
Assesses complexity levels:
- **Reading Level**: Text complexity analysis
- **Cognitive Load**: Reasoning requirements
- **Domain Knowledge**: Required expertise level

### Consistency Metric
Validates logical and factual coherence:
- **Factual Consistency**: Accuracy verification
- **Logical Consistency**: Reasoning validation
- **Format Consistency**: Structure uniformity

---

## Configuration Options

### Metric Thresholds
```json
{
  "diversity": {
    "threshold": 0.7,
    "min_lexical_diversity": 0.6,
    "min_semantic_diversity": 0.7
  },
  "bias": {
    "threshold": 0.8,
    "demographic_keywords": ["gender", "ethnicity", "age"]
  },
  "difficulty": {
    "threshold": 0.6,
    "target_difficulty": "intermediate"
  },
  "consistency": {
    "threshold": 0.9,
    "check_factual_consistency": true
  }
}
```

### Analysis Settings
- **fail_on_any_failure**: Stop on first failed metric
- **output_format**: json, text, or markdown
- **detailed_reporting**: Include per-item analysis

---

## Error Handling

### Common Error Responses

**400 Bad Request:**
```json
{
  "detail": "Invalid dataset format. Expected array of objects with question/answer fields."
}
```

**422 Unprocessable Entity:**
```json
{
  "detail": "Dataset too small for reliable analysis. Minimum 10 items required."
}
```

**500 Internal Server Error:**
```json
{
  "detail": "Quality analysis failed due to metric execution error"
}
```

---

## Best Practices

### Dataset Preparation
- Ensure consistent question/answer format
- Include relevant metadata for better analysis
- Use diverse, representative samples
- Validate data quality before analysis

### Metric Selection
- Enable all metrics for comprehensive analysis
- Adjust thresholds based on use case requirements
- Consider domain-specific quality requirements

### Result Interpretation
- Review individual metric scores and details
- Address failed metrics before dataset use
- Use summary scores for quick quality assessment
- Consider metric weights for custom scoring

### Performance Optimization
- Analyze large datasets in batches
- Cache results for repeated analysis
- Use appropriate metric subsets for quick checks