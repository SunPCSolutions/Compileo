# Prompts API in Compileo

## Overview

The Compileo Prompts API provides comprehensive management capabilities for AI prompts used throughout the system. It enables creation, retrieval, updating, and deletion of prompts that guide AI model behavior for various tasks.

## Base URL: `/api/v1/prompts`

---

## 1. List Prompts

**Endpoint:** `GET /`

**Description:** Retrieves a list of all available prompts with optional filtering.

**Query Parameters:**
- `name_filter` (string, optional): Filter prompts by name substring
- `limit` (integer, optional, default: 50): Maximum number of prompts to return

**Success Response (200 OK):**
```json
{
  "prompts": [
    {
      "id": 1,
      "name": "medical_qa_generation",
      "content": "Generate medical Q&A pairs based on the following context...",
      "created_at": "2024-01-15T10:30:00Z",
      "updated_at": "2024-01-20T14:45:00Z"
    },
    {
      "id": 2,
      "name": "taxonomy_extraction",
      "content": "Extract taxonomy categories from the provided text...",
      "created_at": "2024-01-16T09:15:00Z",
      "updated_at": null
    }
  ],
  "total": 2
}
```

---

## 2. Create Prompt

**Endpoint:** `POST /`

**Description:** Creates a new prompt in the system.

**Request Body:**
```json
{
  "name": "custom_extraction_prompt",
  "content": "You are an expert at extracting structured information from medical texts. Given the following text, identify and categorize all medical conditions, treatments, and symptoms mentioned..."
}
```

**Parameters:**
- `name` (string, required): Unique identifier for the prompt
- `content` (string, required): Full prompt text content

**Success Response (201 Created):**
```json
{
  "id": 3,
  "name": "custom_extraction_prompt",
  "content": "You are an expert at extracting structured information from medical texts...",
  "created_at": "2024-01-21T11:00:00Z"
}
```

**Error Response (400 Bad Request):**
```json
{
  "detail": "Prompt with name 'custom_extraction_prompt' already exists"
}
```

---

## 3. Get Prompt by ID

**Endpoint:** `GET /{prompt_id}`

**Description:** Retrieves a specific prompt by its unique identifier.

**Path Parameters:**
- `prompt_id` (integer, required): Prompt identifier

**Success Response (200 OK):**
```json
{
  "id": 3,
  "name": "custom_extraction_prompt",
  "content": "You are an expert at extracting structured information from medical texts...",
  "created_at": "2024-01-21T11:00:00Z",
  "updated_at": "2024-01-21T15:30:00Z"
}
```

**Error Response (404 Not Found):**
```json
{
  "detail": "Prompt with ID 999 not found"
}
```

---

## 4. Get Prompt by Name

**Endpoint:** `GET /by-name/{name}`

**Description:** Retrieves a specific prompt by its name identifier.

**Path Parameters:**
- `name` (string, required): Prompt name identifier

**Success Response (200 OK):** Same as Get Prompt by ID

**Error Response (404 Not Found):**
```json
{
  "detail": "Prompt 'nonexistent_prompt' not found"
}
```

---

## 5. Update Prompt

**Endpoint:** `PUT /{prompt_id}`

**Description:** Updates an existing prompt's name and/or content.

**Path Parameters:**
- `prompt_id` (integer, required): Prompt identifier

**Request Body:**
```json
{
  "name": "updated_extraction_prompt",
  "content": "Updated prompt content with additional instructions..."
}
```

**Parameters:**
- `name` (string, optional): New name for the prompt
- `content` (string, optional): Updated prompt content

**Success Response (200 OK):**
```json
{
  "id": 3,
  "name": "updated_extraction_prompt",
  "content": "Updated prompt content with additional instructions...",
  "created_at": "2024-01-21T11:00:00Z",
  "updated_at": "2024-01-21T16:45:00Z"
}
```

---

## 6. Delete Prompt

**Endpoint:** `DELETE /{prompt_id}`

**Description:** Permanently removes a prompt from the system.

**Path Parameters:**
- `prompt_id` (integer, required): Prompt identifier

**Success Response (200 OK):**
```json
{
  "message": "Prompt 3 deleted successfully"
}
```

---

## Prompt Types and Usage

### Dataset Generation Prompts
Used for creating training data from processed documents:
- Question-answer pair generation
- Multiple choice question creation
- Fill-in-the-blank exercises
- Explanation generation

### Taxonomy Extraction Prompts
Guide AI models in categorizing content:
- Category identification and classification
- Hierarchical taxonomy construction
- Content tagging and labeling
- Semantic categorization

### Quality Assessment Prompts
Support quality analysis workflows:
- Content evaluation criteria
- Bias detection guidelines
- Difficulty assessment frameworks
- Consistency validation rules

### Custom Analysis Prompts
Domain-specific or specialized prompts:
- Medical content analysis
- Legal document processing
- Technical documentation parsing
- Research paper summarization

---

## Best Practices

### Prompt Design
- **Clear Instructions**: Provide explicit, unambiguous guidance
- **Context Setting**: Include relevant background and constraints
- **Output Formatting**: Specify desired response structure
- **Error Handling**: Include guidance for edge cases

### Naming Conventions
- **Descriptive Names**: Use clear, descriptive identifiers
- **Consistent Prefixes**: Group related prompts by functionality
- **Version Indicators**: Include version numbers for iterations
- **Domain Specificity**: Reflect domain or use case in naming

### Content Management
- **Version Control**: Track prompt evolution and improvements
- **Testing**: Validate prompts across different content types
- **Performance Monitoring**: Track prompt effectiveness over time
- **Regular Updates**: Refine prompts based on results and feedback

### Security Considerations
- **Input Validation**: Sanitize prompt content and metadata
- **Access Control**: Implement appropriate permission levels
- **Audit Logging**: Track prompt creation, modification, and usage
- **Content Review**: Review prompts for sensitive or inappropriate content

---

## Integration Examples

### Using Prompts in Dataset Generation
```python
# Retrieve prompt for dataset generation
prompt = get_prompt_by_name("medical_qa_generation")

# Use in AI model interaction
response = ai_model.generate(
    prompt=prompt.content,
    context=document_content,
    parameters={"max_questions": 10}
)
```

### Dynamic Prompt Selection
```python
# Select appropriate prompt based on content type
if content_type == "medical":
    prompt = get_prompt_by_name("medical_extraction")
elif content_type == "legal":
    prompt = get_prompt_by_name("legal_analysis")
else:
    prompt = get_prompt_by_name("general_extraction")
```

### Prompt Versioning
```python
# Use versioned prompts for consistency
prompt_v2 = get_prompt_by_name("extraction_prompt_v2.1")
results = process_content(content, prompt_v2.content)
```

---

## Error Handling

### Common Error Responses

**400 Bad Request:**
```json
{
  "detail": "Prompt with name 'duplicate_name' already exists"
}
```

**404 Not Found:**
```json
{
  "detail": "Prompt with ID 999 not found"
}
```

**500 Internal Server Error:**
```json
{
  "detail": "Database connection failed during prompt creation"
}
```

---

## Rate Limiting

- **Read Operations**: 100 requests per minute
- **Write Operations**: 30 requests per minute
- **Bulk Operations**: 10 requests per minute

This prompts API provides a centralized system for managing AI model guidance, ensuring consistent and effective prompt usage across all Compileo operations.