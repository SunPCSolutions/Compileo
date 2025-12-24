# Settings GUI User Guide

## Overview

The Settings page provides comprehensive configuration options for Compileo's AI models, system security, and processing parameters. This includes API key management for the Compileo system itself, as well as multiple AI providers (Ollama, Gemini, Grok, OpenAI). This guide covers security configuration and Ollama parameter fine-tuning.

## 🔒 API Authentication & Security

Compileo implements an **"Auto-Lock"** security model. This allows for a simplified setup while ensuring production-grade security once configured.

### Configuring System API Keys

To secure your Compileo instance:
1.  Navigate to the **Settings** page in the Compileo GUI.
2.  Locate the **🔗 API Configuration** section.
3.  Enter one or more strong, secret values in the **API Keys** field (supports comma-separated lists).
4.  Click **Save Settings**.

**Security Behavior:**
*   **Unsecured Mode**: If the API Keys field is empty (and no environment variables like `COMPILEO_API_KEYS` are set), Compileo allows all requests.
*   **Secured Mode**: Once a key is saved, the API immediately "locks" and requires an `X-API-Key` header for all further access from both the GUI and external tools.

### Provider API Keys
The settings page also allows you to manage keys for external LLM providers (Gemini, Grok, OpenAI). These are stored in the database and used by Compileo's backend to perform AI tasks.

## Ollama Parameter Configuration

Compileo supports advanced configuration of Ollama AI models through role-specific parameters. Each AI processing role (parsing, taxonomy, classification, generation) can have customized Ollama parameters.

### Accessing Parameter Settings

1. Navigate to the Settings page in the Compileo GUI
2. Scroll to the AI Model Configuration section
3. Select an Ollama model for any processing role (parsing, taxonomy, classification, generation)
4. Parameter input fields will appear below the model dropdown

### Parameter Layout

Parameters are arranged in a compact 4-column by 2-row grid for each Ollama role:

**Row 1:** temperature | repeat_penalty | top_p | top_k
**Row 2:** num_predict | seed | num_ctx | (empty)

### Available Parameters

#### Temperature (0.0-2.0)
Controls randomness in AI responses:
- **Lower values (0.0-0.5)**: More focused, deterministic responses
- **Higher values (1.0-2.0)**: More creative, varied responses
- **Default**: 0.1 (parsing), 0.8 (other roles)

#### Repeat Penalty (0.0-2.0)
Reduces repetition in generated text:
- **Lower values (< 1.0)**: Allow more repetition
- **Higher values (> 1.0)**: Strongly penalize repetition
- **Default**: 1.1

#### Top P (0.0-1.0)
Nucleus sampling parameter:
- **Lower values (0.1-0.5)**: More focused responses
- **Higher values (0.9-1.0)**: More diverse responses
- **Default**: 0.9

#### Top K (0-100)
Limits candidate tokens for sampling:
- **Lower values (1-20)**: More focused responses
- **Higher values (50-100)**: More diverse responses
- **Default**: 40

#### Num Predict (1-4096)
Maximum number of tokens to generate:
- **Lower values (100-500)**: Shorter responses
- **Higher values (1000-4000)**: Longer responses
- **Default**: 1024 (parsing), 8192 (parsing with higher limit)

#### Seed (0-4294967295)
Random seed for reproducible results:
- **Same seed**: Identical responses for same input
- **Different/random seed**: Varied responses
- **Default**: None (random)

#### Num Ctx (1-131072)
Context window size for Ollama models:
- **Lower values (4096-8192)**: Faster processing, less memory usage
- **Higher values (32768-131072)**: Better context understanding, more memory usage
- **Default**: 60000 (balanced performance and capability)

### Role-Specific Recommendations

#### Document Parsing
- **Temperature**: 0.1 (deterministic OCR output)
- **Num Predict**: 8192 (handle long documents)
- **Num Ctx**: 32768 (sufficient context for document understanding)
- **Other parameters**: Conservative defaults for accuracy

#### Taxonomy Generation
- **Temperature**: 0.8 (balanced creativity)
- **Num Predict**: 2048 (comprehensive taxonomies)
- **Num Ctx**: 65536 (large context for analyzing multiple chunks)
- **Top P**: 0.95 (diverse category suggestions)

#### Classification
- **Temperature**: 0.3 (consistent categorization)
- **Num Predict**: 512 (concise classifications)
- **Num Ctx**: 16384 (focused context for individual classifications)
- **Repeat Penalty**: 1.2 (avoid repetitive classifications)

#### Dataset Generation
- **Temperature**: 0.8 (creative variations)
- **Num Predict**: 1024 (balanced response length)
- **Num Ctx**: 32768 (adequate context for chunk-based generation)
- **Top P**: 0.9 (diverse question/answer pairs)

### Saving Configuration

1. Adjust parameter values as needed
2. Click **Save Settings** to persist changes
3. Parameters are immediately applied to subsequent processing jobs

### Parameter Validation

The interface includes built-in validation:
- **Range checking**: Values must be within specified ranges
- **Type validation**: Numeric inputs only
- **Real-time feedback**: Invalid values show error indicators

### Troubleshooting

#### Parameters Not Applied
- Ensure settings are saved after changes
- Restart any running jobs to pick up new parameters
- Check that the correct Ollama model is selected

#### Unexpected AI Behavior
- Try conservative parameter values first (temperature 0.1-0.3)
- Adjust one parameter at a time to isolate effects
- Reset to defaults if issues persist

#### Performance Issues
- Lower num_predict for faster processing
- Reduce temperature for more consistent outputs
- Monitor Ollama resource usage

## Global Settings

### Log Level Configuration
Compileo provides a project-wide logging system that can be controlled through the GUI, API, or CLI. This allows you to adjust the verbosity of logs based on your needs.

#### Available Log Levels
- **none**: Disables all logging output. Useful for production environments where minimal noise is desired.
- **error**: Only logs critical errors and exceptions. Recommended for standard usage.
- **debug**: Enables extensive log reporting, including internal process details and JSON-structured debug information. Intended for developers and troubleshooting.

#### Configuring Log Level in GUI
1. Navigate to the **General** tab in the Settings page.
2. Locate the **Log Level** dropdown.
3. Select your desired level (**none**, **error**, or **debug**).
4. Click **Save Settings**. The new log level is applied immediately to all system components, including background workers.

## Advanced Configuration

### Custom Model Selection

Parameters are configured per Ollama model. Different models can have different parameter sets:

1. Select different Ollama models for different roles
2. Configure parameters independently for each model
3. Compare results across different model+parameter combinations

### Batch Processing

Configured parameters apply to all batch processing operations:
- Document parsing jobs
- Taxonomy generation tasks
- Dataset creation workflows
- Classification operations

### Integration with CLI/API

Parameters configured in the GUI are automatically used by:
- CLI commands using Ollama models
- API endpoints processing with Ollama
- Background job execution

## Best Practices

1. **Start Conservative**: Begin with default or low parameter values
2. **Test Incrementally**: Change one parameter at a time
3. **Document Settings**: Note parameter combinations that work well
4. **Role-Specific Tuning**: Use different parameters for different AI tasks
5. **Monitor Performance**: Adjust based on output quality and processing speed

## Support

For issues with parameter configuration:
- Check Ollama server logs for API errors
- Verify parameter ranges are valid
- Ensure settings are properly saved

## Plugin Management

The Settings page includes a dedicated **Plugins** tab for extending Compileo's functionality.

### Accessing Plugins
1. Click the **Plugins** tab at the top of the Settings page.
2. View the list of installed plugins or upload new ones.

### Features
- **Upload Plugin**: Install new plugins by uploading `.zip` files.
- **List Plugins**: View details of installed plugins including version and author.
- **Uninstall**: Remove plugins that are no longer needed.

For detailed instructions, refer to the [Plugin Management User Guide](plugins.md).
- Test with default parameters first