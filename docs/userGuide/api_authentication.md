# API Authentication Guide

Compileo implements mandatory API key authentication for all protected endpoints to ensure secure access to document processing and dataset engineering features.

## Authentication Mechanism

Compileo uses a custom header-based authentication mechanism. All requests to protected endpoints must include a valid API key in the request headers.

**Header Name:** `X-API-Key`

### Public Endpoints (No Auth Required)
The following endpoints are accessible without authentication:
*   `/` (Root API information)
*   `/health` (Service health check)
*   `/docs` (Swagger UI documentation)
*   `/redoc` (ReDoc documentation)
*   `/openapi.json` (OpenAPI specification)

---

## Setting Up API Keys

API keys can be defined via the GUI, CLI arguments, or environment variables.

### 🔐 Auto-Lock Model
Compileo uses an "Auto-Lock" security model for maximum user-friendliness:
*   **Unsecured Mode**: If no API keys are defined anywhere, the API allows all requests. This is ideal for initial setup.
*   **Secured Mode**: As soon as at least one API key is defined (via GUI, CLI, or Env), the API "locks" and requires a valid key for all protected endpoints.

### 1. Via GUI (Recommended)
1.  Open the Compileo Web GUI.
2.  Go to **Settings** > **🔗 API Configuration**.
3.  Enter your desired key in the **API Key** field and save.
4.  The API will immediately lock down using this key.

### 2. Via CLI
You can pass a static API key when starting the backend server:
```bash
uvicorn src.compileo.api.main:app --host 0.0.0.0 --port 8000
```
*Note: Set API keys via GUI Settings after startup.*

### 3. Via Environment Variables
Set these in your shell, `.env` file, or Docker configuration:
*   `COMPILEO_API_KEYS`: A comma-separated list of authorized keys.

**Example (.env):**
```env
COMPILEO_API_KEYS=key1,key2,key3
```

---

## Configuration Priorities

The API consolidates keys from all sources. If multiple sources define keys, they are all added to the list of authorized keys.

If **no API keys** are defined anywhere, the API operates in **Unsecured Mode**.

---

## Usage Examples

### Using `curl`
```bash
curl -X GET "http://localhost:8000/api/v1/projects" \
  -H "X-API-Key: your_secret_key"
```

### Using Python `requests`
```python
import requests

API_URL = "http://localhost:8000/api/v1"
API_KEY = "your_secret_key"

headers = {
    "X-API-Key": API_KEY
}

response = requests.get(f"{API_URL}/projects", headers=headers)

if response.status_code == 200:
    print("Successfully authenticated!")
    print(response.json())
elif response.status_code == 401:
    print("Authentication failed: Invalid or missing API key")
```

---

## Security Best Practices

1.  **Use Strong Keys**: Generate random, high-entropy strings for your API keys.
2.  **Environment Isolation**: Use different keys for development, staging, and production environments.
3.  **Secure Storage**: Never commit your `.env` file containing real API keys to version control. Add `.env` to your `.gitignore`.
4.  **HTTPS**: In production environments, always serve the API over HTTPS to protect the API key in transit.
