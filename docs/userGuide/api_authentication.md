# API Authentication Guide

Compileo implements a complete multi-user authentication system using JWT (JSON Web Tokens) for secure access to all protected endpoints. Each user has strict data isolation, ensuring they can only access their own projects, documents, and data.

## Authentication Mechanism

Compileo uses JWT-based authentication with Bearer token authorization. All requests to protected endpoints must include a valid JWT token in the Authorization header.

**Header Name:** `Authorization`
**Header Value:** `Bearer <jwt_token>`

### Public Endpoints (No Auth Required)
The following endpoints are accessible without authentication:
*   `/` (Root API information)
*   `/health` (Service health check)
*   `/docs` (Swagger UI documentation)
*   `/redoc` (ReDoc documentation)
*   `/openapi.json` (OpenAPI specification)
*   `/auth/register` (User registration)
*   `/auth/login` (User login)

---

## User Registration

Before accessing protected endpoints, you must create a user account.

### Via API
```bash
curl -X POST "http://localhost:8000/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "username": "myusername",
    "password": "secure_password_123"
  }'
```

**Response:**
```json
{
  "id": "user_123",
  "email": "user@example.com",
  "username": "myusername",
  "is_active": true,
  "created_at": "2024-01-15T10:30:00Z"
}
```

### Via GUI
1. Open the Compileo Web GUI at `http://localhost:8501`
2. Click "Register" and fill in your details
3. Your account will be created automatically

---

## User Login

After registration, authenticate to receive a JWT token.

### Via API
```bash
curl -X POST "http://localhost:8000/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=user@example.com&password=secure_password_123"
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 86400,
  "user": {
    "id": "user_123",
    "email": "user@example.com",
    "username": "myusername"
  }
}
```

### Via GUI
1. Open the Compileo Web GUI at `http://localhost:8501`
2. Enter your email/username and password
3. You'll be automatically logged in and redirected to the main interface

---

## Using JWT Tokens

All protected API requests must include the JWT token in the Authorization header.

### Using `curl`
```bash
curl -X GET "http://localhost:8000/api/v1/projects" \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
```

### Using Python `requests`
```python
import requests

API_URL = "http://localhost:8000/api/v1"
JWT_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."

headers = {
    "Authorization": f"Bearer {JWT_TOKEN}"
}

response = requests.get(f"{API_URL}/projects", headers=headers)

if response.status_code == 200:
    print("Successfully authenticated!")
    print(response.json())
elif response.status_code == 401:
    print("Authentication failed: Invalid or expired token")
```

### Token Expiration
- JWT tokens expire after 24 hours
- When a token expires, you'll receive a 401 Unauthorized response
- Simply login again to get a fresh token

---

## Data Isolation

Compileo's multi-user architecture ensures complete data isolation:

*   **User-Scoped Data**: Each user can only access projects, documents, and data they own
*   **File Isolation**: User files are stored in separate directories (`storage/{user_id}/...`)
*   **Database Isolation**: All database queries include `owner_id` filtering
*   **Job Isolation**: Background jobs run with the submitting user's context

---

## Security Best Practices

1.  **Strong Passwords**: Use complex passwords with mixed characters, numbers, and symbols
2.  **Token Security**: Never expose JWT tokens in logs, URLs, or client-side code
3.  **HTTPS**: Always use HTTPS in production to protect tokens in transit
4.  **Token Rotation**: Implement automatic token refresh for long-running applications
5.  **Secure Storage**: Store tokens securely (environment variables, secure key stores)
6.  **Session Management**: Implement proper logout to invalidate tokens server-side

---

## Error Responses

### Authentication Errors
```json
{
  "detail": "Not authenticated",
  "status_code": 401
}
```

### Expired Token
```json
{
  "detail": "Token has expired",
  "status_code": 401
}
```

### Invalid Token
```json
{
  "detail": "Invalid authentication credentials",
  "status_code": 401
}
```

---

## Complete Authentication Flow Example

```python
import requests

BASE_URL = "http://localhost:8000"

# 1. Register a new user
register_data = {
    "email": "user@example.com",
    "username": "myusername",
    "password": "secure_password_123"
}

register_response = requests.post(f"{BASE_URL}/auth/register", json=register_data)
print("Registration:", register_response.json())

# 2. Login to get JWT token
login_data = {
    "username": "user@example.com",
    "password": "secure_password_123"
}

login_response = requests.post(f"{BASE_URL}/auth/login", data=login_data)
token_data = login_response.json()
jwt_token = token_data["access_token"]
print("Login successful, got JWT token")

# 3. Use token for authenticated requests
headers = {"Authorization": f"Bearer {jwt_token}"}

# Create a project
project_data = {
    "name": "My Research Project",
    "description": "Document analysis project"
}

project_response = requests.post(f"{BASE_URL}/api/v1/projects",
                               json=project_data, headers=headers)
print("Created project:", project_response.json())

# List user's projects
projects_response = requests.get(f"{BASE_URL}/api/v1/projects", headers=headers)
print("User projects:", projects_response.json())
