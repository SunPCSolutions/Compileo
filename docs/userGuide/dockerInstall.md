# Docker Installation Guide

This guide provides step-by-step instructions for installing and running Compileo using Docker and Docker Compose.

## Prerequisites

Before installing Compileo via Docker, ensure you have the following:

### System Requirements
- **Docker**: Version 20.10 or later
- **Docker Compose**: Version 2.0 or later
- **At least 4GB RAM** (8GB recommended)
- **At least 10GB free disk space**

### Installing Docker

#### Ubuntu/Debian
```bash
# Update package index
sudo apt update

# Install Docker
sudo apt install docker.io docker-compose-plugin

# Start and enable Docker service
sudo systemctl start docker
sudo systemctl enable docker

# Add your user to the docker group (optional, allows running without sudo)
sudo usermod -aG docker $USER
# Log out and back in for the group change to take effect
```

#### macOS
```bash
# Install Docker Desktop from https://www.docker.com/products/docker-desktop
# Or using Homebrew
brew install --cask docker
```

#### Windows
```bash
# Install Docker Desktop from https://www.docker.com/products/docker-desktop
```

#### Verify Installation
```bash
# Check Docker version
docker --version

# Check Docker Compose version
docker compose version

# Test Docker installation
docker run hello-world
```

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/compileo.git
cd compileo
```

### 2. Configure Environment

Copy the example environment file and edit it:

```bash
cp .env.example .env
```

Edit `.env` with your configuration. You can copy the template and uncomment optional parameters as needed:

```bash
# API Keys for authentication (can be set via GUI after deployment)
COMPILEO_API_KEYS=your-api-key-1,your-api-key-2

# Redis connection URL (required for job queuing)
REDIS_URL=redis://redis:6379/0

# API Base URL (for GUI to connect to API)
API_BASE_URL=http://compileo-api:8000

# AI Model API Keys (optional, for enhanced features)
GEMINI_API_KEY=your-gemini-api-key
GROK_API_KEY=your-grok-api-key
HUGGINGFACE_API_KEY=your-huggingface-api-key
OPENAI_API_KEY=your-openai-api-key

# --- Optional Parameters (Uncomment to customize) ---

# CORS_ORIGINS=http://localhost:3000,http://localhost:8080
# RATE_LIMIT_REQUESTS=100
# RATE_LIMIT_WINDOW=60
# LOG_LEVEL=INFO

# API_BASE_URL clarification:
# Within the Docker network, the GUI connects to the API using the
# service name 'compileo-api' and port 8000.
```

### 3. Build and Start Services

```bash
# Build and start all services
docker compose up --build

# Or run in background
docker compose up --build -d
```

### 4. Access the Application

Once the services are running:

- **Web GUI**: http://localhost:8501
- **API Documentation**: http://localhost:8000/docs
- **API Base URL**: http://localhost:8000

## Service Overview

Compileo runs as three main services:

### compileo-api
- **Port**: 8000
- **Purpose**: REST API backend with FastAPI
- **Features**: Document processing, dataset generation, job queuing
- **Health Check**: http://localhost:8000/docs

### compileo-gui
- **Port**: 8501
- **Purpose**: Web interface built with Streamlit
- **Features**: Dataset creation wizard, project management, quality analysis
- **Health Check**: http://localhost:8501/healthz

### redis
- **Port**: 6380 (external), 6379 (internal)
- **Purpose**: Job queuing and caching
- **Features**: RQ job queue, session storage

## Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `REDIS_URL` | Redis connection URL for job queuing | `redis://redis:6379/0` | Yes |
| `API_BASE_URL` | API base URL for GUI | `http://compileo-api:8000` | Yes |
| `COMPILEO_API_KEYS` | API authentication keys | - | No |
| `GEMINI_API_KEY` | Google Gemini API key | - | No |
| `GROK_API_KEY` | xAI Grok API key | - | No |
| `HUGGINGFACE_API_KEY` | HuggingFace API key | - | No |
| `OPENAI_API_KEY` | OpenAI API key | - | No |
| `LICENSE_ENCRYPTION_KEY` | Encryption key for sensitive settings | auto-generated | No |
| `LOG_LEVEL` | Logging level | `INFO` | No |
| `RATE_LIMIT_REQUESTS` | API rate limiting requests per window | `100` | No |
| `RATE_LIMIT_WINDOW` | API rate limiting time window (seconds) | `60` | No |
| `CORS_ORIGINS` | Allowed CORS origins (comma-separated) | `http://localhost:3000,http://localhost:8080` | No |

### API Keys Setup

For enhanced features, configure AI model API keys in the `.env` file (these can also be configured via the GUI settings later):

```bash
# Google Gemini (for document classification)
GEMINI_API_KEY=your-gemini-api-key

# xAI Grok (for advanced reasoning)
GROK_API_KEY=your-grok-api-key

# HuggingFace (for model downloads)
HUGGINGFACE_API_KEY=your-huggingface-api-key

# OpenAI (for GPT models)
OPENAI_API_KEY=your-openai-api-key
```

### Volume Management

Compileo uses Docker volumes for data persistence with a **multi-tenant database architecture**:

- `compileo_storage`: Application data including:
  - `storage/system/system.db` - System database (users, sessions, licenses)
  - `storage/{user_id}/tenant.db` - Per-user tenant databases
  - `storage/{user_id}/uploads/` - User uploaded documents
  - `storage/{user_id}/parsed/` - Parsed document files
  - `storage/{user_id}/chunks/` - Generated chunks
  - `storage/{user_id}/datasets/` - Generated datasets
- `compileo_plugins`: Plugin storage
- `compileo_hf_models`: HuggingFace model cache (Pre-population recommended for large models)
- `redis_data`: Redis persistence

#### Multi-Tenant Database Structure

Compileo uses a **physically isolated multi-tenant architecture**:

```
storage/
├── system/
│   └── system.db              # Global auth data (users, sessions, licenses)
├── {user_id_1}/
│   ├── tenant.db              # User 1's isolated database
│   ├── uploads/{project_id}/
│   ├── parsed/{project_id}/
│   ├── chunks/{project_id}/
│   └── datasets/
├── {user_id_2}/
│   ├── tenant.db              # User 2's isolated database
│   └── ...
```

**Benefits:**
- True data isolation between users
- Encrypted API keys at rest
- Easier backup/restore per tenant
- Compliance with data residency requirements

#### Pre-populating HuggingFace Models
If the HuggingFace parser hangs during model download in Docker, you can manually copy model weights from a local cache:

```bash
docker cp src/compileo/features/ingestion/hf_models/models--nanonets--Nanonets-OCR2-3B compileo-compileo-api-1:/app/src/compileo/features/ingestion/hf_models/
docker exec -u 0 compileo-compileo-api-1 chown -R compileo:compileo /app/src/compileo/features/ingestion/hf_models/
```
To view volume data:
```bash
# List volumes
docker volume ls | grep compileo

# Inspect volume contents
docker run --rm -v compileo_compileo_storage:/data alpine ls -la /data
```

## Usage

### Starting Services

```bash
# Start all services
docker compose up

# Start in background
docker compose up -d

# Start specific service
docker compose up compileo-gui

# Start with rebuild
docker compose up --build
```

### Stopping Services

```bash
# Stop all services
docker compose down

# Stop and remove volumes (WARNING: deletes data)
docker compose down -v
```

### Viewing Logs

```bash
# View all logs
docker compose logs

# View specific service logs
docker compose logs compileo-api

# Follow logs in real-time
docker compose logs -f compileo-gui
```

### Updating

```bash
# Pull latest changes
git pull

# Rebuild and restart
docker compose up --build -d
```

## Troubleshooting

### Common Issues

#### Services Won't Start

**Check system resources:**
```bash
# Check available memory
free -h

# Check disk space
df -h

# Check Docker system
docker system df
```

**Check logs:**
```bash
docker compose logs
```

#### GUI Can't Connect to API

**Verify API_BASE_URL:**
```bash
# Check environment variable
docker compose exec compileo-gui env | grep API_BASE_URL
```

**Test API connectivity:**
```bash
# From GUI container
docker compose exec compileo-gui curl -f http://compileo-api:8000/docs

# From host
curl -f http://localhost:8000/docs
```

#### Redis Connection Issues

**Check Redis status:**
```bash
docker compose ps redis
docker compose logs redis
```

**Test Redis connectivity:**
```bash
docker compose exec redis redis-cli ping
```

#### Permission Issues

**Fix volume permissions:**
```bash
# Reset volumes (WARNING: deletes data)
docker compose down -v
docker compose up --build -d
```

#### Build Failures

**Clear Docker cache:**
```bash
docker system prune -a
docker compose build --no-cache
```

### Performance Issues

#### Memory Usage
```bash
# Check container memory usage
docker stats

# Limit memory usage in docker-compose.yml
services:
  compileo-api:
    deploy:
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 2G
```

#### GPU Support
For NVIDIA GPU support, ensure NVIDIA Container Toolkit is installed:

```bash
# Install NVIDIA Docker support
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### Data Management

#### Backup Data
```bash
# Backup volumes
docker run --rm -v compileo_compileo_storage:/data -v $(pwd):/backup alpine tar czf /backup/backup.tar.gz -C /data .
```

#### Restore Data
```bash
# Restore volumes
docker run --rm -v compileo_compileo_storage:/data -v $(pwd):/backup alpine tar xzf /backup/backup.tar.gz -C /data
```

#### Reset Database

**⚠️ WARNING: This deletes ALL user data including all tenant databases and uploaded files.**

```bash
# Stop services
docker compose down

# Remove storage volume (deletes all tenant databases and files)
docker volume rm compileo_compileo_storage

# Restart services (creates fresh system database)
docker compose up -d
```

#### Backup and Restore Individual Tenant

To backup a specific user's data:

```bash
# Find the user ID from the system database
docker compose exec compileo-api python -c "
from src.compileo.storage.src.database import get_system_db_connection
with get_system_db_connection() as conn:
    cursor = conn.cursor()
    cursor.execute('SELECT id, email FROM users')
    for row in cursor.fetchall():
        print(f'{row[0]}: {row[1]}')
"

# Backup specific tenant database
docker run --rm \
  -v compileo_compileo_storage:/data \
  -v $(pwd):/backup \
  alpine tar czf /backup/tenant_backup.tar.gz -C /data {user_id}

# Restore specific tenant
docker run --rm \
  -v compileo_compileo_storage:/data \
  -v $(pwd):/backup \
  alpine tar xzf /backup/tenant_backup.tar.gz -C /data
```

## Advanced Configuration

### Custom Docker Compose

Create a `docker-compose.override.yml` for custom configuration:

```yaml
version: '3.8'

services:
  compileo-api:
    environment:
      - LOG_LEVEL=DEBUG
    ports:
      - "8001:8000"  # Custom port

  compileo-gui:
    ports:
      - "8502:8501"  # Custom port
```

### Development Setup

For development with live code reloading:

```yaml
version: '3.8'

services:
  compileo-api:
    volumes:
      - ./src:/app/src  # Mount source code
    environment:
      - PYTHONPATH=/app
    command: uvicorn src.compileo.api.main:app --host 0.0.0.0 --port 8000 --reload

  compileo-gui:
    volumes:
      - ./src:/app/src  # Mount source code
    environment:
      - PYTHONPATH=/app
```

### Production Deployment

For production deployments, use the dedicated production Docker Compose configuration:

```bash
# Production deployment with Cython compilation
# This builds optimized, compiled Python modules for better performance

docker-compose -f docker-compose.prod.yml up -d

# With custom worker scaling
docker-compose -f docker-compose.prod.yml up -d --scale compileo-worker=4
```

#### Production Configuration

The `docker-compose.prod.yml` file provides:
- **Cython Compilation**: Source code is compiled to C for performance and IP protection
- **Worker Scaling**: Horizontal and vertical scaling of job workers
- **Health Checks**: Built-in health monitoring for all services
- **Resource Limits**: Configurable CPU and memory limits

#### Worker Scaling

Compileo supports horizontal worker scaling through the `WORKER_REPLICAS` environment variable:

**Horizontal Scaling** (more worker containers):
```bash
# Deploy with 4 worker containers
docker-compose -f docker-compose.prod.yml up -d --scale compileo-worker=4

# Or set via environment variable
export WORKER_REPLICAS=4
docker-compose -f docker-compose.prod.yml up -d
```

Each worker container processes one job at a time. The total concurrent job capacity equals the number of worker replicas.

#### Production Environment Variables

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `WORKER_REPLICAS` | Number of worker containers | 2 | `4` |
| `COMPILEO_JWT_SECRET_KEY` | JWT signing key | (required) | `your-secret-key` |
| `COMPILEO_LICENSE_KEY` | License key for SaaS | (optional) | `XXXX-XXXX` |
| `LICENSE_ENCRYPTION_KEY` | Encryption key for API keys | auto-generated | `base64-encoded-key` |

**Important:** `LICENSE_ENCRYPTION_KEY` is used to encrypt sensitive settings (API keys) in the database. If you reset this key, existing encrypted settings will be lost.

#### Production Resource Configuration

```yaml
# docker-compose.prod.yml (excerpt)
services:
  compileo-api:
    environment:
      - LOG_LEVEL=WARNING
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: '4'
    restart: unless-stopped

  compileo-worker:
    deploy:
      replicas: 4
      resources:
        limits:
          memory: 4G
          cpus: '2'
    restart: unless-stopped

  redis:
    command: redis-server --appendonly yes --maxmemory 512mb --maxmemory-policy allkeys-lru
```

#### Building Production Images

```bash
# Build with Cython compilation
docker-compose -f docker-compose.prod.yml build

# Build specific service
docker-compose -f docker-compose.prod.yml build compileo-api

# Build with no cache (clean build)
docker-compose -f docker-compose.prod.yml build --no-cache
```

#### Monitoring Production Deployment

```bash
# Check service health
docker-compose -f docker-compose.prod.yml ps

# View worker logs
docker-compose -f docker-compose.prod.yml logs -f compileo-worker

# Check all service logs
docker-compose -f docker-compose.prod.yml logs -f

# View resource usage
docker stats
```

## Getting Help

### Documentation
- [API Documentation](http://localhost:8000/docs) - Interactive API docs
- [GUI User Guide](gui_user_guide.md) - Web interface guide
- [Configuration Guide](configuration.md) - Advanced configuration

### Common Commands
```bash
# View service status
docker compose ps

# Execute commands in containers
docker compose exec compileo-api bash
docker compose exec compileo-gui bash

# View resource usage
docker stats

# Clean up
docker system prune
docker volume prune
```

### Support
If you encounter issues:
1. Check the [troubleshooting section](#troubleshooting) above
2. Review the logs: `docker compose logs`
3. Check the [GitHub Issues](https://github.com/your-org/compileo/issues) for similar problems
4. Create a new issue with your Docker version, OS, and complete error logs

## Next Steps

Once Compileo is running:

1. **Access the Web GUI** at http://localhost:8501
2. **Create your first project** using the dataset creation wizard
3. **Upload documents** (PDF, DOCX, TXT, etc.)
4. **Configure AI models** in the Settings page
5. **Generate datasets** using the guided workflow

For detailed usage instructions, see the [GUI User Guide](gui_user_guide.md).