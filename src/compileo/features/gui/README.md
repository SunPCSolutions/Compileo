# Compileo GUI

A Streamlit-based web interface for the Compileo dataset creation and curation pipeline.

## Features

- **Dataset Creation Wizard**: Step-by-step guided workflow for creating datasets
- **Interactive Refinement**: Review, edit, and regenerate dataset entries
- **Taxonomy Browser**: Explore and manage hierarchical taxonomies
- **Quality Metrics Dashboard**: Visualize dataset quality metrics
- **Benchmarking Visualization**: Performance comparison across models
- **Real-time Monitoring**: Track long-running tasks and job progress

## Project Structure

```
src/compileo/features/gui/
├── main.py                    # Streamlit app entry point
├── config.py                  # GUI configuration
├── pages/                     # Streamlit pages
│   ├── home.py               # Dashboard/home page
│   ├── projects.py           # Project management
│   ├── wizard.py             # Dataset creation wizard
│   ├── taxonomy.py           # Taxonomy browser
│   ├── quality.py            # Quality metrics dashboard
│   ├── benchmarking.py       # Benchmarking visualization
│   └── settings.py           # Application settings
├── components/                # Reusable UI components
│   ├── wizard_steps.py       # Wizard step components
├── services/                  # API communication services
│   ├── api_client.py         # Base API client
├── state/                     # State management
│   ├── session_state.py      # Streamlit session state wrapper
│   └── wizard_state.py       # Wizard-specific state
├── utils/                     # Utility functions
└── memory-bank/              # Project documentation
```

## Installation

### Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
pip install -r requirements-gui.txt
```

2. Set environment variables:
```bash
export API_BASE_URL="http://localhost:8000"
export API_KEY="your-api-key"
```

3. Run the GUI:
```bash
streamlit run src/compileo/features/gui/main.py
```

### Docker Development

1. Build and run with Docker Compose:
```bash
docker-compose up --build
```

2. Access the GUI at `http://localhost:8501`

## Configuration

### Environment Variables

- `API_BASE_URL`: Backend API URL (default: `http://localhost:8000`)
- `API_KEY`: API authentication key
- `STREAMLIT_SERVER_PORT`: GUI port (default: 8501)
- `STREAMLIT_SERVER_ADDRESS`: GUI bind address (default: 0.0.0.0)

### GUI Configuration

Edit `config.py` to customize:
- Page settings and theme
- File upload limits
- API timeouts
- Default quality thresholds

## Usage

### Dataset Creation Wizard

1. **Project Selection**: Choose existing project or create new one
2. **Document Upload**: Upload documents (PDF, DOCX, TXT, MD, CSV, JSON, XML)
3. **Processing Configuration**: Set chunking strategy and analysis options
4. **Taxonomy Selection**: Choose or create taxonomy for categorization
5. **Generation Parameters**: Configure dataset purpose and quality settings
6. **Review & Generate**: Confirm settings and start generation
7. **Interactive Refinement**: Review and improve generated dataset

### API Integration

The GUI communicates with the FastAPI backend via REST API calls. All endpoints are documented in the backend API documentation.

## Development

### Running Tests

```bash
# Unit tests
pytest tests/gui/

# Integration tests
pytest tests/gui/ -k "integration"

# All tests
pytest tests/gui/ --cov=src/compileo/features/gui
```

### Code Style

Follow PEP 8 guidelines and use type hints. Format code with Black:

```bash
black src/compileo/features/gui/
```

### Adding New Features

1. Create feature branch
2. Add tests for new functionality
3. Update documentation
4. Submit pull request

## Deployment

### Production Deployment

1. Build Docker image:
```bash
docker build -f Dockerfile.gui -t compileo-gui .
```

2. Run with production settings:
```bash
docker run -p 8501:8501 \
  -e API_BASE_URL="https://api.compileo.com" \
  -e API_KEY="production-key" \
  compileo-gui
```

### Cloud Deployment Options

- **Streamlit Cloud**: Direct GitHub deployment
- **Heroku**: Container-based deployment
- **AWS/GCP/Azure**: Full infrastructure control

## Troubleshooting

### Common Issues

1. **API Connection Failed**
   - Check API_BASE_URL and API_KEY
   - Verify backend is running
   - Check network connectivity

2. **File Upload Errors**
   - Verify file format is supported
   - Check file size limits
   - Ensure proper permissions

3. **Session State Issues**
   - Clear browser cache
   - Restart Streamlit app
   - Check session state configuration

### Debug Mode

Enable debug information in the wizard page by toggling "Show Debug Info".

## Contributing

1. Fork the repository
2. Create feature branch
3. Make changes with tests
4. Submit pull request

## License

See main project LICENSE file.