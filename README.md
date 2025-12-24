## 📖 Documentation
For extensive documentation, including detailed API guides, GUI walkthroughs, plugin development tutorials, and CLI references, visit our [Compileo Documentation Site](https://sunpcsolutions.github.io/Compileo/).

# 🔬 Compileo: The Ultimate AI-Powered Document Processing & Dataset Engineering Suite

**Compileo** is a modular platform designed to process, analyze, structure, and generate data. Whether you're processing 1,000 page  PDFs, scraping JavaScript heavy websites, or engineering datasets for LLM fine-tuning and personal study, Compileo provides a unified, AI-driven lifecycle for the modern data era.

---

## 🌟 What can Compileo do?

Compileo isn't just a parser—it's a comprehensive **data engineering ecosystem**. It automates the complex journey from "Messy Source" to Validated Structured Data.

### 📖 Multi-Source Knowledge Consolidation
Imagine you have several thick textbooks and want to create a specialized dataset focused on **disease treatments** and **symptoms**. Compileo can:
1.  **Ingest** all books simultaneously (PDF, DOCX, etc.).
2.  **Discover** unified **"Treatment"** and **"Symptom"** taxonomies across all sources automatically.
3.  **Extract** every mention of treatments, dosages, clinical manifestations, and so on for its associated taxonomy for each disease.
4.  **Consolidate** this into one unified, high-quality Q&A dataset for training or study.

### 🕹️ Three Ways to Work
Compileo is designed for every workflow:
*   **Web GUI**: A user friendly Streamlit interface including a 7 step guided wizard and .
*   **REST API**: Seamlessly integrate data processing, chunking, NER, extraction, and dataset generation into your own applications.
*   **CLI**: Automate heavy duty processing with powerful command line parameters.

---

## 🚀  Features

### 📄 Intelligent Document Processing & AI-Assisted Chunking
*   **Massive PDF Autonomy**: Automatically splits 1,000+ page documents into manageable segments with intelligent context-aware chunking ensuring LLM token limits are never hit while preserving context.
*   **Two-Pass VLM Parsing**: Employs a "Skim and Extract" methodology using Vision Language Models (Grok, Gemini, ChatGPT, Huggingface, Ollama) to first understand document layout and then extract high-fidelity Markdown.
*   **AI-Assisted Chunking**: Want to chunk your book by chapter but each chapter has a different title? You can instruct AI how to chunk it (Each Chapter starts with... I need you to split the document at this point) or ask AI to advise you about the best strategy for chunking (This is the pattern that start each chapter. Which splitting strategy should I use?). Compileo's AI will analyze your documents, pattern examples, and your splitting goal to recommend the optimal **Semantic**, **Character**, **Token**, **Delimiter** or **Schema-based** chunking strategy.

### 🧠 Semantic Data Engineering
*   **AI-Assisted Taxonomy**: Don't waste weeks defining categories. Compileo can build hierarchical knowledge trees automatically based on your goals. You can also manually define some categories and let Compileo extend them.
*   **Multi-Stage Extraction**: Performs **Hierarchical Classification**, moving from coarse-grained categories to fine-grained entities based on your custom or generated taxonomy.
*   **Context-Aware NER or Full Text**: Uses parent context during extraction to disambiguate entities and discover deep relationships between concepts, then extract name entities or full text.

### 🧪 Advanced Quality Control & Evaluation
*   **Two-step AI Confidence Scoring**: Every extracted entity and relationship is assigned an **AI confidence level** (0.0 - 1.0), allowing you to filter for only the most reliable data.
*   **Deep Quality Metrics**: Automated scoring for **Lexical Diversity**, **Demographic Bias**, **Answer Coherence**, and **Target Audience Alignment** via the `datasetqual` module.
*   **Fine-Tuned Model Testing**: Use the `benchmarking` module to evaluate how your **fine-tuned models** perform on your custom datasets using industry-standard metrics (Accuracy, F1, BLEU, ROUGE) and suites like GLUE or MMLU.

### 🔌 Developer Extensibility
*   **Robust Plugin System**: Effortlessly extend Compileo by adding custom plugins via a simple `.zip` package architecture.
*   **Custom Exports**: Out-of-the-box support for **Scrapy** URL Scraping and **Anki** flashcard export, allowing you to turn any technical document into a high quality study deck.

---

## 💻 System Requirements

*   **CPU**: 4-core processor minimum (8-core recommended).
*   **RAM**: 8GB minimum (16GB recommended for heavy processing).
*   **GPU (Optional)**: NVIDIA GPU with 8GB+ VRAM. Required for **HuggingFace** local inference and advanced system performance monitoring.
*   **Storage**: 25GB free disk space.
*   **Operating System**: Linux, macOS, or Windows.

---

## �️ Installation

### 🐳 Option 1: Docker
The fastest way to deploy the full stack (API, GUI, and Redis).

1.  **Clone & Prepare**:
    ```bash
    git clone https://github.com/SunPCSolutions/Compileo.git
    cd compileo
    cp .env.example .env  # Configure COMPILEO_API_KEYS (optional - can also be set via GUI after deployment)
    ```
2.  **Launch**:
    ```bash
    docker compose up --build -d
    ```
3.  **Access**:
    *   **Web GUI**: `http://localhost:8501`
    *   **API Docs**: `http://localhost:8000/docs`

### 🔐 API Authentication & Security
Compileo implements an **"Auto-Lock"** security model designed for zero-config startup without sacrificing security.

*   **Unsecured Mode (Default)**: If no API keys are defined, Compileo allows all requests. This is ideal for first-time setup and local experimentation.
*   **Secured Mode**: As soon as you define an API key, the system "locks" and strictly requires that key for all API and GUI operations.

#### **How to Secure Your Instance (Choose One):**
1.  **GUI (Recommended)**: Launch Compileo, go to **Settings > 🔗 API Configuration**, enter one or more **API Keys**, and click **Save**. The system locks instantly.
2.  **CLI**: Start the API with the `--api-key` flag:
    ```bash
    uvicorn src.compileo.api.main:app --host 0.0.0.0 --port 8000
    ```
    *Note: Set API keys via GUI Settings after startup.*
3.  **Environment**: Define `COMPILEO_API_KEY=your_secret_key` in your `.env` or Docker configuration.

#### **How to Connect to a Secured Instance:**
All API requests must include the following header:
```http
X-API-Key: your_secret_key
```

---

### 🐍 Option 2: Python Environment
Ideal for local development, CLI automation, or custom integrations.

**Prerequisites**: A running Redis server (`sudo apt install redis-server`).

1.  **Setup Environment**:
    ```bash
    cd <github clone folder>
    python -m venv .venv
    source .venv/bin/activate  # Windows: .venv\Scripts\activate
    ```
2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Start Services**:
    ```bash
    # 1. Start the API server
    uvicorn src.compileo.api.main:app --host 0.0.0.0 --port 8000

    # 2. Start the Web GUI (In a new terminal)
    streamlit run src/compileo/features/gui/main.py --server.port 8501 --server.address 0.0.0.0
    ```
    *Note: For API security, set API keys via the GUI Settings after startup.*

---

## 📄 License
Apache-2.0 license