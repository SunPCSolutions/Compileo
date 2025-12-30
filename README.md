---
title: Compileo
emoji: 🚀
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
---

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
*   **Web GUI**: A user friendly Streamlit interface including a 7 step guided wizard.
*   **REST API**: Seamlessly integrate data processing, chunking, NER, extraction, and dataset generation into your own applications.
*   **CLI**: Automate heavy duty processing with powerful command line parameters.

---

## 🚀  Features

### 📄 Intelligent Document Processing & AI-Assisted Chunking
*   **Massive PDF Autonomy**: Automatically splits 1,000+ page documents into manageable segments with intelligent context-aware chunking ensuring LLM token limits are never hit while preserving context.
*   **Two-Pass VLM Parsing**: Employs a "Skim and Extract" methodology using Vision Language Models (Grok, Gemini, ChatGPT, Huggingface, Ollama) to first understand document layout and then extract high-fidelity Markdown.

<div align="center">
  <a href="docs/img/DocumentProcessing.png"><img src="docs/img/DocumentProcessing.png" width="400" alt="Intelligent Document Processing Workflow"></a>
  <p><i>Intelligent Document Processing Workflow</i></p>
</div>

*   **AI-Assisted Chunking**: Want to chunk your book by chapter but each chapter has a different title? You can instruct AI how to chunk it (Each Chapter starts with... I need you to split the document at this point) or ask AI to advise you about the best strategy for chunking. Compileo's AI will analyze your documents and recommend the optimal **Semantic**, **Character**, **Token**, **Delimiter** or **Schema-based** chunking strategy.

<div align="center">
  <a href="docs/img/chunking_AIassist.png"><img src="docs/img/chunking_AIassist.png" width="400" alt="AI-Assisted Chunking Interface"></a>
  <p><i>AI-Assisted Chunking Interface</i></p>
</div>

### 🧠 Semantic Data Engineering
*   **AI-Assisted Taxonomy**: Don't waste weeks defining categories. Compileo can build hierarchical knowledge trees automatically based on your goals.

<div align="center">
  <a href="docs/img/Taxonomy_generating.png"><img src="docs/img/Taxonomy_generating.png" width="400" alt="AI-Assisted Taxonomy Generation"></a>
  <p><i>AI-Assisted Taxonomy Generation</i></p>
</div>

*   **Multi-Stage Extraction**: Performs **Hierarchical Classification**, moving from coarse-grained categories to fine-grained entities based on your custom or generated taxonomy.

<div align="center">
  <a href="docs/img/Extraction_main.png"><img src="docs/img/Extraction_main.png" width="400" alt="Multi-Stage Entity Extraction"></a>
  <p><i>Multi-Stage Entity Extraction</i></p>
</div>

*   **Dataset Engineering**: Transform extracted entities into high-quality datasets for RAG or fine-tuning.

<div align="center">
  <a href="docs/img/dataset_example.png"><img src="docs/img/dataset_example.png" width="400" alt="Generated Dataset Preview"></a>
  <p><i>Generated Dataset Preview</i></p>
</div>

### 🧪 Advanced Quality Control & Evaluation
*   **Deep Quality Metrics**: Automated scoring for **Lexical Diversity**, **Demographic Bias**, **Answer Coherence**, and **Target Audience Alignment** via the `datasetqual` module.

<div align="center">
  <a href="docs/img/datasetqual.png"><img src="docs/img/datasetqual.png" width="400" alt="Dataset Quality Evaluation"></a>
  <p><i>Dataset Quality Evaluation</i></p>
</div>

*   **Fine-Tuned Model Testing**: Use the `benchmarking` module to evaluate how your **fine-tuned models** perform on your custom datasets using industry-standard metrics (Accuracy, F1, BLEU, ROUGE).

<div align="center">
  <a href="docs/img/benchmark.png"><img src="docs/img/benchmark.png" width="400" alt="Model Performance Benchmarking"></a>
  <p><i>Model Performance Benchmarking</i></p>
</div>

### ⚡ High-Concurrency Job Management
*   **Asynchronous Processing**: All heavy-duty tasks are handled by a robust **Redis-backed queue (RQ)**, allowing for background processing without blocking the API or GUI.
*   **Real-time Monitoring**: Track the status, progress, and results of all ingestion, parsing, and extraction jobs.

<div align="center">
  <a href="docs/img/jobhandle.png"><img src="docs/img/jobhandle.png" width="400" alt="Asynchronous Job Management"></a>
  <p><i>Asynchronous Job Management</i></p>
</div>

### 🔌 Developer Extensibility
*   **Robust Plugin System**: Effortlessly extend Compileo by adding custom plugins via a simple `.zip` package architecture.
*   **Custom Exports**: Out-of-the-box support for **Scrapy** URL Scraping and **Anki** flashcard export.

---

## 💻 System Requirements

*   **CPU**: 4-core processor minimum (8-core recommended).
*   **RAM**: 8GB minimum (16GB recommended for heavy processing).
*   **GPU (Optional)**: NVIDIA GPU with 8GB+ VRAM. Required for **HuggingFace** local inference.
*   **Storage**: 25GB free disk space.
*   **Operating System**: Linux, macOS, or Windows.

---

## 🛠️ Installation

### 🐳 Option 1: Docker
The fastest way to deploy the full stack (API, GUI, and Redis).

1.  **Clone & Prepare**:
    ```bash
    git clone https://github.com/SunPCSolutions/Compileo.git
    cd compileo
    cp .env.example .env  # Configure COMPILEO_API_KEYS (optional)
    ```
2.  **Launch**:
    ```bash
    docker compose up --build -d
    ```
3.  **Access**:
    *   **Web GUI**: `http://localhost:8501`
    *   **API Docs**: `http://localhost:8000/docs`

### 🔐 API Authentication & Security
Compileo implements an **"Auto-Lock"** security model.

*   **Unsecured Mode (Default)**: If no API keys are defined, Compileo allows all requests.
*   **Secured Mode**: Defining an API key "locks" the system, requiring that key for all operations.

#### **How to Secure Your Instance (Choose One):**
1.  **GUI (Recommended)**: Go to **Settings > 🔗 API Configuration**, enter an **API Key**, and click **Save**.
2.  **CLI**: Start the API with the `--api-key` flag.
3.  **Environment**: Define `COMPILEO_API_KEY=your_secret_key` in your `.env`.

#### **How to Connect to a Secured Instance:**
Include the following header: `X-API-Key: your_secret_key`

---

### 🐍 Option 2: Python Environment
Ideal for local development or CLI automation.

**Prerequisites**: A running Redis server.

1.  **Setup Environment**:
    ```bash
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

    # 2. Start the Web GUI
    streamlit run src/compileo/features/gui/main.py --server.port 8501 --server.address 0.0.0.0
    ```

---

## 📄 License
Apache-2.0 license
