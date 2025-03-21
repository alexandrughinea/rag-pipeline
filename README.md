# RAG Pipeline

A Retrieval Augmented Generation (RAG) pipeline built with FastAPI, ChromaDB, and the Transformers library, designed to process various file types, extract their content, generate embeddings, and enhance information retrieval.

## Features

### Document Processing
- Support for multiple file formats:
  - Raw Text files (TXT, MD)
  - Documents (PDF, DOC, DOCX)
  - Markup files (XML)
  - Comma separated value files (CSV)
  - Images (with OCR)
  - Videos 
    - frame extraction and OCR
    - xclip temporal attention (in development)
  - Audio (in development)
- Automatic chunking and preprocessing
- Image cleaning and optimization
- Video frame analysis with relevance filtering

### Embedding Generation
- Configurable embedding models ()
- Efficient text chunking
- Batched processing for large documents
- Model dimension validation

### Vector Storage
- Persistent storage using ChromaDB
- Automatic batch processing for large datasets
- Metadata tracking
- Dimension compatibility checks

### Private Local Inference
- Support any model which is supported by transformers
- Metal acceleration for Apple Silicon
- Configurable inference parameters
- Conversation history tracking

### Simplified Diagram

```mermaid
graph LR
  upload["/upload"] --> FP[FileProcessor]
  FP --> EG[EmbeddingsGenerator]
  EG --> ChromaDB

  search["/search"] --> ChromaDB

  query["/query"] --> LLM[LLMHandler]
  LLM --> ChromaDB
  LLM --> SQLite

  stream["/query/conversation-stream"] --> LLM

  history["/history"] --> SQLite

  classDef endpoint fill:#e1bee7,stroke:#333,stroke-width:2px
  classDef processor fill:#c8e6c9,stroke:#333,stroke-width:2px
  classDef storage fill:#bbdefb,stroke:#333,stroke-width:2px

  class upload,search,query,stream,history endpoint
  class FP,EG,LLM processor
  class ChromaDB,SQLite storage
```

## Installation

1. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # Unix/MacOS
# or
.\venv\Scripts\activate    # Windows
```

2. Install dependencies:
```bash
# For production
pip install -r requirements.txt

# For development
pip install -r requirements-dev.txt
```

3. Set up environment variables in `.env`:
```env
# API Settings
API_HOST=0.0.0.0
API_PORT=8000

# Vector Store
VECTOR_STORAGE_DIR=./storage

# Embedding Model
EMBEDDING_MODEL=multi-qa-distilbert-cos-v1
EMBEDDING_CHUNK_SIZE=500

# LLM Settings
LLM_MODEL_TYPE=llama
LLM_MAX_TOKENS=512
LLM_TEMPERATURE=0.7
LLM_TOP_P=0.95

# API Authentication
API_KEY=your_secret_key_here
```

## Preparation

### 1. Get your LLM model from huggingface

```bash
huggingface-cli login --token YOUR_ACCESS_TOKEN
huggingface-cli download TheBloke/Llama-2-7B-Chat-GGUF
```

### 2. Update your .env with the chosen LLM model and type

```env
LLM_MODEL_PATH_OR_REPO_ID=TheBloke/Llama-2-7B-Chat-GGUF
LLM_MODEL_TYPE=llama
```

### 3. Define the LLMs behaviour
```env
LLM_MODEL_BEHAVIOUR_CONTEXT=...
```

## Preparation

### 1. Get your LLM model from huggingface

```bash
huggingface-cli login --token YOUR_ACCESS_TOKEN
huggingface-cli download meta-llama/Llama-3.1-8B
```

### 2. Update your .env with the chosen LLM model and type

```env
LLM_MODEL_PATH_OR_REPO_ID=meta-llama/Llama-3.1-8B
LLM_MODEL_TYPE=llama
```

### 3. Define the LLMs behaviour
```env
LLM_MODEL_BEHAVIOUR_CONTEXT=...
```

## Usage

#### Starting the Server
```bash
python src/main.py
```

### 1. Adding and Processing Files

```bash
# Upload a document
curl -X POST http://localhost:8000/upload \
  -H "x-api-key: your_secret_key_here" \
  -F "file=@document.pdf"
```

#### 2. Search storage for similar documents:

```bash 
curl -X GET http://localhost:8000/search?query=Test&limit=20 \
  -H "x-api-key: your_secret_key_here"
```

#### 3. Question example:

```bash 
# Ask a question about the file
curl -X POST -H "Content-Type: application/json" \
  -H "x-api-key: your_secret_key_here" \
  -d '{"query": "What are the main points in the document?"}' \
  http://localhost:8000/query/stream
```

#### 4. Conversation example:

```bash 
# Start new conversation
curl -X POST -H "Content-Type: application/json" \
  -H "x-api-key: your_secret_key_here" \
  -d '{"query": "What are the main points in the document?"}' \
  http://localhost:8000/query/conversation-stream

# Continue conversation with ID
curl -X POST -H "Content-Type: application/json" \
  -H "x-api-key: your_secret_key_here" \
  -d '{"query": "Can you elaborate on the first point?", "conversation_id": 1}' \
  http://localhost:8000/query/conversation-stream
```

#### 4. History:

```bash 
# Retrieve query history
curl -X GET http://localhost:8000/history?limit=5 \
  -H "x-api-key: your_secret_key_here"
```

## Configuration

### API Authentication
- The API is protected with an API key
- Set the `API_KEY` environment variable in your `.env` file:
  ```bash
  API_KEY=your_secret_key_here
  ```
- Include the API key in requests using the `x-api-key` header:
  ```bash
  curl -X POST http://localhost:8000/upload \
    -H "x-api-key: your_secret_key_here" \
    -F "file=@document.pdf"
  ```

### Hardware Requirements
- Apple Silicon (M1/M2) or Intel CPU
- 16 GB RAM (32 GB recommended)
- 1 TB Storage (2 TB recommended)

### Vector Store
- `VECTOR_STORAGE_DIR`: Location for persistent storage
- Maximum batch size: 5000 documents per batch

### Embedding Model
- Default model: `multi-qa-distilbert-cos-v1`
- Configurable chunk size for text splitting
- Automatic dimension validation

### LLM Configuration
- Model parameters:
  - `temperature`: 0.0 - 1.0 (default: 0.7)
  - `top_p`: 0.0 - 1.0 (default: 0.95)
  - `max_tokens`: 1 - model context length (default: 512)

### Image Processing
- Minimum DPI: 300
- Minimum dimensions: 800x600
- Automatic metadata cleaning

### Video Processing
- Frame extraction with similarity detection
- OCR confidence threshold: 60
- Automatic dark/blur frame filtering

## Development

### Code Formatting
```bash
# Format code
ruff format .
```

### Running Tests (@todo)
```bash
pytest
# With coverage
pytest --cov=src tests/
```

## Limitations and Considerations

- Maximum batch size of 5000 documents for vector storage
- Memory usage depends on document size and embedding dimensions
- Video processing can be resource-intensive
- Local LLM performance depends on hardware capabilities

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
