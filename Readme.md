# 🤖 RAG API — FastAPI + LangChain + Grok + FAISS

A lightweight Retrieval-Augmented Generation (RAG) API built with FastAPI. It uses **Grok** (by xAI) as the LLM, **FAISS** as a local vector store, and **LangChain** as the orchestration layer.

## Why this stack?

- **Grok API** — OpenAI-compatible, generous free tier for experimentation
- **FAISS** — Fast, local vector search with no external DB setup needed
- **LangChain** — Handles chaining, retrieval, memory, and text splitting
- **FastAPI** — Async, auto-generated Swagger docs at `/docs`

## Project Structure

```
.
├── app.py              # Main application
├── .env                # Environment variables (not committed)
├── requirements.txt    # Python dependencies
├── faiss_index/        # Auto-generated FAISS index (after ingestion)
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.10+
- A Grok API key from [xAI Console](https://console.x.ai/)

### Installation

```bash
# Clone the repo
git clone https://github.com/your-username/rag-api.git
cd rag-api

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```env
GROK_API_KEY=xai-your-key-here
GROK_BASE_URL=https://api.x.ai/v1
FAISS_INDEX_PATH=./faiss_index
```

### Run

```bash
python app.py
```

The server starts at `http://localhost:8000`. Open `http://localhost:8000/docs` for the interactive Swagger UI.

## API Endpoints

### `POST /ingest` — Ingest documents

Feed raw text into the FAISS vector store. Texts are automatically chunked.

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "FastAPI is a modern, high-performance Python web framework.",
      "FAISS is a library for efficient similarity search, built by Meta."
    ],
    "chunk_size": 500,
    "chunk_overlap": 50
  }'
```

### `POST /chat` — RAG-powered Q&A

Retrieves relevant chunks from FAISS and sends them as context to Grok. Maintains conversation history.

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is FAISS?"}'
```

### `POST /ask` — Direct LLM chat

Talk to Grok directly without any retrieval. Useful for general questions.

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Explain RAG in simple terms"}'
```

### `DELETE /reset` — Clear everything

Wipes the FAISS index and conversation memory.

```bash
curl -X DELETE http://localhost:8000/reset
```

## Embeddings Note

The app defaults to using the Grok API for embeddings. If Grok doesn't support an embeddings endpoint, switch to a free local model by updating `get_embeddings()` in `app.py`:

```python
from langchain_huggingface import HuggingFaceEmbeddings

def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
```

Then install the extra dependency:

```bash
pip install langchain-huggingface sentence-transformers
```

## Requirements

```
fastapi
uvicorn[standard]
langchain
langchain-openai
langchain-community
faiss-cpu
pydantic
python-dotenv
```

## Roadmap

- [ ] File upload endpoint (PDF, TXT, CSV)
- [ ] Streaming responses
- [ ] Multiple conversation sessions
- [ ] Configurable retrieval strategies
- [ ] Docker support

## License

MIT