# RAG Backend API

A production-ready Retrieval Augmented Generation (RAG) API built with FastAPI, LangChain, and ChromaDB. This system enables context-grounded question answering over uploaded PDF documents with strict anti-hallucination measures.

## What is RAG?

Retrieval Augmented Generation (RAG) is a technique that combines information retrieval with language generation. Instead of relying solely on the LLM's training data, RAG:

1. **Retrieves** relevant context from a knowledge base (vector store)
2. **Augments** the user's query with this context
3. **Generates** an answer grounded in the retrieved documents

This approach reduces hallucinations and enables answering questions about documents the LLM wasn't trained on.

## Architecture

```
┌─────────────┐
│   FastAPI   │  ← REST API endpoints
└──────┬──────┘
       │
       ├─── POST /upload  → PDF Ingestion
       │                    ↓
       │              ┌─────────────┐
       │              │  ChromaDB   │  ← Vector Store
       │              └─────────────┘
       │
       └─── POST /query  → RAG Pipeline
                            ↓
                    ┌───────────────┐
                    │ Query Transform│  ← Follow-up question handling
                    └───────┬───────┘
                            ↓
                    ┌───────────────┐
                    │   Retriever   │  ← Similarity search (k=5)
                    └───────┬───────┘
                            ↓
                    ┌───────────────┐
                    │  RAG Chain    │  ← Context + LLM
                    └───────┬───────┘
                            ↓
                    ┌───────────────┐
                    │   Response    │  ← Answer + Sources + Confidence
                    └───────────────┘
```

### Key Components

1. **PDF Ingestion** (`services/ingestion.py`)
   - Loads PDFs using LangChain
   - Splits into chunks (1000 chars, 200 overlap)
   - Creates embeddings (OpenAI or Gemini)
   - Stores in ChromaDB with metadata

2. **Retriever** (`services/retriever.py`)
   - Loads ChromaDB collection
   - Performs similarity search (top 5 results)

3. **Query Transformation** (`services/query_transform.py`)
   - Converts follow-up questions into standalone queries
   - Uses chat history for context

4. **RAG Chain** (`services/rag_chain.py`)
   - Orchestrates the complete pipeline
   - Combines retrieval + generation
   - Enforces anti-hallucination prompts

5. **Anti-Hallucination Prompt** (`prompts/system_prompt.py`)
   - Strict instructions to answer only from context
   - Explicit refusal when answer not found
   - Confidence scoring

## Setup Instructions

### 1. Prerequisites

- Python 3.10 or higher
- pip (Python package manager)

### 2. Install Dependencies

```bash
cd rag_backend
pip install -r requirements.txt
```

### 3. Environment Configuration

Create a `.env` file in the `rag_backend` directory:

```bash
# LLM Provider: "openai" or "gemini"
LLM_PROVIDER=openai

# OpenAI Settings
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_EMBEDDING_MODEL=text-embedding-ada-002
OPENAI_LLM_MODEL=gpt-3.5-turbo

# Gemini Settings (if using Gemini)
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_EMBEDDING_MODEL=models/gemini-embedding-001
GEMINI_LLM_MODEL=gemini-pro

# ChromaDB Settings
CHROMA_PERSIST_DIR=./chroma_db
```

### 4. Rate limits and large PDFs (Gemini free tier)

- Gemini free tier allows about **100 embedding requests per minute**. A long PDF (e.g. 50+ pages) produces many chunks and can hit this limit.
- The app **throttles** ingestion: it processes chunks in batches with a short delay between batches to stay under the limit. Large PDFs will take longer to ingest but should complete.
- If you see a **429 (quota exceeded)** error, the app will **retry** a few times after a delay. You can also wait about a minute and upload again.
- For very large documents, consider splitting into smaller PDFs or using a paid plan for higher limits.

### 5. Run the Server

```bash
# From rag_backend directory
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Or use Python directly:

```bash
python -m app.main
```

The API will be available at:
- API: http://localhost:8000
- Swagger Docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## How to Switch LLM Providers

The system supports switching between OpenAI and Google Gemini via configuration.

### Using OpenAI (Default)

Set in `.env`:
```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_LLM_MODEL=gpt-3.5-turbo  # or gpt-4
OPENAI_EMBEDDING_MODEL=text-embedding-ada-002
```

### Using Google Gemini

Set in `.env`:
```bash
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_key
GEMINI_LLM_MODEL=gemini-pro
GEMINI_EMBEDDING_MODEL=models/gemini-embedding-001
```

The system automatically uses the correct models and API keys based on `LLM_PROVIDER`.

## How Query Transformation Works

Query transformation converts follow-up questions into standalone queries that can be understood without conversation history.

### Example:

**Conversation:**
1. User: "What is machine learning?"
2. Assistant: "Machine learning is..."
3. User: "What are its applications?" ← Follow-up question

**Transformation:**
- Original: "What are its applications?"
- Transformed: "What are the applications of machine learning?"

The transformation uses the LLM to rewrite queries by incorporating context from the last 5 messages in chat history. If there's no history or the query is already standalone, it returns the original query.

## API Endpoints

### POST /upload

Upload and ingest a PDF document.

**Request:**
- `file`: PDF file (multipart/form-data)
- `collection_name`: Optional, defaults to "default"

**Response:**
```json
{
  "message": "Successfully ingested PDF",
  "collection_name": "default",
  "chunks_stored": 42,
  "status": "success"
}
```

### POST /query

Query the RAG system.

**Request Body:**
```json
{
  "collection_name": "default",
  "query": "What is the main topic of this document?",
  "chat_history": [
    {
      "role": "user",
      "content": "What is machine learning?"
    },
    {
      "role": "assistant",
      "content": "Machine learning is..."
    }
  ]
}
```

**Response:**
```json
{
  "answer": "The main topic is...",
  "sources": [
    {
      "content": "Retrieved text chunk...",
      "page_number": 1,
      "source_file": "document.pdf",
      "similarity_score": null
    }
  ],
  "confidence": "High",
  "transformed_query": "What is the main topic of this document?",
  "metadata": {
    "num_sources": 5,
    "llm_provider": "openai",
    "llm_model": "gpt-3.5-turbo"
  }
}
```

### GET /health

Health check endpoint.

### GET /

Root endpoint with system information.

## Example Usage

### 1. Upload a PDF

```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@document.pdf" \
  -F "collection_name=my_docs"
```

### 2. Query the System

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "collection_name": "my_docs",
    "query": "What is the main topic?",
    "chat_history": []
  }'
```

### 3. Follow-up Question

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "collection_name": "my_docs",
    "query": "What are its applications?",
    "chat_history": [
      {
        "role": "user",
        "content": "What is machine learning?"
      },
      {
        "role": "assistant",
        "content": "Machine learning is a subset of AI..."
      }
    ]
  }'
```

## Features

✅ **PDF Upload & Processing** - Automatic chunking and embedding  
✅ **Vector Storage** - Persistent ChromaDB storage  
✅ **Context-Grounded Answers** - Strict anti-hallucination prompts  
✅ **Query Transformation** - Follow-up question handling  
✅ **Confidence Scoring** - Low/Medium/High confidence levels  
✅ **Source References** - Page numbers and file names  
✅ **LLM Swappable** - Easy switching between OpenAI and Gemini  
✅ **CORS Enabled** - Ready for frontend integration  
✅ **Swagger Documentation** - Interactive API docs  

## Project Structure

```
rag_backend/
│
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration & environment
│   ├── services/
│   │   ├── __init__.py
│   │   ├── ingestion.py     # PDF processing
│   │   ├── retriever.py     # Vector retrieval
│   │   ├── query_transform.py  # Query rewriting
│   │   └── rag_chain.py     # RAG pipeline
│   ├── prompts/
│   │   ├── __init__.py
│   │   └── system_prompt.py # Anti-hallucination prompts
│   └── models/
│       ├── __init__.py
│       └── schemas.py       # Pydantic models
│
├── chroma_db/               # Persistent vector store
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Error Handling

The system includes comprehensive error handling:

- **File Validation**: Only PDF files accepted
- **API Key Validation**: Clear errors if keys are missing
- **Collection Errors**: Handles missing collections gracefully
- **LLM Errors**: Catches and reports API failures

## Logging

All operations are logged with appropriate levels:
- INFO: Normal operations (ingestion, retrieval, queries)
- WARNING: Non-critical issues (query transformation fallback)
- ERROR: Critical failures

## Design Constraints

- ✅ No agents - Simple, explicit pipeline
- ✅ No complex chains - Straightforward RAG flow
- ✅ Modular architecture - Easy to understand and extend
- ✅ Beginner-friendly - Clear code structure

## License

This project is provided as-is for educational and production use.

