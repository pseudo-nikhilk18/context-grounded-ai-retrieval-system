"""
FastAPI main application with RAG endpoints.
"""
import os
import tempfile
import logging
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.models.schemas import (
    UploadResponse,
    QueryRequest,
    QueryResponse,
    ChatMessage
)
from app.services.ingestion import ingest_document
from app.services.rag_chain import run_rag

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RAG API",
    description="Retrieval Augmented Generation API with PDF ingestion and context-grounded answering",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "RAG API is running",
        "llm_provider": "gemini",
        "llm_model": settings.GEMINI_LLM_MODEL,
        "embedding_model": settings.GEMINI_EMBEDDING_MODEL
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(
    file: UploadFile = File(...),
    collection_name: str = "default"
):
    """
    Upload and ingest a PDF document.
    
    Args:
        file: PDF file to upload
        collection_name: Name of the ChromaDB collection (default: "default")
        
    Returns:
        UploadResponse with chunks stored count
    """
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    # Save file temporarily
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        logger.info(f"Temporary file saved: {tmp_file_path}")
        
        # Ingest document
        try:
            chunks_stored = ingest_document(tmp_file_path, collection_name)
            logger.info(f"Ingested {chunks_stored} chunks into collection '{collection_name}'")
            
            return UploadResponse(
                message=f"Successfully ingested PDF",
                collection_name=collection_name,
                chunks_stored=chunks_stored,
                status="success"
            )
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
                logger.info(f"Cleaned up temporary file: {tmp_file_path}")
                
    except Exception as e:
        logger.error(f"Error uploading PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest
):
    """
    Query the RAG system with a question.
    
    Args:
        request: QueryRequest with collection_name, query, and chat_history
        
    Returns:
        QueryResponse with answer, sources, and confidence
    """
    try:
        logger.info(f"Query received for collection '{request.collection_name}': {request.query}")
        
        # Convert chat history to dict format
        chat_history = [
            {"role": msg.role, "content": msg.content}
            for msg in request.chat_history
        ]
        
        # Run RAG pipeline
        answer, sources, confidence, transformed_query = run_rag(
            collection_name=request.collection_name,
            user_query=request.query,
            chat_history=chat_history
        )
        
        # Build response
        response = QueryResponse(
            answer=answer,
            sources=sources,
            confidence=confidence,
            transformed_query=transformed_query if transformed_query != request.query else None,
            metadata={
                "num_sources": len(sources),
                "llm_provider": "gemini",
                "llm_model": settings.GEMINI_LLM_MODEL
            }
        )
        
        logger.info(f"Query completed. Confidence: {confidence}, Sources: {len(sources)}")
        return response
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

