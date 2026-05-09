"""
Pydantic models for request/response schemas.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    """Response model for PDF upload endpoint."""
    message: str
    collection_name: str
    chunks_stored: int
    status: str = "success"


class ChatMessage(BaseModel):
    """Single chat message in history."""
    role: str = Field(..., description="Role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    collection_name: str = Field(..., description="Name of the ChromaDB collection")
    query: str = Field(..., description="User's question")
    chat_history: List[ChatMessage] = Field(
        default_factory=list,
        description="Previous conversation history"
    )


class Source(BaseModel):
    """Source document metadata."""
    content: str = Field(..., description="Retrieved text chunk")
    page_number: Optional[int] = Field(None, description="Page number if available")
    source_file: Optional[str] = Field(None, description="Source filename")
    similarity_score: Optional[float] = Field(None, description="Similarity score if available")


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    answer: str = Field(..., description="AI-generated answer")
    sources: List[Source] = Field(..., description="Retrieved source documents")
    confidence: str = Field(..., description="Confidence level: Low/Medium/High")
    transformed_query: Optional[str] = Field(None, description="Transformed query if applicable")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

