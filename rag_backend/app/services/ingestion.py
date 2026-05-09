"""
Simple PDF ingestion for Gemini RAG
"""

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.config import settings


def ingest_document(file_path: str, collection_name: str) -> int:

    # Load PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    print("Loaded pages:", len(documents))

    # Split text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(documents)
    print("Chunks created:", len(chunks))

    # Add metadata
    filename = os.path.basename(file_path)
    for chunk in chunks:
        chunk.metadata["source_file"] = filename

    # Gemini embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model=settings.GEMINI_EMBEDDING_MODEL,
        google_api_key=settings.GEMINI_API_KEY,
    )

    # Store in Chroma
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=settings.CHROMA_PERSIST_DIR,
    )

    vectordb.persist()

    return len(chunks)