"""
Retriever for Gemini RAG
"""

from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.config import settings


def get_retriever(collection_name: str):

    embeddings = GoogleGenerativeAIEmbeddings(
        model=settings.GEMINI_EMBEDDING_MODEL,
        google_api_key=settings.GEMINI_API_KEY,
    )

    vectordb = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=settings.CHROMA_PERSIST_DIR,
    )

    retriever = vectordb.as_retriever(
        search_kwargs={"k": 4}
    )

    return retriever