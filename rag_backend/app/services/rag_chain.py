"""
Simple RAG pipeline (Gemini)
"""

from typing import List, Dict, Tuple

from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.prompts import ChatPromptTemplate

from app.config import settings
from app.services.retriever import get_retriever
from app.services.query_transform import transform_query
from app.prompts.system_prompt import get_system_prompt
from app.models.schemas import Source


def run_rag(
    collection_name: str,
    user_query: str,
    chat_history: List[Dict],
) -> Tuple[str, List[Source], str, str]:

    #  Query rewrite
    transformed_query = transform_query(chat_history, user_query)

    # Retrieve docs
    retriever = get_retriever(collection_name)
    docs = retriever.invoke(transformed_query)

    # Build context
    context_parts = []
    sources = []

    for i, doc in enumerate(docs):
        context_parts.append(f"[Doc {i+1}]\n{doc.page_content}")

        sources.append(
            Source(
                content=doc.page_content[:500],
                page_number=doc.metadata.get("page"),
                source_file=doc.metadata.get("source_file"),
                similarity_score=None,
            )
        )

    context = "\n\n".join(context_parts)

    # # Prompt
    system_prompt = get_system_prompt(context)

    # prompt = ChatPromptTemplate.from_messages([
    #     ("system", system_prompt),
    #     ("human", "{question}")
    # ])

    # # Gemini LLM
    # llm = ChatGoogleGenerativeAI(
    #     model=settings.GEMINI_LLM_MODEL,
    #     temperature=0,
    #     google_api_key=settings.GEMINI_API_KEY,
    # )

    # chain = prompt | llm
    # response = chain.invoke({"question": transformed_query})
    # Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model=settings.GEMINI_LLM_MODEL,
        temperature=0,
        google_api_key=settings.GEMINI_API_KEY,
    )

    full_prompt = f"""
    {system_prompt}

    User Question:
    {transformed_query}
    """

    response = llm.invoke(full_prompt)

    answer = response.content.strip()

    # simple confidence
    confidence = "High" if len(docs) > 0 else "Low"

    return answer, sources, confidence, transformed_query